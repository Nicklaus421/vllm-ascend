#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Unit tests for the HybriMoE expert cache (host-side logic)."""

from unittest.mock import MagicMock

import torch

from vllm_ascend._310p.hybrimoe.cache import HybriMoECache, HybriMoELayerState
from vllm_ascend._310p.hybrimoe.config import HybriMoEConfig


def _make_state(num_experts: int = 8, num_slots: int = 3) -> HybriMoELayerState:
    return HybriMoELayerState("model.layers.0.mlp.experts", num_experts, num_slots)


class _FakeLayer:
    """Minimal stand-in for AscendHybriMoEFusedMoE310 (CPU tensors)."""

    def __init__(self, num_experts: int, num_slots: int, hidden: int = 4, intermediate: int = 2):
        self.layer_name = "model.layers.0.mlp.experts"
        self.w13_weight = torch.nn.Parameter(
            torch.full((num_slots, 2 * intermediate, hidden), -1, dtype=torch.int8), requires_grad=False
        )
        self.w2_weight = torch.nn.Parameter(
            torch.full((num_slots, hidden, intermediate), -1, dtype=torch.int8), requires_grad=False
        )
        self.w13_weight_scale = torch.nn.Parameter(torch.zeros(num_slots, 2 * intermediate), requires_grad=False)
        self.w2_weight_scale = torch.nn.Parameter(torch.zeros(num_slots, hidden), requires_grad=False)
        self.host_w13_int8 = torch.arange(num_experts * 2 * intermediate * hidden, dtype=torch.int8).view(
            num_experts, 2 * intermediate, hidden
        )
        self.host_w2_int8 = torch.arange(num_experts * hidden * intermediate, dtype=torch.int8).view(
            num_experts, hidden, intermediate
        )
        self.host_w13_scale = torch.rand(num_experts, 2 * intermediate)
        self.host_w2_scale = torch.rand(num_experts, hidden)


class TestMRSUpdate:
    def test_score_update_formula(self):
        state = _make_state(num_experts=4)
        ids = torch.tensor([1, 3])
        scores = torch.tensor([0.5, 0.25])
        state.mrs_update(ids, scores, alpha=0.5)
        # First update: S = 0.5 * s + 0.5 * 0 on selected, decayed elsewhere.
        assert torch.isclose(state.scores[1], torch.tensor(0.25))
        assert torch.isclose(state.scores[3], torch.tensor(0.125))
        assert state.scores[0] == 0.0 and state.scores[2] == 0.0

        # Second update with a different expert set: decay applies to all.
        state.mrs_update(torch.tensor([0]), torch.tensor([1.0]), alpha=0.5)
        assert torch.isclose(state.scores[0], torch.tensor(0.5))
        assert torch.isclose(state.scores[1], torch.tensor(0.125))

    def test_high_score_experts_are_kept(self):
        state = _make_state(num_experts=4, num_slots=2)
        state.expert_to_slot = torch.tensor([0, 1, -1, -1], dtype=torch.int32)
        state.slot_to_expert = [0, 1]
        state.scores = torch.tensor([0.9, 0.1, 0.0, 0.0])
        victim_slot = state.select_victim_slot(protected=set())
        assert victim_slot == 1  # expert 1 has the lowest score

    def test_protected_experts_are_never_evicted(self):
        state = _make_state(num_experts=4, num_slots=2)
        state.expert_to_slot = torch.tensor([0, 1, -1, -1], dtype=torch.int32)
        state.slot_to_expert = [0, 1]
        state.scores = torch.tensor([0.9, 0.1, 0.0, 0.0])
        victim_slot = state.select_victim_slot(protected={1})
        assert victim_slot == 0


class TestTransfers:
    def test_enqueue_transfer_copies_weights_and_updates_maps(self):
        layer = _FakeLayer(num_experts=8, num_slots=2)
        cache = HybriMoECache(HybriMoEConfig({"enabled": True}))
        cache._copy_stream = MagicMock()
        state = cache.register_layer(layer, num_experts=8, num_slots=2)

        event = cache.enqueue_transfer(state, expert=5, protected=set(), stream=cache.copy_stream)
        assert event is not None
        assert state.is_resident(5)
        slot = int(state.expert_to_slot[5].item())
        assert torch.equal(layer.w13_weight.data[slot], layer.host_w13_int8[5])
        assert torch.equal(layer.w2_weight.data[slot], layer.host_w2_int8[5])
        assert torch.equal(layer.w13_weight_scale.data[slot], layer.host_w13_scale[5])
        assert 5 in state.in_flight

    def test_enqueue_transfer_resident_is_noop(self):
        layer = _FakeLayer(num_experts=8, num_slots=2)
        cache = HybriMoECache(HybriMoEConfig({"enabled": True}))
        state = cache.register_layer(layer, num_experts=8, num_slots=2)
        state.expert_to_slot[3] = 1
        state.slot_to_expert[1] = 3
        assert cache.enqueue_transfer(state, expert=3, protected=set(), stream=MagicMock()) is None

    def test_eviction_on_full_cache(self):
        layer = _FakeLayer(num_experts=8, num_slots=2)
        cache = HybriMoECache(HybriMoEConfig({"enabled": True}))
        cache._copy_stream = MagicMock()
        state = cache.register_layer(layer, num_experts=8, num_slots=2)
        state.expert_to_slot[0] = 0
        state.expert_to_slot[1] = 1
        state.slot_to_expert = [0, 1]
        state.scores = torch.tensor([0.9, 0.1, 0, 0, 0, 0, 0, 0], dtype=torch.float32)

        cache.enqueue_transfer(state, expert=7, protected={0}, stream=cache.copy_stream)
        # Expert 1 (lowest score) evicted; expert 7 took its slot.
        assert int(state.expert_to_slot[1].item()) == -1
        assert int(state.expert_to_slot[7].item()) == 1
        assert state.slot_to_expert[1] == 7

    def test_batch_transfers_load_all_missing(self):
        layer = _FakeLayer(num_experts=8, num_slots=4)
        cache = HybriMoECache(HybriMoEConfig({"enabled": True}))
        cache._copy_stream = MagicMock()
        state = cache.register_layer(layer, num_experts=8, num_slots=4)
        cache.enqueue_transfers(state, [1, 2, 3], protected=set(), stream=cache.copy_stream)
        for e in (1, 2, 3):
            assert state.is_resident(e)
            assert torch.equal(layer.w13_weight.data[int(state.expert_to_slot[e].item())], layer.host_w13_int8[e])
        events = cache.collect_transfer_events(state, [1, 2, 3])
        assert len(events) == 3  # all share the batch event, popped individually
        assert state.misses == 3

    def test_batch_eviction_protects_activated(self):
        layer = _FakeLayer(num_experts=8, num_slots=2)
        cache = HybriMoECache(HybriMoEConfig({"enabled": True}))
        cache._copy_stream = MagicMock()
        state = cache.register_layer(layer, num_experts=8, num_slots=2)
        state.expert_to_slot[0] = 0
        state.expert_to_slot[1] = 1
        state.slot_to_expert = [0, 1]
        state.scores = torch.tensor([0.9, 0.1, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
        cache.enqueue_transfers(state, [7], protected={1}, stream=cache.copy_stream)
        # expert 0 evicted (expert 1 protected); expert 7 took slot 0
        assert int(state.expert_to_slot[0].item()) == -1
        assert int(state.expert_to_slot[7].item()) == 0
        assert state.slot_to_expert[0] == 7
