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
"""Unit tests for HybriMoE weight loading interception and dequantization."""

import torch

from vllm_ascend._310p.hybrimoe.fused_moe import AscendHybriMoEFusedMoE310, derive_num_slots
from vllm_ascend._310p.hybrimoe.utils import dequant_int8_per_channel

HIDDEN = 8
INTERMEDIATE = 4
NUM_SLOTS = 2
NUM_EXPERTS = 4


def _make_bare_layer():
    """AscendHybriMoEFusedMoE310 instance without running __init__."""
    layer = object.__new__(AscendHybriMoEFusedMoE310)
    layer.intermediate_size_per_partition = INTERMEDIATE
    # NPU slot parameters (CPU tensors in the UT).
    layer.w13_weight = torch.nn.Parameter(
        torch.zeros(NUM_SLOTS, 2 * INTERMEDIATE, HIDDEN, dtype=torch.int8), requires_grad=False
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.zeros(NUM_SLOTS, HIDDEN, INTERMEDIATE, dtype=torch.int8), requires_grad=False
    )
    layer.w13_weight_scale = torch.nn.Parameter(torch.zeros(NUM_SLOTS, 2 * INTERMEDIATE), requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(torch.zeros(NUM_SLOTS, HIDDEN), requires_grad=False)
    layer.w13_weight_offset = torch.nn.Parameter(torch.zeros(NUM_SLOTS, 2 * INTERMEDIATE, 1), requires_grad=False)
    layer.w2_weight_offset = torch.nn.Parameter(torch.zeros(NUM_SLOTS, HIDDEN, 1), requires_grad=False)
    # Host buffers.
    layer.host_w13_int8 = torch.zeros(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, dtype=torch.int8)
    layer.host_w2_int8 = torch.zeros(NUM_EXPERTS, HIDDEN, INTERMEDIATE, dtype=torch.int8)
    layer.host_w13_scale = torch.zeros(NUM_EXPERTS, 2 * INTERMEDIATE)
    layer.host_w2_scale = torch.zeros(NUM_EXPERTS, HIDDEN)
    layer.host_w13_offset = torch.zeros(NUM_EXPERTS, 2 * INTERMEDIATE)
    layer.host_w2_offset = torch.zeros(NUM_EXPERTS, HIDDEN)
    return layer


def test_weight_loader_routes_w13_shards():
    layer = _make_bare_layer()
    w1 = torch.full((INTERMEDIATE, HIDDEN), 3, dtype=torch.int8)
    w3 = torch.full((INTERMEDIATE, HIDDEN), 5, dtype=torch.int8)
    ok = layer.weight_loader(layer.w13_weight, w1, "w13_weight", "w1", 2, return_success=True)
    assert ok
    ok = layer.weight_loader(layer.w13_weight, w3, "w13_weight", "w3", 2, return_success=True)
    assert ok
    assert torch.all(layer.host_w13_int8[2, :INTERMEDIATE] == 3)
    assert torch.all(layer.host_w13_int8[2, INTERMEDIATE:] == 5)
    # The NPU slot parameter must not be touched by the loader.
    assert torch.all(layer.w13_weight.data == 0)


def test_weight_loader_routes_w2_and_scales():
    layer = _make_bare_layer()
    w2 = torch.full((HIDDEN, INTERMEDIATE), 7, dtype=torch.int8)
    assert layer.weight_loader(layer.w2_weight, w2, "w2_weight", "w2", 1, return_success=True)
    assert torch.all(layer.host_w2_int8[1] == 7)

    w1_scale = torch.full((INTERMEDIATE, 1), 0.5)
    w3_scale = torch.full((INTERMEDIATE, 1), 0.25)
    layer.weight_loader(layer.w13_weight_scale, w1_scale, "w13_weight_scale", "w1", 1)
    layer.weight_loader(layer.w13_weight_scale, w3_scale, "w13_weight_scale", "w3", 1)
    assert torch.all(layer.host_w13_scale[1, :INTERMEDIATE] == 0.5)
    assert torch.all(layer.host_w13_scale[1, INTERMEDIATE:] == 0.25)

    w2_scale = torch.full((HIDDEN, 1), 2.0)
    layer.weight_loader(layer.w2_weight_scale, w2_scale, "w2_weight_scale", "w2", 1)
    assert torch.all(layer.host_w2_scale[1] == 2.0)


def test_dequant_matches_fp32_reference():
    torch.manual_seed(0)
    w_int8 = torch.randint(-8, 8, (NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN), dtype=torch.int8)
    scale = torch.rand(NUM_EXPERTS, 2 * INTERMEDIATE) + 0.5
    out = dequant_int8_per_channel(w_int8, scale)
    reference = (w_int8.float() * scale.unsqueeze(-1)).to(torch.bfloat16)
    assert torch.equal(out, reference)


def test_derive_num_slots():
    class _Cfg:
        npu_cache_slots_per_layer = None
        npu_cache_budget_gb = 1.0

    # bytes/slot = 3*8*4 int8 + (2*4+8)*4 fp32 = 96 + 64 = 160; 1GiB / (48 * 160) -> large
    slots = derive_num_slots(_Cfg(), num_experts=128, hidden_size=8, intermediate_size=4, num_moe_layers=48)
    assert slots == 128  # clamped to num_experts

    _Cfg.npu_cache_slots_per_layer = 7
    assert derive_num_slots(_Cfg(), 128, 8, 4, 48) == 7

    _Cfg.npu_cache_slots_per_layer = None
    _Cfg.npu_cache_budget_gb = 1e-6  # tiny budget -> clamped to 1
    assert derive_num_slots(_Cfg(), 128, 8, 4, 48) == 1
