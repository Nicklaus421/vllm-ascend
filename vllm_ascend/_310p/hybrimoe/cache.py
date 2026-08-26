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
"""HybriMoE expert cache: NPU slot tables + MRS score-aware replacement.

Every MoE layer owns `num_slots` NPU-resident expert slots. The host keeps
the full expert set (int8 weights for H2D transfer, dequantized weights
for CPU compute) and tracks:

  - slot_to_expert / expert_to_slot: the residency mapping (host only; the
    remapped topk ids are built on the host and copied H2D per forward),
  - MRS priority scores: S = alpha * TopP(routing_scores) + (1 - alpha) * S,
  - in_flight transfers (expert -> event) so a consumer can wait on a
    prefetch instead of issuing a duplicate transfer.

All slot mutations go through enqueue_transfer(), which keeps the residency
maps and the in-flight table consistent.
"""

from __future__ import annotations

import torch
from vllm.logger import logger

from .config import HybriMoEConfig

# Sentinel slot for CPU-bound (token, expert) pairs on the NPU side: their
# routing weight is zeroed, so the sentinel slot's compute contributes
# exactly nothing to the combined output.
SENTINEL_SLOT = 0


class HybriMoELayerState:
    """Host-side cache state of one MoE layer."""

    def __init__(self, layer_name: str, num_experts: int, num_slots: int):
        self.layer_name = layer_name
        self.num_experts = num_experts
        self.num_slots = num_slots
        # Host-side state; pin the device explicitly because this may be
        # constructed while the default device is the NPU (model loading).
        self.expert_to_slot = torch.full((num_experts,), -1, dtype=torch.int32, device="cpu")
        self.slot_to_expert: list[int] = [-1] * num_slots
        self.scores = torch.zeros(num_experts, dtype=torch.float32, device="cpu")
        # expert id -> (slot, event) for transfers that have been enqueued on
        # a copy/prefetch stream but not yet consumed by the compute stream.
        self.in_flight: dict[int, tuple[int, object]] = {}
        # Back-reference to the FusedMoE layer (host weight buffers and NPU
        # slot parameters), set by register_layer().
        self.layer = None
        # Slot-granularity token dispatcher (num_experts == num_slots), set by
        # the scheme after weight loading.
        self.dispatcher = None
        # Cache telemetry.
        self.hits = 0
        self.misses = 0

    def resident_experts(self) -> list[int]:
        return torch.nonzero(self.expert_to_slot >= 0).flatten().tolist()

    def is_resident(self, expert: int) -> bool:
        return int(self.expert_to_slot[expert].item()) >= 0

    def mrs_update(self, top_p_ids: torch.Tensor, top_p_scores: torch.Tensor, alpha: float) -> None:
        """Minus-Recent-Score update: S = alpha * TopP(s) + (1 - alpha) * S.

        Args:
            top_p_ids: unique expert ids with the highest aggregated routing
                scores in this iteration (host int64 tensor).
            top_p_scores: aggregated routing scores for those ids (fp32).
            alpha: averaging coefficient.
        """
        self.scores.mul_(1.0 - alpha)
        self.scores.index_add_(0, top_p_ids, top_p_scores, alpha=alpha)

    def select_victim_slot(self, protected: set[int]) -> int:
        """Pick the slot whose resident expert has the lowest MRS score.

        Experts in `protected` (currently activated / in-flight) are never
        evicted. Falls back to any resident expert if every resident expert
        is protected (correct but may thrash; happens only when the number
        of activated experts exceeds the slot count).
        """
        resident = self.expert_to_slot >= 0
        candidates = resident.clone()
        if protected:
            protected_ids = torch.tensor(sorted(protected), dtype=torch.int64)
            candidates[protected_ids] = False
        pool = candidates if bool(candidates.any()) else resident
        # Score -inf for non-candidates so argmin never picks them.
        masked_scores = torch.where(pool, self.scores, torch.full_like(self.scores, float("inf")))
        victim_expert = int(torch.argmin(masked_scores).item())
        return int(self.expert_to_slot[victim_expert].item())


class HybriMoECache:
    """Process-wide expert cache for all HybriMoE MoE layers."""

    def __init__(self, config: HybriMoEConfig):
        self.config = config
        self.layers: dict[str, HybriMoELayerState] = {}
        self._copy_stream = None
        self._prefetch_stream = None

    # ------------------------------------------------------------------
    # Streams (created lazily so importing this module never touches NPU).
    # ------------------------------------------------------------------
    @property
    def copy_stream(self):
        if self._copy_stream is None:
            self._copy_stream = torch.npu.Stream()
        return self._copy_stream

    @property
    def prefetch_stream(self):
        if self._prefetch_stream is None:
            self._prefetch_stream = torch.npu.Stream()
        return self._prefetch_stream

    # ------------------------------------------------------------------
    # Layer registration
    # ------------------------------------------------------------------
    def register_layer(self, layer: torch.nn.Module, num_experts: int, num_slots: int) -> HybriMoELayerState:
        state = HybriMoELayerState(layer.layer_name, num_experts, num_slots)
        state.layer = layer
        self.layers[layer.layer_name] = state
        return state

    def get_layer(self, layer_name: str) -> HybriMoELayerState:
        return self.layers[layer_name]

    # ------------------------------------------------------------------
    # Residency management
    # ------------------------------------------------------------------
    def enqueue_transfer(
        self,
        state: HybriMoELayerState,
        expert: int,
        protected: set[int],
        stream,
    ):
        """Enqueue an H2D transfer of `expert` into an NPU slot.

        Returns the completion event of the transfer, or None if the expert
        is already resident (and not still in flight).
        """
        pending = state.in_flight.get(expert)
        if pending is not None:
            return pending[1]
        if state.is_resident(expert):
            return None

        free_slots = [s for s, e in enumerate(state.slot_to_expert) if e < 0]
        if free_slots:
            slot = free_slots[0]
        else:
            slot = state.select_victim_slot(protected)
            victim = state.slot_to_expert[slot]
            state.expert_to_slot[victim] = -1
            state.slot_to_expert[slot] = -1
            stale = state.in_flight.pop(victim, None)
            if stale is not None:
                # The victim's transfer finished or is irrelevant now; the
                # slot copy below is stream-ordered after it anyway.
                stale[1].synchronize()

        layer = state.layer
        with torch.npu.stream(stream):
            layer.w13_weight.data[slot].copy_(layer.host_w13_int8[expert], non_blocking=True)
            layer.w2_weight.data[slot].copy_(layer.host_w2_int8[expert], non_blocking=True)
            layer.w13_weight_scale.data[slot].copy_(layer.host_w13_scale[expert], non_blocking=True)
            layer.w2_weight_scale.data[slot].copy_(layer.host_w2_scale[expert], non_blocking=True)
            event = stream.record_event()

        state.slot_to_expert[slot] = expert
        state.expert_to_slot[expert] = slot
        state.in_flight[expert] = (slot, event)
        state.misses += 1
        return event

    def collect_transfer_events(self, state: HybriMoELayerState, experts: list[int]) -> list:
        """Events the compute stream must wait on before using `experts`."""
        events = []
        consumed = []
        for expert in experts:
            pending = state.in_flight.get(expert)
            if pending is not None:
                events.append(pending[1])
                consumed.append(expert)
        for expert in consumed:
            del state.in_flight[expert]
        return events

    def ensure_resident(self, state: HybriMoELayerState, experts: list[int]) -> list:
        """Prefill path: make every expert in `experts` NPU-resident.

        Returns the events to wait on. Eviction protects the activated set so
        the current layer forward never evicts an expert it still needs
        (unless the activated set exceeds the slot count).
        """
        protected = set(experts)
        events = []
        for expert in experts:
            if state.is_resident(expert) and expert not in state.in_flight:
                state.hits += 1
                continue
            event = self.enqueue_transfer(state, expert, protected, self.copy_stream)
            if event is not None:
                events.append(event)
        return events

    def hit_rate(self) -> float:
        hits = sum(s.hits for s in self.layers.values())
        misses = sum(s.misses for s in self.layers.values())
        total = hits + misses
        return hits / total if total else 0.0

    def log_hit_rates(self) -> None:
        logger.info("HybriMoE expert cache hit rate: %.4f", self.hit_rate())
