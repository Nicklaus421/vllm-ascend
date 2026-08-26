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
"""HybriMoE impact-driven inter-layer prefetching.

Because residual connections keep hidden states similar across consecutive
layers, the routing of upcoming layers can be approximated by reusing the
current MoE input through the gates of those layers (paper section IV-C).

At every decode forward of layer l this module:
  1. consumes the ready predictions issued by earlier layers, simulates the
     HSS makespan with and without each candidate expert cached, and
     prefetches the experts with the highest positive gain on a dedicated
     prefetch stream, then
  2. launches activation predictions for layers l+1..l+lookahead (small gate
     matmul on the NPU + async D2H).

Prediction results are staged in a ring of pinned buffers keyed by
``layer_position % lookahead``; a slot is only reused once its previous
prediction has been consumed, so in-flight predictions are never clobbered.
The D2H is launched several layers ahead of consumption, so the host never
blocks on it in the common case (`event.query()` polling).
"""

from __future__ import annotations

import torch
from vllm.logger import logger

from .cache import HybriMoECache, HybriMoELayerState
from .config import HybriMoEConfig
from .registry import HybriMoERegistry
from .scheduler import CostModel, simulate_makespan
from .utils import pin_memory_if_available


class ImpactDrivenPrefetcher:
    def __init__(
        self,
        config: HybriMoEConfig,
        cache: HybriMoECache,
        registry: HybriMoERegistry,
        cost_model: CostModel,
    ):
        self.config = config
        self.cache = cache
        self.registry = registry
        self.cost_model = cost_model
        self.lookahead = config.prefetch_lookahead
        self.prefetch_size = config.prefetch_size
        # Ring of pinned prediction buffers (lazily allocated), indexed by
        # target layer position % lookahead.
        self._pred_buffers: list[torch.Tensor | None] = [None] * self.lookahead
        # target layer position -> (event, buffer, num_tokens, top_k)
        self._pending: dict[int, tuple[object, torch.Tensor, int, int]] = {}

    def maybe_prefetch(
        self,
        state: HybriMoELayerState,
        x: torch.Tensor,
        top_k: int,
        scoring_func: str,
    ) -> None:
        position = self.registry.position_of(state.layer_name)
        num_tokens = x.shape[0]

        # 1. Consume ready (or discard stale) predictions first, so the ring
        #    slots are free before new predictions are launched.
        for next_position in list(self._pending):
            if next_position <= position:
                del self._pending[next_position]
                continue
            event, buffer, tokens, k = self._pending[next_position]
            if not event.query():
                continue
            del self._pending[next_position]
            self._issue_prefetches(next_position, buffer[: tokens * k])

        # 2. Launch predictions for upcoming layers into free ring slots.
        busy_rings = {p % self.lookahead for p in self._pending}
        main_stream = torch.npu.current_stream()
        for next_position in self.registry.next_positions(position, self.lookahead):
            ring = next_position % self.lookahead
            if ring in busy_rings:
                continue
            buffer = self._pred_buffers[ring]
            needed = num_tokens * top_k
            if buffer is None or buffer.numel() < needed:
                buffer = pin_memory_if_available(torch.empty(needed, dtype=torch.int64, device="cpu"))
                self._pred_buffers[ring] = buffer
            gate = self.registry.gate_modules[next_position]
            logits, _ = gate(x)
            scores = torch.softmax(logits.float(), dim=-1)
            _, predicted_ids = scores.topk(top_k, dim=-1)
            buffer[:needed].copy_(predicted_ids.view(-1), non_blocking=True)
            self._pending[next_position] = (main_stream.record_event(), buffer, num_tokens, top_k)
            busy_rings.add(ring)

    # ------------------------------------------------------------------
    def _issue_prefetches(self, next_position: int, predicted_ids: torch.Tensor) -> None:
        next_state = self.cache.get_layer(self.registry.layer_names[next_position])
        num_experts = next_state.num_experts
        counts_tensor = torch.bincount(predicted_ids, minlength=num_experts)
        activated = torch.nonzero(counts_tensor).flatten().tolist()
        counts = {e: int(counts_tensor[e].item()) for e in activated}

        resident = next_state.expert_to_slot.tolist()
        in_flight = set(next_state.in_flight)
        cached = [e for e in activated if resident[e] >= 0 or e in in_flight]
        uncached = [e for e in activated if resident[e] < 0 and e not in in_flight]
        if not uncached:
            return

        base_makespan = simulate_makespan(cached, uncached, counts, self.cost_model)
        gains = []
        for expert in uncached:
            with_expert = simulate_makespan(
                cached + [expert],
                [e for e in uncached if e != expert],
                counts,
                self.cost_model,
            )
            gain = base_makespan - with_expert - self.cost_model.move_time()
            gains.append((gain, expert))
        gains.sort(reverse=True)

        protected = set(activated) | in_flight
        issued = 0
        for gain, expert in gains:
            if issued >= self.prefetch_size or gain <= 0:
                break
            self.cache.enqueue_transfer(next_state, expert, protected, self.cache.prefetch_stream)
            issued += 1
        if issued:
            logger.debug(
                "HybriMoE prefetch: issued %d expert transfers for %s",
                issued,
                next_state.layer_name,
            )
