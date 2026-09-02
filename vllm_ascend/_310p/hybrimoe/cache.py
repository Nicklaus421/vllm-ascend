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
import torch_npu
from vllm.logger import logger

from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ

from .config import HybriMoEConfig

# Sentinel slot for CPU-bound (token, expert) pairs on the NPU side: their
# routing weight is zeroed, so the sentinel slot's compute contributes
# exactly nothing to the combined output.
SENTINEL_SLOT = 0

# Number of experts processed per H2D -> NZ-cast -> slot-copy chunk; bounds
# the ND staging memory to a few hundred MB.
_STAGING_ROWS = 32


class HybriMoELayerState:
    """Host-side cache state of one MoE layer."""

    def __init__(self, layer_name: str, num_experts: int, num_slots: int):
        self.layer_name = layer_name
        self.num_experts = num_experts
        self.num_slots = num_slots
        # All experts fit in the slot cache: the MoE forward can bypass every
        # HybriMoE mechanism (remap, D2H pack, MRS, miss handling) entirely.
        self.full_resident = num_slots >= num_experts
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
        # Top-1 dispatcher for the wave-streaming (prefill) path.
        self.dispatcher_top1 = None
        # NPU mirror of expert_to_slot for the pipelined decode path (device
        # side remap without a host sync). Updated on every slot mutation,
        # stream-ordered after the corresponding weight transfer.
        self.dev_expert_to_slot: torch.Tensor | None = None
        # Event of the last compact top-1 forward; guards dev-buffer reuse.
        self.last_compact_event = None
        # Frozen dispatch parameter objects (constant per layer; avoids
        # re-allocating them on every forward).
        self.routing_params = None
        self.quant_params = None
        # Pre-computed batched-DMA descriptors (swap_blocks_batch), built
        # lazily on first transfer.
        self.batch_params = None
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

    def select_victim_slots(self, count: int, protected: set[int]) -> list[int]:
        """Batch victim selection: slots of the `count` lowest-score eligible residents."""
        resident = self.expert_to_slot >= 0
        candidates = resident.clone()
        if protected:
            protected_ids = torch.tensor(sorted(protected), dtype=torch.int64)
            candidates[protected_ids] = False
        pool = candidates if bool(candidates.any()) else resident
        masked_scores = torch.where(pool, self.scores, torch.full_like(self.scores, float("inf")))
        count = min(count, int(pool.sum().item()))
        if count <= 0:
            return []
        victim_experts = torch.argsort(masked_scores)[:count]
        return [int(self.expert_to_slot[v].item()) for v in victim_experts.tolist()]


class HybriMoECache:
    """Process-wide expert cache for all HybriMoE MoE layers."""

    def __init__(self, config: HybriMoEConfig):
        self.config = config
        self.layers: dict[str, HybriMoELayerState] = {}
        self._copy_stream = None
        self._prefetch_stream = None
        self._staging_copy = None
        self._staging_prefetch = None
        # Whether slot weights are kept in FRACTAL_NZ format (validated once
        # at startup; falls back to ND on failure).
        self.use_nz_slots = True
        self.nz_validated = False
        self._foreach_copy_ok: dict[str, bool] = {}

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
    # Batched DMA (aclrtMemcpyBatchAsync via swap_blocks_batch)
    # ------------------------------------------------------------------
    def _get_staging(self, layer: torch.nn.Module, prefetch: bool):
        """ND staging rows for the H2D->NZ-cast->slot-copy transfer pipeline.

        One pair per stream (copy / prefetch) so transfers on the two streams
        never share staging rows.
        """
        attr = "_staging_prefetch" if prefetch else "_staging_copy"
        staging = getattr(self, attr, None)
        if staging is None:
            staging = (
                torch.empty(_STAGING_ROWS, *layer.host_w13_int8.shape[1:], dtype=torch.int8, device="npu"),
                torch.empty(_STAGING_ROWS, *layer.host_w2_int8.shape[1:], dtype=torch.int8, device="npu"),
            )
            setattr(self, attr, staging)
        return staging

    def _get_batch_params(self, state: HybriMoELayerState, prefetch: bool):
        """Batched-DMA descriptors for weight H2D into ND staging rows.

        Scales are copied per expert (they are tiny); only the two weight
        matrices go through the batched path. Returns None when unavailable.
        """
        key = "batch_params_prefetch" if prefetch else "batch_params_copy"
        params = getattr(state, key, None)
        if params is not None:
            return params or None
        try:
            # The batched op requires a CANN build with aclrtMemcpyBatchAsync;
            # probe it explicitly (build_params itself never fails).
            _ = torch.ops._C_ascend.swap_blocks_batch
            from vllm_ascend.simple_kv_offload.npu_mem_ops import DIRECTION_H2D, build_params

            layer = state.layer
            staging13, staging2 = self._get_staging(layer, prefetch)
            src = {"w13": layer.host_w13_int8, "w2": layer.host_w2_int8}
            dst = {"w13": staging13, "w2": staging2}
            params = build_params(src, dst, DIRECTION_H2D)
        except Exception:  # noqa: BLE001 - fall back to per-tensor copies
            logger.warning_once(
                "HybriMoE: batched DMA (swap_blocks_batch) unavailable; falling back to per-tensor copies."
            )
            params = False
        setattr(state, key, params)
        return params or None

    def validate_nz_layout(self, layer: torch.nn.Module) -> bool:
        """Self-check of the batch-major NZ layout assumption (run once).

        A single-expert NZ cast must byte-match slot 0 of the whole-stack NZ
        cast; otherwise per-slot NZ updates would silently corrupt weights.
        On mismatch the cache falls back to ND slots (correct, slower).
        """
        probe = layer.host_w13_int8[0:1].npu()
        probe_nz = torch_npu.npu_format_cast(probe, ACL_FORMAT_FRACTAL_NZ)
        ok = bool(torch.equal(probe_nz[0], layer.w13_weight.data[0]))
        self.use_nz_slots = ok
        if not ok:
            logger.warning(
                "HybriMoE: per-slot NZ layout validation failed; falling back to ND slot weights "
                "(MoE grouped matmul will be slower)."
            )
        return ok

    # ------------------------------------------------------------------
    # Residency management
    # ------------------------------------------------------------------
    def enqueue_transfer(
        self,
        state: HybriMoELayerState,
        expert: int,
        protected: set[int],
        stream,
        count_miss: bool = True,
    ):
        """Enqueue an H2D transfer of `expert` into an NPU slot.

        Returns the completion event of the transfer, or None if the expert
        is already resident (and not still in flight). `count_miss` is False
        for prefetch fills (they are not demand misses).
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
                # The victim's slot is about to be overwritten on `stream`;
                # order the overwrite after the victim's pending transfer on
                # device instead of blocking the host.
                stream.wait_event(stale[1])

        self._copy_experts_to_slots(state, [(expert, slot)], stream, prefetch=False)
        state.slot_to_expert[slot] = expert
        state.expert_to_slot[expert] = slot
        with torch.npu.stream(stream):
            self._write_mirror(state)
            event = stream.record_event()

        state.in_flight[expert] = (slot, event)
        if count_miss:
            state.misses += 1
        return event

    @staticmethod
    def _write_mirror(state: HybriMoELayerState) -> None:
        """Mirror the host residency map to the NPU (caller holds the stream)."""
        if state.dev_expert_to_slot is not None:
            state.dev_expert_to_slot.copy_(state.expert_to_slot, non_blocking=True)

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

    def enqueue_transfers(
        self,
        state: HybriMoELayerState,
        experts: list[int],
        protected: set[int],
        stream,
        count_miss: bool = True,
    ) -> None:
        """Batch version of enqueue_transfer with vectorized victim selection.

        All copies share one completion event (they are stream-ordered anyway);
        consumers must call collect_transfer_events() before using the experts.
        """
        misses = [e for e in experts if e not in state.in_flight and not state.is_resident(e)]
        if not misses:
            return
        free = [s for s, e in enumerate(state.slot_to_expert) if e < 0]
        if len(misses) > len(free):
            victim_slots = state.select_victim_slots(len(misses) - len(free), protected)
            for slot in victim_slots:
                victim = state.slot_to_expert[slot]
                if victim >= 0:
                    state.expert_to_slot[victim] = -1
                    state.slot_to_expert[slot] = -1
                    stale = state.in_flight.pop(victim, None)
                    if stale is not None:
                        # Order the slot overwrite after the victim's pending
                        # transfer on device instead of blocking the host.
                        stream.wait_event(stale[1])
            free.extend(victim_slots)

        assignments = list(zip(misses, free))
        self._copy_experts_to_slots(state, assignments, stream, prefetch=False)
        for expert, slot in assignments:
            state.slot_to_expert[slot] = expert
            state.expert_to_slot[expert] = slot
        with torch.npu.stream(stream):
            self._write_mirror(state)
            event = stream.record_event()
        for expert, slot in assignments:
            state.in_flight[expert] = (slot, event)
        if count_miss:
            state.misses += len(assignments)

    def collect_all_transfer_events(self, state: HybriMoELayerState) -> list:
        """Pop every in-flight transfer event of this layer (pipelined decode)."""
        events = [event for _, event in state.in_flight.values()]
        state.in_flight.clear()
        return events

    def _copy_experts_to_slots(
        self,
        state: HybriMoELayerState,
        assignments: list[tuple[int, int]],
        stream,
        prefetch: bool,
    ) -> None:
        """Transfer expert weights host -> NPU slots in NZ format.

        Pipeline per chunk: batched H2D of the ND weights into staging rows,
        one batched npu_format_cast, then per-slot NZ block copies; scales are
        copied directly (no NZ). Falls back to plain per-expert ND copies when
        NZ slots are disabled by layout validation.
        """
        layer = state.layer
        use_nz = self.use_nz_slots
        batch_params = self._get_batch_params(state, prefetch) if use_nz else None
        staging13, staging2 = self._get_staging(layer, prefetch) if use_nz else (None, None)
        with torch.npu.stream(stream):
            for start in range(0, len(assignments), _STAGING_ROWS):
                chunk = assignments[start : start + _STAGING_ROWS]
                n = len(chunk)
                experts = [expert for expert, _ in chunk]
                slots = [slot for _, slot in chunk]
                if use_nz:
                    if batch_params is not None:
                        from vllm_ascend.simple_kv_offload.npu_mem_ops import copy_blocks

                        copy_blocks(experts, list(range(n)), batch_params)
                    else:
                        self._foreach_copy(
                            [staging13[i] for i in range(n)] + [staging2[i] for i in range(n)],
                            [layer.host_w13_int8[e] for e in experts] + [layer.host_w2_int8[e] for e in experts],
                            group="staging_weights",
                        )
                    nz13 = torch_npu.npu_format_cast(staging13[:n], ACL_FORMAT_FRACTAL_NZ)
                    nz2 = torch_npu.npu_format_cast(staging2[:n], ACL_FORMAT_FRACTAL_NZ)
                    # NOTE: foreach_copy does not support NZ (internal format)
                    # destinations, so the slot NZ blocks go in their own
                    # group; scales are plain-format and share another.
                    self._foreach_copy(
                        [layer.w13_weight.data[s] for s in slots] + [layer.w2_weight.data[s] for s in slots],
                        list(nz13.unbind(0)) + list(nz2.unbind(0)),
                        group="nz_slot_weights",
                    )
                    self._foreach_copy(
                        [layer.w13_weight_scale.data[s] for s in slots]
                        + [layer.w2_weight_scale.data[s] for s in slots],
                        [layer.host_w13_scale[e] for e in experts] + [layer.host_w2_scale[e] for e in experts],
                        group="scales",
                    )
                else:
                    self._foreach_copy(
                        [layer.w13_weight.data[s] for s in slots]
                        + [layer.w2_weight.data[s] for s in slots]
                        + [layer.w13_weight_scale.data[s] for s in slots]
                        + [layer.w2_weight_scale.data[s] for s in slots],
                        [layer.host_w13_int8[e] for e in experts]
                        + [layer.host_w2_int8[e] for e in experts]
                        + [layer.host_w13_scale[e] for e in experts]
                        + [layer.host_w2_scale[e] for e in experts],
                        group="nd_all",
                    )

    def _foreach_copy(self, dsts: list[torch.Tensor], srcs: list[torch.Tensor], group: str) -> None:
        """Batched copy via torch._foreach_copy_ with per-group fallback.

        Some destinations are unsupported by foreach_copy (e.g. NZ / internal
        format tensors): the first failing group is remembered and all later
        copies of that group use a plain loop. copy_ is idempotent, so
        re-copying a partially-foreach'd group after a launch-time failure is
        safe.
        """
        if not dsts:
            return
        if self._foreach_copy_ok.get(group, True):
            try:
                torch._foreach_copy_(dsts, srcs, non_blocking=True)
                return
            except Exception:  # noqa: BLE001
                self._foreach_copy_ok[group] = False
                logger.warning_once(
                    "HybriMoE: torch._foreach_copy_ unsupported for %s; using per-tensor copies.", group
                )
        for dst, src in zip(dsts, srcs):
            dst.copy_(src, non_blocking=True)

    def hit_rate(self) -> float:
        hits = sum(s.hits for s in self.layers.values())
        misses = sum(s.misses for s in self.layers.values())
        total = hits + misses
        return hits / total if total else 0.0

    def log_hit_rates(self) -> None:
        logger.info("HybriMoE expert cache hit rate: %.4f", self.hit_rate())
