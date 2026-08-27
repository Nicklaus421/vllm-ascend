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
"""HybriMoE FusedMoE layer + W8A8 scheme for Ascend 310P.

Implements the HybriMoE forward path (https://arxiv.org/abs/2504.05897):

  - Prefill: every activated expert is made NPU-resident (on-demand H2D with
    MRS eviction) and computed by the regular grouped matmul.
  - Decode: the HSS scheduler splits activated experts between the NPU cache
    and the CPU; the NPU computes its share via a slot-remapped grouped
    matmul (CPU-bound pairs get the sentinel slot with routing weight 0,
    which contributes exactly 0 at combine time), the CPU computes its share
    asynchronously from the host dequantized copies, and the partial results are
    summed.

Weight layout:
  - NPU: `num_slots`-shaped stacked int8 weights + fp32 scales (ND layout;
    NZ casting of single slots is left as a future optimization, see the
    design doc / R1).
  - Host: pinned int8 weights + fp32 scales (source of truth for H2D) and
    pageable dequantized weights in params_dtype (for CPU compute, optional).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch
from vllm.logger import logger

from vllm_ascend._310p.fused_moe.experts_selector import select_experts
from vllm_ascend._310p.fused_moe.fused_moe import AscendFusedMoE310
from vllm_ascend._310p.fused_moe.moe_mlp import quant_apply_mlp
from vllm_ascend._310p.fused_moe.token_dispatcher import TokenDispatcherWithAllGather310
from vllm_ascend._310p.hybrimoe.cache import SENTINEL_SLOT, HybriMoELayerState
from vllm_ascend._310p.hybrimoe.runtime import HybriMoERuntime
from vllm_ascend._310p.hybrimoe.scheduler import hss_schedule
from vllm_ascend._310p.hybrimoe.utils import dequant_int8_per_channel, pin_memory_if_available
from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ops.fused_moe.moe_comm_method import FusedExpertsResult
from vllm_ascend.ops.fused_moe.moe_stage_contracts import MoETokenDispatchInput
from vllm_ascend.ops.fused_moe.moe_stage_params import MoEQuantParams, MoERoutingParams
from vllm_ascend.quantization.method_adapters import AscendFusedMoEMethod
from vllm_ascend.quantization.methods.base import AscendMoEScheme
from vllm_ascend.quantization.quant_type import QuantType


def derive_num_slots(config, num_experts: int, hidden_size: int, intermediate_size: int, num_moe_layers: int) -> int:
    """Number of NPU cache slots per MoE layer from the configured budget."""
    if config.npu_cache_slots_per_layer is not None:
        slots = config.npu_cache_slots_per_layer
    else:
        bytes_per_slot = (2 * intermediate_size * hidden_size) + (hidden_size * intermediate_size)
        bytes_per_slot += (2 * intermediate_size + hidden_size) * 4  # fp32 scales
        budget_bytes = config.npu_cache_budget_gb * float(2**30)
        slots = int(budget_bytes // (num_moe_layers * bytes_per_slot))
    if slots < 1:
        logger.warning(
            "HybriMoE npu_cache_budget_gb=%.2f is too small for even one expert slot per layer; "
            "clamping to 1 slot per layer.",
            config.npu_cache_budget_gb,
        )
    if slots >= num_experts:
        logger.warning(
            "HybriMoE derived %d slots per layer >= num_experts (%d): the entire expert set would be "
            "NPU-resident, which defeats the purpose of expert offloading and wastes HBM. Check that "
            "npu_cache_budget_gb (%.2f) and the MoE layer count (%d) are as intended.",
            slots,
            num_experts,
            config.npu_cache_budget_gb,
            num_moe_layers,
        )
    return max(1, min(slots, num_experts))


class AscendHybriMoEW8A8DynamicScheme310(AscendMoEScheme):
    """HybriMoE W8A8-dynamic MoE scheme for 310P.

    Same weight format as AscendW8A8DynamicFusedMoEMethod310, except the NPU
    parameters only hold `num_slots` expert slots instead of all experts.
    """

    quant_type: QuantType = QuantType.W8A8

    def __init__(self):
        self.config = get_ascend_config().hybrimoe_config

    # The NPU parameters are the cache slots: [num_slots, ...].
    def get_weight(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        num_slots = self._num_slots(num_experts, intermediate_size_per_partition, hidden_sizes)
        return {
            "w13_weight": torch.empty(num_slots, 2 * intermediate_size_per_partition, hidden_sizes, dtype=torch.int8),
            "w2_weight": torch.empty(num_slots, hidden_sizes, intermediate_size_per_partition, dtype=torch.int8),
        }

    def get_dynamic_quant_param(
        self, num_experts: int, intermediate_size_per_partition: int, hidden_sizes: int, params_dtype: torch.dtype
    ) -> dict[str, Any]:
        num_slots = self._num_slots(num_experts, intermediate_size_per_partition, hidden_sizes)
        return {
            "w13_weight_scale": torch.empty(num_slots, 2 * intermediate_size_per_partition, dtype=torch.float32),
            "w13_weight_offset": torch.empty(num_slots, 2 * intermediate_size_per_partition, 1, dtype=params_dtype),
            "w2_weight_scale": torch.empty(num_slots, hidden_sizes, dtype=torch.float32),
            "w2_weight_offset": torch.empty(num_slots, hidden_sizes, 1, dtype=params_dtype),
        }

    def _num_slots(self, num_experts: int, intermediate_size: int, hidden_size: int) -> int:
        from vllm.config import get_current_vllm_config

        # An explicit slot count wins over budget-based derivation and does
        # not require knowing the MoE layer count.
        if self.config.npu_cache_slots_per_layer is not None:
            return max(1, min(self.config.npu_cache_slots_per_layer, num_experts))

        model_config = get_current_vllm_config().model_config
        # vLLM resolves the text-side config for both plain LM and VL models
        # (whose HF config may nest it under text_config or other keys).
        text_config = getattr(model_config, "hf_text_config", None) or model_config.hf_config
        num_moe_layers = getattr(text_config, "num_hidden_layers", None)
        if num_moe_layers is None:
            raise RuntimeError(
                "HybriMoE: could not determine the number of MoE layers from the model config "
                f"(type={getattr(model_config.hf_config, 'model_type', '?')}). Set hybrimoe_config."
                "npu_cache_slots_per_layer explicitly to bypass budget-based derivation."
            )
        return derive_num_slots(self.config, num_experts, hidden_size, intermediate_size, num_moe_layers)

    # ------------------------------------------------------------------
    # Post-loading: dequant, initial slot fill, cache registration.
    # ------------------------------------------------------------------
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        runtime = HybriMoERuntime.get()
        num_experts = layer.global_num_experts
        num_slots = layer.w13_weight.shape[0]

        if runtime.config.host_store_bf16:
            layer.dequant_host_weights()

        # Initial cache content: experts 0..num_slots-1 (all MRS scores are
        # equal at this point, so the choice is arbitrary).
        layer.w13_weight.data.copy_(layer.host_w13_int8[:num_slots])
        layer.w2_weight.data.copy_(layer.host_w2_int8[:num_slots])
        layer.w13_weight_scale.data.copy_(layer.host_w13_scale[:num_slots])
        layer.w2_weight_scale.data.copy_(layer.host_w2_scale[:num_slots])

        layer.create_hybrimoe_staging_buffers()
        state = runtime.cache.register_layer(layer, num_experts, num_slots)
        for slot in range(num_slots):
            state.slot_to_expert[slot] = slot
            state.expert_to_slot[slot] = slot
        state.dispatcher = TokenDispatcherWithAllGather310(
            top_k=layer.top_k,
            num_experts=num_slots,
            num_local_experts=num_slots,
        )
        # Top-1 dispatcher for the wave-streaming (prefill) path: each
        # (token, expert) pair is dispatched as an independent token.
        state.dispatcher_top1 = TokenDispatcherWithAllGather310(
            top_k=1,
            num_experts=num_slots,
            num_local_experts=num_slots,
        )
        logger.info_once(
            "HybriMoE layer %s: %d experts, %d NPU cache slots",
            layer.layer_name,
            num_experts,
            num_slots,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        is_prefill: bool = True,
        enable_force_load_balance: bool = False,
        log2phy: torch.Tensor | None = None,
        global_redundant_expert_num: int = 0,
        pertoken_scale: Any | None = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        mc2_mask: torch.Tensor | None = None,
        tid2eid: Any | None = None,
    ) -> torch.Tensor:
        if getattr(layer, "zero_expert_num", 0) > 0:
            raise RuntimeError("HybriMoE does not support zero experts yet.")
        if expert_map is not None:
            raise RuntimeError("HybriMoE does not support expert parallelism (expert_map must be None).")
        if apply_router_weight_on_input:
            raise RuntimeError("HybriMoE does not support apply_router_weight_on_input yet.")

        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            global_num_experts=num_experts,
        )

        runtime = HybriMoERuntime.get()
        state = runtime.cache.get_layer(layer.layer_name)
        num_tokens = x.shape[0]
        # NOTE: the phase is decided by the token count; the `is_prefill`
        # argument is never set by the 310P forward path.
        if num_tokens > runtime.config.decode_token_threshold:
            out = self._streaming_forward(runtime, state, layer, x, topk_ids, topk_weights, router_logits, scoring_func)
        else:
            out = self._decode_forward(
                runtime, state, layer, x, topk_ids, topk_weights, router_logits, top_k, scoring_func
            )
        # forward_impl expects a FusedExpertsResult, not a bare tensor.
        return FusedExpertsResult(routed_out=out)

    # ------------------------------------------------------------------
    # Prefill / large-batch: stream experts through the slot cache in waves.
    # ------------------------------------------------------------------
    def _streaming_forward(
        self,
        runtime: HybriMoERuntime,
        state: HybriMoELayerState,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        router_logits: torch.Tensor,
        scoring_func: str,
    ) -> torch.Tensor:
        """Wave-based streaming forward.

        The activated expert set can exceed the slot count here, so experts
        are processed in waves of at most num_slots. Each wave batch-evicts
        and transfers its missing members (overlapping the previous wave's
        grouped matmul on the copy stream), then runs a compacted top-1
        dispatch over exactly this wave's (token, expert) pairs and
        accumulates into the output. Every activated pair is computed exactly
        once, so the result is mathematically identical to the full MoE.
        """
        cache = runtime.cache
        num_tokens = x.shape[0]
        top_k = topk_ids.shape[1]
        num_slots = state.num_slots
        main_stream = torch.npu.current_stream()
        timing = runtime.config.profile_phases
        t0 = time.perf_counter()

        # Single blocking D2H of the routing pack.
        flat_ids_pin = layer.pin_topk_ids[: num_tokens * top_k]
        flat_ids_pin.copy_(topk_ids.view(-1), non_blocking=True)
        flat_w_pin = layer.pin_topk_w[: num_tokens * top_k]
        flat_w_pin.copy_(topk_weights.view(-1), non_blocking=True)
        top_p_scores, top_p_ids = _routing_scores(router_logits, scoring_func).topk(layer.hybrimoe_top_p, dim=-1)
        layer.pin_top_p_ids[: num_tokens * layer.hybrimoe_top_p].copy_(top_p_ids.view(-1), non_blocking=True)
        layer.pin_top_p_scores[: num_tokens * layer.hybrimoe_top_p].copy_(top_p_scores.view(-1), non_blocking=True)
        main_stream.record_event().synchronize()
        if timing:
            runtime.record_phase("stream.d2h_sync", t0)
            t0 = time.perf_counter()

        self._mrs_update(runtime, state, layer, num_tokens)

        flat_ids = flat_ids_pin
        counts_tensor = torch.bincount(flat_ids, minlength=state.num_experts)
        activated = torch.nonzero(counts_tensor).flatten().tolist()
        resident_now = state.expert_to_slot.tolist()
        in_flight = set(state.in_flight)
        cached = [e for e in activated if resident_now[e] >= 0 or e in in_flight]
        missing = [e for e in activated if resident_now[e] < 0 and e not in in_flight]
        state.hits += len(cached)

        # Wave partition: resident experts first (no transfer), then missing
        # experts in slot-sized chunks. Waves are disjoint and cover all
        # activated experts exactly once.
        waves: list[list[int]] = []
        first = cached + missing[: max(0, num_slots - len(cached))]
        if first:
            waves.append(first)
        rest = missing[max(0, num_slots - len(cached)) :]
        for start in range(0, len(rest), num_slots):
            waves.append(rest[start : start + num_slots])

        # (token, expert) pair data on the host, shared by all waves.
        pair_tokens = torch.arange(num_tokens).repeat_interleave(top_k)
        out = torch.zeros_like(x)
        if timing:
            runtime.record_phase("stream.sched", t0)

        for wave_index, wave in enumerate(waves):
            t_wave = time.perf_counter() if timing else 0.0
            protected = set().union(*waves[wave_index:])  # current + future waves
            to_load = [e for e in wave if int(state.expert_to_slot[e].item()) < 0 and e not in state.in_flight]
            if to_load:
                cache.enqueue_transfers(state, to_load, protected, cache.copy_stream)
            events = cache.collect_transfer_events(state, wave)

            # Compact pair selection: exactly this wave's (token, expert) pairs.
            wave_member = torch.zeros(state.num_experts, dtype=torch.bool)
            wave_member[wave] = True
            positions = torch.nonzero(wave_member[flat_ids]).flatten()
            n = positions.numel()
            sel_tokens = pair_tokens[positions]
            sel_slots = state.expert_to_slot[flat_ids[positions]].to(torch.int32)
            sel_weights = flat_w_pin[positions]

            copy_stream = cache.copy_stream
            with torch.npu.stream(copy_stream):
                layer.dev_topk_ids[:n].copy_(sel_slots, non_blocking=True)
                layer.dev_topk_w[:n].copy_(sel_weights, non_blocking=True)
                layer.dev_pair_tokens[:n].copy_(sel_tokens, non_blocking=True)
                ready_event = copy_stream.record_event()
            main_stream.wait_event(ready_event)
            for event in events:
                main_stream.wait_event(event)
            if timing:
                runtime.record_phase("stream.wave_host", t_wave)
                t_wave = time.perf_counter()

            sel_tokens_npu = layer.dev_pair_tokens[:n]
            x_wave = x.index_select(0, sel_tokens_npu)
            dispatch_output = state.dispatcher_top1.token_dispatch(
                MoETokenDispatchInput(
                    hidden_states=x_wave,
                    topk_weights=layer.dev_topk_w[:n].view(n, 1),
                    topk_ids=layer.dev_topk_ids[:n].view(n, 1),
                    routing=MoERoutingParams(
                        expert_map=None,
                        global_redundant_expert_num=0,
                        mc2_mask=None,
                        apply_router_weight_on_input=False,
                    ),
                    quant=MoEQuantParams(quant_type=QuantType.W8A8),
                )
            )
            mlp_out = quant_apply_mlp(
                hidden_states=dispatch_output.hidden_states,
                w1=layer.w13_weight,
                w1_scale=layer.w13_weight_scale,
                w2=layer.w2_weight,
                w2_scale=layer.w2_weight_scale,
                group_list=dispatch_output.group_list,
                group_list_type=dispatch_output.group_list_type,
            )
            wave_out = state.dispatcher_top1.token_combine(mlp_out, dispatch_output.combine_metadata)
            out.index_add_(0, sel_tokens_npu, wave_out)
            if timing:
                runtime.record_phase("stream.wave_npu", t_wave)

        # Cross-layer prefetch: stream the next layers' predicted experts
        # while the rest of this layer (attention etc.) runs.
        if runtime.prefetcher is not None:
            t_prefetch = time.perf_counter() if timing else 0.0
            runtime.prefetcher.maybe_prefetch(state, x, top_k, scoring_func, prefill=True)
            if timing:
                runtime.record_phase("stream.prefetch", t_prefetch)
        if timing:
            runtime.record_phase("stream.total", t0)
            runtime.tick_phase_log()
        return out

    # ------------------------------------------------------------------
    # Decode: HSS hybrid CPU/NPU scheduling.
    # ------------------------------------------------------------------
    def _decode_forward(
        self,
        runtime: HybriMoERuntime,
        state: HybriMoELayerState,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        scoring_func: str,
    ) -> torch.Tensor:
        cache = runtime.cache
        num_tokens = x.shape[0]
        num_experts = state.num_experts
        executor = runtime.get_executor(x.shape[1], dtype=layer.params_dtype)
        buffer_index = executor.next_buffer()
        main_stream = torch.npu.current_stream()
        timing = runtime.config.profile_phases
        t0 = time.perf_counter()
        t_total = t0

        # --- Async D2H of hidden states + routing pack; one host sync. ---
        # The hidden states are only consumed by the CPU experts.
        if runtime.config.enable_cpu_experts:
            in_buf = executor.in_buffer(buffer_index)
            in_buf[:num_tokens].copy_(x, non_blocking=True)
        flat_ids_pin = layer.pin_topk_ids[: num_tokens * top_k]
        flat_ids_pin.copy_(topk_ids.view(-1), non_blocking=True)
        flat_w_pin = layer.pin_topk_w[: num_tokens * top_k]
        flat_w_pin.copy_(topk_weights.view(-1), non_blocking=True)
        top_p_scores, top_p_ids = _routing_scores(router_logits, scoring_func).topk(layer.hybrimoe_top_p, dim=-1)
        layer.pin_top_p_ids[: num_tokens * layer.hybrimoe_top_p].copy_(top_p_ids.view(-1), non_blocking=True)
        layer.pin_top_p_scores[: num_tokens * layer.hybrimoe_top_p].copy_(top_p_scores.view(-1), non_blocking=True)
        main_stream.record_event().synchronize()
        if timing:
            runtime.record_phase("decode.d2h_sync", t0)
            t0 = time.perf_counter()

        # --- HSS scheduling decision (host). ---
        flat_ids = flat_ids_pin
        counts_tensor = torch.bincount(flat_ids, minlength=state.num_experts)
        activated = torch.nonzero(counts_tensor).flatten().tolist()
        if not runtime.config.enable_cpu_experts and len(activated) > state.num_slots:
            # Without a CPU fallback, an over-large activated set cannot be
            # handled by the slot cache alone; stream this layer in waves.
            return self._streaming_forward(
                runtime, state, layer, x, topk_ids, topk_weights, router_logits, scoring_func
            )

        self._mrs_update(runtime, state, layer, num_tokens)

        counts = {e: int(counts_tensor[e].item()) for e in activated}
        resident = state.expert_to_slot.tolist()
        in_flight = set(state.in_flight)
        cached = [e for e in activated if resident[e] >= 0 or e in in_flight]
        uncached = [e for e in activated if resident[e] < 0 and e not in in_flight]
        result = hss_schedule(cached, uncached, counts, runtime.cost_model)
        cpu_expert_set = set(result.cpu_experts)
        if not runtime.config.enable_cpu_experts:
            # The CPU is slower than a transfer on this machine: every
            # activated expert is computed on the NPU; uncached ones are
            # transferred instead.
            result.npu_experts.extend(result.cpu_experts)
            result.cpu_experts = []
            cpu_expert_set = set()
        if timing:
            runtime.record_phase("decode.sched", t0)
            t0 = time.perf_counter()

        # --- Transfers for NPU-assigned but uncached experts. ---
        uncached_set = set(uncached)
        protected = set(activated)
        to_move = [e for e in result.npu_experts if e in uncached_set]
        # Slot-capacity guard: if the activated set is too large to evict
        # enough non-activated residents (large decode batches), the excess
        # transfers fall back to CPU compute instead of thrashing the cache.
        n_resident = sum(1 for s in resident if s >= 0)
        evictable = sum(1 for e, s in enumerate(resident) if s >= 0 and e not in protected)
        capacity = (state.num_slots - n_resident) + evictable
        if len(to_move) > capacity:
            overflow = to_move[capacity:]
            result.npu_experts = [e for e in result.npu_experts if e not in set(overflow)]
            result.cpu_experts.extend(overflow)
            cpu_expert_set = set(result.cpu_experts)
            to_move = to_move[:capacity]
        cache.enqueue_transfers(state, to_move, protected, cache.copy_stream)
        transfer_events = cache.collect_transfer_events(state, result.npu_experts)

        # --- Slot remap + zero-weight sentinel for CPU-bound pairs. ---
        is_cpu_expert = torch.zeros(num_experts, dtype=torch.bool)
        if cpu_expert_set:
            is_cpu_expert[list(cpu_expert_set)] = True
        cpu_mask = is_cpu_expert[flat_ids]
        slot_ids = state.expert_to_slot[flat_ids]
        npu_ids_host = torch.where(cpu_mask, SENTINEL_SLOT, slot_ids).to(torch.int32)
        npu_w_host = torch.where(cpu_mask, 0.0, flat_w_pin)

        copy_stream = cache.copy_stream
        with torch.npu.stream(copy_stream):
            layer.dev_topk_ids[: num_tokens * top_k].copy_(npu_ids_host, non_blocking=True)
            layer.dev_topk_w[: num_tokens * top_k].copy_(npu_w_host, non_blocking=True)
            ready_event = copy_stream.record_event()
        if timing:
            runtime.record_phase("decode.transfer_remap", t0)
            t0 = time.perf_counter()

        # --- CPU experts: asynchronous, overlapped with the NPU below. ---
        handle = None
        if cpu_expert_set:
            assignments = _build_cpu_assignments(flat_ids, flat_w_pin, cpu_mask, result.cpu_experts, num_tokens, top_k)
            handle = executor.submit(layer, assignments, buffer_index, num_tokens)
        if timing:
            runtime.record_phase("decode.cpu_submit", t0)
            t0 = time.perf_counter()

        # --- NPU experts. ---
        main_stream.wait_event(ready_event)
        for event in transfer_events:
            main_stream.wait_event(event)
        if result.npu_experts:
            npu_topk_ids = layer.dev_topk_ids[: num_tokens * top_k].view(num_tokens, top_k)
            npu_topk_w = layer.dev_topk_w[: num_tokens * top_k].view(num_tokens, top_k)
            out = self._npu_fused(state, layer, x, npu_topk_ids, npu_topk_w)
        else:
            out = torch.zeros_like(x)
        if timing:
            runtime.record_phase("decode.npu_launch", t0)
            t0 = time.perf_counter()

        # --- Impact-driven prefetch for subsequent layers. ---
        if runtime.prefetcher is not None:
            runtime.prefetcher.maybe_prefetch(state, x, top_k, scoring_func, prefill=False)
        if timing:
            runtime.record_phase("decode.prefetch", t0)
            t0 = time.perf_counter()

        # --- Combine the CPU partial result. ---
        if handle is not None:
            h2d_event = handle.wait_and_h2d(copy_stream, executor.consume_event(buffer_index))
            main_stream.wait_event(h2d_event)
            out = out + handle.out_npu[:num_tokens].to(out.dtype)
            executor.mark_consumed(buffer_index, main_stream.record_event())
        if timing:
            runtime.record_phase("decode.cpu_wait_combine", t0)
            runtime.record_phase("decode.total", t_total)
            runtime.tick_phase_log()
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _mrs_update(
        self, runtime: HybriMoERuntime, state: HybriMoELayerState, layer: torch.nn.Module, num_tokens: int
    ) -> None:
        """MRS score update from the staged top-p routing scores.

        Scores are aggregated across tokens with a per-expert max before the
        top-p selection (the reference implementation is batch-1; this is the
        batched generalization).
        """
        top_p = layer.hybrimoe_top_p
        ids = layer.pin_top_p_ids[: num_tokens * top_p]
        scores = layer.pin_top_p_scores[: num_tokens * top_p]
        aggregated = torch.zeros(state.num_experts, dtype=torch.float32)
        aggregated.index_reduce_(0, ids, scores, "amax", include_self=False)
        top_scores, top_ids = aggregated.topk(min(top_p, state.num_experts))
        state.mrs_update(top_ids, top_scores, runtime.config.alpha)

    def _npu_fused(
        self,
        state: HybriMoELayerState,
        layer: torch.nn.Module,
        x: torch.Tensor,
        npu_topk_ids: torch.Tensor,
        npu_topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Grouped-matmul over the NPU-resident slots of one MoE layer."""
        dispatch_output = state.dispatcher.token_dispatch(
            MoETokenDispatchInput(
                hidden_states=x,
                topk_weights=npu_topk_weights,
                topk_ids=npu_topk_ids,
                routing=MoERoutingParams(
                    expert_map=None,
                    global_redundant_expert_num=0,
                    mc2_mask=None,
                    apply_router_weight_on_input=False,
                ),
                quant=MoEQuantParams(quant_type=QuantType.W8A8),
            )
        )
        mlp_out = quant_apply_mlp(
            hidden_states=dispatch_output.hidden_states,
            w1=layer.w13_weight,
            w1_scale=layer.w13_weight_scale,
            w2=layer.w2_weight,
            w2_scale=layer.w2_weight_scale,
            group_list=dispatch_output.group_list,
            group_list_type=dispatch_output.group_list_type,
        )
        return state.dispatcher.token_combine(mlp_out, dispatch_output.combine_metadata)


def _routing_scores(router_logits: torch.Tensor, scoring_func: str) -> torch.Tensor:
    """Full routing scores per token, matching the model's scoring function."""
    logits = router_logits.float()
    if scoring_func == "sigmoid":
        return torch.sigmoid(logits)
    # "softmax" (default) and any other function fall back to softmax; the MRS
    # scores only need to be comparable within one layer.
    return torch.softmax(logits, dim=-1)


def _build_cpu_assignments(
    flat_ids: torch.Tensor,
    flat_weights: torch.Tensor,
    cpu_mask: torch.Tensor,
    cpu_experts: list[int],
    num_tokens: int,
    top_k: int,
) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
    """Per-expert (token indices, routing weights) lists for the CPU worker."""
    cpu_positions = torch.nonzero(cpu_mask).flatten()
    cpu_expert_ids = flat_ids[cpu_positions]
    token_of_position = torch.arange(num_tokens).repeat_interleave(top_k)
    assignments = []
    for expert in cpu_experts:
        selected = cpu_positions[cpu_expert_ids == expert]
        assignments.append((expert, token_of_position[selected], flat_weights[selected]))
    return assignments


class AscendHybriMoEMethod(AscendFusedMoEMethod):
    """Marker adapter wrapping AscendHybriMoEW8A8DynamicScheme310."""


class AscendHybriMoEFusedMoE310(AscendFusedMoE310):
    """FusedMoE layer with host-offloaded expert weights for HybriMoE.

    The upstream weight loader drives `weight_loader` once per expert shard;
    this override routes every shard into pinned host buffers instead of the
    (slot-shaped) NPU parameters.
    """

    is_hybrimoe_layer = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.quant_method, AscendHybriMoEMethod):
            raise RuntimeError(
                "HybriMoE requires the W8A8 (W8A8_DYNAMIC) ModelSlim quantization for MoE layers, "
                f"but layer {self.layer_name} uses {self.quant_method.__class__.__name__}."
            )
        config = get_ascend_config().hybrimoe_config
        num_experts = self.global_num_experts
        hidden_size = self.hidden_size
        intermediate_size = self.intermediate_size_per_partition
        # Host weight buffers (source of truth for H2D transfers). NOTE: the
        # default device is the NPU during model construction, so the device
        # must be pinned to CPU explicitly here.
        self.host_w13_int8 = torch.empty(
            num_experts, 2 * intermediate_size, hidden_size, dtype=torch.int8, device="cpu"
        )
        self.host_w2_int8 = torch.empty(num_experts, hidden_size, intermediate_size, dtype=torch.int8, device="cpu")
        self.host_w13_scale = torch.empty(num_experts, 2 * intermediate_size, dtype=torch.float32, device="cpu")
        self.host_w2_scale = torch.empty(num_experts, hidden_size, dtype=torch.float32, device="cpu")
        self.host_w13_offset = torch.zeros(num_experts, 2 * intermediate_size, dtype=torch.float32, device="cpu")
        self.host_w2_offset = torch.zeros(num_experts, hidden_size, dtype=torch.float32, device="cpu")
        # Pin the multi-GB buffers for async H2D (default on; disable via
        # hybrimoe_config.pin_host_weights if the driver rejects large
        # registered-memory regions).
        if config.pin_host_weights:
            self.host_w13_int8 = pin_memory_if_available(self.host_w13_int8)
            self.host_w2_int8 = pin_memory_if_available(self.host_w2_int8)
            self.host_w13_scale = pin_memory_if_available(self.host_w13_scale)
            self.host_w2_scale = pin_memory_if_available(self.host_w2_scale)
        # Dequantized copies for CPU compute, in the model's params_dtype
        # (float16 on 310P; pageable, never go H2D).
        if config.host_store_bf16:
            self.host_w13_dequant = torch.empty(
                num_experts, 2 * intermediate_size, hidden_size, dtype=self.params_dtype, device="cpu"
            )
            self.host_w2_dequant = torch.empty(
                num_experts, hidden_size, intermediate_size, dtype=self.params_dtype, device="cpu"
            )

    # ------------------------------------------------------------------
    # Weight loading interception
    # ------------------------------------------------------------------
    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        # TP=1 / EP=1: the global expert id is the local one; no shard
        # splitting is needed beyond the w1/w3 halves of w13.
        weight = loaded_weight.detach().cpu()
        intermediate_size = self.intermediate_size_per_partition
        if param is self.w13_weight:
            if shard_id == "w1":
                self.host_w13_int8[expert_id, :intermediate_size].copy_(weight)
            elif shard_id == "w3":
                self.host_w13_int8[expert_id, intermediate_size:].copy_(weight)
            else:
                raise ValueError(f"HybriMoE: unexpected shard_id {shard_id!r} for w13_weight")
        elif param is self.w2_weight:
            self.host_w2_int8[expert_id].copy_(weight)
        elif param is self.w13_weight_scale:
            flat = weight.reshape(-1).float()
            if shard_id == "w1":
                self.host_w13_scale[expert_id, :intermediate_size].copy_(flat)
            elif shard_id == "w3":
                self.host_w13_scale[expert_id, intermediate_size:].copy_(flat)
            else:
                raise ValueError(f"HybriMoE: unexpected shard_id {shard_id!r} for w13_weight_scale")
        elif param is self.w2_weight_scale:
            self.host_w2_scale[expert_id].copy_(weight.reshape(-1).float())
        elif param is self.w13_weight_offset:
            flat = weight.reshape(-1).float()
            if shard_id == "w1":
                self.host_w13_offset[expert_id, :intermediate_size].copy_(flat)
            elif shard_id == "w3":
                self.host_w13_offset[expert_id, intermediate_size:].copy_(flat)
        elif param is self.w2_weight_offset:
            self.host_w2_offset[expert_id].copy_(weight.reshape(-1).float())
        else:
            logger.warning_once("HybriMoE: weight_loader received unexpected param for %s; skipped.", weight_name)
            return False if return_success else None
        return True if return_success else None

    # ------------------------------------------------------------------
    # Post-loading helpers (called from the scheme)
    # ------------------------------------------------------------------
    def dequant_host_weights(self) -> None:
        """Dequantize the host int8 weights into params_dtype for CPU compute."""
        self.host_w13_dequant.copy_(
            dequant_int8_per_channel(self.host_w13_int8, self.host_w13_scale, out_dtype=self.params_dtype)
        )
        self.host_w2_dequant.copy_(
            dequant_int8_per_channel(self.host_w2_int8, self.host_w2_scale, out_dtype=self.params_dtype)
        )

    def create_hybrimoe_staging_buffers(self) -> None:
        """Pinned host + NPU staging buffers for the per-forward routing pack."""
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        top_k = self.top_k
        top_p = min(get_ascend_config().hybrimoe_config.top_p_factor * top_k, self.global_num_experts)
        self.hybrimoe_top_p = top_p
        self.pin_topk_ids = pin_memory_if_available(torch.empty(max_tokens * top_k, dtype=torch.int64, device="cpu"))
        self.pin_topk_w = pin_memory_if_available(torch.empty(max_tokens * top_k, dtype=torch.float32, device="cpu"))
        self.pin_top_p_ids = pin_memory_if_available(torch.empty(max_tokens * top_p, dtype=torch.int64, device="cpu"))
        self.pin_top_p_scores = pin_memory_if_available(
            torch.empty(max_tokens * top_p, dtype=torch.float32, device="cpu")
        )
        self.dev_topk_ids = torch.empty(max_tokens * top_k, dtype=torch.int32, device="npu")
        self.dev_topk_w = torch.empty(max_tokens * top_k, dtype=self.params_dtype, device="npu")
        # Pair->token index buffer for the wave-streaming (prefill) path.
        self.dev_pair_tokens = torch.empty(max_tokens * top_k, dtype=torch.int64, device="npu")
