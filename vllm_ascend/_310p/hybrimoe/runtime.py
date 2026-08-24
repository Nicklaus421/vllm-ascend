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
"""Process-wide HybriMoE runtime: cache + scheduler + executor + prefetcher.

The runtime is created lazily on first use (model build triggers weight
loading, which already needs the cache) and fully initialized by
initialize_post_load() once the model is loaded: builds the gate registry,
calibrates the cost model and creates the CPU executor / prefetcher.
"""

from __future__ import annotations

import torch
from vllm.logger import logger

from .cache import HybriMoECache
from .config import HybriMoEConfig
from .cpu_worker import CPUExpertExecutor
from .scheduler import CostModel, calibrate_cost_model
from .utils import pin_memory_if_available


class HybriMoERuntime:
    _instance: HybriMoERuntime | None = None

    def __init__(self, config: HybriMoEConfig):
        self.config = config
        self.cache = HybriMoECache(config)
        self.cost_model = CostModel.load_or_default(config.calibration_cache_path)
        self.executor: CPUExpertExecutor | None = None
        self.prefetcher = None
        self.registry = None

    @classmethod
    def get(cls) -> HybriMoERuntime:
        if cls._instance is None:
            from vllm_ascend.ascend_config import get_ascend_config

            config = get_ascend_config().hybrimoe_config
            if not config.enabled:
                raise RuntimeError("HybriMoERuntime requested but hybrimoe_config.enabled is False.")
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    # ------------------------------------------------------------------
    # Lazy component access
    # ------------------------------------------------------------------
    def get_executor(self, hidden_size: int) -> CPUExpertExecutor:
        if self.executor is None:
            self.executor = CPUExpertExecutor(
                hidden_size=hidden_size,
                max_tokens=self.config.decode_token_threshold,
                num_cpu_threads=self.config.num_cpu_threads,
                host_store_bf16=self.config.host_store_bf16,
            )
        return self.executor

    # ------------------------------------------------------------------
    # Post-load initialization (called once by the model runner)
    # ------------------------------------------------------------------
    def initialize_post_load(self, model: torch.nn.Module) -> None:
        from .prefetch import ImpactDrivenPrefetcher
        from .registry import build_moe_registry

        self.registry = build_moe_registry(model, self.cache)
        if not self.registry.layer_names:
            raise RuntimeError(
                "HybriMoE is enabled but no HybriMoE MoE layers were found in the model. "
                "Check that the model is a W8A8-quantized MoE model running on Ascend 310P."
            )
        first_layer = self.cache.get_layer(self.registry.layer_names[0]).layer
        hidden_size = first_layer.host_w13_int8.shape[2]
        intermediate_size = first_layer.host_w13_int8.shape[1] // 2

        if self.config.calibration_enabled:
            self.cost_model = self._calibrate(hidden_size, intermediate_size)
            if self.config.calibration_cache_path:
                self.cost_model.save(self.config.calibration_cache_path)
        logger.info("HybriMoE cost model: %s", self.cost_model.to_json())

        # The executor is created lazily on the first decode forward, but
        # creating it here surfaces allocation problems at startup.
        self.get_executor(hidden_size)

        if self.config.prefetch_size > 0 and self.config.prefetch_lookahead > 0:
            self.prefetcher = ImpactDrivenPrefetcher(self.config, self.cache, self.registry, self.cost_model)
        logger.info(
            "HybriMoE initialized: %d MoE layers, %d experts/layer, %d NPU slots/layer",
            len(self.registry.layer_names),
            self.cache.get_layer(self.registry.layer_names[0]).num_experts,
            self.cache.get_layer(self.registry.layer_names[0]).num_slots,
        )

    # ------------------------------------------------------------------
    # Cost-model calibration on the actual machine
    # ------------------------------------------------------------------
    def _calibrate(self, hidden_size: int, intermediate_size: int) -> CostModel:
        from vllm_ascend._310p.fused_moe.moe_mlp import quant_apply_mlp

        iterations = self.config.calibration_iterations

        # NPU per-expert MLP timing (W8A8 dynamic grouped matmul, 1 expert).
        npu_w13 = torch.randint(-8, 8, (1, 2 * intermediate_size, hidden_size), dtype=torch.int8, device="npu")
        npu_w2 = torch.randint(-8, 8, (1, hidden_size, intermediate_size), dtype=torch.int8, device="npu")
        npu_w13_scale = torch.ones(1, 2 * intermediate_size, dtype=torch.float32, device="npu")
        npu_w2_scale = torch.ones(1, hidden_size, dtype=torch.float32, device="npu")

        def npu_mlp_fn(token_num: int) -> None:
            x = torch.randn(token_num, hidden_size, dtype=torch.bfloat16, device="npu")
            group_list = torch.tensor([token_num], dtype=torch.int64, device="npu")
            quant_apply_mlp(
                hidden_states=x,
                w1=npu_w13,
                w1_scale=npu_w13_scale,
                w2=npu_w2,
                w2_scale=npu_w2_scale,
                group_list=group_list,
                group_list_type=0,
            )
            torch.npu.synchronize()

        # CPU per-expert MLP timing (bf16 dequantized weights).
        cpu_w13 = torch.randn(2 * intermediate_size, hidden_size, dtype=torch.bfloat16)
        cpu_w2 = torch.randn(hidden_size, intermediate_size, dtype=torch.bfloat16)

        def cpu_mlp_fn(token_num: int) -> None:
            x = torch.randn(token_num, hidden_size, dtype=torch.bfloat16)
            gate = x @ cpu_w13[:intermediate_size].t()
            up = x @ cpu_w13[intermediate_size:].t()
            hidden = torch.nn.functional.silu(gate) * up
            _ = hidden @ cpu_w2.t()

        # H2D transfer timing for one expert (int8 weights + fp32 scales).
        host_w13 = pin_memory_if_available(torch.empty(2 * intermediate_size, hidden_size, dtype=torch.int8))
        host_w2 = pin_memory_if_available(torch.empty(hidden_size, intermediate_size, dtype=torch.int8))
        dev_w13 = torch.empty_like(npu_w13)
        dev_w2 = torch.empty_like(npu_w2)

        def move_fn() -> None:
            dev_w13[0].copy_(host_w13, non_blocking=True)
            dev_w2[0].copy_(host_w2, non_blocking=True)
            torch.npu.synchronize()

        return calibrate_cost_model(
            expert_shapes=(hidden_size, intermediate_size),
            iterations=iterations,
            cpu_mlp_fn=cpu_mlp_fn,
            npu_mlp_fn=npu_mlp_fn,
            move_fn=move_fn,
        )
