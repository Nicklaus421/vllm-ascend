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
"""Configuration for HybriMoE hybrid CPU-NPU MoE inference on Ascend 310P.

This module must stay free of torch / torch_npu imports so that it can be
imported from vllm_ascend.ascend_config on any platform.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class HybriMoEConfig:
    """Configuration Object for hybrimoe_config from additional_config.

    Usage (online)::

        vllm serve <model> --additional-config \
            '{"hybrimoe_config": {"enabled": true, "npu_cache_budget_gb": 8.0}}'

    Usage (offline)::

        llm = LLM(model, additional_config={"hybrimoe_config": {"enabled": true}})
    """

    def __init__(self, config: dict | None = None):
        if config is None:
            config = {}
        self.enabled: bool = bool(config.get("enabled", False))
        # NPU-side expert cache capacity. If npu_cache_slots_per_layer is set
        # it takes precedence; otherwise slots are derived from the budget at
        # model-build time.
        self.npu_cache_budget_gb: float = float(config.get("npu_cache_budget_gb", 8.0))
        self.npu_cache_slots_per_layer: int | None = config.get("npu_cache_slots_per_layer")
        # MRS (score-aware caching) parameters:
        #   S = alpha * TopP(routing_scores) + (1 - alpha) * S, P = top_p_factor * top_k
        self.alpha: float = float(config.get("alpha", 0.5))
        self.top_p_factor: int = int(config.get("top_p_factor", 2))
        # Impact-driven prefetching: reuse the current hidden states through the
        # gates of the next `prefetch_lookahead` layers and prefetch up to
        # `prefetch_size` experts per forward.
        self.prefetch_lookahead: int = int(config.get("prefetch_lookahead", 3))
        self.prefetch_size: int = int(config.get("prefetch_size", 2))
        # 0 means "use all physical cores".
        self.num_cpu_threads: int = int(config.get("num_cpu_threads", 0))
        # Store a dequantized copy of expert weights (in the model's
        # params_dtype) on the host for CPU compute. False halves host memory
        # at the cost of on-the-fly dequant.
        self.host_store_bf16: bool = bool(config.get("host_store_bf16", True))
        # Allow the HSS scheduler to assign experts to CPU compute. On hosts
        # whose CPU is slower than an H2D transfer (e.g. 310P boards with weak
        # host CPUs), disabling this is strictly faster: uncached experts are
        # always transferred to the NPU instead.
        self.enable_cpu_experts: bool = bool(config.get("enable_cpu_experts", True))
        # Run the impact-driven prefetcher only every N forwards (predictions
        # stay valid across steps; the host-side simulation is not free).
        self.prefetch_interval: int = int(config.get("prefetch_interval", 1))
        # Pin the (multi-GB) int8 host weight buffers for async H2D. On hosts
        # with ample RAM this is strongly recommended: pageable copies block
        # the host per expert and are far slower. Disable only if the 310P
        # driver rejects large registered-memory regions.
        self.pin_host_weights: bool = bool(config.get("pin_host_weights", True))
        # Per-phase host-side timing of the HybriMoE forward path, logged
        # periodically. For performance debugging only.
        self.profile_phases: bool = bool(config.get("profile_phases", False))
        # Token count at or below which a forward is treated as decode and the
        # hybrid CPU/NPU schedule is applied.
        self.decode_token_threshold: int = int(config.get("decode_token_threshold", 32))
        # Cost-model calibration run at startup (measures NPU/CPU compute and
        # H2D transfer times on the actual machine).
        calibration = config.get("calibration", {})
        self.calibration_enabled: bool = bool(calibration.get("enabled", True))
        self.calibration_cache_path: str | None = calibration.get("cache_path", None)
        self.calibration_iterations: int = int(calibration.get("iterations", 20))
        self._validate()

    def _validate(self):
        if self.npu_cache_slots_per_layer is not None and self.npu_cache_slots_per_layer <= 0:
            raise ValueError(
                "hybrimoe_config.npu_cache_slots_per_layer must be a positive integer, "
                f"got {self.npu_cache_slots_per_layer}"
            )
        if self.npu_cache_budget_gb <= 0:
            raise ValueError(f"hybrimoe_config.npu_cache_budget_gb must be positive, got {self.npu_cache_budget_gb}")
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError(f"hybrimoe_config.alpha must be in (0, 1], got {self.alpha}")
        if self.top_p_factor < 1:
            raise ValueError(f"hybrimoe_config.top_p_factor must be >= 1, got {self.top_p_factor}")
        if self.prefetch_lookahead < 0:
            raise ValueError(f"hybrimoe_config.prefetch_lookahead must be >= 0, got {self.prefetch_lookahead}")
        if self.prefetch_size < 0:
            raise ValueError(f"hybrimoe_config.prefetch_size must be >= 0, got {self.prefetch_size}")
        if self.prefetch_interval < 1:
            raise ValueError(f"hybrimoe_config.prefetch_interval must be >= 1, got {self.prefetch_interval}")
        if self.num_cpu_threads < 0:
            raise ValueError(f"hybrimoe_config.num_cpu_threads must be >= 0, got {self.num_cpu_threads}")
        if self.decode_token_threshold < 1:
            raise ValueError(f"hybrimoe_config.decode_token_threshold must be >= 1, got {self.decode_token_threshold}")

    def validate_against_vllm_config(self, vllm_config: "VllmConfig"):
        """Hardware / parallelism guards. Called when the config is enabled."""
        from vllm_ascend.utils import is_310p

        if not is_310p():
            raise RuntimeError("HybriMoE is only supported on Ascend 310P.")
        parallel_config = vllm_config.parallel_config
        if parallel_config.tensor_parallel_size > 1:
            raise RuntimeError("HybriMoE does not support tensor parallelism. Please set --tensor-parallel-size to 1.")
        if parallel_config.pipeline_parallel_size > 1:
            raise RuntimeError(
                "HybriMoE does not support pipeline parallelism. Please set --pipeline-parallel-size to 1."
            )
        if parallel_config.enable_expert_parallel:
            raise RuntimeError("HybriMoE does not support expert parallelism. Please remove --enable-expert-parallel.")
        if vllm_config.speculative_config is not None:
            raise RuntimeError("HybriMoE is not compatible with speculative decoding yet.")


def hybrimoe_enabled_from_additional_config(additional_config: dict[str, Any] | None) -> bool:
    """Lightweight enable check usable before AscendConfig is initialized."""
    from vllm_ascend import envs

    if envs.VLLM_ASCEND_HYBRIMOE_ENABLED:
        return True
    if not additional_config:
        return False
    return bool(additional_config.get("hybrimoe_config", {}).get("enabled", False))
