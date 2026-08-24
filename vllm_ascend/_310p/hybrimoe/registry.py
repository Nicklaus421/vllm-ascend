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
"""Registry mapping HybriMoE MoE layers to their position and router gate.

Built once after model load by walking the model's module tree. The gate of
layer i is used by the impact-driven prefetcher to predict expert
activations of subsequent layers from the current hidden states.
"""

from __future__ import annotations

import torch
from vllm.logger import logger

from .cache import HybriMoECache


class HybriMoERegistry:
    def __init__(self):
        # MoE layers in execution order.
        self.layer_names: list[str] = []
        # Router gate modules aligned with layer_names (a ReplicatedLinear;
        # called as gate(x) -> (logits, bias)).
        self.gate_modules: list[torch.nn.Module] = []
        self._position: dict[str, int] = {}

    def position_of(self, layer_name: str) -> int:
        return self._position[layer_name]

    def next_positions(self, position: int, lookahead: int) -> list[int]:
        return [p for p in range(position + 1, min(position + 1 + lookahead, len(self.layer_names)))]


def build_moe_registry(model: torch.nn.Module, cache: HybriMoECache) -> HybriMoERegistry:
    registry = HybriMoERegistry()
    modules = dict(model.named_modules())
    for name, module in modules.items():
        if not getattr(module, "is_hybrimoe_layer", False):
            continue
        if name not in cache.layers and module.layer_name not in cache.layers:
            raise RuntimeError(f"HybriMoE layer {name} was not registered with the cache during weight loading.")
        layer_name = module.layer_name
        gate = _find_gate(modules, name, layer_name)
        registry._position[layer_name] = len(registry.layer_names)
        registry.layer_names.append(layer_name)
        registry.gate_modules.append(gate)
    if not registry.layer_names:
        logger.warning("HybriMoE registry found no HybriMoE MoE layers in the model.")
    return registry


def _find_gate(modules: dict[str, torch.nn.Module], module_name: str, layer_name: str) -> torch.nn.Module:
    """Locate the router gate sibling of a FusedMoE layer.

    For qwen3_5_moe / qwen3_moe style models the FusedMoE lives at
    ``<...>.mlp.experts`` and the gate at ``<...>.mlp.gate``.
    """
    candidates = []
    for prefix in (module_name, layer_name):
        if prefix.endswith(".experts"):
            candidates.append(prefix[: -len(".experts")] + ".gate")
    for candidate in candidates:
        gate = modules.get(candidate)
        if gate is not None:
            return gate
    raise RuntimeError(
        f"HybriMoE: could not locate the router gate for MoE layer {layer_name!r} "
        f"(tried {candidates}). The impact-driven prefetcher requires the gate module."
    )
