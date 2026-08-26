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
"""Small shared helpers for the HybriMoE module."""

from __future__ import annotations

import torch


def pin_memory_if_available(tensor: torch.Tensor) -> torch.Tensor:
    """Pin a host tensor when the platform supports it, else return as-is.

    Pinned memory enables true async H2D/D2H copies on the 310P; the fallback
    keeps unit tests on CPU-only machines working.
    """
    try:
        return tensor.pin_memory()
    except RuntimeError:
        return tensor


# Number of experts dequantized per chunk; bounds the fp32 temporary memory
# to a few hundred MB.
_DEQUANT_EXPERT_CHUNK = 8


def dequant_int8_per_channel(
    w_int8: torch.Tensor,
    scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    chunk: int = _DEQUANT_EXPERT_CHUNK,
) -> torch.Tensor:
    """Dequantize per-channel int8 weights: w = q * scale.

    Args:
        w_int8: [E, out, in] int8 weights.
        scale: [E, out] fp32 per-output-channel scales.
        out_dtype: dtype of the dequantized weights (the model's params_dtype,
            float16 on 310P).
        chunk: experts per chunk (bounds the fp32 temporary).

    Returns:
        [E, out, in] dequantized weights in `out_dtype`.
    """
    num_experts = w_int8.shape[0]
    out = torch.empty(num_experts, *w_int8.shape[1:], dtype=out_dtype, device=w_int8.device)
    for start in range(0, num_experts, chunk):
        end = min(start + chunk, num_experts)
        dequantized = w_int8[start:end].float() * scale[start:end].unsqueeze(-1)
        out[start:end].copy_(dequantized.to(out_dtype))
    return out
