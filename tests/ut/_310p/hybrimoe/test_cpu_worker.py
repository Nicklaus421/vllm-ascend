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
"""Unit tests for the HybriMoE CPU expert executor (numerics, host-only)."""

from unittest.mock import MagicMock

import torch
import torch.nn.functional as F

from vllm_ascend._310p.hybrimoe.cpu_worker import CPUExpertExecutor

HIDDEN = 16
INTERMEDIATE = 8
MAX_TOKENS = 4


class _FakeLayer:
    def __init__(self, num_experts: int):
        self.host_w13_dequant = torch.randn(num_experts, 2 * INTERMEDIATE, HIDDEN, dtype=torch.bfloat16)
        self.host_w2_dequant = torch.randn(num_experts, HIDDEN, INTERMEDIATE, dtype=torch.bfloat16)


def _reference_mlp(layer, expert: int, rows: torch.Tensor) -> torch.Tensor:
    w13 = layer.host_w13_dequant[expert]
    gate = rows @ w13[:INTERMEDIATE].t()
    up = rows @ w13[INTERMEDIATE:].t()
    return (F.silu(gate) * up) @ layer.host_w2_dequant[expert].t()


def test_cpu_executor_matches_reference():
    torch.manual_seed(0)
    layer = _FakeLayer(num_experts=4)
    executor = CPUExpertExecutor(hidden_size=HIDDEN, max_tokens=MAX_TOKENS, num_cpu_threads=2, device="cpu")
    num_tokens = 3
    buffer_index = executor.next_buffer()
    x = torch.randn(num_tokens, HIDDEN, dtype=torch.bfloat16)
    executor.in_buffer(buffer_index)[:num_tokens].copy_(x)

    # token 0 -> experts 1, 2; token 1 -> expert 2; token 2 -> experts 1, 3
    assignments = [
        (1, torch.tensor([0, 2]), torch.tensor([0.5, 0.25])),
        (2, torch.tensor([0, 1]), torch.tensor([0.5, 1.0])),
        (3, torch.tensor([2]), torch.tensor([0.75])),
    ]
    handle = executor.submit(layer, assignments, buffer_index, num_tokens)
    copy_stream = MagicMock()
    handle.wait_and_h2d(copy_stream, None)
    result = handle.out_npu[:num_tokens]

    expected = torch.zeros(num_tokens, HIDDEN)
    for expert, token_idx, weights in assignments:
        for t, w in zip(token_idx.tolist(), weights.tolist()):
            expected[t] += _reference_mlp(layer, expert, x[t : t + 1]).float().squeeze(0) * w
    assert torch.allclose(result, expected, atol=2e-2, rtol=2e-2)


def test_cpu_executor_empty_assignments():
    layer = _FakeLayer(num_experts=2)
    executor = CPUExpertExecutor(hidden_size=HIDDEN, max_tokens=MAX_TOKENS, num_cpu_threads=2, device="cpu")
    handle = executor.submit(layer, [], executor.next_buffer(), 2)
    # No workers were engaged; reduction over an empty list must not happen
    # here because the caller skips wait_and_h2d for empty submissions.
    assert handle._futures == []
