#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
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


import pytest

from tests.e2e.conftest import VllmRunner


@pytest.mark.skip(reason="Requires a single Ascend 310P and the Qwen3.6-35B-A3B-w8a8 model; nightly only")
def test_qwen3_6_moe_hybrimoe_tp1_w8a8():
    """HybriMoE hybrid CPU-NPU MoE inference on a single 310P chip."""
    example_prompts = [
        "Hello, my name is",
    ]
    max_tokens = 8
    with VllmRunner(
        "Eco-Tech/Qwen3.6-35B-A3B-w8a8",
        tensor_parallel_size=1,
        enforce_eager=True,
        quantization="ascend",
        max_model_len=8192,
        max_num_batched_tokens=2048,
        max_num_seqs=16,
        gpu_memory_utilization=0.9,
        additional_config={
            "hybrimoe_config": {
                "enabled": True,
                "npu_cache_budget_gb": 8.0,
                "prefetch_lookahead": 3,
                "prefetch_size": 2,
            }
        },
    ) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)
