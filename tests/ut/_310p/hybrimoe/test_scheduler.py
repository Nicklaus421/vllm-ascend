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
"""Unit tests for the HybriMoE HSS scheduler (host-only, no torch required)."""

import random

from vllm_ascend._310p.hybrimoe.scheduler import (
    CostModel,
    _fit_linear,
    hss_schedule,
    simulate_makespan,
)

# Reference coefficients from the HybriMoE paper implementation.
REF_CPU_COEF = 1.0
REF_CPU_INTERCEPT = 0.79
REF_NPU_CONST = 0.5
REF_MOVE = 0.41791011235117914


def _reference_hss(load_experts, unload_experts, id_counts):
    """Faithful port of HSS() from PKU-SEC-Lab/HybriMoE (experts.py)."""
    gpu_total_time = 0.0
    cpu_total_time = 0.0
    move_total_time = 0.0
    gpu_queue = []
    cpu_queue = []

    load_experts_sorted = sorted(load_experts, key=lambda idx: id_counts[idx], reverse=True)
    unload_experts_sorted = sorted(unload_experts, key=lambda idx: id_counts[idx], reverse=True)
    sorted_experts = load_experts_sorted + unload_experts_sorted

    while sorted_experts:
        first_idx = sorted_experts[0]
        last_idx = sorted_experts[-1]
        is_move = first_idx in unload_experts
        gpu_time = REF_NPU_CONST
        if len(gpu_queue) == 0:
            gpu_time += 0.0  # reference intercept for gpu is 0
        cpu_time = REF_CPU_COEF * id_counts[last_idx]
        if len(cpu_queue) == 0:
            cpu_time += REF_CPU_INTERCEPT
        if is_move:
            move_time = REF_MOVE
            if move_total_time + move_time > gpu_total_time:
                gpu_time += move_total_time + move_time - gpu_total_time
        if gpu_total_time + gpu_time <= cpu_total_time + cpu_time:
            if is_move:
                move_total_time += REF_MOVE
            gpu_total_time += gpu_time
            gpu_queue.append(first_idx)
            sorted_experts.pop(0)
        else:
            cpu_total_time += cpu_time
            cpu_queue.append(last_idx)
            sorted_experts.pop(-1)

    return cpu_queue, gpu_queue


def _reference_cost_model() -> CostModel:
    return CostModel(
        cpu_coef=REF_CPU_COEF,
        cpu_startup=REF_CPU_INTERCEPT,
        npu_coef=0.0,
        npu_per_expert=REF_NPU_CONST,
        move_const=REF_MOVE,
    )


def test_hss_matches_reference_randomized():
    rng = random.Random(20260821)
    cost_model = _reference_cost_model()
    for _ in range(500):
        num_experts = rng.randint(1, 16)
        experts = list(range(num_experts))
        rng.shuffle(experts)
        num_cached = rng.randint(0, num_experts)
        cached = experts[:num_cached]
        uncached = experts[num_cached:]
        counts = {e: rng.randint(1, 64) for e in experts}

        result = hss_schedule(cached, uncached, counts, cost_model)
        ref_cpu, ref_gpu = _reference_hss(cached, uncached, counts)

        assert result.cpu_experts == ref_cpu
        assert result.npu_experts == ref_gpu
        # Every activated expert is assigned exactly once.
        assert sorted(result.cpu_experts + result.npu_experts) == list(range(num_experts))


def test_hss_all_cached_still_uses_cpu():
    cost_model = _reference_cost_model()
    # All experts cached: the CPU should still take low-load experts.
    counts = {0: 100, 1: 1, 2: 1, 3: 1}
    result = hss_schedule([0, 1, 2, 3], [], counts, cost_model)
    assert result.npu_experts[0] == 0  # highest load on the NPU
    assert set(result.cpu_experts) | set(result.npu_experts) == {0, 1, 2, 3}
    assert result.cpu_experts  # CPU is not idle


def test_hss_empty_inputs():
    result = hss_schedule([], [], {}, _reference_cost_model())
    assert result.cpu_experts == []
    assert result.npu_experts == []
    assert result.makespan_ms == 0.0


def test_hss_uncached_expert_waits_for_transfer():
    cost_model = _reference_cost_model()
    # One huge uncached expert: the NPU can only start it after the transfer.
    result = hss_schedule([], [7], {7: 1000}, cost_model)
    assert result.npu_experts == [7]
    assert result.makespan_ms >= REF_MOVE + REF_NPU_CONST


def test_simulate_makespan_prefetch_gain_is_positive():
    cost_model = _reference_cost_model()
    counts = {0: 50, 1: 40, 2: 30}
    without = simulate_makespan([0], [1, 2], counts, cost_model)
    with_expert = simulate_makespan([0, 1], [2], counts, cost_model)
    assert with_expert <= without


def test_fit_linear():
    coef, intercept = _fit_linear([1.0, 2.0, 3.0, 4.0], [3.0, 5.0, 7.0, 9.0])
    assert abs(coef - 2.0) < 1e-9
    assert abs(intercept - 1.0) < 1e-9


def test_cost_model_json_roundtrip(tmp_path):
    model = CostModel(cpu_coef=1.5, cpu_startup=0.3, npu_coef=0.01, npu_per_expert=0.4, move_const=0.2)
    path = tmp_path / "cost_model.json"
    model.save(str(path))
    loaded = CostModel.load_or_default(str(path))
    assert loaded == model
