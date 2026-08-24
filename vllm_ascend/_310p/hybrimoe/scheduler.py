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
"""HybriMoE hybrid scheduling strategy (HSS) and its cost model.

This is a cleaned-up port of the HSS() scheduler from the reference
implementation (https://github.com/PKU-SEC-Lab/HybriMoE,
ktransformers/operators/experts.py), with a calibratable cost model.

Scheduling rules (paper section IV-B):
  - NPU priority: compute cached experts first, higher load first.
  - CPU priority: compute uncached experts, lower load first; the CPU may
    also take low-load cached experts when it would otherwise idle.
  - Transfer priority: high-load uncached experts are moved to the NPU
    first; a transfer must finish before the NPU can compute that expert.

The greedy simulation below reduces the NP-hard mapping problem to a
linear scan by always assigning the next expert (highest-load remaining
for the NPU side, lowest-load remaining for the CPU side) to whichever
device would finish earlier.

Everything in this module is host-only and device-agnostic so it can be
unit-tested without an NPU.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

# Fallback coefficients (milliseconds), taken from the reference
# implementation's hard-coded values. They are only used when on-device
# calibration is disabled and no cached calibration file exists.
_FALLBACK_CPU_COEF_MS = 1.0
_FALLBACK_CPU_STARTUP_MS = 0.79
_FALLBACK_NPU_COEF_MS = 0.0
_FALLBACK_NPU_PER_EXPERT_MS = 0.5
_FALLBACK_MOVE_MS = 0.41791011235117914


def _fit_linear(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares fit of y = coef * x + intercept (closed form, 2 params)."""
    n = len(xs)
    if n == 0:
        raise ValueError("cannot fit a linear model with no samples")
    if n == 1:
        return 0.0, ys[0]
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-12:
        return 0.0, sum_y / n
    coef = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - coef * sum_x) / n
    return coef, intercept


@dataclass
class CostModel:
    """Linear cost model for the HSS scheduler (all times in milliseconds).

    cpu_time(n, first) = cpu_startup (first expert only) + cpu_coef * n
    npu_time(n)        = npu_per_expert + npu_coef * n
    move_time()        = move_const                     (H2D transfer of one expert)

    (n = tokens routed to one expert). The CPU startup term models the
    first-expert cache-cold penalty observed in the paper (Fig. 3e); the NPU
    per-expert term is charged for every expert (grouped-matmul launch).
    """

    cpu_coef: float = _FALLBACK_CPU_COEF_MS
    cpu_startup: float = _FALLBACK_CPU_STARTUP_MS
    npu_coef: float = _FALLBACK_NPU_COEF_MS
    npu_per_expert: float = _FALLBACK_NPU_PER_EXPERT_MS
    move_const: float = _FALLBACK_MOVE_MS

    def cpu_time(self, token_num: int, first: bool = False) -> float:
        """CPU compute time of one expert; `first` adds the startup penalty."""
        return self.cpu_coef * token_num + (self.cpu_startup if first else 0.0)

    def npu_time(self, token_num: int) -> float:
        """NPU compute time of one expert (per-expert cost is always charged)."""
        return self.npu_per_expert + self.npu_coef * token_num

    def move_time(self) -> float:
        return self.move_const

    def to_json(self) -> str:
        return json.dumps(
            {
                "cpu_coef": self.cpu_coef,
                "cpu_startup": self.cpu_startup,
                "npu_coef": self.npu_coef,
                "npu_per_expert": self.npu_per_expert,
                "move_const": self.move_const,
            }
        )

    @classmethod
    def from_json(cls, payload: str) -> CostModel:
        data = json.loads(payload)
        return cls(
            cpu_coef=float(data["cpu_coef"]),
            cpu_startup=float(data["cpu_startup"]),
            npu_coef=float(data["npu_coef"]),
            npu_per_expert=float(data["npu_per_expert"]),
            move_const=float(data["move_const"]),
        )

    @classmethod
    def load_or_default(cls, cache_path: str | None) -> CostModel:
        if cache_path and os.path.exists(cache_path):
            with open(cache_path) as f:
                return cls.from_json(f.read())
        return cls()

    def save(self, cache_path: str) -> None:
        with open(cache_path, "w") as f:
            f.write(self.to_json())


@dataclass
class HSSResult:
    """Output of one HSS scheduling simulation."""

    # Experts to compute on the NPU, in execution order (highest load first).
    npu_experts: list[int] = field(default_factory=list)
    # Experts to compute on the CPU, in execution order (lowest load first).
    cpu_experts: list[int] = field(default_factory=list)
    # Simulated finish time (ms) of the slower device.
    makespan_ms: float = 0.0


def hss_schedule(
    cached_experts: list[int],
    uncached_experts: list[int],
    token_counts: dict[int, int],
    cost_model: CostModel,
) -> HSSResult:
    """Decide the CPU/NPU split for one MoE layer forward (decode).

    Args:
        cached_experts: activated experts already resident in NPU slots.
        uncached_experts: activated experts only present on the host.
        token_counts: number of tokens routed to each activated expert.
        cost_model: calibrated cost model.

    Returns:
        HSSResult with the NPU/CPU expert lists and the simulated makespan.
    """
    # Candidate order follows the reference implementation: cached experts
    # first (descending load), then uncached experts (descending load). The
    # NPU always takes from the front (highest priority), the CPU from the
    # back (lowest load; uncached first, matching "CPU prioritizes uncached
    # experts, and takes low-load cached ones when idle").
    merged = sorted(cached_experts, key=lambda e: token_counts[e], reverse=True) + sorted(
        uncached_experts, key=lambda e: token_counts[e], reverse=True
    )
    uncached = set(uncached_experts)

    npu_total = 0.0
    cpu_total = 0.0
    move_total = 0.0
    npu_queue: list[int] = []
    cpu_queue: list[int] = []

    while merged:
        first = merged[0]
        last = merged[-1]
        needs_move = first in uncached

        npu_time = cost_model.npu_time(token_counts[first])
        cpu_time = cost_model.cpu_time(token_counts[last], first=not cpu_queue)
        if needs_move:
            # The transfer of `first` serializes behind earlier transfers and
            # must complete before the NPU may start computing it.
            move_finish = move_total + cost_model.move_time()
            if move_finish > npu_total:
                npu_time += move_finish - npu_total

        if npu_total + npu_time <= cpu_total + cpu_time:
            if needs_move:
                move_total += cost_model.move_time()
            npu_total += npu_time
            npu_queue.append(first)
            merged.pop(0)
        else:
            cpu_total += cpu_time
            cpu_queue.append(last)
            merged.pop(-1)

    return HSSResult(
        npu_experts=npu_queue,
        cpu_experts=cpu_queue,
        makespan_ms=max(npu_total, cpu_total),
    )


def simulate_makespan(
    cached_experts: list[int],
    uncached_experts: list[int],
    token_counts: dict[int, int],
    cost_model: CostModel,
) -> float:
    """Simulated finish time for a hypothetical cache state (prefetch gain)."""
    return hss_schedule(cached_experts, uncached_experts, token_counts, cost_model).makespan_ms


class _WallClock:
    """Small helper so calibration code reads linearly."""

    def __init__(self):
        self._start = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0


def calibrate_cost_model(
    expert_shapes: tuple[int, int],
    iterations: int,
    cpu_mlp_fn=None,
    npu_mlp_fn=None,
    move_fn=None,
) -> CostModel:
    """Measure cost-model coefficients on the actual machine.

    Args:
        expert_shapes: (hidden_size, moe_intermediate_size) of one expert.
        iterations: repetitions per measurement point; the median is used.
        cpu_mlp_fn: callable(token_num) -> None running one CPU expert MLP.
        npu_mlp_fn: callable(token_num) -> None running one NPU expert MLP.
        move_fn: callable() -> None transferring one expert H2D.

    Any fn left as None keeps the fallback coefficient for that term.
    """
    hidden_size, _ = expert_shapes
    del hidden_size  # shapes are baked into the callables by the caller

    model = CostModel()
    token_points = [1, 8, 32]

    if cpu_mlp_fn is not None:
        xs, ys = [], []
        for n in token_points:
            samples = []
            # one warmup run to stabilize allocator / frequency
            cpu_mlp_fn(n)
            for _ in range(iterations):
                with _WallClock() as clock:
                    cpu_mlp_fn(n)
                samples.append(clock.elapsed_ms())
            xs.append(float(n))
            ys.append(sorted(samples)[len(samples) // 2])
        model.cpu_coef, model.cpu_startup = _fit_linear(xs, ys)

    if npu_mlp_fn is not None:
        xs, ys = [], []
        for n in token_points:
            samples = []
            npu_mlp_fn(n)
            for _ in range(iterations):
                with _WallClock() as clock:
                    npu_mlp_fn(n)
                samples.append(clock.elapsed_ms())
            xs.append(float(n))
            ys.append(sorted(samples)[len(samples) // 2])
        model.npu_coef, model.npu_per_expert = _fit_linear(xs, ys)

    if move_fn is not None:
        samples = []
        move_fn()
        for _ in range(iterations):
            with _WallClock() as clock:
                move_fn()
            samples.append(clock.elapsed_ms())
        model.move_const = sorted(samples)[len(samples) // 2]

    return model
