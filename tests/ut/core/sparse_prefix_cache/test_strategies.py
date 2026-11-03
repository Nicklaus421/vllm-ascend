#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
"""Unit tests for the DP-optimal sparse checkpoint placement strategies."""

import numpy as np
import pytest

from vllm_ascend.core.sparse_prefix_cache import (
    HistogramTracker,
    backtrack,
    balanced_positions,
    laplace_smoothing,
    solve_dp,
)


# ---------------------------------------------------------------------------
# Independent naive ground truth for the DP (semantic cost, no cumsum tricks)
# ---------------------------------------------------------------------------
def _naive_dp(hist: np.ndarray, budget: int):
    """Brute-force DP over split points using the literal recompute cost.

    ``cost(s, j) = sum_{t=s}^{j} (t - s) * p[t]`` where ``p`` is the normalized
    histogram: if a checkpoint sits at ``s``, an overlap depth ``t`` in
    ``[s, j]`` forces recomputing ``t - s`` tokens.
    """
    N = len(hist) - 1
    M = min(budget, N)
    p = np.array(hist, dtype=np.float64)
    total = p.sum()
    assert total > 1e-12
    p = p / total

    inf = float("inf")
    dp = [[inf] * (N + 1) for _ in range(M + 1)]
    back = [[0] * (N + 1) for _ in range(M + 1)]

    dp[0][0] = 0.0
    for j in range(1, N + 1):
        dp[0][j] = sum(t * p[t] for t in range(1, j + 1))

    for m in range(1, M + 1):
        dp[m][0] = 0.0
        for j in range(1, N + 1):
            best = inf
            best_s = 0
            for s in range(1, j + 1):
                seg_cost = sum((t - s) * p[t] for t in range(s, j + 1))
                c = dp[m - 1][s - 1] + seg_cost
                if c < best:
                    best = c
                    best_s = s
            dp[m][j] = best
            back[m][j] = best_s

    positions = []
    j = N
    for m in range(M, 0, -1):
        if j <= 0:
            break
        s = back[m][j]
        positions.append(s)
        j = s - 1
    positions.sort()
    return positions, dp[M][N]


def _naive_dp_weighted(hist: np.ndarray, budget: int, attn_weight: float):
    """Brute-force DP for the attention-aware cost model.

    ``cost(s, j) = sum_{t=s}^{j} [(t - s) + attn_weight/2 * (t^2 - s^2)] * p[t]``
    where the ``attn_weight`` term models full-attention recompute cost growing
    with the absolute position, and the ``(t - s)`` term is the
    position-independent recurrent cost.
    """
    N = len(hist) - 1
    M = min(budget, N)
    p = np.array(hist, dtype=np.float64)
    total = p.sum()
    assert total > 1e-12
    p = p / total
    half_w = attn_weight / 2.0

    inf = float("inf")
    dp = [[inf] * (N + 1) for _ in range(M + 1)]
    back = [[0] * (N + 1) for _ in range(M + 1)]

    dp[0][0] = 0.0
    for j in range(1, N + 1):
        dp[0][j] = sum((t + half_w * t * t) * p[t] for t in range(1, j + 1))

    for m in range(1, M + 1):
        dp[m][0] = 0.0
        for j in range(1, N + 1):
            best = inf
            best_s = 0
            for s in range(1, j + 1):
                seg_cost = sum(
                    ((t - s) + half_w * (t * t - s * s)) * p[t]
                    for t in range(s, j + 1)
                )
                c = dp[m - 1][s - 1] + seg_cost
                if c < best:
                    best = c
                    best_s = s
            dp[m][j] = best
            back[m][j] = best_s

    positions = []
    j = N
    for m in range(M, 0, -1):
        if j <= 0:
            break
        s = back[m][j]
        positions.append(s)
        j = s - 1
    positions.sort()
    return positions, dp[M][N]


def _weighted_recompute_cost(
    positions: list[int], p: np.ndarray, n: int, attn_weight: float
) -> float:
    """Evaluate the attention-aware recompute cost of a placement."""
    half_w = attn_weight / 2.0
    total = 0.0
    for t in range(1, n + 1):
        candidates = [s for s in positions if s <= t]
        s = max(candidates) if candidates else 0
        total += ((t - s) + half_w * (t * t - s * s)) * p[t]
    return total


# ---------------------------------------------------------------------------
# solve_dp / backtrack
# ---------------------------------------------------------------------------
def test_solve_dp_degenerate_inputs():
    assert solve_dp(np.array([0.0]), 1) is None  # N == 0
    assert solve_dp(np.array([1.0, 1.0]), 0) is None  # M == 0
    assert solve_dp(np.zeros(5), 3) is None  # zero total mass


def test_backtrack_matches_naive_on_uniform():
    rng = np.random.RandomState(0)
    for _ in range(50):
        N = rng.randint(1, 30)
        budget = rng.randint(1, 6)
        hist = rng.uniform(0.1, 1.0, size=N + 1)

        result = solve_dp(hist, budget)
        assert result is not None
        all_back, n, m = result
        got = backtrack(all_back, m, n)

        want, _ = _naive_dp(hist, budget)
        assert got == want, f"hist={hist.tolist()}, budget={budget}"


def test_solve_dp_matches_naive_cost():
    rng = np.random.RandomState(1)
    for _ in range(50):
        N = rng.randint(1, 25)
        budget = rng.randint(1, 5)
        hist = rng.uniform(0.05, 1.0, size=N + 1)

        all_back, n, m = solve_dp(hist, budget)
        positions = backtrack(all_back, m, n)

        # Evaluate the returned placement's objective directly: for each
        # overlap depth t, the recompute cost is the distance to the largest
        # checkpoint at or before t (or t itself when none precedes it).
        p = hist / hist.sum()

        def cost_of(pos: list[int]) -> float:
            total = 0.0
            for t in range(1, n + 1):
                candidates = [s for s in pos if s <= t]
                total += (t - max(candidates)) * p[t] if candidates else t * p[t]
            return total

        _, want_cost = _naive_dp(hist, budget)
        assert np.isclose(cost_of(positions), want_cost, atol=1e-9)


def test_solve_dp_attn_weight_matches_naive():
    rng = np.random.RandomState(7)
    for _ in range(50):
        N = rng.randint(1, 25)
        budget = rng.randint(1, 5)
        hist = rng.uniform(0.1, 1.0, size=N + 1)
        attn_weight = rng.uniform(0.0, 2.0)

        result = solve_dp(hist, budget, attn_weight=attn_weight)
        assert result is not None
        all_back, n, m = result
        got = backtrack(all_back, m, n)
        want, _ = _naive_dp_weighted(hist, budget, attn_weight)
        assert got == want, (
            f"hist={hist.tolist()}, budget={budget}, attn_weight={attn_weight}"
        )


def test_solve_dp_attn_weight_zero_equals_unit_cost():
    # attn_weight=0 must reproduce the original unit-cost placement exactly.
    rng = np.random.RandomState(8)
    for _ in range(30):
        N = rng.randint(1, 25)
        budget = rng.randint(1, 5)
        hist = rng.uniform(0.1, 1.0, size=N + 1)
        a = solve_dp(hist, budget, attn_weight=0.0)
        b = solve_dp(hist, budget)
        assert a is not None and b is not None
        assert backtrack(a[0], a[2], a[1]) == backtrack(b[0], b[2], b[1])


def test_solve_dp_attn_weight_improves_weighted_cost():
    # Flat overlap distribution: the attention-aware placement must shift
    # checkpoints deeper (deep recompute is more expensive per token) and
    # achieve a strictly lower weighted recompute cost than the unit placement.
    N = 40
    hist = np.ones(N + 1)
    hist[0] = 0.0
    budget = 3
    attn_weight = 1.0

    all_back, n, m = solve_dp(hist, budget, attn_weight=attn_weight)
    weighted_pos = backtrack(all_back, m, n)
    unit_all_back, unit_n, unit_m = solve_dp(hist, budget, attn_weight=0.0)
    unit_pos = backtrack(unit_all_back, unit_m, unit_n)

    assert weighted_pos != unit_pos
    # Checkpoints shift toward the deep end under the attention-aware model.
    assert sum(weighted_pos) > sum(unit_pos)

    p = hist / hist.sum()
    weighted_cost_weighted = _weighted_recompute_cost(
        weighted_pos, p, n, attn_weight
    )
    weighted_cost_unit = _weighted_recompute_cost(unit_pos, p, n, attn_weight)
    assert weighted_cost_weighted < weighted_cost_unit


def test_backtrack_positions_sorted_and_bounded():
    hist = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    all_back, n, m = solve_dp(hist, budget=3)
    positions = backtrack(all_back, m, n)
    assert positions == sorted(positions)
    assert all(1 <= p <= n for p in positions)
    assert len(positions) <= 3


# ---------------------------------------------------------------------------
# HistogramTracker
# ---------------------------------------------------------------------------
def test_tracker_fallback_is_balanced():
    tracker = HistogramTracker(max_len=100, budget=4)
    pos = tracker.get_positions(100)
    assert pos == balanced_positions(100, 4)


def test_tracker_frozen_positions_stable():
    tracker = HistogramTracker(max_len=64, budget=3, mode="frozen")
    for d in [10, 20, 30, 40, 50]:
        tracker.observe(d)
    tracker.freeze()
    first = tracker.get_positions(64)
    second = tracker.get_positions(64)
    assert first == second
    # Positions must be <= seq_len.
    assert all(0 < p <= 64 for p in first)


def test_tracker_binning_returns_multiples_of_bin_size():
    tracker = HistogramTracker(max_len=128, budget=4, mode="frozen", bin_size=16)
    for d in [16, 32, 48, 64, 80, 96]:
        tracker.observe(d)
    tracker.freeze()
    positions = tracker.get_positions(128)
    assert positions, "expected non-empty positions"
    assert all(p % 16 == 0 for p in positions)


def test_tracker_exp_decay_discounts_old_observations():
    tracker = HistogramTracker(max_len=32, budget=2, mode="exp_decay", gamma=0.5)
    tracker.observe(8)
    counts_after_first = tracker.counts.copy()
    tracker.observe(8)
    # First observation's bin was decayed by gamma before the second increment.
    decayed_first = counts_after_first * 0.5
    decayed_first[8] += 1.0
    np.testing.assert_allclose(tracker.counts, decayed_first)


def test_tracker_periodic_replans():
    tracker = HistogramTracker(
        max_len=32, budget=2, mode="periodic", replan_interval=3
    )
    for d in [8, 16, 24]:
        tracker.observe(d)
    assert tracker.n_obs == 3
    # Replan triggers exactly on multiples of replan_interval.
    assert tracker._n_solves == 0
    tracker.get_positions(32)  # 3 % 3 == 0 -> solve
    assert tracker._n_solves == 1


def test_laplace_smoothing_up_to_last_nonzero():
    counts = np.array([0.0, 2.0, 0.0, 3.0, 0.0, 0.0])
    smoothed = laplace_smoothing(counts, alpha=1.0)
    # Last non-zero bin is index 3, so 4 active bins share the total
    # pseudo-count mass of 1.0 -> +0.25 per bin. Bins past the last non-zero
    # bin are untouched.
    np.testing.assert_allclose(smoothed[0], 0.25)
    np.testing.assert_allclose(smoothed[1], 2.25)
    np.testing.assert_allclose(smoothed[2], 0.25)
    np.testing.assert_allclose(smoothed[3], 3.25)
    assert smoothed[4] == 0.0
    assert smoothed[5] == 0.0


def test_laplace_smoothing_does_not_flood_wide_histogram():
    # A wide, sparse histogram must not be flooded into uniformity: the total
    # pseudo-count mass stays small relative to the real observations, so the
    # DP sees the true distribution instead of an (almost) flat one.
    counts = np.zeros(2000)
    counts[100] = 50.0  # one strong cluster
    counts[1500] = 30.0  # a second cluster
    counts[1999] = 20.0
    smoothed = laplace_smoothing(counts, alpha=1.0)
    # The smoothing mass is 1.0 spread over 2000 bins, i.e. negligible per bin.
    assert smoothed[0] < 0.001
    assert smoothed.sum() == pytest.approx(counts.sum() + 1.0)
    # The cluster contrast is preserved.
    assert smoothed[100] > 49.9
