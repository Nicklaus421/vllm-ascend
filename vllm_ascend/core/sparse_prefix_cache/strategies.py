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
"""DP-optimal sparse checkpoint placement for hybrid (attention + recurrent) LLMs.

Port of the placement logic from the reference implementation of
*Sparse Prefix Caching for Hybrid and Recurrent LLM Serving* (arXiv:2605.05219).

The core idea: recurrent (GatedDeltaNet) state can resume from a single stored
state, so instead of caching it densely we store checkpoints at a sparse set of
positions chosen to minimize expected recompute cost under the empirical
prefix-overlap distribution. The placement problem reduces to an ``O(NM)``
dynamic program (the paper's Corollary 1), solved here with vectorized NumPy.

This module is pure CPU-side planning logic and has no NPU / vLLM dependency so
it can be unit-tested on any host.
"""

from collections import deque
from functools import lru_cache

import numpy as np

# ---------------------------------------------------------------------------
# Balanced fallback (used before the first DP solve)
# ---------------------------------------------------------------------------
def balanced_positions(seq_len: int, budget: int) -> list[int]:
    """Return ``budget`` evenly spaced checkpoint positions in ``[1, seq_len]``."""
    if budget <= 0:
        return []
    block = max(1, seq_len // (budget + 1))
    return list(range(block, seq_len + 1, block))[:budget]


def fixed_interval_positions(
    seq_len: int, interval: int, block_size: int
) -> list[int]:
    """Return prefix-consistent checkpoint positions every ``interval`` tokens.

    This is the deterministic placement used by the v1 sparse prefix cache: the
    scheduler-side coordinator and the TP worker must independently derive the
    *same* checkpoint positions for a prefix without sharing state across
    processes. Unlike :func:`balanced_positions` (whose spacing depends on the
    full prefix length), these positions are fixed multiples of ``interval`` and
    therefore a monotone subset as ``seq_len`` grows — so a hit at ``P`` recovers
    the exact checkpoint ``C <= P`` that was captured when the longer prefix was
    originally stored.

    Each position is clipped down to a ``block_size`` boundary (the granularity
    at which vLLM commits recurrent state) and de-duplicated.
    """
    clipped: list[int] = []
    for pos in range(interval, seq_len + 1, interval):
        c = (pos // block_size) * block_size
        if c <= 0 or c in clipped:
            continue
        clipped.append(c)
    return clipped


# ---------------------------------------------------------------------------
# DP-optimal checkpoint placement (Corollary 1 from the paper)
# ---------------------------------------------------------------------------
def solve_dp(
    hist: np.ndarray,
    budget: int,
    attn_weight: float = 0.0,
) -> tuple[list[np.ndarray | None], int, int] | None:
    """Solve the DP forward pass in ``O(NM)`` via the convex hull trick.

    Args:
        hist: overlap-depth histogram of length ``N + 1`` where ``hist[t]`` is
            the weight of overlap depth ``t`` for ``t = 0..N``.
        budget: checkpoint budget ``M``.
        attn_weight: per-bin attention-cost coefficient. The recompute cost of a
            token at position ``u`` is ``1 + attn_weight * u``: the
            ``attn_weight * u`` term models the full-attention layers, whose
            per-token cost grows with the absolute position ``u`` (they attend
            to ``[0, u)``), while the ``1`` is the position-independent
            recurrent/state cost. ``0.0`` recovers the original unit-cost model.

    Returns:
        ``(all_back, N, M)`` where ``all_back[m][j]`` stores the optimal split
        point for ``m`` checkpoints covering positions ``1..j``. Returns
        ``None`` for an empty histogram or zero budget.

    The recurrence ``dp[m][j] = min_{s<=j} (dp[m-1][s-1] + cost(s, j))`` with
    ``cost(s, j) = sum_{t=s}^j [(t - s) + attn_weight/2 * (t^2 - s^2)] * p[t]``
    reduces to a lower-envelope query over lines ``y = slope * x + B[s]``
    evaluated at ``x = C0[j]``, where ``C0``/``C1``/``C2`` are the cumulative
    sums of ``p``, ``t * p`` and ``t^2 * p``,
    ``slope = -(attn_weight/2 * s^2 + s)`` and
    ``B[s] = dp[m-1][s-1] - attn_weight/2 * C2[s-1] - C1[s-1]
             + (attn_weight/2 * s^2 + s) * C0[s-1]``.
    The slopes are monotone decreasing for ``attn_weight >= 0`` and the query
    points ``C0[j]`` monotone non-decreasing, so a single monotone deque
    evaluates each row in ``O(N)`` (total ``O(NM)``) instead of the naive
    ``O(N^2 M)``.
    """
    N = len(hist) - 1  # hist[0..N], positions 1..N
    M = min(budget, N)
    if N <= 0 or M <= 0:
        return None

    p = np.array(hist, dtype=np.float64)
    total = p.sum()
    if total < 1e-12:
        return None
    p = p / total

    t_idx = np.arange(N + 1, dtype=np.float64)
    c0 = np.cumsum(p)                  # c0[j] = sum_{k=0}^j p[k]
    c1 = np.cumsum(p * t_idx)          # c1[j] = sum_{k=0}^j k * p[k]
    c2 = np.cumsum(p * t_idx * t_idx)  # c2[j] = sum_{k=0}^j k^2 * p[k]

    half_w = attn_weight / 2.0
    # dp_prev[j] == dp[0][j] = sum_{k=1}^j (k + half_w * k^2) * p[k]: the cost
    # of recomputing [0, j] with no checkpoint (the k=0 term is zero).
    dp_prev = c1 + half_w * c2
    all_back: list[np.ndarray | None] = [None]

    for _ in range(1, M + 1):
        dp_curr = np.zeros(N + 1)
        back_curr = np.zeros(N + 1, dtype=np.intp)
        # Monotone lower envelope of lines y = slope * x + B[s] for s = 1..N.
        hull: deque[tuple[float, float, int]] = deque()
        for j in range(1, N + 1):
            s = float(j)
            slope = -(half_w * s * s + s)
            intercept = (
                dp_prev[j - 1] - half_w * c2[j - 1] - c1[j - 1]
                + (half_w * s * s + s) * c0[j - 1]
            )
            # Drop the previous line when the new one renders it never-optimal:
            # inter(prev2, prev) >= inter(prev, new).
            while len(hull) >= 2:
                m1, b1, _ = hull[-2]
                m2, b2, _ = hull[-1]
                if (b2 - b1) * (m2 - slope) >= (intercept - b2) * (m1 - m2):
                    hull.pop()
                else:
                    break
            hull.append((slope, intercept, j))
            x = c0[j]
            # Advance past lines no longer minimal at x. Strict ``>`` keeps the
            # earliest (smallest ``s``) line on ties, matching ``np.argmin``.
            while (
                len(hull) >= 2
                and hull[0][0] * x + hull[0][1] > hull[1][0] * x + hull[1][1]
            ):
                hull.popleft()
            dp_curr[j] = c1[j] + half_w * c2[j] + hull[0][0] * x + hull[0][1]
            back_curr[j] = hull[0][2]
        dp_prev = dp_curr
        all_back.append(back_curr)

    return all_back, N, M


def backtrack(all_back: list[np.ndarray | None], M: int, j: int) -> list[int]:
    """Backtrack from ``dp[M, j]`` to recover checkpoint positions (bin indices)."""
    positions: list[int] = []
    for m in range(M, 0, -1):
        if j <= 0:
            break
        table = all_back[m]
        assert table is not None
        s = int(table[j])
        positions.append(s)
        j = s - 1
    positions.sort()
    return positions


def _bin_index(value: int, bin_size: int) -> int:
    return value // bin_size


def _bin_to_pos(bin_idx: int, bin_size: int) -> int:
    """Convert a bin index back to a token position (use the bin upper edge)."""
    return bin_idx * bin_size


def laplace_smoothing(counts: np.ndarray, alpha: float) -> np.ndarray:
    """Apply Laplace smoothing up to the last non-zero bin.

    ``alpha`` is the *total* pseudo-count mass, spread uniformly over the active
    bins ``[0, last_nonzero_bin]``. Treating it as a total (rather than a
    per-bin additive constant) keeps the smoothing weak and scale-invariant: a
    histogram spanning thousands of bins no longer gets ``alpha``
    pseudo-observations added to *every* bin, which previously flooded the
    empirical counts and collapsed the DP to a near-uniform placement.
    """
    smoothed = counts.copy()
    nz = np.argwhere(smoothed > 0)
    if len(nz) == 0:
        return smoothed
    max_bin_pos = nz.max().item()
    n_active = max_bin_pos + 1
    smoothed[:n_active] += alpha / n_active
    return smoothed


class HistogramTracker:
    """Tracks the overlap-depth histogram for distribution-aware checkpointing.

    Three modes control when the DP is re-solved:

    - ``frozen``: accumulate during warmup, solve once, freeze.
    - ``periodic``: re-solve every ``replan_interval`` observations.
    - ``exp_decay``: exponential decay weighting (``gamma**(n-1-i)``) to follow
      distribution drift, re-solve every ``replan_interval``.

    ``bin_size > 1`` bins overlap depths into groups, smoothing the histogram
    and speeding up the DP. Returned positions are multiples of ``bin_size``.
    """

    def __init__(
        self,
        max_len: int,
        budget: int,
        mode: str = "frozen",
        gamma: float = 0.99,
        replan_interval: int = 100,
        alpha: float = 1.0,
        bin_size: int = 1,
        adaptive_backtrack: bool = True,
        attn_weight: float = 0.0,
    ) -> None:
        self.max_len = max_len
        self.budget = budget
        self.mode = mode
        self.gamma = gamma
        self.alpha = alpha  # Laplace smoothing
        self.replan_interval = replan_interval
        self.bin_size = max(1, bin_size)
        self.adaptive_backtrack = adaptive_backtrack
        self.attn_weight = attn_weight  # attention-aware recompute cost

        n_bins = max_len // self.bin_size + 1
        self.counts = np.zeros(n_bins, dtype=np.float64)
        self.n_obs = 0
        self._dp_result: tuple[list[np.ndarray | None], int, int] | None = None
        self._fixed_positions: list[int] | None = None
        self._dirty = True
        self._n_solves = 0
        self.histogram_log: list[dict[str, object]] = []

    def observe(self, overlap_depth: int) -> None:
        """Record an observed overlap depth."""
        b = min(
            _bin_index(max(int(overlap_depth), 0), self.bin_size),
            len(self.counts) - 1,
        )
        if self.mode == "exp_decay":
            self.counts *= self.gamma
        self.counts[b] += 1.0
        self.n_obs += 1
        self._dirty = True

    def solve(self) -> None:
        """Solve the DP on the current (binned) histogram and cache the tables."""
        self.histogram_log.append(
            {
                "n_obs": self.n_obs,
                "counts": self.counts.copy(),
            }
        )
        smoothed = laplace_smoothing(self.counts, self.alpha)
        self._dp_result = solve_dp(
            smoothed, self.budget, attn_weight=self.attn_weight
        )
        if self._dp_result is None:
            self._fixed_positions = None
        else:
            _, N, M = self._dp_result
            positions = self._backtrack_bins(N)
            token_positions = [_bin_to_pos(b, self.bin_size) for b in positions]
            if not self.adaptive_backtrack:
                self._fixed_positions = token_positions
        self._backtrack_bins.cache_clear()
        self._n_solves += 1
        self._dirty = False

    @lru_cache(maxsize=256)
    def _backtrack_bins(self, j_bin: int) -> list[int]:
        """Backtrack from bin index ``j_bin``, returning bin-level positions."""
        assert self._dp_result is not None
        all_back, N, M = self._dp_result
        j = min(j_bin, N)
        return backtrack(all_back, M, j)

    def get_positions(self, seq_len: int) -> list[int]:
        """Get checkpoint positions for a request of the given length."""
        if self._dp_result is None:
            # No DP solution yet — use balanced fallback until the first solve.
            return balanced_positions(seq_len, self.budget)

        should_solve = False
        if self.mode == "frozen":
            pass
        elif self.mode in ("periodic", "exp_decay"):
            if self._dirty and self.n_obs % self.replan_interval == 0:
                should_solve = True

        if should_solve:
            self.solve()

        if not self.adaptive_backtrack and self._fixed_positions is not None:
            return [p for p in self._fixed_positions if 0 < p <= seq_len]

        j_bin = seq_len // self.bin_size
        bin_positions = self._backtrack_bins(j_bin)
        return [
            _bin_to_pos(b, self.bin_size)
            for b in bin_positions
            if 0 < b <= j_bin
        ]

    def freeze(self) -> None:
        """Solve and freeze — no further updates to positions."""
        self.solve()
        self.mode = "frozen"
