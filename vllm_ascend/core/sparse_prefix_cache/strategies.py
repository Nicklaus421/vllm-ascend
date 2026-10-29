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


# ---------------------------------------------------------------------------
# DP-optimal checkpoint placement (Corollary 1 from the paper)
# ---------------------------------------------------------------------------
def solve_dp(
    hist: np.ndarray,
    budget: int,
) -> tuple[list[np.ndarray | None], int, int] | None:
    """Solve the DP forward pass (vectorized).

    Args:
        hist: overlap-depth histogram of length ``N + 1`` where ``hist[t]`` is
            the weight of overlap depth ``t`` for ``t = 0..N``.
        budget: checkpoint budget ``M``.

    Returns:
        ``(all_back, N, M)`` where ``all_back[m][j]`` stores the optimal split
        point for ``m`` checkpoints covering positions ``1..j``. Returns
        ``None`` for an empty histogram or zero budget.
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

    cum_p = np.zeros(N + 2)
    cum_pt = np.zeros(N + 2)
    t_idx = np.arange(N + 1, dtype=np.float64)
    cum_p[1:] = np.cumsum(p)
    cum_pt[1:] = np.cumsum(p * t_idx)
    cpt_j = cum_pt[1:]
    cp_j = cum_p[1:]

    dp_prev = cpt_j.copy()
    all_back: list[np.ndarray | None] = [None]

    for m in range(1, M + 1):
        dp_curr = np.full(N + 1, np.inf)
        back_curr = np.zeros(N + 1, dtype=np.intp)
        dp_curr[0] = 0.0
        for j in range(1, N + 1):
            s_range = np.arange(1, j + 1)
            costs = (
                dp_prev[s_range - 1]
                + cpt_j[j]
                - cum_pt[s_range]
                - s_range * (cp_j[j] - cum_p[s_range])
            )
            idx = np.argmin(costs)
            dp_curr[j] = costs[idx]
            back_curr[j] = s_range[idx]
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
    """Apply Laplace smoothing up to the last non-zero bin."""
    smoothed = counts.copy()
    nz = np.argwhere(smoothed > 0)
    if len(nz) == 0:
        return smoothed
    max_bin_pos = nz.max().item()
    smoothed[:max_bin_pos] += alpha
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
    ) -> None:
        self.max_len = max_len
        self.budget = budget
        self.mode = mode
        self.gamma = gamma
        self.alpha = alpha  # Laplace smoothing
        self.replan_interval = replan_interval
        self.bin_size = max(1, bin_size)
        self.adaptive_backtrack = adaptive_backtrack

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
        self._dp_result = solve_dp(smoothed, self.budget)
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
