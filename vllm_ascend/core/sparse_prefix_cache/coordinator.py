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
"""Coordinator that ties placement, capture and hit/recompute together.

This is the *standalone* sparse-prefix-cache layer. It does not touch the
MambaManager or the block pool; it only

1. asks the :class:`~vllm_ascend.core.sparse_prefix_cache.strategies.HistogramTracker`
   where to place recurrent-state checkpoints for a prefix length,
2. captures those checkpoints into a
   :class:`~vllm_ascend.core.sparse_prefix_cache.store.SparsePrefixCache`, and
3. on a prefix-cache hit resolves the nearest checkpoint ``C <= P`` and emits
   the recompute plan ``[C, P)``.

The tensor plumbing (reading/writing ``kv_cache[0]``/``kv_cache[1]`` and
rewinding ``num_computed_tokens``) lives in the runner, not here. This module
only owns the pure orchestration, so it is unit-testable without an NPU or
vLLM runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Hashable

from vllm_ascend.core.sparse_prefix_cache.checkpoint import (
    PrefixCheckpointStore,
    RecurrentCheckpoint,
)
from vllm_ascend.core.sparse_prefix_cache.store import SparsePrefixCache
from vllm_ascend.core.sparse_prefix_cache.strategies import (
    HistogramTracker,
    fixed_interval_positions,
)

if TYPE_CHECKING:
    import torch

#: (recurrent_states, conv_states) keyed by GDN layer index.
StateSnapshot = tuple[dict[int, "torch.Tensor"], dict[int, "torch.Tensor"]]


def get_checkpoint_positions(seq_len: int) -> list[int]:
    """Return checkpoint positions for ``seq_len`` under the configured strategy.

    This is the single placement entry point shared by the scheduler-side
    coordinator (hit resolution) and the TP worker (capture/restore), so both
    derive identical positions.

    - ``fixed_interval``: deterministic, prefix-consistent, cross-process safe
      with no shared state.
    - ``hist_exp_decay``: DP-optimal placement. The DP is solved once in the
      EngineCore process and its frozen, prefix-consistent positions are read
      through the Manager-backed shared state (``shared.py``). Before the first
      solve (warmup) the deterministic fixed-interval placement is used as a
      cross-process-consistent fallback.
    """
    from vllm_ascend import envs

    strategy = envs.VLLM_ASCEND_SPARSE_PREFIX_CACHE_STRATEGY
    interval = int(envs.VLLM_ASCEND_SPARSE_PREFIX_CACHE_CHECKPOINT_INTERVAL)
    block_size = int(envs.VLLM_ASCEND_SPARSE_PREFIX_CACHE_KERNEL_BLOCK_SIZE)

    if strategy == "fixed_interval":
        return fixed_interval_positions(seq_len, interval, block_size)

    if strategy == "hist_exp_decay":
        from vllm_ascend.core.sparse_prefix_cache.shared import (
            get_sparse_positions,
        )

        positions = get_sparse_positions()
        if positions is None:
            # DP not solved yet: warmup fallback (prefix-consistent, safe).
            return fixed_interval_positions(seq_len, interval, block_size)
        # positions are already sorted and block-clipped by the publisher;
        # filtering to ``<= seq_len`` preserves prefix-consistency.
        return [p for p in positions if 0 < p <= seq_len]

    raise NotImplementedError(
        f"unknown sparse prefix cache strategy {strategy!r}; expected "
        "'fixed_interval' or 'hist_exp_decay'."
    )


@dataclass(frozen=True)
class SparseHit:
    """A resolved prefix-cache hit against the sparse checkpoint store.

    ``checkpoint_position`` (``C``) is the largest stored checkpoint ``<=``
    ``overlap_depth`` (``P``); attention KV is dense up to ``P`` while the
    recurrent state is only valid up to ``C``. The runner rewinds
    ``num_computed_tokens`` to ``C`` and recomputes ``[C, P)``.
    """

    prefix_key: Hashable
    checkpoint_position: int
    overlap_depth: int

    @property
    def recompute_start(self) -> int:
        """First token position that must be recomputed (inclusive)."""
        return self.checkpoint_position

    @property
    def recompute_end(self) -> int:
        """First token position that must *not* be recomputed (exclusive)."""
        return self.overlap_depth

    @property
    def gap(self) -> int:
        """Number of tokens to recompute."""
        return self.overlap_depth - self.checkpoint_position

    @property
    def is_exact(self) -> bool:
        """True when the checkpoint sits exactly at the overlap depth."""
        return self.checkpoint_position == self.overlap_depth


class SparsePrefixCacheCoordinator:
    """Orchestrates sparse recurrent-state checkpointing.

    The coordinator is driven by the single-threaded scheduler loop and is not
    thread-safe, matching the reference implementation.
    """

    def __init__(
        self,
        tracker: HistogramTracker,
        cache: SparsePrefixCache,
        block_size: int = 16,
    ) -> None:
        self.tracker = tracker
        self.cache = cache
        self.block_size = max(1, block_size)
        # Sparse hits resolved by the KV-cache coordinator that still need their
        # checkpoint materialized by the worker before the recompute prefill.
        # Keyed by request id: it is available on both the scheduler side
        # (``Request.request_id``) and the worker side (``req_id``), whereas
        # ``block_hashes`` is a scheduler-only concept.
        self._pending_hits: dict[str, SparseHit] = {}

    @classmethod
    def from_envs(
        cls,
        max_len: int,
        num_checkpoints: int,
    ) -> "SparsePrefixCacheCoordinator":
        """Build a coordinator from ``vllm_ascend.envs``.

        ``max_len`` is the maximum prefix length and ``num_checkpoints`` is the
        DP checkpoint budget ``M`` (derived by the runner from the model's
        recurrent-state byte size and ``VLLM_ASCEND_SPARSE_PREFIX_CACHE_BUDGET_GB``).
        """
        # Lazy import: envs has no vLLM dependency at import time, but importing
        # it eagerly would pull the ``vllm_ascend`` package __init__ (which does).
        from vllm_ascend import envs

        strategy = envs.VLLM_ASCEND_SPARSE_PREFIX_CACHE_STRATEGY
        mode = "exp_decay" if strategy == "hist_exp_decay" else "frozen"

        tracker = HistogramTracker(
            max_len=max_len,
            budget=num_checkpoints,
            mode=mode,
            gamma=float(envs.VLLM_ASCEND_SPARSE_PREFIX_CACHE_GAMMA),
            replan_interval=int(envs.VLLM_ASCEND_SPARSE_PREFIX_CACHE_REPLAN_INTERVAL),
            alpha=float(envs.VLLM_ASCEND_SPARSE_PREFIX_CACHE_LAPLACE_ALPHA),
            bin_size=int(envs.VLLM_ASCEND_SPARSE_PREFIX_CACHE_BIN_SIZE),
        )

        budget_bytes = int(
            float(envs.VLLM_ASCEND_SPARSE_PREFIX_CACHE_BUDGET_GB) * 1024**3
        )
        cache = SparsePrefixCache(gdn_budget_bytes=budget_bytes)

        block_size = int(envs.VLLM_ASCEND_SPARSE_PREFIX_CACHE_KERNEL_BLOCK_SIZE)
        return cls(tracker, cache, block_size)

    # ------------------------------------------------------------------
    # Placement
    # ------------------------------------------------------------------
    def checkpoint_positions(self, seq_len: int) -> list[int]:
        """Return checkpoint positions for a prefix of ``seq_len`` tokens.

        Positions are clipped down to ``block_size`` boundaries (the granularity
        at which vLLM commits recurrent state) and de-duplicated.
        """
        clipped: list[int] = []
        for pos in self.tracker.get_positions(seq_len):
            c = self._clip_to_block(pos)
            if c <= 0 or c in clipped:
                continue
            clipped.append(c)
        return sorted(clipped)

    def _clip_to_block(self, pos: int) -> int:
        return (pos // self.block_size) * self.block_size

    def observe(self, overlap_depth: int) -> None:
        """Feed an observed overlap depth into the placement histogram."""
        self.tracker.observe(overlap_depth)

    # ------------------------------------------------------------------
    # Capture / restore
    # ------------------------------------------------------------------
    def capture(
        self,
        prefix_key: Hashable,
        seq_len: int,
        snapshots: dict[int, StateSnapshot],
    ) -> None:
        """Capture checkpoints for a finished prefill of ``seq_len`` tokens.

        ``snapshots`` maps a checkpoint position to ``(recurrent_states,
        conv_states)``, each keyed by GDN layer index. The runner is responsible
        for slicing the live ``kv_cache[0]``/``kv_cache[1]`` tensors at each
        position before calling here; the coordinator clones the slices so the
        store owns its memory and is independent of the live cache.
        """
        store = PrefixCheckpointStore()
        for pos in self.checkpoint_positions(seq_len):
            snapshot = snapshots.get(pos)
            if snapshot is None:
                continue
            recurrent, conv = snapshot
            store.checkpoints[pos] = RecurrentCheckpoint(
                position=pos,
                recurrent_states={i: t.clone() for i, t in recurrent.items()},
                conv_states={i: t.clone() for i, t in conv.items()},
            )
        if store.checkpoints:
            self.cache.put(prefix_key, store)

    def restore(self, hit: SparseHit, recurrent_states, conv_states) -> bool:
        """Copy a hit's checkpoint state into the provided live tensors.

        Returns ``False`` if the checkpoint is no longer cached (evicted).
        """
        ckpt = self.cache.best_checkpoint(hit.prefix_key, hit.overlap_depth)
        if ckpt is None or ckpt.position != hit.checkpoint_position:
            return False
        for i, t in ckpt.recurrent_states.items():
            recurrent_states[i].copy_(t)
        for i, t in ckpt.conv_states.items():
            conv_states[i].copy_(t)
        return True

    # ------------------------------------------------------------------
    # Hit resolution
    # ------------------------------------------------------------------
    def lookup(self, prefix_key: Hashable, overlap_depth: int) -> SparseHit | None:
        """Resolve the nearest checkpoint ``<= overlap_depth`` for a prefix hit."""
        ckpt = self.cache.best_checkpoint(prefix_key, overlap_depth)
        if ckpt is None:
            return None
        return SparseHit(
            prefix_key=prefix_key,
            checkpoint_position=ckpt.position,
            overlap_depth=overlap_depth,
        )

    # ------------------------------------------------------------------
    # Pending-hit handoff (coordinator -> worker)
    # ------------------------------------------------------------------
    def register_pending_hit(self, request_id: str, hit: SparseHit) -> None:
        """Record a resolved hit whose checkpoint must be restored by the worker."""
        self._pending_hits[request_id] = hit

    def take_pending_hit(self, request_id: str) -> SparseHit | None:
        """Pop (and thus consume) a pending hit for ``request_id``, if any."""
        return self._pending_hits.pop(request_id, None)

    def stats(self) -> dict[str, object]:
        return {
            "type": "SparsePrefixCacheCoordinator",
            "block_size": self.block_size,
            "tracker": {
                "mode": self.tracker.mode,
                "n_obs": self.tracker.n_obs,
                "n_solves": self.tracker._n_solves,
            },
            "cache": self.cache.stats(),
            "pending_hits": len(self._pending_hits),
        }
