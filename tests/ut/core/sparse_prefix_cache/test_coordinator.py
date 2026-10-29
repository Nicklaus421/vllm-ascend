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
"""Unit tests for the sparse prefix-cache coordinator."""

from vllm_ascend.core.sparse_prefix_cache import (
    SparseHit,
    SparsePrefixCacheCoordinator,
)
from vllm_ascend.core.sparse_prefix_cache.store import SparsePrefixCache
from vllm_ascend.core.sparse_prefix_cache.strategies import HistogramTracker


class FakeTensor:
    """Minimal tensor stand-in exposing nelement/element_size/clone/copy_."""

    def __init__(self, nbytes: int) -> None:
        self._nbytes = nbytes

    def nelement(self) -> int:
        return 1

    def element_size(self) -> int:
        return self._nbytes

    def to(self, device):
        return self

    def clone(self) -> "FakeTensor":
        return FakeTensor(self._nbytes)

    def copy_(self, other) -> "FakeTensor":
        return self


def _make_coordinator(
    budget: int = 4, block_size: int = 16
) -> SparsePrefixCacheCoordinator:
    tracker = HistogramTracker(max_len=1024, budget=budget)
    cache = SparsePrefixCache(gdn_budget_bytes=10_000)
    return SparsePrefixCacheCoordinator(tracker, cache, block_size=block_size)


# ---------------------------------------------------------------------------
# SparseHit
# ---------------------------------------------------------------------------
def test_sparse_hit_properties():
    hit = SparseHit(prefix_key="k", checkpoint_position=48, overlap_depth=100)
    assert hit.recompute_start == 48
    assert hit.recompute_end == 100
    assert hit.gap == 52
    assert not hit.is_exact


def test_sparse_hit_exact():
    hit = SparseHit(prefix_key="k", checkpoint_position=64, overlap_depth=64)
    assert hit.gap == 0
    assert hit.is_exact


# ---------------------------------------------------------------------------
# checkpoint_positions (clipping + dedup)
# ---------------------------------------------------------------------------
def test_checkpoint_positions_clipped_to_block():
    coord = _make_coordinator(budget=4, block_size=16)
    # balanced fallback for seq_len=100, budget=4 -> [20, 40, 60, 80]
    # clipped to multiples of 16 -> [16, 32, 48, 80]
    assert coord.checkpoint_positions(100) == [16, 32, 48, 80]


def test_checkpoint_positions_dedup_when_clipped_together():
    coord = _make_coordinator(budget=4, block_size=32)
    # [20, 40, 60, 80] clip to [0, 32, 32, 64] -> 0 dropped, 40/60 deduped.
    assert coord.checkpoint_positions(100) == [32, 64]


# ---------------------------------------------------------------------------
# capture / lookup
# ---------------------------------------------------------------------------
def test_capture_and_lookup():
    coord = _make_coordinator(budget=4, block_size=16)
    snapshots = {
        pos: ({0: FakeTensor(100)}, {0: FakeTensor(50)})
        for pos in coord.checkpoint_positions(100)
    }
    coord.capture("k", 100, snapshots)

    store = coord.cache.get("k")
    assert store is not None and store.num_checkpoints == 4

    hit = coord.lookup("k", 50)
    assert hit is not None
    assert hit.checkpoint_position == 48
    assert hit.overlap_depth == 50
    assert hit.gap == 2


def test_lookup_miss():
    coord = _make_coordinator()
    assert coord.lookup("missing", 100) is None


def test_lookup_none_when_checkpoints_after_seq_len():
    coord = _make_coordinator(budget=4, block_size=16)
    snapshots = {
        pos: ({0: FakeTensor(100)}, {0: FakeTensor(50)})
        for pos in coord.checkpoint_positions(100)
    }
    coord.capture("k", 100, snapshots)
    # all checkpoints are > 15
    assert coord.lookup("k", 15) is None


# ---------------------------------------------------------------------------
# restore
# ---------------------------------------------------------------------------
def test_restore_copies_checkpoint_state():
    coord = _make_coordinator(budget=4, block_size=16)
    snapshots = {
        pos: ({0: FakeTensor(100)}, {0: FakeTensor(50)})
        for pos in coord.checkpoint_positions(100)
    }
    coord.capture("k", 100, snapshots)
    hit = coord.lookup("k", 50)
    assert hit is not None

    recurrent = {0: FakeTensor(0)}
    conv = {0: FakeTensor(0)}
    assert coord.restore(hit, recurrent, conv) is True


def test_restore_false_on_evicted():
    coord = _make_coordinator(budget=4, block_size=16)
    hit = SparseHit(prefix_key="k", checkpoint_position=48, overlap_depth=50)
    assert coord.restore(hit, {0: FakeTensor(0)}, {0: FakeTensor(0)}) is False
