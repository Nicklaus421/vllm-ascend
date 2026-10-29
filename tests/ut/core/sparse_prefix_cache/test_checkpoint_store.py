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
"""Unit tests for the sparse checkpoint containers and budgeted store."""

from vllm_ascend.core.sparse_prefix_cache import (
    PrefixCheckpointStore,
    RecurrentCheckpoint,
    SparsePrefixCache,
)
from vllm_ascend.core.sparse_prefix_cache.store import _truncate_gdn


class FakeTensor:
    """Minimal tensor stand-in exposing just ``nelement``/``element_size``."""

    def __init__(self, nbytes: int) -> None:
        self._nbytes = nbytes

    def nelement(self) -> int:
        return 1

    def element_size(self) -> int:
        return self._nbytes

    def to(self, device):
        return self


def _make_ckpt(position: int, nbytes: int) -> RecurrentCheckpoint:
    return RecurrentCheckpoint(
        position=position,
        recurrent_states={0: FakeTensor(nbytes)},
        conv_states={0: FakeTensor(nbytes // 2)},
    )


def _make_store(positions: list[int], nbytes: int) -> PrefixCheckpointStore:
    store = PrefixCheckpointStore()
    for pos in positions:
        store.checkpoints[pos] = _make_ckpt(pos, nbytes)
    return store


# ---------------------------------------------------------------------------
# RecurrentCheckpoint
# ---------------------------------------------------------------------------
def test_recurrent_checkpoint_num_bytes():
    ckpt = _make_ckpt(10, 100)
    # recurrent (100) + conv (50)
    assert ckpt.num_bytes() == 150


def test_recurrent_checkpoint_to_returns_self():
    ckpt = _make_ckpt(10, 100)
    assert ckpt.to("cpu") is ckpt


# ---------------------------------------------------------------------------
# PrefixCheckpointStore
# ---------------------------------------------------------------------------
def test_best_checkpoint_nearest_at_or_before():
    store = _make_store([16, 64, 128], 100)
    assert store.best_checkpoint(100).position == 64
    assert store.best_checkpoint(128).position == 128
    assert store.best_checkpoint(15) is None  # all checkpoints > seq_len


def test_best_checkpoint_empty_store():
    store = PrefixCheckpointStore()
    assert store.best_checkpoint(10) is None


def test_gdn_bytes_sums_all_checkpoints():
    store = _make_store([16, 64], 100)
    assert store.gdn_bytes() == 2 * 150


# ---------------------------------------------------------------------------
# _truncate_gdn
# ---------------------------------------------------------------------------
def test_truncate_gdn_evicts_smallest_position_first():
    store = _make_store([16, 64, 128], 100)  # 450 bytes total
    _truncate_gdn(store, gdn_budget=150)
    # Evict positions 16 then 64 until only 128 (150 bytes) remains.
    assert list(store.checkpoints.keys()) == [128]


def test_truncate_gdn_noop_when_under_budget():
    store = _make_store([16, 64], 100)
    _truncate_gdn(store, gdn_budget=1000)
    assert sorted(store.checkpoints.keys()) == [16, 64]


# ---------------------------------------------------------------------------
# SparsePrefixCache
# ---------------------------------------------------------------------------
def test_put_and_get():
    cache = SparsePrefixCache(gdn_budget_bytes=1000)
    store = _make_store([16, 64], 100)
    cache.put("k", store)
    assert cache.n_entries == 1
    assert cache.get("k") is store
    assert cache.get("missing") is None


def test_fifo_eviction_under_budget():
    cache = SparsePrefixCache(gdn_budget_bytes=300)
    cache.put("a", _make_store([16], 100))  # 150 bytes
    cache.put("b", _make_store([32], 100))  # 150 bytes -> total 300
    assert cache.n_entries == 2
    cache.put("c", _make_store([64], 100))  # needs 150, evicts "a"
    assert cache.n_entries == 2
    assert cache.get("a") is None
    assert cache.get("b") is not None
    assert cache.get("c") is not None


def test_best_checkpoint_through_cache():
    cache = SparsePrefixCache(gdn_budget_bytes=1000)
    cache.put("k", _make_store([16, 64, 128], 100))
    ckpt = cache.best_checkpoint("k", 100)
    assert ckpt is not None and ckpt.position == 64
    assert cache.best_checkpoint("missing", 100) is None
