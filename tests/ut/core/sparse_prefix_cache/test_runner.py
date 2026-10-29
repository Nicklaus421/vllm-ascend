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
"""Unit tests for the runner-side sparse prefix cache hook.

The tensor copy functions are monkeypatched (they need torch/NPU); these tests
exercise the hook's *coordination* — capture placement, hit resolution and
restore — against the pure coordinator.
"""

import pytest

from vllm_ascend.core.sparse_prefix_cache import (
    SparsePrefixCacheCoordinator,
    SparsePrefixCacheHook,
)
from vllm_ascend.core.sparse_prefix_cache.runner import (
    capture_checkpoint_state,
    restore_checkpoint_state,
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


def _make_hook(budget: int = 4, block_size: int = 16) -> SparsePrefixCacheHook:
    tracker = HistogramTracker(max_len=1024, budget=budget)
    cache = SparsePrefixCache(gdn_budget_bytes=10_000)
    coord = SparsePrefixCacheCoordinator(tracker, cache, block_size=block_size)
    return SparsePrefixCacheHook(coord, state_copy_funcs=(None, None))


@pytest.fixture
def patch_tensor_copy(monkeypatch):
    monkeypatch.setattr(
        "vllm_ascend.core.sparse_prefix_cache.runner.capture_checkpoint_state",
        lambda kv, blk, idx, funcs: ({0: FakeTensor(100)}, {0: FakeTensor(50)}),
    )
    monkeypatch.setattr(
        "vllm_ascend.core.sparse_prefix_cache.runner.restore_checkpoint_state",
        lambda ckpt, kv, blk, idx, funcs: None,
    )


def test_hook_capture_resolve_restore(patch_tensor_copy):
    hook = _make_hook()
    hook.capture("k", 100, {0: (FakeTensor(0), FakeTensor(0))}, list(range(100)))

    store = hook.coordinator.cache.get("k")
    assert store is not None and store.num_checkpoints == 4

    hit = hook.resolve("k", 50)
    assert hit is not None
    assert hit.checkpoint_position == 48
    assert hit.gap == 2

    assert hook.restore(hit, {0: (FakeTensor(0), FakeTensor(0))}, list(range(100)), 0)
    assert hook.resolve("missing", 50) is None


def test_hook_observe(patch_tensor_copy):
    hook = _make_hook()
    hook.observe(50)
    assert hook.coordinator.tracker.n_obs == 1
