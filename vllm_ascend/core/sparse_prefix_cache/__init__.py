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

from vllm_ascend.core.sparse_prefix_cache.checkpoint import (
    PrefixCheckpointStore,
    RecurrentCheckpoint,
)
from vllm_ascend.core.sparse_prefix_cache.coordinator import (
    SparseHit,
    SparsePrefixCacheCoordinator,
)
from vllm_ascend.core.sparse_prefix_cache.runner import (
    SparsePrefixCacheHook,
    capture_checkpoint_state,
    get_sparse_prefix_cache_hook,
    restore_checkpoint_state,
    set_sparse_prefix_cache_hook,
)
from vllm_ascend.core.sparse_prefix_cache.store import SparsePrefixCache
from vllm_ascend.core.sparse_prefix_cache.strategies import (
    HistogramTracker,
    backtrack,
    balanced_positions,
    laplace_smoothing,
    solve_dp,
)

__all__ = [
    "HistogramTracker",
    "PrefixCheckpointStore",
    "RecurrentCheckpoint",
    "SparseHit",
    "SparsePrefixCache",
    "SparsePrefixCacheCoordinator",
    "SparsePrefixCacheHook",
    "backtrack",
    "balanced_positions",
    "capture_checkpoint_state",
    "get_sparse_prefix_cache_hook",
    "laplace_smoothing",
    "restore_checkpoint_state",
    "set_sparse_prefix_cache_hook",
    "solve_dp",
]
