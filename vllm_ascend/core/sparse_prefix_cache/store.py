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
"""Budgeted cache for sparse recurrent-state checkpoints.

Attention KV is cached by vLLM's native block pool; this store only accounts
for the GDN recurrent-state bytes, which is the budget that sparse checkpoint
placement is trying to shrink.
"""

from collections import OrderedDict
from typing import Hashable

from vllm_ascend.core.sparse_prefix_cache.checkpoint import (
    PrefixCheckpointStore,
)


def _truncate_gdn(
    store: PrefixCheckpointStore, gdn_budget: int
) -> PrefixCheckpointStore:
    """Drop checkpoints (smallest position first) until GDN fits the budget."""
    gdn_size = store.gdn_bytes()
    if gdn_size <= gdn_budget:
        return store
    for pos in sorted(store.checkpoints.keys()):
        ckpt = store.checkpoints.pop(pos)
        gdn_size -= ckpt.num_bytes()
        if gdn_size <= gdn_budget:
            return store
    return store  # all checkpoints removed


class SparsePrefixCache:
    """FIFO cache of sparse recurrent checkpoints under a GDN byte budget.

    Entries are keyed by an opaque hashable prefix identity (the coordinator
    uses the same block-hash chain that vLLM's prefix cache uses). Lookup
    returns the nearest checkpoint at or before ``seq_len``.

    Not thread-safe: it is meant to be driven by the single-threaded scheduler
    loop, matching the reference implementation.
    """

    def __init__(self, gdn_budget_bytes: int) -> None:
        self.gdn_budget = gdn_budget_bytes
        self.gdn_used = 0
        self.entries: OrderedDict[Hashable, tuple[PrefixCheckpointStore, int]] = (
            OrderedDict()
        )

    def put(self, key: Hashable, store: PrefixCheckpointStore) -> None:
        _truncate_gdn(store, self.gdn_budget)
        gdn_b = store.gdn_bytes()
        if gdn_b > self.gdn_budget:
            return

        while self.entries and self.gdn_used + gdn_b > self.gdn_budget:
            _, (_, evicted_gdn) = self.entries.popitem(last=False)
            self.gdn_used -= evicted_gdn

        self.entries[key] = (store, gdn_b)
        self.gdn_used += gdn_b

    def get(self, key: Hashable) -> PrefixCheckpointStore | None:
        entry = self.entries.get(key)
        return entry[0] if entry is not None else None

    def best_checkpoint(self, key: Hashable, seq_len: int):
        store = self.get(key)
        return store.best_checkpoint(seq_len) if store is not None else None

    @property
    def n_entries(self) -> int:
        return len(self.entries)

    def stats(self) -> dict[str, object]:
        return {
            "type": "SparsePrefixCache",
            "n_entries": self.n_entries,
            "gdn_budget_bytes": self.gdn_budget,
            "gdn_used_bytes": self.gdn_used,
        }
