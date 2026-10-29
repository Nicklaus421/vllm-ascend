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
"""Sparse recurrent-state checkpoint containers.

The data model mirrors the reference implementation of
*Sparse Prefix Caching for Hybrid and Recurrent LLM Serving*
(arXiv:2605.05219): a checkpoint stores the GatedDeltaNet recurrent + conv
state for a single token position, and a store holds the sparse set of
checkpoints captured for one token prefix.

Attention KV is deliberately *not* stored here — vLLM already manages it in
the native block pool via hash-based prefix caching. This store only carries
the recurrent state, which is the expensive part being sparsified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass
class RecurrentCheckpoint:
    """GDN recurrent + conv state captured at a single token position.

    ``recurrent_states`` and ``conv_states`` are keyed by the GDN (linear
    attention) layer index; their tensor shapes follow vLLM's per-layer state
    layout (recurrent ``(H, Dk, Dv)``, conv ``(C, K)`` for one sequence).
    """

    position: int
    recurrent_states: dict[int, torch.Tensor]
    conv_states: dict[int, torch.Tensor]

    def to(self, device: torch.device | str) -> RecurrentCheckpoint:
        self.recurrent_states = {
            k: v.to(device) for k, v in self.recurrent_states.items()
        }
        self.conv_states = {
            k: v.to(device) for k, v in self.conv_states.items()
        }
        return self

    def num_bytes(self) -> int:
        total = 0
        for t in self.recurrent_states.values():
            total += t.nelement() * t.element_size()
        for t in self.conv_states.values():
            total += t.nelement() * t.element_size()
        return total


@dataclass
class PrefixCheckpointStore:
    """Sparse recurrent-state checkpoints captured for one token prefix."""

    checkpoints: dict[int, RecurrentCheckpoint] = field(default_factory=dict)

    def best_checkpoint(self, seq_len: int) -> RecurrentCheckpoint | None:
        """Return the checkpoint with the largest position ``<= seq_len``."""
        best: RecurrentCheckpoint | None = None
        for pos in sorted(self.checkpoints.keys()):
            if pos <= seq_len:
                best = self.checkpoints[pos]
        return best

    @property
    def num_checkpoints(self) -> int:
        return len(self.checkpoints)

    def gdn_bytes(self) -> int:
        return sum(c.num_bytes() for c in self.checkpoints.values())

    def to(self, device: torch.device | str) -> PrefixCheckpointStore:
        for ckpt in self.checkpoints.values():
            ckpt.to(device)
        return self
