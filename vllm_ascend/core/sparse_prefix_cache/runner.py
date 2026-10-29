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
"""Runner-side tensor bridge for sparse prefix caching.

The :mod:`~vllm_ascend.core.sparse_prefix_cache.coordinator` owns the *pure*
orchestration (placement, lookup, budget); this module owns the *tensor*
plumbing that moves GatedDeltaNet recurrent/conv state between the live
block-indexed ``kv_cache[0]``/``kv_cache[1]`` tensors and the sparse checkpoint
store.

It reuses vLLM's own state-copy machinery: the per-layer ``MambaStateCopyFunc``
tuple (``get_conv_copy_spec``, ``get_temporal_copy_spec`` for GDN) already knows
how to locate the state slice for a block, so capture is "copy from a live
block to the store" and restore is "copy from the store to a live block" —
exactly what ``preprocess_mamba`` / ``postprocess_mamba`` do between two live
blocks, but across requests via a checkpoint.

The actual call sites live in ``vllm_ascend/worker/model_runner_v1.py`` and the
KV-cache coordinator; see the design doc's "Integration points" section. This
module keeps the tensor logic isolated so it can be reviewed and unit-tested
without the full runner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Hashable

from vllm_ascend.core.sparse_prefix_cache.checkpoint import RecurrentCheckpoint
from vllm_ascend.core.sparse_prefix_cache.coordinator import (
    SparseHit,
    SparsePrefixCacheCoordinator,
)

if TYPE_CHECKING:
    import torch
    from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFunc
    from vllm.v1.kv_cache_interface import KVCacheConfig

#: GDN state copy funcs: (conv, temporal). Matches
#: ``MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()``.
StateCopyFuncs = tuple["MambaStateCopyFunc", "MambaStateCopyFunc"]


def _view_from_spec(
    state: "torch.Tensor",
    start_addr: int,
    num_elements: int,
) -> "torch.Tensor":
    """Return a flat view of ``state`` over ``[start_addr, start_addr+num_elements)``.

    Mirrors ``_tensor_view_from_data_ptr`` in ``patch_mamba_utils.py`` so the
    sparse capture/restore path uses the same pointer arithmetic as the dense
    mamba copy path.
    """
    byte_offset = start_addr - state.data_ptr()
    element_size = state.element_size()
    if byte_offset < 0 or byte_offset % element_size != 0:
        raise RuntimeError("Invalid Mamba state copy pointer.")
    element_offset = byte_offset // element_size
    flat_state = state.view(-1)
    if element_offset + num_elements > flat_state.numel():
        raise RuntimeError("Mamba state copy range exceeds tensor storage.")
    return flat_state.narrow(0, element_offset, num_elements)


def capture_checkpoint_state(
    kv_caches_by_layer: dict[int, tuple["torch.Tensor", "torch.Tensor"]],
    block_ids: list[int],
    src_block_idx: int,
    state_copy_funcs: StateCopyFuncs,
) -> tuple[dict[int, "torch.Tensor"], dict[int, "torch.Tensor"]]:
    """Extract ``(recurrent_states, conv_states)`` at ``src_block_idx``.

    ``kv_caches_by_layer[i]`` is ``(conv_state, ssm_state)`` for GDN layer
    ``i`` (i.e. ``attention.kv_cache[0]`` / ``[1]``), and ``state_copy_funcs``
    is ``(get_conv_copy_spec, get_temporal_copy_spec)``. Returns dicts keyed by
    the same layer index.
    """
    conv_func, temporal_func = state_copy_funcs
    recurrent: dict[int, "torch.Tensor"] = {}
    conv: dict[int, "torch.Tensor"] = {}
    for layer_idx, (conv_state, ssm_state) in kv_caches_by_layer.items():
        conv_spec = conv_func(conv_state, block_ids, src_block_idx, 1)
        temporal_spec = temporal_func(ssm_state, block_ids, src_block_idx, 1)
        conv[layer_idx] = _view_from_spec(
            conv_state, conv_spec.start_addr, conv_spec.num_elements
        ).clone()
        recurrent[layer_idx] = _view_from_spec(
            ssm_state, temporal_spec.start_addr, temporal_spec.num_elements
        ).clone()
    return recurrent, conv


def restore_checkpoint_state(
    checkpoint: RecurrentCheckpoint,
    kv_caches_by_layer: dict[int, tuple["torch.Tensor", "torch.Tensor"]],
    block_ids: list[int],
    dest_block_idx: int,
    state_copy_funcs: StateCopyFuncs,
) -> None:
    """Copy a checkpoint's state into the live block at ``dest_block_idx``."""
    conv_func, temporal_func = state_copy_funcs
    for layer_idx, (conv_state, ssm_state) in kv_caches_by_layer.items():
        conv_spec = conv_func(conv_state, block_ids, dest_block_idx, 1)
        temporal_spec = temporal_func(ssm_state, block_ids, dest_block_idx, 1)
        _view_from_spec(conv_state, conv_spec.start_addr, conv_spec.num_elements).copy_(
            checkpoint.conv_states[layer_idx]
        )
        _view_from_spec(
            ssm_state, temporal_spec.start_addr, temporal_spec.num_elements
        ).copy_(checkpoint.recurrent_states[layer_idx])


class SparsePrefixCacheHook:
    """Runner-side facade over the coordinator.

    The model runner holds a single instance and drives it from three places:

    * after a prefill completes (``capture``);
    * before building attention metadata on a prefix-cache hit (``resolve``);
    * once a hit's checkpoint must be materialized into the fresh sequence's
      state block (``restore``).

    ``capture``/``restore`` take the live per-layer state tensors and the
    request's mamba block table, so the caller only has to pass through what it
    already has in ``forward_context`` and the scheduler output.
    """

    def __init__(
        self,
        coordinator: SparsePrefixCacheCoordinator,
        state_copy_funcs: StateCopyFuncs,
    ) -> None:
        self.coordinator = coordinator
        self.state_copy_funcs = state_copy_funcs

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------
    def capture(
        self,
        prefix_key: Hashable,
        seq_len: int,
        kv_caches_by_layer: dict[int, tuple["torch.Tensor", "torch.Tensor"]],
        block_ids: list[int],
    ) -> None:
        """Capture recurrent-state checkpoints for a finished prefill.

        Checkpoint positions come from the coordinator (DP-optimal + balanced
        fallback, clipped to block boundaries). ``seq_len`` is the total prefix
        length; ``block_ids`` is the request's mamba block table, where the
        state after token ``pos`` lives in ``block_ids[pos // block_size - 1]``.
        """
        block_size = self.coordinator.block_size
        positions = self.coordinator.checkpoint_positions(seq_len)
        snapshots: dict[int, tuple[dict, dict]] = {}
        for pos in positions:
            # pos is a block boundary; the block holding state-after-pos is the
            # one *ending* at pos, i.e. 0-indexed pos//block_size - 1.
            src_block_idx = pos // block_size - 1
            if src_block_idx < 0 or src_block_idx >= len(block_ids):
                continue
            snapshots[pos] = capture_checkpoint_state(
                kv_caches_by_layer, block_ids, src_block_idx, self.state_copy_funcs
            )
        if snapshots:
            self.coordinator.capture(prefix_key, seq_len, snapshots)

    # ------------------------------------------------------------------
    # Hit resolution / restore
    # ------------------------------------------------------------------
    def resolve(self, prefix_key: Hashable, overlap_depth: int) -> SparseHit | None:
        """Resolve a prefix-cache hit to a sparse checkpoint (or ``None``)."""
        return self.coordinator.lookup(prefix_key, overlap_depth)

    def restore(
        self,
        hit: SparseHit,
        kv_caches_by_layer: dict[int, tuple["torch.Tensor", "torch.Tensor"]],
        block_ids: list[int],
        dest_block_idx: int,
    ) -> bool:
        """Materialize ``hit``'s checkpoint into the fresh sequence's state block."""
        ckpt = self.coordinator.cache.best_checkpoint(
            hit.prefix_key, hit.overlap_depth
        )
        if ckpt is None or ckpt.position != hit.checkpoint_position:
            return False
        restore_checkpoint_state(
            ckpt, kv_caches_by_layer, block_ids, dest_block_idx, self.state_copy_funcs
        )
        return True

    def observe(self, overlap_depth: int) -> None:
        """Feed an observed overlap depth into the placement histogram."""
        self.coordinator.observe(overlap_depth)

    def stats(self) -> dict[str, object]:
        return self.coordinator.stats()


# ---------------------------------------------------------------------------
# Process-wide singleton (single-process vLLM engine)
# ---------------------------------------------------------------------------
# The scheduler-side KV-cache coordinator and the worker-side model runner both
# drive the sparse cache. In the default single-process engine they share one
# address space, so the runner registers its hook here and the coordinator reads
# it during ``find_longest_cache_hit``. A multiprocessing engine would need an
# IPC handoff instead (out of scope for the initial port).
_global_hook: "SparsePrefixCacheHook | None" = None


def set_sparse_prefix_cache_hook(hook: "SparsePrefixCacheHook | None") -> None:
    """Register (or clear) the process-wide sparse prefix cache hook."""
    global _global_hook
    _global_hook = hook


def get_sparse_prefix_cache_hook() -> "SparsePrefixCacheHook | None":
    """Return the process-wide sparse prefix cache hook, if registered."""
    return _global_hook
