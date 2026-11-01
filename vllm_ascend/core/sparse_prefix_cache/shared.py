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
"""Cross-process shared state for the DP-optimal (v2) sparse prefix cache.

The overlap-depth histogram and the DP solve both happen in the EngineCore
process (the scheduler-side ``AscendHybridKVCacheCoordinator``). Under TP>1 the
``NPUModelRunner`` lives in worker subprocesses, so the solved checkpoint
positions are published to a ``multiprocessing.Manager`` proxy and read back by
the workers through :func:`get_sparse_positions`.

Lifecycle:

- The EngineCore process creates the manager + proxy with
  :func:`init_sparse_shared_state` and publishes solved positions with
  :func:`publish_sparse_positions`.
- Workers receive the proxy through ``process_kwargs`` (spawn-safe; see
  ``patch_multiproc_executor.py``) and re-attach it with
  :func:`attach_sparse_shared_state`. Under ``fork`` the module singleton is
  inherited anyway, and re-attaching the same proxy is idempotent.
- Before the first solve, :func:`get_sparse_positions` returns ``None``; callers
  fall back to the deterministic fixed-interval placement, which is
  prefix-consistent and cross-process safe.
"""

from __future__ import annotations

from typing import Any

#: The Manager-backed shared dict (or ``None`` when sparse caching is disabled
#: or the ``fixed_interval`` strategy is in use). Set in the EngineCore process
#: and re-attached in each worker subprocess.
_shared_state: Any = None

# Keys stored in the shared dict.
_KEY_POSITIONS = "fixed_positions"
_KEY_READY = "ready"


def init_sparse_shared_state() -> Any:
    """Create (once) and return the Manager-backed shared dict.

    The proxy lives in the EngineCore process that owns the scheduler-side
    coordinator. Idempotent: subsequent calls return the existing proxy.
    """
    global _shared_state
    if _shared_state is not None:
        return _shared_state
    try:
        # Use the same multiprocessing context vLLM uses to spawn the TP
        # workers, so the manager server and the workers agree on start method
        # and authkey (the proxy is then passed through ``process_kwargs``).
        from vllm.utils.system_utils import get_mp_context

        manager = get_mp_context().Manager()
    except Exception:  # pragma: no cover - fallback for unit tests without vllm
        from multiprocessing import Manager

        manager = Manager()

    _shared_state = manager.dict()
    _shared_state[_KEY_POSITIONS] = []
    _shared_state[_KEY_READY] = False
    return _shared_state


def attach_sparse_shared_state(proxy: Any) -> None:
    """Attach a shared-state proxy received from the parent process.

    No-op when ``proxy`` is ``None`` (sparse caching disabled / fixed_interval).
    Under ``fork`` the singleton is inherited already, but re-attaching the same
    proxy is harmless.
    """
    global _shared_state
    if proxy is not None:
        _shared_state = proxy


def get_sparse_shared_state() -> Any:
    """Return the current shared-state proxy (``None`` if not initialized)."""
    return _shared_state


def get_sparse_positions() -> list[int] | None:
    """Return the frozen DP checkpoint positions, or ``None`` before the solve.

    The returned list is already sorted and clipped to block boundaries by the
    publisher; callers filter it to ``<= seq_len`` (prefix-consistent).
    """
    state = _shared_state
    if state is None:
        return None
    if not state.get(_KEY_READY, False):
        return None
    return list(state.get(_KEY_POSITIONS, []))


def publish_sparse_positions(positions: list[int]) -> None:
    """Publish the solved, frozen DP checkpoint positions to the workers."""
    state = _shared_state
    if state is None:
        return
    state[_KEY_POSITIONS] = list(positions)
    state[_KEY_READY] = True
