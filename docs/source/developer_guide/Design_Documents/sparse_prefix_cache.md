# Sparse Prefix Caching for Hybrid LLMs

TL;DR: Cache GatedDeltaNet (GDN) recurrent state at a sparse set of
checkpoint positions instead of every block, and recompute the gap on a
prefix-cache hit. This shrinks the recurrent-state footprint so more
sequences fit in the cache, raising the hit rate.

Based on *Sparse Prefix Caching for Hybrid and Recurrent LLM Serving*
(arXiv:2605.05219) and its reference implementation
[arabel1a/sparse-prefix-caching](https://github.com/arabel1a/sparse-prefix-caching).

## What is Sparse Prefix Caching?

Hybrid models such as Qwen3.5/Qwen3.8 interleave full-attention layers with
linear-attention (recurrent) layers, e.g. GatedDeltaNet (GDN). The two layer
types have very different cache costs:

| State | Size per head | Cache policy |
|---|---|---|
| Attention KV | `d_head` per token | Dense (every block) |
| GDN recurrent state | `d_head²` per head | **Sparse checkpoints** |

The key observation is that a recurrent layer can resume from a single stored
state, so per-token dense caching is unnecessary. Because recurrent state is
far more expensive than KV (ratio
`r = 3·n_v·d_h² / (2·n_kv·d_a)`; `r ≈ 1152` for the target Qwen3.8-27B),
storing fewer checkpoints lets more sequences fit in the same budget and
raises the cache hit rate.

On a prefix-cache hit at position `P`, the nearest stored checkpoint `C <= P`
is restored and the gap `[C, P)` is recomputed (full model, attention reading
cached KV) before continuing with the new tokens.

## Checkpoint Placement

All strategies keep the full attention KV cache. They differ only in where
recurrent-state checkpoints are placed (clipped to block boundaries for kernel
compatibility):

| Strategy | Description |
|---|---|
| `no_cache` | No checkpoints, full recompute |
| `block` | Checkpoint every `block_size` tokens |
| `balanced` | `n_blocks` evenly spaced |
| `sqrt` | `sqrt(L)`-spaced |
| `dyadic` | Positions at powers of two |
| `fixed_interval` | Fixed token interval (prefix-consistent, cross-process safe) |
| `hist_exp_decay` | DP-optimal on the empirical overlap distribution with exponential reweighting for drift |

The v1 implementation ships the deterministic `fixed_interval` strategy (the
default). It is prefix-consistent — positions are fixed multiples of the
interval, so the scheduler-side coordinator and the TP workers derive identical
checkpoint positions from the environment alone, with no cross-process state.

The DP-optimal `hist_exp_decay` strategy (the paper's core contribution,
Corollary 1) is implemented in v2. The overlap-depth histogram accumulates in
the EngineCore process (scheduler-side coordinator), the DP is solved once after
a warmup and frozen, and the resulting prefix-consistent checkpoint positions
are published to the TP workers through a `multiprocessing.Manager` proxy
(`shared.py`). Until the first solve, `hist_exp_decay` falls back to the
deterministic `fixed_interval` placement, so it is always cross-process
consistent.

## Architecture

The design keeps attention KV on vLLM's native hash-based prefix cache
(unchanged) and introduces a separate, standalone sparse checkpoint store for
the GDN recurrent state. This is intentionally less invasive than extending the
block pool: the recurrent state is decoupled from the KV cache and managed on
its own budget.

```
vllm_ascend/core/sparse_prefix_cache/
├── __init__.py
├── strategies.py     # DP solver + HistogramTracker + balanced fallback
├── checkpoint.py     # RecurrentCheckpoint / PrefixCheckpointStore
├── store.py          # budgeted FIFO cache (KV budget + GDN budget)
├── coordinator.py    # hit -> (hit_pos, checkpoint_pos) -> trigger recompute
└── shared.py         # Manager-backed cross-process DP placement (v2)
```

### Recompute path

vLLM advances attention KV and recurrent state in lockstep. Sparse caching
deliberately breaks that coupling: after a hit at `P`, attention KV is dense up
to `P` but recurrent state is only valid up to `C <= P`. The coordinator
therefore rewinds the effective `num_computed_tokens` to `C` and runs a
chunked prefill over `[C, P)` with an initial recurrent state.

This reuses the existing resume machinery rather than new operators:

- `has_initial_state` (`context_lens > 0`)
- `clear_ssm_states`
- `chunk_gated_delta_rule(initial_state=...)`

The recurrent state at layer `i` depends on the attention output at layer
`i-1`, so the recompute runs the full model over `[C, P)` with attention
reading the cached KV for `[0, C)`.

## Configuration

Environment variables are defined centrally in `vllm_ascend/envs.py`:

- `VLLM_ASCEND_SPARSE_PREFIX_CACHE_ENABLED`
- `VLLM_ASCEND_SPARSE_PREFIX_CACHE_STRATEGY` (`fixed_interval` default;
  `hist_exp_decay` for DP-optimal v2 placement)
- `VLLM_ASCEND_SPARSE_PREFIX_CACHE_BUDGET_GB`
- `VLLM_ASCEND_SPARSE_PREFIX_CACHE_BIN_SIZE`
- `VLLM_ASCEND_SPARSE_PREFIX_CACHE_GAMMA`
- `VLLM_ASCEND_SPARSE_PREFIX_CACHE_REPLAN_INTERVAL`
- `VLLM_ASCEND_SPARSE_PREFIX_CACHE_LAPLACE_ALPHA`
- `VLLM_ASCEND_SPARSE_PREFIX_CACHE_KERNEL_BLOCK_SIZE`
- `VLLM_ASCEND_SPARSE_PREFIX_CACHE_CHECKPOINT_INTERVAL` (v1 fixed-interval
  placement, default 256; prefix-consistent so the coordinator and TP worker
  agree on `C` without cross-process state)
- `VLLM_ASCEND_SPARSE_PREFIX_CACHE_WARMUP_REQUESTS` (v2, default 100; number
  of observed cache hits before the DP is solved once and frozen)
- `VLLM_ASCEND_SPARSE_PREFIX_CACHE_NUM_CHECKPOINTS` (v2, default 16; the DP
  checkpoint budget `M`, sized so `M * recurrent_state_bytes` fits the memory
  budget)

## How It Works

### HistogramTracker (DP-optimal)

Tracks the empirical distribution of prefix-overlap depth across requests and
solves an `O(NM)` dynamic program to place `M` checkpoints where they minimize
expected recompute cost. Three modes control replanning:

- `frozen`: accumulate during warmup, solve once, freeze.
- `periodic`: re-solve every `replan_interval` observations.
- `exp_decay`: exponential decay weighting (`gamma^(n-1-i)`) to follow drift,
  re-solve every `replan_interval`.

Before the first DP solve, `get_positions` falls back to balanced placement.

### Checkpoint store

`PrefixCheckpointStore` holds, for one token prefix, the sparse
`RecurrentCheckpoint`s (recurrent + conv states per layer) plus the full
attention KV. `best_checkpoint(seq_len)` returns the latest checkpoint at or
before `seq_len`. The store enforces separate KV and GDN byte budgets with FIFO
eviction; GDN checkpoints are evicted from the smallest position first.

## Integration points

The pure orchestration is in `vllm_ascend/core/sparse_prefix_cache/`; the tensor
plumbing is `runner.py` (`capture_checkpoint_state` / `restore_checkpoint_state`),
which reuses vLLM's `MambaStateCopyFunc` machinery to move state between live
blocks and the store. Both v1 (`fixed_interval`) and v2 (`hist_exp_decay`) are
**cross-process safe**: the scheduler-side coordinator and the TP workers run in
separate processes, so instead of sharing a store they both derive checkpoint
positions from the same deterministic, prefix-consistent placement, and the
worker keeps its own per-process state buffer keyed by the token prefix.

- **v1** derives positions from the stateless `fixed_interval_positions` (no
  shared state).
- **v2** accumulates the overlap histogram in the coordinator, solves the DP
  once after a warmup, freezes the positions, and publishes them to the workers
  through a `multiprocessing.Manager` proxy (`shared.py`); both sides read the
  same frozen list via `get_checkpoint_positions(seq_len)`.

1. **Capture after prefill** — in `vllm_ascend/worker/model_runner_v1.py`
   (`_update_states_after_model_execute`, entry point `_maybe_capture_sparse_checkpoints`),
   after the recurrent state is committed to its block during the forward, the
   worker slices the state at each checkpoint position and stores it in
   `self._sparse_state_buffer`, keyed by `tuple(prompt_token_ids[:pos])`.
   `block_ids` is the request's mamba block table, where the state after token
   `pos` lives in `block_ids[pos // block_size - 1]`. Multi-position capture
   requires `mamba_cache_mode='all'` (block-indexed recurrent state).

2. **Hit resolution** — in `vllm_ascend/patch/platform/patch_kv_cache_coordinator.py`
   (`AscendHybridKVCacheCoordinator.find_longest_cache_hit`), the full-attention
   group already yields the dense overlap depth `P`. For the Mamba group,
   `_resolve_sparse_mamba_hit(P)` returns the largest checkpoint `C <= P` (`0`
   when none), rewinding the hybrid `num_computed_tokens` to `C`. For v2, the
   coordinator also feeds `P` into the DP histogram via `_observe_sparse_hit(P)`
   (full attention is always processed first, so `P` is the dense overlap depth)
   and, once the warmup threshold is reached, solves the DP once, clips the
   positions to block boundaries, and publishes them to the workers through
   `shared.py`.

3. **Restore + recompute** — in `vllm_ascend/worker/model_runner_v1.py`
   (`execute_model`, `_maybe_restore_sparse_checkpoints`, before the forward), the
   worker looks up `tuple(prompt_token_ids[:C])` in `self._sparse_state_buffer`
   and, on a hit, copies the checkpoint into the sequence's state block
   (`dest_block_idx = C // block_size - 1`). `vllm_ascend/ops/gdn.py`
   (`AscendGatedDeltaNetAttention._forward_core`) then reads the restored
   `initial_state = ssm_state[prefill_state_indices]` and runs
   `chunk_gated_delta_rule(initial_state=...)` over `[C, P)` exactly as the
   existing resume path does. A buffer miss (dense hit or chunked continuation)
   leaves the already-present state untouched.

   The buffer is keyed by the *token prefix* (`tuple(prompt_token_ids[:pos])`),
   not `block_hashes`, because the worker-side `CachedRequestState` exposes
   `prompt_token_ids`/`req_id` but not `block_hashes`.

`vllm_ascend/ops/gdn_attn_builder.py`
(`AscendGDNAttentionMetadataBuilder.build`) is where `common_prefix_len`,
`num_computed_tokens` and `prefill_state_indices` are materialized into
`GDNAttentionMetadata`; the rewind from `P` to `C` reaches it through the
coordinator's reduced `hit_length` (`num_computed_tokens = C`), so no change is
needed there.

Both the coordinator and worker call the shared `get_checkpoint_positions(seq_len)`
entry point, which dispatches on the configured strategy. `fixed_interval` is
prefix-consistent: the positions for a hit of length `P` are exactly the subset
of the positions for the longer originally-captured prefix, so the coordinator
and worker always agree on `C` with zero cross-process communication.

The v2 `hist_exp_decay` placement is also prefix-consistent and stable: the
`HistogramTracker` runs in `frozen` mode with `adaptive_backtrack=False`, so a
single DP solve produces a fixed list of positions that is filtered to
`<= seq_len` on both sides. The solved positions are clipped to block boundaries
by the coordinator before publishing, then shared with the workers through
`shared.py`:

- The EngineCore process creates the Manager proxy
  (`init_sparse_shared_state`) and publishes the frozen positions
  (`publish_sparse_positions`).
- `AscendWorkerProc.make_worker_process` reads the proxy and hands it to each
  worker through `process_kwargs`; `_ascend_worker_main` re-attaches it via
  `attach_sparse_shared_state` before the worker starts (spawn-safe; under
  `fork` the module singleton is inherited).
- `get_checkpoint_positions` reads `get_sparse_positions()` and, before the
  first solve (warmup), falls back to `fixed_interval_positions`.

The v2 capture/restore path reuses the same per-process token-prefix buffer as
v1, so no change is needed in `model_runner_v1.py` beyond what v1 already does.

> **Transition caveat (v2).** The DP solve is a one-time placement switch from
> the warmup `fixed_interval` fallback to the frozen DP positions. A prefix that
> was prefilled (and captured) during warmup at fixed-interval positions, and is
> then hit after the solve, resolves to a DP position that the worker buffer does
> not hold. For full correctness across the transition, run the warmup as a
> distinct throwaway phase and reset the prefix cache (e.g. the engine's
> `/reset_prefix_cache` API) before the steady-state workload, so every prefix is
> re-captured at the frozen DP positions.

## Risks

1. **Coordinator integration** is the hardest part: translating "hit at `P`,
   recurrent state valid to `C`" into `num_computed_tokens = C` plus a recompute
   segment.
2. **Qwen3.8 architecture** is not yet in-tree; verify it matches the Qwen3.5
   GDN layer structure (or port a model patch) before Phase 1.
3. **Recompute cost on NPU** — the extra `[C, P)` prefill on the 910B4 must be
   weighed against the memory saved. Tune the checkpoint interval to maximize
   net benefit.
4. **v2 transition correctness** — the DP solve switches placement from the
   warmup `fixed_interval` fallback to frozen DP positions; without a
   prefix-cache reset across the transition, warmup-era prefixes may resolve to
   a DP checkpoint the worker never captured (see the transition caveat above).
5. **Manager proxy ordering** — the shared-state proxy must reach the TP workers
   at spawn time; `init_sparse_shared_state` is idempotent and called from both
   the coordinator and the executor so the proxy exists regardless of init
   order.
