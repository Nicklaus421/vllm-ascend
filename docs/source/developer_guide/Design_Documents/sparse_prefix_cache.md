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
| `hist_exp_decay` | DP-optimal on the empirical overlap distribution with exponential reweighting for drift |

vLLM Ascend implements the DP-optimal `hist_exp_decay` strategy (the paper's
core contribution, Corollary 1). Simpler strategies are retained as the
fallback used before the first DP solve.

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
└── coordinator.py    # hit -> (hit_pos, checkpoint_pos) -> trigger recompute
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
- `VLLM_ASCEND_SPARSE_PREFIX_CACHE_STRATEGY` (`hist_exp_decay`)
- `VLLM_ASCEND_SPARSE_PREFIX_CACHE_BUDGET_GB`
- `VLLM_ASCEND_SPARSE_PREFIX_CACHE_BIN_SIZE`
- `VLLM_ASCEND_SPARSE_PREFIX_CACHE_GAMMA`
- `VLLM_ASCEND_SPARSE_PREFIX_CACHE_REPLAN_INTERVAL`
- `VLLM_ASCEND_SPARSE_PREFIX_CACHE_LAPLACE_ALPHA`
- `VLLM_ASCEND_SPARSE_PREFIX_CACHE_KERNEL_BLOCK_SIZE`

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

The pure orchestration is in `vllm_ascend/core/sparse_prefix_cache/`; the runner
glue is `runner.py` (`SparsePrefixCacheHook`), which reuses vLLM's
`MambaStateCopyFunc` machinery to move state between live blocks and the store.
The three call sites, against the vllm-ascend v1 runner:

1. **Capture after prefill** — in `vllm_ascend/worker/model_runner_v1.py`
   (`_update_states_after_model_execute`, entry point `_maybe_capture_sparse_checkpoints`),
   after the recurrent state is committed to its block during the forward, call
   `hook.capture(prefix_key, seq_len, kv_caches_by_layer, block_ids)`.
   `block_ids` is the request's mamba block table; the state after token `pos`
   lives in `block_ids[pos // block_size - 1]`. Multi-position capture requires
   `mamba_cache_mode='all'` (block-indexed recurrent state).

2. **Hit resolution** — in `vllm_ascend/patch/platform/patch_kv_cache_coordinator.py`
   (`AscendHybridKVCacheCoordinator.find_longest_cache_hit`), the full-attention
   group already yields the dense overlap depth `P`. For the Mamba group,
   `_resolve_sparse_mamba_hit(block_hashes, P)` consults
   `hook.resolve(prefix_key, P)` instead of the dense block pool; the resulting
   `checkpoint_position` `C` is the mamba hit length, so the hybrid
   `num_computed_tokens` is rewound to `C`. The resolved `SparseHit` is recorded
   as a *pending hit* keyed by `prefix_key` (the block-hash chain) for the
   worker to consume.

3. **Restore + recompute** — in `vllm_ascend/worker/model_runner_v1.py`
   (`execute_model`, `_maybe_restore_sparse_checkpoints`, before the forward), a
   pending hit is popped by `prefix_key` and `hook.restore(...)` copies the
   checkpoint into the sequence's state block (`dest_block_idx =
   C // block_size - 1`). `vllm_ascend/ops/gdn.py`
   (`AscendGatedDeltaNetAttention._forward_core`) then reads the restored
   `initial_state = ssm_state[prefill_state_indices]` and runs
   `chunk_gated_delta_rule(initial_state=...)` over `[C, P)` exactly as the
   existing resume path does.

`vllm_ascend/ops/gdn_attn_builder.py`
(`AscendGDNAttentionMetadataBuilder.build`) is where `common_prefix_len`,
`num_computed_tokens` and `prefill_state_indices` are materialized into
`GDNAttentionMetadata`; the rewind from `P` to `C` reaches it through the
coordinator's reduced `hit_length` (`num_computed_tokens = C`), so no change is
needed there — the pending-hit restore above is the only tensor-side hook.

The coordinator and runner share the hook through a process-wide singleton
(`set_sparse_prefix_cache_hook`/`get_sparse_prefix_cache_hook`), which assumes
the default single-process engine. A multiprocessing engine would need an IPC
handoff instead.

## Risks

1. **Coordinator integration** is the hardest part: translating "hit at `P`,
   recurrent state valid to `C`" into `num_computed_tokens = C` plus a recompute
   segment.
2. **Qwen3.8 architecture** is not yet in-tree; verify it matches the Qwen3.5
   GDN layer structure (or port a model patch) before Phase 1.
3. **Recompute cost on NPU** — the extra `[C, P)` prefill on the 910B4 must be
   weighed against the memory saved. Tune the checkpoint interval to maximize
   net benefit.
