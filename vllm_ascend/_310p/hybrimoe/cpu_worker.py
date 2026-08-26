#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#
"""CPU expert executor for HybriMoE.

Runs the experts that HSS assigned to the CPU, overlapped with the NPU
grouped matmul of the same MoE layer. Each worker thread owns a private
fp32 accumulation buffer so concurrent expert tasks never race; the partial
buffers are reduced once when the handle is collected.

Buffering scheme (per decode step, layers are processed sequentially):
  - input hidden states: D2H into one of two pinned buffers (alternating),
  - CPU output: reduced into a pinned host buffer, then H2D into one of two
    NPU buffers (alternating). A buffer is only reused after the event
    recorded on the consumer stream fires (mark_consumed).
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor

import torch
import torch.nn.functional as F
from vllm.logger import logger

from .utils import pin_memory_if_available

_MAX_WORKERS = 16
_IO_BUFFER_COUNT = 2


def _set_worker_threads(num_threads: int) -> None:
    torch.set_num_threads(num_threads)


class CPUResultHandle:
    """Result of one asynchronous CPU expert submission."""

    def __init__(
        self,
        futures: list[Future],
        worker_outputs: list[torch.Tensor],
        out_host: torch.Tensor,
        out_npu: torch.Tensor,
        buffer_index: int,
        num_tokens: int,
    ):
        self._futures = futures
        self._worker_outputs = worker_outputs
        self._out_host = out_host
        self.out_npu = out_npu
        self.buffer_index = buffer_index
        self.num_tokens = num_tokens

    def wait_and_h2d(self, copy_stream, pending_consume_event) -> object:
        """Block until CPU compute finishes, then enqueue the H2D copy.

        Returns the event recorded on `copy_stream` after the copy; the
        compute stream must wait on it before reading `out_npu`.
        """
        for future in self._futures:
            future.result()
        active = self._worker_outputs
        reduced = self._out_host[: self.num_tokens]
        reduced.copy_(active[0][: self.num_tokens])
        for partial in active[1:]:
            reduced.add_(partial[: self.num_tokens])
        if pending_consume_event is not None:
            copy_stream.wait_event(pending_consume_event)
        with torch.npu.stream(copy_stream):
            self.out_npu[: self.num_tokens].copy_(reduced, non_blocking=True)
            event = copy_stream.record_event()
        return event


class CPUExpertExecutor:
    """Thread-pool executor for CPU-resident experts."""

    def __init__(
        self,
        hidden_size: int,
        max_tokens: int,
        num_cpu_threads: int,
        host_store_bf16: bool = True,
        device: str = "npu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        cores = num_cpu_threads if num_cpu_threads > 0 else (os.cpu_count() or _MAX_WORKERS)
        self.num_workers = max(1, min(cores, _MAX_WORKERS))
        threads_per_worker = max(1, cores // self.num_workers)
        self.pool = ThreadPoolExecutor(
            max_workers=self.num_workers,
            initializer=_set_worker_threads,
            initargs=(threads_per_worker,),
        )
        self.hidden_size = hidden_size
        self.max_tokens = max_tokens
        self.host_store_bf16 = host_store_bf16
        self.dtype = dtype
        self.in_buffers = [
            pin_memory_if_available(torch.empty(max_tokens, hidden_size, dtype=dtype, device="cpu"))
            for _ in range(_IO_BUFFER_COUNT)
        ]
        self.out_host_buffers = [
            pin_memory_if_available(torch.zeros(max_tokens, hidden_size, dtype=torch.float32, device="cpu"))
            for _ in range(_IO_BUFFER_COUNT)
        ]
        self.out_npu_buffers = [
            torch.zeros(max_tokens, hidden_size, dtype=torch.float32, device=device) for _ in range(_IO_BUFFER_COUNT)
        ]
        # Per-worker fp32 accumulation shards (pageable; only the reduced
        # buffer needs to be pinned for H2D).
        self._worker_buffers = [
            [torch.zeros(max_tokens, hidden_size, dtype=torch.float32, device="cpu") for _ in range(_IO_BUFFER_COUNT)]
            for _ in range(self.num_workers)
        ]
        # Latest consumer event per output buffer; the copy stream waits on
        # it before overwriting the buffer.
        self._consume_events: list[object | None] = [None] * _IO_BUFFER_COUNT
        self._buffer_toggle = 0
        self._toggle_lock = threading.Lock()
        logger.info(
            "HybriMoE CPU expert executor: %d workers x %d threads, hidden=%d, max_tokens=%d",
            self.num_workers,
            threads_per_worker,
            hidden_size,
            max_tokens,
        )

    def next_buffer(self) -> int:
        with self._toggle_lock:
            index = self._buffer_toggle
            self._buffer_toggle = (self._buffer_toggle + 1) % _IO_BUFFER_COUNT
        return index

    def in_buffer(self, index: int) -> torch.Tensor:
        return self.in_buffers[index]

    def mark_consumed(self, index: int, event) -> None:
        self._consume_events[index] = event

    def consume_event(self, index: int):
        return self._consume_events[index]

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------
    def submit(
        self,
        layer,
        assignments: list[tuple[int, torch.Tensor, torch.Tensor]],
        buffer_index: int,
        num_tokens: int,
    ) -> CPUResultHandle:
        """Submit CPU expert tasks for one MoE layer.

        Args:
            layer: the HybriMoE FusedMoE layer (owns the host weight buffers).
            assignments: list of (expert_id, token_indices, routing_weights)
                where token_indices indexes rows of the input buffer.
            buffer_index: IO buffer slot for this layer forward.
            num_tokens: number of tokens in this forward.
        """
        in_buf = self.in_buffers[buffer_index]
        # Round-robin the experts over the workers.
        chunks: list[list[tuple[int, torch.Tensor, torch.Tensor]]] = [[] for _ in range(self.num_workers)]
        for i, task in enumerate(assignments):
            chunks[i % self.num_workers].append(task)
        active_workers = [w for w in range(self.num_workers) if chunks[w]]
        worker_outputs = []
        futures = []
        for w in active_workers:
            out_buf = self._worker_buffers[w][buffer_index]
            out_buf.zero_()
            worker_outputs.append(out_buf)
            futures.append(self.pool.submit(self._run_experts, layer, chunks[w], in_buf, out_buf))
        return CPUResultHandle(
            futures=futures,
            worker_outputs=worker_outputs,
            out_host=self.out_host_buffers[buffer_index],
            out_npu=self.out_npu_buffers[buffer_index],
            buffer_index=buffer_index,
            num_tokens=num_tokens,
        )

    def _run_experts(
        self,
        layer,
        tasks: list[tuple[int, torch.Tensor, torch.Tensor]],
        in_buf: torch.Tensor,
        out_buf: torch.Tensor,
    ) -> None:
        for expert_id, token_idx, weights in tasks:
            rows = in_buf[token_idx]
            if self.host_store_bf16:
                w13 = layer.host_w13_dequant[expert_id]
                w2 = layer.host_w2_dequant[expert_id]
            else:
                # On-the-fly dequant: (int8 * per-channel scale) -> executor dtype.
                w13 = (layer.host_w13_int8[expert_id].float() * layer.host_w13_scale[expert_id].unsqueeze(1)).to(
                    self.dtype
                )
                w2 = (layer.host_w2_int8[expert_id].float() * layer.host_w2_scale[expert_id].unsqueeze(1)).to(
                    self.dtype
                )
            intermediate = w13.shape[0] // 2
            gate = rows @ w13[:intermediate].t()
            up = rows @ w13[intermediate:].t()
            hidden = F.silu(gate) * up
            output = hidden @ w2.t()
            out_buf.index_add_(0, token_idx, output.float() * weights.unsqueeze(1))
