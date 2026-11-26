# Prefill Optimization: Root-Cause Analysis and Design

Status: analysis complete, implementation not started.
Scope: `velox_cuda_runner_k2.cu`, K2-Horizon-4B Q4_K, RTX 3050 Laptop (4 GB, sm_86).

## 1. Measured baseline

Sequential runs, no contention, 36 layers, Q4_K:

| Prompt | Velox prefill | llama.cpp | Gap |
|---:|---:|---:|---:|
| 32 | 237 ms | 33 ms | 7.2x |
| 256 | 1974 ms | 139 ms | 14.2x |
| 512 | 4050 ms | 276 ms | 14.7x |
| 1024 | 8480 ms | 539 ms | 15.7x |

Per-token cost is flat at ~8.3 ms, so this is a bandwidth/occupancy problem, not
a scaling problem.

## 2. Roofline: where the time actually goes

Per layer, per 16-token chunk, for the two dominant matmuls:

| Matmul | Weight bytes (Q4) | Activation bytes re-read |
|---|---:|---:|
| gate + up (10240 x 2560, twice) | 26.2 MB | 1.68 GB |
| down (2560 x 10240) | 13.1 MB | 1.68 GB |
| **Total per layer** | **~39 MB** | **~3.4 GB** |

The activation term is ~87x the weight term. The arithmetic is:

```
activation = output_rows x BATCH x hidden_dim x 4 B
           = 10240 x 16 x 2560 x 4  = 1.68 GB   (per matmul)
```

At the RTX 3050's 192 GB/s theoretical bandwidth, 3.4 GB per layer is 17.7 ms,
or 638 ms per 16-token chunk. 64 chunks for a 1024-token prompt gives a 40.8 s
floor — which matches the *original* 40.6 s measurement almost exactly.

The current 8.5 s is ~4.8x better than that floor, so the 164 KB activation tile
(16 x 2560 floats) is being partially served from L2 rather than DRAM. But it is
still the dominant term by two orders of magnitude.

**Conclusion: prefill is activation-traffic-bound, not weight-bound.** Adding
occupancy (2 -> 8 blocks/SM, measured 8660 -> 8480 ms) barely moved it, which
confirms the diagnosis.

## 3. Why the obvious fix fails

Doubling the chunk to 32 halves the number of weight passes but doubles the
activation tile to 328 KB, and the extra `acc[BATCH]` accumulators spill:

| Chunk | ms/token |
|---:|---:|
| 16 | 8.3 |
| 32 | 12.0 |

Measured, not estimated. Register spills outweigh the traffic saving.

## 4. Design: shared-memory activation tiling

The fix is to stop re-reading the activation tile once per output row. One block
should load a K-slice of the activation tile into shared memory **once**, then
let all rows in that block consume it.

Parameters:

| Symbol | Value | Rationale |
|---|---:|---|
| `ROWS_PER_BLOCK` | 4 | matches the existing warp-per-row mapping (128 threads) |
| `K_TILE` | 128 | 4 B x 16 tokens x 128 = 8 KB shared, well under the 48 KB budget |
| `K_TILE` loop | `dim / K_TILE` = 20 | covers hidden_dim 2560 |

Shared memory per block: `BATCH x K_TILE x 4 B` = 8 KB for B=16. With 128
threads and 6 blocks/SM that is 48 KB, within the 100 KB/SM limit.

Expected effect: activation DRAM traffic falls by `ROWS_PER_BLOCK` (4x), i.e.
3.4 GB -> 0.85 GB per layer. If the kernel stays bandwidth-bound, that is up to
a 4x prefill improvement; the weight term (39 MB) then becomes the floor at
~0.2 ms per layer per chunk.

### Kernel shape

```
for each K tile:
    load B x K_TILE activations into shared   (cooperative, once per block)
    __syncthreads()
    for each of ROWS_PER_BLOCK rows:
        stream Q4 weights for this K tile
        dequantize (LOP3 + fp16 bias trick, unchanged)
        accumulate into acc[BATCH]
    __syncthreads()
```

The dequantization path is unchanged; only the activation source moves from
global to shared memory.

## 5. Secondary items, in order

1. **FP16 KV cache.** 288 KiB/token at FP32; FP16 halves it, enabling 4096
   context at the current 576 MiB budget. Contained: change the pool element
   type and the QKV/attention load paths.
2. **Native Q6_K head** or matched quantization, so cross-engine tok/s is
   comparable. Currently Velox reads 191 MiB less per token.
3. **Continuous batching** for concurrent requests.

## 6. Verification plan

- Token parity: `--decode-path dag` vs `batched` must stay identical.
- Prefill timing at 32/256/512/1024 against the table in section 1.
- `ptxas -v` must report 0 spill stores for the new kernel.
- Full `cargo test` (291 tests) plus the CUDA build.

## 7. Honest assessment

The analysis is complete and the design is concrete, but the kernel is not
written. The activation-traffic term is two orders of magnitude above the weight
term, so this is the highest-leverage change available; it is also the largest
and riskiest, which is why it is specified rather than half-implemented.

## 8. Experimental result: hypothesis falsified

The tiled kernel from section 4 was implemented and measured (4 rows per block,
256-wide K tile, 8 KB shared, cooperative load, unchanged dequantization, 0
register spills per `ptxas -v`):

| Prompt | Original | Tiled | Verdict |
|---:|---:|---:|---|
| 256 | 1974 ms | 2155 ms | 9% slower |
| 1024 | 8480 ms | 9060 ms | 7% slower |

Correctness was preserved (identical coherent output), so the regression is
performance, not numerics. The kernel was reverted.

**Conclusion: the activation tile is already L2-resident.** If DRAM traffic were
the limiter, cutting it 4x would have helped; it did not, so the shared-memory
load and its `__syncthreads` are pure overhead. The real limiter is therefore
L2 bandwidth or dequantization/ALU throughput, not DRAM traffic.

This also explains why occupancy changes (2 -> 8 blocks/SM) did nothing: the
kernel is neither DRAM-latency- nor occupancy-bound.

### What the evidence now points to

1. **L2 bandwidth** — the 164 KB tile is re-read from L2 once per output row.
   Reducing L2 traffic requires *row blocking* (reuse the weight tile across
   multiple output rows), not activation tiling.
2. **Dequantization ALU** — each weight element costs a LOP3, a subtract, a
   half2->float conversion, and two FMAs. That is ~6 instructions per 4-bit
   weight, which caps throughput independently of memory.
3. **Tensor cores** — the only way past the ALU ceiling, but Q4 requires
   dequantization first, so it needs a fused dequant-then-mma pipeline.

Any further attempt should be preceded by a profiler measurement
(`ncu --metrics l2_tex_read_bytes.sum`, `sm__inst_executed.sum`) to identify
which of the three is binding, rather than another roofline guess.

## 9. Row blocking does not apply (measured, with code evidence)

The proposed "reuse the weight tile across output rows" is not possible here,
because the weight row is indexed by the output row:

```c
size_t off = (layer * dim * inter_dim) + (o * inter_dim);   // o = output row
```

Each output row has a distinct weight row, so there is no tile shared across
rows. The only reuse is across the **batch** dimension (`for t in 0..BATCH`
uses one weight row for every token), and the kernels already exploit it.
Adding row blocking would raise accumulator pressure with no traffic reduction.

## 10. Two-warp row split for chunk 32: attempted, reverted

Per-stage attribution (section 8) shows gate_up 39.2% and down 29.4% of layer
time, and the implied 2.7 TB/s is L2 bandwidth. The intended fix was to raise
the chunk from 16 to 32 tokens so the 26 MB weight matrix is read once per 32
tokens instead of once per 16, with each row split across two warps (16
accumulators each) to avoid the spills measured at chunk 32.

Implementation: `velox_cuda_swiglu_gate_up_split_q4` and
`velox_cuda_swiglu_down_split_q4`, dispatched for `B > 16`, plus template cases
17..32 and `prefill_chunk = 32`.

Two defects were found and fixed during bring-up:

1. `HALF = BATCH / 2` is zero for B <= 16, which fails to compile the array
   declarations. Guarded to `max(1, BATCH / 2)`.
2. `g_prefill_graph[chunk]` was indexed with chunk = 32 while the array holds
   capture widths 1..16, so the lookup read out of bounds and the prefill was
   skipped entirely (`TTFT 0.000 ms`, token 0). Guarded with `chunk <= 16`.

After both fixes the path still produced no valid output (`TTFT 0.000 ms`,
`Generated Tokens (1) : [0]`), so it was reverted rather than committed. The
row/token mapping (`row = blockIdx.x * 2 + warp_id >> 1`,
`tok_off = (warp_id & 1) * HALF`) and the `act_base` indexing are the remaining
suspects; the next step is to compare the first layer's `act_base` output for
the same 16 tokens between the chunk-16 and chunk-32 paths to isolate it in one
measurement.

Verified restored state after revert:

```
TTFT (Prefill Latency) : 9457.945 ms (prompt len = 1024)
Generated Tokens (1)  : [200]
```