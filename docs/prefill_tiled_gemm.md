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