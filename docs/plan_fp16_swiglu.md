# Implementation Plan: FP16-Activation SwiGLU for Prefill

Status: plan only, no code changes.
Owner: Velox runtime
Target: RTX 3050 Laptop 4 GB, sm_86, K2-Horizon-4B Q4_K

## 1. Objective

Reduce prefill latency by widening the SwiGLU datapath from scalar FP32 to
half2 (2-wide) arithmetic, accepting FP16 activations but keeping FP32
accumulation.

Non-goals:

- Bit-exact equality with the FP32 path (impossible once activations are FP16)
- Tensor cores / `mma` (separate project, see section 10)
- Changes to the decode path, sampling, or KV layout

## 2. Measured baseline

Sequential runs, K2 Q4_K, RTX 3050, 36 layers:

| Prompt | Prefill | tok/s equivalent |
|---:|---:|---:|
| 32 | 237 ms | 135 |
| 256 | 1974 ms | 130 |
| 512 | ~4050 ms | 126 |
| 1024 | **8391 ms** | 122 |

llama.cpp on the same model/GPU: 539 ms at 1024 (~15.6x gap).

Per-stage GPU attribution at 64 tokens (one layer):

| Stage | Share |
|---|---:|
| gate_up | 39.2% |
| down | 29.4% |
| qkv | 17.4% |
| out_proj | 12.0% |
| attn | 1.1% |
| rope | 0.2% |

Roofline: 0.68 TB/s effective against a ~3.84 TB/s L2 peak = **18%**, so the
kernel is issue-bound, not bandwidth-bound. That is why widening the ALU is the
right lever.

A prototype accumulating in half2 measured **7545 ms** at 1024 (~10% faster)
but failed token parity. This plan keeps the speedup and fixes the precision by
accumulating in FP32.

## 3. Design

### 3.1 Precision

| Element | Current | New |
|---|---|---|
| Weight | 4-bit, dequant to FP32 | 4-bit, dequant to **FP16** (exact: 4-bit fits fp16) |
| Activation | FP32 | **FP16** (rounded once per layer) |
| Product | FP32 | FP16 (2-wide) |
| Accumulator | FP32 | **FP32** (converted per k-step) |

Resulting relative error is dominated by activation rounding:
`2^-11` per element, and for a 2560-term dot product the expected relative
error is about `sqrt(2560) * 2^-11 ≈ 2.5e-3`. That is the same order as
llama.cpp's FP16-activation path.

### 3.2 Nibble pairing (must match the FP32 order)

The FP32 kernel pairs nibble `i` with nibble `i+4` via the LOP3 mask
`0x000F000F`:

```
f0 = (n0, n4)  multiplies (x[k],   x[k+1])
f1 = (n1, n5)  multiplies (x[k+2], x[k+3])
f2 = (n2, n6)  multiplies (x[k+4], x[k+5])
f3 = (n3, n7)  multiplies (x[k+6], x[k+7])
```

The half2 kernel must use the same pairing, i.e. `wg[i] = half2(n_i, n_{i+4})`
and `xa[i] = half2(x[k+2i], x[k+2i+1])`. The prototype got this right; it is
called out because a mismatch here produces plausible but wrong output.

### 3.3 Scratch

Half2 activations need `BATCH * hidden_dim / 2 * 2 B` = 40 KB at B=16.

Location: `VELOX_ATTN_SPLIT_BASE` (arena tail, `VELOX_ACTIVATION_ARENA_BYTES -
133120`). This region is owned by the attention combine, which runs *before*
SwiGLU in the same layer, so reuse within a layer is safe.

Existing regions are untouched: prefill (8..92 MiB), verify (~96 MiB), split
(125 MiB tail).

### 3.4 Kernels

New, both behind one flag:

1. `velox_cuda_convert_rows_h2` — FP32 row → half2, one block per token,
   128 threads, 2 elements per thread.
2. `velox_cuda_swiglu_gate_up_half2_q4<BATCH>` — half2 multiply,
   `__half22float2` accumulate into an FP32 `acc_g/acc_u`, FP32 scale apply,
   then the existing warp-shuffle reduction.
3. `velox_cuda_swiglu_down_half2_q4<BATCH>` — same treatment for the down
   projection (phase 2).

The Q4_K dequant stays `scale * sum(nibble*x) + min * sum(x)` with the scale
applied **per 64-element group inside the k loop** — applying the last group's
scale to the whole row was a real bug found during the int8 attempt.

## 4. Acceptance criteria

A change is kept only if **all** hold:

1. **Build**: `build_runner.ps1` exits 0, no register spills for the new kernels.
2. **Oracle**: `--no-cuda-graph --dump-layer0` still prints
   `sum=0.380322` (the FP32 path must be untouched).
3. **Token divergence**: on 5 real prompts (factual, code, open-ended,
   multi-turn, long-context), the greedy token sequences must diverge in
   **< 2%** of positions versus the FP32 path. Exact equality is not required
   and not expected.
4. **Coherence**: outputs remain grammatical and on-topic (manual read).
5. **Speed**: prefill at 1024 must be **< 8391 ms**; target ≤ 7600 ms.
6. **Regression**: `cargo test` (291 tests) passes; decode tok/s unchanged
   within 2%.

If (3) or (5) fails, revert.

## 5. Work sequence

Each step is independently compilable and testable.

| Step | Change | Verification |
|---:|---|---|
| 1 | Add `--swiglu-half` flag (no behaviour change) | oracle unchanged |
| 2 | Add `convert_rows_h2` + call it, discard result | build, no drift |
| 3 | Add half2 gate_up with FP32 accumulate, flag-gated | oracle unchanged (flag off) |
| 4 | Enable flag, compare fingerprint | fingerprint close to 0.380322 |
| 5 | Token divergence on 5 prompts | record divergence rate |
| 6 | Prefill 32/256/512/1024 | record, compare to 8391 ms |
| 7 | Keep or revert per section 4 | commit or `git checkout` |
| 8 | Phase 2: down kernel | repeat 3–7 |
| 9 | CLI: expose `--swiglu-half` for `--run` | help text, parse test |
| 10 | Update `docs/prefill_tiled_gemm.md` with results | doc commit |

## 6. Rollback

Single flag, default off. Revert is `git revert` of one commit or
`git checkout -- src/runners/velox_cuda_runner_k2.cu`. No data or weight
format changes, so rollback is always safe.

## 7. Risks

| Risk | Mitigation |
|---|---|
| Nibble-pairing mismatch | assert against the FP32 layout in a comment and a unit test |
| FP16 accumulation drift | accumulate in FP32 (this plan) |
| Scratch collision with attention | reuse only after attention; verify with the oracle |
| Register spills at BATCH=16 | check `ptxas -v`; fall back to 4 rows per block |
| Divergence above 2% | revert; do not ship |

## 8. Out of scope (future phases)

- **Phase 3**: tensor-core `mma` with FP16 fragments (larger win, larger risk)
- **Phase 4**: continuous batching (#5)
- **Phase 5**: fix the generated decode DAG offsets (`src/ir.rs:496`)

## 9. Open question

The 2% divergence threshold in section 4 is a policy choice. If the user wants
stricter quality, tighten to 0.5% and accept a smaller or no speedup.

## 10. Result: FP16 SwiGLU is slower (measured)

Both variants were implemented and measured at 1024 tokens on the RTX 3050:

| Variant | Layer-0 fingerprint | Prefill 1024 |
|---|---|---|
| FP32 reference | 0.380322 | 8532 ms |
| half2, FP32 accumulate | 0.380470 (correct) | 9788 ms |
| half2, FP16 accumulate, scalar accumulator | 0.379834 (correct) | 9708 ms |

An intermediate FP16-accumulate version reported 7545 ms, but that was an
artifact of a bug that dropped the `min * sum(x)` term and therefore did less
work. With the math corrected, FP16 is **slower**, not faster.

Two real bugs were found and fixed on the way, both worth recording:

1. The conversion kernel covered only 256 of 2560 elements (128 threads x 2)
   and needed a grid-stride loop.
2. A half2 accumulator adds `min * sum(x)` twice, once per component. The
   accumulator must be a scalar half.

Conclusion: the ALU is not the binding constraint for these kernels, so
widening the datapath does not help. The remaining options are tensor cores
(phase 3) or accepting the current prefill performance. This plan is closed
without a landed change.