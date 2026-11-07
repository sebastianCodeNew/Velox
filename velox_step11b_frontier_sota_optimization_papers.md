# Frontier Literature & SOTA Systems/Kernel Survey Through 2026 (Step 11B)
## Algorithmic Foundations, Hardware Mechanisms, and Empirical Validation for Velox

**Document ID:** VELOX-PHASE2-STEP11B-SURVEY  
**Author:** Velox Core Systems Team (Phase II Milestone 1 — Step 11B)  
**Date:** November 7, 2026  
**Target Hardware:**  
1. **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU (GA107M, `sm_86`, 16 SMs, 2,048 CUDA cores, 64 Tensor cores, 192.0 GB/s GDDR6, 4,096 MiB VRAM)  
2. **CPU**: 12th Gen Intel Core i5-12500H (Alder Lake-H, 12 cores: 4 P-cores + 8 E-cores, 16 logical threads, AVX2 + FMA3, 18 MB Intel Smart Cache)  
**Target Workload:** `SmolLM-135M` (`smollm-135m.safetensors`, 134,515,008 parameters, 30 Transformer layers, $B=1$ autoregressive decode)  
**Validation Suite:** Standalone Empirical Challenge Suite in `tests/empirical_challenge_phase2/`  

---

## 1. Executive Overview & Hardware Grounding

### 1.1 The Baseline Paradox and the Memory-Bandwidth Roofline
Velox Phase I proved deterministic zero-dynamic-allocation serving, delivering bit-exact token parity across 140 consecutive runs on both the local NVIDIA RTX 3050 GPU and host Intel Core i5-12500H CPU. However, empirical telemetry revealed that the GPU operates at only **4.08% of its memory bus bandwidth ($7.84$ GB/s out of $192.0$ GB/s)**, achieving **$14.57\text{ tok/s}$** ($68.62\text{ ms/tok}$), while a single CPU core achieved **$18.99\text{ tok/s}$** ($52.69\text{ ms/tok}$).

As mathematically derived in Step 11 (`velox_step11_roofline_bottleneck_diagnostic.md`), single-token autoregressive decoding ($B=1$) has an operational arithmetic intensity of:
$$I_{\text{decode}} = \frac{268,959,744\text{ FLOPs}}{538,060,032\text{ bytes}} \approx \mathbf{0.50\text{ FLOPs/byte}}$$
Because the hardware machine balance (knee point) on the RTX 3050 Laptop GPU is:
$$I_{\text{knee}} = \frac{6,144\text{ GFLOPs/s}}{192.0\text{ GB/s}} = \mathbf{32.0\text{ FLOPs/byte}}$$
decode execution is **64x deep inside the memory-bandwidth bound regime**. Compute units spend over $95\%$ of their cycles waiting for weight streaming over the GDDR6 bus.

The theoretical physical rooflines on the RTX 3050 are:
- **FP32 ($538.06\text{ MB/token}$)**: $\mathbf{356.84\text{ tok/s}}$ ($2.802\text{ ms/token}$).
- **FP16 ($269.03\text{ MB/token}$)**: $\mathbf{713.68\text{ tok/s}}$ ($1.401\text{ ms/token}$).
- **INT4 `w4a16` ($67.26\text{ MB/token}$)**: $\mathbf{2,854.70\text{ tok/s}}$ ($0.350\text{ ms/token}$).

To saturate 75%–85%+ of these physical ceilings ($150\text{--}250+\text{ tok/s}$ in FP32, $400\text{--}800+\text{ tok/s}$ in FP16/INT4 on GPU; $75\text{--}120+\text{ tok/s}$ on CPU), this survey synthesizes the 6 core frontier systems and kernel optimization paradigms through 2026. Every technique is accompanied by its lineage, exact hardware mechanics, production implementation pattern, and empirical validation from `tests/empirical_challenge_phase2/`.

---

## 2. Technique 1: Warp-Specialized Coalesced GEMV & Fast Butterfly Shuffle Reductions

### 2.1 Academic & Systems Lineage
- Volkov, V. (2010). *"Better Performance at Lower Occupancy."* GPU Technology Conference (GTC).
- NVIDIA Corporation (2021–2023). *FasterTransformer: High-Performance Transformer Inference Engine.*
- NVIDIA Corporation (2023–2025). *TensorRT-LLM: A TensorRT Toolbox for Large Language Models.*
- Ye, Z. et al. (2024–2025). *"FlashInfer: Efficient and Customizable LLM Serving Kernels."* ArXiv:2404.14404.

### 2.2 Microarchitectural Mechanics on Ampere sm_86
On the NVIDIA Ampere architecture (`sm_86`), global GDDR6 memory is accessed through the L2 cache via 128-byte transactions, subdivided into four 32-byte sectors.
When a warp (32 threads) issues a memory read instruction:
1. If the 32 addresses fall within a contiguous, aligned 128-byte window, the hardware satisfies the request with **exactly 1 memory transaction (128 bytes)**, achieving $100\%$ coalescing efficiency.
2. In Velox v0.1 (`tinystories_smollm.cu`), thread `tid` was assigned to row $o = tid$:
   ```cuda
   // FLAWED: 1 thread per row
   size_t o = blockIdx.x * blockDim.x + threadIdx.x;
   const float* row = w + (o * in_dim);
   for (size_t k = 0; k < in_dim; ++k) acc += x[k] * row[k];
   ```
   At iteration $k=0$, lane $l$ loads `w[l * 576]`. The stride between adjacent warp lanes is $576 \times 4 = 2,304\text{ bytes}$. Every thread accesses a different 128-byte cache line. Servicing 128 bytes of requested data forces the memory controller to fetch $32 \times 128 = 4,096\text{ bytes}$, creating a catastrophic bottleneck:
   $$\eta_{\text{coalescing}} = \frac{128\text{ bytes}}{4,096\text{ bytes}} = \mathbf{3.125\%}$$

### 2.3 The Coalesced Warp-per-Row Architecture
To achieve $100\%$ coalescing, the mapping is inverted: **1 warp (32 threads) computes 1 output row**, cooperating along the input dimension $K$.

#### 128-Bit Vectorized `float4` Loads (`ld.global.nc.v4.f32`)
Threads issue 128-bit `float4` read-only cache loads:
```cuda
float4 w_q = *reinterpret_cast<const float4*>(&row[k]);
float4 x_q = *reinterpret_cast<const float4*>(&x[k]);
```
PTX emitted: `ld.global.nc.v4.f32 {$f0, $f1, $f2, $f3}, [$r0];`
When a 32-thread warp executes this instruction:
$$32 \text{ threads} \times 16\text{ bytes} = \mathbf{512\text{ contiguous bytes}} = \mathbf{4 \times 128\text{-byte cache lines}}$$
The memory pipeline operates at full hardware burst rate with zero sector waste.

#### 5-Step In-Register Butterfly Shuffle Reduction
After partial accumulation, intra-warp reduction is performed using the hardware register crossbar via `__shfl_down_sync`:
```cuda
#pragma unroll
for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
}
```
- Step 1 (`offset = 16`): Lanes $0\dots 15$ receive from lanes $16\dots 31$.
- Step 2 (`offset = 8`): Lanes $0\dots 7$ receive from lanes $8\dots 15$.
- Step 3 (`offset = 4`): Lanes $0\dots 3$ receive from lanes $4\dots 7$.
- Step 4 (`offset = 2`): Lanes $0\dots 1$ receive from lanes $2\dots 3$.
- Step 5 (`offset = 1`): Lane 0 receives from lane 1.
- Total latency: **Exactly 5 instructions ($\sim 5$ clock cycles)**.
- Shared memory consumed: **0 bytes**.
- Synchronization barrier stalls: **0 cycles**.

### 2.4 Empirical Validation Results (`tests/empirical_challenge_phase2/test1_warp_gemv_float4_shuffle.cu`)
Compiled via `nvcc.exe -O3 -arch=sm_86 -std=c++14`:
- **MLP Intermediate Projection ($M=1536, K=576$, $3.54\text{ MB}$)**:
  - Strided baseline: $0.1177\text{ ms}$ ($30.15\text{ GB/s}$, $15.70\%$ roofline).
  - Warp float4 shuffle: $\mathbf{0.0236\text{ ms}}$ ($\mathbf{150.49\text{ GB/s}}$, $\mathbf{78.38\%}$ roofline).
  - Real Speedup: **$4.99\times$**. Max error vs double oracle: $8.45 \times 10^{-8}$ (Tolerance: $10^{-4}$). **PASSED**.
- **Vocabulary LM Head Projection ($M=49152, K=576$, $113.25\text{ MB}$)**:
  - Strided baseline: $2.0112\text{ ms}$ ($56.41\text{ GB/s}$, $29.38\%$ roofline).
  - Warp float4 shuffle: $\mathbf{0.6227\text{ ms}}$ ($\mathbf{182.18\text{ GB/s}}$, $\mathbf{94.88\%}$ roofline).
  - Real Speedup: **$3.23\times$**. Max error vs double oracle: $8.45 \times 10^{-8}$. **PASSED**.

---

## 3. Technique 2: FlashDecoding & Split-K Parallel KV-Cache Attention

### 3.1 Academic & Systems Lineage
- Dao, T. et al. (2022). *"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness."* NeurIPS 2022.
- Dao, T. (2023). *"FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning."* ICLR 2024.
- Dao, T. et al. (2023). *"Flash-Decoding for long-context inference."* Tri Dao Blog / ArXiv.
- Ye, Z. et al. (2024–2025). *"FlashInfer: High-Throughput Serving for DeepSeek & Long-Context LLMs."*

### 3.2 Microarchitectural Mechanics
During single-token decoding ($B=1$), standard multi-head attention parallelizes only across query heads ($N_q = 9$).
On the RTX 3050 (16 SMs):
- 9 blocks occupy only 9 SMs; **7 SMs (43.75%) remain completely idle**.
- As context length $S$ grows from 64 to 2,048 tokens, attention FLOPs grow linearly ($2 \times 9 \times 64 \times S$), creating an increasingly severe sequential bottleneck on the 9 active SMs.

### 3.3 The Split-K FlashDecoding Formulation
Split-K FlashDecoding divides the temporal context dimension $S$ into $K_{\text{split}}$ independent chunks (e.g. $K_{\text{split}} = 4, 8, 16$):
$$\text{Total Thread Blocks} = N_q \times K_{\text{split}} = 9 \times 8 = \mathbf{72\text{ blocks}}$$
72 blocks completely saturate all 16 SMs ($4.5$ blocks per SM).

#### Mathematical Formulation of Online Partial Softmax & LogSumExp Reduction
Let split $s \in [0, K_{\text{split}})$ process tokens $t \in [\text{start}_s, \text{end}_s)$.
1. **Stage 1 (Per-Split Online Softmax)**:
   For each token $t$:
   $$dot_t = \frac{q \cdot k_t}{\sqrt{d}}$$
   $$m_s^{(\text{new})} = \max(m_s^{(\text{old})}, dot_t)$$
   $$\alpha = e^{m_s^{(\text{old})} - m_s^{(\text{new})}}, \quad \beta = e^{dot_t - m_s^{(\text{new})}}$$
   $$l_s^{(\text{new})} = l_s^{(\text{old})} \cdot \alpha + \beta$$
   $$acc_s^{(\text{new})} = acc_s^{(\text{old})} \cdot \alpha + \beta \cdot v_t$$
   Each split emits partial statistics: $(acc_s, m_s, l_s)$.
2. **Stage 2 (Exact LogSumExp Global Merge)**:
   Across the $K_{\text{split}}$ splits for query head $h$:
   $$m_{\text{global}} = \max_{s \in [0, K_{\text{split}})} m_s$$
   $$l_{\text{global}} = \sum_{s=0}^{K_{\text{split}}-1} l_s \cdot e^{m_s - m_{\text{global}}}$$
   $$O_h = \frac{1}{l_{\text{global}}} \sum_{s=0}^{K_{\text{split}}-1} acc_s \cdot e^{m_s - m_{\text{global}}}$$
   This reduction is **mathematically identical** to standard full-sequence softmax, with zero numerical truncation or loss of precision.

### 3.4 Empirical Validation Results (`tests/empirical_challenge_phase2/test2_split_k_flashdecoding_parity.cu`)
Compiled via `nvcc.exe -O3 -arch=sm_86 -std=c++14`:
- **$S = 64$ tokens ($K_{\text{split}} = 4$)**:
  - Single-Block: $37.54\,\mu\text{s}$. Split-K: $19.01\,\mu\text{s}$ (**$1.98\times$ speedup**). Max error vs double oracle: $7.76 \times 10^{-9}$. **PASSED**.
- **$S = 512$ tokens ($K_{\text{split}} = 8$)**:
  - Single-Block: $279.16\,\mu\text{s}$. Split-K: $48.56\,\mu\text{s}$ (**$5.75\times$ speedup**). Max error vs double oracle: $1.10 \times 10^{-8}$. **PASSED**.
- **$S = 1024$ tokens ($K_{\text{split}} = 8$)**:
  - Single-Block: $590.70\,\mu\text{s}$. Split-K: $93.53\,\mu\text{s}$ (**$6.32\times$ speedup**). Max error vs double oracle: $1.58 \times 10^{-8}$. **PASSED**.
- **$S = 2048$ tokens ($K_{\text{split}} = 16$)**:
  - Single-Block: $1,546.38\,\mu\text{s}$. Split-K: $\mathbf{116.63\,\mu\text{s}}$ (**$13.26\times$ speedup**). Max error vs double oracle: $1.85 \times 10^{-8}$. **PASSED**.

---

## 4. Technique 3: Persistent Cooperative Megakernels & CUDA Graphs

### 4.1 Academic & Systems Lineage
- Spector, B. et al. (2024). *"ThunderKittens: Simple, Fast CUDA Kernels for Machine Learning."* Hazy Research, Stanford University.
- Llama-Megakernel (2024–2025). *Single-Launch Monolithic Autoregressive Transformer Kernels.*
- NVIDIA Corporation (2020–2024). *CUDA Graphs Architecture & Best Practices Guide.*

### 4.2 Microarchitectural Mechanics
In standard LLM serving runtimes, every Transformer layer issues 2–4 separate kernel calls, requiring 60–100 host-side launches per token.
On Windows WDDM, each launch requires $8\text{--}12\,\mu\text{s}$ of driver overhead, totaling $0.5\text{--}0.8\text{ ms}$ of CPU queue delay per token step.

Two frontier architectural solutions eliminate this overhead:
1. **CUDA Graph Capture (`cudaGraphLaunch`)**:
   Because tensor allocations, base pointers, and shapes are 100% static in Velox, passing the sequence step and token index as device pointers (`d_step`, `d_token`) ensures kernel argument pointers remain strictly invariant across token steps.
   The entire 64-kernel sequence is captured once during initialization:
   ```cuda
   cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
   // ... enqueue 64 layer kernels ...
   cudaStreamEndCapture(stream, &graph);
   cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
   ```
   At runtime, executing `cudaGraphLaunch(graphExec, stream)` submits the entire token execution DAG to the GPU in a single sub-microsecond driver call.
2. **Persistent Cooperative Megakernels (`cudaLaunchCooperativeKernel`)**:
   Launches a persistent grid of blocks that stay resident on the SMs throughout the entire token step. Cross-layer synchronization is achieved via hardware grid barriers:
   ```cuda
   cooperative_groups::grid_group grid = cooperative_groups::this_grid();
   // ... Layer 0 ...
   grid.sync();
   // ... Layer 1 ...
   grid.sync();
   ```

### 4.3 Windows WDDM Caveat
On Windows 11 under WDDM display mode, cooperative grids cannot exceed the maximum number of blocks that can simultaneously co-reside on the SMs ($\le 16\text{--}32$ blocks for `sm_86`).
**CUDA Graph Capture (`cudaGraphLaunch`) provides identical sub-microsecond launch benefits ($0.0748\text{ ms}$ vs $0.7689\text{ ms}$, $10.28\times$ speedup) without any cooperative grid block count restrictions.**

---

## 5. Technique 4: Marlin & AWQ/BitBLAS INT4 (`w4a16`) Asynchronous Quantized Weight Kernels

### 5.1 Academic & Systems Lineage
- Frantar, E. et al. (2024). *"Marlin: Fast 4-bit Inference with FP16 Activations."* IST Austria / NeuralMagic. ArXiv:2408.11743.
- Lin, J. et al. (2024). *"AWQ: Activation-aware Weight Quantization for On-Device LLM Compression."* MLSys 2024.
- Microsoft Research (2024–2025). *BitBLAS: High-Performance Mixed-Precision GEMM/GEMV Library.*

### 5.2 Microarchitectural Mechanics & The LOP3 Trick
INT4 quantization packs eight 4-bit signed/unsigned weights into a single 32-bit `uint32_t` register ($0.5\text{ bytes/param}$).
Naive dequantization requires 8 arithmetic bit-shifts (`>>`), 8 bitwise ANDs (`& 0xF`), and 8 integer-to-float conversions, consuming 24 ALU instructions per 8 weights.

Marlin eliminates ALU dequantization overhead using the Ampere hardware instruction **`lop3.b32`** (3-input bitwise boolean lookup) with truth table **`0xEA`**:
1. In IEEE 754 half-precision float (FP16), `0x6400` represents the value $1024.0$ ($E = 25$, sign $= 0$, mantissa $= 0$).
2. The value represented by an FP16 number with exponent $E = 25$ and mantissa fraction $F$ is:
   $$\text{Value} = 2^{25 - 15} \times \left(1 + \frac{F}{1024}\right) = 1024.0 + F$$
3. Placing a 4-bit nibble $V \in [0, 15]$ into the lowest 4 bits of the mantissa yields exactly $1024.0 + V$.
4. The Ampere hardware instruction `lop3.b32 d, a, b, c, 0xEA;` computes:
   $$d = (a \land b) \lor c$$
   Setting $a = \text{packed word}$, $b = \text{mask } 0x000F000F$, and $c = \text{FP16 bias } 0x64006400$ isolates pairs of nibbles and injects the FP16 exponent `0x6400` in **a single hardware cycle**:
   ```cuda
   uint32_t h2 = ptx_lop3_0xEA(v, 0x000F000F, 0x64006400);
   ```
5. Subtracting $1024.0$ in SIMD half-precision (`__hsub2`) yields the exact floating-point weight in a single instruction:
   ```cuda
   half2 w = __hsub2(*reinterpret_cast<half2*>(&h2), sub_bias);
   ```

### 5.3 Empirical Validation Results (`tests/empirical_challenge_phase2/test3_marlin_lop3_dequant.cu`)
Compiled via `nvcc.exe -O3 -arch=sm_86 -std=c++14`:
- Dequantized: **1,048,576 weights** (131,072 packed words).
- Execution time: $0.0931\text{ ms}$ ($50.71\text{ GB/s}$).
- Numerical Parity vs Double Oracle: **Max Error = $0.00\text{e}+00$ (Bit-Exact Parity!)**. **PASSED**.

---

## 6. Technique 5: Ampere `sm_86` Tensor Core Prefill & FlashAttention-2/3 Tiling

### 6.1 Academic & Systems Lineage
- NVIDIA Corporation (2020). *NVIDIA Ampere GA102/GA107 GPU Architecture Whitepaper.*
- Dao, T. (2023–2024). *FlashAttention-2 and FlashAttention-3 Tiling Algorithms.*

### 6.2 Microarchitectural Mechanics
The GA107M GPU contains **64 3rd-Generation Tensor Cores** (4 per SM).
While autoregressive decode ($B=1, S=1$) is memory-bandwidth bound ($I = 0.50$), **prompt prefill ($S > 1$)** computes full matrix-matrix multiplications ($S \times D \times D$) where arithmetic intensity scales with sequence length:
$$I_{\text{prefill}} \approx \frac{S \times D}{S + D} \gg 32\text{ FLOPs/byte}$$
In the prefill regime, execution transitions from GEMV to Tensor Core GEMM via the Ampere PTX instruction:
`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
Peak prefill compute throughput jumps from $6.14\text{ TFLOPs}$ (FP32 CUDA cores) to **$28.5\text{ TFLOPs}$** (FP16 Tensor Cores), slashing Time-To-First-Token (TTFT) by up to $4.6\times$.

Shared memory tiling per block is constrained to $\le 24\text{ KiB}$ to ensure two thread blocks co-reside per SM within the 48 KiB shared memory envelope.

---

## 7. Technique 6: BLIS / llama.cpp AVX2+FMA3 Register-Blocked CPU GEMV & Zero-Alloc Threading

### 7.1 Academic & Systems Lineage
- Van de Geijn, R. et al. (2008–2016). *"BLIS: A Framework for Rapidly Instantiating BLAS Functionality."* ACM TOMS.
- Gerganov, G. et al. (2023–2025). *llama.cpp: Fast LLM Inference on Consumer Hardware.*

### 7.2 Microarchitectural Mechanics
On the Intel Core i5-12500H (Golden Cove P-cores):
- Each core has dual 256-bit FMA units on Execution Ports 0 and 1.
- In scalar GEMV, loading input activation $x[k]$ for every row causes $M \times K$ redundant L1 cache reads.
- **4-Row Register Blocking**:
  Holds 4 output accumulators in `YMM0..YMM3`. Loads an 8-float chunk of $x[k..k+8]$ once into `YMM4`, and issues 4 fused multiply-adds (`_mm256_fmadd_ps`):
  ```c
  __m256 vx = _mm256_loadu_ps(&x[k]);
  acc0 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(&r0[k]), acc0);
  acc1 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(&r1[k]), acc1);
  acc2 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(&r2[k]), acc2);
  acc3 = _mm256_fmadd_ps(vx, _mm256_loadu_ps(&r3[k]), acc3);
  ```
  This cuts L1 activation read traffic by **75%** and keeps both FMA execution ports continuously fed.

### 7.3 Empirical Validation Results (`tests/empirical_challenge_phase2/test4_avx2_fma3_register_blocking.c`)
Compiled via `cl.exe /nologo /O2 /arch:AVX2`:
- **Tiny Block ($M=256, K=576$)**:
  - Scalar ST: $0.0780\text{ ms}$ ($3.78\text{ GFLOP/s}$).
  - AVX2 4-row ST: $\mathbf{0.0074\text{ ms}}$ ($\mathbf{39.64\text{ GFLOP/s}}$, **$10.48\times$ speedup**).
  - Max error vs double oracle: $4.01 \times 10^{-8}$. **PASSED**.
- **MLP Projection ($M=1536, K=576$)**:
  - Scalar ST: $0.4793\text{ ms}$ ($3.69\text{ GFLOP/s}$).
  - AVX2 4-row ST: $\mathbf{0.0665\text{ ms}}$ ($\mathbf{26.61\text{ GFLOP/s}}$, **$7.21\times$ speedup**).
  - Max error vs double oracle: $4.01 \times 10^{-8}$. **PASSED**.
- **16-Thread OpenMP (`bench5_cpu_avx2_gemv.c`)**:
  - Scales to **$0.0264\text{ ms}$ ($67.10\text{ GFLOP/s}$)**, achieving **$17.61\times$ speedup** over scalar baseline.

---

## 8. Master Synthesis: Mapping Frontier Algorithms to Velox Phase II Steps

| Paper / Algorithmic Frontier | Target Microarchitectural Flaw | Velox Implementation Target | Target Hardware Roofline |
| :--- | :--- | :--- | :--- |
| **Warp-per-Row `float4` + Shuffle** | Flaw 2 (Strided uncoalesced) & Flaw 3 (Scalar loads) | `src/codegen.rs` (Step 13) | $>150\text{--}250+\text{ tok/s}$ FP32 ($>80\%$ bus) |
| **Split-K FlashDecoding** | Flaw 1 (Single-SM starvation at $S > 64$) | `src/codegen.rs` (Step 13) | $<120\,\mu\text{s}$ attention at $S=2048$ |
| **Persistent CUDA Graphs** | Flaw 5 (64-kernel driver queue delay) | `src/runners/velox_cuda_runner.cu` (Step 14) | Sub-$80\,\mu\text{s}$ token launch latency |
| **Marlin INT4 `w4a16` (LOP3 0xEA)** | Flaw 7 (FP32 bandwidth tax) | `src/codegen.rs`, `src/runtime.rs` (Step 15) | $400\text{--}800+\text{ tok/s}$ in $<250\text{ MiB}$ VRAM |
| **Ampere `sm_86` Tensor GEMM** | High TTFT prompt prefill | `src/codegen.rs` (Step 15) | $28.5\text{ TFLOPs}$ prefill compute |
| **4-Row AVX2+FMA3 & 16-Thread Pool** | Flaw 6 (Single-core CPU ceiling) | `src/codegen.rs`, `src/runtime.rs` (Step 16) | $75\text{--}120+\text{ tok/s}$ on host CPU |
