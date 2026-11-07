# VELOX LANGUAGE SYNTAX & SYSTEM API SPECIFICATION
## Language Specification & Standard Library Manual
**Document Version**: `3.0.0-PROD`  
**Specification Identifier**: `VELOX-SPEC-2026-STEP3-SYNTAX-001`  
**Standard**: ISO/IEC 14977 EBNF Compliant  
**Compiler Frontend**: `veloxc` (Python 3.12+ Reference Implementation)  
**Target Architecture**: High-Throughput Heterogeneous LLM Serving Runtimes  
**Date**: 2026-11-07  

---

## Executive Overview

Velox is a domain-specific systems programming language and optimizing compiler designed from first principles for Large Language Model (LLM) serving and inference. Contemporary serving stacks (e.g., vLLM, TensorRT-LLM, HuggingFace TGI, llama.cpp, Mistral.rs, Modular MAX) rely on heterogeneous language glue—pairing Python runtimes with C++/CUDA kernels via FFI boundaries. As documented in `velox_step1_serving_flaws_diagnostic.md`, this multi-language architecture induces catastrophic systemic failures:
1. **Garbage Collection Pauses & GIL Contention**: CPython garbage collection and Global Interpreter Lock contention cause high p99 latency spikes ($\ge 80\,\mu\text{s}$) during high-concurrency request scheduling.
2. **Dynamic Memory Fragmentation**: Dynamic PyTorch caching allocators (`CUDACachingAllocator`) fragment VRAM during long-running continuous batching, forcing early eviction and out-of-memory (OOM) crashes.
3. **Absence of Compile-Time Memory and Shape Proofs**: Runtime shape mismatches and unverified VRAM budgets cause unexpected server panics in production.
4. **Borrow Checker Impedance**: Traditional systems languages (e.g., Rust) struggle with alias buffers in dynamic execution graphs and require unsafe pointer escapes for KV-cache pooling.
5. **Driver Launch Latencies**: FFI and host dispatch overheads prevent sub-80µs request cycle times.

Velox replaces this fragile multi-layer stack with a unified, statically typed, mathematically verified language environment. It features:
- **Three Non-Negotiable Runtime Invariants**: Guaranteed zero dynamic heap allocation during serving ($O(1)$ Velox Activation Arena), static compile-time VRAM budget feasibility proofs, and lock-free cooperative fiber event loop scheduling.
- **Physical Memory Tier System**: Explicit type system qualifiers separating Accelerator HBM (`@vram_immutable`, `@paged_vram`), On-Chip SRAM (`@bump_sram`), and Non-pageable Host Memory (`@host_pinned`).
- **First-Class Native KV Caches**: Native language types for `PagedKVCache`, `VirtualContiguousKV`, `LatentKVCache`, `StreamingSinkKVCache`, and `SpeculativeTreeBlock`.
- **Phase-Disjoint Execution Scopes**: Formal execution scopes (`prefill`, `decode`, `chunked_prefill`, `speculative_verify`) that isolate compute-bound GEMM pipelines from memory-bandwidth-bound GEMV loops.
- **Deterministic Diagnostic Engine**: Error codes `E001` through `E025` with source pointers, ASCII carets, and actionable remediations.

---

## Table of Contents

1. [Language Vision & Systems Architecture](#1-language-vision--systems-architecture)
   - 1.1 The Velox Mission: Zero-Overhead LLM Serving
   - 1.2 Three Non-Negotiable Runtime Invariants
   - 1.3 Compilation Model (Source $\to$ AST $\to$ Velox Static Memory Planner $\to$ C99/CUDA $\to$ Executable)
2. [Lexical Conventions & Token Grammar](#2-lexical-conventions--token-grammar)
   - 2.1 Character Set, Whitespace & Source Encoding
   - 2.2 Comments
   - 2.3 Identifiers & Literals
   - 2.4 Reserved Keywords & Memory Tier Qualifiers
3. [Hardware Target Specification (`target hardware`)](#3-hardware-target-specification-target-hardware)
   - 3.1 Syntax & Properties (`device`, `vram_capacity`, `compute_capability`, `safety_factor`)
   - 3.2 Memory Multipliers (`KiB`, `MiB`, `GiB`, `TiB`)
   - 3.3 Compile-Time Static VRAM Feasibility Formulation
4. [Type System & Physical Memory Hierarchy](#4-type-system--physical-memory-hierarchy)
   - 4.1 Scalar & Numeric Types
   - 4.2 Const-Generic Tensor Types (`Tensor<[Dim...], Dtype, BlockScaleSpec>`)
   - 4.3 The Four Disjoint Physical Memory Tiers
   - 4.4 Inter-Tier Asynchronous Data Movement Primitives
5. [Native Key-Value Cache Types](#5-native-key-value-cache-types)
   - 5.1 `PagedKVCache` (Block Paging & 2MB Huge-Page Alignment)
   - 5.2 `VirtualContiguousKV` (CUDA Driver VMM)
   - 5.3 `LatentKVCache` (DeepSeek MLA Latent Compression)
   - 5.4 `StreamingSinkKVCache` (Modulo Ring Arithmetic & Infinite Context)
   - 5.5 `SpeculativeTreeBlock` (2D Non-Causal Masking & Path Compaction)
6. [Model Architecture & Weight Declarations](#6-model-architecture--weight-declarations)
   - 6.1 The `model` Declaration & Const-Generic Parameterization
   - 6.2 Zero-Copy Safetensors Weight Loading (`safetensors(...)`)
   - 6.3 Block Quantization Bindings (INT4 Marlin & FP8 Microscaling)
   - 6.4 Offline Matrix Absorption Declarations (`absorb_mla_q`, `absorb_mla_v`)
7. [Execution Pipeline Scopes](#7-execution-pipeline-scopes)
   - 7.1 `scope prefill`: Compute-Bound GEMM Pipeline
   - 7.2 `scope decode`: Bandwidth-Bound GEMV Pipeline & Sub-80µs Dispatch
   - 7.3 `scope chunked_prefill`: Interleaved Chunked Serving (Sarathi-Serve)
   - 7.4 `scope speculative_verify`: Parallel Speculative Tree Verification
8. [Expressions, Statements & Operator Semantics](#8-expressions-statements--operator-semantics)
   - 8.1 14-Level Operator Precedence Hierarchy
   - 8.2 Matrix Multiplication Contraction Rule (`@`)
   - 8.3 Statements & Control Flow (`let`, `mut`, `if/else`, `for in ..`, `while`, `yield`, `return`)
   - 8.4 Linear Ownership, Mutability & Disjoint Borrowing Axioms
9. [Standard Library Builtin Functions & Kernels Registry](#9-standard-library-builtin-functions--kernels-registry)
   - 9.1 Embedding & Normalization (`embed`, `rmsnorm`, `rotary_embedding`)
   - 9.2 High-Performance Attention Kernels (`flash_prefill`, `flash_decode`, `flash_chunked`, `mla_decode_absorbed`, `flash_tree_attention`)
   - 9.3 Feedforward, Activation & MoE (`silu`, `gelu`, `swiglu`, `moe_gate_topk`, `moe_dispatch_topk`)
   - 9.4 Sampling Kernels (`sample_argmax`, `sample_temperature_topp`, `async_sample`)
10. [Diagnostic Error Catalog (E001 – E025)](#10-diagnostic-error-catalog-e001--e025)
    - 10.1 Diagnostic Architecture & Deterministic Format
    - 10.2 Complete Error Code Reference Table
    - 10.3 Concrete Diagnostic Error Walkthroughs
11. [Appendix: Reference Model Implementations](#11-appendix-reference-model-implementations)
    - 11.1 TinyStories / SmolLM Reference Script (`examples/tinystories_smollm.vlx`)
    - 11.2 Meta Llama-3-8B Reference Script (`examples/llama3_8b.vlx`)
    - 11.3 Alibaba Qwen2.5-7B Reference Script (`examples/qwen2_5_7b.vlx`)
    - 11.4 DeepSeek-V3 MLA Reference Script (`examples/deepseek_v3_mla.vlx`)
    - 11.5 Nanbeige-4.3-33B Reference Script (`examples/nanbeige_33b.vlx`)

---

# 1. Language Vision & Systems Architecture

## 1.1 The Velox Mission: Zero-Overhead LLM Serving

Velox is designed to eliminate the runtime overheads and architectural fragility inherent in gluing high-level interpreted languages to low-level GPU compute APIs. In conventional serving systems, the serving runtime must continually juggle Python interpreter threads, dynamic graph executors, garbage collection sweeps, CUDA context switches, and variable-length tensor memory allocations.

Velox treats LLM serving not as generic dynamic scientific computing, but as a **statically bounded real-time systems problem**. An LLM inference server performs a well-defined cycle of computations: prompt encoding (prefill) and autoregressive token generation (decode). By elevating hardware resource envelopes, KV cache layouts, memory tiers, and execution phases directly into the language type system, Velox guarantees that:
- Every byte of GPU memory is accounted for before code compilation completes.
- No dynamic memory allocation (`malloc`, `cudaMalloc`) ever occurs during active request serving.
- Token generation dispatch loops execute without host-side runtime locks, serialization, or FFI penalties.

## 1.2 Three Non-Negotiable Runtime Invariants

Every program compiled with `veloxc` must satisfy three core mathematical invariants. Violation of any invariant causes deterministic compiler rejection prior to code generation.

### Invariant 1: Zero Dynamic Heap Allocation During Inference ($O(1)$ Velox Activation Arena)
$$\forall t \ge t_{\text{serving\_start}}, \quad \Delta \text{HeapAllocations}(t) = 0$$
During request processing inside any execution scope (`prefill`, `decode`, `chunked_prefill`, `speculative_verify`), invocations of general-purpose dynamic memory allocators (`malloc`, `free`, `cudaMalloc`, `cudaFree`, `new`, `delete`) are strictly prohibited. 
All temporary activation tensors reside in statically pre-allocated memory slices managed by the **Velox Static Memory Planner**. The Velox planner analyzes the operator lifetime intervals across execution scopes and maps every intermediate activation to fixed, reuseable base offsets within a contiguous, statically allocated scratchpad buffer. Memory allocation during serving is strictly $O(1)$ and non-fragmenting. Any attempt to invoke dynamic allocation triggers compiler diagnostic `E001`.

### Invariant 2: Static VRAM Feasibility Proof
$$\mathbf{VRAM}_{\text{total}} = \mathbf{S}_{\text{weights}} + \mathbf{S}_{\text{kv\_cache}}(B_{\max}, S_{\max}) + \mathbf{S}_{\text{velox\_act}} + \mathbf{S}_{\text{system\_reserve}} \le \alpha \cdot \mathbf{VRAM}_{\text{budget}}$$
Where:
- $\mathbf{S}_{\text{weights}}$ is the exact byte footprint of all `@vram_immutable` model parameter weights loaded into memory.
- $\mathbf{S}_{\text{kv\_cache}}(B_{\max}, S_{\max})$ is the maximum physical memory occupied by the declared KV-cache pool at maximum batch size $B_{\max}$ and sequence length $S_{\max}$.
- $\mathbf{S}_{\text{velox\_act}}$ is the maximum clique weight of the activation tensor lifetime interval graph computed by the Velox compiler pass.
- $\mathbf{S}_{\text{system\_reserve}}$ is the non-negotiable driver context and page table overhead (default 512 MiB to 1024 MiB).
- $\alpha \in [0.50, 1.00]$ is the user-configured hardware safety factor (default 0.95).
- $\mathbf{VRAM}_{\text{budget}}$ is the physical memory capacity declared in the `target hardware` block.

If this inequality is not satisfied at compile time, compilation fails with diagnostic `E002`. Out-of-memory (OOM) errors during inference are rendered mathematically impossible.

### Invariant 3: Lock-Free Cooperative Fiber Event Loop with Sub-80µs Host Dispatch
$$\mathcal{T}_{\text{host\_dispatch}} \le 25\,\mu\text{s} \ll 80\,\mu\text{s}$$
The host-side execution runtime is compiled into a single-binary, lock-free C99 cooperative fiber event loop. Host-to-device kernel launches avoid OS thread synchronizations, mutex contentions, and blocking API calls (`cudaStreamSynchronize`, `.item()`). Token sampling results are piped via host-pinned ring buffers using asynchronous DMA signals (`dma_async_d2h`). Blocking synchronization calls inside the decode loop trigger diagnostic `E003`.

## 1.3 Compilation Model

The Velox optimizing compiler (`veloxc`) translates high-level `.vlx` source programs into zero-overhead native executables through a four-stage pipeline:

```
+-----------------------------------------------------------------------------------+
|                            VELOX COMPILATION PIPELINE                             |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  1. Lexer & Parser (veloxc.lexer, veloxc.parser)                                  |
|     Source Code (.vlx)                                                            |
|       --> Token Stream (SourceLocation Tracking)                                  |
|       --> ISO/IEC 14977 EBNF Grammar Validation (81 Production Rules)             |
|       --> Strongly-Typed Abstract Syntax Tree (23 AST Schemas)                    |
|                                                                                   |
|  2. Semantic Analyzer & Type Checker (veloxc.analyzer, veloxc.types)              |
|     AST Validation                                                                |
|       --> Const-Generic Dimension Unification (Fourier-Motzkin Elimination)       |
|       --> Disjoint Memory Tier Qualifier Checking (@vram_immutable, @bump_sram)   |
|       --> Disjoint Slice Lending & Aliasing Verification (Axiom 1)                |
|       --> Diagnostic Verification (E001-E025 Emission on Violation)               |
|                                                                                   |
|  3. Velox Memory & Lifetime Planner (veloxc::memory_planner)                                |
|     Memory Allocation Optimization                                                |
|       --> Static Lifetime Interval Graph Construction (G_int)                     |
|       --> Interval Graph Coloring & Scratchpad Offset Assignment                  |
|       --> Static VRAM Feasibility Proof Certificate Synthesis                     |
|                                                                                   |
|  4. Native Code Generator (veloxc.codegen)                                        |
|     Target Lowering                                                               |
|       --> Emits High-Performance ISO C99 Host Runtime Architecture                |
|       --> Emits Hardware Warp-Specialized CUDA / TMA / WGMMA Kernels              |
|       --> Static Binary Compilation (clang / nvcc) -> Standalone Server Binary    |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

---

# 2. Lexical Conventions & Token Grammar

## 2.1 Character Set, Whitespace & Source Encoding

Velox source files are encoded in UTF-8. 
- Whitespace consists of spaces (`U+0020`), horizontal tabs (`U+0009`), carriage returns (`U+000D`), and line feeds (`U+000A`).
- Whitespace is non-semantic and serves to separate tokens, except inside string literals.
- Tab stops are normalized to 4 spaces for precise diagnostic column computation.
- Line numbers begin at index 1; column offsets begin at index 1.

## 2.2 Comments

Velox supports two forms of comments conforming to C/Rust conventions:
1. **Single-line comments**: Begin with `//` and extend to the terminating newline character (`\n` or `\r\n`).
2. **Multi-line block comments**: Begin with `/*` and terminate with `*/`. Multi-line comments can span multiple lines and may be nested.

```velox
// This is a single-line comment.
/* This is a multi-line 
   block comment. */
/* Nested /* block comments */ are fully supported. */
```

## 2.3 Identifiers & Literals

### Identifiers
Identifiers name variables, weights, models, functions, scopes, and types. An identifier must begin with an ASCII letter (`[a-zA-Z]`) or an underscore (`_`), followed by zero or more ASCII letters, digits (`[0-9]`), or underscores:
```ebnf
IDENTIFIER = ( LETTER | "_" ), { LETTER | DIGIT | "_" } ;
LETTER     = "a" ... "z" | "A" ... "Z" ;
DIGIT      = "0" ... "9" ;
```

### Numeric Literals
Velox supports integer and floating-point literals:
1. **Decimal Integer**: Sequence of decimal digits (`[0-9]+`), e.g., `42`, `4096`.
2. **Hexadecimal Integer**: Prefix `0x` or `0X` followed by hexadecimal digits (`[0-9a-fA-F]+`), e.g., `0x7FFF`, `0x200000`.
3. **Floating-Point**: Decimal integer followed by a period `.`, fractional digits, and optional exponent (`[eE][+-]?[0-9]+`), e.g., `0.95`, `1e-5`, `500000.0`.

### String Literals
String literals are enclosed in double quotes `"` and support standard escape sequences (`\"`, `\\`, `\n`, `\r`, `\t`):
```ebnf
STRING_LITERAL = '"', { ESCAPED_CHAR | ~( '"' | "\" | "\n" ) }, '"' ;
```

## 2.4 Reserved Keywords & Memory Tier Qualifiers

The following character sequences are reserved keywords and cannot be used as identifiers:

### Structure & Declaration Keywords
```
model, target, hardware, device, vram_capacity, compute_capability, safety_factor,
scope, prefill, decode, chunked_prefill, speculative_verify,
weight, tensor, let, mut, const, fn, kernel, return, yield
```

### Control Flow Keywords
```
if, else, for, in, while
```

### Memory & System Keywords
```
import, safetensors, mmap, absorb_mla_q, absorb_mla_v,
dma_async_h2d, dma_async_d2h, tma_load_async, tma_store_async, mbarrier
```

### Memory Tier Attributes
```
@vram_immutable, @paged_vram, @bump_sram, @host_pinned, 
@wired_limit_bounded, @warp_specialized, @kernel_fuse
```

### Scalar Dtype Keywords
```
f32, f16, bf16, fp8_e4m3, fp8_e5m2, fp8_e8m0,
int8, int4, usize, u64, u32, u16, u8, i64, i32, i16, i8, bool
```

---

# 3. Hardware Target Specification (`target hardware`)

## 3.1 Syntax & Properties

Every Velox program must contain exactly one `target hardware` block at file scope. This block defines the accelerator hardware constraints used by the compiler to verify memory feasibility, select tensor core GEMM instructions, and optimize thread-block tile schedules.

```velox
target hardware {
    device: "nvidia_rtx_3050_laptop";
    vram_capacity: 4096 * MiB;
    compute_capability: 8.6;
    safety_factor: 0.95;
}
```

### Property Catalog
1. **`device`** (`STRING_LITERAL`): Informative device identifier string (e.g., `"nvidia_rtx_3050_laptop"`, `"nvidia_a100_80gb"`, `"nvidia_h100_sxm5_cluster"`).
2. **`vram_capacity`** (`ConstExpression`, optional `MemoryUnit`): Total physical accelerator VRAM in bytes. Must evaluate to a positive 64-bit integer.
3. **`compute_capability`** (`FLOAT_LITERAL`): NVIDIA Streaming Multiprocessor (SM) architectural level (e.g., `8.0` for Ampere A100, `8.6` for Ampere RTX 3050, `8.9` for Ada Lovelace RTX 4090, `9.0` for Hopper H100). The minimum supported capability is `7.0` (Volta). Values below `7.0` trigger diagnostic `E007`.
4. **`safety_factor`** (`FLOAT_LITERAL`): Allocation safety budget multiplier $\alpha \in [0.50, 1.00]$. Defaults to `0.95`. This reserves $(1.0 - \alpha)$ of VRAM for CUDA driver runtime contexts, OS display servers, and page table allocations.

## 3.2 Memory Multipliers

For developer ergonomics, the `vram_capacity` field supports standard IEC binary byte multipliers:
- `KiB`: $2^{10} = 1{,}024\text{ bytes}$
- `MiB`: $2^{20} = 1{,}048{,}576\text{ bytes}$
- `GiB`: $2^{30} = 1{,}073{,}741{,}824\text{ bytes}$
- `TiB`: $2^{40} = 1{,}099{,}511{,}627{,}776\text{ bytes}$

```velox
vram_capacity: 80 * GiB;    // Evaluates to 85,899,345,920 bytes
vram_capacity: 4096 * MiB;  // Evaluates to 4,294,967,296 bytes
```

## 3.3 Compile-Time Static VRAM Feasibility Formulation

During compilation, the compiler evaluates the total memory consumption:
$$\mathbf{VRAM}_{\text{total}} = \mathbf{S}_{\text{weights}} + \mathbf{S}_{\text{kv\_cache}}(B_{\max}, S_{\max}) + \mathbf{S}_{\text{velox\_act}} + \mathbf{S}_{\text{system\_reserve}}$$

Where:
1. $\mathbf{S}_{\text{weights}} = \sum_{w \in \text{Weights}} \prod_{d \in \text{shape}(w)} d \times \text{sizeof}(\text{dtype}(w))$
2. $\mathbf{S}_{\text{kv\_cache}}(B_{\max}, S_{\max})$ is evaluated according to the chosen KV-cache type (Section 5).
3. $\mathbf{S}_{\text{velox\_act}}$ is computed by the interval graph coloring allocator over all intermediate activation tensors.
4. $\mathbf{S}_{\text{system\_reserve}}$ is the default system reserve ($512\text{ MiB}$ on consumer GPUs, $1024\text{ MiB}$ on datacenter GPUs).

If $\mathbf{VRAM}_{\text{total}} > \alpha \cdot \mathbf{VRAM}_{\text{budget}}$, compilation terminates immediately with diagnostic `E002`, displaying the exact memory breakdown and indicating how much memory must be pruned or offloaded.

---

# 4. Type System & Physical Memory Hierarchy

## 4.1 Scalar & Numeric Types

Velox provides a strongly-typed numeric foundation tailored to modern AI accelerators and low-precision Tensor Core pipelines:

| Type Keyword | Bit Width | Representation Format | Hardware Execution Target |
|---|---|---|---|
| `f32` | 32 bits | IEEE 754 Single-Precision Float | Standard FP32 ALUs / Accumulator |
| `f16` | 16 bits | IEEE 754 Half-Precision Float | Tensor Cores (Ampere, Ada, Hopper) |
| `bf16` | 16 bits | Brain Floating Point (1s, 8e, 7m) | Tensor Cores (Ampere+) |
| `fp8_e4m3` | 8 bits | OCP / NVIDIA FP8 (1s, 4e, 3m) | Hopper / Blackwell FP8 Tensor Cores |
| `fp8_e5m2` | 8 bits | OCP / NVIDIA FP8 (1s, 5e, 2m) | Hopper / Blackwell FP8 Tensor Cores |
| `fp8_e8m0` | 8 bits | Microscaling Block Scale Exponent | MXFP8 Block Scaling Factors |
| `int8` | 8 bits | 2's Complement Signed Integer | INT8 Tensor Cores / Quantized Embeddings |
| `int4` | 4 bits | 4-bit Signed Integer (Packed) | Marlin / AWQ / GPTQ Dequantization |
| `usize` | 64 bits | Unsigned Architecture Word Size | Pointer Offsets, Buffer Capacities |
| `u64`, `u32`, `u16`, `u8` | 64, 32, 16, 8 bits | Unsigned Integer Formats | Sequence Indices, Token IDs, Bitmasks |
| `i64`, `i32`, `i16`, `i8` | 64, 32, 16, 8 bits | Signed Integer Formats | Relative Coordinate Offsets |
| `bool` | 8 bits | Boolean (`true` or `false`) | Branch Conditions, Predication Masks |

### Type Promotion & Precision Guards
Implicit downcasting of floating-point precision (e.g., assigning an `f32` expression into an `fp8_e4m3` tensor) is strictly forbidden. Any narrowing conversion without an explicit quantization intrinsic triggers diagnostic `E024`.

## 4.2 Const-Generic Tensor Types (`Tensor<[Dim...], Dtype, BlockScaleSpec>`)

Tensors are first-class, statically dimensioned types parameterized by compile-time shape tuples:

```velox
// Rank-2 Activation Tensor
Tensor<[B, D], f16>

// Rank-3 Attention Query Tensor
Tensor<[B, S, H * D_H], f16>

// INT4 Block-Quantized Weight Tensor with FP8 Scale
Tensor<[4096, 14336], int4, block_scale=[128, 128], scale_dtype=fp8_e8m0>
```

### Shape Specifications
1. **Rank & Dimensions**: The shape is enclosed in square brackets `[...]`. Each dimension expression may be a positive integer literal, a const-generic parameter (e.g., `B`, `S`, `HIDDEN_DIM`), or a compile-time arithmetic expression (e.g., `H * D_H`, `INTERMEDIATE_DIM / TP`).
2. **Dynamic Wildcard (`*`)**: The wildcard `*` indicates a runtime-inferred dimension bounded by an outer const-generic maximum (e.g., `Tensor<[B, *], u32>`).
3. **Block Scaling**: Quantized weight tensors define the block granularity (e.g., `block_scale=[128, 128]`) and scale data type (`scale_dtype=fp8_e8m0`), natively compiling to Marlin or NVFP4/8 hardware dequantizers.

## 4.3 The Four Disjoint Physical Memory Tiers

In Velox, physical memory locations are encoded directly in the type system. Rather than treating all memory as a uniform virtual address space, Velox enforces **four disjoint physical memory tiers**:

```
+-----------------------------------------------------------------------------------------------+
|                                 VELOX PHYSICAL MEMORY TIERS                                   |
+-----------------------------------------------------------------------------------------------+
| 1. @vram_immutable: Accelerator HBM (GDDR6X/HBM3e). Read-only weights mapped via zero-copy   |
|    safetensors mmap. Write-protected at the MMU level.                                        |
|                                                                                               |
| 2. @paged_vram:      Accelerator HBM. Read-write KV cache and sequence pool managed via       |
|    2.0 MiB hardware-aligned block tables. Zero external fragmentation.                        |
|                                                                                               |
| 3. @bump_sram:       Accelerator On-Chip SMEM (TMA Register File / L1 Shared Memory).         |
|    Ultra-low latency (19-cycle access). Strictly scoped to thread-block lifetimes.             |
|                                                                                               |
| 4. @host_pinned:     Non-pageable Host RAM (OS mlock). DMA-accessible circular ring buffers    |
|    for token dispatch and sampling results. Sub-80us latency.                                 |
+-----------------------------------------------------------------------------------------------+
```

### 1. `@vram_immutable` (Accelerator Static Memory)
- **Physical Location**: High-Bandwidth Memory (HBM) on the accelerator.
- **Mutability**: Strictly read-only ($\mathbf{RO}$). Write operations or passing as a mutable reference (`&mut`) trigger compiler diagnostic `E004`.
- **Allocation Mechanism**: Mapped directly from disk into GPU VRAM using OS zero-copy file mapping (`mmap`) and `cuMemHostRegister` / GPUDirect Storage. No host-side buffer copies.
- **Lifetime**: Process static (allocated at model load, freed at shutdown).

### 2. `@paged_vram` (Accelerator Paged Memory Pool)
- **Physical Location**: High-Bandwidth Memory (HBM) on the accelerator.
- **Mutability**: Read-write ($\mathbf{RW}$).
- **Allocation Mechanism**: Divided into fixed-size physical blocks aligned to 2.0 MiB hardware Huge Page boundaries. Allocated and reclaimed in $O(1)$ time via bitmask registers.
- **Lifetime**: Dynamic request sequence lifetime.

### 3. `@bump_sram` (Accelerator On-Chip Shared Memory / Register File)
- **Physical Location**: Accelerator Streaming Multiprocessor (SM) On-Chip L1 Shared Memory / TMA Register File.
- **Mutability**: Read-write ($\mathbf{RW}$).
- **Addressing**: 32-bit local SMEM offset.
- **Lifetime & Scope Invariant**: An allocation in `@bump_sram` cannot outlive the lexical scope or kernel thread-block that created it. Returning a reference to `@bump_sram` or allowing it to escape the scope triggers compiler diagnostic `E006`.

### 4. `@host_pinned` (Host Non-Pageable Memory)
- **Physical Location**: Host DRAM locked via operating system non-pageable memory primitives (`mlock` on Linux, `VirtualLock` on Windows).
- **Mutability**: Read-write ($\mathbf{RW}$).
- **Addressing**: Mapped into the accelerator address space via PCI-Express / NVLink BAR1 for direct DMA access.
- **Lifetime**: Bound to the host runtime circular ring buffer.
- **Bounded Slab Guarantee**: Bound with `@wired_limit_bounded` to prevent exceeding OS resource limits.

### Memory Tier Disjointness Rules
1. The memory tiers form a set of disjoint storage classes. They do not share a subtyping hierarchy.
2. Casting an immutable reference (`@vram_immutable`) to a mutable tier (`@paged_vram`) triggers diagnostic `E004`.
3. DMA transfers cannot bypass intermediate VRAM stages (e.g., initiating direct DMA between `@bump_sram` and `@host_pinned` triggers diagnostic `E005`).

## 4.4 Inter-Tier Asynchronous Data Movement Primitives

Moving data between physical tiers requires explicit asynchronous transfer intrinsics that compile down to hardware DMA engines and Hopper Tensor Memory Accelerator (TMA) instructions:

```velox
// Load tile from global VRAM into on-chip SMEM via TMA
tma_load_async(src_vram_tile, dst_sram_tile, mbarrier);

// Store computed tile from SMEM back to global VRAM
tma_store_async(src_sram_tile, dst_vram_tile);

// Asynchronously transfer input prompt tokens from Host RAM to VRAM
dma_async_h2d(host_token_buffer, vram_token_tensor, dma_signal);

// Asynchronously stream sampled next token IDs from VRAM to Host RAM
dma_async_d2h(vram_sampled_token, host_token_ring, dma_signal);
```

### TMA Synchronization Rule
Every invocation of `tma_load_async` must be paired with an explicit hardware barrier synchronization (`mbarrier`). Attempting to read SMEM data before committing the barrier triggers compiler diagnostic `E020`.

---

# 5. Native Key-Value Cache Types

Velox replaces unstructured, untyped multidimensional float arrays with five first-class native KV-cache types:

```
+-----------------------------------------------------------------------------------------------+
|                                 VELOX NATIVE KV CACHE TYPES                                   |
+-----------------------------------------------------------------------------------------------+
| 1. PagedKVCache<L, H, D, BS, T>:            Continuous batching with 2MB huge-page blocks.    |
| 2. VirtualContiguousKV<L, H, D, S_MAX, T>:  CUDA Driver VMM for unpaged FlashAttention-3.     |
| 3. LatentKVCache<L, D_C, D_R, T>:           DeepSeek MLA latent compression (98.24% smaller). |
| 4. StreamingSinkKVCache<L, H, D, S, W, T>:  Infinite-context rolling FIFO with attention sinks|
| 5. SpeculativeTreeBlock<N, DEPTH, D, T>:    Multi-candidate tree verification DAG.            |
+-----------------------------------------------------------------------------------------------+
```

## 5.1 `PagedKVCache` (Block Paging & 2MB Huge-Page Alignment)

`PagedKVCache` provides memory-efficient, non-fragmenting storage for standard Multi-Head Attention (MHA) and Grouped Query Attention (GQA).

### Type Declaration
```velox
PagedKVCache<
    const NUM_LAYERS: usize,
    const NUM_KV_HEADS: usize,
    const HEAD_DIM: usize,
    const BLOCK_SIZE: usize,
    type DTYPE
>
```

### Physical Memory Formulation & Alignment
Physical memory is allocated in discrete block units. Each block stores Key and Value tokens for a single sequence across all layers and KV heads:
$$\mathbf{S}_{\text{block}} = 2 \times \text{NUM\_LAYERS} \times \text{NUM\_KV\_HEADS} \times \text{BLOCK\_SIZE} \times \text{HEAD\_DIM} \times \text{sizeof}(\text{DTYPE})$$

For example, in Meta Llama-3-8B ($L=32, H_{kv}=8, D_h=128, \text{DTYPE}=\text{f16}$):
$$\mathbf{S}_{\text{block}} = 2 \times 32 \times 8 \times 16 \times 128 \times 2 = 2{,}097{,}152\text{ bytes} \equiv 2.0\text{ MiB}$$
This achieves exact alignment with the GPU MMU 2.0 MiB Huge Page boundary, eliminating TLB thrashing.

### Invariants
1. `BLOCK_SIZE` must be a power of two ($BS \in \{16, 32, 64\}$). Non-power-of-two block sizes trigger diagnostic `E015`.
2. Total capacity at maximum batch size $B_{\max}$ and sequence length $S_{\max}$:
   $$\mathbf{S}_{\text{kv}} = B_{\max} \times \left\lceil \frac{S_{\max}}{\text{BLOCK\_SIZE}} \right\rceil \times \mathbf{S}_{\text{block}}$$

### Method Interfaces
- `kv.borrow_slot_mut(seq_id: usize, step: usize) -> KVSlotMut`: Obtains a mutable write view for the current generation step.
- `kv.borrow_context_const(seq_id: usize, range: Range) -> KVContextRef`: Obtains a read-only view of historical context for attention.

## 5.2 `VirtualContiguousKV` (CUDA Driver VMM)

For ultra-long-context models (e.g., Nanbeige-33B with 64K context), block table lookups inside attention inner loops degrade memory bandwidth. `VirtualContiguousKV` utilizes low-level CUDA Virtual Memory Management (VMM) APIs (`cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`, `cuMemSetAccess`) to provide a virtually contiguous address space backed by dynamically committed physical 2MB pages.

### Type Declaration
```velox
VirtualContiguousKV<
    const NUM_LAYERS: usize,
    const NUM_KV_HEADS: usize,
    const HEAD_DIM: usize,
    const MAX_SEQ_LEN: usize,
    type DTYPE
>
```

### Architectural Operation
1. At initialization, `cuMemAddressReserve` reserves a contiguous 64-bit virtual address range corresponding to $B_{\max} \times S_{\max}$.
2. As sequences expand, physical 2MB memory handles are created and mapped into virtual addresses on-demand.
3. Attention kernels (e.g., FlashAttention-3) execute with bare pointers (`*const T`, `*mut T`) with zero indirection overhead.

## 5.3 `LatentKVCache` (DeepSeek MLA Latent Compression)

`LatentKVCache` implements Multi-Head Latent Attention (DeepSeek-V2 / DeepSeek-V3), which compresses Key and Value projections into a shared low-rank latent representation.

### Type Declaration
```velox
LatentKVCache<
    const NUM_LAYERS: usize,
    const LATENT_KV_DIM: usize,
    const ROPE_DIM: usize,
    type DTYPE
>
```

### Mathematical Compression Formulation
Standard MHA stores per-token keys and values of dimension $2 \times H \times D_h$. In DeepSeek-V3 ($H=128, D_h=128$):
$$\mathbf{S}_{\text{MHA\_token}} = 2 \times 128 \times 128 \times 1\text{ byte (FP8)} = 32{,}768\text{ bytes/token}$$

`LatentKVCache` stores only the low-rank compressed latent vector $\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c}$ ($d_c = 512$) and the decoupled rotary key $\mathbf{k}_t^R \in \mathbb{R}^{d_R}$ ($d_R = 64$):
$$\mathbf{S}_{\text{MLA\_token}} = (512 + 64) \times 1\text{ byte (FP8)} = 576\text{ bytes/token}$$
This achieves a **98.24% reduction** in KV cache memory footprint ($56.88\times$ compression ratio).

### Offline Matrix Absorption Invariant
At runtime, keys and values are **never decompressed into full MHA tensors in HBM**. Attention executes directly against compressed latents using pre-absorbed projection weights (`absorb_mla_q`, `absorb_mla_v`). Decompressing latent vectors into global memory triggers diagnostic `E018`.

## 5.4 `StreamingSinkKVCache` (Modulo Ring Arithmetic & Infinite Context)

`StreamingSinkKVCache` implements streaming inference (StreamingLLM) for infinite sequence lengths with constant $O(1)$ physical memory.

### Type Declaration
```velox
StreamingSinkKVCache<
    const NUM_LAYERS: usize,
    const NUM_KV_HEADS: usize,
    const HEAD_DIM: usize,
    const NUM_SINKS: usize,
    const WINDOW_SIZE: usize,
    type DTYPE
>
```

### Memory Formulation
$$\mathbf{S}_{\text{streaming}} = 2 \times \text{NUM\_LAYERS} \times \text{NUM\_KV\_HEADS} \times (\text{NUM\_SINKS} + \text{WINDOW\_SIZE}) \times \text{HEAD\_DIM} \times \text{sizeof}(\text{DTYPE}) \equiv \text{Constant}$$
Tokens beyond `NUM_SINKS` wrap cyclically inside a rolling FIFO window using modulo address swizzling:
$$\text{slot}(t) = \begin{cases} t & \text{if } t < \text{NUM\_SINKS} \\ \text{NUM\_SINKS} + ((t - \text{NUM\_SINKS}) \pmod{\text{WINDOW\_SIZE}}) & \text{otherwise} \end{cases}$$

Declaring a `StreamingSinkKVCache` with non-constant or zero parameters triggers diagnostic `E016`.

## 5.5 `SpeculativeTreeBlock` (2D Non-Causal Masking & Path Compaction)

`SpeculativeTreeBlock` provides native data structures for speculative decoding and tree verification (Medusa, EAGLE-2, SpecInfer).

### Type Declaration
```velox
SpeculativeTreeBlock<
    const MAX_CANDIDATE_NODES: usize,
    const MAX_TREE_DEPTH: usize,
    const HEAD_DIM: usize,
    type DTYPE
>
```

### Tree Attention Mask & Compaction
1. Encapsulates candidate token IDs, branch topologies, and parent pointers.
2. Synthesizes a 2D non-causal attention mask:
   $$M_{\text{tree}}[u, v] = \begin{cases} 0.0 & \text{if } v \in \text{Ancestors}(u) \cup \{u\} \\ -\infty & \text{otherwise} \end{cases}$$
3. Upon completion of parallel verification, `commit_accepted_path` gathers the verified candidate tokens into authoritative sequence slots in $O(L)$ time, discarding rejected branches without memory defragmentation.
4. Speculative candidate trees exceeding `MAX_TREE_DEPTH` trigger diagnostic `E017`.

---

# 6. Model Architecture & Weight Declarations

## 6.1 The `model` Declaration & Const-Generic Parameterization

Neural network architectures are declared using the `model` block, parameterized by compile-time const-generic dimensions:

```velox
model Llama3_8B<
    const B_MAX: usize = 32,
    const S_MAX: usize = 4096,
    const VOCAB: usize = 128256,
    const HIDDEN_DIM: usize = 4096,
    const INTERMEDIATE_DIM: usize = 14336,
    const NUM_LAYERS: usize = 32,
    const NUM_HEADS: usize = 32,
    const NUM_KV_HEADS: usize = 8,
    const HEAD_DIM: usize = 128
> {
    // Model members: weights, KV caches, scopes, helper functions
}
```

### Semantic Rules
1. **Strongly-Typed Const Generics**: All generic parameters must be typed as integer scalars (`usize`, `u64`, `u32`). Floating-point or non-scalar generic parameters trigger compiler diagnostic `E009`.
2. **Default Values**: Generic parameters may specify default const-expressions (e.g., `const B_MAX: usize = 32`).
3. **Mandatory Scope Pair**: A model declaration must contain at least one `prefill` scope and one `decode` scope. Omitting either triggers diagnostic `E008`.
4. **Member Symbol Isolation**: Weights, KV caches, scopes, and helper functions share the model's lexical scope. Duplicate member identifiers trigger diagnostic `E022`.

## 6.2 Zero-Copy Safetensors Weight Loading (`safetensors(...)`)

Weights are declared with the `weight` keyword and bound to `.safetensors` files using the `safetensors(filename, tensor_key)` intrinsic:

```velox
@vram_immutable weight w_embed: Tensor<[VOCAB, HIDDEN_DIM], f16> = 
    safetensors("model.safetensors", "model.embed_tokens.weight");

@vram_immutable weight w_q: Tensor<[NUM_LAYERS, HIDDEN_DIM, 4096], f16> = 
    safetensors("model.safetensors", "model.layers.self_attn.q_proj.weight");
```

### Loading Semantics
1. **`@vram_immutable` Invariant**: All `weight` declarations are implicitly or explicitly qualified with `@vram_immutable`. Attempting to mutate a weight or declare it as `mut` triggers diagnostic `E004`.
2. **Zero-Copy Host Bypassing**: The compiler generates code that maps `.safetensors` headers and data payloads directly into accelerator HBM via OS file mapping (`mmap`) and GPUDirect Storage / `cuMemHostRegister`. No intermediate heap allocation occurs on the host CPU.
3. **Metadata Verification**: The compiler checks that the declared tensor shape and dtype match the `.safetensors` header. A missing key triggers diagnostic `E013`; a mismatched data type triggers diagnostic `E014`.

## 6.3 Block Quantization Bindings (INT4 Marlin & FP8 Microscaling)

Velox supports state-of-the-art block-quantized weights for sub-4-bit inference:

```velox
// INT4 Marlin Weight with FP8 Microscaling
@vram_immutable weight w_gate: Tensor<[NUM_LAYERS, HIDDEN_DIM, INTERMEDIATE_DIM], int4, 
    block_scale=[128, 128], scale_dtype=fp8_e8m0> = 
    safetensors("model_int4.safetensors", "model.layers.mlp.gate_proj.weight");
```

- `block_scale=[M, K]`: Specifies the 2D quantization tile dimensions (e.g., $128 \times 128$).
- `scale_dtype`: Specifies the numeric format of the block scaling exponents (`fp8_e8m0`, `f16`).
- Hardware lowering automatically selects asynchronous FP16 $\times$ INT4 Marlin GEMM or Hopper FP8 WGMMA tensor core instructions.

## 6.4 Offline Matrix Absorption Declarations (`absorb_mla_q`, `absorb_mla_v`)

For Multi-Head Latent Attention (DeepSeek-V2 / DeepSeek-V3), Velox provides offline matrix absorption primitives:

```velox
// Absorbed Query-Key Projection: W_absorbed_q = W_uq @ W_uk^T
@vram_immutable weight w_absorbed_q: Tensor<[NUM_LAYERS, NUM_HEADS, LATENT_KV_DIM, HEAD_DIM], fp8_e4m3> = 
    absorb_mla_q(w_uq, w_uk);

// Absorbed Value-Output Projection: W_absorbed_o = W_o @ W_uv
@vram_immutable weight w_absorbed_o: Tensor<[NUM_LAYERS, NUM_HEADS, HEAD_DIM, HIDDEN_DIM], fp8_e4m3> = 
    absorb_mla_v(w_o, w_uv);
```

### Mathematical Principle
In standard attention, query, key, and value vectors are decompressed into $H \times D_h$ matrices:
$$\mathbf{q}_t = \mathbf{c}_t^Q W_{UQ}, \quad \mathbf{k}_t = \mathbf{c}_t^{KV} W_{UK}, \quad \mathbf{v}_t = \mathbf{c}_t^{KV} W_{UV}$$
$$\mathbf{A}_{t, \tau} = \mathbf{q}_t \mathbf{k}_\tau^T = (\mathbf{c}_t^Q W_{UQ}) (W_{UK}^T (\mathbf{c}_\tau^{KV})^T) = \mathbf{c}_t^Q (W_{UQ} W_{UK}^T) (\mathbf{c}_\tau^{KV})^T$$
$$\mathbf{o}_t = \left( \sum_\tau \alpha_{t, \tau} \mathbf{v}_\tau \right) W_O = \sum_\tau \alpha_{t, \tau} (\mathbf{c}_\tau^{KV} W_{UV}) W_O = \sum_\tau \alpha_{t, \tau} \mathbf{c}_\tau^{KV} (W_{UV} W_O)$$

By multiplying $W_{UQ} W_{UK}^T \to W_{\text{absorbed\_q}}$ and $W_{UV} W_O \to W_{\text{absorbed\_o}}$ offline at load time:
1. Keys and values are **never materialized in uncompressed form in HBM**.
2. Attention inner loops execute with reduced memory traffic directly against the compressed $512$-dim latent vectors.

---

# 7. Execution Pipeline Scopes

Velox organizes serving computation into **four phase-disjoint execution scopes**:

```
+-----------------------------------------------------------------------------------------------+
|                                VELOX EXECUTION PIPELINE SCOPES                                |
+-----------------------------------------------------------------------------------------------+
| 1. scope prefill:            High-throughput Compute-Bound GEMM (Prompt Encoding)             |
| 2. scope decode:             Ultra-low-latency Bandwidth-Bound GEMV (Single-Token Generation) |
| 3. scope chunked_prefill:    Interleaved Serving (Sarathi-Serve eliminates pipeline bubbles)  |
| 4. scope speculative_verify: Parallel Non-Causal Tree Verification (SpecInfer / EAGLE-2)     |
+-----------------------------------------------------------------------------------------------+
```

## 7.1 `scope prefill`: Compute-Bound GEMM Pipeline

The `prefill` scope processes full prompt sequence tokens ($S > 1$). Execution is compute-bound, maximizing Tensor Core utilization:

```velox
scope prefill<const B: usize, const S: usize>(
    tokens: Tensor<[B, S], u32>
) -> Tensor<[B, 1], u32> {
    let mut hidden = embed(tokens, w_embed);

    for l in 0..NUM_LAYERS {
        let norm_x = rmsnorm(hidden, w_attn_norm[l]);
        let q = norm_x @ w_q[l];
        let k = norm_x @ w_k[l];
        let v = norm_x @ w_v[l];
        
        let attn_out = flash_prefill(q, k, v, &mut kv_cache, l);
        hidden += attn_out @ w_out[l];

        let mlp_norm = rmsnorm(hidden, w_mlp_norm[l]);
        hidden += swiglu(mlp_norm, w_gate[l], w_up[l], w_down[l]);
    }

    let final_norm = rmsnorm(hidden, w_final_norm);
    let logits = final_norm[:, S - 1, :] @ w_head;
    return sample_argmax(logits);
}
```

## 7.2 `scope decode`: Bandwidth-Bound GEMV Pipeline & Sub-80µs Dispatch

The `decode` scope processes a single generated token ($S = 1$) per active sequence. Execution is strictly memory-bandwidth bound:

```velox
scope decode<const B: usize>(
    next_token: Tensor<[B, 1], u32>,
    step: usize
) -> Tensor<[B, 1], u32> {
    let mut hidden = embed(next_token, w_embed);

    for l in 0..NUM_LAYERS {
        let norm_x = rmsnorm(hidden, w_attn_norm[l]);
        let q = norm_x @ w_q[l];
        let k = norm_x @ w_k[l];
        let v = norm_x @ w_v[l];

        // Disjoint borrow: write new token to slot step
        let mut slot = kv_cache.borrow_slot_mut(0, step);
        slot.write(l, k, v);

        let attn_out = flash_decode(q, &kv_cache, l);
        hidden += attn_out @ w_out[l];

        let mlp_norm = rmsnorm(hidden, w_mlp_norm[l]);
        hidden += swiglu(mlp_norm, w_gate[l], w_up[l], w_down[l]);
    }

    let final_norm = rmsnorm(hidden, w_final_norm);
    let logits = final_norm @ w_head;
    return sample_argmax(logits);
}
```

### Sub-80µs Guarantee
Inside `scope decode`, blocking synchronization calls (`cudaStreamSynchronize`, `.item()`) are prohibited. The compiled runtime launches CUDA Graph instances or streams execution over pre-allocated hardware work queues, achieving $\mathcal{T}_{\text{dispatch}} \le 25\,\mu\text{s}$.

## 7.3 `scope chunked_prefill`: Interleaved Chunked Serving (Sarathi-Serve)

To prevent long prompt prefill passes from causing head-of-line blocking on ongoing decode requests, `scope chunked_prefill` chunks prompts into fixed slices of size $C^*$ (e.g., 512 tokens) and piggybacks them onto batch decode steps:

```velox
scope chunked_prefill<const CHUNK_SIZE: usize = 512, const B_DEC: usize = 16>(
    chunk_tokens: Tensor<[1, CHUNK_SIZE], u32>,
    decode_tokens: Tensor<[B_DEC, 1], u32>
) -> (Tensor<[1, 1], u32>, Tensor<[B_DEC, 1], u32>) {
    // Interleaved execution computes prompt GEMM and batch GEMV simultaneously
    ...
}
```

## 7.4 `scope speculative_verify`: Parallel Speculative Tree Verification

Speculative decoding verifies a tree of candidate tokens generated by a small draft model or Medusa heads:

```velox
scope speculative_verify(
    seq_id: usize,
    tree: &mut SpeculativeTreeBlock<64, 6, HEAD_DIM, f16>
) -> SpeculativePath<6> {
    let tree_mask = tree.synthesize_tree_mask();
    let candidate_tokens = tree.get_candidate_tokens();
    let mut h = embed(candidate_tokens, w_embed);

    for l in 0..NUM_LAYERS {
        ...
        let attn_tree = flash_tree_attention(q, k, v, tree_mask, &kv_cache, l);
        ...
    }

    let logits = rmsnorm(h, w_norm) @ w_head;
    let accepted_path = tree.resolve_best_path(logits);
    tree.commit_accepted_path(&mut kv_cache, accepted_path);
    return accepted_path;
}
```

---

# 8. Expressions, Statements & Operator Semantics

## 8.1 14-Level Operator Precedence Hierarchy

Velox expressions conform to a strict 14-level precedence hierarchy:

| Precedence Level | Category | Operators | Associativity | Description |
|---|---|---|---|---|
| **Level 1 (Highest)** | Primary & Postfix | `.` `[]` `()` `[..]` | Left-to-Right | Member access, indexing, call invocations, tensor slices |
| **Level 2** | Unary | `!` `~` `-` `*` `&` `&mut` | Right-to-Left | Logical NOT, bitwise NOT, arithmetic negation, dereference, borrows |
| **Level 3** | Matrix Multiplication | `@` | Left-to-Right | Hardware Tensor Core / WGMMA contraction |
| **Level 4** | Multiplicative | `*` `/` `%` | Left-to-Right | Elementwise multiplication, division, modulo |
| **Level 5** | Additive | `+` `-` | Left-to-Right | Elementwise addition and subtraction |
| **Level 6** | Bitwise Shifts | `<<` `>>` | Left-to-Right | Bitwise left and right shifts |
| **Level 7** | Relational | `<` `<=` `>` `>=` | Left-to-Right | Numerical inequality comparisons |
| **Level 8** | Equality | `==` `!=` | Left-to-Right | Value equivalence and inequality checks |
| **Level 9** | Bitwise AND | `&` | Left-to-Right | Bitwise AND mask |
| **Level 10** | Bitwise XOR | `^` | Left-to-Right | Bitwise XOR difference |
| **Level 11** | Bitwise OR | `\|` | Left-to-Right | Bitwise OR combination |
| **Level 12** | Logical AND | `&&` | Left-to-Right | Short-circuit boolean conjunction |
| **Level 13** | Logical OR | `\|\|` | Left-to-Right | Short-circuit boolean disjunction |
| **Level 14 (Lowest)** | Assignment Statements | `=` `+=` `-=` `*=` `/=` | Right-to-Left | Value assignment and in-place compound assignment |

*Note: In Velox, assignments are statements (`AssignmentStmt`), terminating with a semicolon `;`. They are placed at Level 14 to indicate that all Right-Hand Side expressions bind with higher precedence.*

## 8.2 Matrix Multiplication Contraction Rule (`@`)

Matrix multiplication is a first-class language operator represented by `@`. 

### Typing Judgment
$$\frac{\Gamma; \Delta \vdash A : \text{Tensor}\langle [d_1, \dots, d_{n-1}, K_A], \tau_1 \rangle \quad \Gamma; \Delta \vdash B : \text{Tensor}\langle [K_B, d_{n+1}, \dots, d_m], \tau_2 \rangle \quad \Delta \models K_A \equiv K_B}{\Gamma; \Delta \vdash A @ B : \text{Tensor}\langle [d_1, \dots, d_{n-1}, d_{n+1}, \dots, d_m], \text{Promote}(\tau_1, \tau_2) \rangle}$$

### Dimension Contraction Invariant
The inner contraction dimension $K_A$ of the left-hand operand must statically unify with the outer dimension $K_B$ of the right-hand operand ($K_A \equiv K_B$). If the dimensions do not match, compilation terminates with diagnostic `E010`:
```velox
let x: Tensor<[B, 4096], f16> = ...;
let w: Tensor<[2048, 8192], f16> = ...;
let y = x @ w; // Error E010: Inner contraction dimension 4096 != 2048
```

## 8.3 Statements & Control Flow

### Variable Bindings (`let` and `mut`)
Variables are immutable by default:
```velox
let hidden = embed(tokens, w_embed); // Immutable binding
let mut x = norm_x;                   // Mutable binding
x += residual;                        // In-place mutation allowed
```

### Control Flow
1. **Conditional Statements (`if / else`)**:
   ```velox
   if (step == 0) {
       kv_cache.reset();
   } else {
       kv_cache.advance_step();
   }
   ```
2. **Bounded Loops (`for in ..`)**:
   ```velox
   for l in 0..NUM_LAYERS {
       // Layer forward pass
   }
   ```
   *Loop bounds must be compile-time constants or statically bounded expressions. Dynamic non-constant loop bounds inside execution fibers trigger diagnostic `E021`.*
3. **While Loops (`while`)**:
   ```velox
   while (curr_step < max_steps) {
       curr_step += 1;
   }
   ```
4. **Returns & Yields (`return`, `yield`)**:
   ```velox
   return sample_argmax(logits);
   yield(next_token);
   ```

## 8.4 Linear Ownership, Mutability & Disjoint Borrowing Axioms

Velox applies compile-time ownership tracking to guarantee that GPU buffers and KV-cache blocks are never concurrently mutated:

### Axiom 1: Disjoint Borrow Rule
Let $C$ be a `PagedKVCache` or `VirtualContiguousKV`. Multiple borrows $\text{ref}_1, \dots, \text{ref}_n$ into $C$ may coexist within the same execution block if and only if their index tuples are statically proven disjoint:
$$\forall i \ne j, \quad (\text{seq}_i \ne \text{seq}_j) \lor (\text{step}_i \ne \text{step}_j) \lor (\text{layer}_i \ne \text{layer}_j)$$

If two borrow expressions cannot be proven disjoint at compile time, the second borrow is rejected with diagnostic `E012`.

---

# 9. Standard Library Builtin Functions & Kernels Registry

The Velox Standard Library provides native, highly-optimized hardware intrinsics for Transformer architectures.

## 9.1 Embedding & Normalization

### 1. `embed`
Maps integer token IDs into dense hidden representation vectors:
```velox
fn embed<const B: usize, const S: usize, const D: usize, const VOCAB: usize, type T>(
    token_ids: Tensor<[B, S], u32>,
    w_embed: @vram_immutable Tensor<[VOCAB, D], T>
) -> Tensor<[B, S, D], T>;
```
- **Operation**: Parallel table lookup: $\mathbf{X}[b, s, :] = \mathbf{W}_{\text{embed}}[\text{token\_ids}[b, s], :]$.
- **Hardware Lowering**: Coalesced global memory reads with zero intermediate register spills.

### 2. `rmsnorm`
Root Mean Square Layer Normalization:
```velox
fn rmsnorm<const B: usize, const S: usize, const D: usize, type T>(
    x: Tensor<[B, S, D], T>,
    w_gamma: @vram_immutable Tensor<[D], T>,
    eps: f32 = 1e-5
) -> Tensor<[B, S, D], T>;
```
- **Mathematical Formulation**:
  $$\text{RMS}(x) = \sqrt{\frac{1}{D} \sum_{i=1}^D x_i^2 + \epsilon}, \quad \text{RMSNorm}(x)_i = \frac{x_i}{\text{RMS}(x)} \cdot \gamma_i$$
- **Hardware Lowering**: Warp-level reduction using `__shfl_xor_sync` across 32 threads. Single-pass SMEM caching.

### 3. `rotary_embedding`
Rotary Position Embedding (RoPE) applied to query and key vectors:
```velox
fn rotary_embedding<const B: usize, const S: usize, const H: usize, const D: usize, type T>(
    q: Tensor<[B, S, H, D], T>,
    k: Tensor<[B, S, H, D], T>,
    cos_table: @vram_immutable Tensor<[S, D / 2], f32>,
    sin_table: @vram_immutable Tensor<[S, D / 2], f32>,
    step_offset: usize = 0
) -> (Tensor<[B, S, H, D], T>, Tensor<[B, S, H, D], T>);
```
- **Mathematical Formulation**: Rotates pairs of coordinate components:
  $$\begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix} \leftarrow \begin{pmatrix} \cos(m \theta_i) & -\sin(m \theta_i) \\ \sin(m \theta_i) & \cos(m \theta_i) \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}$$

## 9.2 High-Throughput Attention

### 1. `flash_prefill`
Fused FlashAttention-2/3 prefill kernel with online softmax:
```velox
fn flash_prefill<const B: usize, const S: usize, const H: usize, const H_KV: usize, const D: usize, type T>(
    q: Tensor<[B, S, H * D], T>,
    k: Tensor<[B, S, H_KV * D], T>,
    v: Tensor<[B, S, H_KV * D], T>,
    kv_cache: &mut PagedKVCache<LAYERS, H_KV, D, BLOCK_SIZE, T>,
    layer_idx: usize
) -> Tensor<[B, S, H * D], T>;
```
- **Operation**: Appends new keys and values to the paged cache and computes causal attention across all prompt tokens in a single fused kernel.
- **Tiling**: $128 \times 64$ SMEM tiles with ping-pong double buffering.

### 2. `flash_decode`
Paged FlashAttention GEMV decode kernel:
```velox
fn flash_decode<const B: usize, const H: usize, const H_KV: usize, const D: usize, type T>(
    q: Tensor<[B, 1, H * D], T>,
    kv_cache: &PagedKVCache<LAYERS, H_KV, D, BLOCK_SIZE, T>,
    layer_idx: usize
) -> Tensor<[B, 1, H * D], T>;
```
- **Operation**: Evaluates attention for single-token queries across historical KV blocks fetched via page tables.
- **Optimization**: Split-K parallel reduction across sequence blocks for small batch sizes ($B=1$).

### 3. `mla_decode_absorbed`
Absorbed Multi-Head Latent Attention decode kernel (DeepSeek-V2 / DeepSeek-V3):
```velox
fn mla_decode_absorbed<const B: usize, const H: usize, const D: usize, const LATENT_DIM: usize, const ROPE_DIM: usize, type T>(
    q_absorbed: Tensor<[B, 1, H, LATENT_DIM], T>,
    q_rope: Tensor<[B, 1, H, ROPE_DIM], T>,
    kv_cache: &LatentKVCache<LAYERS, LATENT_DIM, ROPE_DIM, T>,
    layer_idx: usize
) -> Tensor<[B, 1, H * D], T>;
```
- **Operation**: Computes inner product between pre-absorbed queries $\mathbf{q}_{\text{absorbed}}$ and compressed latent cache $\mathbf{c}^{KV}$, bypassing uncompressed key/value materialization in VRAM.

### 4. `flash_tree_attention`
Non-causal speculative tree attention kernel:
```velox
fn flash_tree_attention<const N: usize, const H: usize, const D: usize, type T>(
    q: Tensor<[1, N, H * D], T>,
    k: Tensor<[1, N, H * D], T>,
    v: Tensor<[1, N, H * D], T>,
    tree_mask: Tensor<[N, N], f32>,
    kv_cache: &PagedKVCache<LAYERS, H_KV, D, BLOCK_SIZE, T>,
    layer_idx: usize
) -> Tensor<[1, N, H * D], T>;
```

### 5. `flash_chunked_piggyback`
Interleaved chunked prefill and decode attention kernel:
```velox
fn flash_chunked_piggyback<const CHUNK_SIZE: usize, const B_DEC: usize, const H: usize, const D: usize, type T>(
    q_c: Tensor<[1, CHUNK_SIZE, H * D], T>,
    k_c: Tensor<[1, CHUNK_SIZE, H * D], T>,
    v_c: Tensor<[1, CHUNK_SIZE, H * D], T>,
    q_d: Tensor<[B_DEC, 1, H * D], T>,
    kv_cache: &mut PagedKVCache<LAYERS, H_KV, D, BLOCK_SIZE, T>,
    layer_idx: usize
) -> (Tensor<[1, CHUNK_SIZE, H * D], T>, Tensor<[B_DEC, 1, H * D], T>);
```

## 9.3 Feedforward, Activation & MoE

### 1. `silu` and `gelu`
```velox
fn silu<const B: usize, const S: usize, const D: usize, type T>(x: Tensor<[B, S, D], T>) -> Tensor<[B, S, D], T>;
fn gelu<const B: usize, const S: usize, const D: usize, type T>(x: Tensor<[B, S, D], T>) -> Tensor<[B, S, D], T>;
```

### 2. `swiglu`
Fused Swish-Gated Linear Unit MLP block:
```velox
fn swiglu<const B: usize, const S: usize, const D_IN: usize, const D_OUT: usize, type T>(
    x: Tensor<[B, S, D_IN], T>,
    w_gate: @vram_immutable Tensor<[D_IN, D_OUT], T>,
    w_up: @vram_immutable Tensor<[D_IN, D_OUT], T>,
    w_down: @vram_immutable Tensor<[D_OUT, D_IN], T>
) -> Tensor<[B, S, D_IN], T>;
```
- **Mathematical Formulation**:
  $$\text{SwiGLU}(x) = (\text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}})) W_{\text{down}}$$
- **Optimization**: Fuses the gate and up GEMMs and SiLU elementwise multiplication into a single register-retained kernel pass.

### 3. `moe_gate_topk` and `moe_dispatch_topk`
Mixture-of-Experts routing and dispatch:
```velox
fn moe_gate_topk<const B: usize, const S: usize, const D: usize, const NUM_EXPERTS: usize, const TOP_K: usize, type T>(
    x: Tensor<[B, S, D], T>,
    w_gate: @vram_immutable Tensor<[D, NUM_EXPERTS], f32>
) -> (Tensor<[B, S, TOP_K], usize>, Tensor<[B, S, TOP_K], f32>);

fn moe_dispatch_topk<const B: usize, const S: usize, const D: usize, const NUM_EXPERTS: usize, const TOP_K: usize, const D_EXP: usize, type T>(
    x: Tensor<[B, S, D], T>,
    indices: Tensor<[B, S, TOP_K], usize>,
    weights: Tensor<[B, S, TOP_K], f32>,
    w_experts: @vram_immutable Tensor<[NUM_EXPERTS, D, D_EXP], T>,
    capacity_factor: f32 = 2.0
) -> Tensor<[B, S, D], T>;
```

## 9.4 Sampling Kernels

### 1. `sample_argmax`
Greedy decoding (temperature = 0):
```velox
fn sample_argmax<const B: usize, const VOCAB: usize, type T>(
    logits: Tensor<[B, 1, VOCAB], T>
) -> Tensor<[B, 1], u32>;
```
- **Hardware Lowering**: Warp-aggregated tree reduction across $V$ vocabulary entries, outputting winning token IDs in $O(\log V)$ steps.

### 2. `sample_temperature_topp`
Stochastic Nucleus (Top-$p$) Sampling:
```velox
fn sample_temperature_topp<const B: usize, const VOCAB: usize, type T>(
    logits: Tensor<[B, 1, VOCAB], T>,
    temperature: f32 = 0.7,
    top_p: f32 = 0.9,
    seed: u64 = 42
) -> Tensor<[B, 1], u32>;
```

---

# 10. Diagnostic Error Catalog (E001 – E025)

## 10.1 Diagnostic Architecture & Deterministic Format

The Velox compiler diagnostic engine produces human-readable, deterministic diagnostics. Every diagnostic contains:
1. **Header**: Error code (`E001`–`E025`) and formal diagnostic name.
2. **Source Location**: Exact file path, line number, and column pointer.
3. **Visual Caret (`^`)**: Exact visual underline highlighting the offending token span.
4. **Detailed Explanation**: Formal explanation linking the error to language invariants or memory safety axioms.
5. **Remediation**: Actionable code replacement or configuration adjustment.

## 10.2 Complete Error Code Reference Table

| Code | Diagnostic Name | Category | Trigger Condition & Description | Remediation |
|---|---|---|---|---|
| **E001** | `DYNAMIC_HEAP_ALLOCATION_PROHIBITED` | Invariant 1 Violation | Calling `malloc`, `cudaMalloc`, `new` inside serving execution scopes. | Use statically dimensioned `@bump_sram` or Velox-allocated tensors. |
| **E002** | `TARGET_VRAM_BUDGET_EXCEEDED` | Invariant 2 Violation | Statically calculated memory footprint exceeds target hardware safety budget. | Reduce batch size $B_{\max}$, sequence length $S_{\max}$, or quantize weights. |
| **E003** | `BLOCKING_HOST_SYNC_IN_DECODE` | Invariant 3 Violation | Calling synchronous host-device primitives (`cudaStreamSynchronize`, `.item()`) inside `decode`. | Use asynchronous DMA ring buffers (`dma_async_d2h`) and non-blocking fibers. |
| **E004** | `ILLEGAL_MEMORY_TIER_CAST` | Memory Tier Violation | Attempting to cast an immutable tier (`@vram_immutable`) to a mutable tier (`@paged_vram`). | Allocate a mutable destination buffer and copy data explicitly. |
| **E005** | `INVALID_DIRECT_DMA_PATH` | Memory Tier Violation | Initiating DMA between incompatible memory tiers without intermediate staging (e.g. SRAM to Host). | Stage the transfer through global `@paged_vram`. |
| **E006** | `SRAM_REFERENCE_ESCAPES_SCOPE` | Lifetime & Aliasing | Pointer or reference to `@bump_sram` outlives its thread-block or enclosing lexical scope. | Copy data to `@paged_vram` via `tma_store_async` before scope exit. |
| **E007** | `UNSUPPORTED_HARDWARE_TARGET` | Target Configuration | Declaring compute capability $< 7.0$ (Volta). | Target a modern accelerator with compute capability $\ge 7.0$. |
| **E008** | `MISSING_EXECUTION_SCOPE` | Architecture Invariant | Model declaration omits mandatory `prefill` or `decode` execution scope. | Define both `scope prefill` and `scope decode` in the model. |
| **E009** | `INVALID_CONST_GENERIC_PARAM` | Const-Generics | Non-integer, negative, or dynamic generic dimension parameter. | Ensure all const-generic parameters are positive scalar integers (`usize`). |
| **E010** | `TENSOR_MATMUL_DIM_MISMATCH` | Shape Verification | Inner contraction dimensions do not unify in matrix multiplication: $\text{Dim}(A, -1) \ne \text{Dim}(B, 0)$. | Align the projection weight matrix dimension with the input activation dimension. |
| **E011** | `SYMBOLIC_SHAPE_UNIFICATION_FAILURE` | Shape Verification | Symbolic dimension constraints cannot be satisfied across operator DAG. | Verify hidden dimension multipliers and head count divisibility. |
| **E012** | `KV_CACHE_BORROW_CONFLICT` | Lifetime & Aliasing | Two overlapping mutable borrows into the same KV cache slot within the same scope. | Ensure sequence IDs or slot coordinates are provably distinct (Axiom 1). |
| **E013** | `SAFETENSORS_KEY_NOT_FOUND` | Weight Binding | Declared weight key does not exist in the provided `.safetensors` file. | Verify the tensor key in the `.safetensors` metadata header. |
| **E014** | `SAFETENSORS_DTYPE_MISMATCH` | Weight Binding | Data type in `.safetensors` header differs from declared `TensorType` without explicit quantizer. | Match the tensor dtype in the declaration or declare block quantization parameters. |
| **E015** | `BLOCK_SIZE_NOT_POWER_OF_TWO` | KV Cache Architecture | `PagedKVCache` block size parameter is not a power of two ($BS \notin \{16, 32, 64\}$). | Set block size to a valid power of two ($16, 32, \text{or } 64$). |
| **E016** | `UNBOUNDED_STREAMING_WINDOW` | KV Cache Architecture | Declaring `StreamingSinkKVCache` with non-constant or zero sink/window parameters. | Provide positive constant integers for sink and window dimensions. |
| **E017** | `SPECULATIVE_TREE_DEPTH_EXCEEDED` | Speculative Decoding | Candidate tree verification depth exceeds declared `MAX_TREE_DEPTH`. | Increase `MAX_TREE_DEPTH` in `SpeculativeTreeBlock` or prune the draft tree. |
| **E018** | `UNABSORBED_MLA_DECOMPRESSION` | Performance Invariant | Materializing full $H \times D$ key/value tensors from a `LatentKVCache` into global HBM. | Use `absorb_mla_q` and `absorb_mla_v` with `mla_decode_absorbed`. |
| **E019** | `HOST_PINNED_BUFFER_OVERFLOW` | Memory Tier Violation | Writing more tokens into host-pinned ring buffer than allocated capacity. | Increase ring buffer capacity or advance consumer read pointer. |
| **E020** | `ASYNC_TMA_BARRIER_MISUSE` | Concurrency & Async | Issuing `tma_load_async` without an associated hardware `mbarrier` synchronization. | Pair `tma_load_async` with `mbarrier.wait()`. |
| **E021** | `NON_CONSTANT_LOOP_BOUND_IN_FIBER` | Invariant 1 Violation | Dynamic, unbounded loop inside token dispatch fiber without static trip-count bounds. | Replace dynamic bounds with const-bounded range expressions. |
| **E022** | `DUPLICATE_MODEL_MEMBER_NAME` | Grammar & AST | Declaring two weights, KV caches, or scopes with identical identifiers inside a model. | Rename the duplicate member to a unique identifier. |
| **E023** | `UNDEFINED_IDENTIFIER_REFERENCE` | Frontend Semantics | Referencing a variable or tensor identifier not present in the current lexical scope. | Declare the variable with `let` before use or check spelling. |
| **E024** | `REPRESENTATION_PRECISION_LOSS` | Quantization & Dtype | Implicit downcasting of floating-point precision without an explicit quantizer. | Add an explicit quantization cast (e.g. `to_fp8(x)`). |
| **E025** | `UNALIGNED_SCRATCHPAD_ACCESS` | Hardware Architecture | Memory access that violates 128-byte hardware transaction alignment on global VRAM. | Align tensor leading dimension to a multiple of 128 bytes. |

## 10.3 Concrete Diagnostic Error Walkthroughs

### Walkthrough 1: Diagnostic `E001` (Dynamic Heap Allocation Prohibited)
```
================================================================================
VELOX COMPILER ERROR [E001]: Dynamic Heap Allocation Prohibited
================================================================================
File: models/llama3_serve.vlx, Line 142, Column 17:
  141 | scope decode(token_id: u32, seq_id: usize) -> Tensor<[1, VOCAB], f16> {
> 142 |     let temp = malloc(1048576);
      |                ^^^^^^^^^^^^^^^
      |                Dynamic allocation call detected inside decode scope!

Explanation:
  Velox enforces Invariant 1: Zero Dynamic Heap Allocation During Inference.
  All memory must be statically planned via the Velox Static Memory Planner or allocated
  within designated @bump_sram / @paged_vram storage classes. Calls to heap
  allocators (malloc, cudaMalloc) are strictly prohibited.

Remediation:
  Declare a statically shaped local tensor instead:
    `let temp: @bump_sram Tensor<[1, 1024], f16>;`
================================================================================
```

### Walkthrough 2: Diagnostic `E010` (Tensor Matmul Dimension Mismatch)
```
================================================================================
VELOX COMPILER ERROR [E010]: Tensor Matrix Multiplication Dimension Mismatch
================================================================================
File: models/deepseek_v3.vlx, Line 88, Column 22:
  87 |     let x: Tensor<[1, 4096], f16> = get_hidden_state();
> 88 |     let out = x @ self.w_gate;
     |               ^   ^^^^^^^^^^^
     |               |   Inner contraction dimension: 2048
     |               Outer dimension: 4096

Dimension Conflict:
  LHS Tensor Shape:           [1, 4096] (Dimension -1 = 4096)
  RHS Tensor Shape:           [2048, 8192] (Dimension 0 = 2048)
  Condition Violated:         Dim(LHS, -1) == Dim(RHS, 0) [4096 != 2048]

Remediation:
  Verify model weight projection dimensions. Did you mean `self.w_up`
  which has shape `[4096, 8192]`?
================================================================================
```

### Walkthrough 3: Diagnostic `E006` (SRAM Reference Escapes Scope)
```
================================================================================
VELOX COMPILER ERROR [E006]: Shared Memory (SRAM) Reference Escapes Scope
================================================================================
File: kernels/attention.vlx, Line 64, Column 9:
  62 | fn compute_tile(...) -> &@bump_sram Tensor<[128, 64], f16> {
  63 |     let mut tile: @bump_sram Tensor<[128, 64], f16>;
> 64 |     return &tile;
     |            ^^^^^
     |            Reference to @bump_sram cannot be returned from function!

Explanation:
  Memory allocated in @bump_sram resides in thread-block Shared Memory (SMEM).
  Returning a reference to SMEM across a function or kernel boundary violates
  the SRAM Escape Invariant. SMEM is discarded at kernel exit.

Remediation:
  Pass the destination buffer as an argument or commit data to @paged_vram
  using `tma_store_async` before returning.
================================================================================
```

### Walkthrough 4: Diagnostic `E012` (KV Cache Borrow Conflict)
```
================================================================================
VELOX COMPILER ERROR [E012]: Key-Value Cache Borrow Conflict
================================================================================
File: runtime/scheduler.vlx, Line 112, Column 24:
  111 |     let mut slot_a = kv.borrow_slot_mut(seq_id = 4, slot_idx = 10);
> 112 |     let mut slot_b = kv.borrow_slot_mut(seq_id = 4, slot_idx = 10);
      |                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      |                        Conflicting mutable borrow of slot (seq=4, slot=10)!

Explanation:
  Velox enforces Axiom 1 (Disjoint Borrow Rule). Two mutable references to the
  identical sequence ID and slot coordinate cannot coexist simultaneously
  within the same execution block.

Remediation:
  Ensure slot indices are distinct or serialize mutations sequentially.
================================================================================
```

---

# 11. Appendix: Complete Reference Implementations

This appendix contains the complete source code for all five Velox reference models. Each implementation is self-contained, adheres strictly to the 81 ISO/IEC 14977 EBNF rules, and enforces compile-time VRAM budget feasibility.

---

## 11.1 TinyStories / SmolLM Reference Script (`examples/tinystories_smollm.vlx`)

```velox
// ============================================================================
// Velox Reference Model: SmolLM-135M / TinyStories Architecture
// File: examples/tinystories_smollm.vlx
// ============================================================================

target hardware {
    device: "nvidia_rtx_3050_laptop";
    vram_capacity: 4096 * MiB;
    compute_capability: 8.6;
    safety_factor: 0.95;
}

model SmolLM_135M<
    const B_MAX: usize = 8,
    const S_MAX: usize = 2048,
    const VOCAB: usize = 49152,
    const HIDDEN_DIM: usize = 576,
    const INTERMEDIATE_DIM: usize = 1536,
    const NUM_LAYERS: usize = 30,
    const NUM_HEADS: usize = 9,
    const NUM_KV_HEADS: usize = 3,
    const HEAD_DIM: usize = 64
> {
    // Model Weights
    @vram_immutable weight w_embed: Tensor<[VOCAB, HIDDEN_DIM], f16> = 
        safetensors("smollm-135m.safetensors", "model.embed_tokens.weight");

    @vram_immutable weight w_q: Tensor<[NUM_LAYERS, HIDDEN_DIM, 576], f16> = 
        safetensors("smollm-135m.safetensors", "model.layers.q_proj.weight");
    @vram_immutable weight w_k: Tensor<[NUM_LAYERS, HIDDEN_DIM, 192], f16> = 
        safetensors("smollm-135m.safetensors", "model.layers.k_proj.weight");
    @vram_immutable weight w_v: Tensor<[NUM_LAYERS, HIDDEN_DIM, 192], f16> = 
        safetensors("smollm-135m.safetensors", "model.layers.v_proj.weight");
    @vram_immutable weight w_out: Tensor<[NUM_LAYERS, 576, HIDDEN_DIM], f16> = 
        safetensors("smollm-135m.safetensors", "model.layers.o_proj.weight");

    @vram_immutable weight w_gate: Tensor<[NUM_LAYERS, HIDDEN_DIM, INTERMEDIATE_DIM], f16> = 
        safetensors("smollm-135m.safetensors", "model.layers.gate_proj.weight");
    @vram_immutable weight w_up: Tensor<[NUM_LAYERS, HIDDEN_DIM, INTERMEDIATE_DIM], f16> = 
        safetensors("smollm-135m.safetensors", "model.layers.up_proj.weight");
    @vram_immutable weight w_down: Tensor<[NUM_LAYERS, INTERMEDIATE_DIM, HIDDEN_DIM], f16> = 
        safetensors("smollm-135m.safetensors", "model.layers.down_proj.weight");

    @vram_immutable weight w_attn_norm: Tensor<[NUM_LAYERS, HIDDEN_DIM], f16> = 
        safetensors("smollm-135m.safetensors", "model.layers.input_layernorm.weight");
    @vram_immutable weight w_mlp_norm: Tensor<[NUM_LAYERS, HIDDEN_DIM], f16> = 
        safetensors("smollm-135m.safetensors", "model.layers.post_attention_layernorm.weight");
    @vram_immutable weight w_final_norm: Tensor<[HIDDEN_DIM], f16> = 
        safetensors("smollm-135m.safetensors", "model.norm.weight");
    @vram_immutable weight w_head: Tensor<[HIDDEN_DIM, VOCAB], f16> = 
        safetensors("smollm-135m.safetensors", "lm_head.weight");

    // Native Paged Key-Value Cache Pool
    let mut kv_cache: PagedKVCache<NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, 16, f16>;

    // Execution Pipeline: Prefill Phase
    scope prefill<const B: usize, const S: usize>(
        tokens: Tensor<[B, S], u32>
    ) -> Tensor<[B, 1], u32> {
        let mut hidden = embed(tokens, w_embed);

        for l in 0..NUM_LAYERS {
            let norm_x = rmsnorm(hidden, w_attn_norm[l]);
            let q = norm_x @ w_q[l];
            let k = norm_x @ w_k[l];
            let v = norm_x @ w_v[l];
            
            let attn_out = flash_prefill(q, k, v, &mut kv_cache, l);
            hidden += attn_out @ w_out[l];

            let mlp_norm_x = rmsnorm(hidden, w_mlp_norm[l]);
            let gate = mlp_norm_x @ w_gate[l];
            let up = mlp_norm_x @ w_up[l];
            let act = silu(gate) * up;
            hidden += act @ w_down[l];
        }

        let final_norm = rmsnorm(hidden, w_final_norm);
        let last_hidden = final_norm[:, S - 1, :];
        let logits = last_hidden @ w_head;
        return sample_argmax(logits);
    }

    // Execution Pipeline: Decode Phase
    scope decode<const B: usize>(
        token: Tensor<[B, 1], u32>,
        step: usize
    ) -> Tensor<[B, 1], u32> {
        let mut hidden = embed(token, w_embed);

        for l in 0..NUM_LAYERS {
            let norm_x = rmsnorm(hidden, w_attn_norm[l]);
            let q = norm_x @ w_q[l];
            let k = norm_x @ w_k[l];
            let v = norm_x @ w_v[l];

            let mut slot = kv_cache.borrow_slot_mut(0, step);
            slot.write(l, k, v);

            let attn_out = flash_decode(q, &kv_cache, l);
            hidden += attn_out @ w_out[l];

            let mlp_norm_x = rmsnorm(hidden, w_mlp_norm[l]);
            let gate = mlp_norm_x @ w_gate[l];
            let up = mlp_norm_x @ w_up[l];
            let act = silu(gate) * up;
            hidden += act @ w_down[l];
        }

        let final_norm = rmsnorm(hidden, w_final_norm);
        let logits = final_norm @ w_head;
        return sample_argmax(logits);
    }
}
```

---

## 11.2 Meta Llama-3-8B Reference Script (`examples/llama3_8b.vlx`)

```velox
// ============================================================================
// Velox Reference Model: Meta Llama-3-8B Architecture
// File: examples/llama3_8b.vlx
// ============================================================================

target hardware {
    device: "nvidia_a100_80gb";
    vram_capacity: 80 * GiB;
    compute_capability: 8.0;
    safety_factor: 0.95;
}

model Llama3_8B<
    const B_MAX: usize = 16,
    const S_MAX: usize = 4096,
    const VOCAB: usize = 128256,
    const HIDDEN_DIM: usize = 4096,
    const INTERMEDIATE_DIM: usize = 14336,
    const NUM_LAYERS: usize = 32,
    const NUM_HEADS: usize = 32,
    const NUM_KV_HEADS: usize = 8,
    const HEAD_DIM: usize = 128
> {
    // Model Weights (Safetensors Mmap)
    @vram_immutable weight w_embed: Tensor<[VOCAB, HIDDEN_DIM], f16> = 
        safetensors("llama-3-8b.safetensors", "model.embed_tokens.weight");

    @vram_immutable weight w_q: Tensor<[NUM_LAYERS, HIDDEN_DIM, 4096], f16> = 
        safetensors("llama-3-8b.safetensors", "model.layers.self_attn.q_proj.weight");
    @vram_immutable weight w_k: Tensor<[NUM_LAYERS, HIDDEN_DIM, 1024], f16> = 
        safetensors("llama-3-8b.safetensors", "model.layers.self_attn.k_proj.weight");
    @vram_immutable weight w_v: Tensor<[NUM_LAYERS, HIDDEN_DIM, 1024], f16> = 
        safetensors("llama-3-8b.safetensors", "model.layers.self_attn.v_proj.weight");
    @vram_immutable weight w_o: Tensor<[NUM_LAYERS, 4096, HIDDEN_DIM], f16> = 
        safetensors("llama-3-8b.safetensors", "model.layers.self_attn.o_proj.weight");

    @vram_immutable weight w_gate: Tensor<[NUM_LAYERS, HIDDEN_DIM, INTERMEDIATE_DIM], f16> = 
        safetensors("llama-3-8b.safetensors", "model.layers.mlp.gate_proj.weight");
    @vram_immutable weight w_up: Tensor<[NUM_LAYERS, HIDDEN_DIM, INTERMEDIATE_DIM], f16> = 
        safetensors("llama-3-8b.safetensors", "model.layers.mlp.up_proj.weight");
    @vram_immutable weight w_down: Tensor<[NUM_LAYERS, INTERMEDIATE_DIM, HIDDEN_DIM], f16> = 
        safetensors("llama-3-8b.safetensors", "model.layers.mlp.down_proj.weight");

    @vram_immutable weight w_input_norm: Tensor<[NUM_LAYERS, HIDDEN_DIM], f16> = 
        safetensors("llama-3-8b.safetensors", "model.layers.input_layernorm.weight");
    @vram_immutable weight w_post_norm: Tensor<[NUM_LAYERS, HIDDEN_DIM], f16> = 
        safetensors("llama-3-8b.safetensors", "model.layers.post_attention_layernorm.weight");
    @vram_immutable weight w_norm: Tensor<[HIDDEN_DIM], f16> = 
        safetensors("llama-3-8b.safetensors", "model.norm.weight");
    @vram_immutable weight w_lm_head: Tensor<[HIDDEN_DIM, VOCAB], f16> = 
        safetensors("llama-3-8b.safetensors", "lm_head.weight");

    // Native Paged Key-Value Cache Pool (2.0 MiB block granularity)
    let mut kv: PagedKVCache<NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, 16, f16>;

    // Execution Pipeline: Prefill Phase
    scope prefill<const B: usize, const S: usize>(
        tokens: Tensor<[B, S], u32>
    ) -> Tensor<[B, 1], u32> {
        let mut h = embed(tokens, w_embed);

        for l in 0..NUM_LAYERS {
            let norm_x = rmsnorm(h, w_input_norm[l], eps = 1e-5);
            let q = norm_x @ w_q[l];
            let k = norm_x @ w_k[l];
            let v = norm_x @ w_v[l];

            let attn_context = flash_prefill(q, k, v, &mut kv, l);
            h += attn_context @ w_o[l];

            let mlp_norm = rmsnorm(h, w_post_norm[l], eps = 1e-5);
            h += swiglu(mlp_norm, w_gate[l], w_up[l], w_down[l]);
        }

        let out_norm = rmsnorm(h, w_norm, eps = 1e-5);
        let final_logits = out_norm[:, S - 1, :] @ w_lm_head;
        return sample_argmax(final_logits);
    }

    // Execution Pipeline: Decode Phase
    scope decode<const B: usize>(
        next_token: Tensor<[B, 1], u32>,
        step: usize
    ) -> Tensor<[B, 1], u32> {
        let mut h = embed(next_token, w_embed);

        for l in 0..NUM_LAYERS {
            let norm_x = rmsnorm(h, w_input_norm[l], eps = 1e-5);
            let q = norm_x @ w_q[l];
            let k = norm_x @ w_k[l];
            let v = norm_x @ w_v[l];

            let mut slot = kv.borrow_slot_mut(0, step);
            slot.write(l, k, v);

            let attn_context = flash_decode(q, &kv, l);
            h += attn_context @ w_o[l];

            let mlp_norm = rmsnorm(h, w_post_norm[l], eps = 1e-5);
            h += swiglu(mlp_norm, w_gate[l], w_up[l], w_down[l]);
        }

        let out_norm = rmsnorm(h, w_norm, eps = 1e-5);
        let final_logits = out_norm @ w_lm_head;
        return sample_argmax(final_logits);
    }
}
```

---

## 11.3 Alibaba Qwen2.5-7B Reference Script (`examples/qwen2_5_7b.vlx`)

```velox
// ============================================================================
// Velox Reference Model: Alibaba Qwen2.5-7B Architecture
// File: examples/qwen2_5_7b.vlx
// ============================================================================

target hardware {
    device: "nvidia_rtx_4090";
    vram_capacity: 24 * GiB;
    compute_capability: 8.9;
    safety_factor: 0.95;
}

model Qwen2_5_7B<
    const B_MAX: usize = 8,
    const S_MAX: usize = 8192,
    const VOCAB: usize = 152064,
    const HIDDEN_DIM: usize = 3584,
    const INTERMEDIATE_DIM: usize = 18944,
    const NUM_LAYERS: usize = 28,
    const NUM_HEADS: usize = 28,
    const NUM_KV_HEADS: usize = 4,
    const HEAD_DIM: usize = 128
> {
    // Model Weights (Tied Word Embeddings)
    @vram_immutable weight w_embed: Tensor<[VOCAB, HIDDEN_DIM], f16> = 
        safetensors("qwen2.5-7b.safetensors", "model.embed_tokens.weight");

    // Attention Projections & Explicit Bias Tensors
    @vram_immutable weight w_q: Tensor<[NUM_LAYERS, HIDDEN_DIM, 3584], f16> = 
        safetensors("qwen2.5-7b.safetensors", "model.layers.self_attn.q_proj.weight");
    @vram_immutable weight b_q: Tensor<[NUM_LAYERS, 3584], f16> = 
        safetensors("qwen2.5-7b.safetensors", "model.layers.self_attn.q_proj.bias");
    @vram_immutable weight w_k: Tensor<[NUM_LAYERS, HIDDEN_DIM, 512], f16> = 
        safetensors("qwen2.5-7b.safetensors", "model.layers.self_attn.k_proj.weight");
    @vram_immutable weight b_k: Tensor<[NUM_LAYERS, 512], f16> = 
        safetensors("qwen2.5-7b.safetensors", "model.layers.self_attn.k_proj.bias");
    @vram_immutable weight w_v: Tensor<[NUM_LAYERS, HIDDEN_DIM, 512], f16> = 
        safetensors("qwen2.5-7b.safetensors", "model.layers.self_attn.v_proj.weight");
    @vram_immutable weight b_v: Tensor<[NUM_LAYERS, 512], f16> = 
        safetensors("qwen2.5-7b.safetensors", "model.layers.self_attn.v_proj.bias");
    @vram_immutable weight w_o: Tensor<[NUM_LAYERS, 3584, HIDDEN_DIM], f16> = 
        safetensors("qwen2.5-7b.safetensors", "model.layers.self_attn.o_proj.weight");

    // MLP Weights
    @vram_immutable weight w_gate: Tensor<[NUM_LAYERS, HIDDEN_DIM, INTERMEDIATE_DIM], f16> = 
        safetensors("qwen2.5-7b.safetensors", "model.layers.mlp.gate_proj.weight");
    @vram_immutable weight w_up: Tensor<[NUM_LAYERS, HIDDEN_DIM, INTERMEDIATE_DIM], f16> = 
        safetensors("qwen2.5-7b.safetensors", "model.layers.mlp.up_proj.weight");
    @vram_immutable weight w_down: Tensor<[NUM_LAYERS, INTERMEDIATE_DIM, HIDDEN_DIM], f16> = 
        safetensors("qwen2.5-7b.safetensors", "model.layers.mlp.down_proj.weight");

    // Layer Normalizations
    @vram_immutable weight w_input_norm: Tensor<[NUM_LAYERS, HIDDEN_DIM], f16> = 
        safetensors("qwen2.5-7b.safetensors", "model.layers.input_layernorm.weight");
    @vram_immutable weight w_post_norm: Tensor<[NUM_LAYERS, HIDDEN_DIM], f16> = 
        safetensors("qwen2.5-7b.safetensors", "model.layers.post_attention_layernorm.weight");
    @vram_immutable weight w_norm: Tensor<[HIDDEN_DIM], f16> = 
        safetensors("qwen2.5-7b.safetensors", "model.norm.weight");

    // Native Paged Key-Value Cache Pool
    let mut kv: PagedKVCache<NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, 16, f16>;

    // Execution Pipeline: Prefill Phase
    scope prefill<const B: usize, const S: usize>(
        tokens: Tensor<[B, S], u32>
    ) -> Tensor<[B, 1], u32> {
        let mut h = embed(tokens, w_embed);

        for l in 0..NUM_LAYERS {
            let norm_x = rmsnorm(h, w_input_norm[l], eps = 1e-6);
            let q = (norm_x @ w_q[l]) + b_q[l];
            let k = (norm_x @ w_k[l]) + b_k[l];
            let v = (norm_x @ w_v[l]) + b_v[l];

            let attn_context = flash_prefill(q, k, v, &mut kv, l);
            h += attn_context @ w_o[l];

            let mlp_norm = rmsnorm(h, w_post_norm[l], eps = 1e-6);
            h += swiglu(mlp_norm, w_gate[l], w_up[l], w_down[l]);
        }

        let out_norm = rmsnorm(h, w_norm, eps = 1e-6);
        // Tied embeddings: reuse w_embed as transposed projection head
        let logits = out_norm[:, S - 1, :] @ w_embed.T;
        return sample_argmax(logits);
    }

    // Execution Pipeline: Decode Phase
    scope decode<const B: usize>(
        next_token: Tensor<[B, 1], u32>,
        step: usize
    ) -> Tensor<[B, 1], u32> {
        let mut h = embed(next_token, w_embed);

        for l in 0..NUM_LAYERS {
            let norm_x = rmsnorm(h, w_input_norm[l], eps = 1e-6);
            let q = (norm_x @ w_q[l]) + b_q[l];
            let k = (norm_x @ w_k[l]) + b_k[l];
            let v = (norm_x @ w_v[l]) + b_v[l];

            let mut slot = kv.borrow_slot_mut(0, step);
            slot.write(l, k, v);

            let attn_context = flash_decode(q, &kv, l);
            h += attn_context @ w_o[l];

            let mlp_norm = rmsnorm(h, w_post_norm[l], eps = 1e-6);
            h += swiglu(mlp_norm, w_gate[l], w_up[l], w_down[l]);
        }

        let out_norm = rmsnorm(h, w_norm, eps = 1e-6);
        let logits = out_norm @ w_embed.T;
        return sample_argmax(logits);
    }
}
```

---

## 11.4 DeepSeek-V3 MLA Reference Script (`examples/deepseek_v3_mla.vlx`)

```velox
// ============================================================================
// Velox Reference Model: DeepSeek-V3 Multi-Head Latent Attention (MLA) & MoE
// File: examples/deepseek_v3_mla.vlx
// ============================================================================

target hardware {
    device: "nvidia_h100_sxm5_cluster";
    vram_capacity: 1280 * GiB;
    compute_capability: 9.0;
    safety_factor: 0.95;
}

model DeepSeek_V3_MLA<
    const B_MAX: usize = 32,
    const S_MAX: usize = 8192,
    const VOCAB: usize = 129280,
    const HIDDEN_DIM: usize = 7168,
    const NUM_LAYERS: usize = 61,
    const NUM_HEADS: usize = 128,
    const HEAD_DIM: usize = 128,
    const LATENT_KV_DIM: usize = 512,
    const ROPE_DIM: usize = 64,
    const Q_LATENT_DIM: usize = 1536,
    const NUM_ROUTED_EXPERTS: usize = 256,
    const NUM_SHARED_EXPERTS: usize = 1,
    const TOP_K: usize = 8,
    const EXPERT_INTERMEDIATE_DIM: usize = 2048
> {
    // Model Weights (Native FP8 with Matrix Absorption)
    @vram_immutable weight w_embed: Tensor<[VOCAB, HIDDEN_DIM], fp8_e4m3> = 
        safetensors("deepseek-v3.safetensors", "model.embed_tokens.weight");

    // MLA Projections
    @vram_immutable weight w_dq: Tensor<[NUM_LAYERS, HIDDEN_DIM, Q_LATENT_DIM], fp8_e4m3> = 
        safetensors("deepseek-v3.safetensors", "model.layers.self_attn.q_a_proj.weight");
    @vram_immutable weight w_uq: Tensor<[NUM_LAYERS, Q_LATENT_DIM, NUM_HEADS * HEAD_DIM], fp8_e4m3> = 
        safetensors("deepseek-v3.safetensors", "model.layers.self_attn.q_b_proj.weight");
    @vram_immutable weight w_qr: Tensor<[NUM_LAYERS, Q_LATENT_DIM, NUM_HEADS * ROPE_DIM], fp8_e4m3> = 
        safetensors("deepseek-v3.safetensors", "model.layers.self_attn.q_rope_proj.weight");

    @vram_immutable weight w_dkv: Tensor<[NUM_LAYERS, HIDDEN_DIM, LATENT_KV_DIM], fp8_e4m3> = 
        safetensors("deepseek-v3.safetensors", "model.layers.self_attn.kv_a_proj_with_mqa.weight");
    @vram_immutable weight w_kr: Tensor<[NUM_LAYERS, HIDDEN_DIM, ROPE_DIM], fp8_e4m3> = 
        safetensors("deepseek-v3.safetensors", "model.layers.self_attn.k_rope_proj.weight");
    @vram_immutable weight w_uk: Tensor<[NUM_LAYERS, LATENT_KV_DIM, NUM_HEADS * HEAD_DIM], fp8_e4m3> = 
        safetensors("deepseek-v3.safetensors", "model.layers.self_attn.k_b_proj.weight");
    @vram_immutable weight w_uv: Tensor<[NUM_LAYERS, LATENT_KV_DIM, NUM_HEADS * HEAD_DIM], fp8_e4m3> = 
        safetensors("deepseek-v3.safetensors", "model.layers.self_attn.v_b_proj.weight");
    @vram_immutable weight w_o: Tensor<[NUM_LAYERS, NUM_HEADS * HEAD_DIM, HIDDEN_DIM], fp8_e4m3> = 
        safetensors("deepseek-v3.safetensors", "model.layers.self_attn.o_proj.weight");

    // Pre-Computed Absorbed Weights (Offline Matrix Absorption eliminates full KV decompression)
    @vram_immutable weight w_absorbed_q: Tensor<[NUM_LAYERS, NUM_HEADS, LATENT_KV_DIM, HEAD_DIM], fp8_e4m3> = 
        absorb_mla_q(w_uq, w_uk);
    @vram_immutable weight w_absorbed_o: Tensor<[NUM_LAYERS, NUM_HEADS, HEAD_DIM, HIDDEN_DIM], fp8_e4m3> = 
        absorb_mla_v(w_o, w_uv);

    // MoE Routing Gate & Routed Expert Slices
    @vram_immutable weight w_moe_gate: Tensor<[NUM_LAYERS, HIDDEN_DIM, NUM_ROUTED_EXPERTS], f32> = 
        safetensors("deepseek-v3.safetensors", "model.layers.moe.gate.weight");
    @vram_immutable weight w_routed_experts: Tensor<[NUM_LAYERS, NUM_ROUTED_EXPERTS, HIDDEN_DIM, EXPERT_INTERMEDIATE_DIM], fp8_e4m3> = 
        safetensors("deepseek-v3.safetensors", "model.layers.moe.experts.weight");
    @vram_immutable weight w_shared_expert: Tensor<[NUM_LAYERS, HIDDEN_DIM, EXPERT_INTERMEDIATE_DIM * 2], fp8_e4m3> = 
        safetensors("deepseek-v3.safetensors", "model.layers.moe.shared_expert.weight");

    // Normalizations & Head
    @vram_immutable weight w_input_norm: Tensor<[NUM_LAYERS, HIDDEN_DIM], fp8_e4m3> = 
        safetensors("deepseek-v3.safetensors", "model.layers.input_layernorm.weight");
    @vram_immutable weight w_post_norm: Tensor<[NUM_LAYERS, HIDDEN_DIM], fp8_e4m3> = 
        safetensors("deepseek-v3.safetensors", "model.layers.post_attention_layernorm.weight");
    @vram_immutable weight w_norm: Tensor<[HIDDEN_DIM], fp8_e4m3> = 
        safetensors("deepseek-v3.safetensors", "model.norm.weight");
    @vram_immutable weight w_head: Tensor<[HIDDEN_DIM, VOCAB], fp8_e4m3> = 
        safetensors("deepseek-v3.safetensors", "lm_head.weight");

    // Native Compressed Latent Key-Value Cache Pool
    let mut kv_latent: LatentKVCache<NUM_LAYERS, LATENT_KV_DIM, ROPE_DIM, fp8_e4m3>;

    // Execution Pipeline: Prefill Phase
    scope prefill<const B: usize, const S: usize>(
        tokens: Tensor<[B, S], u32>
    ) -> Tensor<[B, 1], u32> {
        let mut h = embed(tokens, w_embed);

        for l in 0..NUM_LAYERS {
            let norm_x = rmsnorm(h, w_input_norm[l]);
            
            // MLA Latent Projections
            let c_kv = norm_x @ w_dkv[l];
            let k_r = norm_x @ w_kr[l];
            let c_q = norm_x @ w_dq[l];
            let q_c = c_q @ w_uq[l];
            let q_r = c_q @ w_qr[l];

            // Append compressed representation into Latent KV Pool
            kv_latent.append_latent(0, S, c_kv, k_r);

            // Compute attention
            let attn_context = flash_prefill(q_c, c_kv, c_kv, &mut kv_latent, l);
            h += attn_context @ w_o[l];

            // MoE Block Routing
            let mlp_norm = rmsnorm(h, w_post_norm[l]);
            let (top_indices, top_weights) = moe_gate_topk(mlp_norm, w_moe_gate[l]);
            
            // Dynamic capacity-bounded expert routing (gamma_cap = 2.0)
            let routed_out = moe_dispatch_topk(mlp_norm, top_indices, top_weights, w_routed_experts[l], capacity_factor = 2.0);
            let shared_out = swiglu(mlp_norm, w_shared_expert[l]);
            h += routed_out + shared_out;
        }

        let out_norm = rmsnorm(h, w_norm);
        let logits = out_norm[:, S - 1, :] @ w_head;
        return sample_argmax(logits);
    }

    // Execution Pipeline: Decode Phase with Offline Matrix Absorption
    scope decode<const B: usize>(
        next_token: Tensor<[B, 1], u32>,
        step: usize
    ) -> Tensor<[B, 1], u32> {
        let mut h = embed(next_token, w_embed);

        for l in 0..NUM_LAYERS {
            let norm_x = rmsnorm(h, w_input_norm[l]);

            // Decode projects directly into compressed latent space
            let c_kv = norm_x @ w_dkv[l];
            let k_r = norm_x @ w_kr[l];
            let c_q = norm_x @ w_dq[l];
            let q_r = c_q @ w_qr[l];

            let mut slot = kv_latent.borrow_slot_mut(0, step);
            slot.write(l, c_kv, k_r);

            // Absorbed WGMMA attention: keys and values are NEVER decompressed into HBM!
            let q_absorbed = c_q @ w_absorbed_q[l];
            let attn_out = mla_decode_absorbed(q_absorbed, q_rope, &kv_latent, l);
            h += attn_out @ w_absorbed_o[l];

            // MoE Dispatch
            let mlp_norm = rmsnorm(h, w_post_norm[l]);
            let (top_indices, top_weights) = moe_gate_topk(mlp_norm, w_moe_gate[l]);
            let routed_out = moe_dispatch_topk(mlp_norm, top_indices, top_weights, w_routed_experts[l], capacity_factor = 2.0);
            let shared_out = swiglu(mlp_norm, w_shared_expert[l]);
            h += routed_out + shared_out;
        }

        let out_norm = rmsnorm(h, w_norm);
        let logits = out_norm @ w_head;
        return sample_argmax(logits);
    }
}
```

---

## 11.5 Nanbeige-4.3-33B Reference Script (`examples/nanbeige_33b.vlx`)

```velox
// ============================================================================
// Velox Reference Model: Nanbeige-4.3-33B High-Throughput Dense Architecture
// File: examples/nanbeige_33b.vlx
// ============================================================================

target hardware {
    device: "nvidia_h100_node";
    vram_capacity: 320 * GiB;
    compute_capability: 9.0;
    safety_factor: 0.95;
}

model Nanbeige_33B<
    const B_MAX: usize = 4,
    const S_MAX: usize = 65536,
    const VOCAB: usize = 156992,
    const HIDDEN_DIM: usize = 6144,
    const INTERMEDIATE_DIM: usize = 16384,
    const NUM_LAYERS: usize = 60,
    const NUM_HEADS: usize = 48,
    const NUM_KV_HEADS: usize = 8,
    const HEAD_DIM: usize = 128,
    const TP: usize = 4
> {
    // Model Weights (Sharded across TP=4)
    @vram_immutable weight w_embed: Tensor<[VOCAB, HIDDEN_DIM], f16> = 
        safetensors("nanbeige-33b.safetensors", "model.embed_tokens.weight");

    @vram_immutable weight w_q: Tensor<[NUM_LAYERS, HIDDEN_DIM, 6144 / TP], f16> = 
        safetensors("nanbeige-33b.safetensors", "model.layers.self_attn.q_proj.weight");
    @vram_immutable weight w_k: Tensor<[NUM_LAYERS, HIDDEN_DIM, 1024 / TP], f16> = 
        safetensors("nanbeige-33b.safetensors", "model.layers.self_attn.k_proj.weight");
    @vram_immutable weight w_v: Tensor<[NUM_LAYERS, HIDDEN_DIM, 1024 / TP], f16> = 
        safetensors("nanbeige-33b.safetensors", "model.layers.self_attn.v_proj.weight");
    @vram_immutable weight w_o: Tensor<[NUM_LAYERS, 6144 / TP, HIDDEN_DIM], f16> = 
        safetensors("nanbeige-33b.safetensors", "model.layers.self_attn.o_proj.weight");

    @vram_immutable weight w_gate: Tensor<[NUM_LAYERS, HIDDEN_DIM, INTERMEDIATE_DIM / TP], f16> = 
        safetensors("nanbeige-33b.safetensors", "model.layers.mlp.gate_proj.weight");
    @vram_immutable weight w_up: Tensor<[NUM_LAYERS, HIDDEN_DIM, INTERMEDIATE_DIM / TP], f16> = 
        safetensors("nanbeige-33b.safetensors", "model.layers.mlp.up_proj.weight");
    @vram_immutable weight w_down: Tensor<[NUM_LAYERS, INTERMEDIATE_DIM / TP, HIDDEN_DIM], f16> = 
        safetensors("nanbeige-33b.safetensors", "model.layers.mlp.down_proj.weight");

    @vram_immutable weight w_input_norm: Tensor<[NUM_LAYERS, HIDDEN_DIM], f16> = 
        safetensors("nanbeige-33b.safetensors", "model.layers.input_layernorm.weight");
    @vram_immutable weight w_post_norm: Tensor<[NUM_LAYERS, HIDDEN_DIM], f16> = 
        safetensors("nanbeige-33b.safetensors", "model.layers.post_attention_layernorm.weight");
    @vram_immutable weight w_norm: Tensor<[HIDDEN_DIM], f16> = 
        safetensors("nanbeige-33b.safetensors", "model.norm.weight");
    @vram_immutable weight w_head: Tensor<[HIDDEN_DIM, VOCAB], f16> = 
        safetensors("nanbeige-33b.safetensors", "lm_head.weight");

    // Native Virtual Memory Key-Value Cache (Low-Level CUDA VMM 2MB Physical Pages)
    let mut kv_vmm: VirtualContiguousKV<NUM_LAYERS, NUM_KV_HEADS / TP, HEAD_DIM, S_MAX, f16>;

    // Speculative Tree Drafting State Block
    let mut tree_block: SpeculativeTreeBlock<64, 6, HEAD_DIM, f16>;

    // Pipeline 1: Full Sequence Prefill
    scope prefill<const B: usize, const S: usize>(
        tokens: Tensor<[B, S], u32>
    ) -> Tensor<[B, 1], u32> {
        let mut h = embed(tokens, w_embed);

        for l in 0..NUM_LAYERS {
            let norm_x = rmsnorm(h, w_input_norm[l]);
            let q = norm_x @ w_q[l];
            let k = norm_x @ w_k[l];
            let v = norm_x @ w_v[l];

            let attn_context = flash_prefill(q, k, v, &mut kv_vmm, l);
            h += attn_context @ w_o[l];

            let mlp_norm = rmsnorm(h, w_post_norm[l]);
            h += swiglu(mlp_norm, w_gate[l], w_up[l], w_down[l]);
        }

        let out_norm = rmsnorm(h, w_norm);
        let logits = out_norm[:, S - 1, :] @ w_head;
        return sample_argmax(logits);
    }

    // Pipeline 2: Interleaved Chunked Prefill (Sarathi-Serve)
    scope chunked_prefill<const CHUNK_SIZE: usize = 512, const B_DEC: usize = 16>(
        chunk_tokens: Tensor<[1, CHUNK_SIZE], u32>,
        decode_tokens: Tensor<[B_DEC, 1], u32>
    ) -> (Tensor<[1, 1], u32>, Tensor<[B_DEC, 1], u32>) {
        // Compute chunked prefill GEMM piggybacked with batch decode GEMV
        let h_chunk = embed(chunk_tokens, w_embed);
        let h_decode = embed(decode_tokens, w_embed);

        for l in 0..NUM_LAYERS {
            let norm_chunk = rmsnorm(h_chunk, w_input_norm[l]);
            let norm_decode = rmsnorm(h_decode, w_input_norm[l]);

            let q_c = norm_chunk @ w_q[l];
            let k_c = norm_chunk @ w_k[l];
            let v_c = norm_chunk @ w_v[l];

            let q_d = norm_decode @ w_q[l];

            // Fused kernel evaluates chunk prefill and decode attention simultaneously
            let (attn_c, attn_d) = flash_chunked_piggyback(q_c, k_c, v_c, q_d, &mut kv_vmm, l);
            h_chunk += attn_c @ w_o[l];
            h_decode += attn_d @ w_o[l];

            let mlp_c = rmsnorm(h_chunk, w_post_norm[l]);
            let mlp_d = rmsnorm(h_decode, w_post_norm[l]);
            h_chunk += swiglu(mlp_c, w_gate[l], w_up[l], w_down[l]);
            h_decode += swiglu(mlp_d, w_gate[l], w_up[l], w_down[l]);
        }

        let next_chunk_token = sample_argmax(rmsnorm(h_chunk, w_norm)[:, CHUNK_SIZE - 1, :] @ w_head);
        let next_dec_tokens = sample_argmax(rmsnorm(h_decode, w_norm) @ w_head);
        return (next_chunk_token, next_dec_tokens);
    }

    // Pipeline 3: Standard Single-Token Decode
    scope decode<const B: usize>(
        next_token: Tensor<[B, 1], u32>,
        step: usize
    ) -> Tensor<[B, 1], u32> {
        let mut h = embed(next_token, w_embed);

        for l in 0..NUM_LAYERS {
            let norm_x = rmsnorm(h, w_input_norm[l]);
            let q = norm_x @ w_q[l];
            let k = norm_x @ w_k[l];
            let v = norm_x @ w_v[l];

            let mut slot = kv_vmm.borrow_slot_mut(0, step);
            slot.write(l, k, v);

            let attn_context = flash_decode(q, &kv_vmm, l);
            h += attn_context @ w_o[l];

            let mlp_norm = rmsnorm(h, w_post_norm[l]);
            h += swiglu(mlp_norm, w_gate[l], w_up[l], w_down[l]);
        }

        let out_norm = rmsnorm(h, w_norm);
        let logits = out_norm @ w_head;
        return sample_argmax(logits);
    }

    // Pipeline 4: Speculative Tree Verification (SpecInfer / EAGLE-2)
    scope speculative_verify(
        seq_id: usize,
        tree: &mut SpeculativeTreeBlock<64, 6, HEAD_DIM, f16>
    ) -> SpeculativePath<6> {
        // Synthesize 2D Non-Causal Attention Mask from Candidate DAG
        let tree_mask = tree.synthesize_tree_mask();
        let candidate_tokens = tree.get_candidate_tokens();
        let mut h = embed(candidate_tokens, w_embed);

        for l in 0..NUM_LAYERS {
            let norm_x = rmsnorm(h, w_input_norm[l]);
            let q = norm_x @ w_q[l];
            let k = norm_x @ w_k[l];
            let v = norm_x @ w_v[l];

            // Evaluate non-causal tree attention across all candidate branches
            let attn_tree = flash_tree_attention(q, k, v, tree_mask, &kv_vmm, l);
            h += attn_tree @ w_o[l];

            let mlp_norm = rmsnorm(h, w_post_norm[l]);
            h += swiglu(mlp_norm, w_gate[l], w_up[l], w_down[l]);
        }

        let verification_logits = rmsnorm(h, w_norm) @ w_head;
        let accepted_path = tree.resolve_best_path(verification_logits);

        // Commit accepted candidate tokens into authoritative KV cache (O(L) path compaction)
        tree.commit_accepted_path(&mut kv_vmm, accepted_path);
        return accepted_path;
    }
}
```

---

*End of Velox Language Syntax & System API Specification (`VELOX-SPEC-2026-STEP3-SYNTAX-001`).*
