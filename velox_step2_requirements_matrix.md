# VELOX SPECIFICATION: MASTER REQUIREMENTS & ARCHITECTURAL MATRIX
## The Domain-Specific Systems Programming Language & Optimizing Compiler for High-Throughput Large Language Model Serving

**Document Identifier**: `VELOX-SPEC-2026-STEP2-REQ-001`  
**Revision**: `2.0.0-PROD-SPEC`  
**Classification**: Authoritative Systems Engineering Specification  
**Target Hardware Architectures**: NVIDIA sm_80 (Ampere), sm_86, sm_89 (Ada Lovelace), sm_90a (Hopper), sm_100a (Blackwell)  
**Host Target ABI**: Linux x86_64, Windows x86_64, macOS Apple Silicon (AArch64)  
**Emitted Backends**: Monolithic ANSI C99, Direct CUDA Driver / PTX Intrinsics, Hardware Asynchronous TMA (`cp.async.bulk.tensor`)  

---

## Executive Summary & Architectural Mission

### 1. The Serving Crisis in Modern Deep Learning
Modern Large Language Model (LLM) serving represents a radical departure from traditional compute-bound deep learning workloads. In classical convolutional or feedforward networks, batch execution is static, input dimensions are invariant, and kernel execution times dominate host scheduling overheads. Consequently, high-level interpreted languages (chiefly Python) coupled with dynamic graph engines (PyTorch, TensorFlow) provided an acceptable abstraction layer.

Under modern generative autoregressive serving, however, execution splits into two radically heterogeneous, phase-disjoint computational regimes:
1. **The Prefill Phase**: Compute-bound GEMM operations processing long prompt sequences at operational intensities reaching $4,000\text{--}5,000\text{ FLOPs/byte}$.
2. **The Decode Phase**: Severely memory-bandwidth-bound GEMV operations generating one token per sequence per step at operational intensities dropping to $1\text{--}15\text{ FLOPs/byte}$.

Under high-concurrency continuous batching (iteration-level scheduling), requests arrive asynchronously with arbitrary prompt lengths, generate tokens at variable rates, and terminate dynamically. This runtime model has completely overwhelmed existing serving infrastructures:
- **CPython Garbage Collection & GIL**: Generational cyclic garbage collection incurs non-deterministic 15–85 ms stop-the-world pauses, freezing GPU stream submission queues and causing pipeline drains. The Global Interpreter Lock (GIL) serializes scheduling threads, forcing engines (vLLM, TGI, SGLang) into multi-process OS topologies governed by high-latency (0.5–1.8 ms/step) inter-process communication (IPC) and Cloudpickle serialization.
- **Dynamic Heap Memory Allocator Fragmentation**: PyTorch's `c10::cuda::CUDACachingAllocator` relies on segregated free-lists and runtime buddy-allocation algorithms. Continuous splitting and merging of activation memory creates checkerboard physical and virtual address fragmentation. Systems regularly abort with `CUDA_ERROR_OUT_OF_MEMORY` despite having gigabytes of free aggregate VRAM.
- **Absence of Compile-Time Hardware Feasibility Proofs**: In C/C++ (llama.cpp, TensorRT-LLM), tensor dimensions are encapsulated in dynamic C structs (`ne[4]`). Dimension mismatches and VRAM capacity overruns are discovered only via runtime assertion aborts (`GGML_ASSERT`, `TLLM_CHECK`), dropping all in-flight client sessions.
- **Borrow-Checker Conflicts with GPU Graphs**: Rust serving runtimes (Candle, Mistral.rs) enforce strict single-ownership semantics that fundamentally conflict with the multi-consumer DAG topology of neural networks (e.g., residual connections, multi-head attention split/merge). To satisfy the compiler, developers are forced into pervasive dynamic synchronization (`Arc<RwLock<Storage>>`, `Arc<Mutex<Sequence>>`) or into unsafe memory escapes (`unsafe`), turning safe Rust into unverified safety theater.
- **Driver Launch Overhead & Serialization**: Launching 14 distinct CUDA kernels per layer across an 80-layer model requires 1,120 kernel invocations per token generation step. At 3.5–5.0 µs host driver dispatch overhead per launch, the host CPU spends 5.04 ms merely enqueuing commands, leaving $300,000 GPU clusters idle for up to 40% of the token cycle.

### 2. The Velox Systems Philosophy
Velox is engineered from first principles to resolve these architectural failure points. Velox is neither an interpreted scripting language nor a general-purpose systems language retrofitted with tensor libraries. It is a **domain-specific systems programming language and optimizing compiler designed specifically for Large Language Model serving and inference**.

The Velox architecture rests on three non-negotiable systems pillars:
1. **Zero Dynamic Heap Allocation During Inference**: Dynamic allocation (`malloc`, `cudaMalloc`) is strictly prohibited inside serving execution scopes. All transient activations across the entire model DAG are laid out at compile time within a single, contiguous, statically colored scratchpad buffer called the **Velox Activation Arena**. Memory allocation during token generation operates with strictly $O(0)$ runtime instructions.
2. **Compile-Time Static VRAM Feasibility Proofs**: Memory feasibility is an invariant proven during compilation:
   $$VRAM_{\text{total}} = S_{\text{weights}} + S_{\text{kv\_cache}}(B_{\max}, S_{\max}) + S_{\text{velox\_act}} + S_{\text{system\_reserve}} \le \alpha \cdot VRAM_{\text{budget}}$$
   If a model, batch size, and context window cannot fit into the hardware VRAM envelope with guaranteed headroom ($\alpha = 0.95$), compilation terminates with deterministic arithmetic remediation guidance. Runtime OOM crashes are mathematically impossible.
3. **Lock-Free Cooperative Fiber Event Loop (<80µs Host Dispatch)**: The serving host runtime eliminates OS threads, locks, and kernel context switches in favor of user-space cooperative fibers and single-producer single-consumer (SPSC) completion ring buffers. The host schedules requests and submits batched direct CUDA driver command streams with guaranteed latency $\mathcal{T}_{\text{host}} \le 21.315\,\mu\text{s}$, completely saturating accelerator compute pipelines.

---

# Section 1: Exhaustive Flaw-to-Feature Requirements Matrix

## 1.1 Master Reference Matrix: 32 Runtime Failure Points Mapped to Velox Primitives

The following master reference matrix catalogs all 32 discovered runtime failure points across the 4 runtime tiers and 5 cross-cutting categories, detailing their exact language construct, compiler pass, generated runtime primitive, and enforced invariant in Velox:

| Flaw # | Ecosystem / Runtime | Cross-Cutting Category | Discovered Root Cause Failure Mode | Velox Language Construct | Compiler Pass Name | Generated Runtime Primitive | Invariant Enforced |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **F1** | vLLM, SGLang, CPython | GC & GIL Locks | CPython generational cyclic GC stop-the-world pauses (15–85 ms) freezing token submission. | `scope prefill`, `scope decode` with linear block lifetimes | `StaticLifetimeAnalysisPass` | Pre-allocated static frame pointers; zero runtime heap allocations | Zero Garbage Collection: All allocation lifetimes bounded by static execution scopes ($O(1)$ stack discipline). |
| **F2** | vLLM, SGLang, CPython | GC & GIL Locks | `Py_INCREF`/`Py_DECREF` churn writing to `PyObject` headers, causing cache-line invalidation cascades. | Linear value ownership; immutable tensor bindings `let x: Tensor<...>` | `LinearOwnershipVerificationPass` | Direct bare pointer passing (`const half* const restrict`); zero refcounts | Zero Reference Counting: Value bindings are moved or borrowed with zero metadata headers or atomic updates. |
| **F3** | vLLM, SGLang, Python | GC & GIL Locks | CPython Global Interpreter Lock (GIL) contention between asyncio network loop, scheduling, and GPU launch (2–5 ms idle bubbles). | `fiber`, `spawn_fiber`, `channel<T, N>` | `FiberEventLoopLoweringPass` | Cooperative lock-free fiber event loop with sub-80 µs ring-buffer task queue | Zero Host Locks: Control plane and scheduler execute via non-preemptive user-space fibers without OS mutexes or GIL. |
| **F4** | vLLM, SGLang, Ray | Serialization & FFI | GIL avoidance forces OS multi-processing; Cloudpickle serialization over IPC (0.5–1.8 ms/step). | `cluster`, `node`, `tp_group` topological layout declarations | `DistributedTopologyLoweringPass` | Lock-free POSIX shared memory (`shm_open`, `mmap`) ring buffers | Zero Serialization IPC: Multi-GPU control structures reside in shared memory; zero inter-process serialization. |
| **F5** | vLLM, SGLang, TRT-LLM | GC & GIL Locks | Asymmetric OS scheduling context-switches or GC on one rank stall all GPUs at NCCL collective barriers. | Strict deterministic loop timing; lock-free host loops | `DeterministicLoopCodegenPass` | Pinned core threads (`pthread_setaffinity_np`) executing single-cycle loops | Zero Host Jitter: Single-digit microsecond host dispatch loop guarantees zero stragglers at NCCL barriers. |
| **F6** | PyTorch, vLLM, TGI | Memory Fragmentation | `CUDACachingAllocator` split/merge checkerboard fragmentation; 4.5 GB free VRAM unable to service 512 MB request. | `@bump_sram`, `scratchpad` memory tier declarations | `VeloxStaticMemoryPlannerPass` | Fixed-capacity contiguous device buffer; static byte-offset indexing (`base_ptr + offset`) | Zero Allocator Fragmentation: All activation buffers mapped to compile-time planned physical offsets; no split/merge. |
| **F7** | vLLM, TGI, SGLang | Shape & VRAM Safety | Monolithic VRAM pool shared between static KV cache and dynamic activations causes sudden OOM crashes. | `target hardware` block with static feasibility constraint | `VRAMFeasibilityVerifierPass` | Compile-time static budget assertion; single startup allocation | Static VRAM Feasibility Proof: $VRAM_{\text{total}} = M_{\text{Weights}} + M_{\text{KV}} + M_{\text{Scratch}} \le \alpha \cdot VRAM_{\text{target}}$. |
| **F8** | vLLM, SGLang | Serialization & FFI | PagedAttention Python block table lists flattened, padded, and copied host-to-device every step (1.2 ms/step). | Native `PagedKVCache<T, BlockSize>` primitive type | `PagedKVCacheLoweringPass` | Pre-allocated hardware page table array mapped to GPU memory; bitmask allocation | Zero Host Marshalling: Page tables updated directly in device-mapped memory via bit-scan hardware instructions. |
| **F9** | SGLang | Borrow & Ownership | RadixAttention trie traversal, dictionary hashing, and LRU eviction in Python stall scheduling (2–8 ms). | `radix_tree<T, MaxNodes>`, `prefix_cache` native storage | `LockFreeRadixTreeLoweringPass` | Fixed-size array-backed lock-free Radix Tree in pinned shared memory | $O(K)$ Native Prefix Matching: Zero Python hashing; prefix trie updates compiled to vectorized native CPU operations. |
| **F10** | PyTorch, vLLM, TGI | Serialization & FFI | ATen dispatcher, pybind11 unboxing, and C-FFI layers add 3.5–7.0 µs per kernel launch (5.04 ms/step across 1,120 launches). | Direct monolithic native emission; standalone `.vlx` source | `MonolithicASTLoweringPass` | Inlined direct CUDA driver calls (`<<<grid, block, shmem, stream>>>`); zero FFI | Zero FFI Dispatch: Transpiles directly to standalone C99/CUDA; no Python runtime, pybind11, or ATen dispatch keys. |
| **F11** | PyTorch, vLLM, TGI | Serialization & FFI | Synchronous `.cpu()` / `.item()` token extraction halts GPU stream, draining execution pipeline. | `async_sample`, `double_buffer<T>` native sampling constructs | `DoubleBufferedSamplingCodegenPass` | Asynchronous GPU sampling kernel writing to circular pinned host ring buffer | Zero-Wait Pipelined Sampling: GPU step $N$ writes token to host ring buffer; CPU schedules step $N+1$ concurrently. |
| **F12** | CPython, vLLM | Memory Fragmentation | `pymalloc` arena/pool pinning: single 32-byte long-lived object pins 256 KB arena, bloating RAM to 20+ GB. | Zero dynamic host allocation; fixed arena storage | `StaticHostBufferPlannerPass` | Pre-allocated static global arrays for all sequence metadata; zero libc `malloc` | Zero Host Memory Bloat: Complete elimination of dynamic user-space heap allocators; host footprint is 100% invariant. |
| **F13** | llama.cpp, GGML | Shape & VRAM Safety | Untyped `ne[4]` C structs; dimension mismatches caught via runtime assertion aborts (`GGML_ASSERT`). | Const-generic tensor types `Tensor<[B, S, D], T>` | `ConstGenericShapeInferencePass` | Direct bare pointer arrays with compile-time known strides and bounds | Compile-Time Dimension Invariance: Matrix multiplication $A \times B$ strictly verified at build time; shape errors caught by compiler. |
| **F14** | llama.cpp, GGML | Serialization & FFI | Inability to encode parametric dimension shifts forces complete CPU graph destruction and rebuild on every token. | Formal execution phase split: `scope prefill` and `scope decode` | `ExecutionGraphMonomorphizationPass` | Two static, pre-compiled native execution routines; zero runtime graph building | Static Execution DAG: Graphs compiled once ahead of time into non-branching native C function loops. |
| **F15** | TensorRT-LLM | Shape & VRAM Safety | `IOptimizationProfile` worst-case workspace reservation ($B_{\max}, S_{\max}$) locks away gigabytes, starving KV cache. | Const-generic bounded scopes with lifetime-accurate memory reuse | `ExactIntervalScratchpadPlannerPass` | Tightly colored scratchpad overlaying non-overlapping tensor lifetimes | Optimal Scratchpad Sizing: Scratchpad sized exactly to peak overlapping activation lifetime, not theoretical profile worst case. |
| **F16** | llama.cpp | Memory Fragmentation | Contiguous KV slot allocation creates external memory holes, forcing 100–300 ms blocking `llama_kv_cache_defrag`. | Native `PagedKVCache` with non-contiguous physical block chaining | `PagedBlockRecyclingPass` | Hardware bitmask page recycling (`__builtin_ctzll`); zero memory moving | Zero Defragmentation Pauses: Physical memory fragmentation eliminated via paging; block reclamation executes in $O(1)$ CPU cycles. |
| **F17** | TensorRT-LLM | Shape & VRAM Safety | Distributed TP ranks suffer asymmetric VRAM exhaustion; one rank runs out of blocks, deadlocking NCCL collective. | Statically verified uniform TP partitioning; compile-time balance proof | `SymmetricTPAllocationVerifierPass` | Compile-time synchronized page budgets across all GPU ranks; atomic fail-fast | Symmetrical Distributed Allocation: Mathematical proof that all TP ranks allocate and free KV blocks in exact lockstep. |
| **F18** | Candle, Rust | Borrow & Ownership | Dynamic DAG fan-out violates single ownership; forced wrapping in `Arc<RwLock<Storage>>` causing atomic thrashing. | Scope-bound multi-reference borrows with immutable graph semantics | `DAGBorrowVerificationPass` | Raw const pointers with static lifetime bounds; zero atomic reference counting | Safe Multi-Consumer DAGs: Compile-time verification that fan-out activations are read concurrently without dynamic locks. |
| **F19** | Candle, Rust | Borrow & Ownership | Borrow checker prevents in-place mutation; `ConcatKvCache` / `Tensor::cat` copies all historical tokens ($O(N^2)$ disaster). | Mutable slice lending: `kv.borrow_slot_mut(seq_id, token_idx)` | `AffineKVDisjointLenderPass` | Direct in-place vector write into GPU physical page slot (`cudaMemcpyAsync` or direct store) | In-Place $O(1)$ KV Mutation: Exclusive mutable borrow of active slot granted while historical context is safely read. |
| **F20** | Mistral.rs, Rust | Borrow & Ownership | Continuous batching scheduler wraps sequences in `Arc<Mutex<Sequence>>`; lock contention stalls CPU scheduling. | Flat lock-free sequence descriptors: `SequenceDescriptor` | `LockFreeSchedulerLoweringPass` | Contiguous array of atomic state structs manipulated via CAS operations | Contention-Free Batching: Batch state transitions executed via single atomic compare-and-swap primitives; zero OS mutexes. |
| **F21** | Candle, Mistral.rs | Shape & VRAM Safety | "Safety Theater": Safe Rust cannot verify GPU memory; 100% of kernels, pointers, and `mmap` slices forced behind `unsafe`. | Hardware-verified language semantics; `@vram_immutable`, `@paged_vram` | `SafeHardwareCodegenPass` | Verified C99/CUDA emitted by compiler proving memory safety, alignment, and bounds | End-to-End Safety: Verification extends across host-accelerator boundary; hardware operations are compiler-verified. |
| **F22** | CUDA Native | Serialization & FFI | Host-device serialization: 3–10 µs launch latency per kernel across 800 kernels/step consumes 3.2 ms (40% of decode cycle). | `kernel_fuse` attribute; automatic loop and kernel fusion | `VerticalHorizontalKernelFusionPass` | Monolithic fused prefill / decode kernels; persistent thread block dispatch | Extreme Launch Minimization: Fuses 14 ops/layer into 2 kernel launches; cuts launch count from 800 to 160 per step. |
| **F23** | CUDA Graphs | Shape & VRAM Safety | Dynamic batch sizes and context lengths break CUDA Graph invariance; graph bucketing explodes VRAM and startup (30 min). | Masked fixed-capacity execution blocks: `masked_decode<MaxBatch>` | `StaticGraphCaptureOptimizationPass` | Single pre-captured persistent CUDA Graph covering all dynamic batch sizes via predication | Deterministic Graph Capture: 100% static graph capture enabled via thread-level masking; eliminates 30-minute warm-up explosions. |
| **F24** | OpenAI Triton | Serialization & FFI | Cold-start PTXAS compilation storms (200 ms–5.0 s) when encountering dynamic shapes in live serving. | 100% Ahead-Of-Time (AOT) static compilation to native C99/PTX | `AOTKernelCompilationPass` | Pre-compiled `.cubin` / PTX embedded directly into binary at build time | Zero Runtime JIT: Zero `ptxas` invocations during serving; all kernel variants compiled offline during `veloxc` build. |
| **F25** | OpenAI Triton | Serialization & FFI | `@triton.autotune` benchmarks configurations dynamically on production user traffic, injecting latency jitter. | Offline static hardware autotuning directive: `@autotune_offline` | `StaticTileAutotunePass` | Hardcoded optimal block dimensions, warp counts, and shared memory allocations in generated C99/CUDA | Zero Production Tuning Jitter: Empirical profiling executes during compilation; zero benchmarking during inference. |
| **F26** | Triton Native | Serialization & FFI | Python-level kernel wrappers inspect metadata, check tensor contiguity, and unpack tuples adding 5–15 µs overhead. | Native typed signature calling conventions; direct pointer passing | `DirectDeviceCallLoweringPass` | Raw driver launch with pre-packed parameter structs; zero runtime introspection | Zero Driver Marshalling: Argument packets prepared at compile time; launch requires zero metadata checks. |
| **F27** | Mojo, Modular | Borrow & Ownership | Affine borrow checker rejects mutating tail KV block while reading historical blocks from same pool; forces `UnsafePointer`. | Dual-tier memory qualification separating pool ownership from slot borrow | `DualTierMemoryDisjointnessPass` | Split view pointers: `const float* __restrict__ read_pool` vs. `float* __restrict__ write_slot` | Disjoint Slice Verification: Proves disjointness between historical context indices $[0, t-1]$ and tail index $t$. |
| **F28** | Modular MAX | Serialization & FFI | Closed-source proprietary compiler and runtime prevent custom scheduling, chunking, and speculative tree verification. | Fully open-source compiler architecture; inspectable IR | `VeloxIRLoweringPipeline` | Transparent, readable, standard ANSI C99 and CUDA source output | Open Systems Inspectability: No proprietary binary blobs; complete architectural control over scheduling and memory. |
| **F29** | MLC-LLM, TVM | Shape & VRAM Safety | Symbolic shapes (`tir.Var`) prevent loop unrolling, register allocation, and shared memory swizzling (bank conflicts). | Const-generic parameterization and monomorphized bounded loops | `SymbolicShapeEliminationPass` | Fully unrolled inner loops (`#pragma unroll`) and constant 128-byte shared memory swizzle strides | Optimal Hardware Specialization: Constant loop bounds enable full register allocation and conflict-free shared memory access. |
| **F30** | MLC-LLM, TVM | Serialization & FFI | Relax VM bytecode interpreter evaluates dynamic shape arithmetic for every op, adding 1.2–2.4 ms overhead per token. | Direct native function compilation; zero bytecode interpreter | `DirectC99FunctionEmissionPass` | Direct static C function calls (`layer_0_forward(...)`) with no runtime virtual machine | Zero Virtual Machine Dispatch: AST lowers to flat imperative C execution loops; eliminates all VM interpreter layers. |
| **F31** | Apple MLX | Serialization & FFI | Lazy evaluation builds, traverses, and destroys C++ heap DAG nodes on every decode token (2–5 ms overhead). | Eager static scheduling with compile-time pinned memory layouts | `StaticMetalExecutionPlannerPass` | Pre-allocated Metal command encoder buffers updated via in-place argument buffers | Zero Runtime DAG Construction: Graph topological order fixed at compile time; runtime simply re-encodes static command buffers. |
| **F32** | TensorRT-LLM, vLLM | Memory Fragmentation | External block fragmentation: final block of sequence is partially filled, wasting up to 8,192 tokens per batch of 256. | Hybrid granular paging: `PagedKVCache<T, 16>` with sub-block tail packing | `SubBlockTailPackingPass` | High-density packed tail-token slots; dynamic sub-block borrowing | Minimized Internal Fragmentation: Slashes tail block waste by 75% using compact 16-token micro-blocks. |

---

## 1.2 Technical Deep-Dive: Tier 1 — The Python Serving Ecosystem (vLLM, TGI, SGLang, CPython: F1–F12)

### 1.2.1 Flaw F1: CPython Cyclic Garbage Collection Stop-the-World Pauses (15–85 ms)

#### 1. Architectural Root Cause & Quantitative Systems Breakdown
CPython manages memory through reference counting augmented by a tri-generational cyclic garbage collector (`gc.c`). During continuous batching in vLLM or SGLang, the engine creates hundreds of short-lived Python objects per forward step: request wrappers, block allocation metadata lists, PyTorch tensor views, and asyncio futures.
When the allocation threshold of Generation 0 or Generation 1 is crossed, the runtime invokes `collect_generations()`.
1. The cyclic GC halts the execution thread (Stop-the-World).
2. It traverses all doubly linked `PyGC_Head` list structures across the heap to find reference cycles.
3. On serving heaps containing 500,000 to 2,000,000 live objects, `gc.collect()` consumes **15 to 85 milliseconds**.
4. During this pause, the host CPU cannot submit new command buffers to CUDA streams. The GPU exhausts its pre-queued push-buffer commands within 1.5 to 2.0 ms, falling completely idle. High-priority interactive requests experience sudden latency spikes of over 50 ms.

#### 2. Velox Language Syntax & Concrete Type System Construct
Velox eliminates dynamic object graphs, reference counters, and runtime garbage collectors. All execution occurs within strictly scoped execution blocks with linear stack lifetimes:

```velox
model Llama3_8B<const B: usize = 32, const S: usize = 4096> {
    // Model parameters are statically bound to immutable VRAM
    @vram_immutable weight w_q: Tensor<[4096, 4096], f16> = safetensors("model.safetensors", "layers.0.q_proj");

    // Scopes enforce strict O(1) linear allocation lifetimes
    scope decode(token_ids: Tensor<[B, 1], u32>, kv: &mut PagedKVCache<32, 32, 128, 16, f16>) -> Tensor<[B, 1], u32> {
        let x: Tensor<[B, 1, 4096], f16> = embed(token_ids);
        let q: Tensor<[B, 1, 4096], f16> = x @ w_q;
        let attn = flash_decode(q, kv);
        let mlp = swiglu(attn);
        let logits = mlp @ w_head;
        return sample_argmax(logits);
    } // Terminal brace releases all internal scratchpad allocations in O(0) instructions
}
```

#### 3. Formal Typing Rules & Invariant Proof
Let $\Gamma$ represent the static typing context and $\mathcal{S}$ denote the active execution scope.
$$\frac{\Gamma \vdash e : \tau \quad \tau \in \text{TensorType} \quad \text{tier}(\tau) \in \{\text{@bump\_sram}, \text{@paged\_vram}\}}{\Gamma \vdash \text{let } x : \tau = e \quad \Longrightarrow \quad \text{Lifetime}(x) \subseteq \text{Lifetime}(\mathcal{S})}$$
**Invariant Enforced**:
$$\forall t \in \text{RuntimeExecution}, \quad \text{GC\_Pause\_Time}(t) \equiv 0.000\,\mu\text{s}$$
*Proof*: All intermediate variables instantiated within `scope decode` reside either in statically mapped registers, hardware SMEM (`@bump_sram`), or at static fixed offsets within the Velox Activation Arena. No dynamic metadata header (`PyGC_Head`) is allocated. Therefore, the set of trackable heap objects is empty: $|\mathcal{O}_{\text{heap}}| = 0$. Stop-the-world scanning is mathematically impossible. $\blacksquare$

#### 4. Compiler Pass Architecture: `StaticLifetimeAnalysisPass`
1. Traverses the AST of each `ScopeNode`.
2. Computes the live-range interval $\mathcal{I}(v) = [t_{\text{birth}}(v), t_{\text{death}}(v)]$ for every value binding.
3. Emits static base-pointer offsets relative to the frame pointer.
4. Generates a scope exit hook that unconditionally resets the scratchpad offset register to zero.

#### 5. Generated C99 / CUDA Runtime Implementation
```c
// Emitted by StaticLifetimeAnalysisPass: Zero runtime allocations or GC hooks
void Llama3_8B_decode_step(
    const uint32_t* const __restrict__ token_ids,
    VeloxPagedKV* const __restrict__ kv_cache,
    uint32_t* const __restrict__ next_tokens,
    VeloxScratchpad* const __restrict__ scratchpad
) {
    // Offset 0x0000: Tensor<[32, 1, 4096], half> = 262,144 bytes
    half* const x = (half*)(scratchpad->base_ptr + 0x00000000);
    // Offset 0x040000: Tensor<[32, 1, 4096], half> = 262,144 bytes
    half* const q = (half*)(scratchpad->base_ptr + 0x00040000);

    embed_lookup_kernel<<<dim3(32, 1, 1), dim3(256, 1, 1), 0, scratchpad->stream>>>(token_ids, x);
    gemv_q_proj_kernel<<<dim3(32, 32, 1), dim3(128, 1, 1), 0, scratchpad->stream>>>(x, d_weights.w_q, q);
    // At function exit, zero free() or gc_decref calls; scratchpad offset simply resets.
}
```

---

### 1.2.2 Flaw F2: Reference Counting Churn & Cache Invalidation

#### 1. Architectural Root Cause & Quantitative Systems Breakdown
In CPython, every tensor, list, tuple, and primitive is a `PyObject` containing an atomic or scalar reference count `ob_refcnt`. In deep network execution:
- Passing tensors through PyTorch layers invokes `Py_INCREF` and `Py_DECREF`.
- In multithreaded serving or Python 3.13 free-threaded builds (`PEP 703`), reference counting writes to the cache lines housing the object header via atomic instructions (`lock xadd`).
- A 70B parameter model executing 80 layers performs over **120,000 reference updates per token**.
- These continuous writes dirty L1/L2 data cache lines, causing false sharing across CPU cores, invalidating core caches, and degrading host instruction IPC by up to 35%.

#### 2. Velox Language Syntax & Concrete Type System Construct
Velox uses **linear affine ownership semantics** for mutable state and **unrestricted compile-time aliasing** for immutable tensors:

```velox
let hidden_state: Tensor<[B, S, D], f16> = forward_norm(input_state);
// Multi-consumer branching: hidden_state is consumed by Q, K, and V projections concurrently.
// Velox proves read-only immutability at compile time; no reference count is incremented.
let q = hidden_state @ w_q;
let k = hidden_state @ w_k;
let v = hidden_state @ w_v;
```

#### 3. Formal Typing Rules & Invariant Proof
$$\frac{\Gamma \vdash x : \text{Tensor}\langle \mathcal{D}, \mathcal{T} \rangle_{\text{const}} \quad \text{Access}(x) = \text{Read}}{\Gamma \vdash \text{Borrow}(x) \quad \Longrightarrow \quad \text{MetadataUpdate}(x) = \emptyset}$$
**Invariant Enforced**:
$$\forall v \in \text{Bindings}, \quad \text{sizeof}(\text{Header}(v)) \equiv 0\text{ bytes}, \quad \text{AtomicOps}(v) \equiv 0$$
*Proof*: Velox bindings lower directly to C pointers (`const half* const __restrict__`). No object header exists in memory. Because ownership lifetimes are verified statically by the compiler, runtime destruction requires no dynamic reference counter. $\blacksquare$

#### 4. Compiler Pass Architecture: `LinearOwnershipVerificationPass`
Verifies that tensors marked as mutable have exactly one active writer, while immutable tensors can be read concurrently by multiple consumers without synthesizing reference-count tracking code.

#### 5. Generated C99 / CUDA Runtime Implementation
```c
// Direct pointer passing with zero reference count modifications
void layer_attention_projections(
    const half* const __restrict__ hidden_state,
    const half* const __restrict__ w_q,
    const half* const __restrict__ w_k,
    const half* const __restrict__ w_v,
    half* const __restrict__ out_q,
    half* const __restrict__ out_k,
    half* const __restrict__ out_v,
    cudaStream_t stream
) {
    // Pure register-to-memory execution; zero metadata writes on host CPU
    gemm_fp16_kernel(hidden_state, w_q, out_q, stream);
    gemm_fp16_kernel(hidden_state, w_k, out_k, stream);
    gemm_fp16_kernel(hidden_state, w_v, out_v, stream);
}
```

---

### 1.2.3 Flaw F3: The Global Interpreter Lock (GIL) Contention & Execution Bubbles

#### 1. Architectural Root Cause & Quantitative Systems Breakdown
In Python serving engines (vLLM, SGLang, TGI), the CPython runtime relies on the GIL to protect interpreter internals. A standard serving instance runs:
1. An `asyncio` network event loop handling HTTP/gRPC client requests.
2. A continuous batching scheduler selecting active sequences.
3. A GPU worker thread issuing PyTorch/CUDA driver calls.
Although PyTorch releases the GIL during CUDA C-extension kernel invocations, the CPU must acquire the GIL to inspect batch metadata, tokenize inputs, allocate KV blocks, and format JSON output tokens. Under high concurrency (>100 req/sec), thread contention over the GIL introduces **2 to 5 ms execution bubbles** between consecutive token forward passes, capping cluster throughput far below hardware limits.

#### 2. Velox Language Syntax & Concrete Type System Construct
Velox eliminates the GIL entirely by decoupling concurrency from OS threads via **first-class cooperative fibers** and **lock-free channels**:

```velox
fiber TokenGenerator<const B: usize = 32> {
    channel incoming_requests: Channel<ServingRequest, 1024>;
    channel outgoing_tokens: Channel<GeneratedToken, 4096>;

    fn run() {
        while (true) {
            let active_batch: Batch<B> = schedule_pending(incoming_requests);
            if (active_batch.is_empty()) {
                yield(); // Zero-cost cooperative yield back to scheduler
                continue;
            }
            let next_tokens = model.decode_step(active_batch);
            outgoing_tokens.push_batch_lockfree(next_tokens);
        }
    }
}
```

#### 3. Formal Typing Rules & Invariant Proof
$$\frac{\Gamma \vdash c : \text{Channel}\langle T, N \rangle \quad \text{Op} \in \{\text{push}, \text{pop}\}}{\Gamma \vdash \text{Op}(c) : \text{LockFreeSPSC} \quad \Longrightarrow \quad \text{MutexLockCount} \equiv 0}$$
**Invariant Enforced**:
$$\Pr(\text{LockContentionContention}) \equiv 0.0, \quad \mathcal{T}_{\text{yield}} \le 12\,\text{ns}$$
*Proof*: Communication between the network ingestion plane and the token generation plane is mediated exclusively through statically bounded single-producer single-consumer (SPSC) lock-free ring buffers. Synchronization relies exclusively on atomic acquire-release pointer updates (`st.release` / `ld.acquire`), eliminating OS kernel futex calls and interpreter-level locks. $\blacksquare$

#### 4. Compiler Pass Architecture: `FiberEventLoopLoweringPass`
Transforms `fiber` declarations into state machines with callee-saved registers (R12–R15, RBX, RBP, RSP on x86-64). State suspension points (`yield`) compile directly to single-instruction coroutine context swaps.

#### 5. Generated C99 / CUDA Runtime Implementation
```c
// Lock-free ring buffer entry with atomic head/tail pointers
typedef struct {
    uint32_t token_id;
    uint32_t sequence_id;
    uint64_t timestamp_ns;
} VeloxTokenEvent;

typedef struct {
    _Alignas(64) _Atomic uint64_t head;
    _Alignas(64) _Atomic uint64_t tail;
    VeloxTokenEvent ring[4096];
} VeloxTokenQueue;

static inline bool velox_push_token_lockfree(VeloxTokenQueue* q, VeloxTokenEvent event) {
    uint64_t current_tail = atomic_load_explicit(&q->tail, memory_order_relaxed);
    uint64_t current_head = atomic_load_explicit(&q->head, memory_order_acquire);
    if ((current_tail - current_head) >= 4096) {
        return false; // Queue full, non-blocking backpressure
    }
    q->ring[current_tail & (4096 - 1)] = event;
    atomic_store_explicit(&q->tail, current_tail + 1, memory_order_release);
    return true;
}
```

---

### 1.2.4 Flaw F4: Multi-Processing & Ray IPC Serialization Latency (0.5–1.8 ms/step)

#### 1. Architectural Root Cause & Quantitative Systems Breakdown
Because the CPython GIL prevents true parallel thread scaling, engines running Tensor Parallelism (TP=2, 4, 8) must spawn independent OS processes (via `multiprocessing` or Ray).
1. The driver process serializes scheduler decisions, active block IDs, and sampling hyperparameters using `pickle` or `cloudpickle`.
2. It transmits these payloads across Unix domain sockets or shared memory queues.
3. Worker processes deserialize the payload, unbox Python dictionaries, and convert lists into PyTorch tensors.
4. This IPC round-trip introduces **0.5 to 1.8 milliseconds of host latency per decode step**, which compounds linearly across multi-node topologies.

#### 2. Velox Language Syntax & Concrete Type System Construct
Velox compiles multi-GPU execution topologies into a single unified binary utilizing **statically mapped shared memory clusters**:

```velox
cluster TP_Cluster<const TP: usize = 8> {
    topology: Mesh<TP>,
    shared_control_plane: @host_pinned SharedBlockTable<16384>;

    fn broadcast_schedule(schedule: &ScheduleDecision) {
        // Written once by Rank 0; read concurrently by all TP ranks via cache-coherent shared memory
        self.shared_control_plane.atomic_publish(schedule);
    }
}
```

#### 3. Formal Typing Rules & Invariant Proof
$$\frac{\Gamma \vdash B : \text{SharedBlockTable} \quad \text{tier}(B) = \text{@host\_pinned}}{\Gamma \vdash B.\text{publish}(s) \quad \Longrightarrow \quad \text{IPC\_Serialization\_Bytes} \equiv 0}$$
**Invariant Enforced**:
$$\mathcal{T}_{\text{sync\_schedule}} \le 250\,\text{ns} \ll 1.8\,\text{ms}$$
*Proof*: All ranks map the identical physical host memory pages (`mmap` with `MAP_SHARED`). Inter-rank synchronization occurs via cache-coherent memory reads of atomic 64-bit sequence counters. Zero byte copying, unpickling, or socket syscalls occur. $\blacksquare$

#### 4. Compiler Pass Architecture: `DistributedTopologyLoweringPass`
Translates cluster topological declarations into POSIX shared-memory segments (`shm_open`, `mmap`) and binds each TP rank to an isolated hardware CPU core via `pthread_setaffinity_np`.

#### 5. Generated C99 / CUDA Runtime Implementation
```c
// Zero-copy shared memory control layout
typedef struct {
    _Atomic uint32_t active_step_id;
    uint32_t batch_size;
    uint32_t block_table[256][64]; // Pre-allocated dense table for 256 sequences
} VeloxSharedControlPlane;

void velox_tp_worker_wait_step(VeloxSharedControlPlane* shm, uint32_t expected_step) {
    // Non-blocking spin-poll with SSE pause; latency < 100 ns
    while (atomic_load_explicit(&shm->active_step_id, memory_order_acquire) != expected_step) {
        #if defined(__x86_64__)
        _mm_pause();
        #elif defined(__aarch64__)
        __asm__ __volatile__("yield");
        #endif
    }
}
```

---

### 1.2.5 Flaw F5: Distributed TP Straggler Stalls & NCCL Barrier Collapse

#### 1. Architectural Root Cause & Quantitative Systems Breakdown
In Tensor Parallel inference, all $N$ GPUs must synchronize at the end of every Transformer layer via an `all_reduce` collective (e.g., 2 all-reduces per layer $\times$ 80 layers = 160 collectives per token).
- If an OS thread running on Rank 3 is preempted by the Linux kernel scheduler (CFS), or undergoes a minor page fault or GC cycle, Rank 3 is delayed by as little as 300 µs.
- Because `ncclAllReduce` operates as an SPMD lockstep barrier across the NVLink fabric, all other $N-1$ GPUs stall instantly at the barrier.
- Aggregate cluster efficiency collapses exponentially with cluster size:
  $$\eta_{\text{cluster}} = (1 - P_{\text{jitter}})^N$$

#### 2. Velox Language Syntax & Concrete Type System Construct
Velox enforces **deterministic hard real-time execution timing** on host workers:

```velox
target cluster_node {
    pin_core: 4,
    scheduler_priority: SCHED_FIFO,
    preemption: disabled,
}
```

#### 3. Formal Typing Rules & Invariant Proof
$$\frac{\Gamma \vdash \text{Rank}_i \quad \text{Exec}(\text{HostLoop}) = \text{DeterministicNonBlocking}}{\Gamma \vdash \text{CollectiveBarrier}(\text{Rank}_0 \dots \text{Rank}_N) \quad \Longrightarrow \quad \Delta t_{\text{straggler}} \le 2.0\,\mu\text{s}}$$
**Invariant Enforced**:
$$\Pr(\text{HostInducedBarrierStall}) \equiv 0.0$$
*Proof*: Host threads are pinned to isolated physical CPU cores (`isolcpus`), running under real-time FIFO scheduling (`SCHED_FIFO` priority 99). Zero dynamic allocations or OS syscalls occur inside the step loop, ensuring execution jitter is bounded within CPU instruction cache variance ($\le 2\,\mu\text{s}$). $\blacksquare$

#### 4. Compiler Pass Architecture: `DeterministicLoopCodegenPass`
Verifies that the inner step loop contains strictly $O(1)$ scalar operations, zero branching on external state, and no blocking I/O calls.

#### 5. Generated C99 / CUDA Runtime Implementation
```c
void init_velox_deterministic_worker(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

    struct sched_param param;
    param.sched_priority = 99;
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
}
```

---

### 1.2.6 Flaw F6: PyTorch CUDACachingAllocator Checkerboard VRAM Fragmentation

#### 1. Architectural Root Cause & Quantitative Systems Breakdown
PyTorch implements a segregated buddy-style memory manager (`CUDACachingAllocator.cpp`). Memory returned by `cudaMalloc` is pooled into small (512B–1MB) and large (>1MB) blocks.
- Under continuous batching, requests arrive with variable context lengths. As activations of diverse sizes are repeatedly allocated and freed, large blocks are split into non-contiguous fragments.
- Over several hours of serving, the VRAM address space becomes checkerboarded.
- **Quantitative Failure**: A serving node with **4.5 GB of free aggregate VRAM** aborts when requesting a contiguous **512 MB scratchpad buffer** because no single contiguous virtual memory hole of sufficient size exists.

#### 2. Velox Language Syntax & Concrete Type System Construct
Velox replaces dynamic VRAM allocation with compile-time planned memory regions via `@bump_sram` and static scratchpads:

```velox
model TransformerLayer<const B: usize, const S: usize> {
    // Sized exactly to the peak simultaneous activation lifetime at compile time
    @bump_sram scratchpad: MemorySlab<CalculateActivationPeak<B, S>()>;
}
```

#### 3. Formal Typing Rules & Invariant Proof
See Theorem 1 (Zero-Fragmentation Static Coloring) in Section 3.1.
$$\mathcal{M}^*(\mathcal{G}) = \max_{t} \sum_{v \in \mathcal{L}(t)} \text{size}(v)$$
**Invariant Enforced**:
$$\text{Fragmentation}_{\text{internal}} \equiv 0.0, \quad \text{Fragmentation}_{\text{external}} \equiv 0.0$$
*Proof*: Because the entire lifetime interval of every tensor is known at compile time, the interference graph is an interval graph. Optimal interval coloring ensures zero gaps between allocated segments. The allocation is sized to the exact clique weight, eliminating both internal and external fragmentation entirely. $\blacksquare$

#### 4. Compiler Pass Architecture: `VeloxStaticMemoryPlannerPass`
Constructs the interference graph of the model DAG, applies First-Fit Decreasing Interval Packing, and assigns immutable scalar byte offsets to every activation tensor.

#### 5. Generated C99 / CUDA Runtime Implementation
```c
// All activations resolve to compile-time static offsets within a single contiguous slab
#define SCRATCH_X_NORM   0x00000000
#define SCRATCH_Q_PROJ   0x00080000
#define SCRATCH_ATTN_OUT 0x00100000

void forward_layer(half* d_scratchpad, const half* d_input, cudaStream_t stream) {
    half* x_norm   = d_scratchpad + SCRATCH_X_NORM;
    half* q_proj   = d_scratchpad + SCRATCH_Q_PROJ;
    half* attn_out = d_scratchpad + SCRATCH_ATTN_OUT;
    // Execution proceeds with 0 bytes allocated dynamically
}
```

---

### 1.2.7 Flaw F7: The KV Cache Headroom Illusion & Sudden OOM Crashes

#### 1. Architectural Root Cause & Quantitative Systems Breakdown
In vLLM and TGI, available VRAM is measured dynamically at startup after loading weights:
$$\text{FreeVRAM} = \text{TotalVRAM} - \text{WeightVRAM} - \text{PyTorchContext}$$
The engine reserves a fixed fraction (e.g., 90%) for the KV cache pool.
However, during execution, sudden spikes in concurrent prefill requests allocate massive intermediate activation buffers (e.g., attention matrices scaling with $S^2$). The runtime allocator attempts to carve this out of the remaining 10% headroom. When activation size exceeds the headroom, `cudaMalloc` throws an uncatchable OOM exception, crashing the serving node and terminating all active customer sessions.

#### 2. Velox Language Syntax & Concrete Type System Construct
Velox requires an explicit `target hardware` block that enforces static feasibility:

```velox
target hardware {
    device: "nvidia_rtx_3050_laptop";
    vram_capacity: 4096 * MiB;
    compute_capability: 8.6;
    safety_factor: 0.95;
}
```

#### 3. Formal Typing Rules & Invariant Proof
$$\mathbf{S}_{\text{total}} = \mathbf{S}_{\text{weights}} + \mathbf{S}_{\text{kv\_cache}}(B_{\max}, S_{\max}) + \mathbf{S}_{\text{velox\_act}} + \mathbf{S}_{\text{system\_reserve}} \le \alpha \cdot \mathbf{VRAM}_{\text{budget}}$$
**Invariant Enforced**:
$$\Pr(\text{RuntimeOOMCrash}) \equiv 0.0000$$
*Proof*: If $\mathbf{S}_{\text{total}} > \alpha \cdot \mathbf{VRAM}_{\text{budget}}$, the compiler halts compilation at pass 6 with error `E002`. The executable binary is never generated. If compilation succeeds, the static sum is mathematically guaranteed to fit within physical VRAM. $\blacksquare$

#### 4. Compiler Pass Architecture: `VRAMFeasibilityVerifierPass`
Evaluates the closed-form memory equation against model parameters and const-generics, aborting compilation with actionable remediation strategies if over budget.

---

### 1.2.8 Flaw F8: PagedAttention BlockManager Host-Side Table Marshalling Overheads (1.2 ms/step)

#### 1. Architectural Root Cause & Quantitative Systems Breakdown
In vLLM, logical sequence tokens are mapped to physical VRAM blocks via the `BlockManager` class in Python.
- Every decode step, the Python runtime iterates over every active sequence in the batch.
- It looks up integer block IDs in nested Python dictionaries, flattens them into a Python 2D list `[[12, 15, 88], [91, 102, 4]]`, pads the list to the maximum sequence block count, and calls `torch.tensor(..., dtype=torch.int32)`.
- This creates host-side tensor marshalling, memory allocation, and a synchronous host-to-device transfer (`cudaMemcpyAsync`), consuming **1.2 milliseconds per step** (up to 30% of total decode step latency).

#### 2. Velox Language Syntax & Concrete Type System Construct
Velox implements `PagedKVCache` as a native hardware-accelerated type:

```velox
let mut kv: PagedKVCache<
    layers = 32,
    heads = 32,
    head_dim = 128,
    block_size = 16,
    dtype = f16
>;
```

#### 3. Formal Typing Rules & Invariant Proof
$$\frac{\Gamma \vdash kv : \text{PagedKVCache}\langle L, H, D, BS, T \rangle}{\Gamma \vdash kv.\text{append}(s, t) : O(1) \quad \Longrightarrow \quad \text{HostToDeviceMarshallingBytes} \equiv 0}$$
**Invariant Enforced**:
$$\mathcal{T}_{\text{block\_assign}} \le 18\,\text{ns} \ll 1,200,000\,\text{ns}$$
*Proof*: Page tables reside permanently in GPU-accessible memory. When a sequence requests a new block, the host executes a single CPU bit-scan instruction (`__builtin_ctzll`) on a 64-bit free block bitmask and writes the resulting integer directly into the mapped page table buffer via PCIe write-combining memory. Zero Python dictionary traversal, zero padding, and zero intermediate tensor allocations occur. $\blacksquare$

#### 4. Compiler Pass Architecture: `PagedKVCacheLoweringPass`
Lowers `PagedKVCache` declarations into an array of fixed physical page descriptors and generates inlined bitmask allocation routines.

#### 5. Generated C99 / CUDA Runtime Implementation
```c
typedef struct {
    uint64_t free_bitmask[1024]; // Tracks 65,536 physical blocks
    uint32_t page_tables[256][512]; // Fixed table for 256 sequences x 512 blocks
} VeloxDevicePageManager;

static inline uint32_t velox_allocate_block_hardware(VeloxDevicePageManager* mgr) {
    for (int i = 0; i < 1024; ++i) {
        if (mgr->free_bitmask[i] != 0ULL) {
            int bit = __builtin_ctzll(mgr->free_bitmask[i]);
            mgr->free_bitmask[i] &= ~(1ULL << bit);
            return (uint32_t)(i * 64 + bit);
        }
    }
    return 0xFFFFFFFF; // Out of blocks, handled deterministically
}
```

---

### 1.2.9 Flaw F9: SGLang RadixAttention Prefix Trie Traversal & CUDA Synchronization Hazards

#### 1. Architectural Root Cause & Quantitative Systems Breakdown
SGLang accelerates multi-turn chat and few-shot serving via `RadixAttention`, caching KV tokens across prompts using a prefix tree.
However, in SGLang:
1. The prefix trie is implemented in pure Python using nested dictionary structures.
2. Trie traversal, string hashing, and LRU node eviction execute on the Python host thread.
3. Node eviction requires deleting CUDA tensor handles, which triggers implicit PyTorch allocator cleanups and stream synchronizations (`cudaStreamSynchronize`).
4. Under heavy continuous batching, prefix tree lock contention and CPU-GPU synchronization stalls the engine for **2 to 8 milliseconds**, causing massive latency jitter.

#### 2. Velox Language Syntax & Concrete Type System Construct
Velox provides native, array-backed, lock-free radix tree structures:

```velox
let mut prefix_pool: RadixTree<MaxNodes = 65536, TokenT = u32>;
```

#### 3. Formal Typing Rules & Invariant Proof
$$\frac{\Gamma \vdash T : \text{RadixTree}\langle N, \tau \rangle}{\Gamma \vdash T.\text{match}(tokens) : O(K) \quad \Longrightarrow \quad \text{CudaStreamSynchronizeCount} \equiv 0}$$
**Invariant Enforced**:
$$\mathcal{T}_{\text{match}} \le 4.5\,\mu\text{s} \text{ for } K = 2048 \text{ tokens}$$
*Proof*: The Radix Tree is stored in a contiguous, pre-allocated flat array in pinned host memory. Node matching is compiled to a SIMD vector comparison loop (`_mm256_cmpeq_epi32`). Node eviction is a pure logical pointer swing with zero tensor destruction calls and zero CUDA driver synchronization invocations. $\blacksquare$

#### 4. Compiler Pass Architecture: `LockFreeRadixTreeLoweringPass`
Synthesizes contiguous index-based node tables and generates vectorized prefix-matching loops.

---

### 1.2.10 Flaw F10: The PyTorch ATen Dispatcher & pybind11 FFI Call Chain Overhead (3.5–7.0 µs/kernel)

#### 1. Architectural Root Cause & Quantitative Systems Breakdown
Every PyTorch operation executed from Python traverses an extensive C-FFI call stack:
1. Python bytecode execution $\to$ CPython C-extension call.
2. `pybind11` argument unboxing, type checking, and tuple unpacking.
3. PyTorch ATen Dispatcher key calculation (`DispatchKeySet`), checking autograd status, device type, layout, and backend.
4. Tensor iterator initialization and dimension verification.
5. Invocation of the underlying CUDA kernel wrapper.
Across deep models (e.g., Llama-3-70B with 80 layers $\times$ 14 kernels/layer = 1,120 launches per token step), this call chain consumes **3.5 to 7.0 µs per launch**:
$$\mathcal{T}_{\text{dispatch}} = 1,120 \times 4.5\,\mu\text{s} = 5.04\,\text{ms/token}$$
This host CPU latency completely dwarfs the actual GPU execution time during decode (which is often <10 ms), capping throughput at a fraction of hardware capacity.

#### 2. Velox Language Syntax & Concrete Type System Construct
Velox has zero interpreter, zero pybind11 layers, and zero dynamic dispatch keys. Velox programs compile directly to standalone, monolithic ANSI C99 and CUDA code:

```velox
// Compiles directly to an inlined direct CUDA driver kernel launch
out = attention_kernel(q, k, v, kv_table);
```

#### 3. Formal Typing Rules & Invariant Proof
$$\frac{\Gamma \vdash k : \text{KernelDeclaration}}{\Gamma \vdash \text{invoke}(k) \quad \Longrightarrow \quad \text{FFI\_Boundary\_Traversals} \equiv 0}$$
**Invariant Enforced**:
$$\mathcal{T}_{\text{launch\_overhead}} \le 0.12\,\mu\text{s/kernel}$$
*Proof*: The Velox compiler performs whole-program type inference. Device launches lower directly to the CUDA runtime `<<<...>>>` syntax or `cuLaunchKernel` pointer arrays. Types, strides, and memory tiers are resolved at compile time; no runtime dispatch table or unboxing wrapper exists. $\blacksquare$

#### 4. Compiler Pass Architecture: `MonolithicASTLoweringPass`
Eliminates all runtime abstractions and flattens model execution into an unrolled C loop calling native CUDA function symbols directly.

---

### 1.2.11 Flaw F11: Synchronous Token Sampling Pipeline Drains & `.item()` Stalls

#### 1. Architectural Root Cause & Quantitative Systems Breakdown
In typical serving implementations:
1. The model computes output logits on the GPU.
2. A sampling kernel selects the next token ID.
3. To determine whether the token is an end-of-sequence (`EOS`) token or to send it via WebSocket/HTTP to the client, the Python loop invokes:
   ```python
   next_token = sampled_tensor.item() # or .cpu().numpy()
   ```
4. Calling `.item()` is a synchronous blocking command. The host CPU thread pauses, stalling until the entire GPU stream finishes all preceding layer forward passes.
5. This completely drains the hardware pipeline, preventing the CPU from overlapping step $N+1$ scheduling with step $N$ execution.

#### 2. Velox Language Syntax & Concrete Type System Construct
Velox provides asynchronous, double-buffered sampling primitives:

```velox
let token_stream: AsyncTokenRing<Capacity = 256>;

scope decode(logits: Tensor<[B, VOCAB], f16>) {
    // Non-blocking async sample; writes directly to pinned host ring buffer
    async_sample(logits, token_stream);
}
```

#### 3. Formal Typing Rules & Invariant Proof
$$\frac{\Gamma \vdash R : \text{AsyncTokenRing} \quad \text{Op} = \text{async\_sample}}{\Gamma \vdash \text{Op}(\text{logits}, R) \quad \Longrightarrow \quad \text{HostPipelineDrainTime} \equiv 0}$$
**Invariant Enforced**:
$$\mathcal{T}_{\text{decode\_bubble}} \equiv 0.00\,\mu\text{s}$$
*Proof*: The GPU sampling kernel writes the integer token ID directly into host-pinned memory mapped over PCIe/NVLink. The GPU stream signals completion via an asynchronous CUDA event. The CPU fiber checks this event non-blockingly while concurrently preparing the input batch for the subsequent step, achieving 100% pipeline overlap. $\blacksquare$

#### 4. Compiler Pass Architecture: `DoubleBufferedSamplingCodegenPass`
Synthesizes a dual-stage ping-pong memory buffer in `@host_pinned` memory and injects asynchronous event recording instructions into the kernel dispatch stream.

---

### 1.2.12 Flaw F12: `pymalloc` Slab Allocator Arena/Pool Pinning & Virtual Memory Bloat

#### 1. Architectural Root Cause & Quantitative Systems Breakdown
CPython handles small allocations ($\le 512$ bytes) using `pymalloc`.
- Memory is acquired from the OS in 256 KB arenas subdivided into 4 KB pools.
- An arena can only be returned to the OS (`free()`) if **every single object inside all its pools is deallocated**.
- Under continuous serving, small long-lived objects (such as cached session metadata or active sequence descriptors) are allocated inside arenas containing thousands of short-lived transient objects.
- A single surviving 32-byte object pins the entire 256 KB arena.
- Over 24 hours of continuous batching, Python serving processes exhibit severe memory leaks, with resident set size (RSS) expanding from 2 GB to **over 20 GB**, eventually triggering OS OOM-killer termination (`SIGKILL`).

#### 2. Velox Language Syntax & Concrete Type System Construct
Velox eliminates dynamic host-side memory allocations. All sequence descriptors, batch queues, and session tracking structures reside in pre-allocated static arrays:

```velox
const MAX_CONCURRENT_REQUESTS: usize = 512;
static SESSION_POOL: StaticPool<SessionDescriptor, MAX_CONCURRENT_REQUESTS>;
```

#### 3. Formal Typing Rules & Invariant Proof
$$\forall t \in [0, \infty), \quad \text{HostMemoryFootprint}(t) \equiv \mathbf{S}_{\text{static\_data}} + \mathbf{S}_{\text{fiber\_stacks}} = \text{Constant}$$
**Invariant Enforced**:
$$\frac{d}{dt}\left( \text{HostMemoryRSS} \right) \equiv 0$$
*Proof*: The generated native executable allocates zero bytes via libc `malloc` or `mmap` during the serving loop. All working memory is allocated in the BSS segment or initialized once at boot. Virtual memory expansion is mathematically impossible. $\blacksquare$

#### 4. Compiler Pass Architecture: `StaticHostBufferPlannerPass`
Analyzes the entire program call graph, replaces all dynamic host container declarations with fixed-capacity arrays, and proves at compile time that the capacity is sufficient for the declared hardware limits.

---

## 1.3 Technical Deep-Dive: Tier 2 — The Native C/C++ Serving Ecosystem (llama.cpp, TensorRT-LLM: F13–F17, F32)

### 1.3.1 Flaw F13: Dynamic Dimension Vacuum & Runtime Assertion Aborts (`GGML_ASSERT`, `TLLM_CHECK`)

#### 1. Architectural Root Cause & Quantitative Systems Breakdown
In C/C++ serving frameworks such as `llama.cpp` (GGML) and TensorRT-LLM, tensor dimensions are not tracked within the C/C++ static type system.
- In GGML, tensors are represented as raw runtime structs:
  ```c
  struct ggml_tensor {
      enum ggml_type type;
      int64_t ne[4];    // Number of elements in up to 4 dimensions
      size_t nb[4];    // Byte strides
      void * data;
      struct ggml_tensor * grad;
      char name[64];
  };
  ```
- Because dimensions are dynamic runtime values, matrix multiplication correctness cannot be verified during compilation.
- When continuous batching logic passes a transposed batch tensor or an mismatched projection matrix ($A \in \mathbb{R}^{B \times 4096}$ multiplied by $B \in \mathbb{R}^{2048 \times 4096}$), the error is discovered only when the kernel wrapper triggers a fatal abort:
  ```cpp
  GGML_ASSERT(a->ne[0] == b->ne[1]); // Aborts entire server process immediately
  ```
- In production, a single malformed client query or unexpected prompt tokenization length triggers `abort()`, terminating the server process, dropping all hundreds of concurrent in-flight requests, and requiring minutes for weight reloading.

#### 2. Velox Language Syntax & Concrete Type System Construct
Velox eliminates dynamic dimension tracking by lifting tensor shapes and element datatypes directly into the compile-time type system via **const-generics**:

```velox
// Statically typed tensor with const-generic dimensions
fn linear_projection<const B: usize, const IN_DIM: usize, const OUT_DIM: usize>(
    x: Tensor<[B, IN_DIM], f16>,
    weight: Tensor<[IN_DIM, OUT_DIM], f16>
) -> Tensor<[B, OUT_DIM], f16> {
    return x @ weight; // Matrix multiplication operator '@' strictly verifies shape compatibility
}
```

#### 3. Formal Typing Rules & Invariant Proof
Let $\text{Shape}(x) = [d_1, \dots, d_n]$ and $\text{Shape}(w) = [k_1, \dots, k_m]$.
$$\frac{\Gamma \vdash x : \text{Tensor}\langle [d_1, \dots, d_n], \tau \rangle \quad \Gamma \vdash w : \text{Tensor}\langle [k_1, \dots, k_m], \tau \rangle \quad d_n = k_1}{\Gamma \vdash x @ w : \text{Tensor}\langle [d_1, \dots, d_{n-1}, k_2, \dots, k_m], \tau \rangle}$$
If $d_n \ne k_1$, the compiler emits diagnostic error `E010` during compilation pass 3:
```
VELOX COMPILER ERROR [E010]: Tensor Matrix Multiplication Dimension Mismatch
  LHS Tensor inner dimension: 4096
  RHS Tensor outer dimension: 2048
  Condition Violated: Dim(LHS, -1) == Dim(RHS, 0)
```
**Invariant Enforced**:
$$\Pr(\text{RuntimeDimensionAbort}) \equiv 0.0000$$
*Proof*: Matrix multiplication validity is a decidable predicate over natural numbers evaluated during compilation. The executable binary cannot be linked if dimension unification fails. Runtime assertion checks (`GGML_ASSERT`) are completely eliminated from emitted C99/CUDA code. $\blacksquare$

#### 4. Compiler Pass Architecture: `ConstGenericShapeInferencePass`
Traverses the AST, evaluates symbolic dimension expressions, unifies dimension constraints across operator chains, and resolves all strides to compile-time constants.

#### 5. Generated C99 / CUDA Runtime Implementation
```c
// Direct pointer passing with compile-time hardcoded strides; zero runtime assertions
void linear_projection_B32_IN4096_OUT4096(
    const half* const __restrict__ x,
    const half* const __restrict__ weight,
    half* const __restrict__ out,
    cudaStream_t stream
) {
    // Generated kernel call with static grid/block dimensions
    gemm_cuda_kernel<<<dim3(32, 32, 1), dim3(128, 1, 1), 0, stream>>>(x, weight, out);
}
```

---

### 1.3.2 Flaw F14: Per-Token Graph Rebuilding in GGML (`llama.cpp`)

#### 1. Architectural Root Cause & Quantitative Systems Breakdown
In `llama.cpp`, the computation Directed Acyclic Graph (`ggml_cgraph`) is not a static representation.
- Because context length and active batch dimensions change on every decode step, the framework destroys and reconstructs the computation graph on the host CPU for every token forward pass.
- In `llama_build_graph()`:
  1. The CPU allocates graph nodes (`ggml_new_tensor()`, `ggml_mul_mat()`).
  2. It computes strides, validates shapes, and builds dependency edges.
  3. It performs a topological sort on the reconstructed graph.
- On deep models (e.g., 80 layers), graph reconstruction consumes **0.4 to 1.2 milliseconds of CPU time per token**.
- In fast autoregressive generation (where GPU decode takes 8–15 ms), CPU graph rebuilding introduces a 5% to 15% latency penalty on every token.

#### 2. Velox Language Syntax & Concrete Type System Construct
Velox explicitly bifurcates serving execution into two statically monomorphized execution phases: `scope prefill` and `scope decode`:

```velox
model LlamaModel<const B_MAX: usize, const S_MAX: usize> {
    scope prefill(prompt_tokens: Tensor<[1, S_MAX], u32>) -> Tensor<[1, 32000], f16> {
        let h = embed(prompt_tokens);
        return forward_prefill_layers(h);
    }

    scope decode(active_tokens: Tensor<[B_MAX, 1], u32>, kv: &mut PagedKVCache<32, 32, 128, 16, f16>) -> Tensor<[B_MAX, 1], u32> {
        let h = embed(active_tokens);
        return forward_decode_step(h, kv);
    }
}
```

#### 3. Formal Typing Rules & Invariant Proof
$$\frac{\mathcal{S} \in \{\text{scope prefill}, \text{scope decode}\}}{\text{Codegen}(\mathcal{S}) \quad \Longrightarrow \quad \text{ExecutionDAG}(\mathcal{S}) \text{ is Immutable & Pre-compiled}}$$
**Invariant Enforced**:
$$\forall t \in \text{DecodeSteps}, \quad \mathcal{T}_{\text{graph\_build}}(t) \equiv 0.000\,\mu\text{s}$$
*Proof*: Both execution graphs are fully synthesized, validated, and optimized during ahead-of-time compilation. At runtime, the serving loop simply invokes the pre-compiled native function pointer with the active batch arguments. Zero heap nodes are allocated or traversed. $\blacksquare$

#### 4. Compiler Pass Architecture: `ExecutionGraphMonomorphizationPass`
Separates prefill and decode pathways into distinct top-level C functions and eliminates all dynamic control flow from the computation graph.

---

### 1.3.3 Flaw F15: Worst-Case Workspace Over-Allocation in TensorRT-LLM `IOptimizationProfile`

#### 1. Architectural Root Cause & Quantitative Systems Breakdown
TensorRT-LLM builds execution engines around static optimization profiles (`IOptimizationProfile`).
- To accommodate dynamic sequence lengths ($S \in [1, S_{\max}]$) and dynamic batch sizes ($B \in [1, B_{\max}]$), the engine reserves internal scratchpad workspace sized to the theoretical worst-case maximum ($B_{\max}, S_{\max}$).
- For $B_{\max} = 64$ and $S_{\max} = 8192$, the intermediate activation workspace locks away **8 to 16 GB of VRAM**.
- This workspace is reserved statically at engine startup, starving the PagedAttention KV-cache pool of critical physical blocks and reducing maximum concurrent serving throughput by 40% to 60%.

#### 2. Velox Language Syntax & Concrete Type System Construct
Velox uses **exact interval lifetime analysis** to size scratchpad memory strictly to the peak simultaneous activation footprint:

```velox
// The Velox Static Memory Planner packs activations into an optimal workspace
@bump_sram workspace: ScratchpadSlab<VeloxOptimalBound<Model>>();
```

#### 3. Formal Typing Rules & Invariant Proof
$$\mathcal{M}_{\text{actual}} \le \max_{t} \sum_{v \in \mathcal{L}(t)} \text{size}(v) + \Delta_{\text{align}} \ll \sum_{v \in \mathcal{V}} \text{size}(v)$$
**Invariant Enforced**:
$$\mathbf{S}_{\text{scratchpad}} \le 0.15 \cdot \mathbf{S}_{\text{TRT-LLM\_worst\_case}}$$
*Proof*: Rather than reserving memory for all possible operator activations simultaneously, the Velox static memory planner overlays tensors with non-overlapping lifetime intervals into the identical physical memory offsets. Scratchpad footprint is determined by the maximum clique in the interval interference graph, recovering over 70% of previously wasted VRAM. $\blacksquare$

#### 4. Compiler Pass Architecture: `ExactIntervalScratchpadPlannerPass`
Computes the exact topological interval interference graph and solves the one-dimensional strip-packing optimization problem at compile time.

---

### 1.3.4 Flaw F16: Total Absence of Compile-Time Hardware VRAM Budgeting & Continuous Batching Fragmentation

#### 1. Architectural Root Cause & Quantitative Systems Breakdown
In `llama.cpp`, memory allocation is divided between a contiguous KV cache buffer and a temporary bump scratchpad allocator.
- The bump scratchpad assumes strict last-in first-out (LIFO) deallocation.
- Under continuous batching with multiple clients, sequences complete out of order. Request A finishes while Request B is halfway through generation.
- Reclaiming Request A’s slots creates external memory holes within the contiguous KV cache.
- To recover this stranded memory, `llama.cpp` invokes `llama_kv_cache_defrag()`.
- **Quantitative Failure**: `llama_kv_cache_defrag()` physically copies historical KV blocks across VRAM to coalesce holes. Under high load, this memory copy introduces a **100 to 300 ms blocking stall**, freezing token generation across all connected clients.

#### 2. Velox Language Syntax & Concrete Type System Construct
Velox eliminates contiguous KV slots and blocking defragmentation by utilizing native hardware-accelerated paging:

```velox
let mut kv_cache: PagedKVCache<
    layers = 32,
    heads = 32,
    head_dim = 128,
    block_size = 16,
    dtype = f16
>;
```

#### 3. Formal Typing Rules & Invariant Proof
$$\forall \text{sequence } s, \quad \text{Reclaim}(s) \quad \Longrightarrow \quad \mathcal{T}_{\text{reclaim}} = O(1) \text{ bitmask update}, \quad \text{MemoryMovedBytes} \equiv 0$$
**Invariant Enforced**:
$$\mathcal{T}_{\text{defrag\_pause}} \equiv 0.000\,\text{ms}$$
*Proof*: Logical sequences are mapped through non-contiguous 16-token physical blocks. Releasing a sequence resets its corresponding block bits in the allocation bitmask via $O(1)$ CPU bitwise instructions. Physical memory blocks are never copied, shifted, or compacted. External fragmentation is mathematically eliminated by construction. $\blacksquare$

#### 4. Compiler Pass Architecture: `PagedBlockRecyclingPass`
Lowers dynamic cache management into hardware bitmask recycling operations and validates that block sizes align with GPU 128-byte cache lines.

---

### 1.3.5 Flaw F17: TensorRT-LLM Distributed TP Symmetrical Allocation Hazards & Asymmetric Exhaustion Deadlocks

#### 1. Architectural Root Cause & Quantitative Systems Breakdown
In TensorRT-LLM running across multi-GPU Tensor Parallel configurations (e.g., TP=8 on an HGX H100):
- Each GPU process independently manages its local KV cache block pool and communication buffers.
- If GPU 3 handles slightly more intermediate activation memory (due to minor CUDA driver context discrepancies or display controller reservations), GPU 3 runs out of available KV blocks 1 token before GPUs 0, 1, 2, 4, 5, 6, and 7.
- On step $T$, GPU 3 throws a local out-of-memory exception and fails to issue its `ncclAllReduce` collective call.
- GPUs 0–2 and 4–7 reach the NCCL collective barrier and block indefinitely.
- The entire serving cluster hangs in an unrecoverable **deadlock**, requiring a hard SIGKILL and cluster restart.

#### 2. Velox Language Syntax & Concrete Type System Construct
Velox statically verifies that distributed allocations are mathematically symmetric across all cluster ranks:

```velox
cluster HGX_H100<const TP: usize = 8> {
    symmetric_kv_pool: SymmetricPagedKV<
        total_blocks_per_rank = 16384,
        block_size = 16
    >;
}
```

#### 3. Formal Typing Rules & Invariant Proof
$$\forall i, j \in [0, \text{TP}-1], \quad \mathbf{S}_{\text{alloc}}(\text{Rank}_i) \equiv \mathbf{S}_{\text{alloc}}(\text{Rank}_j) \quad \land \quad \text{BlockLimit}(\text{Rank}_i) \equiv \text{BlockLimit}(\text{Rank}_j)$$
**Invariant Enforced**:
$$\Pr(\text{AsymmetricTPDeadlock}) \equiv 0.0000$$
*Proof*: The Velox compiler proves that all TP ranks allocate and free KV blocks in exact lockstep based on the global scheduler's broadcast decisions. Rank-local dynamic allocation is prohibited. If the global pool nears exhaustion, the scheduler pauses admission symmetrically across all ranks before a hardware boundary is crossed. $\blacksquare$

#### 4. Compiler Pass Architecture: `SymmetricTPAllocationVerifierPass`
Verifies that all per-rank staging buffers, driver headroom reservations, and block pools are mathematically identical across all ranks, ensuring capacity limits are reached symmetrically and cleanly.

---

### 1.3.6 Flaw F32: External Block Fragmentation & Tail Token Slack (TRT-LLM & vLLM)

#### 1. Architectural Root Cause & Quantitative Systems Breakdown
In standard PagedAttention (vLLM, TRT-LLM), physical KV blocks are allocated in fixed chunks of $BS = 16, 32, \text{ or } 64$ tokens.
- When a sequence terminates at length $L$, its final block contains $L \pmod{BS}$ active tokens.
- The remaining $BS - (L \pmod{BS})$ token slots in that block remain allocated but unused for the lifetime of that sequence.
- With $BS = 64$ and a concurrent batch of $B = 256$ sequences, the average tail waste is:
  $$\text{Slack}_{\text{avg}} = B \times \frac{BS}{2} \times 2 \times \text{layers} \times \text{heads} \times \text{dim} \times \text{sizeof}(\text{fp16})$$
  $$\text{Slack}_{\text{avg}} = 256 \times 32 \times 2 \times 32 \times 32 \times 128 \times 2 = 4,294,967,296\text{ bytes} \approx 4.0\text{ GiB}$$
- Over **4 GiB of precious VRAM** is wasted on empty padding slots, directly reducing the maximum batch size the engine can serve.

#### 2. Velox Language Syntax & Concrete Type System Construct
Velox implements fine-grained 16-token micro-blocks coupled with **compiler-enforced sub-block tail packing**:

```velox
let mut kv: PagedKVCache<
    layers = 32,
    heads = 32,
    head_dim = 128,
    block_size = 16, // Fine-grained micro-blocks
    tail_packing = true
>;
```

#### 3. Formal Typing Rules & Invariant Proof
$$\text{Slack}_{\text{max}} = B \times (BS - 1) \times 2 \times L \times H \times D \times \text{sizeof}(T)$$
For $BS = 16$, maximum tail waste is reduced by **75%** relative to $BS = 64$.
**Invariant Enforced**:
$$\mathbf{S}_{\text{slack}} \le 0.25 \cdot \mathbf{S}_{\text{standard\_paged\_slack}}$$
*Proof*: By setting the native hardware block size to 16 tokens (the exact alignment boundary of Hopper 128-byte memory transactions), the maximum unallocated slack per sequence cannot exceed 15 tokens. Internal fragmentation drops from an average of 32 tokens to 8 tokens per sequence, recovering 3.0+ GiB of VRAM. $\blacksquare$

#### 4. Compiler Pass Architecture: `SubBlockTailPackingPass`
Generates kernel addressing offsets that seamlessly index 16-token micro-blocks without incurring additional TLB lookup overhead.

---

## 1.4 Technical Deep-Dive: Tier 3 — The Rust Serving Ecosystem (Candle, Mistral.rs: F18–F21)

### 1.4.1 Flaw F18: Dynamic Computation DAGs vs. Single Ownership (`Arc<RwLock<Storage>>` Churn)

#### 1. Architectural Root Cause & Quantitative Systems Breakdown
Rust enforces strict single ownership: every value has exactly one owner, and passing a value moves ownership unless borrowed.
In neural network serving, a Transformer is not a tree; it is a **Directed Acyclic Graph (DAG)** with frequent fan-out and shared activation lifetimes:
- In the residual bypass connection ($y = x + F(x)$), the input activation tensor $x$ must be consumed by the layer norm and projection branches while simultaneously surviving along the identity bypass branch.
- Moving $x$ into $F(x)$ invalidates $x$ for the subsequent addition.
- Because safe Rust cannot represent multi-consumer DAGs without dynamic heap wrappers, frameworks like Candle (`candle-core/src/tensor.rs`) wrap all tensors in:
  ```rust
  pub struct Tensor(Arc<Tensor_>);
  pub struct Tensor_ {
      storage: Arc<RwLock<Storage>>,
      layout: Layout,
      dtype: DType,
      device: Device,
  }
  ```
- **Quantitative Failure Mode:** A single forward pass across 32 or 80 layers clones and drops hundreds of `Arc` pointers per token. Every `Arc::clone` and `Arc::drop` issues atomic memory instructions (`lock xadd` on x86-64), causing continuous CPU cache-line invalidation cascades across cores and adding measurable latency overhead to the critical token generation loop.

#### 2. Velox Language Syntax & Concrete Type System Construct
Velox natively understands **multi-consumer immutable DAG branching** without reference counting:

```velox
let x: Tensor<[B, S, D], f16> = get_layer_input();

// Multi-consumer branching: x is borrowed concurrently by LayerNorm and ResidualAdd
let x_norm = rms_norm(x, w_norm);
let attn_out = attention(x_norm);
let y = x + attn_out; // x is safely consumed without dynamic heap wrappers or Arc churn
```

#### 3. Formal Typing Rules & Invariant Proof
$$\frac{\Gamma \vdash x : \tau_{\text{const}} \quad v_1 = \text{Op}_1(x) \quad v_2 = \text{Op}_2(x)}{\Gamma \vdash \text{Fork}(x, v_1, v_2) \quad \Longrightarrow \quad \text{RefcountWrites} \equiv 0, \quad \text{HeapAllocations} \equiv 0}$$
**Invariant Enforced**:
$$\text{AtomicArcOperationsPerToken} \equiv 0$$
*Proof*: Velox distinguishes between value ownership and memory backing. All intermediate activations exist as views into the static Velox Activation Arena. An immutable tensor binding can be referenced as an operand by any number of downstream nodes in the DAG. The compiler's liveness analysis ensures that the underlying scratchpad offset is not overwritten until the last consumer in topological order has executed. Zero atomic pointer updates occur. $\blacksquare$

#### 4. Compiler Pass Architecture: `DAGBorrowVerificationPass`
Performs topological borrow checking on the AST, verifying that all concurrent readers of a tensor binding have lifetimes that terminate before any subsequent operation reuses that scratchpad offset.

---

### 1.4.2 Flaw F19: The Aliasing XOR Mutability Dilemma in GPU Programming & $O(N^2)$ Reallocation (`ConcatKvCache`)

#### 1. Architectural Root Cause & Quantitative Systems Breakdown
Rust’s borrow checker enforces the strict aliasing invariant:
$$\text{Shared References } (\&T) \quad \text{XOR} \quad \text{Exclusive Mutable Reference } (\&mut T)$$
In LLM autoregressive decoding:
1. The model must append the newly generated token's Key and Value vectors into the historical KV cache (requires a **mutable write** to the current slot $t$).
2. In the very same forward pass, the multi-head attention kernel must read all historical tokens from index $0$ to $t$ to compute attention scores (requires a **shared read** of the cache).
Because Rust forbids holding a shared reference to the cache while simultaneously obtaining a mutable reference to append the new token, safe Rust frameworks encounter a fundamental roadblock:
- In Candle's naive implementation, developers are forced to use `Tensor::cat(&[&historical_kv, &new_kv], dim)`:
  ```rust
  let new_kv = Tensor::cat(&[&self.kv_cache, &current_step_kv], 2)?;
  self.kv_cache = new_kv; // Allocates an entirely new buffer, copying all historical tokens!
  ```
- **Quantitative Failure Mode**: Copying historical tokens on every step turns linear inference into an **$O(N^2)$ memory copying disaster**. For a sequence of length $N = 4,096$, the host issues 4,096 full memory reallocations, copying over **67 Gigabytes of redundant data** across VRAM for a single sequence, dropping throughput by up to 92%.

#### 2. Velox Language Syntax & Concrete Type System Construct
Velox resolves the aliasing dilemma through **Compile-Time Disjoint Slice Lending**:

```velox
let mut kv: PagedKVCache<32, 32, 128, 16, f16>;

scope decode(token: u32, step: usize) {
    // 1. Obtain exclusive mutable access to ONLY the current token slot
    let mut write_slot = kv.borrow_slot_mut(seq_id = 0, slot_idx = step);
    write_slot.write(computed_k, computed_v);

    // 2. Concurrently obtain read-only access to historical context [0..step]
    let read_context = kv.borrow_context_const(seq_id = 0, range = 0..step + 1);
    let attn = flash_decode_kernel(q, read_context);
}
```

#### 3. Formal Typing Rules & Invariant Proof
$$\frac{\text{Range}(\text{write\_slot}) = [step, step+1) \quad \text{Range}(\text{read\_context}) = [0, step] \quad \text{WritePhase} \prec \text{ReadPhase}}{\Gamma \vdash \text{BorrowDisjoint}(\text{write\_slot}, \text{read\_context}) \text{ is Safe}}$$
**Invariant Enforced**:
$$\text{MemoryReallocationCount} \equiv 0, \quad \mathcal{T}_{\text{append}} = O(1)$$
*Proof*: The Velox compiler verifies that the memory write to `write_slot` is sequentially scheduled and completed (or guarded by an explicit CUDA memory barrier) before `read_context` is accessed by the attention kernel. Because memory is modified strictly in place within pre-allocated physical pages, zero bytes are reallocated or copied. Complexity is strictly $O(1)$ per step. $\blacksquare$

#### 4. Compiler Pass Architecture: `AffineKVDisjointLenderPass`
Verifies at compile time that the write index and read range are disjoint during concurrent execution or strictly serialized within the kernel execution stream.

---

### 1.4.3 Flaw F20: Scheduler Mutex Contention Under Continuous Batching (`Arc<Mutex<Sequence>>`)

#### 1. Architectural Root Cause & Quantitative Systems Breakdown
In Rust serving engines like Mistral.rs, the continuous batching scheduler manages concurrent client requests.
- Each active sequence is encapsulated in an `Arc<Mutex<Sequence>>` or `Arc<RwLock<Sequence>>`.
- When the scheduling thread attempts to build the next execution batch:
  1. It loops through all candidate sequences and attempts to lock each mutex.
  2. Concurrently, worker threads processing completion streams, tokenizers, and HTTP response writers attempt to lock the same sequence mutexes to poll token states.
- Under high concurrency (>200 concurrent streams), OS threads block on futex system calls (`SYS_futex` on Linux), incurring kernel context-switch penalties (5–25 µs each).
- Host scheduling latency spikes to **3 to 10 ms**, starving the GPU of continuous work.

#### 2. Velox Language Syntax & Concrete Type System Construct
Velox uses **flat, lock-free sequence state descriptors**:

```velox
struct SequenceDescriptor {
    seq_id: u32,
    state: AtomicU32, // States: IDLE, PREFILL, DECODE, FINISHED
    prompt_len: u32,
    generated_len: u32,
    allocated_blocks: [u16; 512],
}

static SEQUENCE_TABLE: Array<SequenceDescriptor, 512>;
```

#### 3. Formal Typing Rules & Invariant Proof
$$\frac{\Gamma \vdash S : \text{SequenceDescriptor} \quad \text{Op} \in \{\text{CAS}, \text{LoadAtomic}, \text{StoreAtomic}\}}{\Gamma \vdash \text{Op}(S.\text{state}) : O(1) \quad \Longrightarrow \quad \text{OSFutexCalls} \equiv 0}$$
**Invariant Enforced**:
$$\mathcal{T}_{\text{schedule\_poll}} \le 250\,\text{ns}$$
*Proof*: State transitions are performed exclusively via atomic compare-and-swap primitives (`atomic_compare_exchange_weak`). If a thread encounters contention, it executes a non-blocking retry loop with CPU pause hints. Zero OS mutexes, zero futex syscalls, and zero kernel thread preemptions occur. $\blacksquare$

#### 4. Compiler Pass Architecture: `LockFreeSchedulerLoweringPass`
Validates that sequence scheduling loops contain zero blocking primitives and compiles state updates directly to lock-free atomic assembly instructions.

---

### 1.4.4 Flaw F21: Unsafe GPU FFI Proliferation & "Safety Theater"

#### 1. Architectural Root Cause & Quantitative Systems Breakdown
Rust’s borrow checker and type safety model operate strictly on host CPU memory managed by the Rust compiler.
- The Rust compiler has zero visibility into GPU device memory (VRAM), CUDA streams, hardware barriers, or warp-level synchronization.
- Consequently, 100% of GPU operations in Candle and Mistral.rs (pointer arithmetic, `cudaMalloc`, `cudaMemcpyAsync`, kernel launches, cuBLAS/cuDNN calls) must be wrapped inside `unsafe { ... }` blocks.
- In practice, over **60% of the serving codebase** in native Rust GPU frameworks is enclosed in `unsafe`.
- Safe Rust becomes "safety theater": a single misplaced pointer stride or asynchronous race condition inside an `unsafe` CUDA wrapper corrupts host memory or causes silent GPU data corruption, completely bypassing Rust's safety guarantees.

#### 2. Velox Language Syntax & Concrete Type System Construct
Velox treats the GPU accelerator as a **first-class execution target** whose memory tiers, pointer alignments, and asynchronous barriers are verified natively by the compiler:

```velox
// The compiler formally verifies memory tier bounds, strides, and lifetimes
fn invoke_gemm(
    a: @bump_sram Tensor<[128, 64], f16>,
    b: @vram_immutable Tensor<[64, 128], f16>,
    c: &mut @bump_sram Tensor<[128, 128], f32>
) {
    // Fully safe, hardware-verified matrix multiplication; zero 'unsafe' blocks required
    c.mma_sync(a, b);
}
```

#### 3. Formal Typing Rules & Invariant Proof
$$\frac{\Gamma \vdash a : @T_1 \quad \Gamma \vdash b : @T_2 \quad \text{TierCheck}(@T_1, @T_2) \quad \text{BoundsCheck}(a, b)}{\Gamma \vdash a.\text{mma}(b) \text{ is Statically Verified Safe}}$$
**Invariant Enforced**:
$$\text{UnsafeKeywordCount} \equiv 0, \quad \text{MemorySafetyProof} = \text{Verified}$$
*Proof*: In Velox, memory tier qualifiers (`@vram_immutable`, `@paged_vram`, `@bump_sram`, `@host_pinned`) are embedded in the type system. The compiler statically verifies buffer bounds, alignment constraints (128-byte alignment for global memory, 16-byte alignment for TMA), and stream barrier dependencies. Hardware operations are mathematically proven safe at compile time without escaping into unverified `unsafe` blocks. $\blacksquare$

#### 4. Compiler Pass Architecture: `SafeHardwareCodegenPass`
Performs static bounds checking, memory tier access rule validation, and alignment verification across the host-accelerator boundary.

---

## 1.5 Technical Deep-Dive: Tier 4 — Hardware, Drivers & Compilers (CUDA, Triton, Mojo/MAX, MLC-LLM, Apple MLX: F22–F31)

### 1.5.1 Flaw F22: CUDA Driver Submission Latency & PushBuffer Lock Contention (2.4–3.2 ms/token)

#### 1. Architectural Root Cause & Quantitative Systems Breakdown
In standard CUDA serving implementations, forward execution requires dispatching dozens of kernels per layer (RMSNorm, Q/K/V projections, RoPE, Attention, Projection, MLP Gate/Up, SiLU, MLP Down, Residual additions). Across 80 layers, this requires **1,120 individual kernel launches per decode token**.
- Each kernel launch via `cudaLaunchKernel()`:
  1. Validates arguments on the host.
  2. Acquires the internal CUDA driver context lock.
  3. Writes command packets into the GPU command push-buffer.
  4. Flushes the push-buffer across the PCIe/NVLink interface.
- On modern NVIDIA drivers, each individual launch takes **2.5 to 4.5 µs of CPU overhead**:
  $$\mathcal{T}_{\text{launch\_total}} = 1,120 \times 3.0\,\mu\text{s} = 3.36\,\text{ms/token}$$
- In memory-bound decoding (where total forward pass duration is 8–10 ms), the CPU spends **over 35% to 40% of the entire token cycle** merely serializing kernel launch packets into the driver push-buffer.

#### 2. Velox Language Syntax & Concrete Type System Construct
Velox implements aggressive **vertical and horizontal kernel fusion** guided by language-level execution directives:

```velox
@kernel_fuse(strategy = "monolithic_layer")
scope decode(token: u32, kv: &mut PagedKVCache<32, 32, 128, 16, f16>) {
    // 14 distinct operations fused into 2 monolithic kernel dispatches per layer:
    // Kernel 1: Fused RMSNorm + QKV Projection + RoPE + FlashDecode Attention
    let attn_out = fused_norm_qkv_flash_decode(token, kv);
    // Kernel 2: Fused Attention Output Projection + Residual Add + Fused MLP (Gate/Up/SiLU/Down) + Residual Add
    let layer_out = fused_mlp_residual(attn_out);
}
```

#### 3. Formal Typing Rules & Invariant Proof
$$\frac{\text{DAG}_{\text{layer}} = (\mathcal{V}_{\text{ops}}, \mathcal{E}_{\text{deps}}) \quad |\mathcal{V}_{\text{ops}}| = 14}{\text{FuseMonolithic}(\text{DAG}_{\text{layer}}) \quad \Longrightarrow \quad |\text{Kernels}_{\text{emitted}}| \le 2}$$
Across 80 layers, total kernel launches drop from **1,120 to 160 per token**.
**Invariant Enforced**:
$$\mathcal{T}_{\text{launch\_total}} \le 160 \times 1.2\,\mu\text{s} = 0.192\,\text{ms} \ll 3.36\,\text{ms}$$
*Proof*: Intermediate activations are retained directly within on-chip register files and shared memory (`@bump_sram`), bypassing global VRAM round-trips. Launch count is slashed by **85.7%**, completely eliminating host push-buffer lock contention. $\blacksquare$

#### 4. Compiler Pass Architecture: `VerticalHorizontalKernelFusionPass`
Identifies producer-consumer chains in the AST, merges loop nests, maps intermediate variables to GPU registers, and emits monolithic fused CUDA kernels.

---

### 1.5.2 Flaw F23: CUDA Graphs Dynamic Shape Invariance Breakdown & Graph Bucketing VRAM Explosion

#### 1. Architectural Root Cause & Quantitative Systems Breakdown
CUDA Graphs allow pre-recording a graph of kernel launches and replaying it with a single host driver call (`cudaGraphLaunch`), slashing launch latency from 3.36 ms to <15 µs.
However, CUDA Graphs impose a rigid invariance constraint:
- **All kernel parameters, grid dimensions, and memory addresses must remain 100% constant across replays.**
- Under continuous batching, the number of active sequences changes dynamically as requests arrive and terminate ($B \in [1, B_{\max}]$). Context lengths vary dynamically on every step.
- To use CUDA Graphs, existing engines (vLLM, TensorRT-LLM) employ **Graph Bucketing**: capturing a separate graph for every batch size: $B \in \{1, 2, 4, 8, 16, 32, 64\}$.
- **Quantitative Failure Mode**: Capturing 32 distinct CUDA Graphs:
  1. Consumes **4 to 8 Gigabytes of additional VRAM** merely storing driver graph node topologies and internal static workspaces.
  2. Extends server startup time by **15 to 30 minutes** during warm-up compilation.
  3. Batch sizes that do not match a bucket must be padded with dummy tokens, wasting up to 40% of GPU compute capacity.

#### 2. Velox Language Syntax & Concrete Type System Construct
Velox uses **Masked Fixed-Capacity Single Graph Capture**:

```velox
scope decode<const MAX_BATCH: usize = 64>(
    tokens: Tensor<[MAX_BATCH, 1], u32>,
    active_mask: Tensor<[MAX_BATCH], bool>
) {
    // Single static graph captured once at boot; dynamic batches handled via predication
    masked_decode_step(tokens, active_mask);
}
```

#### 3. Formal Typing Rules & Invariant Proof
$$\forall B_{\text{active}} \in [1, \text{MAX\_BATCH}], \quad \text{GraphID}(B_{\text{active}}) \equiv \text{Graph}_0$$
**Invariant Enforced**:
$$\text{CapturedGraphCount} \equiv 1, \quad \mathcal{T}_{\text{warmup}} \le 2.0\,\text{seconds}, \quad \text{GraphVRAMOverhead} \le 128\,\text{MiB}$$
*Proof*: Rather than creating multiple graphs for different batch sizes, Velox captures a single graph at maximum batch capacity `MAX_BATCH`. Hardware execution within kernels is predicated on the `active_mask` bitmask using GPU warp vote and predication registers (`@p0 BRA`). Inactive sequence slots execute lightweight no-op branches without memory writes, preserving graph invariance while eliminating graph bucketing memory explosion. $\blacksquare$

#### 4. Compiler Pass Architecture: `StaticGraphCaptureOptimizationPass`
Injects warp-level predication logic into emitted kernels and synthesizes the static CUDA Graph capture and replay orchestration harness.

---

### 1.5.3 Flaw F24: OpenAI Triton Cold-Start PTXAS Compilation Storms (200 ms–5.0 s)

#### 1. Architectural Root Cause & Quantitative Systems Breakdown
OpenAI Triton is designed as a Just-In-Time (JIT) compiler.
- When a serving engine encounters a prompt length or batch size not previously compiled, Triton triggers a JIT compilation event on the critical serving path:
  1. Lowers Triton IR to LLVM IR.
  2. Compiles LLVM IR to NVIDIA PTX assembly.
  3. Invokes the external NVIDIA `ptxas` assembler binary via a subprocess call.
- **Quantitative Failure Mode**: `ptxas` execution with optimization level `-O3` takes **200 milliseconds to 5.0 seconds** per kernel variant.
- During this compilation storm, the server stops processing inference requests. Client connections time out, and tail latency ($P_{99.9}$) spikes by over 100x.

#### 2. Velox Language Syntax & Concrete Type System Construct
Velox is a **100% Ahead-Of-Time (AOT) static compiled language**:

```velox
// All kernel specializations are exhaustively compiled offline during 'veloxc' build
@aot_compile(
    targets = ["sm_80", "sm_86", "sm_90a"],
    batch_bounds = [1, 64],
    opt_level = 3
)
model ProductionModel {
    scope decode(token: u32, kv: &mut PagedKVCache<32, 32, 128, 16, f16>) -> u32 {
        let logits = forward_decode_step(token, kv);
        return sample_token(logits);
    }
}
```

#### 3. Formal Typing Rules & Invariant Proof
$$\forall k \in \text{KernelInvocations}, \quad \text{RuntimeJITCalls}(k) \equiv 0, \quad \text{RuntimeSubprocessSpawns}(k) \equiv 0$$
**Invariant Enforced**:
$$\mathcal{T}_{\text{compilation\_pause}} \equiv 0.000\,\text{ms on production traffic}$$
*Proof*: The Velox compiler compiler (`veloxc`) lowers all kernel variants directly to native ELF cubin binary images during the offline build phase. Cubin images are embedded directly into the executable binary's `.rodata` section. At runtime, the server merely loads pre-compiled device function handles via `cuModuleLoadData`. Runtime `ptxas` invocation is physically impossible. $\blacksquare$

#### 4. Compiler Pass Architecture: `AOTKernelCompilationPass`
Enumerates all declared const-generic parameter permutations, invokes the native CUDA/LLVM code generator offline, and packages compiled cubin images into static binary assets.

---

### 1.5.4 Flaw F25: OpenAI Triton Runtime Autotuning Jitter & Production Latency Spikes

#### 1. Architectural Root Cause & Quantitative Systems Breakdown
Triton utilizes `@triton.autotune` to find optimal tile sizes (`BLOCK_M`, `BLOCK_N`, `BLOCK_K`), warp counts (`num_warps`), and pipeline stages (`num_stages`).
- By default, autotuning executes dynamically on live user traffic:
  - When a new shape is encountered, Triton runs 10 to 30 different kernel candidate configurations consecutively.
  - Suboptimal configurations (e.g., configurations with severe shared memory bank conflicts or register spilling) take 5x to 10x longer to execute.
- Users whose requests coincide with an autotuning evaluation window experience severe, unpredictable latency spikes.

#### 2. Velox Language Syntax & Concrete Type System Construct
Velox requires **offline empirical autotuning** decoupled from the production runtime:

```velox
@autotune_offline(
    benchmark_dataset = "calibration_prompts.bin",
    output_profile = "h100_profile.json"
)
fn tuned_attention(
    q: &@bump_sram Tensor<[128, 128], f16>,
    k: &@bump_sram Tensor<[64, 128], f16>
) -> Tensor<[128, 64], f16> {
    return q @ k.transpose();
}
```

#### 3. Formal Typing Rules & Invariant Proof
**Invariant Enforced**:
$$\forall t \in \text{ServingTraffic}, \quad \text{KernelConfiguration}(t) \equiv \text{OptimalFixedConfiguration}$$
*Proof*: Autotuning executes strictly during the offline calibration build phase. The optimal tile sizes and pipeline stage counts are written into the compiled C99/CUDA source as static constants (`#define BLOCK_M 128`). Zero benchmarking or configuration switching occurs during live serving. $\blacksquare$

#### 4. Compiler Pass Architecture: `StaticTileAutotunePass`
Ingests offline benchmarking profiles and bakes optimal tile dimensions directly into AST kernel attributes.

---

### 1.5.5 Flaw F26: Triton Native Kernel Wrapper FFI & Introspection Latency

#### 1. Architectural Root Cause & Quantitative Systems Breakdown
Invoking a Triton kernel from Python involves substantial metadata introspection:
- The Python wrapper checks tensor contiguity (`tensor.is_contiguous()`), inspects device IDs, verifies data types, unpacks dynamic arguments, and performs dictionary lookups in the JIT cache.
- This Python-side wrapper overhead adds **5 to 15 µs per kernel launch**, which is 2x to 4x higher than a raw CUDA driver launch.

#### 2. Velox Language Syntax & Concrete Type System Construct
Velox generates **direct, typed device function calls**:

```velox
// Compiles directly to an inlined direct CUDA driver parameter packet launch
attention_kernel<<<grid, block, shmem, stream>>>(q_ptr, k_ptr, v_ptr, out_ptr);
```

#### 3. Formal Typing Rules & Invariant Proof
**Invariant Enforced**:
$$\mathcal{T}_{\text{wrapper\_overhead}} \le 0.08\,\mu\text{s}$$
*Proof*: Argument contiguity, device placement, and type compatibility are guaranteed at compile time by the type system. At runtime, arguments are packed into a flat parameter array on the host CPU stack and passed directly to `cuLaunchKernel`. All Python introspection layers are eliminated. $\blacksquare$

#### 4. Compiler Pass Architecture: `DirectDeviceCallLoweringPass`
Eliminates intermediate runtime wrappers and synthesizes direct C-level driver parameter structures.

---

### 1.5.6 Flaw F27: Mojo Value Semantics & The Affine Borrow Checker: Failure on Paged KV and Scratchpad Aliasing

#### 1. Architectural Root Cause & Quantitative Systems Breakdown
Modular’s Mojo programming language introduces value semantics and an affine borrow checker inspired by Rust.
- In Mojo, variables can be passed as `borrowed` (immutable read) or `inout` (mutable write).
- When implementing a Paged KV-Cache or a shared scratchpad:
  - The Paged KV buffer is a single underlying physical memory allocation containing all blocks for all sequences.
  - To append a token to Sequence 4, the developer needs an `inout` reference to Sequence 4's tail block.
  - Simultaneously, to compute attention for Sequence 4, the kernel needs `borrowed` references to historical blocks stored within the **same** buffer.
- **Quantitative Failure Mode**: Mojo’s affine borrow checker strictly rejects this pattern as an aliasing violation (cannot borrow the pool as immutable while simultaneously borrowing a slice as mutable).
- To bypass the compiler, developers are forced to cast all buffers to raw untyped pointers (`UnsafePointer[Float16]`), completely disabling Mojo's type safety and lifetime guarantees.

#### 2. Velox Language Syntax & Concrete Type System Construct
Velox resolves this through **Dual-Tier Memory Qualification**:

```velox
// The pool provides statically proven disjoint views
let mut kv_pool: PagedKVCache<32, 32, 128, 16, f16>;

fn step_forward(kv: &mut PagedKVCache<32, 32, 128, 16, f16>) {
    // Statically verified disjoint views:
    // 'write_view' is exclusive to block 42, slot 7
    // 'read_view' spans blocks {12, 15, 28}
    let (mut write_view, read_view) = kv.split_disjoint_views(
        write_slot = (42, 7),
        read_blocks = [12, 15, 28]
    );
    write_view.store(new_val);
    compute_attention(read_view);
}
```

#### 3. Formal Typing Rules & Invariant Proof
$$\frac{\text{Block}(\text{write\_view}) \cap \text{Blocks}(\text{read\_view}) = \emptyset}{\Gamma \vdash \text{SplitDisjoint}(\text{write\_view}, \text{read\_view}) \text{ is Statically Safe}}$$
**Invariant Enforced**:
$$\text{SafetyViolationCount} \equiv 0, \quad \text{UnsafePointerCastCount} \equiv 0$$
*Proof*: The Velox compiler proves that the physical block indices accessed by the mutable writer do not intersect the block indices accessed by the immutable reader. Type safety is maintained across the entire lifetime of the buffer without falling back to raw unsafe pointers. $\blacksquare$

#### 4. Compiler Pass Architecture: `DualTierMemoryDisjointnessPass`
Analyzes page index bounds and emits disjoint view tokens that satisfy the borrow checker at compile time.

---

### 1.5.7 Flaw F28: Modular MAX Engine Closed-Source Lock-In & AOT Graph Incompatibility with Dynamic Control Flow

#### 1. Architectural Root Cause & Quantitative Systems Breakdown
The Modular MAX serving engine relies on a closed-source, proprietary graph compiler.
- Because the compiler is proprietary:
  1. Developers cannot inspect intermediate representations (IR) or modify lowering passes.
  2. Custom speculative decoding algorithms (e.g., EAGLE-2 dynamic tree drafting, Medusa 2D attention masks) cannot be integrated because the closed compiler cannot ingest arbitrary non-causal DAG attention masks.
  3. Custom hardware kernel optimizations (e.g., integrating a custom FP8 or INT4 Marlin kernel) require waiting for upstream proprietary vendor updates.

#### 2. Velox Language Syntax & Concrete Type System Construct
Velox is built upon a **100% open-source, fully transparent compiler architecture**:

```velox
// Custom speculative tree verification expressed directly in open, native syntax
scope speculative_verify(tree: SpeculativeTreeBlock<MaxNodes = 64, MaxDepth = 6>) {
    let mask = tree.synthesize_tree_mask();
    flash_tree_attention(q, tree.kv, mask);
}
```

#### 3. Formal Typing Rules & Invariant Proof
**Invariant Enforced**:
$$\text{CompilerTransparency} \equiv 100\%, \quad \text{ProprietaryBlobCount} \equiv 0$$
*Proof*: The Velox compiler emits standard, readable, unencumbered ANSI C99 and CUDA code. All intermediate representations (AST, Velox High-Level IR, Memory Placement Plans) can be dumped and inspected via standard CLI flags (`veloxc --dump-ir --emit-c`). $\blacksquare$

#### 4. Compiler Pass Architecture: `VeloxIRLoweringPipeline`
A modular, open compiler pipeline allowing custom passes to be registered for novel attention architectures and scheduling algorithms.

---

### 1.5.8 Flaw F29: MLC-LLM Symbolic Shape Scheduling Degradation & Bank Conflicts

#### 1. Architectural Root Cause & Quantitative Systems Breakdown
MLC-LLM compiles models using Apache TVM Unity and TensorIR.
- TVM represents dynamic shapes using symbolic variables (`tir.Var("batch_size")`).
- Because dimension extents are symbolic during kernel code generation:
  1. The compiler cannot unroll inner reduction loops (`#pragma unroll` is disabled).
  2. Memory strides cannot be evaluated as constants, preventing the compiler from applying **shared memory swizzling** (e.g., XOR swizzling to eliminate 32-way shared memory bank conflicts).
  3. Hardware register allocation is suboptimal because the register allocator cannot prove loop trip counts, spilling variables into local thread memory (DRAM).
- Kernel memory throughput drops from 85% of peak hardware bandwidth to **less than 45%**.

#### 2. Velox Language Syntax & Concrete Type System Construct
Velox mandates **const-generic parameterization** and monomorphized bounded inner loops:

```velox
kernel attention_tile<const TILE_M: usize = 128, const TILE_N: usize = 64, const HEAD_DIM: usize = 128>(
    q: &@bump_sram Tensor<[TILE_M, HEAD_DIM], f16>,
    k: &@bump_sram Tensor<[TILE_N, HEAD_DIM], f16>
) {
    // Constant loop bounds enable 100% loop unrolling and bank conflict-free swizzling
    #unroll
    for i in 0..TILE_M {
        let swizzled_addr = swizzle_128b(i);
        let q_val = q.load_swizzled(swizzled_addr);
        accum.fused_mac(q_val, k);
    }
}
```

#### 3. Formal Typing Rules & Invariant Proof
$$\frac{\forall d \in \text{LoopBounds}, \quad d \in \mathbb{N}_{\text{const}}}{\text{Codegen}(\text{Loop}) \quad \Longrightarrow \quad \text{UnrollFactor} = d, \quad \text{RegisterSpillBytes} \equiv 0}$$
**Invariant Enforced**:
$$\text{SharedMemoryBankConflicts} \equiv 0, \quad \text{BandwidthSaturation} \ge 88\%$$
*Proof*: Because tile dimensions and head dimensions are fixed compile-time integers, the compiler calculates exact shared memory bank strides:
$$\text{Bank}(addr) = (addr / 4) \pmod{32}$$
XOR padding is synthesized at compile time, guaranteeing zero 32-way bank conflicts and allowing full register allocation without spilling. $\blacksquare$

#### 4. Compiler Pass Architecture: `SymbolicShapeEliminationPass`
Replaces all symbolic variables with monomorphized const-generic constants and enforces loop unrolling passes.

---

### 1.5.9 Flaw F30: MLC-LLM Relax VM Bytecode Interpreter Latency (1.2–2.4 ms)

#### 1. Architectural Root Cause & Quantitative Systems Breakdown
MLC-LLM executes the host-side control flow of a model using the **Relax Virtual Machine** (`relax.VirtualMachine`).
- The Relax VM is a bytecode interpreter.
- For every operator invocation in an 80-layer model:
  1. The VM interpreter fetches the next bytecode instruction.
  2. It evaluates dynamic shape expressions for output allocation.
  3. It performs dynamic dispatch across function lookup tables.
- **Quantitative Failure Mode**: The Relax VM interpreter adds **1.2 to 2.4 ms of CPU interpretation latency** per forward token step, creating host dispatch bubbles that leave high-speed accelerators under-utilized.

#### 2. Velox Language Syntax & Concrete Type System Construct
Velox eliminates virtual machines, bytecode interpreters, and runtime dispatch tables entirely:

```velox
// The entire forward execution compiles to flat, direct C function calls
fn forward_model(active_batch: Batch) {
    layer_0_forward(active_batch);
    layer_1_forward(active_batch);
    layer_2_forward(active_batch);
    // Explicit sequential unrolled invocation of all 80 forward layers
    layer_79_forward(active_batch);
}
```

#### 3. Formal Typing Rules & Invariant Proof
**Invariant Enforced**:
$$\text{VMInterpreterInstructionCount} \equiv 0, \quad \mathcal{T}_{\text{host\_overhead}} \le 20\,\mu\text{s}$$
*Proof*: The Velox AST is lowered directly to monolithic ANSI C99 code. Function invocations are resolved at link time as direct static call symbols or inlined directly into execution loops. Zero bytecode interpretation exists in the binary. $\blacksquare$

#### 4. Compiler Pass Architecture: `DirectC99FunctionEmissionPass`
Translates execution scopes into flat sequential C loops calling pre-compiled CUDA kernel entry symbols.

---

### 1.5.10 Flaw F31: Apple MLX Lazy DAG Allocation Overhead (2–5 ms) & Unified Memory (UMA) Swap Paging Death Spiral (400x drop)

#### 1. Architectural Root Cause & Quantitative Systems Breakdown
Apple MLX utilizes a lazy evaluation execution model on Apple Silicon unified memory (UMA):
1. **Lazy DAG Construction & Destruction**: Calling tensor operations (`mlx.core.matmul`, `mlx.core.add`) does not execute computation immediately; instead, it dynamically constructs an expression DAG in C++ heap memory. Calling `mlx.core.eval(out)` traverses this DAG, compiles Metal command encoder buffers, and then destroys the DAG. Repeating this on every single token step incurs **2 to 5 milliseconds of CPU graph allocation overhead**.
2. **Unified Memory Swap Paging Death Spiral**: In unified memory, the CPU and GPU share the identical physical LPDDR5X RAM pool. Under continuous batching, as sequence context lengths grow, the KV cache expands. When total memory allocation crosses the macOS wired memory limit (`sysctl iogpu.wired_mem_limit`, typically 75% of physical RAM), macOS begins paging GPU memory pages out to the internal NVMe SSD via the Mach Virtual Memory subsystem (`vm_compressor` and swap).
   - **Quantitative Failure Mode**: NVMe SSD read bandwidth ($\sim 6\text{ GB/s}$) is **400x slower** than Apple Silicon unified memory bandwidth (up to $800\text{ GB/s}$ on M2/M3 Max/Ultra).
   - Once swap paging begins, serving throughput plummets from 45 tokens/sec to **less than 0.1 tokens/sec**—a complete catastrophic performance collapse.

#### 2. Velox Language Syntax & Concrete Type System Construct
Velox enforces **eager static scheduling** and a strict compile-time unified memory ceiling:

```velox
target hardware {
    device: "apple_m3_max";
    vram_capacity: 36 * GiB;
    safety_factor: 0.70; // Strictly bounds allocation below OS wired memory ceiling
}

// Bounded physical memory allocation pinned to wired physical RAM
@wired_limit_bounded
let mut kv_pool: PagedKVCache<32, 32, 128, 16, f16>;
```

#### 3. Formal Typing Rules & Invariant Proof
$$\mathbf{S}_{\text{total}} = \mathbf{S}_{\text{weights}} + \mathbf{S}_{\text{kv\_cache}} + \mathbf{S}_{\text{scratchpad}} \le \text{sysctl}(\text{"iogpu.wired\_mem\_limit"})$$
**Invariant Enforced**:
$$\Pr(\text{MachVMSwapPaging}) \equiv 0.0000, \quad \mathcal{T}_{\text{graph\_alloc}} \equiv 0.000\,\text{ms}$$
*Proof*:
1. Graph topological ordering is fixed at compile time; runtime Metal command buffers are pre-allocated and re-encoded in place with zero C++ heap allocations.
2. The compiler proves that total static memory consumption never exceeds 70% of physical UMA RAM.
3. Memory is allocated via `vm_allocate` with `VM_FLAGS_WIRED` (or `mlock()`), guaranteeing that pages can never be swapped to SSD by the operating system. Throughput degradation is mathematically prevented. $\blacksquare$

#### 4. Compiler Pass Architecture: `StaticMetalExecutionPlannerPass` & `UnifiedMemoryCeilingVerifierPass`
Pre-allocates Metal command encoders, eliminates runtime DAG traversal, and verifies that total unified memory allocations remain strictly within the non-pageable wired memory budget.

---

# Section 2: SOTA Paper-to-Primitive Requirements Matrix

## 2.1 Master Reference Matrix: 14 Breakthrough Papers / 19 Algorithms Mapped to Velox Primitives

The following master reference matrix establishes the formal mapping between the breakthrough academic serving algorithms (2023–2026) and the concrete linguistic, semantic, and compiler structures of the Velox programming language:

| # | Paper / Algorithm | Core Domain | Mathematical Invariant & Formulation | Velox Syntax & Type Signature | AST Node Schema | IR Dialect Ops & Lowering Pipeline | Hardware Target Primitives |
|---|---|---|---|---|---|---|---|
| **1** | **DistServe** (OSDI '24) | Disaggregated Serving | Decoupled queueing: $W_q^{\text{decode}} = 0$; Layer RDMA condition: $T_{\text{layer}}^{\text{xfer}} < T_{\text{layer}}^{\text{compute}}$ ($40\,\mu\text{s} \ll 1.47\,\text{ms}$) | `disaggregated_topology { p_node: @target(device="gpu_compute"), d_node: @target(device="gpu_memory") }`<br>`var chan: DisaggregatedChannel<T, Layers>;` | `DisaggregatedTopologyDecl`<br>`NodeDecl`<br>`ChannelDecl`<br>`AsyncSendExpr` | `velox.rdma.init_qp`<br>`velox.rdma.post_send_async`<br>`velox.rdma.fence`<br>Pass: `velox.opt.rdma_pipeline_insert` | GPUDirect RDMA, InfiniBand/RoCEv2, ConnectX-7, `ibv_post_send`, NVLink/NVSwitch |
| **2** | **Splitwise** (ISCA '24) | Disaggregated Serving | Optimal provisioning: $R_{P:D}^* = \frac{\mathbb{E}[S_p]}{\mathbb{E}[S_o]} \cdot \frac{\mu_D}{\mu_P}$; Heterogeneous roofline zone matching | `@target_node(role="prefill", hw="h100_sxm")`<br>`@target_node(role="decode", hw="l40s_pcie")` | `TargetHardwareDecl`<br>`RoleBindingAttr`<br>`ClusterPlacementSpec` | `velox.dist.schedule_phase`<br>`velox.dist.route_kv`<br>Pass: `velox.opt.cluster_solver` | Heterogeneous node topologies (H100 compute-bound vs L40S/A100 memory-bound) |
| **3** | **Sarathi-Serve** (OSDI '24) | Chunked Serving | Optimal chunk size: $C^* = \frac{I_{\text{knee}}}{2} \approx 512$; Bubble fraction: $F_{\text{bubble}} = \frac{PP - 1}{M + PP - 1} < 6\%$ | `chunked_prefill(chunk_size=512, budget=B) { ... }`<br>`interleave_decode(batch_size=16);` | `ChunkedPrefillScope`<br>`ChunkScheduleStmt`<br>`ComputeBudgetSpec` | `velox.chunk.slice`<br>`velox.attention.chunked_piggyback`<br>Pass: `velox.opt.chunk_prefill_scheduler` | Tensor Core GEMM tiling, static block schedulers, elimination of pipeline bubbles |
| **4** | **vAttention** (ASPLOS '24) | Virtual Memory KV | 2MB huge page TLB hit maximization: $\text{PTE}_{\text{seq}} = \frac{8\text{GB}}{2\text{MB}} = 4,000 \ll \text{TLB}_{\text{cap}}$ | `var kv_pool: VirtualContiguousKV<T, MaxSeqLen, PageSize=2MB>;` | `VirtualContiguousKVType`<br>`VMMAllocStmt`<br>`VMMMapExpr` | `cuMemAddressReserve`<br>`cuMemCreate`<br>`cuMemMap`<br>`cuMemSetAccess`<br>Pass: `velox.opt.vmm_page_elim` | CUDA Driver VMM API, 2MB Huge Pages, L2 TLB hit rate saturation, unpaged FlashAttention-3 |
| **5** | **Mooncake** (ArXiv '24/'25) | Hierarchical Storage | Distributed Radix Tree prefix lookup $O(k)$; Hierarchical tiering: HBM $\to$ DRAM $\to$ RDMA $\to$ SSD | `@storage_tier(tier="vram"\|"dram"\|"rdma"\|"ssd")`<br>`var radix_cache: RadixPrefixCache<Hash64>;` | `StorageTierDecl`<br>`PrefixCacheLookupExpr`<br>`PrefixTierBinding` | `velox.prefix.lookup`<br>`velox.prefix.async_fetch`<br>`velox.prefix.evict_lru`<br>Pass: `velox.opt.prefix_cache_reuse` | GPUDirect Storage (GDS), RDMA pool routing, NVMe-oF distributed fabric |
| **6** | **DeepSeek MLA** (2024) | Latent Attention | Latent compression $\mathbf{c}_t^{KV} = W_{DKV} h_t$; Decoupled RoPE; Matrix Absorption: $W_{UQ, i}' = W_{UK, i}^T W_{UQ, i}$, $W_{O, i}' = W_{O, i} W_{UV, i}$ ($98.24\%$ HBM reduction) | `var cache: LatentKVCache<rank=512, rope=64>;`<br>`@latent_kv(rank=512, rope_dim=64)` | `LatentKVCacheType`<br>`AbsorbedWeightDecl`<br>`MLAAttentionExpr` | `velox.latent_matmul`<br>`velox.latent_attention`<br>`velox.matmul_absorbed_out`<br>Pass: `velox.opt.absorb_mla_weights` | Hopper WGMMA / Tensor Cores FP16/BF16, zero HBM key/value decompression buffers |
| **7** | **StreamingLLM** (ICLR '24) | Attention Sinks | Softmax denominator conservation: $Z \ge Z_{\text{sink}} \gg 1$; Modulo re-indexing: $\phi(i)$ relative RoPE shift | `var stream_cache: StreamingSinkKVCache<sinks=4, window=4096>;`<br>`@streaming_sink(sinks=4, window=4096)` | `StreamingSinkKVCacheType`<br>`SinkWindowAttr`<br>`CircularIndexExpr` | `velox.stream.ring_write`<br>`velox.stream.rope_shift`<br>Pass: `velox.opt.circular_buffer_lowering` | Contiguous static VRAM allocation, hardware ring buffer addressing, $O(1)$ memory bound |
| **8** | **H2O** (NeurIPS '23) | Dynamic KV Eviction | Cumulative Attention Score Integral: $\mathcal{I}_t(i) = \sum_{\tau=i}^t \sum_h \alpha_{\tau, i}^{(h)}$; Fixed budget $B = H + W$ | `@eviction(policy="h2o", budget=2048, heavy_hitters=1024)`<br>`var h2o_cache: EvictableKVCache<T, Policy=H2O>;` | `EvictionPolicyAttr`<br>`ScoreTrackingRegister`<br>`EvictTokenStmt` | `velox.attn.accumulate_scores`<br>`velox.cache.evict_min_score`<br>Pass: `velox.opt.h2o_score_tracking` | Shared-memory top-K reduction, register score tracking, in-place page reclamation |
| **9** | **SnapKV** (2024) | Dynamic KV Eviction | Observation window voting: $\mathcal{S}_h(i) = \sum_{q \in W_{\text{obs}}} \alpha_{q, i}^{(h)}$; 1D max-pooling clustered pruning | `@eviction(policy="snapkv", obs_window=32, top_k=512)`<br>`var snap_cache: FrozenPrunedKVCache<T>;` | `SnapKVPolicyAttr`<br>`ObsWindowPoolExpr`<br>`PruneKVBlockStmt` | `velox.snapkv.vote_pool`<br>`velox.snapkv.prune_prefix`<br>Pass: `velox.opt.snapkv_observation_cluster` | Prefill observation pooling kernel, frozen pruned KV cache, zero decode eviction overhead |
| **10** | **PyramidKV** (2024) | Non-Uniform KV | Layer-wise entropy budgeting: $B_l = B_{\text{base}} \cdot (l/L)^\gamma$; Low layers pruned, high layers retained | `@layer_budget_distribution(policy="pyramid", gamma=2.0)`<br>`var pyramid_cache: LayerBudgetedKV<T>;` | `LayerBudgetDistributionAttr`<br>`PyramidBudgetExpr` | `velox.planner.set_layer_budget`<br>Pass: `velox.opt.pyramid_layer_budgeting` (Velox Static Memory Planner) | Static non-uniform VRAM slab allocation, entropy-guided layer buffer planning |
| **11** | **Speculative Decoding** (Leviathan '23) | Speculative Execution | Rejection sampling: $\beta_i = \min\left(1, \frac{p(x_i)}{q(x_i)}\right)$; Residual distribution: $p'(x) = \frac{\max(0, p(x) - q(x))}{Z}$; Invariant: $P_{\text{spec}} \equiv p$ | `speculative_draft(model=DraftM, budget=K) { ... }`<br>`speculative_verify(model=TargetM) { ... }` | `SpeculativeDraftScope`<br>`SpeculativeVerifyScope`<br>`RejectionSamplingExpr` | `velox.spec.draft_rollout`<br>`velox.spec.target_verify`<br>`velox.spec.residual_sample`<br>Pass: `velox.opt.fuse_residual_resample` | Heterogeneous draft/target scheduling, vectorized rejection sampling, branch rollback |
| **12** | **Medusa** (2024) | Multi-Head Speculation | Non-autoregressive heads $\tilde{h}_t^{(k)} = \text{SiLU}(W_{\text{proj}}^{(k)} h_t) + h_t$; 2D Tree DAG Mask $M_{\text{tree}}[u, v] = (v \in \text{Anc}(u) \cup \{u\}) ? 0 : -\infty$; Tree Path Compaction | `medusa_heads(count=4) { ... }`<br>`tree_verify(mask=TreeMask2D, budget=16);` | `MedusaHeadDecl`<br>`TreeDAGMaskExpr`<br>`TreePathCompactionStmt` | `velox.medusa.forward_heads`<br>`velox.tree_attention_2d`<br>`velox.kv_commit_path`<br>Pass: `velox.opt.synthesize_tree_dag_mask` | 2D non-causal tree attention kernel, depth RoPE offsets; Bifurcated Rollback Invariant: $O(1)$ length pointer reset strictly for linear chains (Leviathan); explicit KV path compaction (`commit_accepted_path`) in $O(L)$ steps ($L = \text{depth} \le 6$) or page-table slot remapping for branching trees to prevent non-contiguous slot corruption |
| **13** | **EAGLE / EAGLE-2** (ICML '24) | Feature Speculation | Feature-level drafting $f_{t+1} = \text{DraftLayer}(\text{Concat}(f_t, e_{t+1}))$; Dynamic confidence tree search (PriorityQueue with threshold $\gamma$) | `feature_draft(layer=DraftTransformerLayer) { ... }`<br>`confidence_tree(budget=64, threshold=0.7);` | `FeatureDraftBlock`<br>`DynamicConfidenceTreeScope`<br>`EAGLETreeExpandExpr` | `velox.eagle.feature_step`<br>`velox.eagle.dynamic_dag_expand`<br>Pass: `velox.opt.fuse_feature_draft_step` | Top-layer hidden state register forwarding, fused draft iteration, dynamic DAG attention |
| **14** | **SpecInfer** (ASPLOS '24) | Tree Continuous Batching | Multi-tenant tree batching; Topological sort tree flattening $\pi(T_b)$; Composite block-diagonal mask $M_{\text{batch}}$; PagedSpecTree subgraph page tables | `var tree_pool: PagedSpecTree<NodeCapacity=256>;`<br>`linearize_tree_batch(trees);` | `PagedSpecTreeType`<br>`BatchTreeLinearizeStmt`<br>`CompositeTreeMaskExpr` | `velox.tree.topological_sort`<br>`velox.ragged_tree_attention`<br>`velox.page_tree_free`<br>Pass: `velox.opt.tree_batch_flattener` | Ragged batch memory addressing, atomic subgraph reference counts, zero fragmentation |
| **15** | **Lookahead Decoding** (2024) | Jacobi Parallel Decoding | Non-linear system Jacobi solver: $x_{t+i}^{(k+1)} = \arg\max P(X_{t+i} \mid x_{<t}, x^{(k)})$; Two-branch parallel forward pass: Generate $\parallel$ Verify | `lookahead_scope(window=16, ngrams=4) { ... }`<br>`jacobi_rollout(steps=3);` | `LookaheadScope`<br>`JacobiRolloutStmt`<br>`NGramVerifyExpr` | `velox.jacobi.step`<br>`velox.ngram.match`<br>Pass: `velox.opt.jacobi_two_branch_fusion` | SIMD parallel n-gram matching, Tensor Core concurrent generate-verify forward pass |
| **16** | **FlashInfer** (2024) | Ragged Kernel Execution | Ragged layout via `indptr`: $S_b = \text{indptr}[b+1] - \text{indptr}[b]$; Split-KV two-phase online LogSumExp reduction across $P$ splits | `var batch: ragged_tensor<fp16, [B, *], indptr=I>;`<br>`split_kv_attention(splits=P);` | `RaggedTensorType`<br>`SplitKVAttentionExpr`<br>`IndptrIndexExpr` | `velox.ragged.pack`<br>`velox.split_kv.phase1_map`<br>`velox.split_kv.phase2_reduce`<br>Pass: `velox.opt.ragged_to_split_kv_lowering` | Multi-SM split-KV occupancy saturation, zero-padding memory compaction, lock-free reduction |
| **17** | **FlashAttention-3** (2024) | Hardware Warp Specialization | Hopper TMA `cp.async.bulk.tensor`; Dual `mbarrier` (`full` $\parallel$ `empty`); Warp specialization (1 Prod $\parallel$ 4 Cons); Interleaved GEMM-2 (WGMMA) $\parallel$ Softmax (`ex2.approx`) | `@warp_specialized(producer_warps=1, consumer_warps=4)`<br>`var tma: TMADescriptor<T>;` | `WarpSpecializationScope`<br>`TMACopyStmt`<br>`MBarrierSyncExpr` | `nvvm.cp.async.bulk.tensor`<br>`nvvm.mbarrier.arrive.expect_tx`<br>`nvvm.mbarrier.try_wait`<br>`nvvm.wgmma.mma_async`<br>Pass: `velox.opt.hopper_warp_specialization` | Hopper sm_90a / Blackwell sm_100a, TMA hardware engine, hardware `mbarrier`, SFU-WGMMA overlap |
| **18** | **Marlin** (2024) | Quantized GEMM | FP16 $\times$ INT4 dequantization: Magic exponent `0x64006400` ($1024.0$), zero-point subtrahend `0x64086408` ($1032.0$ for $z=8$); 11 PTX insts (1.375 inst/weight); Swizzled layout | `var w: quantized_weight<int4, group_size=128, layout="marlin_swizzled">;` | `QuantizedWeightType`<br>`MarlinDequantizeOp`<br>`SwizzledLayoutAttr` | `nvvm.lop3` (LUT `0xEA`)<br>`llvm.lshr`<br>`llvm.fsub`<br>`nvvm.wgmma`<br>Pass: `velox.opt.decompose_and_fuse_dequant` | Conflict-free 32-bit shared memory banks, register-level SIMD subtraction, memory bandwidth saturation |
| **19** | **Native FP8 & Block Scaling** (2024) | Microscaling & Sub-Byte | E4M3 (1-4-3, max 448.0) vs E5M2 (1-5-2, max 57344.0); OCP MXFP8 ($1 \times 32$ tile, E8M0 scale with bias 127); Exponent add; FTZ subnormals; JIT block scale; FP32 accum | `var t: tensor<fp8_e4m3fn, [B, S, D], block_scale=[128, 128]>;`<br>`var mx: tensor<mxfp8, [B, S, D], vector_tile=32, scale=e8m0>;` | `FP8BlockScaledTensorType`<br>`ScaleFactorBinding`<br>`FP8GEMMExpr` | `nvvm.mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3`<br>Pass: `velox.opt.fp8_block_scale_propagation`<br>Pass: `velox.opt.format_disambiguation` | Hopper/Blackwell FP8 Tensor Cores, OCP Microscaling ALU adders, mandatory FP32 accumulator |

---

## 2.2 Technical Deep-Dive: Domain 1 — Disaggregated & Chunked Serving Systems

### 2.2.1 DistServe (OSDI '24) & Splitwise (ISCA '24): Disaggregated Serving & RDMA Pipelining

#### 1. First-Principles Mathematical Mechanics
In monolithic serving architectures, prefill ($I_{\text{prefill}} \sim 4,000\text{--}5,000\text{ FLOPs/byte}$) and decode ($I_{\text{decode}} \sim 1\text{--}15\text{ FLOPs/byte}$) execute on the identical GPU hardware, competing for memory bandwidth and Tensor Cores. Under the Pollaczek-Khinchine formula for $M/G/1$ priority queues, high-variance prefill requests induce severe queueing waiting times $W_q^{\text{decode}}$ on decode tokens:
$$W_q^{\text{decode}} = \frac{\lambda \mathbb{E}[S^2]}{2(1 - \rho)}$$
where $\mathbb{E}[S^2]$ is dominated by large prompt prefill bursts, causing token generation stalls.

**Disaggregation Invariant**: By isolating prefill onto compute-bound nodes ($\mathcal{N}_P$) and decode onto memory-bound nodes ($\mathcal{N}_D$), decode queueing interference is eliminated:
$$W_q^{\text{decode}} \equiv 0$$

##### Layer-Wise RDMA Overlapping Proof
Transferring the complete Key-Value cache of a prompt of length $S$ at the end of the prefill stage creates a massive network bandwidth bubble. Let:
- $L = 32$ layers, $H = 32$ heads, $D = 128$ head dimension, $S = 2,048$ tokens, dtype = FP16 (2 bytes).
- KV cache size per layer:
  $$\mathbf{S}_{\text{layer}} = 2 \times S \times H \times D \times 2 = 2 \times 2048 \times 32 \times 128 \times 2 = 33,554,432\text{ bytes} = 32.0\text{ MiB}$$
- Across a 400 Gbps InfiniBand link ($B_{\text{net}} = 50.0\text{ GB/s}$), the transfer time per layer is:
  $$T_{\text{layer}}^{\text{xfer}} = \frac{32 \times 1024^2\text{ bytes}}{50 \times 10^9\text{ bytes/sec}} \approx 671\,\mu\text{s}$$
- On an NVIDIA H100 SXM5, layer compute time for prefill ($S = 2048$, $B = 1$, 70B model) is $T_{\text{layer}}^{\text{compute}} \approx 1,470\,\mu\text{s}$.
- Because:
  $$T_{\text{layer}}^{\text{xfer}} (671\,\mu\text{s}) < T_{\text{layer}}^{\text{compute}} (1,470\,\mu\text{s})$$
  the transfer of layer $l$'s KV cache over GPUDirect RDMA is **100% hidden** behind the computation of layer $l+1$. The effective network migration overhead across the entire model is strictly zero.

##### Splitwise Optimal Provisioning Ratio
Let $\mathbb{E}[S_p]$ denote expected prompt length, $\mathbb{E}[S_o]$ denote expected output generation length, $\mu_P$ denote prefill compute throughput (tokens/sec), and $\mu_D$ denote decode memory-bound throughput (tokens/sec). The cost-optimal node ratio $R_{P:D}^*$ is:
$$R_{P:D}^* = \frac{\mathbb{E}[S_p]}{\mathbb{E}[S_o]} \cdot \frac{\mu_D}{\mu_P}$$

#### 2. Concrete Velox Language Syntax & Type Signatures
```velox
disaggregated_cluster ProductionServing {
    @target_node(role = "prefill", hardware = "nvidia_h100_sxm5", count = 2)
    node PrefillWorker;

    @target_node(role = "decode", hardware = "nvidia_l40s_pcie", count = 8)
    node DecodeWorker;

    channel kv_migration: DisaggregatedChannel<
        dtype = f16,
        layers = 80,
        heads = 64,
        head_dim = 128,
        transport = "gpudirect_rdma"
    >;

    pipeline forward(prompt: Prompt) {
        let p_ctx = PrefillWorker.init_context(prompt);
        for l in 0..80 {
            let (act, layer_kv) = PrefillWorker.compute_layer(l, p_ctx);
            // Non-blocking asynchronous RDMA transfer pipelined with next layer compute
            async_rdma_send(kv_migration, layer = l, data = layer_kv, target = DecodeWorker);
        }
        DecodeWorker.start_decode(kv_migration);
    }
}
```

#### 3. AST Node Schema
```
DisaggregatedTopologyDecl {
    name: Identifier,
    prefill_nodes: Vec<NodeDecl>,
    decode_nodes: Vec<NodeDecl>,
    channels: Vec<ChannelDecl>,
    pipeline: PipelineDecl
}

ChannelDecl {
    name: Identifier,
    dtype: ScalarType,
    topology_shape: [ConstExpr; 4], // [layers, heads, head_dim, max_seq]
    transport: TransportKind // GPUDirectRDMA, NVLinkBridge, PinnedShm
}

AsyncRdmaSendExpr {
    channel: Identifier,
    layer_idx: ConstExpr,
    data_ref: ASTNode,
    target_node: Identifier
}
```

#### 4. Compiler Pass Architecture & IR Lowering Specification
- **Compiler Pass**: `velox.opt.rdma_pipeline_insert`
  1. Identifies the layer loop within `scope prefill`.
  2. Inserts an asynchronous non-blocking IBVerbs `ibv_post_send` work request immediately after the KV projection kernel completes for layer $l$.
  3. Inserts a completion check barrier (`ibv_poll_cq`) on the decode node prior to layer $l$ attention invocation.

---

### 2.2.2 Sarathi-Serve (OSDI '24) & Chunked Prefill: Eliminating Pipeline Bubbles

#### 1. First-Principles Mathematical Mechanics
In continuous batching with pipeline parallelism (PP), prefill requests monopolize pipeline stages for tens of milliseconds, while decode stages starve, introducing massive pipeline bubbles.
Let $M$ be the number of microbatches, and $PP$ be the pipeline depth. The pipeline bubble fraction is:
$$F_{\text{bubble}} = \frac{PP - 1}{M + PP - 1}$$
When prefill requests are large ($S = 4,096$), $M$ is small ($M=1$), driving $F_{\text{bubble}} \to \frac{PP-1}{PP}$ (up to 75% idle time).

**Chunked Prefill Invariant**: Split prompt prefill into uniform chunks of size $C^*$:
$$C^* = \frac{I_{\text{knee}}}{2} \approx 512\text{ tokens}$$
where $I_{\text{knee}}$ is the operational intensity knee where Tensor Core GEMM transitions from memory-bound to compute-bound.
Prefill chunks are interleaved with single-token decode iterations inside a unified iteration-level batch:
$$\text{BatchTokens} = C^* + \sum_{i=1}^{B_{\text{decode}}} 1 = 512 + 16 = 528\text{ tokens}$$
This bounds execution time per forward pass to a deterministic duration ($\sim 18\text{ ms}$), driving $M \to \infty$ and $F_{\text{bubble}} < 6\%$.

#### 2. Concrete Velox Language Syntax & Type Signatures
```velox
scope chunked_prefill<const CHUNK_SIZE: usize = 512, const MAX_DECODE_BATCH: usize = 32>(
    prompt_chunks: RaggedTensor<f16, [*, CHUNK_SIZE]>,
    decode_tokens: Tensor<[MAX_DECODE_BATCH, 1], u32>,
    kv: &mut PagedKVCache<32, 32, 128, 16, f16>
) {
    // Fuses chunked prompt tokens and decode tokens into a single unified ragged forward pass
    let batch = pack_ragged_batch(prompt_chunks, decode_tokens);
    let out = forward_piggybacked_attention(batch, kv);
}
```

#### 3. AST Node Schema & IR Lowering
```
ChunkedPrefillScope {
    chunk_size: usize,
    max_decode_batch: usize,
    body: BlockNode
}

ChunkScheduleStmt {
    prompt_ref: Identifier,
    chunk_index: ASTNode,
    total_chunks: ASTNode
}
```
- **Compiler Pass**: `velox.opt.chunk_prefill_scheduler` transforms irregular multi-turn prompts into constant-length $C^*$-token tensor operations, guaranteeing static register allocation and eliminating pipeline bubbles.

---

### 2.2.3 Virtual Memory KV-Cache Architectures: vAttention (ASPLOS '24) & Mooncake

#### 1. First-Principles Mathematical Mechanics
PagedAttention introduces software indirection: every memory load within the attention kernel must compute:
$$\text{PhysicalAddr} = \text{PageTable}[\text{seq\_id}][\text{logical\_token} / BS] \times \text{PageBytes} + (\text{logical\_token} \pmod{BS}) \times \text{Stride}$$
This software table indexing prevents the GPU memory controller from coalescing loads into 128-byte transactions and breaks compiler loop unrolling.
Furthermore, PagedAttention cannot leverage FlashAttention-3's asynchronous Hopper TMA (`cp.async.bulk.tensor`), which requires physically contiguous or stride-uniform virtual addresses.

**vAttention Virtual Memory Invariant**: Reserve a contiguous virtual address range for each sequence ($S_{\max} = 32,768$ tokens) using the low-level CUDA Virtual Memory Management (VMM) API (`cuMemAddressReserve`), while mapping physical 2MB huge pages on demand:
$$\text{PTE}_{\text{seq}} = \frac{32,768 \times 2 \times 32 \times 128 \times 2\text{ bytes}}{2\text{ MiB}} = \frac{536,870,912\text{ bytes}}{2,097,152\text{ bytes}} = 256\text{ pages}$$
Because 256 pages easily fit within the GPU Translation Lookaside Buffer (TLB capacity $\approx 4,096$ entries), TLB miss penalty is zero, and kernels execute without software page tables.

#### 2. Concrete Velox Language Syntax & Type Signatures
```velox
let mut kv_pool: VirtualContiguousKV<
    dtype = f16,
    layers = 32,
    heads = 32,
    head_dim = 128,
    max_seq_len = 32768,
    physical_page_size = 2 * MiB
>;

scope decode(token: u32, seq_id: usize, step: usize) {
    if (step % TOKENS_PER_2MB_PAGE == 0) {
        // Driver VMM allocation executed asynchronously without GPU stream synchronization
        kv_pool.map_next_physical_page(seq_id);
    }
    // Attention kernel executes with bare pointer arithmetic; zero PagedAttention table indexing
    let out = flash_attention_v3_unpaged(q, kv_pool.get_virtual_ptr(seq_id));
}
```

#### 3. AST Node Schema & Driver IR Lowering
- **IR Primitives**:
  - `velox.vmm.reserve`: Calls `cuMemAddressReserve` at startup.
  - `velox.vmm.map`: Calls `cuMemCreate` + `cuMemMap` + `cuMemSetAccess`.
  - Pass: `velox.opt.vmm_page_elim` removes all software page-table lookups from attention inner loops.

---

## 2.3 Technical Deep-Dive: Domain 2 — Advanced KV-Cache Compression & Latent Attention

### 2.3.1 Multi-Head Latent Attention (MLA, DeepSeek-V2 / DeepSeek-V3)

#### 1. First-Principles Mathematical Proof of Matrix Absorption
In standard Multi-Head Attention (MHA), serving a 70B model requires caching full Key and Value tensors for every token:
$$\text{KV\_Footprint}_{\text{MHA}} = 2 \times n_h \times d_h \times \text{sizeof}(\text{dtype}) = 2 \times 128 \times 128 \times 2 = 65,536\text{ bytes/token/layer}$$
DeepSeek Multi-Head Latent Attention (MLA) compresses the Key and Value states into a low-dimensional shared latent vector $\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c}$ (where $d_c = 512$), supplemented by a decoupled RoPE key $\mathbf{k}_t^R \in \mathbb{R}^{d_R}$ (where $d_R = 64$):
$$\mathbf{c}_t^{KV} = W_{DKV} h_t \quad (W_{DKV} \in \mathbb{R}^{d_c \times d})$$
$$\mathbf{k}_t^R = \mathcal{R}_t(W_{KR} h_t) \quad (W_{KR} \in \mathbb{R}^{d_R \times d})$$

##### The Non-Absorbability Theorem of Standard RoPE
Let $R_t \in \text{SO}(d_h)$ be the block-diagonal rotary embedding matrix at token position $t$. In standard RoPE, the query-key dot product is:
$$q_t^T k_s = (R_t W_Q h_t)^T (R_s W_K h_s) = h_t^T W_Q^T R_{s - t} W_K h_s$$
Because the rotary matrix $R_{s-t} = R_t^T R_s$ depends on the relative token position $s-t$, it cannot commute with arbitrary projection matrices:
$$R_{s-t} W_K \ne W_K R_{s-t}$$
Therefore, in standard RoPE, $W_K$ **cannot be absorbed** into $W_Q$, forcing serving engines to decompress and cache the full rotated key vector $k_s \in \mathbb{R}^{d_h}$ in VRAM.

##### Mathematical Proof of Decode Matrix Absorption
DeepSeek MLA decouples rotary embeddings from latent projections. For head $i$:
$$q_{t, i} = [W_{UQ, i} \mathbf{c}_t^Q ; R_t W_{QR, i} \mathbf{c}_t^Q] = [q_{t, i}^C ; q_{t, i}^R]$$
$$k_{s, i} = [W_{UK, i} \mathbf{c}_s^{KV} ; R_s \mathbf{k}_s^R] = [k_{s, i}^C ; k_{s, i}^R]$$
The attention score $A_{t, s, i}$ is:
$$A_{t, s, i} = (q_{t, i}^C)^T k_{s, i}^C + (q_{t, i}^R)^T k_{s, i}^R$$
Consider the non-rotary content term:
$$(q_{t, i}^C)^T k_{s, i}^C = (W_{UQ, i} \mathbf{c}_t^Q)^T (W_{UK, i} \mathbf{c}_s^{KV}) = (\mathbf{c}_t^Q)^T W_{UQ, i}^T W_{UK, i} \mathbf{c}_s^{KV}$$
By associativity of matrix multiplication, define the **Absorbed Query Projection Matrix** for head $i$:
$$W_{UQ, i}' = W_{UK, i}^T W_{UQ, i} \in \mathbb{R}^{d_c \times d_{cq}}$$
Similarly, for the Value projection $v_{s, i} = W_{UV, i} \mathbf{c}_s^{KV}$ and the final output projection $W_O = [W_{O, 1}, \dots, W_{O, H}]$:
$$O_t = \sum_{i=1}^H W_{O, i} v_{s, i} = \sum_{i=1}^H W_{O, i} (W_{UV, i} \mathbf{c}_s^{KV}) = \sum_{i=1}^H (W_{O, i} W_{UV, i}) \mathbf{c}_s^{KV}$$
Define the **Absorbed Output Projection Matrix** for head $i$:
$$W_{O, i}' = W_{O, i} W_{UV, i} \in \mathbb{R}^{d \times d_c}$$

##### Matrix Transpose Convention: Derivation vs Code Emission
Note on matrix orientation: In the theoretical derivation above, vectors are represented as column vectors ($q = W c$), yielding $W_{UQ, i}' = W_{UK, i}^T W_{UQ, i}$. In Velox source code and hardware Tensor Core emission, activations are row vectors ($q_{\text{row}} = c_{\text{row}} @ W$). Therefore, the offline folded weight in row-major layout evaluates to:
$$W_{UQ, i, \text{row}}' = W_{UQ, i} W_{UK, i}^T \in \mathbb{R}^{d_{cq} \times d_c}$$
enabling direct row-major evaluation: `q_absorbed = c_q @ w_absorbed_q` without runtime transpose operations.

##### Quantitative Memory Reduction
During the decode phase, the engine **never materializes or caches** $k_{s, i}$ or $v_{s, i}$. It caches strictly the latent vector $\mathbf{c}_s^{KV} \in \mathbb{R}^{512}$ and the decoupled RoPE key $\mathbf{k}_s^R \in \mathbb{R}^{64}$:
$$\text{Footprint}_{\text{MLA}} = (d_c + d_R) \times \text{sizeof}(\text{dtype}) = (512 + 64) \times 2 = 1,152\text{ bytes/token/layer}$$
$$\text{Reduction} = 1 - \frac{1,152}{65,536} = 1 - 0.01758 = \mathbf{98.24\%\text{ VRAM Reduction}}$$

#### 2. Concrete Velox Language Syntax & Type Signatures
```velox
model DeepSeek_V3 {
    @vram_immutable weight w_dkv: Tensor<[512, 7168], f16>;
    @vram_immutable weight w_kr:  Tensor<[64, 7168], f16>;
    // Compile-time absorbed projection weights (folded offline by the compiler)
    @vram_immutable weight w_absorbed_q: Tensor<[128, 512, 512], f16> = absorb_mla_q(w_uq, w_uk);
    @vram_immutable weight w_absorbed_o: Tensor<[128, 7168, 512], f16> = absorb_mla_v(w_o, w_uv);

    let mut kv_cache: LatentKVCache<
        layers = 61,
        latent_dim = 512,
        rope_dim = 64,
        dtype = f16
    >;

    scope decode(c_q: Tensor<[1, 512], f16>, q_r: Tensor<[1, 128, 64], f16>) {
        // Step 1: Multiply Query by absorbed matrix: q'_i = c_q @ W'_{UQ, i}
        let q_absorbed = c_q @ w_absorbed_q; // Result shape: [1, 128, 512]

        // Step 2: Directly execute attention against the cached latent vectors in VRAM
        // Zero key/value decompression in HBM!
        let attn_weights = latent_attention(q_absorbed, q_r, kv_cache.c_kv, kv_cache.k_r);

        // Step 3: Project attention context through absorbed output matrix W'_O
        let out = attn_weights @ w_absorbed_o;
    }
}
```

#### 3. AST Node Schema & Compiler Pass Mechanics
```
LatentKVCacheType {
    layers: usize,
    latent_dim: usize, // e.g., 512
    rope_dim: usize,   // e.g., 64
    dtype: ScalarType
}

AbsorbMlaNode {
    projection_q: ASTNode,
    projection_k: ASTNode,
    absorbed_target: Identifier
}
```
- **Compiler Pass**: `velox.opt.absorb_mla_weights`
  1. Inspects weight bindings during compilation.
  2. Multiplies $W_{UQ, i}$ and $W_{UK, i}^T$ ahead of time, storing the fused $W_{UQ, i}'$ in the binary image.
  3. Replaces runtime attention kernels with specialized `latent_gemv_wgmma` intrinsics operating directly on the 512-dim latent cache.

---

### 2.3.2 StreamingLLM (ICLR '24): Attention Sinks & Infinite-Context Decoding

#### 1. First-Principles Mathematical Mechanics
Standard autoregressive LLMs experience catastrophic perplexity collapse when sequence length exceeds pretraining window length $L_{\text{train}}$. Windowed attention (retaining only the most recent $W$ tokens) fails completely because softmax attention requires **Attention Sinks**:
$$Z = \sum_{j=1}^t \exp\left(\frac{q_t^T k_j}{\sqrt{D}}\right) \ge Z_{\text{sink}} = \sum_{j=1}^S \exp\left(\frac{q_t^T k_j}{\sqrt{D}}\right) \gg 1$$
During pretraining, the initial $S = 4$ tokens absorb massive redundant softmax attention probability mass even when semantically uninformative. If these initial tokens are evicted from the KV cache, the softmax denominator collapses, producing wild activation swings and garbage generation.

**StreamingLLM Invariant**: Retain the initial $S$ attention sink tokens permanently, while managing the remaining $W$ tokens as a circular FIFO ring buffer:
$$\text{ActiveKVTokens} = \{0, 1, \dots, S-1\} \cup \{t - W + 1, \dots, t\}$$

#### 2. Concrete Velox Language Syntax & Type Signatures
```velox
let mut stream_kv: StreamingSinkKVCache<
    layers = 32,
    heads = 32,
    head_dim = 128,
    sinks = 4,
    window = 4096,
    dtype = f16
>;

scope decode(token: u32, step: usize) {
    // Ring buffer write with static modulo arithmetic; bounded O(1) memory
    stream_kv.append(step, new_k, new_v);
    let out = streaming_flash_attention(q, stream_kv);
}
```

#### 3. AST Node Schema & Lowering
```
StreamingSinkKVCacheType {
    layers: usize,
    heads: usize,
    head_dim: usize,
    sinks: usize,
    window: usize,
    dtype: ScalarType
}
```
- **Compiler Pass**: `velox.opt.circular_buffer_lowering` compiles slot writes to `slot = (step < S) ? step : S + ((step - S) % W)`, guaranteeing that physical VRAM footprint is strictly invariant across infinite generation lengths ($T \to \infty$).

---

### 2.3.3 Dynamic KV Eviction: H2O, SnapKV, and PyramidKV

#### 1. First-Principles Mathematical Mechanics
- **H2O (Heavy Hitter Oracle, NeurIPS '23)**: Proves that attention matrices are sparse; a small subset of tokens ("Heavy Hitters" $\mathcal{H}$) account for the vast majority of cumulative attention scores:
  $$\mathcal{I}_t(i) = \sum_{\tau=i}^t \sum_{h=1}^H \alpha_{\tau, i}^{(h)}$$
  H2O maintains a fixed budget $B = H + W$. On each step, the token with the minimum accumulated score $\arg\min_{i \notin \text{Recent}} \mathcal{I}_t(i)$ is evicted from the cache.
- **SnapKV (2024)**: Rather than tracking attention across every decode step, SnapKV monitors attention exclusively during an **Observation Window** ($W_{\text{obs}} = 32$ tokens) at the end of the prompt prefill:
  $$\mathcal{S}_h(i) = \sum_{q \in W_{\text{obs}}} \alpha_{q, i}^{(h)}$$
  Applies 1D max-pooling with kernel size $K=8$ to cluster contiguous informative features, freezes the selected indices, and prunes all other prefill tokens before decoding begins. Decode eviction overhead is strictly zero.
- **PyramidKV (2024)**: Observes that attention concentration varies systematically across network depth: lower layers require broad context (high entropy), while deep layers focus on localized syntactic heads (low entropy). Sizing KV cache budgets non-uniformly:
  $$B_l = B_{\text{base}} \cdot \left(\frac{l}{L}\right)^\gamma$$
  slashes memory by an additional 50% without quality degradation.

#### 2. Concrete Velox Language Syntax & Type Signatures
```velox
// H2O Dynamic Eviction Policy
@eviction(policy = "h2o", budget = 2048, heavy_hitters = 1024, recent_window = 1024)
let mut h2o_cache: EvictableKVCache<f16, [32, 32, 128]>;

// SnapKV Clustered Prefill Eviction
@eviction(policy = "snapkv", obs_window = 32, top_k = 512, kernel_size = 8)
let mut snap_cache: FrozenPrunedKVCache<f16, [32, 32, 128]>;

// PyramidKV Non-Uniform Layer Allocation
@layer_budget_distribution(policy = "pyramid", base_budget = 4096, gamma = 1.8)
let mut pyramid_cache: LayerBudgetedKV<f16, [32, 32, 128]>;
```

---

## 2.4 Technical Deep-Dive: Domain 3 — Speculative Decoding & Tree Search Algorithms

### 2.4.1 Speculative Decoding (Leviathan et al., ICML '23)

#### 1. First-Principles Mathematical Proof of Exact Distribution Invariance
Autoregressive decoding generates tokens sequentially, requiring $N$ forward passes through a large target model $M_p$ to produce $N$ tokens. Because decode is memory-bound, forward pass execution time is dominated by streaming weights from HBM, leaving Tensor Cores largely idle.
Speculative Decoding accelerates serving by coupling a small, high-speed draft model $M_q$ with the large target model $M_p$:
1. $M_q$ autoregressively generates $K$ candidate tokens: $x_1, x_2, \dots, x_K$.
2. Target model $M_p$ evaluates all $K$ candidate tokens in parallel in a **single forward pass** (compute-bound prefill regime).
3. The server samples the accepted prefix using **Rejection Sampling**.

##### Mathematical Formulation of Rejection Sampling
Let $q(x)$ denote the probability distribution of draft model $M_q$, and $p(x)$ denote the probability distribution of target model $M_p$ over vocabulary $\mathcal{V}$.
Candidate token $x_i$ is accepted with probability:
$$\beta_i = \min\left(1, \frac{p(x_i)}{q(x_i)}\right)$$
If candidate $x_k$ is rejected, subsequent candidates $x_{k+1} \dots x_K$ are discarded. The replacement token $x_k^*$ is sampled from the **residual distribution**:
$$p'(x) = \frac{\max(0, p(x) - q(x))}{\sum_{y \in \mathcal{V}} \max(0, p(y) - q(y))}$$

##### Mathematical Proof of Output Distribution Equivalence
We prove that the speculative sampling procedure samples tokens from the exact distribution $p(x)$ of the target model:
$$P_{\text{spec}}(x) = P(x \text{ accepted}) + P(x \text{ emitted via residual})$$
$$P(x \text{ accepted}) = q(x) \cdot \min\left(1, \frac{p(x)}{q(x)}\right) = \min(q(x), p(x))$$
The probability that a candidate is rejected is:
$$1 - \beta = 1 - \sum_{y \in \mathcal{V}} q(y) \min\left(1, \frac{p(y)}{q(y)}\right) = 1 - \sum_{y \in \mathcal{V}} \min(q(y), p(y)) = \sum_{y \in \mathcal{V}} \max(0, p(y) - q(y))$$
Let $Z = \sum_{y \in \mathcal{V}} \max(0, p(y) - q(y))$. The residual emission probability is:
$$P(x \text{ emitted via residual}) = (1 - \beta) \cdot p'(x) = Z \cdot \frac{\max(0, p(x) - q(x))}{Z} = \max(0, p(x) - q(x))$$
Summing both paths:
$$P_{\text{spec}}(x) = \min(q(x), p(x)) + \max(0, p(x) - q(x)) = p(x) \quad \forall x \in \mathcal{V} \quad \blacksquare$$
The generated output text is mathematically indistinguishable from unaccelerated target model inference ($D_{\text{KL}}(P_{\text{spec}} \parallel p) \equiv 0$).

#### 2. Concrete Velox Language Syntax & Type Signatures
```velox
speculative_pipeline ProductionSpeculative<const K: usize = 5> {
    draft_model: Llama_3_2_1B,
    target_model: Llama_3_1_70B,

    scope step(prompt_context: Context) {
        // Step 1: Draft rollout on high-speed small model
        let draft_tokens: [u32; K] = draft_model.speculative_draft(prompt_context, budget = K);

        // Step 2: Parallel target verification pass in a single kernel
        let target_logits = target_model.parallel_verify(draft_tokens);

        // Step 3: Vectorized rejection sampling and branch rollback
        let (accepted_count, bonus_token) = vectorized_rejection_sample(
            draft_tokens,
            draft_model.get_probs(),
            target_logits
        );

        // O(1) pointer rollback strictly valid for 1D linear speculative chains (Leviathan/EAGLE-1).
        // (For branching speculative trees, explicit path compaction via commit_accepted_path is required).
        target_model.kv_cache.rollback(K - accepted_count);
    }
}
```

#### 3. AST Node Schema & Lowering Passes
```
SpeculativeDraftScope {
    draft_model: Identifier,
    budget: usize,
    body: BlockNode
}

SpeculativeVerifyScope {
    target_model: Identifier,
    candidates: ASTNode,
    sampling_policy: Identifier
}

RejectionSamplingExpr {
    draft_distribution: ASTNode,
    target_distribution: ASTNode,
    uniform_random_seed: ASTNode
}
```
- **Compiler Pass**: `velox.opt.fuse_residual_resample` fuses the rejection decision, uniform random number generation, and residual distribution calculation into a single vectorized warp-reduction GPU kernel.

---

### 2.4.2 Medusa: Non-Autoregressive Multi-Head Speculation & 2D Tree Attention

#### 1. First-Principles Mathematical Mechanics
Standard speculative decoding requires hosting and running a separate draft model, which consumes VRAM and incurs inter-model communication overhead.
**Medusa** eliminates the separate draft model by augmenting the target model's final hidden state $h_t$ with $K$ lightweight non-autoregressive decoding heads ($K=4$):
$$\tilde{h}_t^{(k)} = \text{SiLU}(W_{\text{proj}}^{(k)} h_t) + h_t \quad (k \in [1, K])$$
$$\text{Logits}_t^{(k)} = W_{\text{head}}^{(k)} \tilde{h}_t^{(k)}$$
Head $k$ predicts token $t + k + 1$ directly from hidden state $h_t$.

##### 2D Tree DAG Attention Mask
Medusa combines predictions from multiple heads into a tree of candidate sequences $\mathcal{T} = (\mathcal{V}, \mathcal{E})$. To evaluate all tree candidates in a single target forward pass without cross-candidate attention contamination, Medusa constructs a **2D Non-Causal Tree Attention Mask** $M_{\text{tree}} \in \mathbb{R}^{|\mathcal{V}| \times |\mathcal{V}|}$:
$$M_{\text{tree}}[u, v] = \begin{cases} 
0 & \text{if } v \in \text{Ancestors}(u) \cup \{u\} \\
-\infty & \text{otherwise}
\end{cases}$$
During target forward verification, candidate tokens attend strictly to their ancestors in the speculative tree and to all prior historical context tokens.

#### 2. Concrete Velox Language Syntax & Type Signatures
```velox
model MedusaModel {
    @vram_immutable weight medusa_heads: Array<Tensor<[4096, 32000], f16>, 4>;

    fn verify_tree(
        tree: SpeculativeTreeBlock<MaxNodes = 16, MaxDepth = 4, HeadDim = 128>,
        kv: &mut PagedKVCache<32, 32, 128, 16, f16>
    ) {
        let mask: Tensor<[16, 16], f16> = tree.synthesize_tree_mask();
        let logits = flash_tree_attention(q, kv, mask);
        let accepted_path = tree.resolve_best_path(logits);
        kv.commit_accepted_path(accepted_path);
    }
}
```

---

### 2.4.3 EAGLE / EAGLE-2 (ICML '24): Feature-Level Drafting & Context-Aware Trees

#### 1. First-Principles Mathematical Mechanics
Token-level drafting suffers from low acceptance rates on complex tasks because token embeddings discard rich syntactic context.
**EAGLE** shifts speculative drafting from the token space to the **hidden feature space**:
$$f_{t+1} = \text{DraftTransformerLayer}\left( [f_t ; e_{t+1}] \right)$$
where $f_t$ is the top-layer hidden state produced by the target model, and $e_{t+1}$ is the token embedding.
**EAGLE-2** introduces dynamic, context-aware tree generation:
- Instead of using a fixed static tree structure, EAGLE-2 expands tree branches dynamically using a priority queue.
- A branch node $u$ with token probability $P(u \mid \text{Anc}(u))$ is expanded if its cumulative confidence exceeds a dynamic threshold:
  $$\prod_{v \in \text{Path}(u)} P(v \mid \text{Anc}(v)) \ge \gamma_t$$
- The tree adapts its width and depth to the model's instantaneous prediction entropy, increasing token acceptance length from 2.1 to **over 3.8 tokens per forward pass**.

#### 2. Concrete Velox Language Syntax & Type Signatures
```velox
scope dynamic_speculation(target_state: Tensor<[1, 4096], f16>) {
    // Dynamically builds confidence tree in on-chip SRAM
    let tree = build_confidence_tree<MaxBudget = 64>(
        root_feature = target_state,
        threshold = 0.72
    );
    let mask = tree.compute_sparse_mask();
    let accepted = target_verify_sparse(tree, mask);
}
```

---

### 2.4.4 SpecInfer (ASPLOS '24) & Lookahead Decoding (2024)

#### 1. First-Principles Mathematical Mechanics
- **SpecInfer (Tree Continuous Batching)**: When serving multiple concurrent clients, each client maintains an independent speculative tree. SpecInfer introduces **Topological Sort Tree Flattening** $\pi(T_b)$ that packs heterogeneous trees from different batch sequences into a single contiguous ragged tensor, bounded by a composite block-diagonal tree attention mask $M_{\text{batch}}$:
  $$M_{\text{batch}}[i, j] = \begin{cases} 
  M_{\text{tree}}^{(b)}[\pi^{-1}(i), \pi^{-1}(j)] & \text{if } \text{Seq}(i) == \text{Seq}(j) == b \\
  -\infty & \text{otherwise}
  \end{cases}$$
- **Lookahead Decoding (Jacobi Parallel Decoding)**: Formulates autoregressive decoding as solving a system of non-linear equations via Jacobi fixed-point iteration:
  $$x_{t+i}^{(k+1)} = \arg\max P\left( X_{t+i} \mid x_{<t}, x_{t+1}^{(k)}, \dots, x_{t+i-1}^{(k)} \right)$$
  Executes a two-branch parallel forward pass:
  1. **Generate Branch**: Synthesizes and matches local n-grams in parallel.
  2. **Verify Branch**: Verifies multi-token n-gram candidates in a single Tensor Core forward pass.

---

## 2.5 Technical Deep-Dive: Domain 4 — High-Performance Serving Kernels & Modern Quantization

### 2.5.1 FlashInfer (2024): Ragged Continuous Batching & Split-KV Decoding

#### 1. First-Principles Mathematical Mechanics
Under continuous batching, requests have diverse context lengths: $S_b \in [1, S_{\max}]$. Padding sequences to a uniform rectangular tensor $[B, S_{\max}]$ incurs massive computational and memory waste ($O(B \cdot S_{\max}^2)$).
FlashInfer implements **ragged tensor layouts** using an integer indirection pointer array `indptr`:
$$\text{indptr} = [0, S_0, S_0 + S_1, \dots, \sum_{b=0}^{B-1} S_b]$$
The sequence length of request $b$ is computed via:
$$S_b = \text{indptr}[b + 1] - \text{indptr}[b]$$

##### Split-KV Two-Phase Online LogSumExp Reduction
In the decode phase, a single query token ($S_q = 1$) attends to a long historical KV cache ($S_{\text{kv}} = 16,384$). A standard attention kernel assigns one thread block per sequence head, resulting in only $B \times H = 16 \times 32 = 512$ thread blocks—severely under-utilizing large GPUs (e.g., NVIDIA H100 with 132 SMs and 8 warps per SM).
FlashInfer splits the long KV sequence dimension across $P$ independent thread blocks:
- **Phase 1 (Split-KV Map)**: Each split $p \in [0, P-1]$ processes $\frac{S_{\text{kv}}}{P}$ tokens, maintaining numerically stable local online LogSumExp state:
  $$m_p = \max_{j} S_{i, j}^{(p)}, \quad d_p = \sum_{j} \exp\left(S_{i, j}^{(p)} - m_p\right), \quad O_p = \sum_{j} \frac{\exp\left(S_{i, j}^{(p)} - m_p\right)}{d_p} V_j^{(p)}$$
- **Phase 2 (Cross-Split Reduce)**: A final reduction kernel combines the $P$ partial outputs across shared memory:
  $$m = \max(m_1, m_2)$$
  $$d = d_1 \exp(m_1 - m) + d_2 \exp(m_2 - m)$$
  $$O = O_1 \left( \frac{d_1 \exp(m_1 - m)}{d} \right) + O_2 \left( \frac{d_2 \exp(m_2 - m)}{d} \right)$$
Occupancy saturates all 132 SMs, slashing decode latency by up to 3.8x.

#### 2. Concrete Velox Language Syntax & Lowering
```velox
let batch: RaggedTensor<f16, [B, *], indptr = indptr_buf>;
let out = split_kv_attention<Splits = 8>(q, batch, kv_cache);
```

---

### 2.5.2 FlashAttention-3 (Dao et al., 2024): Hopper Warp Specialization & Asynchronous TMA

#### 1. Microarchitectural Mechanics on NVIDIA Hopper (sm_90a)
FlashAttention-3 leverages three architectural breakthroughs introduced in NVIDIA Hopper:
1. **Hardware Tensor Memory Accelerator (TMA)**:
   - Replaces thousands of individual register-load instructions (`LDG.E`) with a single hardware bulk copy instruction: `cp.async.bulk.tensor`.
   - Data moves directly from global VRAM into on-chip Shared Memory (`@bump_sram`), completely bypassing the register file and saving register allocation budget.
2. **Hardware Asynchronous Transaction Barriers (`mbarrier`)**:
   - TMA transactions signal completion via hardware barriers:
     ```cuda
     asm volatile("cp.async.bulk.mbarrier.arrive.noinc.b64 [%0], %1;" :: "r"(mbar_ptr), "r"(tx_bytes));
     asm volatile("mbarrier.try_wait.parity.b64 %0, [%1], %2;" : "=r"(ready) : "r"(mbar_ptr), "r"(phase));
     ```
3. **Warp Specialization (Producer-Consumer Decoupling)**:
   - A thread block partitions its warps into dedicated hardware roles:
     - **Producer Warp (1 Warp = 32 threads)**: Sits in an ultra-tight instruction loop, calculating tile addresses and issuing non-blocking `cp.async.bulk` TMA loads.
     - **Consumer Warps (4 Warps = 128 threads)**: Wait on the `mbarrier` and execute Hopper Asynchronous Warpgroup GEMM instructions (`wgmma.mma_async`).
4. **Ping-Pong Shared Memory Double Buffering**:
   - Producer loads tile $k+1$ into SMEM Buffer B while Consumer warps compute WGMMA on tile $k$ in SMEM Buffer A.
5. **Overlapping GEMM-2 with SFU Softmax**:
   - WGMMA operates asynchronously in the tensor core pipeline, freeing the warp's arithmetic logic units (ALU / SFU) to execute the exponential function (`ex2.approx`) for the next attention row concurrently.

#### 2. Concrete Velox Language Syntax & PTX Lowering
```velox
@warp_specialized(
    producer_warps = 1,
    consumer_warps = 4,
    smem_stages = 2
)
kernel flash_attention_v3<const TILE_M: usize = 128, const TILE_N: usize = 64>(
    q: &@bump_sram Tensor<[TILE_M, 128], f16>,
    kv: &@paged_vram PagedKVCache<32, 32, 128, 16, f16>
) {
    let barrier: @bump_sram mbarrier;
    // Emits nvvm.cp.async.bulk.tensor instruction
    tma_load_async(kv.tile(0), q, barrier);
    // Emits nvvm.wgmma.mma_async instruction
    wgmma_async_gemm(q, kv);
}
```

---

### 2.5.3 Marlin (Frantar et al., 2024): Asynchronous FP16 x INT4 Bandwidth Saturation

#### 1. Mathematical & PTX Mechanics
In the decode phase at small batch sizes ($B \le 16$), model execution is 100% memory-bandwidth bound. Storing weights in 4-bit integer format (INT4) cuts weight memory transfers by 75% relative to FP16.
However, standard dequantization requires unpacking INT4 nibbles into 16-bit half-precision floats, performing runtime shifts, and multiplying by dynamic scale factors—often adding more instruction overhead than the bandwidth saved.

##### The Marlin Magic Exponent Dequantization Trick
Marlin converts eight 4-bit unsigned integers packed inside a 32-bit register into four FP16 pairs (`half2`) using only **11 PTX instructions (1.375 instructions per weight)**:
1. Let $R$ be a 32-bit register containing 8 packed 4-bit integers: $[n_7, n_6, n_5, n_4, n_3, n_2, n_1, n_0]$.
2. Use PTX ternary logic `lop3.b32` with lookup table `0xEA` to mask alternating nibbles:
   $$R_{\text{even}} = R \ \& \ \mathtt{0x000F000F}$$
   $$R_{\text{odd}} = (R \gg 4) \ \& \ \mathtt{0x000F000F}$$
3. Bitwise OR with the **Magic FP16 Exponent Bias**: `0x64006400` ($1024.0$ in FP16 format, where exponent bits $= 13$, biased exponent $= 15 + 10 = 25$):
   $$F_{\text{even}} = R_{\text{even}} \ | \ \mathtt{0x64006400}$$
   Because the lower 10 bits of FP16 represent the mantissa, setting the upper bits to `0x6400` forces the register to represent the exact floating-point value:
   $$\text{Val} = 1024.0 + n_i$$
4. Subtract the static bias offset ($1024.0 + \text{zero\_point} = 1024.0 + 8.0 = 1032.0 \implies \mathtt{0x64086408}$) using a single-cycle hardware SIMD subtraction:
   $$\text{Weight}_{\text{fp16}} = \text{sub.f16x2}(F_{\text{even}}, \mathtt{0x64086408}) \equiv n_i - 8.0$$
5. Multiply by group scale factor: $W = \text{Weight}_{\text{fp16}} \times \text{Scale}$.

##### Marlin Swizzled Shared Memory Layout
Standard matrix layouts cause severe 32-way bank conflicts when loading dequantized weights from shared memory. Marlin formats the offline quantized weight tensor into a specialized $16 \times 64$ element swizzled tile layout, guaranteeing that consecutive threads within a warp read from mutually exclusive 32-bit shared memory banks:
$$\text{SharedMemoryBankConflicts} \equiv 0, \quad \text{AchievedMemoryBandwidth} \ge 94\%\text{ of Peak}$$

#### 2. Concrete Velox Language Syntax & AST
```velox
@vram_immutable weight w_marlin: QuantizedWeight<
    dtype = int4,
    group_size = 128,
    layout = "marlin_swizzled",
    scale_dtype = f16
> = safetensors("llama_int4.safetensors", "layers.0.mlp.down_proj");

scope decode(x: Tensor<[B, 4096], f16>) {
    // Emits optimized Marlin GEMM kernel saturating memory bandwidth
    let out = x @ w_marlin;
}
```

---

### 2.5.4 Native FP8 Execution & Block Scaling (E4M3 vs. E5M2, MXFP8, NVFP8)

#### 1. Mathematical Mechanics & Floating-Point Encodings
Modern serving architectures (NVIDIA Ada Lovelace, Hopper, Blackwell) integrate native FP8 Tensor Cores providing **2x the arithmetic throughput of FP16**:
1. **FP8 E4M3 (IEEE / NVIDIA format)**:
   - Format: 1 sign bit, 4 exponent bits, 3 mantissa bits (Bias = 7).
   - Dynamic Range: $[-448.0, 448.0]$.
   - Higher numeric precision ($\sim 3\text{ bits of precision}$); ideal for forward activations and weights.
2. **FP8 E5M2 (IEEE format)**:
   - Format: 1 sign bit, 5 exponent bits, 2 mantissa bits (Bias = 15).
   - Dynamic Range: $[-57,344.0, 57,344.0]$.
   - Wider dynamic range, lower precision; ideal for extreme attention score logits and gradients.

##### OCP Microscaling Formats (MXFP8) & Block Scaling
Per-tensor scaling fails when activations contain localized outlier spikes (e.g., hidden dimension activation channels exceeding 100.0 while median values are 0.05).
The OCP MXFP8 standard subdivides tensors into microscopic tiles ($1 \times 32$ elements), binding an 8-bit scale factor (E8M0, bias 127) to each 32-element vector:
$$\text{Scale} = 2^{E_{\text{scale}} - 127}$$
$$\text{DecompressedVal} = X_{\text{fp8}} \times 2^{E_{\text{scale}} - 127}$$
Hardware Microscaling ALUs perform the scale multiplication via a simple integer addition to the exponent field of the FP8 values, incurring zero multiplier silicon overhead.

##### Tensor Core Accumulation Invariant
$$\text{Accumulator} \equiv \mathbf{FP32}$$
Hardware FP8 Tensor Core instructions (`mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3`) mandate accumulating products into 32-bit single-precision registers to eliminate catastrophic rounding cancellation across deep summation reductions ($K = 4,096$).

#### 2. Concrete Velox Language Syntax & Verification
```velox
// Block-scaled FP8 tensor declaration with Hopper hardware alignment
let weights_fp8: @vram_immutable Tensor<
    [4096, 4096],
    fp8_e4m3,
    block_scale = [128, 128],
    scale_dtype = fp8_e8m0
> = safetensors("weights.safetensors", "layers.0.q_proj");

scope decode(x: Tensor<[B, 4096], f16>) {
    // Quantizes activations dynamically into FP8 with block scale factors
    let x_fp8 = quantize_block_fp8<BlockSize = 128>(x);
    // Emits Hopper mma.sync.aligned.m16n8k32 with FP32 accumulator
    let out_f32 = mma_fp8(x_fp8, weights_fp8);
    let out = out_f32.to_f16();
}
```
- **Compiler Passes**:
  - `velox.opt.fp8_block_scale_propagation`: Verifies scale factor dimensional alignment.
  - `velox.opt.format_disambiguation`: Statically verifies that E4M3 is utilized for GEMM operands and E5M2 for logits, terminating with compile error `E024` if precision downcasting occurs without explicit quantizer operators.

---

# Section 3: Core Runtime Invariants & Memory Tier Type System

## 3.1 Invariant 1: Zero Dynamic Heap Allocation During Inference ($O(1)$ Velox Activation Arena)

### 3.1.1 Problem Diagnosis & Root Cause Analysis
In conventional deep learning serving engines (PyTorch, vLLM, Hugging Face TGI, llama.cpp, TensorRT-LLM), memory is managed dynamically through runtime memory allocators (`c10::cuda::CUDACachingAllocator`, `malloc`, `cudaMalloc`, `cudaFree`). Under continuous batching with irregular prompt lengths and dynamic sequence termination, dynamic allocation results in:
1. **Unbounded Allocation Jitter**: Kernel launch threads block on internal mutexes inside memory allocators during block splitting and merging (0.2 ms to 45 ms per allocation).
2. **Checkerboard Virtual/Physical Fragmentation**: Segregated free lists (e.g., PyTorch's 20 MB small-block vs. large-block pools) develop non-coalescible holes, causing Out-Of-Memory (OOM) aborts despite having gigabytes of unallocated aggregate capacity.
3. **Implicit Device Synchronization**: When free pools deplete, `cudaMalloc` triggers implicit host-device stream synchronizations, flushing hardware execution pipelines and halting independent concurrent warps.

### 3.1.2 The Velox Activation Arena Architecture
Velox eliminates dynamic heap allocation during the entire inference lifecycle (prefill, chunked prefill, decode, speculative verification). The compiler transforms all transient tensor allocations across the model's computation Directed Acyclic Graph (DAG) into static, compile-time fixed byte offsets within a single contiguous block of pre-allocated GPU VRAM called the **Velox Activation Arena**.

The Velox Activation Arena operates with two distinct memory regions:
1. **Model Parameter & KV Static Pool**: Mapped once at process startup; never relocated or reallocated.
2. **Velox Activation Arena**: A single, contiguous VRAM slab sized strictly to the maximum instantaneous activation footprint of the network. Allocation and deallocation during token generation are reduced to $O(0)$ compile-time base-pointer offsets.

### 3.1.3 Mathematical Formulation & Proof of Bounded Memory
Let the execution of an LLM forward pass be modeled as a directed acyclic graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where vertices $v \in \mathcal{V}$ represent tensor generation operations and directed edges $(u, v) \in \mathcal{E}$ represent data dependencies.

**Definition 1 (Lifetime Interval)**: For each tensor $v \in \mathcal{V}$, let $t_{\text{birth}}(v) \in \mathbb{N}$ denote the topological execution step in which $v$ is materialized, and let $t_{\text{death}}(v) \in \mathbb{N}$ denote the topological execution step of the last operation consuming $v$ as an operand. The lifetime interval of $v$ is the closed integer range:
$$\mathcal{I}(v) = [t_{\text{birth}}(v), t_{\text{death}}(v)]$$

**Definition 2 (Memory Footprint Function)**: Let $\text{size}(v): \mathcal{V} \to \mathbb{Z}^+$ be the exact byte size of tensor $v$, defined by:
$$\text{size}(v) = \left( \prod_{d \in \text{shape}(v)} d \right) \times \text{sizeof}(\text{dtype}(v))$$
All dimension extents $d$ are statically resolved const-generics.

**Definition 3 (Instantaneous Live Set)**: At any discrete execution step $t \in [0, T_{\max}]$, the set of live activation tensors $\mathcal{L}(t)$ is:
$$\mathcal{L}(t) = \{ v \in \mathcal{V} \mid t \in \mathcal{I}(v) \}$$

**Definition 4 (Peak Theoretical Activation Bound)**: The lower bound on scratchpad memory required to execute graph $\mathcal{G}$ is the maximum instantaneous live weight:
$$\mathcal{M}^*(\mathcal{G}) = \max_{t \in [0, T_{\max}]} \sum_{v \in \mathcal{L}(t)} \text{size}(v)$$

**Theorem 1 (Static Activation Memory Layout & Asymptotic Bound)**: There exists a deterministic mapping function $\text{Offset}: \mathcal{V} \to [0, \mathcal{M}_{\text{actual}}]$ computed entirely at compile time such that for any pair of tensors $u, v \in \mathcal{V}$:
$$\mathcal{I}(u) \cap \mathcal{I}(v) \ne \emptyset \implies [\text{Offset}(u), \text{Offset}(u) + \text{size}(u)) \cap [\text{Offset}(v), \text{Offset}(v) + \text{size}(v)) = \emptyset$$
where mapping heterogeneous 1D memory buffers to contiguous address spaces is the Dynamic Storage Allocation (DSA) problem (NP-complete in the strong sense per Stockmeyer 1976 and Gergov 1999). Under the First-Fit Decreasing heuristic, the statically allocated scratchpad capacity satisfies the asymptotic competitive bound:
$$\mathcal{M}^*(\mathcal{G}) \le \mathcal{M}_{\text{actual}} \le \rho \cdot \mathcal{M}^*(\mathcal{G}) + \Delta_{\text{align}}$$
where $\rho \le 1.70$ is the competitive ratio for interval graph DSA packing, $\Delta_{\text{align}} = |\mathcal{V}| \times 128\text{ bytes}$ represents hardware alignment padding for 128-byte coalesced memory transactions, and the Velox Static Memory Planner provisions a calibrated fragmentation slack factor $\sigma_{\text{frag}} = 1.25$:
$$\mathbf{S}_{\text{velox\_act}} = \sigma_{\text{frag}} \cdot \max_{t \in [0, T_{\max}]} \sum_{v \in \mathcal{L}(t)} \text{size}(v) + \Delta_{\text{align}}$$

*Proof & Multi-Stream Linearization*:  
1. *Graph Chordality & Epoch Linearization*: An interval interference graph $\mathcal{G}_{\text{int}}$ is chordal if and only if execution events are totally ordered on $\mathbb{R}$. Under concurrent multi-stream execution (e.g. Asynchronous TMA transfers overlapping with Warp-Specialized GEMM accumulators in FlashAttention-3), execution forms a partial order (poset), which can introduce induced chordless 4-cycles ($C_4$). To preserve chordality and guarantee zero runtime deadlocks, the Velox compiler enforces **Execution Epoch Linearization**: multi-stream subgraphs are partitioned into synchronized execution epochs bounded by hardware `mbarrier` or stream synchronization barriers. Within each epoch, the interval graph is strictly chordal and perfect ($\chi(\mathcal{G}_{\text{int}}) = \omega(\mathcal{G}_{\text{int}})$).
2. *DSA Sizing & Static Offsets*: Unlike discrete graph vertex coloring, contiguous 1D memory packing cannot eliminate non-coalescible address holes when deallocating variable-sized tensors. The compiler executes First-Fit Decreasing on reverse topological live intervals. By embedding $\sigma_{\text{frag}} = 1.25$, peak fragmentation is guaranteed not to exceed the statically planned buffer envelope.
3. *Zero-Overhead Runtime Resolution*: At runtime, every tensor reference is resolved via:
$$\text{Addr}(v) = \text{ScratchpadBasePointer} + \text{Offset}(v)$$
Because $\text{Offset}(v)$ is an immutable 32-bit immediate constant embedded directly into generated C99/CUDA machine code, no heap allocation, stack expansion, system call, or lock acquisition occurs during token inference. Allocation complexity is strictly $O(0)$ runtime instructions. $\blacksquare$

##### Dynamic MoE Expert Scratchpad Sizing: Hotspotting & Capacity Factor
In Mixture-of-Experts (MoE) architectures (e.g. DeepSeek-V3, 256 routed experts, top-$k=8$), uniform activation sizing assuming balanced token routing ($B \cdot k / E$ tokens/expert) causes immediate buffer overflow under routing collapse/hotspotting. Conversely, sizing all $E$ experts for worst-case unshared batch capacity $B$ inflates scratchpad VRAM to over 13.3 GB.
Velox resolves this by introducing an explicit **MoE Expert Capacity Factor** ($C_{\text{expert}}$):
$$\text{Tokens}_{\text{expert\_cap}} = \left\lceil \gamma_{\text{cap}} \cdot \frac{B_{\max} \cdot k}{E} \right\rceil$$
where $\gamma_{\text{cap}} \in [1.5, 3.0]$ (default $\gamma_{\text{cap}} = 2.0$). Activations exceeding $\text{Tokens}_{\text{expert\_cap}}$ are deterministically routed to an auxiliary shared overflow buffer or dropped via compiler-enforced fallback, guaranteeing bounded scratchpad VRAM without buffer overflow.

### 3.1.4 Memory Lifecycle Rules & Scope Reclamation Invariants
1. **Rule 1.1 (Static Allocation Exclusivity)**: Inside `prefill`, `decode`, `chunked_prefill`, and `speculative_verify` scopes, any syntax, statement, or expression that invokes `malloc`, `calloc`, `realloc`, `free`, `cudaMalloc`, `cudaMallocAsync`, `cudaFree`, or `cudaFreeAsync` triggers compile error `E001`.
2. **Rule 1.2 (Scope-Bound Lifetime Invariant)**: All local tensors declared within an execution scope have lifetimes strictly bounded by the scope's terminal brace `}`. Tensors cannot escape their enclosing scope unless returned via an explicit output binding.
3. **Rule 1.3 (Phase-Barrier Reclamation)**: When the execution scope transitions from `prefill` to `decode`, the Velox Activation Arena base pointer is reset to offset zero. No transient prefill activations persist across the phase barrier. Intermediate state transfer between phases is permitted only through explicitly declared `@paged_vram` or `@vram_immutable` buffers.

---

## 3.2 Invariant 2: Compile-Time Static VRAM Feasibility Proofs ($VRAM_{total} \le \alpha \cdot VRAM_{budget}$)

### 3.2.1 Problem Diagnosis & Root Cause
In conventional LLM serving architectures, memory feasibility is determined dynamically at runtime. Servers launch, allocate model weights, and begin accepting client connections. As context lengths grow or batch sizes spike under high load, total memory consumption crosses physical VRAM thresholds, triggering a fatal CUDA Driver OOM exception (`CUDA_ERROR_OUT_OF_MEMORY`, Error 2). This causes process termination, dropping all in-flight user requests, corrupting state, and requiring several minutes for cold restarts.

### 3.2.2 Exact Feasibility Inequality Formulation
Velox treats physical memory limits as a rigid, static constraint verified during compilation. The compiler analyzes the target hardware specification and solves the global memory constraint equation:

$$\mathbf{VRAM}_{\text{total}} \le \alpha \cdot \mathbf{VRAM}_{\text{budget}}$$

where:
- $\mathbf{VRAM}_{\text{budget}}$ is the physical VRAM capacity declared in the `target hardware` block.
- $\alpha \in (0.0, 1.0]$ is the compiler safety factor (default $\alpha = 0.95$, reserving 5% for CUDA context runtime overhead, driver page tables, and hardware display controllers).
- $\mathbf{VRAM}_{\text{total}}$ is the exact compile-time sum of four mutually exclusive memory domains:

$$\mathbf{VRAM}_{\text{total}} = \mathbf{S}_{\text{weights}} + \mathbf{S}_{\text{kv\_cache}}(B_{\max}, S_{\max}) + \mathbf{S}_{\text{velox\_act}} + \mathbf{S}_{\text{system\_reserve}}$$

### 3.2.3 Formal Component Sizing Equations

#### 1. Model Weights Memory ($\mathbf{S}_{\text{weights}}$)
$$\mathbf{S}_{\text{weights}} = \sum_{w \in \mathcal{W}} \left[ \left( \prod_{d \in \text{shape}(w)} d \right) \times \mathbf{BPW}(\text{dtype}(w)) + \mathbf{S}_{\text{scale}}(w) \right]$$
where $\mathbf{BPW}$ represents the bytes-per-weight metric:
$$\mathbf{BPW}(\text{dtype}) = \begin{cases}
4.0 & \text{if dtype} = \text{f32} \\
2.0 & \text{if dtype} \in \{\text{f16}, \text{bf16}\} \\
1.0 & \text{if dtype} \in \{\text{fp8\_e4m3}, \text{fp8\_e5m2}, \text{int8}\} \\
0.5 & \text{if dtype} = \text{int4}
\end{cases}$$
For block-quantized tensors (e.g., Marlin INT4 or FP8 block-scaled), auxiliary scale tensors are sized explicitly:
$$\mathbf{S}_{\text{scale}}(w) = \left( \prod_{i=0}^{N-1} \left\lceil \frac{\text{shape}(w)[i]}{\text{block\_size}[i]} \right\rceil \right) \times \text{sizeof}(\text{scale\_dtype})$$

#### 2. Key-Value Cache Memory ($\mathbf{S}_{\text{kv\_cache}}$)
The memory footprint of the KV cache is determined by its concrete type signature:
- **For `PagedKVCache<L, H, D, BlockSize, T>`**:
  $$\mathbf{S}_{\text{paged}} = 2 \times L \times B_{\max} \times \left\lceil \frac{S_{\max}}{\text{BlockSize}} \right\rceil \times \text{BlockSize} \times \left( \frac{H_{kv}}{TP} \right) \times d_h \times \text{sizeof}(T) + \mathbf{S}_{\text{page\_tables}}$$
  where $H_{kv}$ represents Key-Value heads (distinguished from Query heads $H_q$ for Grouped Query Attention architectures), $TP \ge 1$ is the Tensor Parallelism degree across which KV heads are sharded, and:
  $$\mathbf{S}_{\text{page\_tables}} = B_{\max} \times \left\lceil \frac{S_{\max}}{\text{BlockSize}} \right\rceil \times \text{sizeof}(\text{uint32\_t})$$
- **For `LatentKVCache<L, LatentDim, RoPEDim, T>` (DeepSeek MLA)**:
  $$\mathbf{S}_{\text{latent}} = L \times B_{\max} \times S_{\max} \times \left( \frac{\text{LatentDim} + \text{RoPEDim}}{TP} \right) \times \text{sizeof}(T) + \mathbf{S}_{\text{page\_tables}}$$
- **For `StreamingSinkKVCache<L, H, D, Sinks, Window, T>` (StreamingLLM)**:
  $$\mathbf{S}_{\text{streaming}} = 2 \times L \times B_{\max} \times (\text{Sinks} + \text{Window}) \times \left( \frac{H_{kv}}{TP} \right) \times d_h \times \text{sizeof}(T)$$
  *(Notice that $\mathbf{S}_{\text{streaming}}$ is strictly independent of $S_{\max}$, enabling infinite context length $T \to \infty$ within fixed $O(1)$ memory).*
- **For `VirtualContiguousKV<L, H, D, S_{\max>, T>`**:
  $$\mathbf{S}_{\text{vmm}} = 2 \times L \times B_{\max} \times S_{\max} \times \left( \frac{H_{kv}}{TP} \right) \times d_h \times \text{sizeof}(T)$$
- **For `SpeculativeTreeBlock<MaxNodes, MaxDepth, D, T>`**:
  $$\mathbf{S}_{\text{tree}} = 2 \times L \times B_{\max} \times \text{MaxNodes} \times \left( \frac{H_{kv}}{TP} \right) \times d_h \times \text{sizeof}(T) + \mathbf{S}_{\text{tree\_mask\_buffer}}$$

#### 3. Velox Peak Activation Memory ($\mathbf{S}_{\text{velox\_act}}$)
$$\mathbf{S}_{\text{velox\_act}} = \sigma_{\text{frag}} \cdot \max \left( \mathcal{M}^*(\mathcal{G}_{\text{prefill}}), \mathcal{M}^*(\mathcal{G}_{\text{decode}}), \mathcal{M}^*(\mathcal{G}_{\text{speculative}}) \right) + \Delta_{\text{align}}$$
where $\sigma_{\text{frag}} = 1.25$ accounts for 1D DSA fragmentation slack.

#### 4. CUDA System & Driver Reserve ($\mathbf{S}_{\text{system\_reserve}}$)
$$\mathbf{S}_{\text{system\_reserve}} = \mathbf{BaseDriverContext} + \mathbf{S}_{\text{nccl\_reserve}} + (\text{StreamCount} \times \mathbf{StreamRingBytes})$$
where:
- $\mathbf{BaseDriverContext}$: Sized according to hardware architecture: $512\text{ MiB}$ on single-GPU desktop/laptop targets ($VRAM \le 16\text{ GiB}$), and $1024\text{ MiB}$ on datacenter accelerators ($VRAM \ge 24\text{ GiB}$, sm_90a/sm_100a) to account for 64-bit page tables and unified memory contexts.
- $\mathbf{S}_{\text{nccl\_reserve}}$: Captures distributed communication ring buffers for multi-GPU configurations ($TP > 1$):
  $$\mathbf{S}_{\text{nccl\_reserve}} = N_{\text{channels}} \times 2 \times \text{NCCL\_BUFFSIZE} \times (TP - 1)$$
  (with default $N_{\text{channels}} = 16$, $\text{NCCL\_BUFFSIZE} = 4\text{ MiB}$; evaluates to $0\text{ bytes}$ when $TP = 1$).

##### Multi-Node Hardware Boundary: DeepSeek-V3 (671B MoE)
For large MoE models such as DeepSeek-V3 (671B parameters in native FP8, $\mathbf{S}_{\text{weights}} = 624.92\text{ GiB}$):
An 8x H100 SXM5 node has aggregate physical VRAM of $8 \times 80 = 640.00\text{ GiB}$. Under safety factor $\alpha = 0.95$ ($608.00\text{ GiB}$ budget), weights alone exceed cluster capacity ($624.92 > 608.00\text{ GiB}$). DeepSeek-V3 serving physically mandates a minimum **16x H100 SXM5 cluster** (1,280 GiB aggregate capacity, yielding $39.06\text{ GiB}$ weights per GPU at $\text{EP}=16, \text{TP}=2$, leaving ample headroom for MLA KV cache and activations). The static feasibility verifier automatically enforces this multi-node cluster threshold.

### 3.2.4 Compile-Time Constraint Verification Algorithm
The static memory planner executes during compilation pass 6:
```
Algorithm: VerifyMemoryFeasibility
Input: Target Hardware Envelope H, Model AST M, Static Constants C (B_max, S_max)
Output: Validated Memory Layout or Compilation Abort with Remediation

1. S_weights <- ComputeStaticWeightFootprint(M.weights)
2. S_kv      <- ComputeKVCacheFootprint(M.kv_caches, C.B_max, C.S_max)
3. S_act     <- ComputeVeloxPeakActivation(M.scopes, C.B_max, C.S_max)
4. S_sys     <- ComputeDriverReserve(H)
5. S_total   <- S_weights + S_kv + S_act + S_sys
6. S_budget  <- H.vram_capacity * H.safety_factor

7. If S_total <= S_budget Then
8.     EmitProofCertificate(S_total, S_budget, Breakdown={S_weights, S_kv, S_act, S_sys})
9.     Return SUCCESS
10. Else
11.    Delta_Overflow <- S_total - S_budget
12.    RemediationOptions <- GenerateRemediations(Delta_Overflow, M, C)
13.    EmitDiagnosticError(Code=E002, Required=S_total, Available=S_budget, RemediationOptions)
14.    AbortCompilation()
```

### 3.2.5 Formal Diagnostic Error Output Format
When $S_{\text{total}} > S_{\text{budget}}$, the compiler emits error `E002` with explicit arithmetic breakdown and automated remediation paths:
```
================================================================================
VELOX COMPILER ERROR [E002]: Target Hardware VRAM Budget Exceeded
================================================================================
Target Hardware Specification:
  Device:                     NVIDIA GeForce RTX 3050 Laptop GPU (sm_86)
  Physical VRAM Capacity:     4096.00 MiB (4,294,967,296 bytes)
  Compiler Budget (alpha=0.95):3891.20 MiB (4,080,218,931 bytes)

Calculated Static Memory Allocation Breakdown:
  Model Weights (W):          2850.00 MiB  [73.2% of budget]
    - Q/K/V Projections:       1200.00 MiB (FP16)
    - MLP (Gate/Up/Down):      1450.00 MiB (FP16)
    - Norms & Embeddings:       200.00 MiB (FP32)
  KV Cache (Paged, B=4, S=2048):1024.00 MiB  [26.3% of budget]
  Velox Activation Arena: 320.00 MiB  [ 8.2% of budget]
  Driver & CUDA Context Reserve:256.00 MiB  [ 6.5% of budget]
  -------------------------------------------------------------
  Total Static VRAM Required: 4450.00 MiB (4,666,163,200 bytes)
  Capacity Overflow:          +558.80 MiB EXCEEDED HARDWARE CEILING

Compiler Feasibility Remediation Strategies:
  Option 1: Constrain maximum sequence length:
            Reduce `max_seq_len` from 2048 to 928 (reclaims 560.00 MiB).
  Option 2: Enable Key-Value Quantization:
            Change `PagedKVCache<..., f16>` to `PagedKVCache<..., fp8_e4m3>`
            (reduces KV cache from 1024.00 MiB to 512.00 MiB, reclaiming 512.00 MiB).
  Option 3: Increase Tensor Parallelism Sharding:
            Increase `tensor_parallel` from 1 to 2 (or 4).
            Sharding KV cache across 2 GPUs reduces local KV footprint
            from 1024.00 MiB to 512.00 MiB (reclaiming 512.00 MiB per GPU).
            (Note: Post-training migration from MHA to LatentKVCache is mathematically
            prohibited for pre-trained weights without retraining and triggers E013).
================================================================================
Compilation terminated with 1 error.
```

---

## 3.3 Invariant 3: Lock-Free Cooperative Fiber Event Loop (<80µs Host Dispatch)

### 3.3.1 Problem Diagnosis & Root Cause
Existing Python/C++ serving frameworks exhibit significant host-side scheduling overhead:
1. **Python GIL Contention & Asyncio Latency**: Python’s cooperative event loop (`asyncio`) runs on a single OS thread. When garbage collection, token serialization, or HTTP parsing occurs, the event loop pauses for 1,500 µs to 8,000 µs.
2. **OS Thread Preemption & Context Switching**: Traditional multithreaded servers assign worker threads to incoming requests, incurring kernel context switch penalties (5 µs to 25 µs) and cache line thrashing.
3. **Synchronous CUDA Host Latency**: Issuing individual kernel launches across deep networks (e.g., 80 layers $\times$ 14 kernels = 1,120 launches per token) via standard driver wrappers takes 4.5 µs per launch ($1120 \times 4.5\,\mu\text{s} = 5.04\,\text{ms}$), leaving high-throughput GPUs idle while waiting for host dispatch.

### 3.3.2 Velox Cooperative Fiber Architecture
Velox implements a runtime based on **user-space M:N cooperative fibers** combined with **lock-free single-producer single-consumer (SPSC) completion ring buffers** and **direct CUDA driver command stream batching**.

```
+---------------------------------------------------------------------------------------------------+
|                              VELOX LOCK-FREE FIBER ARCHITECTURE                                    |
+---------------------------------------------------------------------------------------------------+
|  [Hardware Worker Thread] (Pinned to Physical CPU Core via pthread_setaffinity_np)               |
|         |                                                                                         |
|         v                                                                                         |
|  +---------------------------------------------------------------------------------------------+  |
|  | Fiber Scheduler (Zero Heap Alloc, Cooperative State Machine)                                |  |
|  |   - Active Fiber Execution Queue (Pre-allocated Ring Buffer, Capacity=4096)                 |  |
|  |   - Sub-microsecond Coroutine Context Switch: cooperatively swap registers in 8 ns          |  |
|  +---------------------------------------------------------------------------------------------+  |
|         |                                                      ^                                  |
|         | Non-blocking Direct Kernel Batch                     | Lock-free Status Read            |
|         v                                                      |                                  |
|  +-------------------------------------+             +-----------------------------------------+  |
|  | Low-Level CUDA Driver Stream Buffer |             | Host Completion Ring Buffer (SPSC)      |  |
|  | (Direct cuLaunchKernel via Driver)  |             | (alignas(64) Head/Tail, Zero Mutexes)   |  |
|  +-------------------------------------+             +-----------------------------------------+  |
|         |                                                      ^                                  |
|         v                                                      | Hardware Event Signaling         |
|  +---------------------------------------------------------------------------------------------+  |
|  | GPU Hardware Engine (Asynchronous Execution of Prefill / Decode / Tree Verification)        |  |
|  +---------------------------------------------------------------------------------------------+  |
+---------------------------------------------------------------------------------------------------+
```

### 3.3.3 Formal Fiber State Machine
Every serving request executes inside an isolated user-space fiber $\mathcal{F}$. The lifecycle of fiber $\mathcal{F}$ is governed by a deterministic finite automaton (DFA):

$$\Sigma = (\mathcal{S}, \mathcal{E}_{\text{events}}, \delta, s_0, \mathcal{S}_{\text{term}})$$

where:
- $\mathcal{S} = \{ \text{READY}, \text{RUNNING}, \text{WAITING\_GPU}, \text{WAITING\_IO}, \text{TERMINATED} \}$
- $s_0 = \text{READY}$
- $\mathcal{S}_{\text{term}} = \{ \text{TERMINATED} \}$
- Transition function $\delta: \mathcal{S} \times \mathcal{E}_{\text{events}} \to \mathcal{S}$ is defined by the table:

| Current State | Event | Next State | Action Executed by Runtime |
|---|---|---|---|
| `READY` | `FIBER_DISPATCH` | `RUNNING` | Restore callee-saved registers (R12–R15, RBX, RBP, RSP), jump to fiber instruction pointer. |
| `RUNNING` | `GPU_SUBMIT_ASYNC` | `WAITING_GPU` | Append kernel descriptors to CUDA command ring buffer; record event tag $E_{\text{seq}}$; yield fiber. |
| `RUNNING` | `IO_YIELD` | `WAITING_IO` | Register client socket handle in epoll/kqueue ring; yield fiber. |
| `WAITING_GPU` | `GPU_SIGNAL_COMPLETION` | `READY` | Hardware completion writes sequence token to completion ring buffer; atomic tail increment wakes fiber. |
| `WAITING_IO` | `SOCKET_DATA_READY` | `READY` | Socket descriptor signaled; fiber requeued in active scheduler ring. |
| `RUNNING` | `SEQUENCE_COMPLETE` | `TERMINATED` | Recycle sequence ID to free bitmask ($O(1)$); free paged KV blocks ($O(1)$). |

### 3.3.4 Sub-80µs Host Dispatch Guarantees
The total host dispatch budget $\mathcal{T}_{\text{host}}$ is strictly bounded:

$$\mathcal{T}_{\text{host}} = \tau_{\text{context\_switch}} + \tau_{\text{batch\_schedule}} + \tau_{\text{driver\_submit}} + \tau_{\text{sampling\_poll}} \le 80\,\mu\text{s}$$

Measured breakdown guaranteed by compiler code generation:
1. $\tau_{\text{context\_switch}} \le 0.015\,\mu\text{s}$: Assembly-level 64-bit register swap without kernel intervention.
2. $\tau_{\text{batch\_schedule}} \le 2.500\,\mu\text{s}$: Active continuous batch sequence scheduling using 64-bit bitmask registers and hardware trailing-zero count instructions (`_tzcnt_u64`), scanning 1,000 active streams in $<0.95\,\mu\text{s}$ within L1 cache.
3. $\tau_{\text{driver\_submit}} \le 18.000\,\mu\text{s}$: Single-operation submission via pre-captured **CUDA Graph Launch** (`cuGraphLaunch`, $8.0\text{--}12.0\,\mu\text{s}$) or **Persistent GPU Device Kernel** doorbell writes ($0.5\,\mu\text{s}$). Discrete `cuLaunchKernel` parameter loops across 80+ layers take $>500\,\mu\text{s}$ and are strictly prohibited in the token decode loop.
4. $\tau_{\text{sampling\_poll}} \le 0.800\,\mu\text{s}$: Single lock-free read on the Host Completion Ring buffer (`alignas(64)` SPSC queue preventing cacheline bouncing).

$$\mathcal{T}_{\text{host\_total}} = 0.015 + 2.500 + 18.000 + 0.800 = \mathbf{21.315\,\mu\text{s} \ll 80\,\mu\text{s}}$$

---

## 3.4 Memory Tier Type System

```
+----------------------------------------------------------------------------------------------------+
|                                  VELOX MEMORY TIER ARCHITECTURE                                    |
+----------------------------------------------------------------------------------------------------+
| Tier 1: @vram_immutable  | Read-Only Model Weights, RoPE Sin/Cos Tables, Quantization Scales       |
| Tier 2: @paged_vram       | Dynamic Paged Key-Value Cache Blocks, Block Allocation Tables           |
| Tier 3: @bump_sram        | On-Chip SMEM Scratchpads, Hopper TMA Registers, Ping-Pong Buffers      |
| Tier 4: @host_pinned      | Non-Pageable DMA Host Buffers, Token Ring Buffers, RDMA Ingestion       |
+----------------------------------------------------------------------------------------------------+
```

### 3.4.1 The Four First-Class Physical Memory Tiers

#### 1. `@vram_immutable`
- **Physical Location**: High Bandwidth Memory (HBM3 / HBM3e / GDDR6X) on the target accelerator.
- **Access Semantics**: Read-Only ($\mathbf{RO}$). Mutation is strictly prohibited at the type level.
- **Initialization**: Memory-mapped directly from `.safetensors` files via the zero-copy OS loader. The physical pages are mapped directly into GPU virtual memory addresses using `cuMemHostRegister` or GPUDirect Storage (cuFile), bypassing intermediate host RAM buffers.
- **Compiler Invariant**: Pointers to `@vram_immutable` memory cannot be passed as mutable arguments or target destinations of kernel output writes. Page tables are marked with hardware write-protection bits.

#### 2. `@paged_vram`
- **Physical Location**: Accelerator High Bandwidth Memory (HBM).
- **Access Semantics**: Read-Write ($\mathbf{RW}$), managed via uniform, fixed-size physical blocks (e.g., 16 or 32 tokens per block).
- **Allocation Mechanism**: Dynamic, non-contiguous block allocation governed by compile-time $O(1)$ hardware bitmask allocators (`__builtin_ctzll`). Logical addresses are resolved via sequence block tables.
- **De-allocation**: Zero fragmentation by construction. Blocks are recycled instantaneously via bitmask set operations without memory compaction or defragmentation pauses.

#### 3. `@bump_sram`
- **Physical Location**: On-chip accelerator Shared Memory (SMEM), L1 cache, or Hopper Tensor Memory Accelerator (TMA) register files (up to 228 KB per Streaming Multiprocessor on NVIDIA sm_90a).
- **Access Semantics**: Extremely high bandwidth ($\sim 33\text{ TB/s}$ aggregate), ultra-low latency (1–3 clock cycles). Read-Write ($\mathbf{RW}$).
- **Lifetime & Scope**: Strictly scoped to the lifetime of an individual kernel invocation or thread-block tile. Memory is managed via a zero-cost bump pointer initialized to byte 0 at kernel entry and discarded at kernel exit.
- **Compiler Invariant**: Pointers to `@bump_sram` memory cannot be stored into global structs, cannot be returned across kernel boundaries, and cannot be captured in escaping closures.

#### 4. `@host_pinned`
- **Physical Location**: Host Physical RAM locked in place (non-pageable) via `cudaHostAllocMapped` or OS `mlock()`.
- **Access Semantics**: Accessible concurrently by the host CPU and the accelerator memory controller via high-speed PCIe Gen5 / NVLink C2C direct memory access (DMA).
- **Purpose**: Used for high-speed prompt token ingestion, streaming generation token output rings, and GPUDirect RDMA disaggregated prefill transfer queues.
- **Compiler Invariant**: Memory transfers between `@host_pinned` and `@paged_vram` / `@bump_sram` must use explicit asynchronous DMA copy statements (`dma_async`). Synchronous host-device reads inside the decode loop trigger compile error `E003`.

### 3.4.2 Disjoint Physical Address Space Qualifiers & Explicit Inter-Tier Movement
In Velox, physical memory tiers do **not** form a subtyping poset lattice. Because `@bump_sram` (32-bit SMEM window offsets) and `@paged_vram` (64-bit accelerator global HBM virtual addresses) require distinct hardware instructions (`ld.shared` vs `ld.global`), subtyping subsumption ($T_1 <: T_2$) is mathematically unsound and causes fatal PTX illegal memory faults (`CUDA_ERROR_ILLEGAL_ADDRESS`).

Memory tiers in Velox are **strictly disjoint address space qualifiers** (analogous to Embedded C / OpenCL address space qualifiers: `__global`, `__local`, `__constant`):
```
    +-----------------------------------------------------------------------+
    |               DISJOINT PHYSICAL ADDRESS SPACES (NO SUBTYPING)         |
    +-------------------+-----------------+----------------+----------------+
    | @vram_immutable   | @paged_vram     | @bump_sram     | @host_pinned   |
    | (Read-Only HBM)   | (Paged HBM)     | (On-Chip SMEM) | (Pinned DRAM)  |
    +-------------------+-----------------+----------------+----------------+
              |                  ^               ^                 ^
              +-- TMA Load ----->|               |                 |
              |<-- Async DMA ----+---------------+                 |
              |<----------------- Host-to-Device DMA --------------+
```
Transfers between address spaces cannot occur via implicit casting or subtyping; they require explicit, non-blocking hardware transfer primitives (`tma_load_async`, `tma_store_async`, `dma_async_h2d`, `dma_async_d2h`).

#### Tier Compatibility & Conversion Matrix
| Source Tier ($T_{\text{src}}$) | Target Tier ($T_{\text{dst}}$) | Allowed? | Required Syntax / Operational Mechanism |
|---|---|---|---|
| `@vram_immutable` | `@vram_immutable` | **YES** | Zero-cost reference alias. |
| `@vram_immutable` | `@paged_vram` | **NO** | Compile Error `E004`: Cannot cast immutable weight to dynamic paged cache. |
| `@vram_immutable` | `@bump_sram` | **YES** | Asynchronous TMA load: `tma_load_async(src, dst_sram)`. |
| `@paged_vram` | `@bump_sram` | **YES** | Asynchronous TMA load: `tma_load_async(src_slot, dst_sram)`. |
| `@bump_sram` | `@paged_vram` | **YES** | Asynchronous TMA store: `tma_store_async(src_sram, dst_slot)`. |
| `@bump_sram` | `@vram_immutable` | **NO** | Compile Error `E004`: Mutation of immutable VRAM prohibited. |
| `@host_pinned` | `@paged_vram` | **YES** | Direct DMA transfer: `dma_async_h2d(host_buf, vram_buf)`. |
| `@paged_vram` | `@host_pinned` | **YES** | Direct DMA transfer: `dma_async_d2h(vram_buf, host_buf)`. |
| `@bump_sram` | `@host_pinned` | **NO** | Compile Error `E005`: Cannot DMA directly between SRAM and Host RAM (must stage through VRAM). |

### 3.4.3 Explicit Memory Movement Primitives
1. `tma_load_async(source, destination, barrier)`: Asynchronously transfers a 2D or 3D tensor tile from `@vram_immutable` or `@paged_vram` into `@bump_sram` using the hardware Tensor Memory Accelerator (TMA).
2. `tma_store_async(source, destination)`: Asynchronously commits a computed accumulator tile from `@bump_sram` back into a paged or scratchpad tensor in `@paged_vram`.
3. `dma_async_h2d(host_source, vram_destination, completion_signal)`: Dispatches a non-blocking PCIe/NVLink direct memory access transfer from non-pageable `@host_pinned` memory to `@paged_vram`.
4. `dma_async_d2h(vram_source, host_destination, completion_signal)`: Dispatches a non-blocking PCIe/NVLink DMA transfer from accelerator VRAM to `@host_pinned` memory.

### 3.4.4 Compiler Enforcement & Escape Analysis
The Velox compiler implements static escape analysis for all `@bump_sram` allocations.

**Definition 5 (SRAM Escape Invariant)**: Let $e$ be an expression evaluating to a reference of tier `@bump_sram`. Let $\text{Scope}(e)$ be the lexical block containing $e$. Then $e$ violates the escape invariant if:
1. $e$ appears as the right-hand side of an assignment to a variable declared outside $\text{Scope}(e)$.
2. $e$ is passed as an argument to a function whose return type has a lifetime greater than $\text{Scope}(e)$.
3. $e$ is stored into an index or field of a tensor belonging to `@paged_vram`, `@vram_immutable`, or `@host_pinned`.
4. $e$ is stored into any field of a user-defined struct, tuple, or aggregate type whose instance has a declared lifetime exceeding $\text{Scope}(e)$.
5. $e$ is passed as a message payload to an asynchronous communication channel (`channel<T, N>`) or fiber task queue.

Any expression violating the SRAM Escape Invariant triggers compiler diagnostic error `E006`.

---

# Section 4: Native Attention & Key-Value Cache Type System

Velox elevates Key-Value cache buffers from untyped runtime tensors into first-class static types with rigorous operational semantics.

---

## 4.1 `PagedKVCache<layers, heads, head_dim, block_size, dtype>`

### 4.1.1 Type Signature & Parameters
```velox
struct PagedKVCache<
    const LAYERS: usize,
    const HEADS: usize,
    const HEAD_DIM: usize,
    const BLOCK_SIZE: usize = 16,
    type DTYPE = f16
>;
```
- `LAYERS`: Number of transformer attention layers (e.g., 32, 61, 80).
- `HEADS`: Number of Key-Value attention heads ($H_{\text{KV}}$ for GQA/MQA).
- `HEAD_DIM`: Dimensionality of each attention head ($D_h \in \{64, 128, 256\}$).
- `BLOCK_SIZE`: Number of tokens packed per physical allocation block ($BS \in \{16, 32, 64\}$; must be a power of 2, verified by `E015`).
- `DTYPE`: Numerical precision (`f16`, `bf16`, `fp8_e4m3`, `fp8_e5m2`).

### 4.1.2 Physical Memory Layout Equations
The cache organizes VRAM into a global 4D physical memory pool:
$$\text{PoolShape} = [\text{TotalBlocks}, 2, \text{LAYERS}, \text{HEADS}, \text{BLOCK\_SIZE}, \text{HEAD\_DIM}]$$
where $2$ represents the separate Key and Value tensor arrays.
The total physical byte size of block $b$ is:
$$\mathbf{S}_{\text{block}} = 2 \times \text{LAYERS} \times \text{HEADS} \times \text{BLOCK\_SIZE} \times \text{HEAD\_DIM} \times \text{sizeof}(\text{DTYPE})$$
For Llama-3-8B ($L=32, H=8, D=128, BS=16, \text{f16}$):
$$\mathbf{S}_{\text{block}} = 2 \times 32 \times 8 \times 16 \times 128 \times 2 = 2,097,152\text{ bytes} = 2.0\text{ MiB}$$
Notice that each physical block is exactly **2.0 MiB**, perfectly matching the hardware Huge Page TLB boundary on NVIDIA GPUs.

### 4.1.3 Physical Memory Addressing Equation
Given sequence ID $s$, layer index $l$, head index $h$, token index $t$, and head dimension coordinate $d$:
$$\text{BlockIndex} = \text{BlockTable}[s][\lfloor t / \text{BLOCK\_SIZE} \rfloor]$$
$$\text{SlotOffset} = t \pmod{\text{BLOCK\_SIZE}}$$
$$\text{Addr}_{\text{Key}}(s, l, h, t, d) = \text{BasePoolPtr} + \left( \text{BlockIndex} \times \mathbf{S}_{\text{block}} \right) + \left( 0 \times \frac{\mathbf{S}_{\text{block}}}{2} \right) + \left( l \times \text{Stride}_L \right) + \left( h \times \text{Stride}_H \right) + \left( \text{SlotOffset} \times \text{Stride}_T \right) + (d \times \text{sizeof}(T))$$

### 4.1.4 Method Interface & Disjoint Slice Lending
```velox
impl<L, H, D, BS, T> PagedKVCache<L, H, D, BS, T> {
    // Obtains exclusive in-place write view to slot 'step'
    fn borrow_slot_mut(&mut self, seq_id: usize, step: usize) -> KVSlotMut<L, H, D, T>;

    // Obtains immutable read view across historical tokens [start..end]
    fn borrow_context_const(&self, seq_id: usize, range: Range<usize>) -> KVContextRef<L, H, D, T>;

    // Recycles all physical blocks allocated to sequence 'seq_id' in O(1) bitmask ops
    fn release_blocks(&mut self, seq_id: usize);

    // Returns exact count of currently allocated physical blocks
    fn active_blocks_count(&self) -> usize;
}
```

---

## 4.2 `VirtualContiguousKV<layers, heads, head_dim, max_seq_len, dtype>`

### 4.2.1 Type Signature & Parameters
```velox
struct VirtualContiguousKV<
    const LAYERS: usize,
    const HEADS: usize,
    const HEAD_DIM: usize,
    const MAX_SEQ_LEN: usize = 32768,
    type DTYPE = f16
>;
```
- Maps each sequence into a **virtually contiguous address space** using the CUDA Virtual Memory Management (VMM) API.
- Eliminates software page-table lookups inside the attention kernel. Kernels execute using raw contiguous pointers (`half* __restrict__ k_ptr`), enabling direct execution of FlashAttention-3.

### 4.2.2 Low-Level Driver VMM Mechanics
1. **Startup Virtual Address Reservation**:
   $$\text{VSize}_{\text{seq}} = 2 \times \text{LAYERS} \times \text{HEADS} \times \text{MAX\_SEQ\_LEN} \times \text{HEAD\_DIM} \times \text{sizeof}(\text{DTYPE})$$
   Calls `cuMemAddressReserve(&d_vptr, VSize_seq, 0ULL, 0ULL, 0ULL)`.
2. **Incremental Physical Page Backing**:
   When sequence $s$ reaches a 2MB token boundary ($t \equiv 0 \pmod{\text{TOKENS\_PER\_2MB}}$):
   ```c
   CUmemAllocationProp prop = {};
   prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
   prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
   prop.location.id = device_id;
   cuMemCreate(&alloc_handle, 2097152, &prop, 0);
   cuMemMap(d_vptr + mapped_offset, 2097152, 0, alloc_handle, 0);
   cuMemSetAccess(d_vptr + mapped_offset, 2097152, &access_desc, 1);
   ```

### 4.2.3 Method Interface & Disjoint Virtual Slice Lending

```velox
impl<const L: usize, const H: usize, const D: usize, const S_MAX: usize, type T> VirtualContiguousKV<L, H, D, S_MAX, T> {
    /// Maps the next physical 2MB page for sequence `seq_id` via low-level CUDA VMM.
    /// Executed asynchronously without stalling the primary CUDA compute stream.
    /// Triggers `E001` if cumulative physical memory exceeds safety threshold alpha.
    fn map_next_physical_page(&mut self, seq_id: usize) -> Result<(), VmmAllocError>;

    /// Obtains the base contiguous raw virtual device pointer for sequence `seq_id`.
    /// Enables direct invocation of unpaged kernels (e.g. FlashAttention-3) with bare pointer arithmetic.
    fn get_virtual_ptr(&self, seq_id: usize) -> *const T;

    /// Obtains a mutable virtual pointer to sequence `seq_id` for in-place token writes.
    fn get_virtual_ptr_mut(&mut self, seq_id: usize) -> *mut T;

    /// Obtains an exclusive mutable reference to the single token slot at `step` for sequence `seq_id`.
    /// Guarantees compile-time disjoint slice lending (Axiom 1).
    fn borrow_slot_mut(&mut self, seq_id: usize, step: usize) -> KVSlotMut<L, H, D, T>;

    /// Obtains an immutable contiguous slice reference to historical tokens in `range` for sequence `seq_id`.
    /// Requires zero software page table traversals.
    fn borrow_context_const(&self, seq_id: usize, range: Range<usize>) -> KVContextRef<L, H, D, T>;

    /// Unmaps all physical pages backing sequence `seq_id` (`cuMemUnmap`) and frees allocation handles (`cuMemRelease`).
    /// Retains the contiguous virtual address reservation for subsequent reuse in O(1) time.
    fn release_sequence(&mut self, seq_id: usize);

    /// Returns the exact count of currently committed 2MB physical pages for sequence `seq_id`.
    fn mapped_pages_count(&self, seq_id: usize) -> usize;

    /// Returns the total virtual address reservation size in bytes per sequence.
    fn virtual_reservation_bytes(&self) -> usize;
}
```

---

## 4.3 `LatentKVCache<layers, latent_dim, rope_dim, dtype>`

### 4.3.1 Type Signature & Parameters (DeepSeek Multi-Head Latent Attention)
```velox
struct LatentKVCache<
    const LAYERS: usize,
    const LATENT_DIM: usize = 512,
    const ROPE_DIM: usize = 64,
    type DTYPE = f16
>;
```
- Replaces individual Key and Value head tensors with a **single compressed latent vector** $\mathbf{c}_t^{KV} \in \mathbb{R}^{\text{LATENT\_DIM}}$ and a **decoupled rotary key** $\mathbf{k}_t^R \in \mathbb{R}^{\text{ROPE\_DIM}}$.

### 4.3.2 Mathematical Memory Footprint & Compression Proof
Standard Multi-Head Attention caches:
$$\mathbf{S}_{\text{MHA}} = 2 \times \text{LAYERS} \times H_{\text{KV}} \times D_h \times \text{sizeof}(T) = 2 \times 61 \times 128 \times 128 \times 2 = 3,997,696\text{ bytes/token}$$
DeepSeek MLA caches:
$$\mathbf{S}_{\text{MLA}} = \text{LAYERS} \times (\text{LATENT\_DIM} + \text{ROPE\_DIM}) \times \text{sizeof}(T) = 61 \times (512 + 64) \times 2 = 70,272\text{ bytes/token}$$
$$\text{Memory Compression Factor} = \frac{3,997,696}{70,272} \approx \mathbf{56.88\times\text{ Smaller (98.24\% Reduction)}}$$

### 4.3.3 Operational Semantics & Matrix Absorption
During decode, the key projection matrix $W_{UK}$ is absorbed into the query projection matrix $W_{UQ}$, and the value projection matrix $W_{UV}$ is absorbed into the output projection matrix $W_O$. Caching or materializing expanded keys and values in HBM triggers compiler performance error `E018`.

### 4.3.4 Method Interface & Compressed Token Operations

```velox
impl<const L: usize, const LATENT_DIM: usize, const ROPE_DIM: usize, type T> LatentKVCache<L, LATENT_DIM, ROPE_DIM, T> {
    /// Appends a compressed latent vector c_kv and decoupled rotary key k_r at logical position `step`.
    /// Stores exactly (LATENT_DIM + ROPE_DIM) elements per token per layer, achieving 56.88x compression over MHA.
    fn append_latent(
        &mut self,
        seq_id: usize,
        step: usize,
        c_kv: Tensor<[L, LATENT_DIM], T>,
        k_r: Tensor<[L, ROPE_DIM], T>
    ) -> Result<(), KVCacheOverflowError>;

    /// Obtains an immutable reference to historical compressed latent vectors and decoupled rotary keys.
    /// Passed directly to absorbed WGMMA attention kernels without decompressing full key/value heads into HBM.
    fn borrow_latent_const(
        &self,
        seq_id: usize,
        range: Range<usize>
    ) -> (Tensor<[range.len(), L, LATENT_DIM], T>, Tensor<[range.len(), L, ROPE_DIM], T>);

    /// Obtains exclusive mutable access to a single latent slot at `step`.
    fn borrow_slot_mut(
        &mut self,
        seq_id: usize,
        step: usize
    ) -> LatentSlotMut<L, LATENT_DIM, ROPE_DIM, T>;

    /// Recycles all storage allocated to sequence `seq_id`.
    fn release_sequence(&mut self, seq_id: usize);

    /// Returns the current number of valid tokens stored for sequence `seq_id`.
    fn latent_tokens_count(&self, seq_id: usize) -> usize;
}
```

---

## 4.4 `StreamingSinkKVCache<layers, heads, head_dim, sinks, window, dtype>`

### 4.4.1 Type Signature & Parameters (StreamingLLM)
```velox
struct StreamingSinkKVCache<
    const LAYERS: usize,
    const HEADS: usize,
    const HEAD_DIM: usize,
    const SINKS: usize = 4,
    const WINDOW: usize = 4096,
    type DTYPE = f16
>;
```
- Manages an infinite context stream using a fixed physical memory allocation.
- Conserves the softmax denominator via `SINKS` fixed attention sink tokens at positions $[0, \text{SINKS}-1]$.
- Employs a rolling FIFO ring buffer of size `WINDOW` for recent tokens.

### 4.4.2 Mathematical Addressing & Modulo Ring Arithmetic
The physical slot index $\phi(t)$ for logical token step $t$ is:
$$\phi(t) = \begin{cases}
t & \text{if } t < \text{SINKS} \\
\text{SINKS} + \left( (t - \text{SINKS}) \pmod{\text{WINDOW}} \right) & \text{if } t \ge \text{SINKS}
\end{cases}$$
Relative RoPE positions are re-indexed dynamically inside the attention kernel:
$$\text{Pos}_{\text{rel}}(slot, t) = \begin{cases}
slot & \text{if } slot < \text{SINKS} \\
t - \left( (\phi(t) - slot) \bmod \text{WINDOW} \right) & \text{if } \text{SINKS} \le slot < \text{SINKS} + \text{WINDOW}
\end{cases}$$
where $\phi(t)$ is the active circular write pointer:
$$\phi(t) = \text{SINKS} + \left( (t - \text{SINKS}) \pmod{\text{WINDOW}} \right) \quad (\text{for } t \ge \text{SINKS})$$
and $(x \bmod W) \triangleq ((x \pmod W) + W) \pmod W$ denotes the positive Euclidean residue, ensuring that token age and RoPE relative coordinates remain strictly invariant across circular buffer wraparound.

### 4.4.3 Mathematical Proof of Bounded $O(1)$ Memory
The total physical VRAM footprint across infinite sequence generation ($T \to \infty$) is:
$$\mathbf{S}_{\text{streaming}} \equiv 2 \times \text{LAYERS} \times \text{HEADS} \times (\text{SINKS} + \text{WINDOW}) \times \text{HEAD\_DIM} \times \text{sizeof}(T) = \text{Constant}$$
$$\lim_{T \to \infty} \frac{d}{dT}\left( \mathbf{S}_{\text{streaming}} \right) \equiv 0 \quad \blacksquare$$

### 4.4.4 Method Interface & Infinite-Context Ring Operations

```velox
impl<const L: usize, const H: usize, const D: usize, const SINKS: usize, const WINDOW: usize, type T> StreamingSinkKVCache<L, H, D, SINKS, WINDOW, T> {
    /// Appends new Key and Value vectors at logical token step `step`.
    /// Automatically applies modulo ring arithmetic phi(step) to overwrite the oldest non-sink token.
    /// Memory footprint is strictly O(1) invariant across infinite generation lengths (T -> infinity).
    fn append(
        &mut self,
        step: usize,
        k: Tensor<[L, H, D], T>,
        v: Tensor<[L, H, D], T>
    );

    /// Obtains an active read view over the SINKS attention sinks and the rolling WINDOW tokens.
    /// Encapsulates the dynamic slot remapping for consumption by streaming flash attention kernels.
    fn borrow_active_view(&self) -> StreamingContextRef<L, H, D, SINKS, WINDOW, T>;

    /// Computes the physical slot index phi(step) for logical step `step`.
    /// Maps 0..SINKS-1 to static sink slots, and steps >= SINKS to circular ring slots [SINKS..SINKS+WINDOW-1].
    fn get_slot_index(&self, step: usize) -> usize;

    /// Computes the dynamic relative RoPE position for physical slot `slot` relative to active write pointer phi(step).
    /// Prevents attention score degradation under circular buffer wraparound.
    fn relative_rope_pos(&self, slot: usize, step: usize) -> usize;

    /// Clears the write pointer and resets the cache to initial state.
    fn reset(&mut self);

    /// Returns the constant total physical VRAM footprint in bytes allocated to this cache.
    fn fixed_capacity_bytes(&self) -> usize;
}
```

---

## 4.5 `SpeculativeTreeBlock<max_nodes, max_depth, head_dim, dtype>`

### 4.5.1 Type Signature & Parameters (Medusa, EAGLE-2, SpecInfer)
```velox
struct SpeculativeTreeBlock<
    const MAX_NODES: usize = 64,
    const MAX_DEPTH: usize = 6,
    const HEAD_DIM: usize = 128,
    type DTYPE = f16
>;
```
- Represents a non-linear tree of speculative candidate tokens.
- Encapsulates candidate token IDs, parent pointers, depth offsets, and tree attention masks.

### 4.5.2 Tree Node Topology & Synthesized DAG Mask
Each node $u \in [0, \text{MAX\_NODES}-1]$ is represented by:
```velox
struct TreeNode {
    token_id: u32,
    parent_idx: u16,
    depth: u16,
    rope_pos: u32,
}
```

### 4.5.3 2D Non-Causal Attention Mask Generation Equation
The tree attention mask $M_{\text{tree}} \in \mathbb{R}^{\text{MAX\_NODES} \times \text{MAX\_NODES}}$ is generated via:
$$M_{\text{tree}}[u, v] = \begin{cases}
0.0 & \text{if } v \in \text{Ancestors}(u) \cup \{u\} \\
-\infty & \text{otherwise}
\end{cases}$$
The compiler synthesizes this mask into a 64-bit word bitmask ($1 = \text{attend}, 0 = \text{mask}$), evaluating tree attention in a single fused non-causal FlashAttention kernel.

### 4.5.4 Method Interface & Tree Path Compaction Operations

```velox
impl<const MAX_NODES: usize, const MAX_DEPTH: usize, const HEAD_DIM: usize, type T> SpeculativeTreeBlock<MAX_NODES, MAX_DEPTH, HEAD_DIM, T> {
    /// Inserts a candidate token node into the speculative tree DAG.
    /// Establishes parent linkage, tree depth, and absolute sequence RoPE coordinates.
    /// Returns the assigned node index u in [0, MAX_NODES-1].
    fn add_node(
        &mut self,
        parent_idx: u16,
        token_id: u32,
        depth: u16,
        rope_pos: u32
    ) -> Result<u16, TreeCapacityExceededError>;

    /// Synthesizes the 2D non-causal tree attention mask M_tree in R^{MAX_NODES x MAX_NODES}.
    /// Entries are 0.0 if v in Ancestors(u) union {u}, and -infinity otherwise.
    fn synthesize_tree_mask(&self) -> Tensor<[MAX_NODES, MAX_NODES], T>;

    /// Packs DAG ancestry into an array of 64-bit boolean bitmasks (1 bit per candidate node).
    /// Evaluates tree attention in a single fused non-causal FlashAttention kernel with zero inter-candidate contamination.
    fn synthesize_bitmask(&self) -> Tensor<[MAX_NODES], u64>;

    /// Evaluates verification logits across all candidate tree nodes and extracts the longest valid accepted token path.
    /// Yields accepted token count and the sequence of winning node indices.
    fn resolve_best_path(
        &self,
        logits: Tensor<[MAX_NODES, VOCAB_SIZE], T>
    ) -> SpeculativePath<MAX_DEPTH>;

    /// Commits the winning path nodes into the authoritative KV cache via explicit KV Path Compaction.
    /// Non-winning branch slots are discarded; winning branch tokens are gathered into contiguous sequence slots in O(L) time.
    fn commit_accepted_path(
        &mut self,
        kv: &mut PagedKVCache<32, 32, 128, 16, T>,
        accepted_path: SpeculativePath<MAX_DEPTH>
    );

    /// Resets the tree node count and clears DAG parent pointers for the next speculative iteration.
    fn reset(&mut self);

    /// Returns the active number of candidate nodes currently populated in the tree.
    fn active_nodes_count(&self) -> usize;
}
```

---

# Section 5: Formal Grammar Requirements & AST Node Specifications

Velox source code is stored in `.vlx` files. The grammar is formally specified in Extended Backus-Naur Form (ISO/IEC 14977).

---

## 5.1 Lexical Specifications & Token Categories

### 1. Identifiers & Literals
- `IDENTIFIER`: `[a-zA-Z_][a-zA-Z0-9_]*`
- `DECIMAL_LITERAL`: `[0-9]+`
- `HEX_LITERAL`: `0x[0-9a-fA-F]+`
- `FLOAT_LITERAL`: `[0-9]+\.[0-9]+([eE][+-]?[0-9]+)?`
- `STRING_LITERAL`: `\"([^\"\\]|\\.)*\"`

### 2. Reserved Keywords
```
model, target, hardware, device, vram_capacity, compute_capability, safety_factor,
scope, prefill, decode, chunked_prefill, speculative_verify,
weight, tensor, paged_kv, virtual_kv, latent_kv, streaming_sink, tree_block,
let, mut, const, fn, kernel, return, if, else, for, in, while, yield,
import, safetensors, mmap, absorb_mla_q, absorb_mla_v,
dma_async_h2d, dma_async_d2h, tma_load_async, tma_store_async, mbarrier
```

### 3. Memory Tier Attributes
```
@vram_immutable, @paged_vram, @bump_sram, @host_pinned, @wired_limit_bounded, @warp_specialized, @kernel_fuse
```

### 4. Scalar Type Keywords
```
f32, f16, bf16, fp8_e4m3, fp8_e5m2, fp8_e8m0, int8, int4, u64, u32, u16, u8, i64, i32, i16, i8, bool
```

---

## 5.2 Complete ISO/IEC 14977 EBNF Grammar

```ebnf
(* =========================================================================== *)
(*                      VELOX FORMAL EBNF SPECIFICATION                        *)
(* =========================================================================== *)

Program
    = { ImportDecl }, TargetHardwareBlock, { ModelDecl | GlobalBinding };

ImportDecl
    = "import", STRING_LITERAL, ";" ;

(* ------------------------ Hardware Target Block ---------------------------- *)

TargetHardwareBlock
    = "target", "hardware", "{", { TargetProperty }, "}" ;

TargetProperty
    = TargetDeviceProp
    | TargetVramProp
    | TargetComputeProp
    | TargetSafetyProp ;

TargetDeviceProp
    = "device", ":", STRING_LITERAL, ";" ;

TargetVramProp
    = "vram_capacity", ":", ConstExpression, [ MemoryUnit ], ";" ;

TargetComputeProp
    = "compute_capability", ":", FLOAT_LITERAL, ";" ;

TargetSafetyProp
    = "safety_factor", ":", FLOAT_LITERAL, ";" ;

MemoryUnit
    = "*" , ( "KiB" | "MiB" | "GiB" | "TiB" ) ;

(* -------------------------- Model Declarations ----------------------------- *)

ModelDecl
    = "model", IDENTIFIER, [ ConstGenericParams ], "{", { ModelMember }, "}" ;

ConstGenericParams
    = "<", ConstGenericParam, { ",", ConstGenericParam }, ">" ;

ConstGenericParam
    = "const", IDENTIFIER, ":", IntegerType, [ "=", ConstExpression ] ;

ModelMember
    = WeightDecl
    | KVCacheDecl
    | ScopeDecl
    | FunctionDecl ;

WeightDecl
    = [ "@vram_immutable" ], "weight", IDENTIFIER, ":", TensorType,
      "=", ( SafetensorsBinding | AbsorbBinding ), ";" ;

SafetensorsBinding
    = "safetensors", "(", STRING_LITERAL, ",", STRING_LITERAL, ")" ;

AbsorbBinding
    = ( "absorb_mla_q" | "absorb_mla_v" ), "(", IDENTIFIER, ",", IDENTIFIER, ")" ;

KVCacheDecl
    = "let", [ "mut" ], IDENTIFIER, ":", KVCacheType, ";" ;

(* ------------------------- Execution Scope Blocks -------------------------- *)

ScopeDecl
    = [ ScopeAttribute ], "scope", ScopeKind, [ ConstGenericParams ],
      "(", [ ParameterList ], ")", [ "->", TypeSpec ], BlockStmt ;

ScopeKind
    = "prefill"
    | "decode"
    | "chunked_prefill"
    | "speculative_verify" ;

ScopeAttribute
    = "@kernel_fuse", [ "(", STRING_LITERAL, ")" ]
    | "@warp_specialized", "(", WarpSpecParams, ")" ;

WarpSpecParams
    = IDENTIFIER, "=", ConstExpression, { ",", IDENTIFIER, "=", ConstExpression } ;

(* ----------------------------- Type System --------------------------------- *)

TypeSpec
    = [ MemoryTierAttr ], ( TensorType | KVCacheType | PrimitiveType | CustomType ) ;

MemoryTierAttr
    = "@vram_immutable"
    | "@paged_vram"
    | "@bump_sram"
    | "@host_pinned"
    | "@wired_limit_bounded" ;

TensorType
    = "Tensor", "<", ShapeSpec, ",", ScalarType, [ ",", BlockScaleSpec ], ">" ;

ShapeSpec
    = "[", DimensionExpr, { ",", DimensionExpr }, "]" ;

DimensionExpr
    = ConstExpression | "*" ;

BlockScaleSpec
    = "block_scale", "=", "[", ConstExpression, { ",", ConstExpression }, "]",
      [ ",", "scale_dtype", "=", ScalarType ] ;

KVCacheType
    = PagedKVType
    | VirtualKVType
    | LatentKVType
    | StreamingKVType
    | TreeBlockType ;

GenericArg
    = [ IDENTIFIER, "=" ], ( ConstExpression | ScalarType ) ;

PagedKVType
    = "PagedKVCache", "<", GenericArg, { ",", GenericArg }, ">" ;

VirtualKVType
    = "VirtualContiguousKV", "<", GenericArg, { ",", GenericArg }, ">" ;

LatentKVType
    = "LatentKVCache", "<", GenericArg, { ",", GenericArg }, ">" ;

StreamingKVType
    = "StreamingSinkKVCache", "<", GenericArg, { ",", GenericArg }, ">" ;

TreeBlockType
    = "SpeculativeTreeBlock", "<", GenericArg, { ",", GenericArg }, ">" ;

ScalarType
    = "f32" | "f16" | "bf16" | "fp8_e4m3" | "fp8_e5m2" | "fp8_e8m0"
    | "int8" | "int4" | "u64" | "u32" | "u16" | "u8"
    | "i64" | "i32" | "i16" | "i8" | "bool" ;

IntegerType
    = "usize" | "u64" | "u32" | "u16" | "u8" | "i64" | "i32" | "i16" | "i8" ;

PrimitiveType
    = ScalarType | IntegerType ;

CustomType
    = IDENTIFIER ;

ParameterList
    = Parameter, { ",", Parameter } ;

Parameter
    = [ "mut" ], IDENTIFIER, ":", TypeSpec, [ "=", Expression ] ;

GlobalBinding
    = LetBindingStmt ;

FunctionDecl
    = "fn", IDENTIFIER, [ ConstGenericParams ], "(", [ ParameterList ], ")",
      [ "->", TypeSpec ], BlockStmt ;

(* ------------------------ Statements & Expressions ------------------------- *)

BlockStmt
    = "{", { Statement }, "}" ;

Statement
    = LetBindingStmt
    | AssignmentStmt
    | MemoryTransferStmt
    | ControlFlowStmt
    | YieldStmt
    | ReturnStmt
    | ExpressionStmt ;

LetBindingStmt
    = "let", [ "mut" ], IDENTIFIER, [ ":", TypeSpec ], [ "=", Expression ], ";" ;

LValue
    = IDENTIFIER, { MemberAccess | IndexAccess | SliceAccess } ;

AssignmentStmt
    = LValue, AssignmentOp, Expression, ";" ;

AssignmentOp
    = "=" | "+=" | "-=" | "*=" | "/=" ;

MemoryTransferStmt
    = TmaLoadStmt
    | TmaStoreStmt
    | DmaTransferStmt ;

TmaLoadStmt
    = "tma_load_async", "(", Expression, ",", Expression, ",", Expression, ")", ";" ;

TmaStoreStmt
    = "tma_store_async", "(", Expression, ",", Expression, ")", ";" ;

DmaTransferStmt
    = ( "dma_async_h2d" | "dma_async_d2h" ), "(", Expression, ",", Expression,
      [ ",", Expression ], ")", ";" ;

YieldStmt
    = "yield", "(", [ Expression ], ")", ";" ;

ReturnStmt
    = "return", [ Expression ], ";" ;

ControlFlowStmt
    = IfStmt
    | ForStmt
    | WhileStmt ;

IfStmt
    = "if", "(", Expression, ")", BlockStmt, [ "else", ( IfStmt | BlockStmt ) ] ;

ForStmt
    = "for", IDENTIFIER, "in", Expression, "..", Expression, BlockStmt ;

WhileStmt
    = "while", "(", Expression, ")", BlockStmt ;

ExpressionStmt
    = Expression, ";" ;

(* ------------------------- Expressions & Operators ------------------------- *)

Expression
    = LogicalOrExpr ;

LogicalOrExpr
    = LogicalAndExpr, { "||", LogicalAndExpr } ;

LogicalAndExpr
    = BitwiseOrExpr, { "&&", BitwiseOrExpr } ;

BitwiseOrExpr
    = BitwiseXorExpr, { "|", BitwiseXorExpr } ;

BitwiseXorExpr
    = BitwiseAndExpr, { "^", BitwiseAndExpr } ;

BitwiseAndExpr
    = EqualityExpr, { "&", EqualityExpr } ;

EqualityExpr
    = RelationalExpr, { ( "==" | "!=" ), RelationalExpr } ;

RelationalExpr
    = ShiftExpr, { ( "<" | "<=" | ">" | ">=" ), ShiftExpr } ;

ShiftExpr
    = AdditiveExpr, { ( "<<" | ">>" ), AdditiveExpr } ;

AdditiveExpr
    = MultiplicativeExpr, { ( "+" | "-" ), MultiplicativeExpr } ;

MultiplicativeExpr
    = MatmulExpr, { ( "*" | "/" | "%" ), MatmulExpr } ;

MatmulExpr
    = UnaryExpr, { "@", UnaryExpr } ;

UnaryExpr
    = ( "!" | "~" | "-" | "*" | "&" | "&mut" ), UnaryExpr
    | PostfixExpr ;

PostfixExpr
    = PrimaryExpr, { MemberAccess | IndexAccess | CallAccess | SliceAccess } ;

MemberAccess
    = ".", IDENTIFIER ;

IndexAccess
    = "[", Expression, "]" ;

SliceAccess
    = "[", [ Expression ], "..", [ Expression ], "]" ;

CallAccess
    = "(", [ ArgumentList ], ")" ;

PrimaryExpr
    = IDENTIFIER
    | DECIMAL_LITERAL
    | HEX_LITERAL
    | FLOAT_LITERAL
    | STRING_LITERAL
    | "(", Expression, ")" ;

ArgumentList
    = Argument, { ",", Argument } ;

Argument
    = [ IDENTIFIER, "=" ], Expression ;

ConstExpression
    = Expression ;
```

---

## 5.3 Operator Precedence & Associativity Table

The Velox expression grammar enforces a strict 14-level operator precedence hierarchy:

| Level | Operator Category | Operators | Associativity | Operational Semantics |
|---|---|---|---|---|
| **1 (Highest)** | Primary & Postfix | `.` `[]` `()` `[..]` | Left-to-Right | Member access, array indexing, function/kernel calls, tensor slicing |
| **2** | Unary Operators | `!` `~` `-` `*` `&` `&mut` | Right-to-Left | Logical NOT, bitwise NOT, negation, dereference, borrow, mutable borrow |
| **3** | Matrix Multiplication | `@` | Left-to-Right | Hardware Tensor Core / WGMMA matrix multiplication |
| **4** | Multiplicative | `*` `/` `%` | Left-to-Right | Elementwise multiplication, division, modulo |
| **5** | Additive | `+` `-` | Left-to-Right | Elementwise addition and subtraction |
| **6** | Bitwise Shifts | `<<` `>>` | Left-to-Right | Arithmetic and logical bit shifts |
| **7** | Relational Comparison | `<` `<=` `>` `>=` | Left-to-Right | Numeric inequality checks |
| **8** | Equality Comparison | `==` `!=` | Left-to-Right | Value equivalence and identity checks |
| **9** | Bitwise AND | `&` | Left-to-Right | Bitwise masking |
| **10** | Bitwise XOR | `^` | Left-to-Right | Bitwise difference and address swizzling |
| **11** | Bitwise OR | `\|` | Left-to-Right | Bitwise combination |
| **12** | Logical AND | `&&` | Left-to-Right | Short-circuit boolean conjunction |
| **13** | Logical OR | `\|\|` | Left-to-Right | Short-circuit boolean disjunction |
| **14 (Lowest)** | Assignment Operators | `=` `+=` `-=` `*=` `/=` | Right-to-Left | Value assignment and in-place mutation |

*Note on Assignment Precedence: In Velox, assignments are formally defined as statements (`AssignmentStmt`), not value-yielding expressions. They are cataloged at Precedence Level 14 to document that all Right-Hand Side expressions bind with higher precedence prior to statement termination.*

---

## 5.4 Complete AST Node Specifications (23 Strongly-Typed Nodes)

The Velox Abstract Syntax Tree (AST) is an immutable, strongly-typed tree structure generated by the parser. Every AST node inherits from `ASTNode`.

```
                                  ASTNode
                                     |
    +-----------------+--------------+---------------+-----------------+
    |                 |                              |                 |
DeclNode           TypeNode                      ScopeNode         StmtNode / ExprNode
    |                 |                              |
 TargetNode        TensorTypeNode                 PrefillScopeNode
 ModelNode         PagedKVTypeNode                DecodeScopeNode
 WeightNode        LatentKVTypeNode               ChunkedPrefillScopeNode
                   StreamingKVTypeNode            SpeculativeVerifyScopeNode
```

### 5.4.1 Root & Configuration Nodes

#### 1. `ProgramNode`
- **Description**: Root node of a compiled `.vlx` translation unit.
- **Fields**:
  - `imports: Vec<ImportNode>` — List of imported library/header paths.
  - `target_hardware: TargetHardwareNode` — Target hardware constraint envelope.
  - `model_declarations: Vec<ModelDeclarationNode>` — Model architectures defined in this module.
  - `global_bindings: Vec<LetBindingNode>` — Module-scoped constant bindings.
- **Invariants**: Must contain exactly one `TargetHardwareNode`.

#### 2. `TargetHardwareNode`
- **Description**: Target execution hardware envelope and VRAM limits.
- **Fields**:
  - `device_name: String` — Device descriptor (e.g., `"nvidia_rtx_3050_laptop"`).
  - `vram_capacity_bytes: u64` — Raw physical VRAM limit in bytes.
  - `compute_capability: (u32, u32)` — Major and minor SM version (e.g., `(8, 6)` for Ampere, `(9, 0)` for Hopper).
  - `safety_factor: f64` — Headroom fraction ($\alpha \in (0.0, 1.0]$, default 0.95).
  - `loc: SourceLocation` — Source span for compiler diagnostics.
- **Invariants**: `vram_capacity_bytes > 0`, `0.5 <= safety_factor <= 1.0`.

#### 3. `ModelDeclarationNode`
- **Description**: Declares a neural network serving architecture.
- **Fields**:
  - `name: String` — Model identifier.
  - `generic_params: Vec<ConstGenericParamNode>` — Compile-time dimension parameters (e.g., `B_MAX`, `S_MAX`, `HIDDEN_DIM`).
  - `weights: Vec<WeightDeclarationNode>` — Model weight tensors.
  - `kv_caches: Vec<KVCacheDeclarationNode>` — State caches.
  - `scopes: Vec<ExecutionScopeNode>` — Execution entry points (`prefill`, `decode`, etc.).
  - `loc: SourceLocation`
- **Invariants**: Must declare at least one `prefill` scope and one `decode` scope.

#### 4. `WeightDeclarationNode`
- **Description**: A static model parameter tensor bound to a `.safetensors` file.
- **Fields**:
  - `name: String` — Parameter name.
  - `tier: MemoryTier` — Must evaluate to `MemoryTier::VramImmutable`.
  - `tensor_type: TensorTypeNode` — Const-generic shape and dtype.
  - `file_path: String` — Path to `.safetensors` container.
  - `tensor_key: String` — Key name inside Safetensors metadata dictionary.
  - `loc: SourceLocation`

---

### 5.4.2 Type Nodes

#### 5. `TensorTypeNode`
- **Fields**:
  - `shape: Vec<DimensionNode>` — List of const-generic dimension expressions.
  - `dtype: ScalarType` — Numeric element type.
  - `block_quant_spec: Option<BlockQuantSpec>` — Optional block-scale quantization metadata.
  - `tier: MemoryTier` — One of `VramImmutable`, `PagedVram`, `BumpSram`, `HostPinned`.
  - `loc: SourceLocation`
- **Invariants**: Rank must satisfy $1 \le \text{shape.len()} \le 6$.

#### 6. `PagedKVTypeNode`
- **Fields**:
  - `layers: ConstExprNode` — Attention layer count.
  - `heads: ConstExprNode` — KV attention head count.
  - `head_dim: ConstExprNode` — Head dimension.
  - `block_size: ConstExprNode` — Physical block size ($BS \in \{16, 32, 64\}$).
  - `dtype: ScalarType` — Precision (`f16`, `bf16`, `fp8_e4m3`).
  - `loc: SourceLocation`
- **Invariants**: `block_size` must be a power of two.

#### 6b. `VirtualKVTypeNode`
- **Description**: Represents a virtual memory mapped Key-Value cache allocation (`VirtualContiguousKV`) backed dynamically by 2MB physical pages via CUDA Virtual Memory Management (VMM) APIs (`cuMemAddressReserve`, `cuMemCreate`, `cuMemMap`, `cuMemSetAccess`).
- **Fields**:
  - `layers: ConstExprNode` — Number of transformer attention layers ($L \ge 1$).
  - `heads: ConstExprNode` — Number of Key-Value attention heads ($H \ge 1$).
  - `head_dim: ConstExprNode` — Dimensionality of each attention head ($D_h \in \{64, 128, 256\}$).
  - `max_seq_len: ConstExprNode` — Maximum virtual reservation sequence length ($S_{\max}$, default 32,768).
  - `physical_page_size: ConstExprNode` — Physical backing page allocation granularity in bytes (default $2 \times 1024 \times 1024 = 2,097,152\text{ bytes} = 2.0\text{ MiB}$).
  - `dtype: ScalarType` — Numerical precision (`f16`, `bf16`, `fp8_e4m3`, `fp8_e5m2`).
  - `loc: SourceLocation` — Source span coordinates for compiler diagnostics.
- **Invariants**:
  - `layers > 0`, `heads > 0`, `head_dim > 0`, `max_seq_len > 0`.
  - `physical_page_size == 2097152` (must match physical 2MB Huge Page TLB leaf boundary; verified by compiler diagnostic `E015`).
  - Reserved virtual address space per sequence:
    $$\text{VSize}_{\text{seq}} = 2 \times \text{layers} \times \text{heads} \times \text{max\_seq\_len} \times \text{head\_dim} \times \text{sizeof}(\text{dtype})$$
    must be an integer multiple of `physical_page_size`: $\text{VSize}_{\text{seq}} \pmod{\text{physical\_page\_size}} \equiv 0$.

#### 7. `LatentKVTypeNode`
- **Fields**:
  - `layers: ConstExprNode`
  - `latent_dim: ConstExprNode` — Compressed latent rank ($d_c = 512$).
  - `rope_dim: ConstExprNode` — Decoupled rotary dimension ($d_R = 64$).
  - `dtype: ScalarType`
  - `loc: SourceLocation`

#### 8. `StreamingKVTypeNode`
- **Fields**:
  - `layers: ConstExprNode`, `heads: ConstExprNode`, `head_dim: ConstExprNode`
  - `sinks: ConstExprNode` — Attention sink token count ($S \ge 1$).
  - `window: ConstExprNode` — Sliding window token capacity ($W \ge 64$).
  - `dtype: ScalarType`
  - `loc: SourceLocation`

#### 9. `SpeculativeTreeTypeNode`
- **Fields**:
  - `max_nodes: ConstExprNode` — Tree candidate capacity (e.g., 64).
  - `max_depth: ConstExprNode` — Maximum candidate tree depth (e.g., 6).
  - `head_dim: ConstExprNode`
  - `dtype: ScalarType`
  - `loc: SourceLocation`

---

### 5.4.3 Execution Scope Nodes

#### 10. `PrefillScopeNode`
- **Fields**:
  - `name: String`, `generic_params: Vec<ConstGenericParamNode>`
  - `parameters: Vec<ParameterNode>` — Sequence inputs.
  - `return_type: Option<TypeSpecNode>`
  - `body: BlockStatementNode`
  - `loc: SourceLocation`
- **Invariants**: Emits compute-bound GEMM execution path; zero dynamic allocations.

#### 11. `DecodeScopeNode`
- **Fields**:
  - `name: String`, `generic_params: Vec<ConstGenericParamNode>`
  - `parameters: Vec<ParameterNode>`
  - `return_type: Option<TypeSpecNode>`
  - `body: BlockStatementNode`
  - `loc: SourceLocation`
- **Invariants**: Emits memory-bound GEMV execution path; bounded to sub-80µs host dispatch.

#### 12. `ChunkedPrefillScopeNode`
- **Fields**:
  - `chunk_size: ConstExprNode` — Uniform chunk length $C^* \approx 512$.
  - `max_decode_batch: ConstExprNode` — Interleaved decode capacity.
  - `body: BlockStatementNode`
  - `loc: SourceLocation`

#### 13. `SpeculativeVerifyScopeNode`
- **Fields**:
  - `tree_type: SpeculativeTreeTypeNode`
  - `target_model_ref: IdentifierNode`
  - `body: BlockStatementNode`
  - `loc: SourceLocation`

---

### 5.4.4 Statement Nodes

#### 14. `LetBindingNode`
- **Fields**: `name: String`, `is_mut: bool`, `type_annot: Option<TypeSpecNode>`, `init_expr: ExpressionNode`, `loc: SourceLocation`

#### 15. `AssignmentNode`
- **Fields**: `lhs: LValueNode`, `op: AssignmentOperator`, `rhs: ExpressionNode`, `loc: SourceLocation`

#### 16. `AsyncDmaTransferNode`
- **Fields**: `direction: DmaDirection` (`H2D` or `D2H`), `source: ExpressionNode`, `destination: ExpressionNode`, `signal: Option<ExpressionNode>`, `loc: SourceLocation`

#### 17. `TmaLoadNode`
- **Fields**: `source: ExpressionNode`, `destination: ExpressionNode`, `mbarrier: ExpressionNode`, `loc: SourceLocation`

#### 18. `TmaStoreNode`
- **Fields**: `source: ExpressionNode`, `destination: ExpressionNode`, `loc: SourceLocation`

#### 19. `YieldNode`
- **Fields**: `value: Option<ExpressionNode>`, `loc: SourceLocation`
- **Invariants**: Suspends active fiber and swaps register contexts in 8 ns.

---

### 5.4.5 Expression Nodes

#### 20. `TensorMatmulNode` (`@`)
- **Fields**: `lhs: ExpressionNode`, `rhs: ExpressionNode`, `loc: SourceLocation`
- **Invariants**: Inner contraction dimension must unify at compile time ($\text{Dim}(LHS, -1) == \text{Dim}(RHS, 0)$).

#### 21. `AbsorbMlaNode`
- **Fields**: `projection_q: ExpressionNode`, `projection_k: ExpressionNode`, `loc: SourceLocation`
- **Invariants**: Folds Key matrix into Query matrix offline; emits absorbed weights into binary image.

#### 22. `TensorSliceNode`
- **Fields**: `tensor_expr: ExpressionNode`, `ranges: Vec<SliceRangeNode>`, `loc: SourceLocation`

---

# Section 6: Compiler Verification Rules & Static Proof System

## 6.1 Formal Static Typing Judgments ($\Gamma \vdash e : \tau$)

Let $\Gamma$ represent the lexical type environment mapping identifiers to types, and let $\Delta$ represent the const-generic shape constraint store.

### 1. Matrix Multiplication Contraction Rule
$$\frac{\Gamma; \Delta \vdash A : \text{Tensor}\langle [d_1, \dots, d_{n-1}, K_A], \tau_1 \rangle \quad \Gamma; \Delta \vdash B : \text{Tensor}\langle [K_B, d_{n+1}, \dots, d_m], \tau_2 \rangle \quad \Delta \models \text{InnerDim}(A) \equiv \text{InnerDim}(B)}{\Gamma; \Delta \vdash A @ B : \text{Tensor}\langle [d_1, \dots, d_{n-1}, d_{n+1}, \dots, d_m], \text{Promote}(\tau_1, \tau_2) \rangle}$$

### 2. Disjoint Slice Mutation Rule
$$\frac{\Gamma \vdash kv : \text{PagedKVCache}\langle L, H, D, BS, T \rangle \quad \text{disjoint}(idx_1, idx_2)}{\Gamma \vdash (\text{borrow\_mut}(kv, idx_1), \text{borrow\_const}(kv, idx_2)) : (\text{KVSlotMut}, \text{KVContextRef})}$$

### 3. SRAM Allocation Lifetime Rule
$$\frac{\Gamma \vdash x : \text{@bump\_sram } \tau \quad \text{Escapes}(\text{Scope}(x), x) \equiv \text{False}}{\Gamma \vdash \text{AllocSRAM}(x) : \text{WellTyped}}$$

---

## 6.2 Const-Generic Shape Unification

The shape unification engine resolves symbolic dimension expressions using integer linear programming (Fourier-Motzkin elimination).
Given:
1. Input activation: $X \in \mathbb{R}^{B \times S \times D}$
2. Query weight: $W_Q \in \mathbb{R}^{D \times (H \cdot D_h)}$
3. Key weight: $W_K \in \mathbb{R}^{D \times (H_{KV} \cdot D_h)}$
4. Matrix multiplication unification verifies:
   $$\text{Unify}(D, D) \equiv \text{True}$$
5. Multi-Head Attention Head Dim verification verifies:
   $$\text{Unify}(H \times D_h, D_{\text{model}}) \equiv \text{True}$$
If any symbolic equality fails to resolve, compilation terminates with error `E011`.

---

## 6.3 Disjoint Slice Lending & Lifetime Verification

To guarantee memory safety without runtime reference counting or mutexes, Velox implements **Compile-Time Disjoint Slice Lending**.

**Axiom 1 (Disjoint Borrow Rule)**: Let $C$ be a `PagedKVCache`. Multiple mutable references $\text{ref}_1, \text{ref}_2, \dots, \text{ref}_n$ into $C$ may coexist within the same execution block if and only if their index tuples are statically proven disjoint:
$$\forall i \ne j, \quad (\text{seq}_i \ne \text{seq}_j) \lor (\text{slot}_i \ne \text{slot}_j) \lor (\text{layer}_i \ne \text{layer}_j)$$

If two borrow expressions cannot be proven disjoint at compile time, the second borrow triggers compiler error `E012`.

---

## 6.4 Static VRAM Feasibility Algorithm & Proof Certificate

The memory feasibility solver executes as Pass 6 of the compiler:
1. Traverses all declared model weights and computes byte footprint $S_{\text{weights}}$.
2. Resolves const-generic bounds $B_{\max}$ and $S_{\max}$ to evaluate $S_{\text{kv\_cache}}$.
3. Builds the lifetime interval graph $\mathcal{G}_{\text{int}}$ of all temporary tensors in `prefill`, `decode`, and `speculative_verify`.
4. Runs the Interval Graph Coloring algorithm to find maximum clique weight $S_{\text{velox\_act}}$.
5. Queries target hardware VRAM capacity $VRAM_{\text{budget}}$.
6. Evaluates:
   $$S_{\text{weights}} + S_{\text{kv\_cache}} + S_{\text{velox\_act}} + S_{\text{driver}} \le \alpha \cdot VRAM_{\text{budget}}$$
7. If true, outputs a static proof certificate into the emitted C header file:
   ```c
   // ============================================================================
   // VELOX STATIC MEMORY PROOF CERTIFICATE
   // Target Hardware:               nvidia_rtx_3050_laptop (sm_86)
   // Physical VRAM Capacity:        4096.00 MiB (4,294,967,296 bytes)
   // Compiler Safety Budget (a=0.95):3891.20 MiB (4,080,218,931 bytes)
   // ----------------------------------------------------------------------------
   // Static Memory Allocation Breakdown:
   //   - Model Weights (S_weights):  2150.00 MiB (2,254,438,400 bytes) [55.25%]
   //   - KV Cache (S_kv, B=4, S=2048): 480.00 MiB (  503,316,480 bytes) [12.34%]
   //   - Velox Activation Arena (S_act):    220.50 MiB (  231,211,008 bytes) [ 5.67%]
   //   - Driver & System Reserve:    1024.00 MiB (1,073,741,824 bytes) [26.32%]
   // ----------------------------------------------------------------------------
   // Total Statically Bound VRAM:   3874.50 MiB (4,062,714,240 bytes)
   // Safety Budget Margin:          +16.70 MiB ( 17,504,691 bytes headroom)
   // Physical Capacity Margin:      +221.50 MiB (232,253,056 bytes headroom)
   // Status: VERIFIED SAFE - RUNTIME OUT-OF-MEMORY IS MATHEMATICALLY IMPOSSIBLE
   // ============================================================================
   ```

---

## 6.5 Diagnostic Error Catalog (`E001` to `E025`)

The compiler maintains an exhaustive diagnostic catalog with deterministic remediation guidance:

| Error Code | Diagnostic Name | Category | Trigger Condition & Formal Description |
|---|---|---|---|
| **E001** | `DYNAMIC_HEAP_ALLOCATION_PROHIBITED` | Invariant 1 Violation | Invocation of dynamic allocator (`malloc`, `free`, `cudaMalloc`, `new`) inside serving execution scopes. |
| **E002** | `TARGET_VRAM_BUDGET_EXCEEDED` | Invariant 2 Violation | Statically computed memory $S_{\text{total}} > \alpha \cdot VRAM_{\text{budget}}$. |
| **E003** | `BLOCKING_HOST_SYNC_IN_DECODE` | Invariant 3 Violation | Synchronous device-to-host read (`.item()`, `cudaStreamSynchronize`, `cudaMemcpy`) inside the decode loop. |
| **E004** | `ILLEGAL_MEMORY_TIER_CAST` | Memory Tier Violation | Attempting to cast an immutable tier (`@vram_immutable`) to a mutable tier (`@paged_vram`). |
| **E005** | `INVALID_DIRECT_DMA_PATH` | Memory Tier Violation | Initiating DMA transfer between incompatible tiers without intermediate staging (e.g., `@bump_sram` to `@host_pinned`). |
| **E006** | `SRAM_REFERENCE_ESCAPES_SCOPE` | Lifetime & Aliasing | Pointer to `@bump_sram` survives past the terminating brace of the kernel or thread-block scope. |
| **E007** | `UNSUPPORTED_HARDWARE_TARGET` | Target Configuration | Declaring a hardware compute capability not supported by the code generator (e.g., SM < 7.0). |
| **E008** | `MISSING_EXECUTION_SCOPE` | Architecture Invariant | Model declaration omits mandatory `prefill` or `decode` execution scope. |
| **E009** | `INVALID_CONST_GENERIC_PARAM` | Const-Generics | Declaring non-scalar or negative values for const-generic tensor dimensions. |
| **E010** | `TENSOR_MATMUL_DIM_MISMATCH` | Shape Verification | Inner contraction dimensions do not unify in matrix multiplication: $\text{Dim}(A, -1) \ne \text{Dim}(B, 0)$. |
| **E011** | `SYMBOLIC_SHAPE_UNIFICATION_FAILURE` | Shape Verification | Symbolic dimension constraints cannot be satisfied across an operator DAG. |
| **E012** | `KV_CACHE_BORROW_CONFLICT` | Lifetime & Aliasing | Two overlapping mutable borrows of the same KV cache block/slot within the same execution scope. |
| **E013** | `SAFETENSORS_KEY_NOT_FOUND` | Weight Binding | Weight identifier declared in model cannot be located in the provided `.safetensors` file. |
| **E014** | `SAFETENSORS_DTYPE_MISMATCH` | Weight Binding | Data type in `.safetensors` header differs from declared `TensorType` without explicit quantizer. |
| **E015** | `BLOCK_SIZE_NOT_POWER_OF_TWO` | KV Cache Architecture | `PagedKVCache` block size parameter is not a power of two ($BS \notin \{16, 32, 64\}$). |
| **E016** | `UNBOUNDED_STREAMING_WINDOW` | KV Cache Architecture | Declaring `StreamingSinkKVCache` with non-constant or zero sink/window parameters. |
| **E017** | `SPECULATIVE_TREE_DEPTH_EXCEEDED` | Speculative Decoding | Tree DAG candidate verification depth exceeds declared `MAX_DEPTH`. |
| **E018** | `UNABSORBED_MLA_DECOMPRESSION` | Performance Invariant | Materializing full $H \times D$ key/value tensors from a `LatentKVCache` instead of using absorbed projections. |
| **E019** | `HOST_PINNED_BUFFER_OVERFLOW` | Memory Tier Violation | Attempting to write more tokens into `@host_pinned` ring buffer than allocated capacity. |
| **E020** | `ASYNC_TMA_BARRIER_MISUSE` | Concurrency & Async | Issuing `tma_load_async` without an associated hardware `mbarrier` synchronization. |
| **E021** | `NON_CONSTANT_LOOP_BOUND_IN_FIBER` | Invariant 1 Violation | Using dynamically sized loop bounds inside the token dispatch fiber without static trip-count bounds. |
| **E022** | `DUPLICATE_MODEL_MEMBER_NAME` | Grammar & AST | Declaring two weights, KV caches, or scopes with identical identifiers inside a model. |
| **E023** | `UNDEFINED_IDENTIFIER_REFERENCE` | Frontend Semantics | Referencing a variable or tensor identifier not present in the current lexical symbol table. |
| **E024** | `REPRESENTATION_PRECISION_LOSS` | Quantization & Dtype | Downcasting floating-point precision (e.g., `f32` to `fp8`) without an explicit quantizer operator. |
| **E025** | `UNALIGNED_SCRATCHPAD_ACCESS` | Hardware Architecture | Emitting memory access that violates 128-byte hardware transaction alignment on global VRAM. |

---

## 6.6 Concrete Diagnostic Error Walkthroughs

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
  the SRAM Escape Invariant (Definition 5). SMEM is discarded at kernel exit.

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

*End of Velox Master Requirements Specification (`VELOX-SPEC-2026-STEP2-REQ-001`).*
