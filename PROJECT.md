# Project: Velox Systems Programming Language — Step 1 Serving Flaws Diagnostic

## Architecture & Scope Overview
Velox is a domain-specific systems programming language engineered for high-throughput, deterministic, zero-overhead Large Language Model (LLM) serving.
Step 1 establishes the empirical and theoretical foundation of Velox by conducting an exhaustive diagnostic of existing LLM serving ecosystems to catalog all architectural failure points, runtime bottlenecks, and safety gaps.

Deliverable target: `C:\Sebastian\Project\Velox\velox_step1_serving_flaws_diagnostic.md`

## Feature Inventory
Every ecosystem, runtime mechanism, and failure mode investigated during the Survey phase is mapped to its diagnostic deliverable chapter and subsequent Velox feature design.

| # | Feature / Ecosystem | Description & Failure Mechanisms | Milestone | Source |
|---|---------------------|----------------------------------|-----------|--------|
| 1 | Python: vLLM & TGI | Cyclic GC stop-the-world pauses (15-85ms), reference counting churn, pymalloc arena memory bloat, GIL contention blocking GPU dispatch, multiprocessing/Ray IPC serialization latency. | M1 | Survey (Explorer 1) |
| 2 | Python: PyTorch Caching Allocator & vLLM/SGLang Memory Managers | Checkerboard VRAM fragmentation under continuous dynamic batching, PagedAttention block table host-to-device marshalling overhead, SGLang RadixAttention tree traversal and CUDA event sync bottlenecks. | M1 | Survey (Explorer 1) |
| 3 | Python: FFI & Dispatch Latency | PyTorch C++ Dispatcher / ATen unboxing (3.5-7.0µs/kernel, 52% of decode cycle time lost to host CPU), synchronous `.item()` sampling pipeline drains. | M1 | Survey (Explorer 1) |
| 4 | C/C++: llama.cpp (GGML) | Lack of compile-time shape safety, runtime assertion crashes (`GGML_ASSERT`), graph destruction and reallocation on every decode step, bump scratchpad slot fragmentation requiring blocking defragmentation (`llama_kv_cache_defrag`). | M1 | Survey (Explorer 2) |
| 5 | C/C++: TensorRT-LLM | Worst-case workspace over-allocation via `IOptimizationProfile`, lack of compile-time hardware VRAM budgeting leading to CUDA OOMs, symmetric KV block allocation in Tensor Parallelism causing asymmetric exhaustion deadlocks. | M1 | Survey (Explorer 2) |
| 6 | Rust: Candle | Dynamic DAG fan-out violating single ownership (`Arc<RwLock<Storage>>` churn), aliasing XOR mutability preventing in-place mutation, `ConcatKvCache` triggering $O(N^2)$ reallocation and copying disasters, unsafe GPU FFI bypasses ("safety theater"). | M1 | Survey (Explorer 2) |
| 7 | Rust: Mistral.rs | `Arc<Mutex<Sequence>>` lock contention in continuous batching, thread synchronization overhead, reliance on external C/CUDA unsafe kernels without compile-time tensor verification. | M1 | Survey (Explorer 2) |
| 8 | Hardware: CUDA & Driver Layer | Host-device kernel launch latency (3-10µs/kernel, 800 kernels/token = 2.4-3.2ms driver submission time, up to 40% of decode latency), CUDA Graph capture collapse under dynamic continuous batching, graph bucketing VRAM explosion and startup delays. | M1 | Survey (Explorer 3) |
| 9 | Hardware: OpenAI Triton | Runtime JIT compilation pauses (200ms-5.0s per kernel) upon unseen dynamic shapes, autotuning jitter during live inference, Python FFI parameter marshaling overhead (15-30µs/kernel). | M1 | Survey (Explorer 3) |
| 10 | Mojo: Modular MAX | Affine borrow checker (`borrowed` vs `inout`) inability to handle concurrent reading and tail-mutation in paged KV pools or inter-layer scratchpad reuse; forced fallback to `UnsafePointer`; closed-source runtime lock-in. | M1 | Survey (Explorer 3) |
| 11 | Modern: MLC-LLM (TVM Unity) | Symbolic shapes (`tir.Var`) degrading TensorIR GPU scheduling (register spilling, bank conflicts), Relax VM interpreter dispatch adding 1.2-2.4ms/token overhead. | M1 | Survey (Explorer 3) |
| 12 | Modern: Apple MLX | Lazy computational DAG allocation/traversal in C++ heap (2-5ms/token CPU overhead), Unified Memory Architecture (UMA) Mach VM page-out and swap paging death spiral (400x slowdown). | M1 | Survey (Explorer 3) |
| 13 | Cross-Cutting Failure Taxonomy | Exhaustive cross-matrix structuring failures across the 5 core categories: GC/GIL, Memory Fragmentation, Compile-Time Safety/VRAM Budgeting, Borrow Checker/Alias Buffers, and FFI/Serialization Overhead. | M1 | Synthesis |
| 14 | Velox Language Feature Mapping | Direct technical mapping from discovered flaws to Velox language rules: Velox Static Memory Planner, const-generic tensor shapes, compile-time VRAM feasibility proof, native `PagedKVCache` type, standalone C99/CUDA codegen. | M1 | Master Plan & Synthesis |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|--------------|--------|
| M1 | Step 1 Serving Flaws Diagnostic | Author comprehensive diagnostic document at `velox_step1_serving_flaws_diagnostic.md` synthesizing all 14 features, verify via 2 Reviewers, 2 Challengers, and 1 Forensic Auditor | Survey Complete | DONE |

## Interface Contracts & Document Structure
The deliverable `C:\Sebastian\Project\Velox\velox_step1_serving_flaws_diagnostic.md` must follow this comprehensive, uncompromised structure:
- **Title & Executive Summary**: Context, mission, high-level comparative findings, execution breakdown.
- **Section 1: Python Serving Ecosystem (vLLM, TGI, SGLang, CPython runtime)**:
  - CPython memory model, cyclic GC, reference counting churn, pymalloc arena bloat.
  - GIL lock contention, asyncio/FastAPI vs GPU engine threads, multiprocessing/Ray IPC.
  - PyTorch CUDA Caching Allocator fragmentation under continuous batching.
  - KV Cache block managers: vLLM PagedAttention table marshalling, SGLang RadixAttention tree sync.
  - PyTorch ATen dispatcher & FFI boundary penalties, synchronous token sampling stalls.
- **Section 2: C/C++ Serving Ecosystem (llama.cpp, TensorRT-LLM)**:
  - Lack of compile-time shape safety and static dimension checking.
  - Lack of compile-time VRAM hardware budgeting (Weights + KV + Scratchpad).
  - Dynamic shape handling & graph recreation in GGML vs TRT-LLM IOptimizationProfile.
  - Memory fragmentation in native continuous batching, GGML scratchpad defrag, TRT-LLM asymmetric TP KV exhaustion.
- **Section 3: Rust Serving Ecosystem (Candle, Mistral.rs)**:
  - Borrow checker conflicts with dynamic computation DAGs (`Arc<RwLock<Storage>>` churn).
  - Aliasing XOR mutability and the in-place mutation dilemma: `ConcatKvCache` $O(N^2)$ reallocation disasters.
  - Continuous batching lock contention in Mistral.rs (`Arc<Mutex<Sequence>>`).
  - Unsafe boundary proliferation and GPU "safety theater".
- **Section 4: Hardware, Kernel & Modern Runtimes (CUDA, Triton, Mojo/MAX, MLC-LLM, MLX)**:
  - Host-device serialization, CUDA driver submission bottlenecks.
  - CUDA Graph capture breakdown under dynamic shapes & batch sizes; graph bucketing VRAM overhead.
  - Triton JIT compilation latency, autotuning jitter, Python FFI parameter marshaling.
  - Mojo borrow checker limitations on paged KV caches and scratchpad reuse; closed-source MAX engine.
  - MLC-LLM TVM Unity symbolic shape scheduling degradation and Relax VM dispatch overhead.
  - Apple MLX lazy DAG building costs and Unified Memory Architecture (UMA) swap paging death spiral.
- **Section 5: Cross-Cutting Failure Taxonomy Matrix**:
  - Detailed matrix evaluating all 10 frameworks across the 5 failure categories.
- **Section 6: Direct Architecture & Language Mandates for Velox**:
  - Direct flaw-to-feature mapping to the Velox language design and compiler passes.

## Code & File Layout
- Project Deliverable: `C:\Sebastian\Project\Velox\velox_step1_serving_flaws_diagnostic.md`
- Master Workflow Plan: `C:\Sebastian\Project\Velox\velox_workflow_plan.md`
- Survey Reports:
  - `C:\Sebastian\Project\Velox\.agents\teamwork\teamwork_explorer_survey_1\analysis.md`
  - `C:\Sebastian\Project\Velox\.agents\teamwork\teamwork_explorer_survey_2\analysis.md`
  - `C:\Sebastian\Project\Velox\.agents\teamwork\teamwork_explorer_survey_3\analysis.md`
- Orchestration Metadata: `C:\Sebastian\Project\Velox\.agents\teamwork\orchestrator\`
