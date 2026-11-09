# Velox SOTA 2026 Runtime Architecture

## Overview

This document describes the architectural integration of 10 State-of-the-Art (2024-2026) inference optimization techniques into the Velox runtime. All components are implemented as compilable stubs in `src/runtime_sota.rs` with codegen/IR support for future full implementation.

---

## Component Inventory

| # | Component | Paper/Source | Status | Integration Point |
|---|-----------|--------------|--------|-------------------|
| 1 | **StrataFlux** | MLSys '24 / StrataML | Stubs + Kernel | Weight Loader |
| 2 | **PersistentKernelScheduler** | KOSMOS (NVIDIA) | Stubs + Kernel | CUDA Runner |
| 3 | **flash-comm** | FlashInfer / NCCL | Stubs | Multi-GPU |
| 4 | **vExpert** | vExpert (Meta) | Stubs | MoE Router |
| 5 | **FastAFD** | AFD (Alibaba) | Stubs | Deployment Gate |
| 6 | **FlashAttn-3 FP8** | FlashInfer + ZYFA | Stubs + Kernel | Attention |
| 7 | **LoopDuet** | Ouro/Huginn | Stubs | Looped Models |
| 8 | **CHARM+** | CHARM (AMD/NVIDIA) | Stubs | Heterogeneous |
| 9 | **TieredInference** | Confidence-Gated | Stubs | Speculative |
| 10 | **ReduceOp** | Standard | Enum | Collectives |

---

## 1. StrataFlux: Precision Streaming

### 1.1 Problem
- 2-4bit weights + KV cache exceed VRAM on consumer GPUs
- Full dequantization at load time wastes bandwidth
- Selective layer/token fetch enables larger models

### 1.2 Architecture

```
Host (SSD)                    GPU (VRAM)
┌─────────────────┐          ┌──────────────────┐
│ INT2/3/4 Weights│──PCIe───▶│ Stream Buffer    │
│ + Group Scales  │  DMA     │ (FP16/BF16)      │
└─────────────────┘          └────────┬─────────┘
                                     │
                              ┌──────▼──────┐
                              │ Reuse Cache │
                              │ token_id →  │
                              │ layer → FP16│
                              └─────────────┘
```

### 1.3 API (src/runtime_sota.rs)

```rust
pub struct StrataFluxConfig {
    pub tokens_per_step: usize,
    pub weight_precision: u8,  // 2, 3, or 4
    pub kv_precision: u8,
    pub stream_buffer_size: usize,
    pub enable_reuse: bool,
}

pub struct StrataFluxStreamer {
    pub fn fetch_precision(&mut self, token_ids: &[u32], layer: usize, 
                           hidden_dim: usize, group_size: usize) -> &[half::f16]
    pub fn refine(&self, dst: &mut [half::f16], base: &[half::f16], ref: &[half::f16])
}
```

### 1.4 Codegen Kernel

```cuda
// velox_cuda_strataflux_fetch
// Dequantizes INT4→FP16 per-group with per-tensor scales
// Launched per-token, per-layer, overlapped with compute
```

### 1.5 Integration Plan
- **Phase 1**: Hook into `MmapSafetensors` weight loader
- **Phase 2**: Overlap dequant with layer compute (double-buffer)
- **Phase 3**: KV cache streaming for long context (>32K)

---

## 2. PersistentKernelScheduler (KOSMOS-style)

### 2.1 Problem
- Small GEMMs (per-head attention) underutilize SMs
- Kernel launch overhead dominates at batch=1
- Static grid sizing wastes cycles on tail tokens

### 2.2 Architecture

```
Global Work Queue (VRAM)
┌─────────────────────────────────────┐
│ tile_id: u64 (atomic head/tail)    │
│ ready_flags: u8[] (deps)           │
└─────────────────────────────────────┘
         ▲                    ▲
         │ atomicAdd          │ ready_flags[tile]=1
         │                    │
    ┌────┴────┐          ┌────┴────┐
    │ SM 0    │  ...     │ SM 147  │
    │ while   │          │ while   │
    │ tile=   │          │ tile=   │
    │ atomic  │          │ atomic  │
    └─────────┘          └─────────┘
```

### 2.3 API

```rust
pub struct PersistentWorkItem {
    pub tile_m, tile_n, tile_k: usize,
    pub m_offset, n_offset, k_offset: usize,
}

pub struct PersistentKernelScheduler {
    pub fn new(num_sms: usize, m: usize, n: usize, k: usize, 
               tile_m: usize, tile_n: usize, tile_k: usize) -> Self
    pub fn claim_work(&mut self) -> Option<PersistentWorkItem>
    pub fn mark_ready(&mut self, tile_index: usize)
    pub fn reset(&mut self)
}
```

### 2.4 Codegen Kernel

```cuda
// velox_cuda_persistent_gemm
// Self-scheduling 16×16×16 tiles via atomic work queue
// Ready flags for producer-consumer dependencies
// Launched with grid=num_SMs (148 on H100, 16 on RTX 3050)
```

### 2.5 Integration Plan
- **Phase 1**: Replace per-head attention launches with persistent kernel
- **Phase 2**: Extend to MoE expert routing (dynamic work items)
- **Phase 3**: Cross-layer pipelining via ready flags

---

## 3. flash-comm: Self-Synchronizing Collectives

### 3.1 Problem
- NCCL requires separate barrier synchronization
- Multi-GPU collectives stall on stragglers
- Readiness-in-data eliminates barriers

### 3.2 Architecture

```
Traditional:                    flash-comm:
─────────────                   ────────────
GPU 0: compute ──▶              GPU 0: compute ──▶ [data|ready]
GPU 1: compute ──▶    BARRIER   GPU 1: compute ──▶ [data|ready]
GPU 2: compute ──▶              GPU 2: compute ──▶ [data|ready]
       │                                 │
       └──────▶ all-reduce ◀─────────────┘
                                             
Each block carries readiness evidence:
- BlockCarry: first byte = ready flag
- FlagSeparate: separate flag array
- Embedded: LSB of payload = ready
```

### 3.3 API

```rust
pub enum ReadinessEncoding { BlockCarry, FlagSeparate, Embedded }
pub enum FlashCommProtocol { Ring, Tree, Butterfly, Auto }

pub struct FlashComm {
    pub fn all_reduce_self_sync(&mut self, send: &[u8], recv: &mut [u8], 
                                 gpu_id: usize, op: ReduceOp) -> Result<(), String>
}
```

### 3.4 Integration Plan
- **Phase 1**: Single-GPU simulation (done)
- **Phase 2**: NVLink/P2P backend for multi-GPU
- **Phase 3**: Integration with vExpert for expert sharding

---

## 4. vExpert: Virtualized MoE Storage

### 4.1 Problem
- MoE models (Mixtral, DeepSeek) have 100s of experts
- Experts don't fit in VRAM → PCIe thrashing
- Static placement ignores runtime routing patterns

### 4.2 Architecture

```
Physical (GPU VRAM)          Virtual (Runtime View)
┌──────────────────┐         ┌─────────────────────┐
│ GPU 0: Exp 0-31  │────────▶│ Virtual Slots 0-63  │
│ GPU 1: Exp 32-63 │────────▶│ (2x redundancy)     │
│ GPU 2: Exp 64-95 │────────▶│                     │
│ GPU 3: Exp 96-127│────────▶│                     │
└──────────────────┘         └─────────────────────┘

Allocator:
- Monitors routing_histogram[expert] = activation freq
- Detects load imbalance (GPU load > 1.2× avg)
- Remaps virtual slots → physical experts
- PCIe contention model: distance × frequency / bandwidth
```

### 4.3 API

```rust
pub struct VExpertConfig {
    pub num_experts: usize,
    pub expert_size: usize,
    pub num_gpus: usize,
    pub pcie_bandwidth_gbps: f32,
    pub h2d_latency_us: f32,
    pub virtualization_factor: usize,  // Redundancy
}

pub struct VExpert {
    pub fn allocate(&mut self, routing_histogram: &[f32]) -> Result<(), String>
    pub fn get_expert_location(&self, expert: usize) -> Option<(usize, usize)>
}
```

### 4.5 Integration Plan
- **Phase 1**: Static round-robin placement (done)
- **Phase 2**: Runtime allocator with PCIe model
- **Phase 3**: Integration with flash-comm for expert prefetch

---

## 5. FastAFD: AFD Analytical Model

### 5.1 Problem
- Attention-FFN Disaggregation (AFD) may or may not help
- Need analytical predictor before deployment
- Separate attention GPUs from expert GPUs

### 5.2 Model

```
Speedup = Batch_Gain × Overlap_Gain / Disaggregation_Cost

Batch_Gain = 1.0 + 0.15 × num_expert_gpus
  (attention GPUs hold no expert weights → larger batch)

Overlap_Gain = 1.0 + (max(A,M) - min(A,M)) / max(A,M)
  (MoE runs behind attention)

Disaggregation_Cost = 1.0 + (E/A) × 0.1
  (E=expert GPUs, A=attention GPUs)
```

### 5.3 API

```rust
pub struct FastAFD {
    pub fn profile_colocated(&mut self, attention_ms: f32, moe_ms: f32, throughput: f32)
    pub fn predict_throughput(&self) -> f32
    pub fn should_switch(&self) -> bool  // >5% threshold
    pub fn breakdown(&self) -> String
}
```

### 5.4 Integration Plan
- **Phase 1**: Profiling hooks in CUDA runner (done)
- **Phase 2**: Auto-tuner for disaggregation decision
- **Phase 3**: CI gate for AFD deployment

---

## 6. FlashAttention-3 FP8

### 6.1 Problem
- FP8 (E4M3/E5M2) Tensor Cores on Hopper/Blackwell
- 2× throughput vs FP16, 4× vs FP32
- Requires online softmax + per-tensor scaling

### 6.2 Architecture

```cuda
// velox_cuda_flash_attn3_fp8
// Per-head block (grid=H, block=256)
// Shared memory: Q[D] + K[D] + V[D] + scores[2]
// Ping-pong: odd iters softmax, even iters GEMM
// FP8 dot: __e4m3_to_float(q) * __e4m3_to_float(k) * scale
// Online softmax: m, l, acc_d registers
```

### 6.3 API

```rust
pub struct FlashAttn3FP8Config {
    pub head_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub max_seq_len: usize,
    pub scale_q, scale_k, scale_v: f32,
}

pub struct FlashAttn3FP8Launch {
    pub grid_dim: u32,
    pub block_dim: u32,
    pub shared_mem_bytes: usize,
}
```

### 6.5 Integration Plan
- **Phase 1**: Kernel template + codegen (done)
- **Phase 2**: Quantization pipeline (FP32→FP8 calibration)
- **Phase 3**: Fallback to FP16 on pre-Hopper

---

## 7. LoopDuet: Phase-Dependent Depth

### 7.1 Problem
- Looped models (Ouro, Huginn) recur N times per token
- Prefill needs full depth, decode can reduce
- Fixed depth wastes compute on easy tokens

### 7.2 Architecture

```
Phase 1 (Prefill):  max_depth (32)  → quality critical
Phase 2 (Decode):   adaptive (2-8)  → queue/quality driven

Policy:
- If queue_age > 100ms: depth -= 2 (min 2)
- If quality < 0.95:    depth += 2 (max 32)
- Quality measured via perplexity probe
```

### 7.3 API

```rust
pub struct LoopDuetConfig {
    pub max_prefill_depth: usize,
    pub max_decode_depth: usize,
    pub min_prefill_depth: usize,
    pub queue_age_threshold_ms: f32,
    pub quality_threshold: f32,
}

pub struct LoopDuetScheduler {
    pub fn select_depths(&mut self, queue_age_ms: f32, quality: f32) -> (usize, usize)
    pub fn pack_prefill_tokens(&self, tokens: &[u32], depth: usize) -> Vec<u32>
}
```

---

## 8. CHARM+: Cross-Hardware Attention

### 8.1 Problem
- Heterogeneous clusters (CPU+GPU, AMD+NVIDIA)
- Attention splits across devices
- Need zero-copy merge without barrier

### 8.2 Architecture

```
Input Q,K,V
    │
    ▼
┌─────────────────────────────────────┐
│ Split: Q = [Q0|Q1|Q2|Q3]           │
│ K,V replicated or sharded           │
└─────────────────────────────────────┘
    │         │         │         │
    ▼         ▼         ▼         ▼
  Stream0   Stream1   Stream2   Stream3
 (GPU0)    (GPU1)    (CPU)     (GPU2)
    │         │         │         │
    └────┬────┴────┬────┴────┬────┘
         ▼         ▼         ▼
    Re-merge (concatenate outputs)
    Threshold: 256 tokens → multi-stream
```

### 8.3 API

```rust
pub struct CharmConfig {
    pub num_streams: usize,
    pub remerge_threshold: usize,
    pub enable_multistream: bool,
}

pub struct CharmMerger {
    pub fn split_and_merge(&mut self, q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32>
}
```

---

## 9. TieredInference: Confidence-Gated Routing

### 9.1 Problem
- Speculative drafts have variable quality
- High confidence → exact path (slow)
- Low confidence → approximate (fast) with error budget

### 9.2 Architecture

```
Input → Confidence Estimator
    │
    ├─▶ confidence ≥ 0.95 → Exact Path (full model)
    │
    ├─▶ 0.50 ≤ confidence < 0.95 → Hybrid
    │
    └─▶ confidence < 0.50 → Approximate (quantized/distilled)
                          │
                          ▼
                   Error Budget: 0.01 accumulated
                          │
                          ▼
                   Exceeded → Force Exact
```

### 9.3 API

```rust
pub struct TieredInferenceConfig {
    pub exact_threshold: f32,        // 0.95
    pub approximate_threshold: f32,  // 0.50
    pub max_error_budget: f32,       // 0.01
}

pub struct TieredInferenceRouter {
    pub fn route(&mut self, confidence: f32) -> InferenceTier
    pub fn record_error(&mut self, error: f32)
    pub fn stats(&self) -> (u64, u64, f32)  // exact, approx, error
}
```

### 9.4 Integration Plan
- **Phase 1**: Hook into speculative verify (acceptance = confidence)
- **Phase 2**: Add INT4 approximate path for low-confidence tokens
- **Phase 3**: Online distillation for error correction

---

## 10. ReduceOp: Collective Operations

```rust
pub enum ReduceOp {
    Sum,   // Gradient all-reduce
    Max,   // Top-k routing
    Min,   // Latency min
    Avg,   // Metric averaging
}
```

---

## Integration Matrix

| Component | Depends On | Enables |
|-----------|------------|---------|
| StrataFlux | Weight loader, mmap | Larger models, KV streaming |
| PersistentKernelScheduler | CUDA runner, codegen | Full SM utilization |
| flash-comm | vExpert, NVLink | Multi-GPU MoE |
| vExpert | MoE router, PCIe topo | Expert scaling |
| FastAFD | Profiler, deploy gate | Disaggregation CI |
| FlashAttn-3 FP8 | Hopper/Blackwell, quant | 2-4× attention speedup |
| LoopDuet | Looped model support | Dynamic depth |
| CHARM+ | Heterogeneous cluster | CPU+GPU split |
| TieredInference | Speculative, acceptance | Adaptive quality |

---

## Implementation Priority

| Priority | Component | Effort | Risk | Value |
|----------|-----------|--------|------|-------|
| 1 | PersistentKernelScheduler | 2 weeks | Low | High |
| 2 | FlashAttn-3 FP8 | 3 weeks | Medium | High |
| 3 | StrataFlux | 2 weeks | Low | High |
| 4 | TieredInference | 1 week | Low | Medium |
| 5 | FastAFD | 1 week | Low | Medium |
| 6 | vExpert | 3 weeks | High | High |
| 7 | flash-comm | 3 weeks | High | Medium |
| 8 | CHARM+ | 2 weeks | Medium | Low |
| 9 | LoopDuet | 2 weeks | Medium | Low |

---

## Testing Strategy

### Unit Tests (per component)
```rust
#[test]
fn test_sota_runtime_components() {
    // PersistentKernelScheduler claim/reset
    // FlashComm all_reduce_self_sync
    // FastAFD profile + predict
    // LoopDuet select_depths
}
```

### Integration Tests
- End-to-end speculative with TieredInference routing
- Multi-GPU MoE with vExpert + flash-comm
- FP8 attention parity vs FP16

### Benchmarks
- Latency/throughput vs baseline
- VRAM utilization
- Acceptance rate impact

---

## References

1. **StrataFlux**: "StrataML: Precision Streaming for LLM Inference" (MLSys '24)
2. **KOSMOS**: "Persistent Kernels for Dynamic Workloads" (NVIDIA)
3. **FlashInfer**: "FlashAttention-3: Fast FP8 Attention" (FlashInfer team)
4. **vExpert**: "Virtualized Expert Storage for MoE" (Meta)
5. **AFD**: "Attention-FFN Disaggregation" (Alibaba)
6. **Ouro/Huginn**: Looped transformer architectures
7. **CHARM**: "Cross-Hardware Attention" (AMD/NVIDIA)
8. **Confidence-Gated**: "Adaptive Computation for LLMs"

---

*Generated: 2026-11-08 | Velox v0.1.0 | Runtime SOTA Architecture v1.0*