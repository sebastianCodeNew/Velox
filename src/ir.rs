//! Velox Intermediate Representation — VIR (`veloxc::ir`)
//!
//! Implements Step 7 SSA Intermediate Representation:
//! - Lowers validated `TypedProgram` and `MemoryPlan` into Static Single Assignment (SSA)
//!   basic blocks and virtual registers bound to 256-byte aligned static byte offsets
//!   in the Velox Activation Arena (`(scratchpad_base + OFFSET)`).
//! - Represents both unfused fine-grained tensor operations and fused SOTA super-kernels
//!   (Fused RMSNorm+QKV+RoPE+Attention and Fused OutProj+Residual+SwiGLU+Residual).
//!
//! Zero external dependencies (`std` only).

use crate::ast::*;
use crate::memory_planner::{align_up_256, MemoryPlan, PlannedKVCachePool, PlannedWeightOffset};
use crate::sema::{TypedModel, TypedProgram};
use crate::types::{DmaDirection, MemoryTier, ScalarType, ScopeKind};
use std::collections::HashMap;
use std::fmt::{self, Write};

/// Virtual SSA register in Velox IR bound to a static Velox Activation Arena byte offset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirReg {
    pub id: usize,
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: ScalarType,
    pub tier: MemoryTier,
    pub arena_offset: u64,
    pub aligned_bytes: u64,
}

impl fmt::Display for VirReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "%{}({}: {:?} {} @ +0x{:X})",
            self.id, self.name, self.shape, self.dtype, self.arena_offset
        )
    }
}

/// Function parameter descriptor in VIR.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirParam {
    pub name: String,
    pub dtype: ScalarType,
    pub shape: Vec<usize>,
    pub is_scalar: bool,
}

/// SSA instruction set for Velox IR (VIR), supporting both unfused operations and SOTA fused kernels.
#[derive(Debug, Clone, PartialEq)]
pub enum VirInst {
    /// Token embedding lookup (`embed(tokens, w_embed)`).
    Embed {
        dst: VirReg,
        tokens_param: String,
        weight_name: String,
    },
    /// Root Mean Square Layer Normalization (`rmsnorm(x, w, eps)`).
    RmsNorm {
        dst: VirReg,
        input: VirReg,
        weight_name: String,
        layer: Option<usize>,
        eps: f32,
    },
    /// Matrix multiplication (`lhs @ rhs [+ bias]`).
    Matmul {
        dst: VirReg,
        lhs: VirReg,
        weight_name: String,
        bias_name: Option<String>,
        layer: Option<usize>,
        transposed_rhs: bool,
    },
    /// Rotary Position Embedding (`rope(x, step)`).
    RoPE {
        dst: VirReg,
        input: VirReg,
        theta: f32,
    },
    /// Append K, V into KV cache slot (`slot.write(l, k, v)`).
    KVWrite {
        kv_name: String,
        layer: usize,
        k: VirReg,
        v: VirReg,
    },
    /// Append compressed latent KV vector (`kv_latent.append_latent(...)`).
    LatentKVAppend {
        kv_name: String,
        layer: usize,
        c_kv: VirReg,
        k_rope: VirReg,
    },
    /// FlashAttention prefill kernel (`flash_prefill(q, k, v, &mut kv, l)`).
    FlashPrefill {
        dst: VirReg,
        q: VirReg,
        k: VirReg,
        v: VirReg,
        kv_name: String,
        layer: usize,
    },
    /// FlashDecoding single-token attention kernel (`flash_decode(q, &kv, l)`).
    FlashDecode {
        dst: VirReg,
        q: VirReg,
        kv_name: String,
        layer: usize,
    },
    /// DeepSeek MLA absorbed latent decoding attention (`mla_decode_absorbed(q_abs, q_r, &kv_latent, l)`).
    MlaDecodeAbsorbed {
        dst: VirReg,
        q_absorbed: VirReg,
        q_rope: VirReg,
        kv_name: String,
        layer: usize,
    },
    /// Sarathi-Serve piggybacked chunked prefill + decode attention.
    FlashChunkedPiggyback {
        dst_chunk: VirReg,
        dst_decode: VirReg,
        q_chunk: VirReg,
        k_chunk: VirReg,
        v_chunk: VirReg,
        q_decode: VirReg,
        kv_name: String,
        layer: usize,
    },
    /// SpecInfer non-causal tree verification attention.
    FlashTreeAttention {
        dst: VirReg,
        q: VirReg,
        k: VirReg,
        v: VirReg,
        kv_name: String,
        layer: usize,
    },
    /// SiLU activation (`z / (1 + exp(-z))`).
    Silu {
        dst: VirReg,
        input: VirReg,
    },
    /// Element-wise Hadamard product (`lhs * rhs`).
    ElementwiseMul {
        dst: VirReg,
        lhs: VirReg,
        rhs: VirReg,
    },
    /// Residual stream addition (`dst = lhs + rhs`).
    ResidualAdd {
        dst: VirReg,
        lhs: VirReg,
        rhs: VirReg,
    },
    /// Unfused or macro SwiGLU MLP block (`swiglu(x, w_gate, w_up, w_down)`).
    SwiGlu {
        dst: VirReg,
        input: VirReg,
        w_gate: String,
        w_up: String,
        w_down: String,
        layer: usize,
    },
    /// Mixture-of-Experts Top-K router gate (`moe_gate_topk`).
    MoeGateTopK {
        dst_indices: VirReg,
        dst_weights: VirReg,
        input: VirReg,
        w_gate: String,
        layer: usize,
        top_k: usize,
    },
    /// Mixture-of-Experts capacity-bounded token dispatch (`moe_dispatch_topk`).
    MoeDispatchTopK {
        dst: VirReg,
        input: VirReg,
        indices: VirReg,
        weights: VirReg,
        w_experts: String,
        layer: usize,
        capacity_factor: f32,
    },
    /// Slice final sequence position `[:, S - 1, :]`.
    SliceLastToken {
        dst: VirReg,
        input: VirReg,
    },
    /// Greedy argmax sampler (`sample_argmax(logits)`).
    SampleArgmax {
        dst: VirReg,
        logits: VirReg,
    },
    /// Async DMA transfer (`dma_async_h2d` / `dma_async_d2h`).
    AsyncDma {
        direction: DmaDirection,
        src: VirReg,
        dst: VirReg,
    },
    /// Hopper/Blackwell TMA async load.
    TmaLoad {
        src: VirReg,
        dst: VirReg,
        mbarrier: String,
    },
    /// Hopper/Blackwell TMA async store.
    TmaStore {
        src: VirReg,
        dst: VirReg,
    },
    // ========================================================================
    // SOTA Fused & Absorbed Super-Instructions (Emitted by Optimizer Passes)
    // ========================================================================
    /// SOTA Pass 1: Compile-time absorbed MLA projection ($W_{UQ}' = W_{UK}^T W_{UQ}$ and $W_O' = W_O W_{UV}$).
    FusedMlaAbsorbedAttn {
        dst_hidden: VirReg,
        input_hidden: VirReg,
        w_input_norm: String,
        w_dq: String,
        w_absorbed_q: String,
        w_qr: String,
        w_dkv: String,
        w_kr: String,
        w_absorbed_o: String,
        kv_name: String,
        layer: usize,
        eps: f32,
    },
    /// SOTA Pass 2 (Kernel 1 of 2 per layer):
    /// Fused Pre-Attention RMSNorm + Horizontal QKV Projection + RoPE + KV Write + FlashAttention/Decode.
    FusedNormQkvRopeAttn {
        dst_attn: VirReg,
        input_hidden: VirReg,
        w_norm: String,
        w_q: String,
        w_k: String,
        w_v: String,
        b_q: Option<String>,
        b_k: Option<String>,
        b_v: Option<String>,
        kv_name: String,
        layer: usize,
        is_decode: bool,
        eps: f32,
    },
    /// SOTA Pass 2 (Kernel 2 of 2 per layer):
    /// Fused Attention Output Projection + Residual Add + Post-Attn RMSNorm + SwiGLU (Gate/Up/SiLU/Down) + Residual Add.
    FusedOutProjResSwiGluRes {
        dst_hidden: VirReg,
        attn_in: VirReg,
        residual_in: VirReg,
        w_out: String,
        w_post_norm: String,
        w_gate: String,
        w_up: String,
        w_down: String,
        layer: usize,
        eps: f32,
    },
}

/// Basic block control-flow terminator.
#[derive(Debug, Clone, PartialEq)]
pub enum VirTerminator {
    Return(Option<VirReg>),
    Branch { target_block: usize },
}

/// SSA Basic Block inside a VIR function.
#[derive(Debug, Clone, PartialEq)]
pub struct VirBasicBlock {
    pub id: usize,
    pub label: String,
    pub instructions: Vec<VirInst>,
    pub terminator: VirTerminator,
}

/// Monomorphized execution routine in Velox IR (`prefill`, `decode`, etc.).
#[derive(Debug, Clone, PartialEq)]
pub struct VirFunction {
    pub name: String,
    pub scope_kind: ScopeKind,
    pub is_monomorphized: bool,
    pub active_mask_predicated: bool,
    pub max_batch_size: usize,
    pub max_seq_len: usize,
    pub params: Vec<VirParam>,
    pub blocks: Vec<VirBasicBlock>,
    pub unoptimized_op_count: usize,
    pub optimized_launch_count: usize,
}

/// Complete Velox IR (VIR) Module for a compiled model.
#[derive(Debug, Clone, PartialEq)]
pub struct VirModule {
    pub model_name: String,
    pub device_name: String,
    pub compute_capability: (u32, u32),
    pub constants: HashMap<String, usize>,
    pub weights: Vec<PlannedWeightOffset>,
    pub kv_pools: Vec<PlannedKVCachePool>,
    pub functions: Vec<VirFunction>,
    pub memory_plan: MemoryPlan,
    pub mla_absorption_applied: bool,
    pub kernel_fusion_applied: bool,
    pub dag_monomorphized: bool,
}

impl VirModule {
    /// Formats the VIR module into a human-readable SSA dump for inspection and testing.
    #[must_use]
    pub fn format_ir(&self) -> String {
        let mut out = String::with_capacity(4096);
        let _ = writeln!(
            out,
            "; ============================================================================"
        );
        let _ = writeln!(out, "; Velox IR (VIR) Module: @{}", self.model_name);
        let _ = writeln!(
            out,
            "; Target Device: {} (sm_{}{})",
            self.device_name, self.compute_capability.0, self.compute_capability.1
        );
        let _ = writeln!(
            out,
            "; SOTA Passes: mla_absorption={}, kernel_fusion={}, monomorphized={}",
            self.mla_absorption_applied, self.kernel_fusion_applied, self.dag_monomorphized
        );
        let _ = writeln!(
            out,
            "; Velox Activation Arena: {} bytes (256B aligned)",
            self.memory_plan.activation_arena_bytes
        );
        let _ = writeln!(
            out,
            "; ============================================================================"
        );

        for func in &self.functions {
            let _ = writeln!(out);
            let _ = writeln!(
                out,
                "define @{} (scope={}, monomorphized={}, active_mask={}, launches={}) {{",
                func.name,
                func.scope_kind,
                func.is_monomorphized,
                func.active_mask_predicated,
                func.optimized_launch_count
            );
            for bb in &func.blocks {
                let _ = writeln!(out, "  {}:", bb.label);
                for inst in &bb.instructions {
                    let _ = writeln!(out, "    {inst:?}");
                }
                let _ = writeln!(out, "    {:?}", bb.terminator);
            }
            let _ = writeln!(out, "}}");
        }
        out
    }
}

/// Lowers a validated `TypedProgram` and its `MemoryPlan`s into `VirModule`s.
#[must_use]
pub fn lower_program_to_vir(typed: &TypedProgram, plans: &[MemoryPlan]) -> Vec<VirModule> {
    let mut modules = Vec::with_capacity(typed.models.len());

    for (idx, model) in typed.models.iter().enumerate() {
        let plan = plans
            .get(idx)
            .cloned()
            .expect("MemoryPlan must exist for every TypedModel");
        let ast_model = &typed.program.model_declarations[idx];
        let module = lower_model_to_vir(model, ast_model, plan);
        modules.push(module);
    }

    modules
}

fn lower_model_to_vir(
    model: &TypedModel,
    ast_model: &ModelDeclarationNode,
    plan: MemoryPlan,
) -> VirModule {
    let num_layers = model.constants.get("NUM_LAYERS").copied().unwrap_or(1);
    let b_max = model.constants.get("B_MAX").copied().unwrap_or(1);
    let s_max = model.constants.get("S_MAX").copied().unwrap_or(2048);
    let hidden_dim = model.constants.get("HIDDEN_DIM").copied().unwrap_or(576);
    let inter_dim = model
        .constants
        .get("INTERMEDIATE_DIM")
        .or_else(|| model.constants.get("EXPERT_INTERMEDIATE_DIM"))
        .copied()
        .unwrap_or(hidden_dim * 4);
    let vocab = model.constants.get("VOCAB").copied().unwrap_or(49152);

    // Identify canonical weight names from model declarations
    let find_weight = |candidates: &[&str], fallback: &str| -> String {
        for &c in candidates {
            if model.weights.iter().any(|w| w.name == c) {
                return c.to_string();
            }
        }
        fallback.to_string()
    };

    let w_embed = find_weight(&["w_embed"], "w_embed");
    let w_q = find_weight(&["w_q", "w_uq"], "w_q");
    let w_k = find_weight(&["w_k", "w_dkv"], "w_k");
    let w_v = find_weight(&["w_v", "w_uv"], "w_v");
    let w_out = find_weight(&["w_out", "w_o"], "w_out");
    let w_gate = find_weight(&["w_gate", "w_shared_expert"], "w_gate");
    let w_up = find_weight(&["w_up", "w_shared_expert"], "w_up");
    let w_down = find_weight(&["w_down", "w_shared_expert"], "w_down");
    let w_attn_norm = find_weight(&["w_attn_norm", "w_input_norm"], "w_attn_norm");
    let w_mlp_norm = find_weight(&["w_mlp_norm", "w_post_norm"], "w_mlp_norm");
    let w_final_norm = find_weight(&["w_final_norm", "w_norm"], "w_final_norm");
    let w_head = find_weight(&["w_head", "w_lm_head", "w_embed"], "w_head");

    let b_q = model.weights.iter().find(|w| w.name == "b_q").map(|w| w.name.clone());
    let b_k = model.weights.iter().find(|w| w.name == "b_k").map(|w| w.name.clone());
    let b_v = model.weights.iter().find(|w| w.name == "b_v").map(|w| w.name.clone());

    let kv_name = model
        .kv_caches
        .first()
        .map(|kv| kv.name.clone())
        .unwrap_or_else(|| "kv_cache".to_string());
    let is_mla = model.kv_caches.iter().any(|kv| kv.kind == crate::types::KVCacheKind::Latent);
    let has_moe = model.weights.iter().any(|w| w.name == "w_moe_gate");

    let mut functions = Vec::with_capacity(ast_model.scopes.len());

    for (s_idx, ast_scope) in ast_model.scopes.iter().enumerate() {
        let kind = ast_scope.kind();
        let scope_plan = plan.scope_plans.get(s_idx);

        // Build offset lookup table from the StaticMemoryPlanner's ScopeArenaPlan
        let mut offset_map: HashMap<String, u64> = HashMap::new();
        let mut next_fallback_offset = 0u64;
        if let Some(sp) = scope_plan {
            for alloc in &sp.allocations {
                offset_map.insert(alloc.name.clone(), alloc.offset_bytes);
                next_fallback_offset =
                    next_fallback_offset.max(alloc.offset_bytes + alloc.aligned_size_bytes);
            }
        }

        let mut reg_counter = 0usize;
        let mut alloc_reg = |name: &str, shape: Vec<usize>, dtype: ScalarType, tier: MemoryTier| -> VirReg {
            let id = reg_counter;
            reg_counter += 1;
            let raw_bytes =
                (shape.iter().product::<usize>() as f64 * dtype.byte_size()).ceil() as u64;
            let aligned_bytes = align_up_256(raw_bytes.max(256));
            let arena_offset = if let Some(&off) = offset_map.get(name) {
                off
            } else {
                let off = next_fallback_offset;
                next_fallback_offset += aligned_bytes;
                off
            };
            VirReg {
                id,
                name: name.to_string(),
                shape,
                dtype,
                tier,
                arena_offset,
                aligned_bytes,
            }
        };

        let seq_extent = if kind.is_prefill_phase() {
            s_max.min(2048)
        } else {
            1
        };
        let batch_extent = b_max;

        let mut insts = Vec::new();
        let hidden_reg = alloc_reg(
            "hidden",
            vec![batch_extent, seq_extent, hidden_dim],
            ScalarType::F16,
            MemoryTier::BumpSram,
        );
        insts.push(VirInst::Embed {
            dst: hidden_reg.clone(),
            tokens_param: "tokens".to_string(),
            weight_name: w_embed.clone(),
        });

        let eps = if model.name.contains("Qwen") {
            1e-6
        } else {
            1e-5
        };

        // Lower each layer into canonical VIR instructions (14 ops/layer before optimization)
        for layer in 0..num_layers {
            let norm_x = alloc_reg(
                "norm_x",
                vec![batch_extent, seq_extent, hidden_dim],
                ScalarType::F16,
                MemoryTier::BumpSram,
            );
            insts.push(VirInst::RmsNorm {
                dst: norm_x.clone(),
                input: hidden_reg.clone(),
                weight_name: w_attn_norm.clone(),
                layer: Some(layer),
                eps,
            });

            if is_mla && kind == ScopeKind::Decode {
                // Unfused MLA decode operations (to be rewritten by MlaAbsorptionPass)
                let c_q = alloc_reg(
                    "c_q",
                    vec![batch_extent, 1, 1536],
                    ScalarType::Fp8E4m3,
                    MemoryTier::BumpSram,
                );
                insts.push(VirInst::Matmul {
                    dst: c_q.clone(),
                    lhs: norm_x.clone(),
                    weight_name: "w_dq".to_string(),
                    bias_name: None,
                    layer: Some(layer),
                    transposed_rhs: false,
                });

                let q_absorbed = alloc_reg(
                    "q_absorbed",
                    vec![batch_extent, 1, 128 * 512],
                    ScalarType::Fp8E4m3,
                    MemoryTier::BumpSram,
                );
                insts.push(VirInst::Matmul {
                    dst: q_absorbed.clone(),
                    lhs: c_q.clone(),
                    weight_name: "w_absorbed_q".to_string(),
                    bias_name: None,
                    layer: Some(layer),
                    transposed_rhs: false,
                });

                let q_r = alloc_reg(
                    "q_r",
                    vec![batch_extent, 1, 128 * 64],
                    ScalarType::Fp8E4m3,
                    MemoryTier::BumpSram,
                );
                insts.push(VirInst::Matmul {
                    dst: q_r.clone(),
                    lhs: c_q,
                    weight_name: "w_qr".to_string(),
                    bias_name: None,
                    layer: Some(layer),
                    transposed_rhs: false,
                });

                let c_kv = alloc_reg(
                    "c_kv",
                    vec![batch_extent, 1, 512],
                    ScalarType::Fp8E4m3,
                    MemoryTier::BumpSram,
                );
                insts.push(VirInst::Matmul {
                    dst: c_kv.clone(),
                    lhs: norm_x.clone(),
                    weight_name: "w_dkv".to_string(),
                    bias_name: None,
                    layer: Some(layer),
                    transposed_rhs: false,
                });

                let k_r = alloc_reg(
                    "k_r",
                    vec![batch_extent, 1, 64],
                    ScalarType::Fp8E4m3,
                    MemoryTier::BumpSram,
                );
                insts.push(VirInst::Matmul {
                    dst: k_r.clone(),
                    lhs: norm_x,
                    weight_name: "w_kr".to_string(),
                    bias_name: None,
                    layer: Some(layer),
                    transposed_rhs: false,
                });

                insts.push(VirInst::LatentKVAppend {
                    kv_name: kv_name.clone(),
                    layer,
                    c_kv,
                    k_rope: k_r,
                });

                let attn_out = alloc_reg(
                    "attn_out",
                    vec![batch_extent, 1, hidden_dim],
                    ScalarType::F16,
                    MemoryTier::BumpSram,
                );
                insts.push(VirInst::MlaDecodeAbsorbed {
                    dst: attn_out.clone(),
                    q_absorbed,
                    q_rope: q_r,
                    kv_name: kv_name.clone(),
                    layer,
                });

                let proj_out = alloc_reg(
                    "proj_out",
                    vec![batch_extent, 1, hidden_dim],
                    ScalarType::F16,
                    MemoryTier::BumpSram,
                );
                insts.push(VirInst::Matmul {
                    dst: proj_out.clone(),
                    lhs: attn_out,
                    weight_name: "w_absorbed_o".to_string(),
                    bias_name: None,
                    layer: Some(layer),
                    transposed_rhs: false,
                });

                insts.push(VirInst::ResidualAdd {
                    dst: hidden_reg.clone(),
                    lhs: hidden_reg.clone(),
                    rhs: proj_out,
                });
            } else {
                // Standard GQA / MHA QKV projections + RoPE + FlashAttention
                let q_reg = alloc_reg(
                    "q",
                    vec![batch_extent, seq_extent, hidden_dim],
                    ScalarType::F16,
                    MemoryTier::BumpSram,
                );
                let k_reg = alloc_reg(
                    "k",
                    vec![batch_extent, seq_extent, hidden_dim],
                    ScalarType::F16,
                    MemoryTier::BumpSram,
                );
                let v_reg = alloc_reg(
                    "v",
                    vec![batch_extent, seq_extent, hidden_dim],
                    ScalarType::F16,
                    MemoryTier::BumpSram,
                );

                insts.push(VirInst::Matmul {
                    dst: q_reg.clone(),
                    lhs: norm_x.clone(),
                    weight_name: w_q.clone(),
                    bias_name: b_q.clone(),
                    layer: Some(layer),
                    transposed_rhs: false,
                });
                insts.push(VirInst::Matmul {
                    dst: k_reg.clone(),
                    lhs: norm_x.clone(),
                    weight_name: w_k.clone(),
                    bias_name: b_k.clone(),
                    layer: Some(layer),
                    transposed_rhs: false,
                });
                insts.push(VirInst::Matmul {
                    dst: v_reg.clone(),
                    lhs: norm_x,
                    weight_name: w_v.clone(),
                    bias_name: b_v.clone(),
                    layer: Some(layer),
                    transposed_rhs: false,
                });

                insts.push(VirInst::RoPE {
                    dst: q_reg.clone(),
                    input: q_reg.clone(),
                    theta: 10000.0,
                });
                insts.push(VirInst::RoPE {
                    dst: k_reg.clone(),
                    input: k_reg.clone(),
                    theta: 10000.0,
                });

                let attn_out = alloc_reg(
                    "attn_out",
                    vec![batch_extent, seq_extent, hidden_dim],
                    ScalarType::F16,
                    MemoryTier::BumpSram,
                );

                if kind.is_decode_phase() {
                    insts.push(VirInst::KVWrite {
                        kv_name: kv_name.clone(),
                        layer,
                        k: k_reg,
                        v: v_reg,
                    });
                    insts.push(VirInst::FlashDecode {
                        dst: attn_out.clone(),
                        q: q_reg,
                        kv_name: kv_name.clone(),
                        layer,
                    });
                } else {
                    insts.push(VirInst::FlashPrefill {
                        dst: attn_out.clone(),
                        q: q_reg,
                        k: k_reg,
                        v: v_reg,
                        kv_name: kv_name.clone(),
                        layer,
                    });
                }

                let o_proj = alloc_reg(
                    "o_proj",
                    vec![batch_extent, seq_extent, hidden_dim],
                    ScalarType::F16,
                    MemoryTier::BumpSram,
                );
                insts.push(VirInst::Matmul {
                    dst: o_proj.clone(),
                    lhs: attn_out,
                    weight_name: w_out.clone(),
                    bias_name: None,
                    layer: Some(layer),
                    transposed_rhs: false,
                });

                insts.push(VirInst::ResidualAdd {
                    dst: hidden_reg.clone(),
                    lhs: hidden_reg.clone(),
                    rhs: o_proj,
                });
            }

            // Post-Attention MLP / MoE Block
            let mlp_norm_x = alloc_reg(
                "mlp_norm_x",
                vec![batch_extent, seq_extent, hidden_dim],
                ScalarType::F16,
                MemoryTier::BumpSram,
            );
            insts.push(VirInst::RmsNorm {
                dst: mlp_norm_x.clone(),
                input: hidden_reg.clone(),
                weight_name: w_mlp_norm.clone(),
                layer: Some(layer),
                eps,
            });

            if has_moe {
                let top_idx = alloc_reg(
                    "top_indices",
                    vec![batch_extent, seq_extent, 8],
                    ScalarType::U32,
                    MemoryTier::BumpSram,
                );
                let top_w = alloc_reg(
                    "top_weights",
                    vec![batch_extent, seq_extent, 8],
                    ScalarType::F32,
                    MemoryTier::BumpSram,
                );
                insts.push(VirInst::MoeGateTopK {
                    dst_indices: top_idx.clone(),
                    dst_weights: top_w.clone(),
                    input: mlp_norm_x.clone(),
                    w_gate: "w_moe_gate".to_string(),
                    layer,
                    top_k: 8,
                });
                let routed_out = alloc_reg(
                    "routed_out",
                    vec![batch_extent, seq_extent, hidden_dim],
                    ScalarType::F16,
                    MemoryTier::BumpSram,
                );
                insts.push(VirInst::MoeDispatchTopK {
                    dst: routed_out.clone(),
                    input: mlp_norm_x,
                    indices: top_idx,
                    weights: top_w,
                    w_experts: "w_routed_experts".to_string(),
                    layer,
                    capacity_factor: 2.0,
                });
                insts.push(VirInst::ResidualAdd {
                    dst: hidden_reg.clone(),
                    lhs: hidden_reg.clone(),
                    rhs: routed_out,
                });
            } else {
                let gate_reg = alloc_reg(
                    "gate",
                    vec![batch_extent, seq_extent, inter_dim],
                    ScalarType::F16,
                    MemoryTier::BumpSram,
                );
                let up_reg = alloc_reg(
                    "up",
                    vec![batch_extent, seq_extent, inter_dim],
                    ScalarType::F16,
                    MemoryTier::BumpSram,
                );
                let act_reg = alloc_reg(
                    "act",
                    vec![batch_extent, seq_extent, inter_dim],
                    ScalarType::F16,
                    MemoryTier::BumpSram,
                );
                let down_reg = alloc_reg(
                    "down",
                    vec![batch_extent, seq_extent, hidden_dim],
                    ScalarType::F16,
                    MemoryTier::BumpSram,
                );

                insts.push(VirInst::Matmul {
                    dst: gate_reg.clone(),
                    lhs: mlp_norm_x.clone(),
                    weight_name: w_gate.clone(),
                    bias_name: None,
                    layer: Some(layer),
                    transposed_rhs: false,
                });
                insts.push(VirInst::Matmul {
                    dst: up_reg.clone(),
                    lhs: mlp_norm_x,
                    weight_name: w_up.clone(),
                    bias_name: None,
                    layer: Some(layer),
                    transposed_rhs: false,
                });
                insts.push(VirInst::Silu {
                    dst: gate_reg.clone(),
                    input: gate_reg.clone(),
                });
                insts.push(VirInst::ElementwiseMul {
                    dst: act_reg.clone(),
                    lhs: gate_reg,
                    rhs: up_reg,
                });
                insts.push(VirInst::Matmul {
                    dst: down_reg.clone(),
                    lhs: act_reg,
                    weight_name: w_down.clone(),
                    bias_name: None,
                    layer: Some(layer),
                    transposed_rhs: false,
                });
                insts.push(VirInst::ResidualAdd {
                    dst: hidden_reg.clone(),
                    lhs: hidden_reg.clone(),
                    rhs: down_reg,
                });
            }
        }

        // Final RMSNorm + LM Head Logits + Argmax Sampling
        let final_norm = alloc_reg(
            "final_norm",
            vec![batch_extent, 1, hidden_dim],
            ScalarType::F16,
            MemoryTier::BumpSram,
        );
        insts.push(VirInst::RmsNorm {
            dst: final_norm.clone(),
            input: hidden_reg,
            weight_name: w_final_norm.clone(),
            layer: None,
            eps,
        });

        let logits = alloc_reg(
            "logits",
            vec![batch_extent, 1, vocab],
            ScalarType::F32,
            MemoryTier::BumpSram,
        );
        insts.push(VirInst::Matmul {
            dst: logits.clone(),
            lhs: final_norm,
            weight_name: w_head.clone(),
            bias_name: None,
            layer: None,
            transposed_rhs: w_head == "w_embed",
        });

        let next_token = alloc_reg(
            "next_token",
            vec![batch_extent, 1],
            ScalarType::U32,
            MemoryTier::HostPinned,
        );
        insts.push(VirInst::SampleArgmax {
            dst: next_token.clone(),
            logits,
        });

        let unoptimized_op_count = insts.len();
        let entry_bb = VirBasicBlock {
            id: 0,
            label: "entry".to_string(),
            instructions: insts,
            terminator: VirTerminator::Return(Some(next_token)),
        };

        let func_name = format!("velox_{}_{}", model.name, kind.as_str());
        let params = if kind.is_prefill_phase() {
            vec![
                VirParam {
                    name: "tokens".to_string(),
                    dtype: ScalarType::U32,
                    shape: vec![batch_extent, seq_extent],
                    is_scalar: false,
                },
                VirParam {
                    name: "seq_len".to_string(),
                    dtype: ScalarType::Usize,
                    shape: vec![1],
                    is_scalar: true,
                },
            ]
        } else {
            vec![
                VirParam {
                    name: "token".to_string(),
                    dtype: ScalarType::U32,
                    shape: vec![batch_extent, 1],
                    is_scalar: false,
                },
                VirParam {
                    name: "step".to_string(),
                    dtype: ScalarType::Usize,
                    shape: vec![1],
                    is_scalar: true,
                },
                VirParam {
                    name: "active_mask".to_string(),
                    dtype: ScalarType::U64,
                    shape: vec![1],
                    is_scalar: true,
                },
            ]
        };

        functions.push(VirFunction {
            name: func_name,
            scope_kind: kind,
            is_monomorphized: false,
            active_mask_predicated: false,
            max_batch_size: batch_extent,
            max_seq_len: s_max,
            params,
            blocks: vec![entry_bb],
            unoptimized_op_count,
            optimized_launch_count: unoptimized_op_count,
        });
    }

    VirModule {
        model_name: model.name.clone(),
        device_name: plan.device_name.clone(),
        compute_capability: plan.compute_capability,
        constants: model.constants.clone(),
        weights: plan.weight_offsets.clone(),
        kv_pools: plan.kv_pools.clone(),
        functions,
        memory_plan: plan,
        mla_absorption_applied: false,
        kernel_fusion_applied: false,
        dag_monomorphized: false,
    }
}
