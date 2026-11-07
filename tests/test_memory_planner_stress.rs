//! Adversarial Stress-Test Suite for Step 6: Velox Static Memory Planner & VRAM Feasibility Prover
//!
//! Covers:
//! 1. 256-byte static byte offset alignment invariant across all allocations.
//! 2. Lifetime interval non-overlap invariant (disjoint memory for overlapping lifetimes).
//! 3. VRAM budget threshold enforcement (exact boundary at alpha = 0.95, E002 emission).
//! 4. Extreme boundary models (1,000 overlapping activations, 1,000 sequential activations, fuzzing).
//! 5. 5 native KV-cache pool types (Paged, VirtualContiguous, Latent, StreamingSink, SpeculativeTree).
//! 6. Compile-Time VRAM Feasibility Proof Certificate generation and C99 static assert validity.
//! 7. End-to-end CLI validation with veloxc.

use std::collections::HashMap;
use veloxc::ast::TargetHardwareNode;
use veloxc::cli::{run_cli_with_io, EXIT_SUCCESS};
use veloxc::diagnostics::DiagnosticCode;
use veloxc::memory_planner::{align_up_256, StaticMemoryPlanner, ARENA_ALIGNMENT_BYTES};
use veloxc::sema::{TypedActivation, TypedKVCache, TypedModel, TypedScope, TypedWeight};
use veloxc::types::{
    KVCacheKind, MemoryTier, ScalarType, ScopeKind, SourceLocation, SourceSpan,
};

fn dummy_span() -> SourceSpan {
    SourceSpan {
        file: "stress_test.vlx".to_string(),
        start: SourceLocation {
            line: 1,
            column: 1,
            offset: 0,
        },
        end: SourceLocation {
            line: 1,
            column: 10,
            offset: 9,
        },
    }
}

// ============================================================================
// Battery 1: 256-Byte Alignment Invariant Across Odd and Unaligned Raw Sizes
// ============================================================================

#[test]
fn test_stress_256_byte_alignment_unaligned_raw_sizes() {
    let unaligned_sizes = [
        1u64, 3, 7, 15, 31, 63, 127, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 4095, 4096,
        4097, 65535, 65536, 65537,
    ];

    let mut activations = Vec::new();
    for (i, &sz) in unaligned_sizes.iter().enumerate() {
        activations.push(TypedActivation {
            id: i,
            name: format!("act_unaligned_{i}"),
            scope: ScopeKind::Prefill,
            shape: vec![sz as usize],
            dtype: ScalarType::U8,
            tier: MemoryTier::BumpSram,
            size_bytes: sz,
            birth_step: 1,
            death_step: 10, // All live concurrently
            span: dummy_span(),
        });
    }

    let scope = TypedScope {
        name: "scope_unaligned_stress".to_string(),
        kind: ScopeKind::Prefill,
        activations,
        total_steps: 10,
        span: dummy_span(),
    };

    let plan = StaticMemoryPlanner::color_scope_activations(&scope);

    // Verify invariant 1: Every assigned offset must be strictly 0 mod 256
    for alloc in &plan.allocations {
        assert_eq!(
            alloc.offset_bytes % ARENA_ALIGNMENT_BYTES,
            0,
            "Activation {} offset {} is not 256-byte aligned!",
            alloc.name,
            alloc.offset_bytes
        );
        assert_eq!(
            alloc.aligned_size_bytes % ARENA_ALIGNMENT_BYTES,
            0,
            "Activation {} aligned size {} is not a multiple of 256!",
            alloc.name,
            alloc.aligned_size_bytes
        );
        assert!(
            alloc.aligned_size_bytes >= alloc.raw_size_bytes,
            "Aligned size must cover raw size"
        );
    }

    assert_eq!(
        plan.actual_colored_peak_bytes % ARENA_ALIGNMENT_BYTES,
        0,
        "Peak activation bytes must be 256-byte aligned"
    );
}

// ============================================================================
// Battery 2: Extreme Boundary Model - 1,000 Overlapping Activations
// ============================================================================

#[test]
fn test_stress_1000_overlapping_activations_pairwise_disjoint() {
    const N: usize = 1000;
    let mut activations = Vec::with_capacity(N);

    // Generate 1,000 concurrent activations with variable pseudo-random sizes
    for i in 0..N {
        // Diverse size distribution between 256 and 65,536 bytes
        let raw_sz = (256 + ((i * 7919) % 65280)) as u64;
        activations.push(TypedActivation {
            id: i,
            name: format!("act_1k_{i}"),
            scope: ScopeKind::Prefill,
            shape: vec![raw_sz as usize],
            dtype: ScalarType::U8,
            tier: MemoryTier::BumpSram,
            size_bytes: raw_sz,
            birth_step: 1,  // All born at step 1
            death_step: 50, // All live until step 50
            span: dummy_span(),
        });
    }

    let scope = TypedScope {
        name: "scope_1000_overlap".to_string(),
        kind: ScopeKind::Prefill,
        activations,
        total_steps: 50,
        span: dummy_span(),
    };

    let plan = StaticMemoryPlanner::color_scope_activations(&scope);
    assert_eq!(plan.allocations.len(), N);

    // Invariant 1: 256-byte alignment
    for alloc in &plan.allocations {
        assert_eq!(
            alloc.offset_bytes % 256,
            0,
            "Offset for {} is not 256B aligned",
            alloc.name
        );
    }

    // Invariant 2: Pairwise non-overlap invariant
    // Across all N * (N - 1) / 2 = 499,500 pairs, check memory intervals are disjoint
    for i in 0..N {
        let a = &plan.allocations[i];
        let a_start = a.offset_bytes;
        let a_end = a_start + a.aligned_size_bytes;

        for j in (i + 1)..N {
            let b = &plan.allocations[j];
            let b_start = b.offset_bytes;
            let b_end = b_start + b.aligned_size_bytes;

            let disjoint = a_end <= b_start || b_end <= a_start;
            assert!(
                disjoint,
                "COLLISION DETECTED between act {} [{}, {}) and act {} [{}, {})!",
                a.name, a_start, a_end, b.name, b_start, b_end
            );
        }
    }

    // Since all 1,000 are concurrently live, actual colored peak must equal sum of all aligned sizes
    let total_expected_bytes: u64 = plan
        .allocations
        .iter()
        .map(|a| a.aligned_size_bytes)
        .sum();
    assert_eq!(plan.actual_colored_peak_bytes, total_expected_bytes);
    assert!(
        (plan.competitive_ratio_rho - 1.0).abs() < 1e-6,
        "Competitive ratio should be 1.0 for completely overlapping set, got {}",
        plan.competitive_ratio_rho
    );
}

// ============================================================================
// Battery 3: Extreme Boundary Model - 1,000 Sequential Activations (Memory Reuse)
// ============================================================================

#[test]
fn test_stress_1000_sequential_activations_memory_reuse() {
    const N: usize = 1000;
    let mut activations = Vec::with_capacity(N);
    let size_per_tensor = 1024 * 1024u64; // 1 MiB each

    // Each activation lives for 1 step: born at 2*i, dies at 2*i + 1
    // The next activation is born at 2*i + 2, dies at 2*i + 3
    // Therefore, NO two activations overlap in time!
    for i in 0..N {
        activations.push(TypedActivation {
            id: i,
            name: format!("seq_act_{i}"),
            scope: ScopeKind::Prefill,
            shape: vec![size_per_tensor as usize],
            dtype: ScalarType::U8,
            tier: MemoryTier::BumpSram,
            size_bytes: size_per_tensor,
            birth_step: 2 * i,
            death_step: 2 * i + 1,
            span: dummy_span(),
        });
    }

    let scope = TypedScope {
        name: "scope_1000_sequential".to_string(),
        kind: ScopeKind::Prefill,
        activations,
        total_steps: 2 * N + 1,
        span: dummy_span(),
    };

    let plan = StaticMemoryPlanner::color_scope_activations(&scope);
    assert_eq!(plan.allocations.len(), N);

    // Invariant: Peak memory must be EXACTLY 1 MiB (reused across all 1,000 activations)
    // rather than 1,000 MiB!
    assert_eq!(
        plan.actual_colored_peak_bytes, size_per_tensor,
        "Sequential activations should achieve 100% memory reuse!"
    );
    assert_eq!(
        plan.theoretical_lower_bound_bytes, size_per_tensor,
        "Theoretical lower bound must equal 1 MiB"
    );
    assert!(
        (plan.competitive_ratio_rho - 1.0).abs() < 1e-6,
        "Competitive ratio should be 1.0"
    );

    // Invariant: All allocations must reuse offset 0
    for alloc in &plan.allocations {
        assert_eq!(
            alloc.offset_bytes, 0,
            "Sequential activation {} must be assigned offset 0",
            alloc.name
        );
    }
}

// ============================================================================
// Battery 4: Deterministic Random Interval Graph Fuzzing (50 Random Scenarios)
// ============================================================================

#[test]
fn test_stress_randomized_interval_graphs_fuzzing() {
    // Pure LCG pseudo-random generator for 100% deterministic reproducibility
    struct Lcg(u64);
    impl Lcg {
        fn next_u32(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            (self.0 >> 32) as u32
        }
        fn range(&mut self, low: usize, high: usize) -> usize {
            low + (self.next_u32() as usize % (high - low + 1))
        }
    }

    let mut rng = Lcg(0xDEADBEEF_CAFE1234);

    for scenario in 0..50 {
        let num_acts = rng.range(20, 100);
        let mut activations = Vec::with_capacity(num_acts);

        for i in 0..num_acts {
            let birth = rng.range(0, 50);
            let death = rng.range(birth, birth + 40);
            let sz = rng.range(1, 100_000) as u64;

            activations.push(TypedActivation {
                id: i,
                name: format!("fuzz_act_{scenario}_{i}"),
                scope: ScopeKind::Decode,
                shape: vec![sz as usize],
                dtype: ScalarType::U8,
                tier: MemoryTier::BumpSram,
                size_bytes: sz,
                birth_step: birth,
                death_step: death,
                span: dummy_span(),
            });
        }

        let scope = TypedScope {
            name: format!("scope_fuzz_{scenario}"),
            kind: ScopeKind::Decode,
            activations,
            total_steps: 100,
            span: dummy_span(),
        };

        let plan = StaticMemoryPlanner::color_scope_activations(&scope);

        // Check 1: 256-byte alignment
        for alloc in &plan.allocations {
            assert_eq!(
                alloc.offset_bytes % 256,
                0,
                "Offset not 256B aligned in scenario {scenario}"
            );
            assert_eq!(
                alloc.aligned_size_bytes % 256,
                0,
                "Size not 256B aligned in scenario {scenario}"
            );
        }

        // Check 2: Pairwise non-overlap invariant for all interfering pairs
        for i in 0..num_acts {
            let a = &plan.allocations[i];
            for j in (i + 1)..num_acts {
                let b = &plan.allocations[j];

                let lifetimes_overlap =
                    a.birth_step <= b.death_step && b.birth_step <= a.death_step;
                if lifetimes_overlap {
                    let a_end = a.offset_bytes + a.aligned_size_bytes;
                    let b_end = b.offset_bytes + b.aligned_size_bytes;
                    let disjoint = a_end <= b.offset_bytes || b_end <= a.offset_bytes;
                    assert!(
                        disjoint,
                        "Scenario {scenario}: Overlapping live activations '{}' [{}, {}] and '{}' [{}, {}] collision: [{}, {}) vs [{}, {})",
                        a.name, a.birth_step, a.death_step,
                        b.name, b.birth_step, b.death_step,
                        a.offset_bytes, a_end,
                        b.offset_bytes, b_end
                    );
                }
            }
        }

        // Check 3: Peak memory >= theoretical lower bound
        assert!(
            plan.actual_colored_peak_bytes >= plan.theoretical_lower_bound_bytes,
            "Actual peak {} cannot be less than theoretical lower bound {}",
            plan.actual_colored_peak_bytes,
            plan.theoretical_lower_bound_bytes
        );
    }
}

// ============================================================================
// Battery 5: VRAM Budget Threshold Exact Boundary Testing (E002)
// ============================================================================

#[test]
fn test_stress_vram_budget_threshold_exact_boundary() {
    let dummy_weight = TypedWeight {
        name: "w_test".to_string(),
        shape: vec![1024, 1024],
        dtype: ScalarType::F16,
        tier: MemoryTier::VramImmutable,
        size_bytes: 3 * 1024 * 1024 * 1024, // 3 GiB
        scale_bytes: 0,
        is_absorbed: false,
        absorb_kind: None,
        file_path: None,
        tensor_key: None,
        span: dummy_span(),
    };

    let dummy_scope = TypedScope {
        name: "prefill".to_string(),
        kind: ScopeKind::Prefill,
        activations: vec![TypedActivation {
            id: 0,
            name: "act_test".to_string(),
            scope: ScopeKind::Prefill,
            shape: vec![1024, 1024],
            dtype: ScalarType::F16,
            tier: MemoryTier::BumpSram,
            size_bytes: 1024 * 1024 * 2, // 2 MiB
            birth_step: 1,
            death_step: 2,
            span: dummy_span(),
        }],
        total_steps: 2,
        span: dummy_span(),
    };

    let model = TypedModel {
        name: "TestModel".to_string(),
        constants: HashMap::new(),
        weights: vec![dummy_weight],
        kv_caches: Vec::new(),
        scopes: vec![dummy_scope],
        span: dummy_span(),
    };

    // Calculate exact required VRAM for this model on a 4 GB GPU (in the 2..24 GiB reserve band)
    let mut planner = StaticMemoryPlanner::new();
    let hw_probe = TargetHardwareNode::new(
        "probe_gpu",
        4096 * 1024 * 1024,
        (8, 6),
        0.95,
        dummy_span(),
    );
    let probe_plan = planner.plan_model(&model, &hw_probe);
    let s_total = probe_plan.total_required_vram_bytes;

    // Case A: Exact threshold match (total <= safe_budget) -> MUST SUCCEED
    let exact_capacity = ((s_total as f64) / 0.95).ceil() as u64;
    let hw_exact = TargetHardwareNode::new("exact_gpu", exact_capacity, (8, 6), 0.95, dummy_span());
    let mut planner_exact = StaticMemoryPlanner::new();
    let plan_exact = planner_exact.plan_model(&model, &hw_exact);
    assert!(
        plan_exact.is_feasible,
        "Model must be feasible when total ({}) <= safe budget ({})",
        plan_exact.total_required_vram_bytes, plan_exact.safe_budget_bytes
    );
    assert!(planner_exact.diagnostics.is_empty());

    // Case B: Budget exceeded by reducing capacity such that safe_budget < s_total -> MUST FAIL WITH E002
    // Since exact_capacity gives safe_budget >= s_total, reducing capacity by a small delta makes safe_budget < s_total
    let delta = ((exact_capacity as f64 * 0.95) - (s_total as f64)).floor() as u64 + 100;
    let capacity_too_small = exact_capacity - (delta / 2).max(10);
    let hw_fail = TargetHardwareNode::new(
        "tight_gpu",
        capacity_too_small,
        (8, 6),
        0.95,
        dummy_span(),
    );
    let mut planner_fail = StaticMemoryPlanner::new();
    let plan_fail = planner_fail.plan_model(&model, &hw_fail);
    assert!(
        !plan_fail.is_feasible,
        "Model must be marked infeasible when total ({}) > safe budget ({})",
        plan_fail.total_required_vram_bytes, plan_fail.safe_budget_bytes
    );
    assert!(
        planner_fail.diagnostics.iter().any(|d| d.code == DiagnosticCode::E002),
        "Planner must emit E002 when safe budget is exceeded"
    );

    // Case C: Safety factor variations
    for &safety in &[0.50, 0.80, 0.90, 0.95, 0.99] {
        let cap = ((s_total as f64) / safety).ceil() as u64;
        let hw = TargetHardwareNode::new("sf_gpu", cap, (8, 6), safety, dummy_span());
        let mut p = StaticMemoryPlanner::new();
        let plan = p.plan_model(&model, &hw);
        assert!(plan.is_feasible);
        assert!(p.diagnostics.is_empty());
    }

    // Case D: Zero VRAM capacity -> MUST FAIL WITH E002
    let hw_zero = TargetHardwareNode::new("zero_gpu", 0, (8, 6), 0.95, dummy_span());
    let mut p_zero = StaticMemoryPlanner::new();
    let plan_zero = p_zero.plan_model(&model, &hw_zero);
    assert!(!plan_zero.is_feasible);
    assert!(p_zero.diagnostics.iter().any(|d| d.code == DiagnosticCode::E002));
}

// ============================================================================
// Battery 6: 5 Native KV Cache Pool Types Footprint Verification
// ============================================================================

#[test]
fn test_stress_5_kv_cache_types_memory_footprint_and_alignment() {
    let b_max = 4u64;
    let s_max = 2048u64;

    // 1. PagedKVCache
    let paged_kv = TypedKVCache {
        name: "paged_kv".to_string(),
        kind: KVCacheKind::Paged,
        layers: 16,
        heads: 4,
        head_dim: 64,
        block_size: 16,
        max_seq_len: 2048,
        latent_dim: 0,
        rope_dim: 0,
        sinks: 0,
        window: 0,
        max_nodes: 0,
        max_depth: 0,
        dtype: ScalarType::F16,
        tier: MemoryTier::PagedVram,
        span: dummy_span(),
    };
    let (data_paged, meta_paged) = StaticMemoryPlanner::compute_kv_cache_bytes(&paged_kv, b_max, s_max);
    assert!(data_paged > 0);
    assert!(meta_paged > 0);
    assert_eq!(align_up_256(data_paged + meta_paged) % 256, 0);

    // 2. VirtualContiguousKV
    let virt_kv = TypedKVCache {
        name: "virt_kv".to_string(),
        kind: KVCacheKind::VirtualContiguous,
        layers: 16,
        heads: 4,
        head_dim: 64,
        block_size: 16,
        max_seq_len: 2048,
        latent_dim: 0,
        rope_dim: 0,
        sinks: 0,
        window: 0,
        max_nodes: 0,
        max_depth: 0,
        dtype: ScalarType::F16,
        tier: MemoryTier::PagedVram,
        span: dummy_span(),
    };
    let (data_virt, meta_virt) = StaticMemoryPlanner::compute_kv_cache_bytes(&virt_kv, b_max, s_max);
    assert!(data_virt > 0);
    assert_eq!(meta_virt, 0);
    // VirtualContiguous must be 2 MiB page-aligned
    assert_eq!(data_virt % (2 * 1024 * 1024), 0);

    // 3. LatentKVCache (DeepSeek MLA)
    let latent_kv = TypedKVCache {
        name: "latent_kv".to_string(),
        kind: KVCacheKind::Latent,
        layers: 16,
        heads: 4,
        head_dim: 64,
        block_size: 16,
        max_seq_len: 2048,
        latent_dim: 512,
        rope_dim: 64,
        sinks: 0,
        window: 0,
        max_nodes: 0,
        max_depth: 0,
        dtype: ScalarType::F16,
        tier: MemoryTier::PagedVram,
        span: dummy_span(),
    };
    let (data_latent, meta_latent) = StaticMemoryPlanner::compute_kv_cache_bytes(&latent_kv, b_max, s_max);
    assert!(data_latent > 0);
    assert!(meta_latent > 0);
    assert_eq!(align_up_256(data_latent + meta_latent) % 256, 0);

    // 4. StreamingSinkKVCache
    let stream_kv = TypedKVCache {
        name: "stream_kv".to_string(),
        kind: KVCacheKind::StreamingSink,
        layers: 16,
        heads: 4,
        head_dim: 64,
        block_size: 16,
        max_seq_len: 2048,
        latent_dim: 0,
        rope_dim: 0,
        sinks: 4,
        window: 512,
        max_nodes: 0,
        max_depth: 0,
        dtype: ScalarType::F16,
        tier: MemoryTier::PagedVram,
        span: dummy_span(),
    };
    let (data_stream, meta_stream) = StaticMemoryPlanner::compute_kv_cache_bytes(&stream_kv, b_max, s_max);
    assert!(data_stream > 0);
    assert_eq!(meta_stream, 0);
    assert_eq!(align_up_256(data_stream) % 256, 0);

    // 5. SpeculativeTreeBlock
    let tree_kv = TypedKVCache {
        name: "tree_kv".to_string(),
        kind: KVCacheKind::SpeculativeTree,
        layers: 16,
        heads: 4,
        head_dim: 64,
        block_size: 16,
        max_seq_len: 2048,
        latent_dim: 0,
        rope_dim: 0,
        sinks: 0,
        window: 0,
        max_nodes: 64,
        max_depth: 8,
        dtype: ScalarType::F16,
        tier: MemoryTier::PagedVram,
        span: dummy_span(),
    };
    let (data_tree, meta_tree) = StaticMemoryPlanner::compute_kv_cache_bytes(&tree_kv, b_max, s_max);
    assert!(data_tree > 0);
    assert_eq!(meta_tree, b_max * 64 * 8); // 64-bit ancestry bitmask per candidate node
    assert_eq!(align_up_256(data_tree + meta_tree) % 256, 0);
}

// ============================================================================
// Battery 7: Proof Certificate Header Formatting & C99 Static Assert
// ============================================================================

#[test]
fn test_stress_proof_certificate_formatting() {
    let dummy_weight = TypedWeight {
        name: "w_q".to_string(),
        shape: vec![512, 512],
        dtype: ScalarType::F16,
        tier: MemoryTier::VramImmutable,
        size_bytes: 512 * 512 * 2,
        scale_bytes: 0,
        is_absorbed: false,
        absorb_kind: None,
        file_path: None,
        tensor_key: None,
        span: dummy_span(),
    };

    let model = TypedModel {
        name: "CertModel".to_string(),
        constants: HashMap::new(),
        weights: vec![dummy_weight],
        kv_caches: Vec::new(),
        scopes: Vec::new(),
        span: dummy_span(),
    };

    let hw = TargetHardwareNode::new("test_gpu", 8192 * 1024 * 1024, (8, 6), 0.95, dummy_span());
    let mut planner = StaticMemoryPlanner::new();
    let plan = planner.plan_model(&model, &hw);
    let cert = plan.format_proof_certificate();

    assert!(cert.contains("#ifndef VELOX_VRAM_PROOF_CERTMODEL_H"));
    assert!(cert.contains("#define VELOX_VRAM_PROOF_CERTMODEL_H"));
    assert!(cert.contains("VERIFIED FEASIBLE (Q.E.D.)"));
    assert!(cert.contains("_Static_assert(VELOX_TOTAL_REQUIRED_VRAM_BYTES <= VELOX_SAFE_VRAM_BUDGET_BYTES"));
    assert!(cert.contains("#define VELOX_HARDWARE_VRAM_CAPACITY_BYTES"));
    assert!(cert.contains("#define VELOX_TIER1_WEIGHT_ARENA_BYTES"));
    assert!(cert.contains("#define VELOX_TIER2_KV_CACHE_POOL_BYTES"));
    assert!(cert.contains("#define VELOX_TIER3_ACTIVATION_ARENA_BYTES"));
    assert!(cert.contains("#define VELOX_TIER4_SYSTEM_RESERVE_BYTES"));
    assert!(cert.contains("#define VELOX_WEIGHT_OFFSET_W_Q"));
}

// ============================================================================
// Battery 8: End-to-End veloxc.exe CLI Verification
// ============================================================================

#[test]
fn test_stress_cli_all_5_models_prove_vram() {
    let models = [
        "examples/tinystories_smollm.vlx",
        "examples/llama3_8b.vlx",
        "examples/qwen2_5_7b.vlx",
        "examples/deepseek_v3_mla.vlx",
        "examples/nanbeige_33b.vlx",
    ];

    for path in models {
        let mut out = Vec::new();
        let mut err = Vec::new();
        let args = vec!["--prove-vram".to_string(), path.to_string()];
        let code = run_cli_with_io(&args, &mut out, &mut err);
        assert_eq!(code, EXIT_SUCCESS, "CLI --prove-vram failed for {path}");

        let stdout_str = String::from_utf8(out).unwrap();
        assert!(
            stdout_str.contains("VELOX COMPILE-TIME VRAM FEASIBILITY PROOF CERTIFICATE"),
            "Proof certificate header missing in {path}"
        );
        assert!(
            stdout_str.contains("VERIFIED FEASIBLE (Q.E.D.)"),
            "Proof feasibility marker missing in {path}"
        );
        assert!(
            stdout_str.contains("_Static_assert"),
            "_Static_assert missing in {path}"
        );
    }
}

// ============================================================================
// Battery 9: System Reserve Step-Function Thresholds
// ============================================================================

#[test]
fn test_stress_system_reserve_thresholds() {
    let gib = 1024 * 1024 * 1024u64;
    let mib = 1024 * 1024u64;

    // Boundary 1: >= 24 GiB -> 512 MiB
    assert_eq!(StaticMemoryPlanner::compute_system_reserve_bytes(100 * gib), 512 * mib);
    assert_eq!(StaticMemoryPlanner::compute_system_reserve_bytes(24 * gib), 512 * mib);

    // Boundary 2: 2 GiB .. < 24 GiB -> 256 MiB
    assert_eq!(StaticMemoryPlanner::compute_system_reserve_bytes(24 * gib - 1), 256 * mib);
    assert_eq!(StaticMemoryPlanner::compute_system_reserve_bytes(16 * gib), 256 * mib);
    assert_eq!(StaticMemoryPlanner::compute_system_reserve_bytes(4 * gib), 256 * mib);
    assert_eq!(StaticMemoryPlanner::compute_system_reserve_bytes(2 * gib), 256 * mib);

    // Boundary 3: < 2 GiB -> cap / 32
    assert_eq!(StaticMemoryPlanner::compute_system_reserve_bytes(2 * gib - 1), (2 * gib - 1) / 32);
    assert_eq!(StaticMemoryPlanner::compute_system_reserve_bytes(1 * gib), 32 * mib);
    assert_eq!(StaticMemoryPlanner::compute_system_reserve_bytes(32 * mib), 1 * mib);
    assert_eq!(StaticMemoryPlanner::compute_system_reserve_bytes(0), 0);
}

// ============================================================================
// Battery 10: MLA Matrix Absorption Offline Weight Folding Invariant
// ============================================================================

#[test]
fn test_stress_mla_matrix_absorption_weight_folding() {
    let w_absorbed_q = TypedWeight {
        name: "w_absorbed_q".to_string(),
        shape: vec![1024, 1024],
        dtype: ScalarType::F16,
        tier: MemoryTier::VramImmutable,
        size_bytes: 2 * 1024 * 1024, // 2 MiB
        scale_bytes: 0,
        is_absorbed: true, // Marked absorbed
        absorb_kind: None,
        file_path: None,
        tensor_key: None,
        span: dummy_span(),
    };
    let w_uk = TypedWeight {
        name: "w_uk".to_string(),
        shape: vec![1024, 512],
        dtype: ScalarType::F16,
        tier: MemoryTier::VramImmutable,
        size_bytes: 1 * 1024 * 1024, // 1 MiB
        scale_bytes: 0,
        is_absorbed: false,
        absorb_kind: None,
        file_path: None,
        tensor_key: None,
        span: dummy_span(),
    };
    let w_uv = TypedWeight {
        name: "w_uv".to_string(),
        shape: vec![1024, 512],
        dtype: ScalarType::F16,
        tier: MemoryTier::VramImmutable,
        size_bytes: 1 * 1024 * 1024, // 1 MiB
        scale_bytes: 0,
        is_absorbed: false,
        absorb_kind: None,
        file_path: None,
        tensor_key: None,
        span: dummy_span(),
    };

    // Scenario A: Offline MLA absorption present -> w_uk and w_uv folded (size = 0)
    let model_absorbed = TypedModel {
        name: "MlaAbsorbed".to_string(),
        constants: HashMap::new(),
        weights: vec![w_absorbed_q.clone(), w_uk.clone(), w_uv.clone()],
        kv_caches: Vec::new(),
        scopes: Vec::new(),
        span: dummy_span(),
    };
    let hw = TargetHardwareNode::new("test_gpu", 8 * 1024 * 1024 * 1024, (8, 6), 0.95, dummy_span());
    let mut p_abs = StaticMemoryPlanner::new();
    let plan_abs = p_abs.plan_model(&model_absorbed, &hw);
    assert_eq!(plan_abs.weight_arena_bytes, 2 * 1024 * 1024); // Only w_absorbed_q takes bytes!
    let w_uk_plan = plan_abs.weight_offsets.iter().find(|w| w.name == "w_uk").unwrap();
    assert_eq!(w_uk_plan.aligned_size_bytes, 0);

    // Scenario B: Offline MLA absorption NOT present -> w_uk and w_uv take full space
    let mut w_absorbed_q_unabsorbed = w_absorbed_q.clone();
    w_absorbed_q_unabsorbed.is_absorbed = false;
    let model_unabsorbed = TypedModel {
        name: "MlaUnabsorbed".to_string(),
        constants: HashMap::new(),
        weights: vec![w_absorbed_q_unabsorbed, w_uk, w_uv],
        kv_caches: Vec::new(),
        scopes: Vec::new(),
        span: dummy_span(),
    };
    let mut p_unabs = StaticMemoryPlanner::new();
    let plan_unabs = p_unabs.plan_model(&model_unabsorbed, &hw);
    assert_eq!(plan_unabs.weight_arena_bytes, 4 * 1024 * 1024); // 2 + 1 + 1 = 4 MiB
}

// ============================================================================
// Battery 11: Empty Scope and Empty Model Boundary Robustness
// ============================================================================

#[test]
fn test_stress_empty_scope_and_empty_model() {
    let empty_scope = TypedScope {
        name: "empty_scope".to_string(),
        kind: ScopeKind::Prefill,
        activations: Vec::new(),
        total_steps: 0,
        span: dummy_span(),
    };
    let plan_scope = StaticMemoryPlanner::color_scope_activations(&empty_scope);
    assert_eq!(plan_scope.actual_colored_peak_bytes, 0);
    assert_eq!(plan_scope.theoretical_lower_bound_bytes, 0);
    assert_eq!(plan_scope.competitive_ratio_rho, 1.0);
    assert!(plan_scope.allocations.is_empty());

    let empty_model = TypedModel {
        name: "EmptyModel".to_string(),
        constants: HashMap::new(),
        weights: Vec::new(),
        kv_caches: Vec::new(),
        scopes: vec![empty_scope],
        span: dummy_span(),
    };
    let hw = TargetHardwareNode::new("test_gpu", 4 * 1024 * 1024 * 1024, (8, 6), 0.95, dummy_span());
    let mut planner = StaticMemoryPlanner::new();
    let plan = planner.plan_model(&empty_model, &hw);
    assert!(plan.is_feasible);
    assert_eq!(plan.weight_arena_bytes, 0);
    assert_eq!(plan.kv_cache_pool_bytes, 0);
    assert_eq!(plan.activation_arena_bytes, 256); // Minimum arena floor is 256 bytes
    assert_eq!(plan.system_reserve_bytes, 256 * 1024 * 1024);
}

// ============================================================================
// Battery 12: Source-Level AST Loop Lifetime Extension and Interval Coloring
// ============================================================================

#[test]
fn test_stress_source_ast_loop_lifetimes_and_coloring() {
    use veloxc::lexer::Lexer;
    use veloxc::parser::Parser;
    use veloxc::sema::SemanticAnalyzer;

    let src = r#"
target hardware {
    device: "sm_86";
    vram_capacity: 4096 * MiB;
    compute_capability: 8.6;
    safety_factor: 0.95;
}

model LoopStressModel<const VOCAB: usize = 1000, const DIM: usize = 64> {
    @vram_immutable weight w_embed: Tensor<[VOCAB, DIM], f16> = safetensors("test.safetensors", "embed");

    scope prefill(x: Tensor<[1, 16], u32>) -> Tensor<[1, 16, DIM], f16> {
        let mut h = embed(x, w_embed);
        for i in 0..4 {
            let temp = h;
            h = temp;
        }
        return h;
    }

    scope decode(x: Tensor<[1, 1], u32>) -> Tensor<[1, 1, DIM], f16> {
        let h = embed(x, w_embed);
        return h;
    }
}
"#;
    let tokens = Lexer::tokenize_source(src, "loop_stress.vlx").expect("Lexing failed");
    let mut parser = Parser::new(tokens, "loop_stress.vlx", src);
    let prog = parser.parse_program_safe().expect("Parsing failed");
    let typed = SemanticAnalyzer::new().analyze(&prog).expect("Sema failed");
    let plans = StaticMemoryPlanner::new().plan_program(&typed).expect("Memory planning failed");

    assert_eq!(plans.len(), 1);
    let plan = &plans[0];
    assert!(plan.is_feasible);

    for sp in &plan.scope_plans {
        for alloc in &sp.allocations {
            assert_eq!(alloc.offset_bytes % 256, 0);
            assert_eq!(alloc.aligned_size_bytes % 256, 0);
        }

        // Pairwise disjointness check
        for i in 0..sp.allocations.len() {
            for j in (i + 1)..sp.allocations.len() {
                let a = &sp.allocations[i];
                let b = &sp.allocations[j];
                if a.birth_step <= b.death_step && b.birth_step <= a.death_step {
                    let a_end = a.offset_bytes + a.aligned_size_bytes;
                    let b_end = b.offset_bytes + b.aligned_size_bytes;
                    assert!(
                        a_end <= b.offset_bytes || b_end <= a.offset_bytes,
                        "Collision in scope {}: '{}' [{}, {}] and '{}' [{}, {}]",
                        sp.scope_name, a.name, a.birth_step, a.death_step,
                        b.name, b.birth_step, b.death_step
                    );
                }
            }
        }
    }
}

// ============================================================================
// Battery 13: Direct veloxc.exe Binary Process Execution on Boundary E002
// ============================================================================

#[test]
fn test_stress_cli_binary_process_execution_e002() {
    let tmp_dir = std::env::temp_dir();
    let bad_model_path = tmp_dir.join("velox_stress_e002_rejection.vlx");
    std::fs::write(
        &bad_model_path,
        r#"
target hardware {
    device: "oversized_vram_test";
    vram_capacity: 512 * MiB;
    compute_capability: 8.6;
    safety_factor: 0.95;
}

model GiantWeights<const V: usize = 200000, const D: usize = 8192> {
    @vram_immutable weight w_giant: Tensor<[V, D], f32> = safetensors("m.safetensors", "w");

    scope prefill(x: Tensor<[1, 16], f32>) -> Tensor<[1, 16], f32> {
        return x;
    }
    scope decode(x: Tensor<[1, 1], f32>) -> Tensor<[1, 1], f32> {
        return x;
    }
}
"#,
    )
    .unwrap();

    let exe_path = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("veloxc.exe");

    if exe_path.exists() {
        let output = std::process::Command::new(&exe_path)
            .arg("--prove-vram")
            .arg(&bad_model_path)
            .output()
            .expect("Failed to execute veloxc.exe process");

        let _ = std::fs::remove_file(&bad_model_path);
        assert!(!output.status.success(), "Process must fail on VRAM overflow");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("E002"), "stderr must contain diagnostic E002, got: {stderr}");
    }
}

