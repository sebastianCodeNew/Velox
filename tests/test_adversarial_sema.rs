//! Adversarial Empirical Stress Test Suite for Step 6 (`veloxc::sema` & `veloxc::memory_planner`)
//!
//! Thoroughly exercises:
//! 1. Dimension mismatch detection (E010, E011).
//! 2. Spatial memory tier violation detection (E004, E005, E006).
//! 3. KV Cache borrow checking across all 5 types (E012, E015, E016, E017, E018).
//! 4. Boundary inputs, zero/negative dimensions, wildcard dimensions, circular/aliased references.
//! 5. Verification that malformed programs result in clean diagnostics rather than compiler panics.

use veloxc::diagnostics::DiagnosticCode;
use veloxc::lexer::Lexer;
use veloxc::parser::Parser;
use veloxc::sema::SemanticAnalyzer;

fn analyze_src(src: &str) -> Result<veloxc::sema::TypedProgram, Vec<veloxc::diagnostics::Diagnostic>> {
    let tokens = match Lexer::tokenize_source(src, "adversarial_test.vlx") {
        Ok(t) => t,
        Err(e) => return Err(vec![veloxc::diagnostics::Diagnostic::error(DiagnosticCode::E023, e.message, e.span)]),
    };
    let mut parser = Parser::new(tokens, "adversarial_test.vlx", src);
    let program = match parser.parse_program_safe() {
        Some(p) => p,
        None => return Err(parser.diagnostics),
    };
    if !parser.diagnostics.is_empty() {
        return Err(parser.diagnostics);
    }
    let sema = SemanticAnalyzer::new();
    sema.analyze(&program)
}

fn assert_has_error(src: &str, expected_code: DiagnosticCode) {
    let res = analyze_src(src);
    assert!(
        res.is_err(),
        "Expected compilation error {:?}, but compilation succeeded!",
        expected_code
    );
    let diags = res.unwrap_err();
    let found = diags.iter().any(|d| d.code == expected_code);
    assert!(
        found,
        "Expected error code {:?}, but got diagnostics: {:?}",
        expected_code,
        diags.iter().map(|d| (&d.code, &d.message)).collect::<Vec<_>>()
    );
}

fn assert_compiles_cleanly(src: &str) {
    let res = analyze_src(src);
    assert!(
        res.is_ok(),
        "Expected source to compile cleanly, but got diagnostics: {:?}",
        res.as_ref().err()
    );
}

// ============================================================================
// 1. Dimension Mismatch Detection (E010, E011)
// ============================================================================

#[test]
fn test_adversarial_e010_matmul_2d_contraction_mismatch() {
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model Matmul2dMismatch {
        @vram_immutable weight w: Tensor<[2048, 4096], f16> = safetensors("m.safetensors", "w");
        scope prefill(x: Tensor<[1, 1024], f16>) -> Tensor<[1, 4096], f16> {
            let y: Tensor<[1, 4096], f16> = x @ w;
            return y;
        }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E010);
}

#[test]
fn test_adversarial_e010_matmul_4d_batched_contraction_mismatch() {
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model Matmul4dMismatch {
        scope prefill(
            a: Tensor<[1, 2, 16, 64], f16>,
            b: Tensor<[1, 2, 128, 32], f16>
        ) -> Tensor<[1, 2, 16, 32], f16> {
            let c = a @ b;
            return c;
        }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E010);
}

#[test]
fn test_adversarial_e010_3d_rhs_bypasses_contraction_check_finding() {
    // 1. Mismatched contraction dimensions (64 != 128) MUST emit E010
    let src_mismatch = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model Matmul3dBypass {
        scope prefill(
            a: Tensor<[2, 16, 64], f16>,
            b: Tensor<[2, 128, 32], f16>
        ) -> Tensor<[2, 16, 32], f16> {
            let c = a @ b;
            return c;
        }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    let res = analyze_src(src_mismatch);
    let has_e010 = res.as_ref().err().is_some_and(|diags| diags.iter().any(|d| d.code == DiagnosticCode::E010));
    assert!(
        has_e010,
        "3D RHS matmul contraction check must emit E010 on dimension mismatch"
    );

    // 2. Matching contraction dimensions (64 == 64) MUST succeed cleanly
    let src_matching = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model Matmul3dMatch {
        scope prefill(
            a: Tensor<[2, 16, 64], f16>,
            b: Tensor<[2, 64, 32], f16>
        ) -> Tensor<[2, 16, 32], f16> {
            let c = a @ b;
            return c;
        }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_compiles_cleanly(src_matching);

    // 3. Absorbed DeepSeek MLA matmul (1536 and [128, 512, 128]) MUST succeed cleanly
    let src_mla = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model MatmulMlaAbsorbed {
        scope prefill(
            c_q: Tensor<[1, 1, 1536], f16>,
            w_abs: Tensor<[128, 512, 128], f16>
        ) -> Tensor<[1, 1, 128], f16> {
            let q_abs = c_q @ w_abs;
            return q_abs;
        }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_compiles_cleanly(src_mla);

    // 4. Broken MLA matmul (1536 vs 256 != 512) MUST emit E010
    let src_mla_broken = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model MatmulMlaBroken {
        scope prefill(
            c_q: Tensor<[1, 1, 1536], f16>,
            w_abs: Tensor<[128, 256, 128], f16>
        ) -> Tensor<[1, 1, 128], f16> {
            let q_abs = c_q @ w_abs;
            return q_abs;
        }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src_mla_broken, DiagnosticCode::E010);
}

#[test]
fn test_adversarial_e010_swiglu_gate_dimension_mismatch() {
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model SwiGluGateMismatch {
        @vram_immutable weight w_gate: Tensor<[2048, 512], f16> = safetensors("m.safetensors", "gate");
        @vram_immutable weight w_up:   Tensor<[1024, 512], f16> = safetensors("m.safetensors", "up");
        @vram_immutable weight w_down: Tensor<[512, 1024], f16> = safetensors("m.safetensors", "down");
        scope prefill(x: Tensor<[1, 1024], f16>) -> Tensor<[1, 1024], f16> {
            // x last dim is 1024, but w_gate first dim is 2048
            let out: Tensor<[1, 1024], f16> = swiglu(x, w_gate, w_up, w_down);
            return out;
        }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E010);
}

#[test]
fn test_adversarial_e010_swiglu_down_dimension_mismatch() {
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model SwiGluDownMismatch {
        @vram_immutable weight w_gate: Tensor<[1024, 512], f16> = safetensors("m.safetensors", "gate");
        @vram_immutable weight w_up:   Tensor<[1024, 512], f16> = safetensors("m.safetensors", "up");
        @vram_immutable weight w_down: Tensor<[256, 1024], f16> = safetensors("m.safetensors", "down");
        scope prefill(x: Tensor<[1, 1024], f16>) -> Tensor<[1, 1024], f16> {
            // w_gate last dim is 512, but w_down first dim is 256
            let out: Tensor<[1, 1024], f16> = swiglu(x, w_gate, w_up, w_down);
            return out;
        }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E010);
}

#[test]
fn test_adversarial_e011_binary_elementwise_dimension_mismatch() {
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model AddMismatch {
        scope prefill(a: Tensor<[4, 8], f16>, b: Tensor<[4, 16], f16>) -> Tensor<[4, 8], f16> {
            let c: Tensor<[4, 8], f16> = a + b;
            return c;
        }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E011);
}

#[test]
fn test_adversarial_e011_tensor_rank_mismatch() {
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model RankMismatch {
        scope prefill(a: Tensor<[4, 8], f16>, b: Tensor<[4, 8, 16], f16>) -> Tensor<[4, 8], f16> {
            let c: Tensor<[4, 8], f16> = a;
            let d: Tensor<[4, 8, 16], f16> = b;
            let res: Tensor<[4, 8], f16> = d;
            return res;
        }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E011);
}

#[test]
fn test_adversarial_e011_rmsnorm_weight_dimension_mismatch() {
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model RmsnormMismatch {
        @vram_immutable weight w_norm: Tensor<[512], f16> = safetensors("m.safetensors", "norm");
        scope prefill(x: Tensor<[1, 1024], f16>) -> Tensor<[1, 1024], f16> {
            let out: Tensor<[1, 1024], f16> = rmsnorm(x, w_norm);
            return out;
        }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E011);
}

// ============================================================================
// 2. Spatial Memory Tier Violation Detection (E004, E005, E006)
// ============================================================================

#[test]
fn test_adversarial_e004_mutation_of_vram_immutable_weight() {
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model MutateWeight {
        @vram_immutable weight w: Tensor<[64, 64], f16> = safetensors("m.safetensors", "w");
        scope prefill(x: Tensor<[64, 64], f16>) -> Tensor<[64, 64], f16> {
            w = x;
            return x;
        }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E004);
}

#[test]
fn test_adversarial_e004_borrow_mut_of_vram_immutable_weight() {
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model BorrowMutWeight {
        @vram_immutable weight w: Tensor<[64, 64], f16> = safetensors("m.safetensors", "w");
        scope prefill(x: Tensor<[64, 64], f16>) -> Tensor<[64, 64], f16> {
            let ref_mut = &mut w;
            return x;
        }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E004);
}

#[test]
fn test_adversarial_e004_assign_to_vram_immutable_tier_annotated_var() {
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model NonImmutableToImmutable {
        scope prefill(x: @bump_sram Tensor<[64, 64], f16>) -> @bump_sram Tensor<[64, 64], f16> {
            let mut imm: @vram_immutable Tensor<[64, 64], f16>;
            imm = x;
            return x;
        }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E004);
}

#[test]
fn test_adversarial_e004_dma_write_into_vram_immutable() {
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model DmaToImmutable {
        @vram_immutable weight w: Tensor<[16, 16], f16> = safetensors("m.safetensors", "w");
        scope prefill(h: @host_pinned Tensor<[16, 16], f16>) -> Tensor<[16, 16], f16> {
            dma_async_h2d(h, w);
            return h;
        }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E004);
}

#[test]
fn test_adversarial_e004_explicit_cast_tier_prohibited() {
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model ExplicitCastTier {
        scope prefill(x: Tensor<[16, 16], f16>) -> Tensor<[16, 16], f16> {
            let y = cast_tier(x);
            return y;
        }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E004);
}

#[test]
fn test_adversarial_e005_illegal_dma_sram_to_host() {
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model DmaSramToHost {
        scope prefill(
            s: @bump_sram Tensor<[16, 16], f16>,
            h: @host_pinned Tensor<[16, 16], f16>
        ) -> Tensor<[16, 16], f16> {
            dma_async_d2h(s, h);
            return s;
        }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E005);
}

#[test]
fn test_adversarial_e005_illegal_dma_host_to_sram() {
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model DmaHostToSram {
        scope prefill(
            h: @host_pinned Tensor<[16, 16], f16>,
            s: @bump_sram Tensor<[16, 16], f16>
        ) -> Tensor<[16, 16], f16> {
            dma_async_h2d(h, s);
            return h;
        }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E005);
}

#[test]
fn test_adversarial_e005_dma_direction_mismatch() {
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model DmaDirMismatch {
        scope prefill(
            s: @bump_sram Tensor<[16, 16], f16>,
            v: @paged_vram Tensor<[16, 16], f16>
        ) -> Tensor<[16, 16], f16> {
            // dma_async_h2d source must be @host_pinned, not @bump_sram
            dma_async_h2d(s, v);
            return s;
        }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E005);
}

#[test]
fn test_adversarial_e006_sram_reference_escape_immut() {
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model SramEscapeImmut {
        scope prefill(s: @bump_sram Tensor<[16, 16], f16>) -> &Tensor<[16, 16], f16> {
            return &s;
        }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E006);
}

#[test]
fn test_adversarial_e006_sram_reference_escape_mut() {
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model SramEscapeMut {
        scope prefill(mut s: @bump_sram Tensor<[16, 16], f16>) -> &mut Tensor<[16, 16], f16> {
            return &mut s;
        }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E006);
}

// ============================================================================
// 3. KV Cache Borrow Checking across all 5 types (E012, E015, E016, E017, E018)
// ============================================================================

#[test]
fn test_adversarial_e012_kv_cache_aliased_mutable_borrow() {
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model KVAliasing {
        let mut kv: PagedKVCache<4, 4, 64, 16, f16>;
        scope prefill(k: Tensor<[1, 8, 4, 64], f16>) -> Tensor<[1, 1], u32> {
            let s1 = kv.borrow_mut_slot(0, 5);
            let s2 = kv.borrow_mut_slot(0, 5);
            let out: Tensor<[1, 1], u32> = argmax(k, -1);
            return out;
        }
        scope decode(x: Tensor<[1, 1], u32>) -> Tensor<[1, 1], u32> { return x; }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E012);
}

#[test]
fn test_adversarial_kv_cache_disjoint_mutable_borrows_succeed() {
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model KVDisjoint {
        let mut kv: PagedKVCache<4, 4, 64, 16, f16>;
        scope prefill(k: Tensor<[1, 8, 4, 64], f16>) -> Tensor<[1, 1], u32> {
            let s1 = kv.borrow_mut_slot(0, 5);
            let s2 = kv.borrow_mut_slot(0, 6);
            let out = argmax(k, -1);
            return out;
        }
        scope decode(x: Tensor<[1, 1], u32>) -> Tensor<[1, 1], u32> { return x; }
    }
    "#;
    assert_compiles_cleanly(src);
}

#[test]
fn test_adversarial_e015_paged_kv_invalid_block_size_8() {
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model BadBlock8 {
        let mut kv: PagedKVCache<4, 4, 64, 8, f16>;
        scope prefill(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E015);
}

#[test]
fn test_adversarial_e015_paged_kv_invalid_block_size_128() {
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model BadBlock128 {
        let mut kv: PagedKVCache<4, 4, 64, 128, f16>;
        scope prefill(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E015);
}

#[test]
fn test_adversarial_paged_kv_valid_block_sizes_16_32_64() {
    for bs in [16, 32, 64] {
        let src = format!(
            r#"
            target hardware {{ device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }}
            model ValidBlock{bs} {{
                let mut kv: PagedKVCache<4, 4, 64, {bs}, f16>;
                scope prefill(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> {{ return x; }}
                scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> {{ return x; }}
            }}
            "#
        );
        assert_compiles_cleanly(&src);
    }
}

#[test]
fn test_adversarial_e016_streaming_sink_zero_sinks() {
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model BadStreamingSinkZero {
        let mut kv: StreamingSinkKVCache<4, 4, 64, 0, 1024, f16>;
        scope prefill(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E016);
}

#[test]
fn test_adversarial_e016_streaming_sink_zero_window() {
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model BadStreamingWindowZero {
        let mut kv: StreamingSinkKVCache<4, 4, 64, 4, 0, f16>;
        scope prefill(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E016);
}

#[test]
fn test_adversarial_e017_speculative_tree_depth_exceeded() {
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model BadTreeDepth {
        let mut tree: SpeculativeTreeBlock<64, 20, 128, f16>;
        scope prefill(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E017);
}

#[test]
fn test_adversarial_e017_speculative_tree_nodes_exceeded() {
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model BadTreeNodes {
        let mut tree: SpeculativeTreeBlock<2048, 6, 128, f16>;
        scope prefill(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E017);
}

#[test]
fn test_adversarial_e018_mla_unabsorbed_access_in_decode() {
    let src = r#"
    target hardware { device: "sm_90"; vram_capacity: 80 * GiB; compute_capability: 9.0; }
    model MlaViolation {
        let mut kv: LatentKVCache<4, 512, 64, f16>;
        @vram_immutable weight w_uk: Tensor<[512, 2048], f16> = safetensors("m.safetensors", "w_uk");
        scope prefill(x: Tensor<[1, 512], f16>) -> Tensor<[1, 512], f16> { return x; }
        scope decode(latent: Tensor<[1, 512], f16>) -> Tensor<[1, 2048], f16> {
            let decompress: Tensor<[1, 2048], f16> = latent @ w_uk;
            return decompress;
        }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E018);
}

#[test]
fn test_adversarial_e018_mla_decompress_kv_call_in_decode() {
    let src = r#"
    target hardware { device: "sm_90"; vram_capacity: 80 * GiB; compute_capability: 9.0; }
    model MlaDecompressCall {
        let mut kv: LatentKVCache<4, 512, 64, f16>;
        scope prefill(x: Tensor<[1, 512], f16>) -> Tensor<[1, 512], f16> { return x; }
        scope decode(latent: Tensor<[1, 512], f16>) -> Tensor<[1, 512], f16> {
            let full_kv = decompress_kv(latent);
            return latent;
        }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E018);
}

// ============================================================================
// 4. Boundary Inputs, Zero/Negative Dimensions, and Diagnostics
// ============================================================================

#[test]
fn test_adversarial_e009_negative_weight_dimension() {
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model NegWeightDim {
        @vram_immutable weight w: Tensor<[-16, 32], f16> = safetensors("m.safetensors", "w");
        scope prefill(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E009);
}

#[test]
fn test_adversarial_e009_zero_weight_dimension() {
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model ZeroWeightDim {
        @vram_immutable weight w: Tensor<[0, 32], f16> = safetensors("m.safetensors", "w");
        scope prefill(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E009);
}

#[test]
fn test_adversarial_e009_negative_model_const_generic() {
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model NegConstGeneric<DIM: usize = -128> {
        scope prefill(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E009);
}

#[test]
fn test_adversarial_e008_missing_mandatory_decode_scope() {
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model MissingDecode {
        scope prefill(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E008);
}

#[test]
fn test_adversarial_e008_missing_mandatory_prefill_scope() {
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model MissingPrefill {
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E008);
}

#[test]
fn test_adversarial_e020_missing_tma_barrier_wait() {
    let src = r#"
    target hardware { device: "sm_90"; vram_capacity: 8192 * MiB; compute_capability: 9.0; }
    model UnsyncedTma {
        @vram_immutable weight w: Tensor<[64, 64], f16> = safetensors("m.safetensors", "w");
        scope prefill(x: Tensor<[1, 64], f16>, bar: mbarrier) -> Tensor<[1, 64], f16> {
            let mut tile: @bump_sram Tensor<[64, 64], f16>;
            tma_load_async(tile, w, bar);
            // Scope exits without waiting on bar -> E020
            return x;
        }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E020);
}

#[test]
fn test_adversarial_e020_tma_read_before_barrier_wait() {
    let src = r#"
    target hardware { device: "sm_90"; vram_capacity: 8192 * MiB; compute_capability: 9.0; }
    model TmaReadBeforeWait {
        @vram_immutable weight w: Tensor<[64, 64], f16> = safetensors("m.safetensors", "w");
        scope prefill(x: Tensor<[1, 64], f16>, bar: mbarrier) -> Tensor<[1, 64], f16> {
            let mut tile: @bump_sram Tensor<[64, 64], f16>;
            tma_load_async(w, tile, bar);
            // Read tile before waiting on barrier!
            let y = tile @ tile;
            mbarrier_wait(bar);
            return x;
        }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E020);
}

#[test]
fn test_adversarial_e021_dynamic_while_loop_prohibited() {
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model DynamicWhile {
        scope prefill(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> {
            while (true) {
                let y = x;
            }
            return x;
        }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E021);
}

#[test]
fn test_adversarial_e022_duplicate_weight_names() {
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model DuplicateWeights {
        @vram_immutable weight w: Tensor<[16, 16], f16> = safetensors("m.safetensors", "w1");
        @vram_immutable weight w: Tensor<[16, 16], f16> = safetensors("m.safetensors", "w2");
        scope prefill(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E022);
}

#[test]
fn test_adversarial_e023_undefined_identifier_clean_diagnostic() {
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model UndefinedVar {
        scope prefill(x: Tensor<[1, 16], f16>) -> Tensor<[1, 16], f16> {
            let y = x + completely_undefined_var;
            return y;
        }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E023);
}

#[test]
fn test_adversarial_e024_implicit_narrowing_cast() {
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model NarrowingCast {
        scope prefill(x: Tensor<[4, 64], f32>) -> Tensor<[4, 64], f16> {
            let y: Tensor<[4, 64], f16> = x;
            return y;
        }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E024);
}

#[test]
fn test_adversarial_e025_unaligned_scratchpad_indexing() {
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model UnalignedSram {
        scope prefill(s: @bump_sram Tensor<[1024], f16>) -> Tensor<[1], f16> {
            // Index 3 of f16 is byte offset 6, not 128-byte aligned -> E025
            let val = s[3];
            return s;
        }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E025);
}

#[test]
fn test_adversarial_e019_host_pinned_index_overflow() {
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model HostPinnedOverflow {
        scope prefill(h: @host_pinned Tensor<[32], f32>) -> Tensor<[1], f32> {
            // Index 64 exceeds bound 32 -> E019
            let val = h[64];
            return h;
        }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E019);
}

#[test]
fn test_adversarial_wildcard_dimension_unification() {
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model WildcardTest {
        @vram_immutable weight w: Tensor<[64, 128], f16> = safetensors("m.safetensors", "w");
        scope prefill(x: Tensor<[*, 64], f16>) -> Tensor<[*, 128], f16> {
            let y = x @ w;
            return y;
        }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_compiles_cleanly(src);
}

#[test]
fn test_adversarial_no_compiler_panic_on_empty_and_garbage_inputs() {
    let inputs = [
        "",
        "   \n\t  ",
        "model",
        "model Empty {}",
        "target hardware {}",
        "// only comments\n/* block comment */",
        "$$$ invalid token @@@",
        "model M { scope prefill() -> void { 1 + ; } }",
    ];

    for (idx, bad_src) in inputs.iter().enumerate() {
        // Compiler must NOT panic on any of these inputs
        let result = std::panic::catch_unwind(|| {
            let _ = analyze_src(bad_src);
        });
        assert!(
            result.is_ok(),
            "Compiler panicked on garbage input #{}: {:?}",
            idx,
            bad_src
        );
    }
}

// ============================================================================
// 5. Deep Probes: Architectural Edge Cases & Invariant Verification
// ============================================================================

#[test]
fn test_probe_return_type_shape_mismatch_behavior() {
    // 1. Scope return shape mismatch (512 != 1024) MUST emit E011
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model ReturnMismatch {
        scope prefill(x: Tensor<[1, 1024], f16>) -> Tensor<[1, 512], f16> {
            return x;
        }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E011);

    // 2. Scope return rank mismatch ([1, 1024, 64] vs [1, 1024]) MUST emit E011
    let src_rank = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model ReturnRankMismatch {
        scope prefill(x: Tensor<[1, 1024, 64], f16>) -> Tensor<[1, 1024], f16> {
            return x;
        }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src_rank, DiagnosticCode::E011);

    // 3. Matching scope return MUST compile cleanly
    let src_match = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model ReturnMatch {
        scope prefill(x: Tensor<[1, 512], f16>) -> Tensor<[1, 512], f16> {
            return x;
        }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_compiles_cleanly(src_match);
}

#[test]
fn test_probe_negative_parameter_dimension_behavior() {
    // 1. Negative dimension in scope parameter (-16) MUST emit E009
    let src_neg = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model NegParamDim {
        scope prefill(x: Tensor<[-16, 32], f16>) -> Tensor<[1, 1], f16> {
            return x;
        }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src_neg, DiagnosticCode::E009);

    // 2. Zero dimension in scope parameter (0) MUST emit E009
    let src_zero = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model ZeroParamDim {
        scope prefill(x: Tensor<[0, 32], f16>) -> Tensor<[1, 1], f16> {
            return x;
        }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src_zero, DiagnosticCode::E009);

    // 3. Positive dimension in scope parameter (16) MUST compile cleanly
    let src_pos = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model PosParamDim {
        scope prefill(x: Tensor<[16, 32], f16>) -> Tensor<[16, 32], f16> {
            return x;
        }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_compiles_cleanly(src_pos);
}

#[test]
fn test_probe_borrow_scope_release_in_inner_block() {
    // Probe whether active_borrows properly tracks lexical blocks:
    // If a borrow occurs inside an inner block { let s = kv.borrow_mut_slot(0, 0); },
    // does it get released when the block ends so the same slot can be borrowed in a subsequent block?
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model BlockBorrowRelease {
        let mut kv: PagedKVCache<4, 4, 64, 16, f16>;
        scope prefill(k: Tensor<[1, 8, 4, 64], f16>) -> Tensor<[1, 1], u32> {
            for i in 0..1 {
                let s1 = kv.borrow_mut_slot(0, 0);
            }
            for j in 0..1 {
                // In separate sequential loop, borrowing slot (0, 0) should be legal
                let s2 = kv.borrow_mut_slot(0, 0);
            }
            let out = argmax(k, -1);
            return out;
        }
        scope decode(x: Tensor<[1, 1], u32>) -> Tensor<[1, 1], u32> { return x; }
    }
    "#;
    let res = analyze_src(src);
    assert!(
        res.is_ok(),
        "Sequential borrows of slot (0, 0) in sequential for loops must succeed! Got: {:?}",
        res.as_ref().err()
    );
}

#[test]
fn test_probe_self_referencing_variable_declaration() {
    // Probe: let x = x + 1; when x has not been defined previously.
    // Must cleanly emit E023 for undefined variable x.
    let src = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; }
    model SelfRef {
        scope prefill(val: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> {
            let x = x + val;
            return x;
        }
        scope decode(x: Tensor<[1, 1], f16>) -> Tensor<[1, 1], f16> { return x; }
    }
    "#;
    assert_has_error(src, DiagnosticCode::E023);
}

#[test]
fn test_probe_first_fit_decreasing_alignment_and_no_collision_oracle() {
    // Directly test the StaticMemoryPlanner coloring engine with artificially overlapping activations
    use veloxc::sema::TypedActivation;
    use veloxc::types::{MemoryTier, ScalarType, ScopeKind, SourceLocation, SourceSpan};

    let span = SourceSpan::new(
        "test.vlx",
        SourceLocation::new(1, 1, 0),
        SourceLocation::new(1, 1, 0),
    );

    // Create 5 activations with heavily overlapping intervals and varying sizes:
    // a0: [1, 10], size 1000
    // a1: [2, 6],  size 3000
    // a2: [5, 8],  size 500
    // a3: [7, 12], size 2000
    // a4: [9, 15], size 4000
    let activations = vec![
        TypedActivation {
            id: 0,
            name: "act0".to_string(),
            scope: ScopeKind::Prefill,
            shape: vec![1000],
            dtype: ScalarType::U8,
            tier: MemoryTier::BumpSram,
            size_bytes: 1000,
            birth_step: 1,
            death_step: 10,
            span: span.clone(),
        },
        TypedActivation {
            id: 1,
            name: "act1".to_string(),
            scope: ScopeKind::Prefill,
            shape: vec![3000],
            dtype: ScalarType::U8,
            tier: MemoryTier::BumpSram,
            size_bytes: 3000,
            birth_step: 2,
            death_step: 6,
            span: span.clone(),
        },
        TypedActivation {
            id: 2,
            name: "act2".to_string(),
            scope: ScopeKind::Prefill,
            shape: vec![500],
            dtype: ScalarType::U8,
            tier: MemoryTier::BumpSram,
            size_bytes: 500,
            birth_step: 5,
            death_step: 8,
            span: span.clone(),
        },
        TypedActivation {
            id: 3,
            name: "act3".to_string(),
            scope: ScopeKind::Prefill,
            shape: vec![2000],
            dtype: ScalarType::U8,
            tier: MemoryTier::BumpSram,
            size_bytes: 2000,
            birth_step: 7,
            death_step: 12,
            span: span.clone(),
        },
        TypedActivation {
            id: 4,
            name: "act4".to_string(),
            scope: ScopeKind::Prefill,
            shape: vec![4000],
            dtype: ScalarType::U8,
            tier: MemoryTier::BumpSram,
            size_bytes: 4000,
            birth_step: 9,
            death_step: 15,
            span: span.clone(),
        },
    ];

    let typed_scope = veloxc::sema::TypedScope {
        name: "test_prefill".to_string(),
        kind: ScopeKind::Prefill,
        activations,
        total_steps: 16,
        span,
    };

    let plan = veloxc::memory_planner::StaticMemoryPlanner::color_scope_activations(&typed_scope);
    assert_eq!(plan.allocations.len(), 5);

    // Verify 256-byte alignment of every allocation offset
    for alloc in &plan.allocations {
        assert_eq!(
            alloc.offset_bytes % 256,
            0,
            "Allocation '{}' offset {} must be 256-byte aligned",
            alloc.name,
            alloc.offset_bytes
        );
        assert_eq!(
            alloc.aligned_size_bytes % 256,
            0,
            "Allocation '{}' size {} must be 256-byte aligned",
            alloc.name,
            alloc.aligned_size_bytes
        );
    }

    // Verify disjoint memory for overlapping intervals
    for i in 0..plan.allocations.len() {
        for j in (i + 1)..plan.allocations.len() {
            let a = &plan.allocations[i];
            let b = &plan.allocations[j];
            let intervals_overlap = a.birth_step <= b.death_step && b.birth_step <= a.death_step;
            if intervals_overlap {
                let a_end = a.offset_bytes + a.aligned_size_bytes;
                let b_end = b.offset_bytes + b.aligned_size_bytes;
                let disjoint = a_end <= b.offset_bytes || b_end <= a.offset_bytes;
                assert!(
                    disjoint,
                    "Memory collision between '{}' [{}..{}) and '{}' [{}..{})",
                    a.name, a.offset_bytes, a_end, b.name, b.offset_bytes, b_end
                );
            }
        }
    }

    // Verify competitive ratio rho <= 1.70
    assert!(
        plan.competitive_ratio_rho <= 1.70,
        "FFD competitive ratio rho = {} exceeds 1.70 upper bound",
        plan.competitive_ratio_rho
    );
}

