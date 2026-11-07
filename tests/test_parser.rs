//! Integration tests for Velox recursive descent parser (`tests/test_parser.rs`)
//!
//! Verifies:
//! - Target hardware declarations and safety factor defaults
//! - Model declarations with const-generics and defaults
//! - Weight declarations with safetensors and offline MLA absorption
//! - All 5 first-class KV cache architectures
//! - Execution scopes (prefill, decode, chunked_prefill, speculative_verify) and attributes
//! - 14-level Pratt operator precedence and associativity
//! - Control flow statements (if/else, for in .., while, yield, return)
//! - Asynchronous hardware transfers (TMA, DMA)
//! - Panic-mode error recovery across statement (;) and block (}) boundaries
//! - Successful parsing of all 5 production reference models in `examples/`

use std::fs;
use std::path::Path;
use veloxc::ast::*;
use veloxc::parser::Parser;
use veloxc::types::*;

fn parse_code(code: &str) -> ProgramNode {
    let mut parser = Parser::from_source(code, "test.vlx").expect("Lexer error");
    match parser.parse() {
        Ok(prog) => prog,
        Err(diags) => {
            let rendered = diags
                .iter()
                .map(|d| format!("[{}]: {}", d.code_str(), d.message))
                .collect::<Vec<_>>()
                .join("\n");
            panic!("Parser failed with diagnostics:\n{rendered}");
        }
    }
}

fn parse_safe(code: &str) -> (Option<ProgramNode>, Vec<veloxc::diagnostics::Diagnostic>) {
    let mut parser = Parser::from_source(code, "test.vlx").expect("Lexer error");
    let prog = parser.parse_program_safe();
    let diags = parser.diagnostics;
    (prog, diags)
}

// ============================================================================
// 1. Target Hardware Declarations
// ============================================================================

#[test]
fn test_target_hardware_full_declaration() {
    let code = r#"
    target hardware {
        device: "nvidia_rtx_3050_laptop";
        vram_capacity: 4096 * MiB;
        compute_capability: 8.6;
        safety_factor: 0.90;
    }
    model M {
        scope prefill() -> u32 { return 0; }
        scope decode() -> u32 { return 0; }
    }
    "#;
    let program = parse_code(code);
    let hw = program.target_hardware.as_ref().expect("Expected TargetHardwareNode");
    assert_eq!(hw.device_name, "nvidia_rtx_3050_laptop");
    assert_eq!(hw.vram_capacity_bytes, 4096 * 1024 * 1024);
    assert_eq!(hw.compute_capability, (8, 6));
    assert!((hw.safety_factor - 0.90).abs() < 1e-5);
}

#[test]
fn test_target_hardware_default_safety_factor() {
    let code = r#"
    target hardware {
        device: "sm_90";
        vram_capacity: 80 * GiB;
        compute_capability: 9.0;
    }
    model M {
        scope prefill() -> u32 { return 0; }
        scope decode() -> u32 { return 0; }
    }
    "#;
    let program = parse_code(code);
    let hw = program.target_hardware.as_ref().expect("Expected TargetHardwareNode");
    assert_eq!(hw.device_name, "sm_90");
    assert_eq!(hw.vram_capacity_bytes, 80 * 1024 * 1024 * 1024);
    assert_eq!(hw.compute_capability, (9, 0));
    assert!((hw.safety_factor - 0.95).abs() < 1e-5);
}

// ============================================================================
// 2. Model Declarations & Const-Generics
// ============================================================================

#[test]
fn test_model_const_generic_parameters() {
    let code = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; safety_factor: 0.95; }
    model Llama<const B: usize = 32, const S: usize = 4096, const HEADS: usize = 32> {
        scope prefill() -> u32 { return 0; }
        scope decode() -> u32 { return 0; }
    }
    "#;
    let program = parse_code(code);
    let model = &program.model_declarations[0];
    assert_eq!(model.name, "Llama");
    assert_eq!(model.generic_params.len(), 3);

    assert_eq!(model.generic_params[0].name, "B");
    assert_eq!(model.generic_params[0].type_name, "usize");
    if let Some(ref def) = model.generic_params[0].default_val {
        if let ExprNode::Literal(lit) = &**def {
            assert_eq!(lit.value, LiteralValue::Decimal(32));
        } else {
            panic!("Expected literal default value");
        }
    } else {
        panic!("Missing default value for B");
    }

    assert_eq!(model.generic_params[1].name, "S");
    assert_eq!(model.generic_params[2].name, "HEADS");
}

// ============================================================================
// 3. Weight Declarations & Safetensors Bindings
// ============================================================================

#[test]
fn test_weight_declarations_with_safetensors() {
    let code = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; safety_factor: 0.95; }
    model Transformer {
        @vram_immutable weight w_q: Tensor<[4096, 4096], f16> = safetensors("model.safetensors", "layers.0.q_proj");
        @vram_immutable weight w_gate: Tensor<[4096, 11008], bf16> = safetensors("model.safetensors", "layers.0.gate_proj");

        scope prefill() -> u32 { return 0; }
        scope decode() -> u32 { return 0; }
    }
    "#;
    let program = parse_code(code);
    let model = &program.model_declarations[0];
    assert_eq!(model.weights.len(), 2);

    let w_q = &model.weights[0];
    assert_eq!(w_q.name, "w_q");
    assert_eq!(w_q.tier, MemoryTier::VramImmutable);
    let t_type = w_q.tensor_type.as_ref().unwrap();
    assert_eq!(t_type.dtype, ScalarType::F16);
    assert_eq!(w_q.file_path.as_deref(), Some("model.safetensors"));
    assert_eq!(w_q.tensor_key.as_deref(), Some("layers.0.q_proj"));

    let w_gate = &model.weights[1];
    assert_eq!(w_gate.name, "w_gate");
    let t_gate_type = w_gate.tensor_type.as_ref().unwrap();
    assert_eq!(t_gate_type.dtype, ScalarType::Bf16);
    assert_eq!(w_gate.tensor_key.as_deref(), Some("layers.0.gate_proj"));
}

#[test]
fn test_weight_offline_mla_absorption() {
    let code = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; safety_factor: 0.95; }
    model DeepSeekMLA {
        @vram_immutable weight w_q: Tensor<[4096, 1536], f16> = safetensors("m.safetensors", "q_proj");
        @vram_immutable weight w_uk: Tensor<[512, 4096], f16> = safetensors("m.safetensors", "uk_proj");
        @vram_immutable weight w_uv: Tensor<[512, 4096], f16> = safetensors("m.safetensors", "uv_proj");
        @vram_immutable weight w_absorbed_q: Tensor<[4096, 512], f16> = absorb_mla_q(w_q, w_uk);
        @vram_immutable weight w_absorbed_v: Tensor<[512, 4096], f16> = absorb_mla_v(w_uv, w_q);

        scope prefill() -> u32 { return 0; }
        scope decode() -> u32 { return 0; }
    }
    "#;
    let program = parse_code(code);
    let model = &program.model_declarations[0];
    assert_eq!(model.weights.len(), 5);

    let w_abs_q = &model.weights[3];
    match &w_abs_q.binding {
        Some(WeightBinding::AbsorbMla(node)) => {
            assert_eq!(node.kind, AbsorbKind::AbsorbMlaQ);
            if let ExprNode::Identifier(id) = &*node.projection_q {
                assert_eq!(id.name, "w_q");
            } else {
                panic!("Expected identifier projection_q");
            }
        }
        _ => panic!("Expected AbsorbMla binding"),
    }

    let w_abs_v = &model.weights[4];
    match &w_abs_v.binding {
        Some(WeightBinding::AbsorbMla(node)) => {
            assert_eq!(node.kind, AbsorbKind::AbsorbMlaV);
            if let ExprNode::Identifier(id) = &*node.projection_q {
                assert_eq!(id.name, "w_uv");
            } else {
                panic!("Expected identifier projection_q");
            }
        }
        _ => panic!("Expected AbsorbMla binding"),
    }
}

// ============================================================================
// 4. KV Cache Declarations (All 5 Types)
// ============================================================================

#[test]
fn test_all_five_kv_cache_types() {
    let code = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; safety_factor: 0.95; }
    model ModelWithAllKVs {
        let mut kv_paged: PagedKVCache<32, 8, 128, 16, f16>;
        let mut kv_virtual: VirtualContiguousKV<32, 8, 128, 65536, f16>;
        let mut kv_latent: LatentKVCache<32, 512, 64, bf16>;
        let mut kv_streaming: StreamingSinkKVCache<32, 8, 128, 4, 1024, f16>;
        let mut kv_tree: SpeculativeTreeBlock<64, 8, 128, f16>;

        scope prefill() -> u32 { return 0; }
        scope decode() -> u32 { return 0; }
    }
    "#;
    let program = parse_code(code);
    let model = &program.model_declarations[0];
    assert_eq!(model.kv_caches.len(), 5);

    let kv0 = &model.kv_caches[0];
    assert_eq!(kv0.name, "kv_paged");
    assert!(kv0.is_mut);
    match &kv0.cache_type {
        Some(ts) => match &ts.kind {
            TypeKind::PagedKV(p) => {
                assert_eq!(p.dtype, ScalarType::F16);
            }
            _ => panic!("Expected PagedKV"),
        },
        _ => panic!("Missing cache type"),
    }

    let kv1 = &model.kv_caches[1];
    assert_eq!(kv1.name, "kv_virtual");
    match &kv1.cache_type {
        Some(ts) => match &ts.kind {
            TypeKind::VirtualKV(_) => {}
            _ => panic!("Expected VirtualKV"),
        },
        _ => panic!("Missing cache type"),
    }

    let kv2 = &model.kv_caches[2];
    assert_eq!(kv2.name, "kv_latent");
    match &kv2.cache_type {
        Some(ts) => match &ts.kind {
            TypeKind::LatentKV(l) => {
                assert_eq!(l.dtype, ScalarType::Bf16);
            }
            _ => panic!("Expected LatentKV"),
        },
        _ => panic!("Missing cache type"),
    }

    let kv3 = &model.kv_caches[3];
    assert_eq!(kv3.name, "kv_streaming");
    match &kv3.cache_type {
        Some(ts) => match &ts.kind {
            TypeKind::StreamingKV(_) => {}
            _ => panic!("Expected StreamingKV"),
        },
        _ => panic!("Missing cache type"),
    }

    let kv4 = &model.kv_caches[4];
    assert_eq!(kv4.name, "kv_tree");
    match &kv4.cache_type {
        Some(ts) => match &ts.kind {
            TypeKind::SpeculativeTree(_) => {}
            _ => panic!("Expected SpeculativeTree"),
        },
        _ => panic!("Missing cache type"),
    }
}

// ============================================================================
// 5. Execution Scopes (prefill, decode, chunked_prefill, speculative_verify)
// ============================================================================

#[test]
fn test_execution_scopes_all_variants_and_attributes() {
    let code = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; safety_factor: 0.95; }
    model NanbeigeServing {
        @kernel_fuse(mode = "fused_prefill_mha")
        scope prefill(tokens: Tensor<[1, 1024], u32>) -> Tensor<[1, 1024, 4096], f16> {
            return 0;
        }

        @warp_specialized(num_warps = 4)
        scope decode(token: Tensor<[1, 1], u32>) -> Tensor<[1, 1, 4096], f16> {
            return 0;
        }

        scope chunked_prefill(chunk: Tensor<[1, 256], u32>, is_first: bool) -> f16 {
            return 0;
        }

        scope speculative_verify(draft_tokens: Tensor<[1, 8], u32>) -> u32 {
            return 0;
        }
    }
    "#;
    let program = parse_code(code);
    let model = &program.model_declarations[0];
    assert_eq!(model.scopes.len(), 4);

    let s0 = &model.scopes[0];
    assert_eq!(s0.kind(), ScopeKind::Prefill);
    match s0 {
        ExecutionScopeNode::Prefill(n) => {
            assert_eq!(n.attributes.len(), 1);
            assert_eq!(n.attributes[0].name, "@kernel_fuse");
            assert_eq!(n.attributes[0].params[0].0, "mode");
        }
        _ => panic!("Expected PrefillScopeNode"),
    }

    let s1 = &model.scopes[1];
    assert_eq!(s1.kind(), ScopeKind::Decode);
    match s1 {
        ExecutionScopeNode::Decode(n) => {
            assert_eq!(n.attributes.len(), 1);
            assert_eq!(n.attributes[0].name, "@warp_specialized");
            assert_eq!(n.attributes[0].params[0].0, "num_warps");
        }
        _ => panic!("Expected DecodeScopeNode"),
    }

    let s2 = &model.scopes[2];
    assert_eq!(s2.kind(), ScopeKind::ChunkedPrefill);

    let s3 = &model.scopes[3];
    assert_eq!(s3.kind(), ScopeKind::SpeculativeVerify);
}

// ============================================================================
// 6. 14-Level Operator Precedence & Associativity
// ============================================================================

#[test]
fn test_matmul_precedence_and_left_associativity() {
    let code = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; safety_factor: 0.95; }
    model PrecedenceTest {
        scope prefill(x: Tensor<[1, 10], f16>) -> f16 {
            let res1 = x @ w + b;
            let res2 = x @ w1 @ w2;
            return 0;
        }
        scope decode() -> u32 { return 0; }
    }
    "#;
    let program = parse_code(code);
    let scope = match &program.model_declarations[0].scopes[0] {
        ExecutionScopeNode::Prefill(n) => n,
        _ => panic!("Expected prefill"),
    };
    let stmts = &scope.body.statements;

    // let res1 = x @ w + b; -> Root is Add (+), Left is Matmul (@)
    if let StmtNode::Let(l1) = &stmts[0] {
        if let Some(ref expr) = l1.init_expr {
            if let ExprNode::Binary(bin) = &**expr {
                assert_eq!(bin.op, BinaryOperator::Add);
                assert!(matches!(&*bin.left, ExprNode::TensorMatmul(_)));
            } else {
                panic!("Expected Binary + for res1");
            }
        } else {
            panic!("Missing init_expr");
        }
    } else {
        panic!("Expected let stmt");
    }

    // let res2 = x @ w1 @ w2; -> Left associative: (x @ w1) @ w2
    if let StmtNode::Let(l2) = &stmts[1] {
        if let Some(ref expr) = l2.init_expr {
            if let ExprNode::TensorMatmul(mat) = &**expr {
                assert!(matches!(&*mat.lhs, ExprNode::TensorMatmul(_)));
            } else {
                panic!("Expected TensorMatmul for res2");
            }
        } else {
            panic!("Missing init_expr");
        }
    } else {
        panic!("Expected let stmt");
    }
}

#[test]
fn test_level_1_postfix_chains_bind_tighter_than_binary() {
    let code = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; safety_factor: 0.95; }
    model PostfixTest {
        scope prefill() -> u32 {
            let res = a.b.c[0..64] + 5;
            return 0;
        }
        scope decode() -> u32 { return 0; }
    }
    "#;
    let program = parse_code(code);
    let scope = match &program.model_declarations[0].scopes[0] {
        ExecutionScopeNode::Prefill(n) => n,
        _ => panic!("Expected prefill"),
    };
    let stmt = &scope.body.statements[0];

    if let StmtNode::Let(l) = stmt {
        if let Some(ref expr) = l.init_expr {
            if let ExprNode::Binary(bin) = &**expr {
                assert_eq!(bin.op, BinaryOperator::Add);
                assert!(matches!(&*bin.left, ExprNode::TensorSlice(_)));
            } else {
                panic!("Expected Binary Add");
            }
        }
    }
}

#[test]
fn test_logical_and_binds_tighter_than_logical_or() {
    let code = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; safety_factor: 0.95; }
    model LogicalPrecedence {
        scope prefill() -> u32 {
            let res = a || b && c;
            return 0;
        }
        scope decode() -> u32 { return 0; }
    }
    "#;
    let program = parse_code(code);
    let scope = match &program.model_declarations[0].scopes[0] {
        ExecutionScopeNode::Prefill(n) => n,
        _ => panic!("Expected prefill"),
    };

    if let StmtNode::Let(l) = &scope.body.statements[0] {
        if let Some(ref expr) = l.init_expr {
            if let ExprNode::Binary(bin) = &**expr {
                assert_eq!(bin.op, BinaryOperator::LogicalOr);
                if let ExprNode::Binary(rbin) = &*bin.right {
                    assert_eq!(rbin.op, BinaryOperator::LogicalAnd);
                } else {
                    panic!("Expected LogicalAnd on right");
                }
            } else {
                panic!("Expected Binary LogicalOr");
            }
        }
    }
}

#[test]
fn test_multiplication_over_addition_over_relational_over_equality() {
    let code = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; safety_factor: 0.95; }
    model PrecedenceHierarchy {
        scope prefill() -> u32 {
            let res = 1 + 2 * 3 < 10 == true;
            return 0;
        }
        scope decode() -> u32 { return 0; }
    }
    "#;
    let program = parse_code(code);
    let scope = match &program.model_declarations[0].scopes[0] {
        ExecutionScopeNode::Prefill(n) => n,
        _ => panic!("Expected prefill"),
    };

    if let StmtNode::Let(l) = &scope.body.statements[0] {
        if let Some(ref expr) = l.init_expr {
            // Root is Equality ==
            if let ExprNode::Binary(eq_node) = &**expr {
                assert_eq!(eq_node.op, BinaryOperator::Eq);
                // Left is Relational <
                if let ExprNode::Binary(lt_node) = &*eq_node.left {
                    assert_eq!(lt_node.op, BinaryOperator::Lt);
                    // Left of relational is Add +
                    if let ExprNode::Binary(add_node) = &*lt_node.left {
                        assert_eq!(add_node.op, BinaryOperator::Add);
                        // Right of add is Mul *
                        if let ExprNode::Binary(mul_node) = &*add_node.right {
                            assert_eq!(mul_node.op, BinaryOperator::Mul);
                        } else {
                            panic!("Expected Mul node");
                        }
                    } else {
                        panic!("Expected Add node");
                    }
                } else {
                    panic!("Expected Lt node");
                }
            } else {
                panic!("Expected Eq node");
            }
        }
    }
}

#[test]
fn test_bitwise_precedence_hierarchy() {
    let code = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; safety_factor: 0.95; }
    model BitwiseTest {
        scope prefill() -> u32 {
            let res = a | b ^ c & d;
            return 0;
        }
        scope decode() -> u32 { return 0; }
    }
    "#;
    let program = parse_code(code);
    let scope = match &program.model_declarations[0].scopes[0] {
        ExecutionScopeNode::Prefill(n) => n,
        _ => panic!("Expected prefill"),
    };

    if let StmtNode::Let(l) = &scope.body.statements[0] {
        if let Some(ref expr) = l.init_expr {
            // Root is Bitwise OR |
            if let ExprNode::Binary(or_node) = &**expr {
                assert_eq!(or_node.op, BinaryOperator::BitOr);
                // Right is Bitwise XOR ^
                if let ExprNode::Binary(xor_node) = &*or_node.right {
                    assert_eq!(xor_node.op, BinaryOperator::BitXor);
                    // Right of XOR is Bitwise AND &
                    if let ExprNode::Binary(and_node) = &*xor_node.right {
                        assert_eq!(and_node.op, BinaryOperator::BitAnd);
                    } else {
                        panic!("Expected BitAnd");
                    }
                } else {
                    panic!("Expected BitXor");
                }
            }
        }
    }
}

// ============================================================================
// 7. Control Flow Statements
// ============================================================================

#[test]
fn test_if_else_control_flow() {
    let code = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; safety_factor: 0.95; }
    model ControlFlow {
        scope decode(pos: usize) -> u32 {
            if (pos > 0) {
                return 1;
            } else {
                return 0;
            }
        }
        scope prefill() -> u32 { return 0; }
    }
    "#;
    let program = parse_code(code);
    let scope = match &program.model_declarations[0].scopes[0] {
        ExecutionScopeNode::Decode(n) => n,
        _ => panic!("Expected decode"),
    };

    if let StmtNode::If(if_node) = &scope.body.statements[0] {
        assert_eq!(if_node.then_branch.statements.len(), 1);
        assert!(if_node.else_branch.is_some());
    } else {
        panic!("Expected IfNode");
    }
}

#[test]
fn test_for_loop_and_while_loop() {
    let code = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; safety_factor: 0.95; }
    model LoopTest {
        scope decode() -> u32 {
            for i in 0..32 {
                yield();
            }
            while (x < 10) {
                x += 1;
            }
            return 0;
        }
        scope prefill() -> u32 { return 0; }
    }
    "#;
    let program = parse_code(code);
    let scope = match &program.model_declarations[0].scopes[0] {
        ExecutionScopeNode::Decode(n) => n,
        _ => panic!("Expected decode"),
    };

    assert!(matches!(&scope.body.statements[0], StmtNode::For(_)));
    assert!(matches!(&scope.body.statements[1], StmtNode::While(_)));
}

// ============================================================================
// 8. Asynchronous Hardware Transfers
// ============================================================================

#[test]
fn test_async_dma_and_tma_transfers() {
    let code = r#"
    target hardware { device: "sm_90"; vram_capacity: 80 * GiB; compute_capability: 9.0; safety_factor: 0.95; }
    model HopperAsync {
        scope decode() -> u32 {
            tma_load_async(src_tensor, dst_smem, mbarrier_token);
            tma_store_async(src_smem, dst_vram);
            dma_async_h2d(host_buffer, vram_buffer, 4096);
            dma_async_d2h(vram_buffer, host_buffer, 4096);
            return 0;
        }
        scope prefill() -> u32 { return 0; }
    }
    "#;
    let program = parse_code(code);
    let scope = match &program.model_declarations[0].scopes[0] {
        ExecutionScopeNode::Decode(n) => n,
        _ => panic!("Expected decode"),
    };
    let stmts = &scope.body.statements;

    assert!(matches!(&stmts[0], StmtNode::TmaLoad(_)));
    assert!(matches!(&stmts[1], StmtNode::TmaStore(_)));
    if let StmtNode::AsyncDmaTransfer(d1) = &stmts[2] {
        assert_eq!(d1.direction, DmaDirection::H2D);
    }
    if let StmtNode::AsyncDmaTransfer(d2) = &stmts[3] {
        assert_eq!(d2.direction, DmaDirection::D2H);
    }
}

// ============================================================================
// 9. Panic Mode Error Recovery
// ============================================================================

#[test]
fn test_panic_mode_recovers_at_statement_semicolon_barrier() {
    let code = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; safety_factor: 0.95; }
    model RecoveryModel {
        scope prefill() -> u32 {
            let corrupted = ; // Syntax error here
            let clean_var = 42; // Should recover and parse cleanly
            return clean_var;
        }
        scope decode() -> u32 { return 0; }
    }
    "#;
    let (ast, diags) = parse_safe(code);
    assert!(!diags.is_empty(), "Expected error diagnostics");
    let program = ast.expect("Expected recovered AST");
    let scope = match &program.model_declarations[0].scopes[0] {
        ExecutionScopeNode::Prefill(n) => n,
        _ => panic!("Expected prefill"),
    };
    let names: Vec<&str> = scope
        .body
        .statements
        .iter()
        .filter_map(|s| match s {
            StmtNode::Let(l) => Some(l.name.as_str()),
            _ => None,
        })
        .collect();
    assert!(names.contains(&"clean_var"), "Clean statement was not parsed!");
}

#[test]
fn test_panic_mode_recovers_at_block_and_declaration_barriers() {
    let code = r#"
    target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; safety_factor: 0.95; }
    model BrokenModel {
        let bad = !!! ; // Malformed statement inside model
        scope prefill() -> u32 { return 1; }
        scope decode() -> u32 { return 2; }
    }
    "#;
    let (ast, diags) = parse_safe(code);
    assert!(!diags.is_empty());
    let program = ast.expect("Expected recovered AST");
    assert_eq!(program.model_declarations.len(), 1);
    assert_eq!(program.model_declarations[0].scopes.len(), 2);
}

#[test]
fn test_negative_parser_no_infinite_loop_on_isolated_rbrace() {
    let code = "}";
    let (ast, diags) = parse_safe(code);
    assert!(!diags.is_empty(), "Expected error diagnostics on isolated '}}'");
    let _ = ast;
}

#[test]
fn test_negative_parser_no_infinite_loop_on_toplevel_fn() {
    let code = "fn test() {}";
    let (ast, diags) = parse_safe(code);
    assert!(!diags.is_empty(), "Expected error diagnostics on top-level fn");
    let _ = ast;
}

#[test]
fn test_negative_parser_no_infinite_loop_on_statement_in_model_body() {
    let code = "model M1 { if (x) {} }";
    let (ast, diags) = parse_safe(code);
    assert!(!diags.is_empty(), "Expected error diagnostics on statement in model body");
    let _ = ast;
}

#[test]
fn test_negative_parser_no_infinite_loop_on_declaration_in_block() {
    let code = "model M1 { scope decode() -> Tensor<[1], u32> { model M2 {} } }";
    let (ast, diags) = parse_safe(code);
    assert!(!diags.is_empty(), "Expected error diagnostics on nested model in scope block");
    let _ = ast;
}

// ============================================================================
// 10. Verification of All 5 Production Reference Models in `examples/`
// ============================================================================

#[test]
fn test_parse_tinystories_smollm() {
    let path = Path::new("examples/tinystories_smollm.vlx");
    let content = fs::read_to_string(path).expect("Failed to read tinystories_smollm.vlx");
    let program = parse_code(&content);
    assert!(program.target_hardware.is_some());
    assert_eq!(program.model_declarations.len(), 1);
    let m = &program.model_declarations[0];
    assert_eq!(m.name, "SmolLM_135M");
    assert!(!m.weights.is_empty());
    assert_eq!(m.scopes.len(), 2);
}

#[test]
fn test_parse_llama3_8b() {
    let path = Path::new("examples/llama3_8b.vlx");
    let content = fs::read_to_string(path).expect("Failed to read llama3_8b.vlx");
    let program = parse_code(&content);
    assert_eq!(program.model_declarations[0].name, "Llama3_8B");
    assert_eq!(program.model_declarations[0].scopes.len(), 2);
}

#[test]
fn test_parse_qwen2_5_7b() {
    let path = Path::new("examples/qwen2_5_7b.vlx");
    let content = fs::read_to_string(path).expect("Failed to read qwen2_5_7b.vlx");
    let program = parse_code(&content);
    assert_eq!(program.model_declarations[0].name, "Qwen2_5_7B");
    assert_eq!(program.model_declarations[0].scopes.len(), 2);
}

#[test]
fn test_parse_deepseek_v3_mla() {
    let path = Path::new("examples/deepseek_v3_mla.vlx");
    let content = fs::read_to_string(path).expect("Failed to read deepseek_v3_mla.vlx");
    let program = parse_code(&content);
    assert_eq!(program.model_declarations[0].name, "DeepSeek_V3_MLA");
    assert_eq!(program.model_declarations[0].scopes.len(), 2);
}

#[test]
fn test_parse_nanbeige_33b() {
    let path = Path::new("examples/nanbeige_33b.vlx");
    let content = fs::read_to_string(path).expect("Failed to read nanbeige_33b.vlx");
    let program = parse_code(&content);
    assert_eq!(program.model_declarations[0].name, "Nanbeige_33B");
    assert_eq!(program.model_declarations[0].scopes.len(), 4);
}
