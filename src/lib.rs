//! Velox Optimizing Compiler & Zero-Copy Serving Runtime Library (`veloxc`)
//!
//! A 100% Native Rust optimizing compiler and zero-copy runtime for the Velox
//! domain-specific systems programming language designed for Large Language Model (LLM)
//! serving and inference.
//!
//! Zero external dependencies (`std` only).

pub mod alloc_counter;
pub mod ast;
pub mod cli;
pub mod codegen;
pub mod diagnostics;
pub mod ir;
pub mod lexer;
pub mod memory_planner;
pub mod optimizer;
pub mod parser;
pub mod runtime;
pub mod runtime_sota;
pub mod sema;
pub mod tokenizer;
pub mod tokens;
pub mod types;

// Re-export core items for ergonomic library usage
pub use alloc_counter::{count_allocations, thread_allocation_count, CountingAllocator};
pub use ast::{
    ASTVisitor, AbsorbMlaNode, AssignmentNode, AsyncDmaTransferNode, ChunkedPrefillScopeNode,
    DecodeScopeNode, ExprNode, LatentKVTypeNode, LetBindingNode, ModelDeclarationNode,
    PagedKVTypeNode, PrefillScopeNode, ProgramNode, SpeculativeTreeTypeNode,
    SpeculativeVerifyScopeNode, StmtNode, StreamingKVTypeNode, TargetHardwareNode, TensorMatmulNode,
    TensorSliceNode, TensorTypeNode, TmaLoadNode, TmaStoreNode, VirtualKVTypeNode,
    WeightDeclarationNode, YieldNode,
};
pub use cli::{
    format_token_stream, run_cli, run_cli_with_io, token_type_to_kind_str, ASTPrettyPrinter,
    CliAction, CliArgs, OutputFormat, EXIT_COMPILATION_ERROR, EXIT_IO_ERROR, EXIT_SUCCESS,
    EXIT_USAGE_ERROR,
};
pub use codegen::{CodeGenerator, CodegenTarget};
pub use diagnostics::{
    Diagnostic, DiagnosticCode, DiagnosticEngine, DiagnosticFormatter, DiagnosticInfo, Severity,
    ALL_CODES,
};
pub use ir::{
    lower_program_to_vir, VirBasicBlock, VirFunction, VirInst, VirModule, VirParam, VirReg,
    VirTerminator,
};
pub use lexer::{LexError, Lexer};
pub use memory_planner::{
    align_up_256, MemoryPlan, PlannedActivationOffset, PlannedKVCachePool, PlannedWeightOffset,
    ScopeArenaPlan, StaticMemoryPlanner, ARENA_ALIGNMENT_BYTES,
};
pub use optimizer::{
    optimize_vir_module, DagMonomorphizationPass, KernelFusionPass, MlaAbsorptionPass,
    OptimizationReport,
};
pub use parser::{binary_binding_power, ParseError, Parser};
pub use runtime::{
    gqa_attention_step_f32, matmul_transposed_f32, rmsnorm_f32, rope_rotate_half_in_place_f32,
    sample_argmax_f32, sample_top_p_f32, swiglu_mlp_residual_f32, MmapSafetensors, SafetensorEntry,
    TransformerConfig, VeloxStaticExecutionEngine,
    StrataFluxConfig, StrataFluxStreamer,
    PersistentKernelScheduler, PersistentWorkItem,
    FlashCommConfig, ReadinessEncoding, FlashCommProtocol, FlashComm, ReduceOp,
    VExpertConfig, VExpert,
    FastAFDConfig, FastAFD,
    FlashAttn3FP8Config, FlashAttn3FP8Launch,
    LoopDuetConfig, LoopDuetScheduler,
    CharmConfig, CharmMerger,
    TieredInferenceConfig, TieredInferenceRouter, InferenceTier,
};
pub use sema::{
    EvaluatedType, FourierMotzkinSolver, SemanticAnalyzer, SymbolicDim, TypedActivation,
    TypedKVCache, TypedModel, TypedProgram, TypedScope, TypedWeight,
};
pub use tokenizer::{build_byte_to_unicode, StreamingDecoder, Tokenizer};
pub use tokens::{Token, TokenType};
pub use types::{
    parse_memory_capacity, AbsorbKind, AssignmentOperator, BinaryOperator, DmaDirection,
    KVCacheKind, MemoryTier, ScalarType, ScopeKind, SourceLocation, SourceSpan, UnaryOperator,
};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Returns the current compiler version.
#[must_use]
pub fn version() -> &'static str {
    VERSION
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert_eq!(version(), "0.1.0");
    }
}
