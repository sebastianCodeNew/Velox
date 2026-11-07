//! Velox Abstract Syntax Tree (`veloxc::ast`)
//!
//! Strongly-typed representation of all 23 core domain AST schemas and supporting nodes
//! formalized in the Velox Master Requirements Matrix and Survey Blueprint.
//!
//! Includes `ASTVisitor` trait and pure `std` JSON serialization helpers.
//! Zero external dependencies (`std` only).

use crate::types::{
    AbsorbKind, AssignmentOperator, BinaryOperator, DmaDirection, MemoryTier, ScalarType,
    ScopeKind, SourceSpan, UnaryOperator,
};

// ============================================================================
// Category 1: Root & Configuration Nodes (Nodes 1 - 4)
// ============================================================================

/// Node 1: Root compilation unit of a `.vlx` file.
/// Contains module imports, target hardware requirements, model definitions, and global bindings.
#[derive(Debug, Clone, PartialEq)]
pub struct ProgramNode {
    pub imports: Vec<ImportNode>,
    pub target_hardware: Option<TargetHardwareNode>,
    pub model_declarations: Vec<ModelDeclarationNode>,
    pub global_bindings: Vec<LetBindingNode>,
    pub span: SourceSpan,
}

impl ProgramNode {
    #[must_use]
    pub fn new(span: SourceSpan) -> Self {
        Self {
            imports: Vec::new(),
            target_hardware: None,
            model_declarations: Vec::new(),
            global_bindings: Vec::new(),
            span,
        }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Module import statement (`import "path/to/module.vlx";`).
#[derive(Debug, Clone, PartialEq)]
pub struct ImportNode {
    pub path: String,
    pub span: SourceSpan,
}

impl ImportNode {
    #[must_use]
    pub fn new(path: impl Into<String>, span: SourceSpan) -> Self {
        Self {
            path: path.into(),
            span,
        }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Node 2: Target execution hardware envelope and VRAM limits.
/// Defines physical device constraints against which compile-time VRAM proofs are evaluated.
#[derive(Debug, Clone, PartialEq)]
pub struct TargetHardwareNode {
    pub device_name: String,
    pub vram_capacity_bytes: u64,
    pub compute_capability: (u32, u32), // e.g. (8, 6) or (9, 0)
    pub safety_factor: f64,             // default 0.95
    pub span: SourceSpan,
}

impl TargetHardwareNode {
    #[must_use]
    pub fn new(
        device_name: impl Into<String>,
        vram_capacity_bytes: u64,
        compute_capability: (u32, u32),
        safety_factor: f64,
        span: SourceSpan,
    ) -> Self {
        Self {
            device_name: device_name.into(),
            vram_capacity_bytes,
            compute_capability,
            safety_factor,
            span,
        }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Node 3: Neural network serving architecture definition.
/// Encapsulates const-generic parameters, weights, KV caches, execution scopes, and helper routines.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelDeclarationNode {
    pub name: String,
    pub generic_params: Vec<ConstGenericParamNode>,
    pub weights: Vec<WeightDeclarationNode>,
    pub kv_caches: Vec<KVCacheDeclarationNode>,
    pub scopes: Vec<ExecutionScopeNode>,
    pub functions: Vec<FunctionDeclarationNode>,
    pub span: SourceSpan,
}

impl ModelDeclarationNode {
    #[must_use]
    pub fn new(name: impl Into<String>, span: SourceSpan) -> Self {
        Self {
            name: name.into(),
            generic_params: Vec::new(),
            weights: Vec::new(),
            kv_caches: Vec::new(),
            scopes: Vec::new(),
            functions: Vec::new(),
            span,
        }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Node 4: Static model parameter tensor bound to a Safetensors file or offline-absorbed projection.
/// Always resides in `@vram_immutable`.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightDeclarationNode {
    pub name: String,
    pub tier: MemoryTier,
    pub tensor_type: Option<TensorTypeNode>,
    pub binding: Option<WeightBinding>,
    pub file_path: Option<String>,
    pub tensor_key: Option<String>,
    pub span: SourceSpan,
}

impl WeightDeclarationNode {
    #[must_use]
    pub fn new(name: impl Into<String>, span: SourceSpan) -> Self {
        Self {
            name: name.into(),
            tier: MemoryTier::VramImmutable,
            tensor_type: None,
            binding: None,
            file_path: None,
            tensor_key: None,
            span,
        }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Weight binding source representation.
#[derive(Debug, Clone, PartialEq)]
pub enum WeightBinding {
    Safetensors(SafetensorsBindingNode),
    AbsorbMla(AbsorbMlaNode),
    Expr(Box<ExprNode>),
}

// ============================================================================
// Category 2: Type & KV-Cache Nodes (Nodes 5 - 10)
// ============================================================================

/// Compile-time dimension parameter (`<const B: usize = 32>`).
#[derive(Debug, Clone, PartialEq)]
pub struct ConstGenericParamNode {
    pub name: String,
    pub type_name: String,
    pub default_val: Option<Box<ExprNode>>,
    pub span: SourceSpan,
}

impl ConstGenericParamNode {
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        type_name: impl Into<String>,
        default_val: Option<Box<ExprNode>>,
        span: SourceSpan,
    ) -> Self {
        Self {
            name: name.into(),
            type_name: type_name.into(),
            default_val,
            span,
        }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Individual tensor dimension (expression or wildcard `*`).
#[derive(Debug, Clone, PartialEq)]
pub struct DimensionNode {
    pub expr: Option<Box<ExprNode>>,
    pub is_wildcard: bool,
    pub span: SourceSpan,
}

impl DimensionNode {
    #[must_use]
    pub fn new_expr(expr: Box<ExprNode>, span: SourceSpan) -> Self {
        Self {
            expr: Some(expr),
            is_wildcard: false,
            span,
        }
    }

    #[must_use]
    pub fn new_wildcard(span: SourceSpan) -> Self {
        Self {
            expr: None,
            is_wildcard: true,
            span,
        }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Block-scale micro-quantization specification (e.g. `block_scale = [128, 128], scale_dtype = fp8_e8m0`).
#[derive(Debug, Clone, PartialEq)]
pub struct BlockQuantSpecNode {
    pub block_shape: Vec<ExprNode>,
    pub scale_dtype: Option<ScalarType>,
    pub span: SourceSpan,
}

impl BlockQuantSpecNode {
    #[must_use]
    pub fn new(block_shape: Vec<ExprNode>, scale_dtype: Option<ScalarType>, span: SourceSpan) -> Self {
        Self {
            block_shape,
            scale_dtype,
            span,
        }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Node 5: Const-generic tensor type with memory tier qualification.
/// Rank must satisfy 1 <= rank <= 6.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorTypeNode {
    pub shape: Vec<DimensionNode>,
    pub dtype: ScalarType,
    pub block_quant_spec: Option<BlockQuantSpecNode>,
    pub tier: MemoryTier,
    pub span: SourceSpan,
}

impl TensorTypeNode {
    #[must_use]
    pub fn new(shape: Vec<DimensionNode>, dtype: ScalarType, tier: MemoryTier, span: SourceSpan) -> Self {
        Self {
            shape,
            dtype,
            block_quant_spec: None,
            tier,
            span,
        }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Node 6: PagedAttention block-chained Key-Value cache type.
/// Block size must be a power-of-two (BS in {16, 32, 64}).
#[derive(Debug, Clone, PartialEq)]
pub struct PagedKVTypeNode {
    pub layers: Box<ExprNode>,
    pub heads: Box<ExprNode>,
    pub head_dim: Box<ExprNode>,
    pub block_size: Box<ExprNode>,
    pub dtype: ScalarType,
    pub tier: MemoryTier,
    pub span: SourceSpan,
}

impl PagedKVTypeNode {
    #[must_use]
    pub fn new(
        layers: Box<ExprNode>,
        heads: Box<ExprNode>,
        head_dim: Box<ExprNode>,
        block_size: Box<ExprNode>,
        dtype: ScalarType,
        tier: MemoryTier,
        span: SourceSpan,
    ) -> Self {
        Self {
            layers,
            heads,
            head_dim,
            block_size,
            dtype,
            tier,
            span,
        }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Node 7: CUDA VMM 2MB Huge-Page virtual-memory-mapped KV cache type.
/// Sequence reservations align with 2MB physical page granularity.
#[derive(Debug, Clone, PartialEq)]
pub struct VirtualKVTypeNode {
    pub layers: Box<ExprNode>,
    pub heads: Box<ExprNode>,
    pub head_dim: Box<ExprNode>,
    pub max_seq_len: Box<ExprNode>,
    pub physical_page_size: Box<ExprNode>,
    pub dtype: ScalarType,
    pub tier: MemoryTier,
    pub span: SourceSpan,
}

impl VirtualKVTypeNode {
    #[must_use]
    pub fn new(
        layers: Box<ExprNode>,
        heads: Box<ExprNode>,
        head_dim: Box<ExprNode>,
        max_seq_len: Box<ExprNode>,
        physical_page_size: Box<ExprNode>,
        dtype: ScalarType,
        tier: MemoryTier,
        span: SourceSpan,
    ) -> Self {
        Self {
            layers,
            heads,
            head_dim,
            max_seq_len,
            physical_page_size,
            dtype,
            tier,
            span,
        }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Node 8: DeepSeek Multi-Head Latent Attention compressed KV cache type.
/// Compresses KV footprint by ~90% via latent rank (`dc`) and decoupled RoPE (`dR`).
#[derive(Debug, Clone, PartialEq)]
pub struct LatentKVTypeNode {
    pub layers: Box<ExprNode>,
    pub latent_dim: Box<ExprNode>,
    pub rope_dim: Box<ExprNode>,
    pub dtype: ScalarType,
    pub tier: MemoryTier,
    pub span: SourceSpan,
}

impl LatentKVTypeNode {
    #[must_use]
    pub fn new(
        layers: Box<ExprNode>,
        latent_dim: Box<ExprNode>,
        rope_dim: Box<ExprNode>,
        dtype: ScalarType,
        tier: MemoryTier,
        span: SourceSpan,
    ) -> Self {
        Self {
            layers,
            latent_dim,
            rope_dim,
            dtype,
            tier,
            span,
        }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Node 9: StreamingLLM attention sink and sliding window cache type.
/// Maintains initial sink tokens (`S >= 1`) and sliding window tokens (`W >= 64`).
#[derive(Debug, Clone, PartialEq)]
pub struct StreamingKVTypeNode {
    pub layers: Box<ExprNode>,
    pub heads: Box<ExprNode>,
    pub head_dim: Box<ExprNode>,
    pub sinks: Box<ExprNode>,
    pub window: Box<ExprNode>,
    pub dtype: ScalarType,
    pub tier: MemoryTier,
    pub span: SourceSpan,
}

impl StreamingKVTypeNode {
    #[must_use]
    pub fn new(
        layers: Box<ExprNode>,
        heads: Box<ExprNode>,
        head_dim: Box<ExprNode>,
        sinks: Box<ExprNode>,
        window: Box<ExprNode>,
        dtype: ScalarType,
        tier: MemoryTier,
        span: SourceSpan,
    ) -> Self {
        Self {
            layers,
            heads,
            head_dim,
            sinks,
            window,
            dtype,
            tier,
            span,
        }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Node 10: SpecInfer / EAGLE speculative verification tree cache type.
/// Bounded candidate verification DAG with static maximum depth. Resides in `@bump_sram`.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeculativeTreeTypeNode {
    pub max_nodes: Box<ExprNode>,
    pub max_depth: Box<ExprNode>,
    pub head_dim: Box<ExprNode>,
    pub dtype: ScalarType,
    pub tier: MemoryTier,
    pub span: SourceSpan,
}

impl SpeculativeTreeTypeNode {
    #[must_use]
    pub fn new(
        max_nodes: Box<ExprNode>,
        max_depth: Box<ExprNode>,
        head_dim: Box<ExprNode>,
        dtype: ScalarType,
        tier: MemoryTier,
        span: SourceSpan,
    ) -> Self {
        Self {
            max_nodes,
            max_depth,
            head_dim,
            dtype,
            tier,
            span,
        }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Full type specification with optional memory tier qualification.
#[derive(Debug, Clone, PartialEq)]
pub struct TypeSpecNode {
    pub tier: Option<MemoryTier>,
    pub kind: TypeKind,
    pub span: SourceSpan,
}

impl TypeSpecNode {
    #[must_use]
    pub fn new(tier: Option<MemoryTier>, kind: TypeKind, span: SourceSpan) -> Self {
        Self { tier, kind, span }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Type classification enum for types in Velox.
#[derive(Debug, Clone, PartialEq)]
pub enum TypeKind {
    Tensor(TensorTypeNode),
    PagedKV(PagedKVTypeNode),
    VirtualKV(VirtualKVTypeNode),
    LatentKV(LatentKVTypeNode),
    StreamingKV(StreamingKVTypeNode),
    SpeculativeTree(SpeculativeTreeTypeNode),
    Primitive(String),
    Custom(String),
    Reference {
        is_mut: bool,
        inner: Box<TypeSpecNode>,
    },
    Tuple(Vec<TypeSpecNode>),
}

// ============================================================================
// Category 3: Execution Scope Nodes (Nodes 11 - 14)
// ============================================================================

/// Node 11: Compute-bound prompt evaluation phase.
/// Synthesizes compute-bound GEMM pipelines; zero dynamic allocation allowed.
#[derive(Debug, Clone, PartialEq)]
pub struct PrefillScopeNode {
    pub name: String,
    pub generic_params: Vec<ConstGenericParamNode>,
    pub parameters: Vec<ParameterNode>,
    pub return_type: Option<TypeSpecNode>,
    pub attributes: Vec<ScopeAttributeNode>,
    pub body: BlockStatementNode,
    pub span: SourceSpan,
}

impl PrefillScopeNode {
    #[must_use]
    pub fn new(name: impl Into<String>, span: SourceSpan) -> Self {
        Self {
            name: name.into(),
            generic_params: Vec::new(),
            parameters: Vec::new(),
            return_type: None,
            attributes: Vec::new(),
            body: BlockStatementNode::new(span.clone()),
            span,
        }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Node 12: Memory-bandwidth-bound single-token generation phase.
/// Bounded to sub-80us host dispatch; synchronous D2H synchronization is fatal.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodeScopeNode {
    pub name: String,
    pub generic_params: Vec<ConstGenericParamNode>,
    pub parameters: Vec<ParameterNode>,
    pub return_type: Option<TypeSpecNode>,
    pub attributes: Vec<ScopeAttributeNode>,
    pub body: BlockStatementNode,
    pub span: SourceSpan,
}

impl DecodeScopeNode {
    #[must_use]
    pub fn new(name: impl Into<String>, span: SourceSpan) -> Self {
        Self {
            name: name.into(),
            generic_params: Vec::new(),
            parameters: Vec::new(),
            return_type: None,
            attributes: Vec::new(),
            body: BlockStatementNode::new(span.clone()),
            span,
        }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Node 13: Interleaved chunked prefill serving scope.
/// Interleaves uniform prefill chunks with decode steps to eliminate pipeline bubbles.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkedPrefillScopeNode {
    pub name: String,
    pub generic_params: Vec<ConstGenericParamNode>,
    pub parameters: Vec<ParameterNode>,
    pub return_type: Option<TypeSpecNode>,
    pub attributes: Vec<ScopeAttributeNode>,
    pub body: BlockStatementNode,
    pub chunk_size: Option<Box<ExprNode>>,
    pub max_decode_batch: Option<Box<ExprNode>>,
    pub span: SourceSpan,
}

impl ChunkedPrefillScopeNode {
    #[must_use]
    pub fn new(name: impl Into<String>, span: SourceSpan) -> Self {
        Self {
            name: name.into(),
            generic_params: Vec::new(),
            parameters: Vec::new(),
            return_type: None,
            attributes: Vec::new(),
            body: BlockStatementNode::new(span.clone()),
            chunk_size: None,
            max_decode_batch: None,
            span,
        }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Node 14: Speculative tree verification serving scope.
/// Validates draft model candidate tokens in parallel with non-causal attention masks.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeculativeVerifyScopeNode {
    pub name: String,
    pub generic_params: Vec<ConstGenericParamNode>,
    pub parameters: Vec<ParameterNode>,
    pub return_type: Option<TypeSpecNode>,
    pub attributes: Vec<ScopeAttributeNode>,
    pub body: BlockStatementNode,
    pub tree_type: Option<SpeculativeTreeTypeNode>,
    pub target_model_ref: String,
    pub span: SourceSpan,
}

impl SpeculativeVerifyScopeNode {
    #[must_use]
    pub fn new(name: impl Into<String>, span: SourceSpan) -> Self {
        Self {
            name: name.into(),
            generic_params: Vec::new(),
            parameters: Vec::new(),
            return_type: None,
            attributes: Vec::new(),
            body: BlockStatementNode::new(span.clone()),
            tree_type: None,
            target_model_ref: String::new(),
            span,
        }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// General execution scope container enum.
#[derive(Debug, Clone, PartialEq)]
pub enum ExecutionScopeNode {
    Prefill(PrefillScopeNode),
    Decode(DecodeScopeNode),
    ChunkedPrefill(ChunkedPrefillScopeNode),
    SpeculativeVerify(SpeculativeVerifyScopeNode),
}

impl ExecutionScopeNode {
    #[must_use]
    pub fn kind(&self) -> ScopeKind {
        match self {
            Self::Prefill(_) => ScopeKind::Prefill,
            Self::Decode(_) => ScopeKind::Decode,
            Self::ChunkedPrefill(_) => ScopeKind::ChunkedPrefill,
            Self::SpeculativeVerify(_) => ScopeKind::SpeculativeVerify,
        }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        match self {
            Self::Prefill(n) => n.span(),
            Self::Decode(n) => n.span(),
            Self::ChunkedPrefill(n) => n.span(),
            Self::SpeculativeVerify(n) => n.span(),
        }
    }
}

// ============================================================================
// Category 4: Statement & Memory Nodes (Nodes 15 - 20)
// ============================================================================

/// Node 15: Variable or local activation binding statement (`let [mut] x: Type = expr;` or tuple pattern).
#[derive(Debug, Clone, PartialEq)]
pub struct LetBindingNode {
    pub name: String,
    pub is_mut: bool,
    pub type_annot: Option<TypeSpecNode>,
    pub init_expr: Option<Box<ExprNode>>,
    pub tuple_pattern: Option<Vec<String>>,
    pub span: SourceSpan,
}

impl LetBindingNode {
    #[must_use]
    pub fn new(name: impl Into<String>, is_mut: bool, span: SourceSpan) -> Self {
        Self {
            name: name.into(),
            is_mut,
            type_annot: None,
            init_expr: None,
            tuple_pattern: None,
            span,
        }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Node 16: In-place mutation assignment statement (`lhs = rhs`, `lhs += rhs`, etc.).
#[derive(Debug, Clone, PartialEq)]
pub struct AssignmentNode {
    pub lhs: Box<ExprNode>,
    pub op: AssignmentOperator,
    pub rhs: Box<ExprNode>,
    pub span: SourceSpan,
}

impl AssignmentNode {
    #[must_use]
    pub fn new(lhs: Box<ExprNode>, op: AssignmentOperator, rhs: Box<ExprNode>, span: SourceSpan) -> Self {
        Self { lhs, op, rhs, span }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Node 17: Asynchronous DMA transfer statement between host and device tiers.
#[derive(Debug, Clone, PartialEq)]
pub struct AsyncDmaTransferNode {
    pub direction: DmaDirection,
    pub source: Box<ExprNode>,
    pub destination: Box<ExprNode>,
    pub signal: Option<Box<ExprNode>>,
    pub span: SourceSpan,
}

impl AsyncDmaTransferNode {
    #[must_use]
    pub fn new(
        direction: DmaDirection,
        source: Box<ExprNode>,
        destination: Box<ExprNode>,
        signal: Option<Box<ExprNode>>,
        span: SourceSpan,
    ) -> Self {
        Self {
            direction,
            source,
            destination,
            signal,
            span,
        }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Node 18: Hardware TMA asynchronous load from global VRAM to Shared Memory.
/// Emits Hopper/Blackwell `cp.async.bulk.tensor` instructions with transaction barrier.
#[derive(Debug, Clone, PartialEq)]
pub struct TmaLoadNode {
    pub source: Box<ExprNode>,
    pub destination: Box<ExprNode>,
    pub mbarrier: Box<ExprNode>,
    pub span: SourceSpan,
}

impl TmaLoadNode {
    #[must_use]
    pub fn new(
        source: Box<ExprNode>,
        destination: Box<ExprNode>,
        mbarrier: Box<ExprNode>,
        span: SourceSpan,
    ) -> Self {
        Self {
            source,
            destination,
            mbarrier,
            span,
        }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Node 19: Hardware TMA asynchronous store from Shared Memory to global VRAM.
#[derive(Debug, Clone, PartialEq)]
pub struct TmaStoreNode {
    pub source: Box<ExprNode>,
    pub destination: Box<ExprNode>,
    pub span: SourceSpan,
}

impl TmaStoreNode {
    #[must_use]
    pub fn new(source: Box<ExprNode>, destination: Box<ExprNode>, span: SourceSpan) -> Self {
        Self {
            source,
            destination,
            span,
        }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Node 20: Cooperative non-preemptive fiber yield statement.
/// Performs 8-nanosecond user-space register context switch to serving event loop.
#[derive(Debug, Clone, PartialEq)]
pub struct YieldNode {
    pub value: Option<Box<ExprNode>>,
    pub span: SourceSpan,
}

impl YieldNode {
    #[must_use]
    pub fn new(value: Option<Box<ExprNode>>, span: SourceSpan) -> Self {
        Self { value, span }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

// ============================================================================
// Category 5: Domain-Specific Expression Nodes (Nodes 21 - 23)
// ============================================================================

/// Node 21: Dedicated Tensor Core matrix multiplication operator (`@`).
/// Compile-time invariant: Inner contraction dimensions must unify.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorMatmulNode {
    pub lhs: Box<ExprNode>,
    pub rhs: Box<ExprNode>,
    pub span: SourceSpan,
}

impl TensorMatmulNode {
    #[must_use]
    pub fn new(lhs: Box<ExprNode>, rhs: Box<ExprNode>, span: SourceSpan) -> Self {
        Self { lhs, rhs, span }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Node 22: Multi-Head Latent Attention offline weight absorption operator.
/// Folds projection matrices offline at build time to slash KV cache footprint.
#[derive(Debug, Clone, PartialEq)]
pub struct AbsorbMlaNode {
    pub kind: AbsorbKind,
    pub projection_q: Box<ExprNode>,
    pub projection_k: Box<ExprNode>,
    pub span: SourceSpan,
}

impl AbsorbMlaNode {
    #[must_use]
    pub fn new(
        kind: AbsorbKind,
        projection_q: Box<ExprNode>,
        projection_k: Box<ExprNode>,
        span: SourceSpan,
    ) -> Self {
        Self {
            kind,
            projection_q,
            projection_k,
            span,
        }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Node 23: Multi-dimensional tensor slicing and sub-view borrow.
/// Extracts zero-copy slice views without dynamic memory allocation.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorSliceNode {
    pub tensor_expr: Box<ExprNode>,
    pub ranges: Vec<SliceRangeNode>,
    pub span: SourceSpan,
}

impl TensorSliceNode {
    #[must_use]
    pub fn new(tensor_expr: Box<ExprNode>, ranges: Vec<SliceRangeNode>, span: SourceSpan) -> Self {
        Self {
            tensor_expr,
            ranges,
            span,
        }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

// ============================================================================
// Supporting Statements, Expressions, & Navigation Nodes
// ============================================================================

/// Function or scope parameter.
#[derive(Debug, Clone, PartialEq)]
pub struct ParameterNode {
    pub name: String,
    pub is_mut: bool,
    pub type_spec: Option<TypeSpecNode>,
    pub default_val: Option<Box<ExprNode>>,
    pub span: SourceSpan,
}

impl ParameterNode {
    #[must_use]
    pub fn new(name: impl Into<String>, is_mut: bool, span: SourceSpan) -> Self {
        Self {
            name: name.into(),
            is_mut,
            type_spec: None,
            default_val: None,
            span,
        }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Internal helper function declaration.
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDeclarationNode {
    pub name: String,
    pub generic_params: Vec<ConstGenericParamNode>,
    pub parameters: Vec<ParameterNode>,
    pub return_type: Option<TypeSpecNode>,
    pub body: BlockStatementNode,
    pub span: SourceSpan,
}

impl FunctionDeclarationNode {
    #[must_use]
    pub fn new(name: impl Into<String>, span: SourceSpan) -> Self {
        Self {
            name: name.into(),
            generic_params: Vec::new(),
            parameters: Vec::new(),
            return_type: None,
            body: BlockStatementNode::new(span.clone()),
            span,
        }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// KV cache state allocation inside a model (`let [mut] kv_name: KVCacheType;`).
#[derive(Debug, Clone, PartialEq)]
pub struct KVCacheDeclarationNode {
    pub name: String,
    pub is_mut: bool,
    pub cache_type: Option<TypeSpecNode>,
    pub span: SourceSpan,
}

impl KVCacheDeclarationNode {
    #[must_use]
    pub fn new(name: impl Into<String>, is_mut: bool, span: SourceSpan) -> Self {
        Self {
            name: name.into(),
            is_mut,
            cache_type: None,
            span,
        }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Safetensors weight loader binding source (`safetensors("path", "key")`).
#[derive(Debug, Clone, PartialEq)]
pub struct SafetensorsBindingNode {
    pub file_path: String,
    pub tensor_key: String,
    pub span: SourceSpan,
}

impl SafetensorsBindingNode {
    #[must_use]
    pub fn new(file_path: impl Into<String>, tensor_key: impl Into<String>, span: SourceSpan) -> Self {
        Self {
            file_path: file_path.into(),
            tensor_key: tensor_key.into(),
            span,
        }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Optimization attribute attached to execution scopes (`@kernel_fuse`, `@warp_specialized`).
#[derive(Debug, Clone, PartialEq)]
pub struct ScopeAttributeNode {
    pub name: String,
    pub params: Vec<(String, ExprNode)>,
    pub span: SourceSpan,
}

impl ScopeAttributeNode {
    #[must_use]
    pub fn new(name: impl Into<String>, span: SourceSpan) -> Self {
        Self {
            name: name.into(),
            params: Vec::new(),
            span,
        }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Return statement (`return [expr];`).
#[derive(Debug, Clone, PartialEq)]
pub struct ReturnNode {
    pub value: Option<Box<ExprNode>>,
    pub span: SourceSpan,
}

impl ReturnNode {
    #[must_use]
    pub fn new(value: Option<Box<ExprNode>>, span: SourceSpan) -> Self {
        Self { value, span }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Conditional if/else statement.
#[derive(Debug, Clone, PartialEq)]
pub struct IfNode {
    pub condition: Box<ExprNode>,
    pub then_branch: BlockStatementNode,
    pub else_branch: Option<Box<StmtNode>>,
    pub span: SourceSpan,
}

impl IfNode {
    #[must_use]
    pub fn new(
        condition: Box<ExprNode>,
        then_branch: BlockStatementNode,
        else_branch: Option<Box<StmtNode>>,
        span: SourceSpan,
    ) -> Self {
        Self {
            condition,
            then_branch,
            else_branch,
            span,
        }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Bounded range iteration loop (`for i in start..end { ... }`).
#[derive(Debug, Clone, PartialEq)]
pub struct ForNode {
    pub var_name: String,
    pub start: Box<ExprNode>,
    pub end: Box<ExprNode>,
    pub body: BlockStatementNode,
    pub span: SourceSpan,
}

impl ForNode {
    #[must_use]
    pub fn new(
        var_name: impl Into<String>,
        start: Box<ExprNode>,
        end: Box<ExprNode>,
        body: BlockStatementNode,
        span: SourceSpan,
    ) -> Self {
        Self {
            var_name: var_name.into(),
            start,
            end,
            body,
            span,
        }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// While loop statement (`while (cond) { ... }`).
#[derive(Debug, Clone, PartialEq)]
pub struct WhileNode {
    pub condition: Box<ExprNode>,
    pub body: BlockStatementNode,
    pub span: SourceSpan,
}

impl WhileNode {
    #[must_use]
    pub fn new(condition: Box<ExprNode>, body: BlockStatementNode, span: SourceSpan) -> Self {
        Self {
            condition,
            body,
            span,
        }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Compound block statement containing sequential statements (`{ ... }`).
#[derive(Debug, Clone, PartialEq)]
pub struct BlockStatementNode {
    pub statements: Vec<StmtNode>,
    pub span: SourceSpan,
}

impl BlockStatementNode {
    #[must_use]
    pub fn new(span: SourceSpan) -> Self {
        Self {
            statements: Vec::new(),
            span,
        }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Evaluation of an expression as a statement (`expr;`).
#[derive(Debug, Clone, PartialEq)]
pub struct ExpressionStmtNode {
    pub expr: Box<ExprNode>,
    pub span: SourceSpan,
}

impl ExpressionStmtNode {
    #[must_use]
    pub fn new(expr: Box<ExprNode>, span: SourceSpan) -> Self {
        Self { expr, span }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Imperative statement union enum.
#[derive(Debug, Clone, PartialEq)]
pub enum StmtNode {
    Let(LetBindingNode),
    Assignment(AssignmentNode),
    AsyncDmaTransfer(AsyncDmaTransferNode),
    TmaLoad(TmaLoadNode),
    TmaStore(TmaStoreNode),
    Yield(YieldNode),
    Return(ReturnNode),
    If(IfNode),
    For(ForNode),
    While(WhileNode),
    Block(BlockStatementNode),
    Expression(ExpressionStmtNode),
}

impl StmtNode {
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        match self {
            Self::Let(n) => n.span(),
            Self::Assignment(n) => n.span(),
            Self::AsyncDmaTransfer(n) => n.span(),
            Self::TmaLoad(n) => n.span(),
            Self::TmaStore(n) => n.span(),
            Self::Yield(n) => n.span(),
            Self::Return(n) => n.span(),
            Self::If(n) => n.span(),
            Self::For(n) => n.span(),
            Self::While(n) => n.span(),
            Self::Block(n) => n.span(),
            Self::Expression(n) => n.span(),
        }
    }
}

/// Binary operation expression (`left op right`).
#[derive(Debug, Clone, PartialEq)]
pub struct BinaryOpNode {
    pub left: Box<ExprNode>,
    pub op: BinaryOperator,
    pub right: Box<ExprNode>,
    pub span: SourceSpan,
}

impl BinaryOpNode {
    #[must_use]
    pub fn new(left: Box<ExprNode>, op: BinaryOperator, right: Box<ExprNode>, span: SourceSpan) -> Self {
        Self { left, op, right, span }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Unary prefix operation expression (`op operand`).
#[derive(Debug, Clone, PartialEq)]
pub struct UnaryOpNode {
    pub op: UnaryOperator,
    pub operand: Box<ExprNode>,
    pub span: SourceSpan,
}

impl UnaryOpNode {
    #[must_use]
    pub fn new(op: UnaryOperator, operand: Box<ExprNode>, span: SourceSpan) -> Self {
        Self { op, operand, span }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Function, kernel, or constructor call (`callee(args)`).
#[derive(Debug, Clone, PartialEq)]
pub struct CallNode {
    pub callee: Box<ExprNode>,
    pub args: Vec<ArgumentNode>,
    pub span: SourceSpan,
}

impl CallNode {
    #[must_use]
    pub fn new(callee: Box<ExprNode>, args: Vec<ArgumentNode>, span: SourceSpan) -> Self {
        Self { callee, args, span }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Call argument with optional parameter keyword name (`[name = ] value`).
#[derive(Debug, Clone, PartialEq)]
pub struct ArgumentNode {
    pub name: Option<String>,
    pub value: Box<ExprNode>,
    pub span: SourceSpan,
}

impl ArgumentNode {
    #[must_use]
    pub fn new(name: Option<String>, value: Box<ExprNode>, span: SourceSpan) -> Self {
        Self { name, value, span }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Struct field or tensor property access (`obj.member`).
#[derive(Debug, Clone, PartialEq)]
pub struct MemberAccessNode {
    pub obj: Box<ExprNode>,
    pub member: String,
    pub span: SourceSpan,
}

impl MemberAccessNode {
    #[must_use]
    pub fn new(obj: Box<ExprNode>, member: impl Into<String>, span: SourceSpan) -> Self {
        Self {
            obj,
            member: member.into(),
            span,
        }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Array or tensor coordinate indexing (`obj[index]`).
#[derive(Debug, Clone, PartialEq)]
pub struct IndexAccessNode {
    pub obj: Box<ExprNode>,
    pub index: Box<ExprNode>,
    pub span: SourceSpan,
}

impl IndexAccessNode {
    #[must_use]
    pub fn new(obj: Box<ExprNode>, index: Box<ExprNode>, span: SourceSpan) -> Self {
        Self { obj, index, span }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Coordinate range for tensor slicing (`start..end`, `start..`, `..end`, `..`, or `:`).
#[derive(Debug, Clone, PartialEq)]
pub struct SliceRangeNode {
    pub start: Option<Box<ExprNode>>,
    pub end: Option<Box<ExprNode>>,
    pub is_full_slice: bool,
    pub span: SourceSpan,
}

impl SliceRangeNode {
    #[must_use]
    pub fn new(
        start: Option<Box<ExprNode>>,
        end: Option<Box<ExprNode>>,
        is_full_slice: bool,
        span: SourceSpan,
    ) -> Self {
        Self {
            start,
            end,
            is_full_slice,
            span,
        }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Named variable or tensor identifier reference.
#[derive(Debug, Clone, PartialEq)]
pub struct IdentifierNode {
    pub name: String,
    pub span: SourceSpan,
}

impl IdentifierNode {
    #[must_use]
    pub fn new(name: impl Into<String>, span: SourceSpan) -> Self {
        Self {
            name: name.into(),
            span,
        }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Evaluated literal constant values.
#[derive(Debug, Clone, PartialEq)]
pub enum LiteralValue {
    Decimal(u64),
    Hex(u64),
    Binary(u64),
    Float(f64),
    String(String),
    Bool(bool),
}

/// Literal constant value AST node.
#[derive(Debug, Clone, PartialEq)]
pub struct LiteralNode {
    pub value: LiteralValue,
    pub raw_text: String,
    pub span: SourceSpan,
}

impl LiteralNode {
    #[must_use]
    pub fn new(value: LiteralValue, raw_text: impl Into<String>, span: SourceSpan) -> Self {
        Self {
            value,
            raw_text: raw_text.into(),
            span,
        }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Tuple expression (`(a, b, ...)`).
#[derive(Debug, Clone, PartialEq)]
pub struct TupleNode {
    pub elements: Vec<ExprNode>,
    pub span: SourceSpan,
}

impl TupleNode {
    #[must_use]
    pub fn new(elements: Vec<ExprNode>, span: SourceSpan) -> Self {
        Self { elements, span }
    }

    #[inline]
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        &self.span
    }
}

/// Value-yielding expression union enum.
#[derive(Debug, Clone, PartialEq)]
pub enum ExprNode {
    TensorMatmul(TensorMatmulNode),
    AbsorbMla(AbsorbMlaNode),
    TensorSlice(TensorSliceNode),
    Binary(BinaryOpNode),
    Unary(UnaryOpNode),
    Call(CallNode),
    MemberAccess(MemberAccessNode),
    IndexAccess(IndexAccessNode),
    Identifier(IdentifierNode),
    Literal(LiteralNode),
    Tuple(TupleNode),
    Parenthesized(Box<ExprNode>),
}

impl ExprNode {
    #[must_use]
    pub fn span(&self) -> &SourceSpan {
        match self {
            Self::TensorMatmul(n) => n.span(),
            Self::AbsorbMla(n) => n.span(),
            Self::TensorSlice(n) => n.span(),
            Self::Binary(n) => n.span(),
            Self::Unary(n) => n.span(),
            Self::Call(n) => n.span(),
            Self::MemberAccess(n) => n.span(),
            Self::IndexAccess(n) => n.span(),
            Self::Identifier(n) => n.span(),
            Self::Literal(n) => n.span(),
            Self::Tuple(n) => n.span(),
            Self::Parenthesized(n) => n.span(),
        }
    }
}

// ============================================================================
// Visitor Infrastructure
// ============================================================================

/// Trait for double-dispatch traversal over Velox AST hierarchies.
pub trait ASTVisitor {
    type Output;

    fn visit_program(&mut self, node: &ProgramNode) -> Self::Output;
    fn visit_target_hardware(&mut self, node: &TargetHardwareNode) -> Self::Output;
    fn visit_model_declaration(&mut self, node: &ModelDeclarationNode) -> Self::Output;
    fn visit_weight_declaration(&mut self, node: &WeightDeclarationNode) -> Self::Output;
    fn visit_tensor_type(&mut self, node: &TensorTypeNode) -> Self::Output;
    fn visit_paged_kv_type(&mut self, node: &PagedKVTypeNode) -> Self::Output;
    fn visit_virtual_kv_type(&mut self, node: &VirtualKVTypeNode) -> Self::Output;
    fn visit_latent_kv_type(&mut self, node: &LatentKVTypeNode) -> Self::Output;
    fn visit_streaming_kv_type(&mut self, node: &StreamingKVTypeNode) -> Self::Output;
    fn visit_speculative_tree_type(&mut self, node: &SpeculativeTreeTypeNode) -> Self::Output;
    fn visit_prefill_scope(&mut self, node: &PrefillScopeNode) -> Self::Output;
    fn visit_decode_scope(&mut self, node: &DecodeScopeNode) -> Self::Output;
    fn visit_chunked_prefill_scope(&mut self, node: &ChunkedPrefillScopeNode) -> Self::Output;
    fn visit_speculative_verify_scope(&mut self, node: &SpeculativeVerifyScopeNode) -> Self::Output;
    fn visit_let_binding(&mut self, node: &LetBindingNode) -> Self::Output;
    fn visit_assignment(&mut self, node: &AssignmentNode) -> Self::Output;
    fn visit_async_dma_transfer(&mut self, node: &AsyncDmaTransferNode) -> Self::Output;
    fn visit_tma_load(&mut self, node: &TmaLoadNode) -> Self::Output;
    fn visit_tma_store(&mut self, node: &TmaStoreNode) -> Self::Output;
    fn visit_yield(&mut self, node: &YieldNode) -> Self::Output;
    fn visit_tensor_matmul(&mut self, node: &TensorMatmulNode) -> Self::Output;
    fn visit_absorb_mla(&mut self, node: &AbsorbMlaNode) -> Self::Output;
    fn visit_tensor_slice(&mut self, node: &TensorSliceNode) -> Self::Output;
    fn visit_stmt(&mut self, node: &StmtNode) -> Self::Output;
    fn visit_expr(&mut self, node: &ExprNode) -> Self::Output;
}

// ============================================================================
// Pure-Rust JSON Serialization Helper (Zero External Crates)
// ============================================================================

/// JSON value model for zero-dependency AST serialization.
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    Integer(i64),
    Unsigned(u64),
    String(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

impl JsonValue {
    /// Serializes value to compact JSON string.
    #[must_use]
    pub fn to_compact_string(&self) -> String {
        let mut out = String::new();
        self.write_compact(&mut out);
        out
    }

    /// Serializes value to formatted indented JSON string.
    #[must_use]
    pub fn to_pretty_string(&self, indent_size: usize) -> String {
        let mut out = String::new();
        self.write_pretty(&mut out, 0, indent_size);
        out
    }

    fn write_compact(&self, out: &mut String) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Self::Number(n) => {
                if n.is_finite() {
                    out.push_str(&n.to_string());
                } else {
                    out.push_str("null");
                }
            }
            Self::Integer(i) => out.push_str(&i.to_string()),
            Self::Unsigned(u) => out.push_str(&u.to_string()),
            Self::String(s) => escape_json_string(s, out),
            Self::Array(items) => {
                out.push('[');
                for (idx, item) in items.iter().enumerate() {
                    if idx > 0 {
                        out.push(',');
                    }
                    item.write_compact(out);
                }
                out.push(']');
            }
            Self::Object(entries) => {
                out.push('{');
                for (idx, (k, v)) in entries.iter().enumerate() {
                    if idx > 0 {
                        out.push(',');
                    }
                    escape_json_string(k, out);
                    out.push(':');
                    v.write_compact(out);
                }
                out.push('}');
            }
        }
    }

    fn write_pretty(&self, out: &mut String, depth: usize, indent_size: usize) {
        let indent = " ".repeat(depth * indent_size);
        let next_indent = " ".repeat((depth + 1) * indent_size);

        match self {
            Self::Null | Self::Bool(_) | Self::Number(_) | Self::Integer(_) | Self::Unsigned(_) | Self::String(_) => {
                self.write_compact(out);
            }
            Self::Array(items) => {
                if items.is_empty() {
                    out.push_str("[]");
                    return;
                }
                out.push_str("[\n");
                for (idx, item) in items.iter().enumerate() {
                    if idx > 0 {
                        out.push_str(",\n");
                    }
                    out.push_str(&next_indent);
                    item.write_pretty(out, depth + 1, indent_size);
                }
                out.push('\n');
                out.push_str(&indent);
                out.push(']');
            }
            Self::Object(entries) => {
                if entries.is_empty() {
                    out.push_str("{}");
                    return;
                }
                out.push_str("{\n");
                for (idx, (k, v)) in entries.iter().enumerate() {
                    if idx > 0 {
                        out.push_str(",\n");
                    }
                    out.push_str(&next_indent);
                    escape_json_string(k, out);
                    out.push_str(": ");
                    v.write_pretty(out, depth + 1, indent_size);
                }
                out.push('\n');
                out.push_str(&indent);
                out.push('}');
            }
        }
    }
}

fn escape_json_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\x08' => out.push_str("\\b"),
            '\x0C' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn span_to_json(span: &SourceSpan) -> JsonValue {
    JsonValue::Object(vec![
        ("file".to_string(), JsonValue::String(span.file.clone())),
        (
            "start".to_string(),
            JsonValue::Object(vec![
                ("line".to_string(), JsonValue::Unsigned(span.start.line as u64)),
                ("col".to_string(), JsonValue::Unsigned(span.start.column as u64)),
                ("offset".to_string(), JsonValue::Unsigned(span.start.offset as u64)),
            ]),
        ),
        (
            "end".to_string(),
            JsonValue::Object(vec![
                ("line".to_string(), JsonValue::Unsigned(span.end.line as u64)),
                ("col".to_string(), JsonValue::Unsigned(span.end.column as u64)),
                ("offset".to_string(), JsonValue::Unsigned(span.end.offset as u64)),
            ]),
        ),
    ])
}

fn expr_to_json(expr: &ExprNode) -> JsonValue {
    match expr {
        ExprNode::Literal(lit) => match lit.value {
            LiteralValue::Decimal(v) => JsonValue::Unsigned(v),
            LiteralValue::Hex(v) => JsonValue::Unsigned(v),
            LiteralValue::Binary(v) => JsonValue::Unsigned(v),
            LiteralValue::Float(f) => JsonValue::Number(f),
            LiteralValue::String(ref s) => JsonValue::String(s.clone()),
            LiteralValue::Bool(b) => JsonValue::Bool(b),
        },
        ExprNode::Identifier(id) => JsonValue::String(id.name.clone()),
        ExprNode::Binary(b) => JsonValue::Object(vec![
            ("lhs".to_string(), expr_to_json(&b.left)),
            ("op".to_string(), JsonValue::String(b.op.as_str().to_string())),
            ("rhs".to_string(), expr_to_json(&b.right)),
        ]),
        ExprNode::Unary(u) => JsonValue::Object(vec![
            ("op".to_string(), JsonValue::String(u.op.as_str().to_string())),
            ("operand".to_string(), expr_to_json(&u.operand)),
        ]),
        ExprNode::TensorMatmul(m) => JsonValue::Object(vec![
            ("lhs".to_string(), expr_to_json(&m.lhs)),
            ("op".to_string(), JsonValue::String("@".to_string())),
            ("rhs".to_string(), expr_to_json(&m.rhs)),
        ]),
        ExprNode::MemberAccess(ma) => JsonValue::Object(vec![
            ("object".to_string(), expr_to_json(&ma.obj)),
            ("member".to_string(), JsonValue::String(ma.member.clone())),
        ]),
        ExprNode::Call(c) => JsonValue::Object(vec![
            ("callee".to_string(), expr_to_json(&c.callee)),
            (
                "arguments".to_string(),
                JsonValue::Array(
                    c.args
                        .iter()
                        .map(|a| {
                            if let Some(ref name) = a.name {
                                JsonValue::Object(vec![
                                    ("name".to_string(), JsonValue::String(name.clone())),
                                    ("value".to_string(), expr_to_json(&a.value)),
                                ])
                            } else {
                                expr_to_json(&a.value)
                            }
                        })
                        .collect(),
                ),
            ),
        ]),
        _ => JsonValue::String(format!("{expr:?}")),
    }
}

fn const_generic_to_json(cg: &ConstGenericParamNode) -> JsonValue {
    let mut entries = vec![
        ("node_type".to_string(), JsonValue::String("ConstGenericParamNode".to_string())),
        ("name".to_string(), JsonValue::String(cg.name.clone())),
        ("type_name".to_string(), JsonValue::String(cg.type_name.clone())),
    ];
    if let Some(ref def) = cg.default_val {
        entries.push(("default_value".to_string(), expr_to_json(def)));
    }
    entries.push(("span".to_string(), span_to_json(&cg.span)));
    JsonValue::Object(entries)
}

fn tensor_type_to_json(tt: &TensorTypeNode) -> JsonValue {
    let shape_arr = tt
        .shape
        .iter()
        .map(|d| {
            if d.is_wildcard {
                JsonValue::String("*".to_string())
            } else if let Some(ref expr) = d.expr {
                expr_to_json(expr)
            } else {
                JsonValue::Null
            }
        })
        .collect();

    JsonValue::Object(vec![
        ("node_type".to_string(), JsonValue::String("TensorTypeNode".to_string())),
        ("shape".to_string(), JsonValue::Array(shape_arr)),
        ("dtype".to_string(), JsonValue::String(tt.dtype.as_str().to_string())),
        ("memory_tier".to_string(), JsonValue::String(tt.tier.as_str().to_string())),
        ("span".to_string(), span_to_json(&tt.span)),
    ])
}

fn weight_to_json(w: &WeightDeclarationNode) -> JsonValue {
    let binding_val = match &w.binding {
        Some(WeightBinding::Safetensors(s)) => JsonValue::Object(vec![
            ("kind".to_string(), JsonValue::String("safetensors".to_string())),
            ("file_path".to_string(), JsonValue::String(s.file_path.clone())),
            ("tensor_key".to_string(), JsonValue::String(s.tensor_key.clone())),
            ("span".to_string(), span_to_json(&s.span)),
        ]),
        Some(WeightBinding::AbsorbMla(a)) => JsonValue::Object(vec![
            ("kind".to_string(), JsonValue::String(match a.kind {
                AbsorbKind::AbsorbMlaQ => "absorb_mla_q".to_string(),
                AbsorbKind::AbsorbMlaV => "absorb_mla_v".to_string(),
            })),
            ("projection_q".to_string(), expr_to_json(&a.projection_q)),
            ("projection_k".to_string(), expr_to_json(&a.projection_k)),
            ("span".to_string(), span_to_json(&a.span)),
        ]),
        Some(WeightBinding::Expr(e)) => JsonValue::Object(vec![
            ("kind".to_string(), JsonValue::String("expr".to_string())),
            ("value".to_string(), expr_to_json(e)),
            ("span".to_string(), span_to_json(e.span())),
        ]),
        None => JsonValue::Null,
    };

    let tensor_type_val = w.tensor_type.as_ref().map_or(JsonValue::Null, tensor_type_to_json);

    JsonValue::Object(vec![
        ("node_type".to_string(), JsonValue::String("WeightDeclarationNode".to_string())),
        ("name".to_string(), JsonValue::String(w.name.clone())),
        ("memory_tier".to_string(), JsonValue::String(w.tier.as_str().to_string())),
        ("tensor_type".to_string(), tensor_type_val),
        ("binding".to_string(), binding_val),
        ("span".to_string(), span_to_json(&w.span)),
    ])
}

fn kv_cache_to_json(kv: &KVCacheDeclarationNode) -> JsonValue {
    let (cache_type_str, params_val, dtype_val, tier_val) = match &kv.cache_type {
        Some(spec) => {
            let tier = spec.tier.map_or_else(
                || match &spec.kind {
                    TypeKind::PagedKV(p) => p.tier.as_str().to_string(),
                    TypeKind::VirtualKV(v) => v.tier.as_str().to_string(),
                    TypeKind::LatentKV(l) => l.tier.as_str().to_string(),
                    TypeKind::StreamingKV(s) => s.tier.as_str().to_string(),
                    TypeKind::SpeculativeTree(st) => st.tier.as_str().to_string(),
                    _ => "@paged_vram".to_string(),
                },
                |t| t.as_str().to_string(),
            );
            match &spec.kind {
                TypeKind::PagedKV(p) => (
                    "PagedKVCache".to_string(),
                    JsonValue::Object(vec![
                        ("layers".to_string(), expr_to_json(&p.layers)),
                        ("heads".to_string(), expr_to_json(&p.heads)),
                        ("head_dim".to_string(), expr_to_json(&p.head_dim)),
                        ("block_size".to_string(), expr_to_json(&p.block_size)),
                    ]),
                    JsonValue::String(p.dtype.as_str().to_string()),
                    JsonValue::String(tier),
                ),
                TypeKind::VirtualKV(v) => (
                    "VirtualContiguousKV".to_string(),
                    JsonValue::Object(vec![
                        ("layers".to_string(), expr_to_json(&v.layers)),
                        ("heads".to_string(), expr_to_json(&v.heads)),
                        ("head_dim".to_string(), expr_to_json(&v.head_dim)),
                        ("max_seq_len".to_string(), expr_to_json(&v.max_seq_len)),
                        ("physical_page_size".to_string(), expr_to_json(&v.physical_page_size)),
                    ]),
                    JsonValue::String(v.dtype.as_str().to_string()),
                    JsonValue::String(tier),
                ),
                TypeKind::LatentKV(l) => (
                    "LatentKVCache".to_string(),
                    JsonValue::Object(vec![
                        ("layers".to_string(), expr_to_json(&l.layers)),
                        ("latent_dim".to_string(), expr_to_json(&l.latent_dim)),
                        ("rope_dim".to_string(), expr_to_json(&l.rope_dim)),
                    ]),
                    JsonValue::String(l.dtype.as_str().to_string()),
                    JsonValue::String(tier),
                ),
                TypeKind::StreamingKV(s) => (
                    "StreamingSinkKVCache".to_string(),
                    JsonValue::Object(vec![
                        ("layers".to_string(), expr_to_json(&s.layers)),
                        ("heads".to_string(), expr_to_json(&s.heads)),
                        ("head_dim".to_string(), expr_to_json(&s.head_dim)),
                        ("sinks".to_string(), expr_to_json(&s.sinks)),
                        ("window".to_string(), expr_to_json(&s.window)),
                    ]),
                    JsonValue::String(s.dtype.as_str().to_string()),
                    JsonValue::String(tier),
                ),
                TypeKind::SpeculativeTree(st) => (
                    "SpeculativeTreeBlock".to_string(),
                    JsonValue::Object(vec![
                        ("max_nodes".to_string(), expr_to_json(&st.max_nodes)),
                        ("max_depth".to_string(), expr_to_json(&st.max_depth)),
                        ("head_dim".to_string(), expr_to_json(&st.head_dim)),
                    ]),
                    JsonValue::String(st.dtype.as_str().to_string()),
                    JsonValue::String(tier),
                ),
                TypeKind::Custom(c) => (
                    c.clone(),
                    JsonValue::Null,
                    JsonValue::Null,
                    JsonValue::String(tier),
                ),
                TypeKind::Primitive(p) => (
                    p.clone(),
                    JsonValue::Null,
                    JsonValue::Null,
                    JsonValue::String(tier),
                ),
                _ => (
                    "Unknown".to_string(),
                    JsonValue::Null,
                    JsonValue::Null,
                    JsonValue::String(tier),
                ),
            }
        }
        None => (
            "unspecified".to_string(),
            JsonValue::Null,
            JsonValue::Null,
            JsonValue::Null,
        ),
    };

    JsonValue::Object(vec![
        ("node_type".to_string(), JsonValue::String("KVCacheDeclarationNode".to_string())),
        ("name".to_string(), JsonValue::String(kv.name.clone())),
        ("is_mut".to_string(), JsonValue::Bool(kv.is_mut)),
        ("cache_type".to_string(), JsonValue::String(cache_type_str)),
        ("dtype".to_string(), dtype_val),
        ("memory_tier".to_string(), tier_val),
        ("params".to_string(), params_val),
        ("span".to_string(), span_to_json(&kv.span)),
    ])
}

fn scope_to_json(s: &ExecutionScopeNode) -> JsonValue {
    let (node_type, kind_str, name, params_count, stmts_count, span) = match s {
        ExecutionScopeNode::Prefill(n) => (
            "PrefillScopeNode",
            "prefill",
            &n.name,
            n.parameters.len(),
            n.body.statements.len(),
            &n.span,
        ),
        ExecutionScopeNode::Decode(n) => (
            "DecodeScopeNode",
            "decode",
            &n.name,
            n.parameters.len(),
            n.body.statements.len(),
            &n.span,
        ),
        ExecutionScopeNode::ChunkedPrefill(n) => (
            "ChunkedPrefillScopeNode",
            "chunked_prefill",
            &n.name,
            n.parameters.len(),
            n.body.statements.len(),
            &n.span,
        ),
        ExecutionScopeNode::SpeculativeVerify(n) => (
            "SpeculativeVerifyScopeNode",
            "speculative_verify",
            &n.name,
            n.parameters.len(),
            n.body.statements.len(),
            &n.span,
        ),
    };

    JsonValue::Object(vec![
        ("node_type".to_string(), JsonValue::String(node_type.to_string())),
        ("kind".to_string(), JsonValue::String(kind_str.to_string())),
        ("name".to_string(), JsonValue::String(name.clone())),
        ("parameters_count".to_string(), JsonValue::Unsigned(params_count as u64)),
        ("statements_count".to_string(), JsonValue::Unsigned(stmts_count as u64)),
        ("span".to_string(), span_to_json(span)),
    ])
}

impl ProgramNode {
    /// Converts `ProgramNode` and its full AST tree into a JSON value.
    #[must_use]
    pub fn to_json_value(&self) -> JsonValue {
        let obj = vec![
            ("node_type".to_string(), JsonValue::String("ProgramNode".to_string())),
            ("span".to_string(), span_to_json(&self.span)),
            (
                "imports".to_string(),
                JsonValue::Array(
                    self.imports
                        .iter()
                        .map(|i| {
                            JsonValue::Object(vec![
                                ("node_type".to_string(), JsonValue::String("ImportNode".to_string())),
                                ("path".to_string(), JsonValue::String(i.path.clone())),
                                ("span".to_string(), span_to_json(&i.span)),
                            ])
                        })
                        .collect(),
                ),
            ),
            (
                "target_hardware".to_string(),
                self.target_hardware
                    .as_ref()
                    .map_or(JsonValue::Null, |th| {
                        JsonValue::Object(vec![
                            ("node_type".to_string(), JsonValue::String("TargetHardwareNode".to_string())),
                            ("device_name".to_string(), JsonValue::String(th.device_name.clone())),
                            ("vram_capacity_bytes".to_string(), JsonValue::Unsigned(th.vram_capacity_bytes)),
                            (
                                "compute_capability".to_string(),
                                JsonValue::Array(vec![
                                    JsonValue::Unsigned(th.compute_capability.0 as u64),
                                    JsonValue::Unsigned(th.compute_capability.1 as u64),
                                ]),
                            ),
                            ("safety_factor".to_string(), JsonValue::Number(th.safety_factor)),
                            ("span".to_string(), span_to_json(&th.span)),
                        ])
                    }),
            ),
            (
                "model_declarations".to_string(),
                JsonValue::Array(
                    self.model_declarations
                        .iter()
                        .map(|m| {
                            JsonValue::Object(vec![
                                ("node_type".to_string(), JsonValue::String("ModelDeclarationNode".to_string())),
                                ("name".to_string(), JsonValue::String(m.name.clone())),
                                ("weights_count".to_string(), JsonValue::Unsigned(m.weights.len() as u64)),
                                ("kv_caches_count".to_string(), JsonValue::Unsigned(m.kv_caches.len() as u64)),
                                ("scopes_count".to_string(), JsonValue::Unsigned(m.scopes.len() as u64)),
                                (
                                    "const_generics".to_string(),
                                    JsonValue::Array(m.generic_params.iter().map(const_generic_to_json).collect()),
                                ),
                                (
                                    "weights".to_string(),
                                    JsonValue::Array(m.weights.iter().map(weight_to_json).collect()),
                                ),
                                (
                                    "kv_caches".to_string(),
                                    JsonValue::Array(m.kv_caches.iter().map(kv_cache_to_json).collect()),
                                ),
                                (
                                    "scopes".to_string(),
                                    JsonValue::Array(m.scopes.iter().map(scope_to_json).collect()),
                                ),
                                ("span".to_string(), span_to_json(&m.span)),
                            ])
                        })
                        .collect(),
                ),
            ),
        ];

        JsonValue::Object(obj)
    }

    /// Serializes entire program AST to formatted JSON.
    #[must_use]
    pub fn to_json(&self) -> String {
        self.to_json_value().to_pretty_string(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ast_node_creation() {
        let span = SourceSpan::synthetic("test.vlx");
        let mut program = ProgramNode::new(span.clone());
        let hw = TargetHardwareNode::new("sm_86", 4096 * 1024 * 1024, (8, 6), 0.95, span.clone());
        program.target_hardware = Some(hw);

        let model = ModelDeclarationNode::new("TestModel", span.clone());
        program.model_declarations.push(model);

        let json = program.to_json();
        assert!(json.contains("TestModel"));
        assert!(json.contains("TargetHardwareNode"));
    }

    #[test]
    fn test_all_23_ast_nodes_instantiation() {
        let span = SourceSpan::synthetic("nodes.vlx");

        // 1. ProgramNode
        let _n1 = ProgramNode::new(span.clone());
        // 2. TargetHardwareNode
        let _n2 = TargetHardwareNode::new("sm_90", 80 * 1024 * 1024 * 1024, (9, 0), 0.95, span.clone());
        // 3. ModelDeclarationNode
        let _n3 = ModelDeclarationNode::new("Llama3", span.clone());
        // 4. WeightDeclarationNode
        let _n4 = WeightDeclarationNode::new("w_q", span.clone());
        // 5. TensorTypeNode
        let _n5 = TensorTypeNode::new(vec![], ScalarType::F16, MemoryTier::VramImmutable, span.clone());
        // 6. PagedKVTypeNode
        let lit_expr = Box::new(ExprNode::Literal(LiteralNode::new(
            LiteralValue::Decimal(32),
            "32",
            span.clone(),
        )));
        let _n6 = PagedKVTypeNode::new(
            lit_expr.clone(),
            lit_expr.clone(),
            lit_expr.clone(),
            lit_expr.clone(),
            ScalarType::F16,
            MemoryTier::PagedVram,
            span.clone(),
        );
        // 7. VirtualKVTypeNode
        let _n7 = VirtualKVTypeNode::new(
            lit_expr.clone(),
            lit_expr.clone(),
            lit_expr.clone(),
            lit_expr.clone(),
            lit_expr.clone(),
            ScalarType::F16,
            MemoryTier::PagedVram,
            span.clone(),
        );
        // 8. LatentKVTypeNode
        let _n8 = LatentKVTypeNode::new(
            lit_expr.clone(),
            lit_expr.clone(),
            lit_expr.clone(),
            ScalarType::Fp8E4m3,
            MemoryTier::PagedVram,
            span.clone(),
        );
        // 9. StreamingKVTypeNode
        let _n9 = StreamingKVTypeNode::new(
            lit_expr.clone(),
            lit_expr.clone(),
            lit_expr.clone(),
            lit_expr.clone(),
            lit_expr.clone(),
            ScalarType::F16,
            MemoryTier::PagedVram,
            span.clone(),
        );
        // 10. SpeculativeTreeTypeNode
        let _n10 = SpeculativeTreeTypeNode::new(
            lit_expr.clone(),
            lit_expr.clone(),
            lit_expr.clone(),
            ScalarType::F16,
            MemoryTier::BumpSram,
            span.clone(),
        );
        // 11. PrefillScopeNode
        let _n11 = PrefillScopeNode::new("prefill", span.clone());
        // 12. DecodeScopeNode
        let _n12 = DecodeScopeNode::new("decode", span.clone());
        // 13. ChunkedPrefillScopeNode
        let _n13 = ChunkedPrefillScopeNode::new("chunked_prefill", span.clone());
        // 14. SpeculativeVerifyScopeNode
        let _n14 = SpeculativeVerifyScopeNode::new("speculative_verify", span.clone());
        // 15. LetBindingNode
        let _n15 = LetBindingNode::new("x", false, span.clone());
        // 16. AssignmentNode
        let _n16 = AssignmentNode::new(
            lit_expr.clone(),
            AssignmentOperator::Assign,
            lit_expr.clone(),
            span.clone(),
        );
        // 17. AsyncDmaTransferNode
        let _n17 = AsyncDmaTransferNode::new(
            DmaDirection::H2D,
            lit_expr.clone(),
            lit_expr.clone(),
            None,
            span.clone(),
        );
        // 18. TmaLoadNode
        let _n18 = TmaLoadNode::new(
            lit_expr.clone(),
            lit_expr.clone(),
            lit_expr.clone(),
            span.clone(),
        );
        // 19. TmaStoreNode
        let _n19 = TmaStoreNode::new(lit_expr.clone(), lit_expr.clone(), span.clone());
        // 20. YieldNode
        let _n20 = YieldNode::new(None, span.clone());
        // 21. TensorMatmulNode
        let _n21 = TensorMatmulNode::new(lit_expr.clone(), lit_expr.clone(), span.clone());
        // 22. AbsorbMlaNode
        let _n22 = AbsorbMlaNode::new(
            AbsorbKind::AbsorbMlaQ,
            lit_expr.clone(),
            lit_expr.clone(),
            span.clone(),
        );
        // 23. TensorSliceNode
        let _n23 = TensorSliceNode::new(lit_expr.clone(), vec![], span);
    }
}
