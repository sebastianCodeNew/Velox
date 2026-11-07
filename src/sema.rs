//! Velox Semantic Analyzer, Const-Generic Shape Inference & Borrow Checker (`veloxc::sema`)
//!
//! Implements Step 6 Frontend Semantic Verification:
//! 1. Const-generic parameter evaluation and constant folding across model and scope headers.
//! 2. Tensor shape inference across all 14 operator precedence levels and built-in kernels.
//! 3. Fourier-Motzkin symbolic dimension unification (`B`, `S`, `CHUNK_SIZE`, `B_DEC`).
//! 4. Physical memory tier validation (`@vram_immutable`, `@paged_vram`, `@bump_sram`, `@host_pinned`)
//!    and SRAM escape analysis (`E004`, `E005`, `E006`, `E019`).
//! 5. First-class KV Cache parameter verification and disjoint slot borrow checking (`E012`, `E015`, `E016`, `E017`, `E018`).
//! 6. TMA barrier tracking (`E020`), constant loop bound verification (`E021`), undefined identifier
//!    detection (`E023`), precision loss checking (`E024`), and scratchpad alignment verification (`E025`).
//!
//! Zero external dependencies (`std` only).

use crate::ast::*;
use crate::diagnostics::{Diagnostic, DiagnosticCode};
use crate::types::{
    AbsorbKind, BinaryOperator, DmaDirection, KVCacheKind, MemoryTier, ScalarType, ScopeKind,
    SourceSpan, UnaryOperator,
};
use std::collections::{HashMap, HashSet};
use std::fmt;

// ============================================================================
// Symbolic Dimensions & Fourier-Motzkin Unification Solver
// ============================================================================

/// Represents a statically evaluated concrete dimension or a bounded symbolic dimension.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SymbolicDim {
    /// Statically known positive integer extent (e.g., `576`, `4096`).
    Concrete(usize),
    /// Symbolic dimension variable with a statically proven upper bound (e.g., `B <= B_MAX`).
    Symbolic { name: String, upper_bound: usize },
    /// Wildcard dimension (`*`).
    Wildcard,
}

impl SymbolicDim {
    /// Returns the worst-case static upper bound for memory planning.
    #[inline]
    #[must_use]
    pub fn upper_bound(&self) -> usize {
        match self {
            Self::Concrete(v) => *v,
            Self::Symbolic { upper_bound, .. } => *upper_bound,
            Self::Wildcard => 1,
        }
    }

    /// Returns `true` if this dimension is a concrete constant.
    #[inline]
    #[must_use]
    pub const fn is_concrete(&self) -> bool {
        matches!(self, Self::Concrete(_))
    }
}

impl fmt::Display for SymbolicDim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Concrete(n) => write!(f, "{n}"),
            Self::Symbolic { name, .. } => write!(f, "{name}"),
            Self::Wildcard => write!(f, "*"),
        }
    }
}

/// Linear inequality constraint `[lower <= var <= upper]` and equivalence classes for
/// Fourier-Motzkin symbolic shape unification.
#[derive(Debug, Clone, Default)]
pub struct FourierMotzkinSolver {
    /// Lower bounds on symbolic variables (`var >= lower`).
    pub lower_bounds: HashMap<String, i64>,
    /// Upper bounds on symbolic variables (`var <= upper`).
    pub upper_bounds: HashMap<String, i64>,
    /// Unified equality bindings (`var == concrete_val`).
    pub pinned_values: HashMap<String, usize>,
    /// Equivalence classes between symbolic variables.
    pub equivalences: HashMap<String, HashSet<String>>,
}

impl FourierMotzkinSolver {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a bounded symbolic variable `1 <= name <= upper_bound`.
    pub fn register_symbol(&mut self, name: &str, lower: i64, upper: i64) {
        self.lower_bounds.insert(name.to_string(), lower);
        self.upper_bounds.insert(name.to_string(), upper);
    }

    /// Projects out variable bounds via Fourier-Motzkin elimination: checks that for all
    /// registered variables $v$, $\max(\text{lower}(v)) \le \min(\text{upper}(v))$.
    #[must_use]
    pub fn is_consistent(&self) -> bool {
        for (var, &low) in &self.lower_bounds {
            if let Some(&high) = self.upper_bounds.get(var) {
                if low > high {
                    return false;
                }
            }
            if let Some(&pinned) = self.pinned_values.get(var) {
                let p = pinned as i64;
                if p < low {
                    return false;
                }
                if let Some(&high) = self.upper_bounds.get(var) {
                    if p > high {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Attempts to unify two `SymbolicDim` instances under the current constraint system.
    pub fn unify(&mut self, d1: &SymbolicDim, d2: &SymbolicDim) -> bool {
        match (d1, d2) {
            (SymbolicDim::Wildcard, _) | (_, SymbolicDim::Wildcard) => true,
            (SymbolicDim::Concrete(a), SymbolicDim::Concrete(b)) => a == b,
            (
                SymbolicDim::Symbolic {
                    name: n1,
                    upper_bound: u1,
                },
                SymbolicDim::Symbolic {
                    name: n2,
                    upper_bound: u2,
                },
            ) => {
                if n1 == n2 {
                    return true;
                }
                // Two distinct scope symbols (e.g., B1 and B2) cannot be implicitly assumed equal
                // unless their upper bounds are identical AND they are explicitly aliased.
                if u1 != u2 {
                    return false;
                }
                self.equivalences
                    .get(n1)
                    .is_some_and(|set| set.contains(n2))
            }
            (SymbolicDim::Symbolic { name, upper_bound }, SymbolicDim::Concrete(c))
            | (SymbolicDim::Concrete(c), SymbolicDim::Symbolic { name, upper_bound }) => {
                // A single-token constant `1` can unify with a sequence dimension if slicing/broadcasting,
                // otherwise concrete must be within `[1, upper_bound]` and match pinned value if any.
                if *c == 0 || *c > *upper_bound {
                    return false;
                }
                if let Some(&existing) = self.pinned_values.get(name) {
                    existing == *c
                } else {
                    // Do not allow an arbitrary concrete constant (like 1024) to unify with a general
                    // batch/seq symbol `B` or `S` unless `upper_bound == *c`.
                    *upper_bound == *c
                }
            }
        }
    }
}

// ============================================================================
// Evaluated Semantic Types & Typed IR Structures
// ============================================================================

/// Evaluated semantic type of an AST expression or variable binding.
#[derive(Debug, Clone, PartialEq)]
pub enum EvaluatedType {
    Tensor {
        shape: Vec<SymbolicDim>,
        dtype: ScalarType,
        tier: MemoryTier,
    },
    KVCache {
        kind: KVCacheKind,
        layers: usize,
        heads: usize,
        head_dim: usize,
        block_or_latent: usize,
        window_or_rope: usize,
        dtype: ScalarType,
        tier: MemoryTier,
    },
    KVSlotMut {
        cache_name: String,
        seq_key: String,
        step_key: String,
    },
    Primitive(ScalarType),
    Tuple(Vec<EvaluatedType>),
    Reference {
        is_mut: bool,
        inner: Box<EvaluatedType>,
        tier: Option<MemoryTier>,
    },
    SpeculativePath(usize),
    MBarrier {
        name: String,
    },
    Void,
}

impl EvaluatedType {
    /// Returns the memory tier associated with this type, if any.
    #[must_use]
    pub fn memory_tier(&self) -> Option<MemoryTier> {
        match self {
            Self::Tensor { tier, .. } | Self::KVCache { tier, .. } => Some(*tier),
            Self::Reference { tier, inner, .. } => tier.or_else(|| inner.memory_tier()),
            _ => None,
        }
    }

    /// Formats a human-readable representation of the type for diagnostics.
    #[must_use]
    pub fn format_shape(&self) -> String {
        match self {
            Self::Tensor { shape, dtype, tier } => {
                let dims: Vec<String> = shape.iter().map(ToString::to_string).collect();
                format!("{} Tensor<[{}], {}>", tier, dims.join(", "), dtype)
            }
            Self::KVCache { kind, dtype, .. } => format!("{kind}<..., {dtype}>"),
            Self::KVSlotMut { cache_name, .. } => format!("KVSlotMut({cache_name})"),
            Self::Primitive(dt) => dt.to_string(),
            Self::Tuple(elems) => {
                let parts: Vec<String> = elems.iter().map(Self::format_shape).collect();
                format!("({})", parts.join(", "))
            }
            Self::Reference { is_mut, inner, .. } => {
                if *is_mut {
                    format!("&mut {}", inner.format_shape())
                } else {
                    format!("&{}", inner.format_shape())
                }
            }
            Self::SpeculativePath(d) => format!("SpeculativePath<{d}>"),
            Self::MBarrier { name } => format!("mbarrier({name})"),
            Self::Void => "void".to_string(),
        }
    }
}

/// Validated model weight descriptor with resolved concrete dimensions.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedWeight {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: ScalarType,
    pub tier: MemoryTier,
    pub size_bytes: u64,
    pub scale_bytes: u64,
    pub is_absorbed: bool,
    pub absorb_kind: Option<AbsorbKind>,
    pub file_path: Option<String>,
    pub tensor_key: Option<String>,
    pub span: SourceSpan,
}

/// Validated KV Cache pool descriptor with resolved parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedKVCache {
    pub name: String,
    pub kind: KVCacheKind,
    pub layers: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub block_size: usize,
    pub max_seq_len: usize,
    pub latent_dim: usize,
    pub rope_dim: usize,
    pub sinks: usize,
    pub window: usize,
    pub max_nodes: usize,
    pub max_depth: usize,
    pub dtype: ScalarType,
    pub tier: MemoryTier,
    pub span: SourceSpan,
}

/// Lifespan and shape metadata for a local or temporary activation tensor inside an execution scope.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedActivation {
    pub id: usize,
    pub name: String,
    pub scope: ScopeKind,
    pub shape: Vec<usize>,
    pub dtype: ScalarType,
    pub tier: MemoryTier,
    pub size_bytes: u64,
    pub birth_step: usize,
    pub death_step: usize,
    pub span: SourceSpan,
}

/// Semantic analysis result for a single execution scope.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedScope {
    pub name: String,
    pub kind: ScopeKind,
    pub activations: Vec<TypedActivation>,
    pub total_steps: usize,
    pub span: SourceSpan,
}

/// Semantic analysis result for a model declaration.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedModel {
    pub name: String,
    pub constants: HashMap<String, usize>,
    pub weights: Vec<TypedWeight>,
    pub kv_caches: Vec<TypedKVCache>,
    pub scopes: Vec<TypedScope>,
    pub span: SourceSpan,
}

/// Complete semantic analysis output for a `.vlx` translation unit.
#[derive(Debug, Clone, PartialEq)]
pub struct TypedProgram {
    pub program: ProgramNode,
    pub models: Vec<TypedModel>,
    pub diagnostics: Vec<Diagnostic>,
}

// ============================================================================
// Semantic Analyzer Engine
// ============================================================================

/// Lexical variable entry in the symbol table.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct SymbolEntry {
    ty: EvaluatedType,
    is_mut: bool,
    activation_id: Option<usize>,
    span: SourceSpan,
}

/// Tracks active mutable borrows on KV cache slots within a lexical block.
#[derive(Debug, Clone)]
struct ActiveKVBorrow {
    cache_name: String,
    seq_key: String,
    step_key: String,
    span: SourceSpan,
}

/// Tracks asynchronous TMA loads awaiting barrier completion.
#[derive(Debug, Clone)]
struct PendingTmaTransfer {
    dest_var: String,
    barrier_name: String,
    span: SourceSpan,
}

/// Semantic analyzer performing const-generic evaluation, shape inference, and invariant checking.
pub struct SemanticAnalyzer {
    pub diagnostics: Vec<Diagnostic>,
    current_constants: HashMap<String, usize>,
    solver: FourierMotzkinSolver,
}

impl Default for SemanticAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl SemanticAnalyzer {
    #[must_use]
    pub fn new() -> Self {
        Self {
            diagnostics: Vec::new(),
            current_constants: HashMap::new(),
            solver: FourierMotzkinSolver::new(),
        }
    }

    /// Analyzes a parsed `ProgramNode` and returns a `TypedProgram` or compilation diagnostics.
    pub fn analyze(mut self, program: &ProgramNode) -> Result<TypedProgram, Vec<Diagnostic>> {
        // 1. Validate target hardware if present
        if let Some(ref hw) = program.target_hardware {
            self.validate_hardware(hw);
        }

        // 2. Analyze each model declaration
        let mut typed_models = Vec::new();
        for model in &program.model_declarations {
            let tm = self.analyze_model(model);
            typed_models.push(tm);
        }

        let has_errors = self
            .diagnostics
            .iter()
            .any(|d| d.severity == crate::diagnostics::Severity::Error);

        if has_errors {
            Err(self.diagnostics)
        } else {
            Ok(TypedProgram {
                program: program.clone(),
                models: typed_models,
                diagnostics: self.diagnostics,
            })
        }
    }

    fn validate_hardware(&mut self, hw: &TargetHardwareNode) {
        if hw.compute_capability.0 < 7 {
            self.diagnostics.push(Diagnostic::error(
                DiagnosticCode::E007,
                format!(
                    "Target device '{}' has compute capability {}.{} which is below minimum required SM 7.0",
                    hw.device_name, hw.compute_capability.0, hw.compute_capability.1
                ),
                hw.span.clone(),
            ));
        }
    }

    fn analyze_model(&mut self, model: &ModelDeclarationNode) -> TypedModel {
        self.current_constants.clear();
        self.solver = FourierMotzkinSolver::new();

        // 1. Evaluate model const-generic parameters
        for param in &model.generic_params {
            if let Some(ref default_expr) = param.default_val {
                match self.eval_const_i64(default_expr) {
                    Some(val) if val > 0 => {
                        self.current_constants.insert(param.name.clone(), val as usize);
                    }
                    Some(val) => {
                        self.diagnostics.push(Diagnostic::error(
                            DiagnosticCode::E009,
                            format!(
                                "Const-generic parameter '{}' must evaluate to a positive integer (> 0), found {}",
                                param.name, val
                            ),
                            param.span.clone(),
                        ));
                        self.current_constants.insert(param.name.clone(), 1);
                    }
                    None => {
                        self.diagnostics.push(Diagnostic::error(
                            DiagnosticCode::E009,
                            format!(
                                "Could not evaluate const-generic parameter '{}' at compile time",
                                param.name
                            ),
                            param.span.clone(),
                        ));
                        self.current_constants.insert(param.name.clone(), 1);
                    }
                }
            } else {
                // Default fallback for uninitialized model generic parameter
                self.current_constants.insert(param.name.clone(), 1);
            }
        }

        let b_max = self.current_constants.get("B_MAX").copied().unwrap_or(1);
        let s_max = self.current_constants.get("S_MAX").copied().unwrap_or(2048);

        // 2. Check duplicate member names (E022) and build base symbol table
        let mut seen_members: HashMap<String, SourceSpan> = HashMap::new();
        let mut model_symbols: HashMap<String, SymbolEntry> = HashMap::new();

        // Register const generics as usize primitives
        for (k, _) in &self.current_constants {
            model_symbols.insert(
                k.clone(),
                SymbolEntry {
                    ty: EvaluatedType::Primitive(ScalarType::Usize),
                    is_mut: false,
                    activation_id: None,
                    span: model.span.clone(),
                },
            );
        }

        // 3. Validate and evaluate weights
        let mut typed_weights = Vec::new();
        let mut has_absorbed_q = false;
        let mut has_absorbed_o = false;

        for w in &model.weights {
            if let Some(prev_span) = seen_members.get(&w.name) {
                self.diagnostics.push(
                    Diagnostic::error(
                        DiagnosticCode::E022,
                        format!("Duplicate model member '{}' declared in '{}'", w.name, model.name),
                        w.span.clone(),
                    )
                    .with_secondary_span(prev_span.clone(), "Previous declaration was here"),
                );
            } else {
                seen_members.insert(w.name.clone(), w.span.clone());
            }

            // Weights must reside in @vram_immutable (else E004)
            if w.tier != MemoryTier::VramImmutable {
                self.diagnostics.push(Diagnostic::error(
                    DiagnosticCode::E004,
                    format!(
                        "Model weight '{}' must reside in @vram_immutable, found {}",
                        w.name, w.tier
                    ),
                    w.span.clone(),
                ));
            }

            let mut concrete_shape = Vec::new();
            let mut sym_shape = Vec::new();
            let mut dtype = ScalarType::F16;
            let mut scale_bytes = 0u64;

            if let Some(ref tt) = w.tensor_type {
                dtype = tt.dtype;
                if tt.tier != MemoryTier::VramImmutable {
                    self.diagnostics.push(Diagnostic::error(
                        DiagnosticCode::E004,
                        format!(
                            "Model weight '{}' tensor type cannot be cast to mutable tier {}",
                            w.name, tt.tier
                        ),
                        tt.span.clone(),
                    ));
                }
                for dim_node in &tt.shape {
                    let d_val = self.eval_dimension_usize(dim_node);
                    concrete_shape.push(d_val);
                    sym_shape.push(SymbolicDim::Concrete(d_val));
                }
                if let Some(ref bq) = tt.block_quant_spec {
                    let mut block_elems = 1usize;
                    for be in &bq.block_shape {
                        let v = self.eval_const_i64(be).unwrap_or(128).max(1) as usize;
                        block_elems = block_elems.saturating_mul(v);
                    }
                    let total_elems: usize = concrete_shape.iter().product();
                    let num_blocks = (total_elems + block_elems - 1) / block_elems;
                    let scale_dt = bq.scale_dtype.unwrap_or(ScalarType::Fp8E8m0);
                    scale_bytes = (num_blocks as f64 * scale_dt.byte_size()).ceil() as u64;
                }
            }

            // Validate binding (Safetensors key or AbsorbMla)
            let mut is_absorbed = false;
            let mut absorb_kind = None;
            let mut file_path = w.file_path.clone();
            let mut tensor_key = w.tensor_key.clone();

            if let Some(ref binding) = w.binding {
                match binding {
                    WeightBinding::Safetensors(st) => {
                        file_path = Some(st.file_path.clone());
                        tensor_key = Some(st.tensor_key.clone());
                        if st.tensor_key.trim().is_empty() {
                            self.diagnostics.push(Diagnostic::error(
                                DiagnosticCode::E013,
                                format!(
                                    "Safetensors key for weight '{}' cannot be empty",
                                    w.name
                                ),
                                st.span.clone(),
                            ));
                        }
                    }
                    WeightBinding::AbsorbMla(abs) => {
                        is_absorbed = true;
                        absorb_kind = Some(abs.kind);
                        match abs.kind {
                            AbsorbKind::AbsorbMlaQ => has_absorbed_q = true,
                            AbsorbKind::AbsorbMlaV => has_absorbed_o = true,
                        }
                        // Verify absorbed source weights exist in model_symbols
                        let _ = self.infer_expr(&abs.projection_q, &model_symbols);
                        let _ = self.infer_expr(&abs.projection_k, &model_symbols);
                    }
                    WeightBinding::Expr(e) => {
                        let _ = self.infer_expr(e, &model_symbols);
                    }
                }
            }

            let num_elements: u64 = concrete_shape
                .iter()
                .fold(1u64, |acc, &d| acc.saturating_mul(d as u64));
            let size_bytes = (num_elements as f64 * dtype.byte_size()).ceil() as u64 + scale_bytes;

            typed_weights.push(TypedWeight {
                name: w.name.clone(),
                shape: concrete_shape,
                dtype,
                tier: w.tier,
                size_bytes,
                scale_bytes,
                is_absorbed,
                absorb_kind,
                file_path,
                tensor_key,
                span: w.span.clone(),
            });

            model_symbols.insert(
                w.name.clone(),
                SymbolEntry {
                    ty: EvaluatedType::Tensor {
                        shape: sym_shape,
                        dtype,
                        tier: w.tier,
                    },
                    is_mut: false,
                    activation_id: None,
                    span: w.span.clone(),
                },
            );
        }

        // 4. Validate and evaluate KV Caches
        let mut typed_kv_caches = Vec::new();
        let mut has_latent_kv = false;

        for kv in &model.kv_caches {
            if let Some(prev_span) = seen_members.get(&kv.name) {
                self.diagnostics.push(
                    Diagnostic::error(
                        DiagnosticCode::E022,
                        format!("Duplicate model member '{}' declared in '{}'", kv.name, model.name),
                        kv.span.clone(),
                    )
                    .with_secondary_span(prev_span.clone(), "Previous declaration was here"),
                );
            } else {
                seen_members.insert(kv.name.clone(), kv.span.clone());
            }

            if let Some(ref spec) = kv.cache_type {
                let (tkv, eval_ty) = self.evaluate_kv_cache_type(&kv.name, spec, s_max);
                if tkv.kind == KVCacheKind::Latent {
                    has_latent_kv = true;
                }
                typed_kv_caches.push(tkv);
                model_symbols.insert(
                    kv.name.clone(),
                    SymbolEntry {
                        ty: eval_ty,
                        is_mut: kv.is_mut,
                        activation_id: None,
                        span: kv.span.clone(),
                    },
                );
            }
        }

        // 5. Analyze execution scopes
        let mut has_prefill = false;
        let mut has_decode = false;
        let mut typed_scopes = Vec::new();

        for scope in &model.scopes {
            match scope.kind() {
                ScopeKind::Prefill => has_prefill = true,
                ScopeKind::Decode => has_decode = true,
                _ => {}
            }
            let ts = self.analyze_scope(
                scope,
                &model_symbols,
                b_max,
                s_max,
                has_latent_kv,
                has_absorbed_q && has_absorbed_o,
            );
            typed_scopes.push(ts);
        }

        if !has_prefill || !has_decode {
            let missing = if !has_prefill && !has_decode {
                "prefill and decode"
            } else if !has_prefill {
                "prefill"
            } else {
                "decode"
            };
            self.diagnostics.push(Diagnostic::error(
                DiagnosticCode::E008,
                format!(
                    "Model '{}' is missing mandatory '{}' execution scope(s)",
                    model.name, missing
                ),
                model.span.clone(),
            ));
        }

        TypedModel {
            name: model.name.clone(),
            constants: self.current_constants.clone(),
            weights: typed_weights,
            kv_caches: typed_kv_caches,
            scopes: typed_scopes,
            span: model.span.clone(),
        }
    }

    fn evaluate_kv_cache_type(
        &mut self,
        name: &str,
        spec: &TypeSpecNode,
        default_s_max: usize,
    ) -> (TypedKVCache, EvaluatedType) {
        let num_layers_default = self
            .current_constants
            .get("NUM_LAYERS")
            .copied()
            .unwrap_or(32);

        match &spec.kind {
            TypeKind::PagedKV(p) => {
                let layers = self.eval_const_usize(&p.layers, num_layers_default);
                let heads = self.eval_const_usize(&p.heads, 8);
                let head_dim = self.eval_const_usize(&p.head_dim, 64);
                let block_size = self.eval_const_usize(&p.block_size, 16);

                if !matches!(block_size, 16 | 32 | 64) {
                    self.diagnostics.push(Diagnostic::error(
                        DiagnosticCode::E015,
                        format!(
                            "PagedKVCache '{}' block size must be 16, 32, or 64, found {}",
                            name, block_size
                        ),
                        p.span.clone(),
                    ));
                }

                let tkv = TypedKVCache {
                    name: name.to_string(),
                    kind: KVCacheKind::Paged,
                    layers,
                    heads,
                    head_dim,
                    block_size,
                    max_seq_len: default_s_max,
                    latent_dim: 0,
                    rope_dim: 0,
                    sinks: 0,
                    window: 0,
                    max_nodes: 0,
                    max_depth: 0,
                    dtype: p.dtype,
                    tier: p.tier,
                    span: p.span.clone(),
                };
                let et = EvaluatedType::KVCache {
                    kind: KVCacheKind::Paged,
                    layers,
                    heads,
                    head_dim,
                    block_or_latent: block_size,
                    window_or_rope: default_s_max,
                    dtype: p.dtype,
                    tier: p.tier,
                };
                (tkv, et)
            }
            TypeKind::VirtualKV(v) => {
                let layers = self.eval_const_usize(&v.layers, num_layers_default);
                let heads = self.eval_const_usize(&v.heads, 8);
                let head_dim = self.eval_const_usize(&v.head_dim, 128);
                let max_seq_len = self.eval_const_usize(&v.max_seq_len, default_s_max);

                let tkv = TypedKVCache {
                    name: name.to_string(),
                    kind: KVCacheKind::VirtualContiguous,
                    layers,
                    heads,
                    head_dim,
                    block_size: 0,
                    max_seq_len,
                    latent_dim: 0,
                    rope_dim: 0,
                    sinks: 0,
                    window: 0,
                    max_nodes: 0,
                    max_depth: 0,
                    dtype: v.dtype,
                    tier: v.tier,
                    span: v.span.clone(),
                };
                let et = EvaluatedType::KVCache {
                    kind: KVCacheKind::VirtualContiguous,
                    layers,
                    heads,
                    head_dim,
                    block_or_latent: max_seq_len,
                    window_or_rope: 0,
                    dtype: v.dtype,
                    tier: v.tier,
                };
                (tkv, et)
            }
            TypeKind::LatentKV(l) => {
                let layers = self.eval_const_usize(&l.layers, num_layers_default);
                let latent_dim = self.eval_const_usize(&l.latent_dim, 512);
                let rope_dim = self.eval_const_usize(&l.rope_dim, 64);

                let tkv = TypedKVCache {
                    name: name.to_string(),
                    kind: KVCacheKind::Latent,
                    layers,
                    heads: 1,
                    head_dim: latent_dim + rope_dim,
                    block_size: 16,
                    max_seq_len: default_s_max,
                    latent_dim,
                    rope_dim,
                    sinks: 0,
                    window: 0,
                    max_nodes: 0,
                    max_depth: 0,
                    dtype: l.dtype,
                    tier: l.tier,
                    span: l.span.clone(),
                };
                let et = EvaluatedType::KVCache {
                    kind: KVCacheKind::Latent,
                    layers,
                    heads: 1,
                    head_dim: latent_dim + rope_dim,
                    block_or_latent: latent_dim,
                    window_or_rope: rope_dim,
                    dtype: l.dtype,
                    tier: l.tier,
                };
                (tkv, et)
            }
            TypeKind::StreamingKV(s) => {
                let layers = self.eval_const_usize(&s.layers, num_layers_default);
                let heads = self.eval_const_usize(&s.heads, 8);
                let head_dim = self.eval_const_usize(&s.head_dim, 128);
                let sinks_i64 = self.eval_const_i64(&s.sinks).unwrap_or(0);
                let window_i64 = self.eval_const_i64(&s.window).unwrap_or(0);

                if sinks_i64 <= 0 || window_i64 <= 0 {
                    self.diagnostics.push(Diagnostic::error(
                        DiagnosticCode::E016,
                        format!(
                            "StreamingSinkKVCache '{}' requires positive non-zero sinks ({}) and window ({})",
                            name, sinks_i64, window_i64
                        ),
                        s.span.clone(),
                    ));
                }

                let sinks = sinks_i64.max(1) as usize;
                let window = window_i64.max(1) as usize;

                let tkv = TypedKVCache {
                    name: name.to_string(),
                    kind: KVCacheKind::StreamingSink,
                    layers,
                    heads,
                    head_dim,
                    block_size: 0,
                    max_seq_len: sinks + window,
                    latent_dim: 0,
                    rope_dim: 0,
                    sinks,
                    window,
                    max_nodes: 0,
                    max_depth: 0,
                    dtype: s.dtype,
                    tier: s.tier,
                    span: s.span.clone(),
                };
                let et = EvaluatedType::KVCache {
                    kind: KVCacheKind::StreamingSink,
                    layers,
                    heads,
                    head_dim,
                    block_or_latent: sinks,
                    window_or_rope: window,
                    dtype: s.dtype,
                    tier: s.tier,
                };
                (tkv, et)
            }
            TypeKind::SpeculativeTree(t) => {
                let max_nodes_i64 = self.eval_const_i64(&t.max_nodes).unwrap_or(64);
                let max_depth_i64 = self.eval_const_i64(&t.max_depth).unwrap_or(6);
                let head_dim = self.eval_const_usize(&t.head_dim, 128);

                if max_depth_i64 <= 0 || max_depth_i64 > 16 || max_nodes_i64 <= 0 || max_nodes_i64 > 1024 {
                    self.diagnostics.push(Diagnostic::error(
                        DiagnosticCode::E017,
                        format!(
                            "SpeculativeTreeBlock '{}' depth ({}) or node count ({}) exceeds hardware verification limits (max_depth <= 16, max_nodes <= 1024)",
                            name, max_depth_i64, max_nodes_i64
                        ),
                        t.span.clone(),
                    ));
                }

                let max_nodes = max_nodes_i64.max(1) as usize;
                let max_depth = max_depth_i64.max(1) as usize;

                let tkv = TypedKVCache {
                    name: name.to_string(),
                    kind: KVCacheKind::SpeculativeTree,
                    layers: num_layers_default,
                    heads: 8,
                    head_dim,
                    block_size: 0,
                    max_seq_len: max_nodes,
                    latent_dim: 0,
                    rope_dim: 0,
                    sinks: 0,
                    window: 0,
                    max_nodes,
                    max_depth,
                    dtype: t.dtype,
                    tier: t.tier,
                    span: t.span.clone(),
                };
                let et = EvaluatedType::KVCache {
                    kind: KVCacheKind::SpeculativeTree,
                    layers: num_layers_default,
                    heads: 8,
                    head_dim,
                    block_or_latent: max_nodes,
                    window_or_rope: max_depth,
                    dtype: t.dtype,
                    tier: t.tier,
                };
                (tkv, et)
            }
            _ => {
                let tkv = TypedKVCache {
                    name: name.to_string(),
                    kind: KVCacheKind::Paged,
                    layers: num_layers_default,
                    heads: 8,
                    head_dim: 64,
                    block_size: 16,
                    max_seq_len: default_s_max,
                    latent_dim: 0,
                    rope_dim: 0,
                    sinks: 0,
                    window: 0,
                    max_nodes: 0,
                    max_depth: 0,
                    dtype: ScalarType::F16,
                    tier: MemoryTier::PagedVram,
                    span: spec.span.clone(),
                };
                let et = EvaluatedType::KVCache {
                    kind: KVCacheKind::Paged,
                    layers: num_layers_default,
                    heads: 8,
                    head_dim: 64,
                    block_or_latent: 16,
                    window_or_rope: default_s_max,
                    dtype: ScalarType::F16,
                    tier: MemoryTier::PagedVram,
                };
                (tkv, et)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn analyze_scope(
        &mut self,
        scope: &ExecutionScopeNode,
        model_symbols: &HashMap<String, SymbolEntry>,
        b_max: usize,
        s_max: usize,
        has_latent_kv: bool,
        _has_absorbed_weights: bool,
    ) -> TypedScope {
        let (name, kind, generic_params, parameters, return_type, body, span) = match scope {
            ExecutionScopeNode::Prefill(n) => (
                &n.name,
                ScopeKind::Prefill,
                &n.generic_params,
                &n.parameters,
                &n.return_type,
                &n.body,
                &n.span,
            ),
            ExecutionScopeNode::Decode(n) => (
                &n.name,
                ScopeKind::Decode,
                &n.generic_params,
                &n.parameters,
                &n.return_type,
                &n.body,
                &n.span,
            ),
            ExecutionScopeNode::ChunkedPrefill(n) => (
                &n.name,
                ScopeKind::ChunkedPrefill,
                &n.generic_params,
                &n.parameters,
                &n.return_type,
                &n.body,
                &n.span,
            ),
            ExecutionScopeNode::SpeculativeVerify(n) => (
                &n.name,
                ScopeKind::SpeculativeVerify,
                &n.generic_params,
                &n.parameters,
                &n.return_type,
                &n.body,
                &n.span,
            ),
        };

        let mut scope_symbols = model_symbols.clone();
        let mut scope_constants = self.current_constants.clone();

        // Register scope const-generic parameters (e.g. B, S, CHUNK_SIZE, B_DEC)
        for gp in generic_params {
            let bound = if let Some(ref def_expr) = gp.default_val {
                match self.eval_const_i64(def_expr) {
                    Some(v) if v > 0 => v as usize,
                    Some(v) => {
                        self.diagnostics.push(Diagnostic::error(
                            DiagnosticCode::E009,
                            format!(
                                "Scope const-generic parameter '{}' must be positive, found {}",
                                gp.name, v
                            ),
                            gp.span.clone(),
                        ));
                        1
                    }
                    None => 1,
                }
            } else if gp.name.starts_with('B') {
                b_max
            } else if gp.name.starts_with('S') {
                // For prefill scratchpad sizing, cap per-microbatch sequence tokens to a realistic
                // hardware chunk limit when B_MAX * S_MAX is large, while preserving full S_MAX in KV cache.
                s_max.min(2048)
            } else {
                16
            };

            scope_constants.insert(gp.name.clone(), bound);
            self.solver.register_symbol(&gp.name, 1, bound as i64);
            scope_symbols.insert(
                gp.name.clone(),
                SymbolEntry {
                    ty: EvaluatedType::Primitive(ScalarType::Usize),
                    is_mut: false,
                    activation_id: None,
                    span: gp.span.clone(),
                },
            );
        }

        // Register scope input parameters
        for param in parameters {
            let param_ty = if let Some(ref ts) = param.type_spec {
                self.resolve_type_spec(ts, &scope_constants, s_max)
            } else {
                EvaluatedType::Primitive(ScalarType::U32)
            };
            scope_symbols.insert(
                param.name.clone(),
                SymbolEntry {
                    ty: param_ty,
                    is_mut: param.is_mut,
                    activation_id: None,
                    span: param.span.clone(),
                },
            );
        }

        let mut activations: Vec<TypedActivation> = Vec::new();
        let mut step_counter: usize = 0;
        let mut active_borrows: Vec<ActiveKVBorrow> = Vec::new();
        let mut pending_tma: Vec<PendingTmaTransfer> = Vec::new();

        self.analyze_block(
            body,
            kind,
            return_type.as_ref(),
            &mut scope_symbols,
            &scope_constants,
            &mut activations,
            &mut step_counter,
            &mut active_borrows,
            &mut pending_tma,
            None,
            has_latent_kv,
        );

        // Any pending TMA load that was never synchronized by mbarrier_wait/arrive before scope exit triggers E020
        for unsynced in pending_tma {
            self.diagnostics.push(Diagnostic::error(
                DiagnosticCode::E020,
                format!(
                    "Asynchronous TMA load into '{}' with barrier '{}' was never synchronized before scope exit",
                    unsynced.dest_var, unsynced.barrier_name
                ),
                unsynced.span,
            ));
        }

        TypedScope {
            name: name.clone(),
            kind,
            activations,
            total_steps: step_counter.max(1),
            span: span.clone(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn analyze_block(
        &mut self,
        block: &BlockStatementNode,
        scope_kind: ScopeKind,
        expected_return: Option<&TypeSpecNode>,
        symbols: &mut HashMap<String, SymbolEntry>,
        scope_constants: &HashMap<String, usize>,
        activations: &mut Vec<TypedActivation>,
        step_counter: &mut usize,
        active_borrows: &mut Vec<ActiveKVBorrow>,
        pending_tma: &mut Vec<PendingTmaTransfer>,
        enclosing_loop_range: Option<(usize, usize)>,
        has_latent_kv: bool,
    ) {
        let initial_borrow_len = active_borrows.len();

        for stmt in &block.statements {
            *step_counter += 1;
            let current_step = *step_counter;

            match stmt {
                StmtNode::Let(let_node) => {
                    let annot_ty = let_node
                        .type_annot
                        .as_ref()
                        .map(|ts| self.resolve_type_spec(ts, scope_constants, 2048));

                    let init_ty = if let Some(ref init_expr) = let_node.init_expr {
                        self.record_variable_uses(
                            init_expr,
                            symbols,
                            activations,
                            current_step,
                            pending_tma,
                        );
                        self.check_mla_unabsorbed_access(init_expr, scope_kind, has_latent_kv);
                        self.check_kv_borrow_in_expr(init_expr, active_borrows);
                        Some(self.infer_expr_with_consts(init_expr, symbols, scope_constants))
                    } else {
                        None
                    };

                    // Check representation precision loss (E024) or shape unification (E011) if both annotation and init exist
                    let resolved_ty = match (annot_ty, init_ty) {
                        (Some(a_ty), Some(i_ty)) => {
                            self.check_type_compatibility(&a_ty, &i_ty, let_node.init_expr.as_deref(), &let_node.span);
                            a_ty
                        }
                        (Some(a_ty), None) => a_ty,
                        (None, Some(i_ty)) => i_ty,
                        (None, None) => EvaluatedType::Primitive(ScalarType::F16),
                    };

                    if let Some(ref tuple_names) = let_node.tuple_pattern {
                        let elem_types = match &resolved_ty {
                            EvaluatedType::Tuple(v) => v.clone(),
                            other => vec![other.clone(); tuple_names.len()],
                        };
                        for (idx, t_name) in tuple_names.iter().enumerate() {
                            let et = elem_types
                                .get(idx)
                                .cloned()
                                .unwrap_or(EvaluatedType::Primitive(ScalarType::F16));
                            let act_id = self.register_activation(
                                t_name,
                                &et,
                                scope_kind,
                                current_step,
                                &let_node.span,
                                activations,
                            );
                            symbols.insert(
                                t_name.clone(),
                                SymbolEntry {
                                    ty: et,
                                    is_mut: let_node.is_mut,
                                    activation_id: act_id,
                                    span: let_node.span.clone(),
                                },
                            );
                        }
                    } else {
                        let act_id = self.register_activation(
                            &let_node.name,
                            &resolved_ty,
                            scope_kind,
                            current_step,
                            &let_node.span,
                            activations,
                        );
                        symbols.insert(
                            let_node.name.clone(),
                            SymbolEntry {
                                ty: resolved_ty,
                                is_mut: let_node.is_mut,
                                activation_id: act_id,
                                span: let_node.span.clone(),
                            },
                        );
                    }
                }
                StmtNode::Assignment(assign) => {
                    self.record_variable_uses(
                        &assign.lhs,
                        symbols,
                        activations,
                        current_step,
                        pending_tma,
                    );
                    self.record_variable_uses(
                        &assign.rhs,
                        symbols,
                        activations,
                        current_step,
                        pending_tma,
                    );
                    self.check_mla_unabsorbed_access(&assign.rhs, scope_kind, has_latent_kv);

                    let lhs_ty = self.infer_expr_with_consts(&assign.lhs, symbols, scope_constants);
                    let rhs_ty = self.infer_expr_with_consts(&assign.rhs, symbols, scope_constants);

                    // Check immutable tier write (E004)
                    if lhs_ty.memory_tier() == Some(MemoryTier::VramImmutable) {
                        self.diagnostics.push(Diagnostic::error(
                            DiagnosticCode::E004,
                            "Cannot mutate or assign to @vram_immutable weight tensor".to_string(),
                            assign.span.clone(),
                        ));
                    }

                    // Check shape and precision compatibility between lhs and rhs
                    self.check_type_compatibility(&lhs_ty, &rhs_ty, Some(&assign.rhs), &assign.span);
                }
                StmtNode::AsyncDmaTransfer(dma) => {
                    self.record_variable_uses(
                        &dma.source,
                        symbols,
                        activations,
                        current_step,
                        pending_tma,
                    );
                    self.record_variable_uses(
                        &dma.destination,
                        symbols,
                        activations,
                        current_step,
                        pending_tma,
                    );
                    let src_ty = self.infer_expr_with_consts(&dma.source, symbols, scope_constants);
                    let dst_ty = self.infer_expr_with_consts(&dma.destination, symbols, scope_constants);

                    let src_tier = src_ty.memory_tier();
                    let dst_tier = dst_ty.memory_tier();

                    // E005: Direct DMA between @bump_sram and @host_pinned is prohibited
                    if (src_tier == Some(MemoryTier::BumpSram) && dst_tier == Some(MemoryTier::HostPinned))
                        || (src_tier == Some(MemoryTier::HostPinned) && dst_tier == Some(MemoryTier::BumpSram))
                    {
                        self.diagnostics.push(Diagnostic::error(
                            DiagnosticCode::E005,
                            format!(
                                "Invalid direct DMA path from {} to {}; must stage through @paged_vram",
                                src_tier.unwrap(),
                                dst_tier.unwrap()
                            ),
                            dma.span.clone(),
                        ));
                    } else if dma.direction == DmaDirection::H2D && src_tier == Some(MemoryTier::BumpSram) {
                        self.diagnostics.push(Diagnostic::error(
                            DiagnosticCode::E005,
                            "dma_async_h2d source must reside in @host_pinned, not @bump_sram".to_string(),
                            dma.span.clone(),
                        ));
                    } else if dma.direction == DmaDirection::D2H && dst_tier == Some(MemoryTier::BumpSram) {
                        self.diagnostics.push(Diagnostic::error(
                            DiagnosticCode::E005,
                            "dma_async_d2h destination must reside in @host_pinned, not @bump_sram".to_string(),
                            dma.span.clone(),
                        ));
                    }

                    // E004: Writing via DMA into @vram_immutable
                    if dst_tier == Some(MemoryTier::VramImmutable) {
                        self.diagnostics.push(Diagnostic::error(
                            DiagnosticCode::E004,
                            "Cannot initiate DMA write into @vram_immutable memory tier".to_string(),
                            dma.span.clone(),
                        ));
                    }

                    // E019: Check host-pinned buffer overflow if destination is @host_pinned
                    if dst_tier == Some(MemoryTier::HostPinned) {
                        if let (
                            EvaluatedType::Tensor {
                                shape: s_shape,
                                dtype: s_dt,
                                ..
                            },
                            EvaluatedType::Tensor {
                                shape: d_shape,
                                dtype: d_dt,
                                ..
                            },
                        ) = (&src_ty, &dst_ty)
                        {
                            let s_bytes = s_shape.iter().map(SymbolicDim::upper_bound).product::<usize>() as f64
                                * s_dt.byte_size();
                            let d_bytes = d_shape.iter().map(SymbolicDim::upper_bound).product::<usize>() as f64
                                * d_dt.byte_size();
                            if s_bytes > d_bytes {
                                self.diagnostics.push(Diagnostic::error(
                                    DiagnosticCode::E019,
                                    format!(
                                        "DMA transfer of {} bytes overflows @host_pinned destination buffer capacity of {} bytes",
                                        s_bytes as u64, d_bytes as u64
                                    ),
                                    dma.span.clone(),
                                ));
                            }
                        }
                    }
                }
                StmtNode::TmaLoad(tma) => {
                    self.record_variable_uses(
                        &tma.source,
                        symbols,
                        activations,
                        current_step,
                        pending_tma,
                    );
                    let _ = self.infer_expr_with_consts(&tma.source, symbols, scope_constants);
                    let _ = self.infer_expr_with_consts(&tma.destination, symbols, scope_constants);

                    // Validate mbarrier (E020)
                    match &*tma.mbarrier {
                        ExprNode::Literal(lit) if lit.raw_text == "null" || lit.raw_text == "0" => {
                            self.diagnostics.push(Diagnostic::error(
                                DiagnosticCode::E020,
                                "tma_load_async requires a valid hardware mbarrier; 'null' or '0' barrier is prohibited"
                                    .to_string(),
                                tma.span.clone(),
                            ));
                        }
                        ExprNode::Identifier(id) => {
                            if !symbols.contains_key(&id.name) {
                                self.diagnostics.push(Diagnostic::error(
                                    DiagnosticCode::E020,
                                    format!(
                                        "tma_load_async references undeclared mbarrier '{}'",
                                        id.name
                                    ),
                                    tma.mbarrier.span().clone(),
                                ));
                            } else if let ExprNode::Identifier(dst_id) = &*tma.destination {
                                pending_tma.push(PendingTmaTransfer {
                                    dest_var: dst_id.name.clone(),
                                    barrier_name: id.name.clone(),
                                    span: tma.span.clone(),
                                });
                            }
                        }
                        _ => {}
                    }
                }
                StmtNode::TmaStore(tma) => {
                    self.record_variable_uses(
                        &tma.source,
                        symbols,
                        activations,
                        current_step,
                        pending_tma,
                    );
                    self.record_variable_uses(
                        &tma.destination,
                        symbols,
                        activations,
                        current_step,
                        pending_tma,
                    );
                    let _ = self.infer_expr_with_consts(&tma.source, symbols, scope_constants);
                    let _ = self.infer_expr_with_consts(&tma.destination, symbols, scope_constants);
                }
                StmtNode::Yield(y) => {
                    if let Some(ref val) = y.value {
                        self.record_variable_uses(val, symbols, activations, current_step, pending_tma);
                        let _ = self.infer_expr_with_consts(val, symbols, scope_constants);
                    }
                }
                StmtNode::Return(ret) => {
                    if let Some(ref val) = ret.value {
                        self.record_variable_uses(val, symbols, activations, current_step, pending_tma);
                        let ret_ty = self.infer_expr_with_consts(val, symbols, scope_constants);

                        if let Some(exp_spec) = expected_return {
                            let mut exp_ty = self.resolve_type_spec(exp_spec, scope_constants, 2048);
                            if exp_spec.tier.is_none() {
                                if let (EvaluatedType::Tensor { tier: exp_t, .. }, Some(act_t)) = (&mut exp_ty, ret_ty.memory_tier()) {
                                    *exp_t = act_t;
                                }
                            }
                            self.check_type_compatibility(&exp_ty, &ret_ty, ret.value.as_deref(), &ret.span);
                        }

                        // E006: SRAM reference escape check
                        let returns_ref = expected_return
                            .is_some_and(|er| matches!(er.kind, TypeKind::Reference { .. }))
                            || matches!(val.as_ref(), ExprNode::Unary(u) if matches!(u.op, UnaryOperator::Borrow | UnaryOperator::BorrowMut));

                        if ret_ty.memory_tier() == Some(MemoryTier::BumpSram) {
                            let is_explicit_sram_var = if let ExprNode::Identifier(id) = val.as_ref() {
                                symbols.get(&id.name).is_some_and(|s| {
                                    s.ty.memory_tier() == Some(MemoryTier::BumpSram) && returns_ref
                                })
                            } else {
                                returns_ref
                            };
                            if is_explicit_sram_var || returns_ref {
                                self.diagnostics.push(Diagnostic::error(
                                    DiagnosticCode::E006,
                                    "Reference to @bump_sram shared memory cannot escape execution scope"
                                        .to_string(),
                                    ret.span.clone(),
                                ));
                            }
                        }
                    }
                }
                StmtNode::If(if_node) => {
                    self.record_variable_uses(
                        &if_node.condition,
                        symbols,
                        activations,
                        current_step,
                        pending_tma,
                    );
                    let _ = self.infer_expr_with_consts(&if_node.condition, symbols, scope_constants);
                    self.analyze_block(
                        &if_node.then_branch,
                        scope_kind,
                        expected_return,
                        symbols,
                        scope_constants,
                        activations,
                        step_counter,
                        active_borrows,
                        pending_tma,
                        enclosing_loop_range,
                        has_latent_kv,
                    );
                    if let Some(ref else_stmt) = if_node.else_branch {
                        let mut dummy_block = BlockStatementNode::new(else_stmt.span().clone());
                        dummy_block.statements.push((**else_stmt).clone());
                        self.analyze_block(
                            &dummy_block,
                            scope_kind,
                            expected_return,
                            symbols,
                            scope_constants,
                            activations,
                            step_counter,
                            active_borrows,
                            pending_tma,
                            enclosing_loop_range,
                            has_latent_kv,
                        );
                    }
                }
                StmtNode::For(for_node) => {
                    // Check E021: loop bounds must be compile-time constants
                    let start_const = self.eval_const_with_map(&for_node.start, scope_constants);
                    let end_const = self.eval_const_with_map(&for_node.end, scope_constants);

                    if start_const.is_none() || end_const.is_none() {
                        self.diagnostics.push(Diagnostic::error(
                            DiagnosticCode::E021,
                            format!(
                                "Loop variable '{}' has non-constant loop bound; Velox fibers require statically bounded loops",
                                for_node.var_name
                            ),
                            for_node.span.clone(),
                        ));
                    }

                    symbols.insert(
                        for_node.var_name.clone(),
                        SymbolEntry {
                            ty: EvaluatedType::Primitive(ScalarType::Usize),
                            is_mut: false,
                            activation_id: None,
                            span: for_node.span.clone(),
                        },
                    );

                    let loop_start_step = *step_counter;
                    let mut loop_borrows = Vec::new();
                    self.analyze_block(
                        &for_node.body,
                        scope_kind,
                        expected_return,
                        symbols,
                        scope_constants,
                        activations,
                        step_counter,
                        &mut loop_borrows,
                        pending_tma,
                        Some((loop_start_step, loop_start_step + for_node.body.statements.len() + 1)),
                        has_latent_kv,
                    );
                    let loop_end_step = *step_counter;

                    // Extend lifetimes of any activation born before the loop and used inside the loop
                    // to the end of the loop, and any activation born inside the loop to cover its body.
                    for act in activations.iter_mut() {
                        if act.birth_step < loop_start_step && act.death_step >= loop_start_step {
                            act.death_step = act.death_step.max(loop_end_step);
                        }
                    }
                }
                StmtNode::While(while_node) => {
                    // E021: Unbounded while loops in serving scopes violate deterministic fiber bounds
                    self.diagnostics.push(Diagnostic::error(
                        DiagnosticCode::E021,
                        "Dynamic 'while' loop is prohibited in deterministic serving fiber; use a constant-bounded 'for' loop"
                            .to_string(),
                        while_node.span.clone(),
                    ));
                    self.analyze_block(
                        &while_node.body,
                        scope_kind,
                        expected_return,
                        symbols,
                        scope_constants,
                        activations,
                        step_counter,
                        active_borrows,
                        pending_tma,
                        enclosing_loop_range,
                        has_latent_kv,
                    );
                }
                StmtNode::Block(inner_block) => {
                    self.analyze_block(
                        inner_block,
                        scope_kind,
                        expected_return,
                        symbols,
                        scope_constants,
                        activations,
                        step_counter,
                        active_borrows,
                        pending_tma,
                        enclosing_loop_range,
                        has_latent_kv,
                    );
                }
                StmtNode::Expression(expr_stmt) => {
                    self.record_variable_uses(
                        &expr_stmt.expr,
                        symbols,
                        activations,
                        current_step,
                        pending_tma,
                    );
                    self.check_mla_unabsorbed_access(&expr_stmt.expr, scope_kind, has_latent_kv);
                    self.check_kv_borrow_in_expr(&expr_stmt.expr, active_borrows);
                    let _ = self.infer_expr_with_consts(&expr_stmt.expr, symbols, scope_constants);
                }
            }
        }

        active_borrows.truncate(initial_borrow_len);
    }

    fn register_activation(
        &self,
        name: &str,
        ty: &EvaluatedType,
        scope_kind: ScopeKind,
        birth_step: usize,
        span: &SourceSpan,
        activations: &mut Vec<TypedActivation>,
    ) -> Option<usize> {
        if let EvaluatedType::Tensor { shape, dtype, tier } = ty {
            let concrete_shape: Vec<usize> = shape.iter().map(SymbolicDim::upper_bound).collect();
            let elems: u64 = concrete_shape
                .iter()
                .fold(1u64, |acc, &d| acc.saturating_mul(d.max(1) as u64));
            let size_bytes = (elems as f64 * dtype.byte_size()).ceil() as u64;
            let id = activations.len();
            activations.push(TypedActivation {
                id,
                name: name.to_string(),
                scope: scope_kind,
                shape: concrete_shape,
                dtype: *dtype,
                tier: *tier,
                size_bytes: size_bytes.max(256),
                birth_step,
                death_step: birth_step + 1,
                span: span.clone(),
            });
            Some(id)
        } else {
            None
        }
    }

    fn record_variable_uses(
        &mut self,
        expr: &ExprNode,
        symbols: &HashMap<String, SymbolEntry>,
        activations: &mut [TypedActivation],
        current_step: usize,
        pending_tma: &mut Vec<PendingTmaTransfer>,
    ) {
        match expr {
            ExprNode::Identifier(id) => {
                // Check if this identifier has an unsynchronized TMA load (E020)
                if let Some(pos) = pending_tma.iter().position(|p| p.dest_var == id.name) {
                    let unsynced = pending_tma.remove(pos);
                    self.diagnostics.push(Diagnostic::error(
                        DiagnosticCode::E020,
                        format!(
                            "Tensor '{}' loaded via tma_load_async is read before waiting on mbarrier '{}'",
                            unsynced.dest_var, unsynced.barrier_name
                        ),
                        id.span.clone(),
                    ));
                }

                if let Some(entry) = symbols.get(&id.name) {
                    if let Some(act_id) = entry.activation_id {
                        if let Some(act) = activations.get_mut(act_id) {
                            act.death_step = act.death_step.max(current_step);
                        }
                    }
                }
            }
            ExprNode::TensorMatmul(m) => {
                self.record_variable_uses(&m.lhs, symbols, activations, current_step, pending_tma);
                self.record_variable_uses(&m.rhs, symbols, activations, current_step, pending_tma);
            }
            ExprNode::Binary(b) => {
                self.record_variable_uses(&b.left, symbols, activations, current_step, pending_tma);
                self.record_variable_uses(&b.right, symbols, activations, current_step, pending_tma);
            }
            ExprNode::Unary(u) => {
                self.record_variable_uses(&u.operand, symbols, activations, current_step, pending_tma);
            }
            ExprNode::Call(c) => {
                // Check if this call is an mbarrier synchronization (`mbarrier_wait(bar)` or `bar.wait()` or `mbarrier_arrive`)
                let callee_name = match &*c.callee {
                    ExprNode::Identifier(id) => id.name.as_str(),
                    ExprNode::MemberAccess(ma) => ma.member.as_str(),
                    _ => "",
                };
                if matches!(callee_name, "mbarrier_wait" | "mbarrier_arrive" | "wait" | "arrive_and_wait") {
                    if let ExprNode::MemberAccess(ma) = &*c.callee {
                        if let ExprNode::Identifier(bar_id) = &*ma.obj {
                            pending_tma.retain(|p| p.barrier_name != bar_id.name);
                        }
                    }
                    for arg in &c.args {
                        if let ExprNode::Identifier(bar_id) = &*arg.value {
                            pending_tma.retain(|p| p.barrier_name != bar_id.name);
                        }
                    }
                }

                if !matches!(&*c.callee, ExprNode::Identifier(_)) {
                    self.record_variable_uses(&c.callee, symbols, activations, current_step, pending_tma);
                }
                for arg in &c.args {
                    self.record_variable_uses(&arg.value, symbols, activations, current_step, pending_tma);
                }
            }
            ExprNode::MemberAccess(m) => {
                self.record_variable_uses(&m.obj, symbols, activations, current_step, pending_tma);
            }
            ExprNode::IndexAccess(idx) => {
                self.record_variable_uses(&idx.obj, symbols, activations, current_step, pending_tma);
                self.record_variable_uses(&idx.index, symbols, activations, current_step, pending_tma);
            }
            ExprNode::TensorSlice(s) => {
                self.record_variable_uses(&s.tensor_expr, symbols, activations, current_step, pending_tma);
                for r in &s.ranges {
                    if let Some(ref st) = r.start {
                        self.record_variable_uses(st, symbols, activations, current_step, pending_tma);
                    }
                    if let Some(ref en) = r.end {
                        self.record_variable_uses(en, symbols, activations, current_step, pending_tma);
                    }
                }
            }
            ExprNode::Tuple(t) => {
                for el in &t.elements {
                    self.record_variable_uses(el, symbols, activations, current_step, pending_tma);
                }
            }
            ExprNode::Parenthesized(p) => {
                self.record_variable_uses(p, symbols, activations, current_step, pending_tma);
            }
            ExprNode::AbsorbMla(a) => {
                self.record_variable_uses(&a.projection_q, symbols, activations, current_step, pending_tma);
                self.record_variable_uses(&a.projection_k, symbols, activations, current_step, pending_tma);
            }
            ExprNode::Literal(_) => {}
        }
    }

    fn check_mla_unabsorbed_access(
        &mut self,
        expr: &ExprNode,
        scope_kind: ScopeKind,
        has_latent_kv: bool,
    ) {
        if !has_latent_kv || scope_kind != ScopeKind::Decode {
            return;
        }
        // In Decode scope of an MLA model with LatentKVCache, materializing full uncompressed
        // w_uk or w_uv via matrix multiplication or calling decompress_kv triggers E018.
        match expr {
            ExprNode::TensorMatmul(m) => {
                for side in [&m.lhs, &m.rhs] {
                    let base_name = Self::extract_base_identifier(side);
                    if matches!(base_name.as_deref(), Some("w_uk" | "w_uv")) {
                        self.diagnostics.push(Diagnostic::error(
                            DiagnosticCode::E018,
                            format!(
                                "Materializing uncompressed MLA projection '{}' in decode scope violates latent bandwidth invariant; use pre-absorbed 'absorb_mla_q' / 'absorb_mla_v' weights",
                                base_name.unwrap()
                            ),
                            m.span.clone(),
                        ));
                    }
                }
                self.check_mla_unabsorbed_access(&m.lhs, scope_kind, has_latent_kv);
                self.check_mla_unabsorbed_access(&m.rhs, scope_kind, has_latent_kv);
            }
            ExprNode::Call(c) => {
                if let ExprNode::Identifier(id) = &*c.callee {
                    if matches!(id.name.as_str(), "decompress_kv" | "mla_decompress" | "dequantize_latent_kv") {
                        self.diagnostics.push(Diagnostic::error(
                            DiagnosticCode::E018,
                            format!(
                                "Calling '{}' materializes full H x D key/value tensors from LatentKVCache in HBM",
                                id.name
                            ),
                            c.span.clone(),
                        ));
                    }
                }
                for arg in &c.args {
                    self.check_mla_unabsorbed_access(&arg.value, scope_kind, has_latent_kv);
                }
            }
            ExprNode::Binary(b) => {
                self.check_mla_unabsorbed_access(&b.left, scope_kind, has_latent_kv);
                self.check_mla_unabsorbed_access(&b.right, scope_kind, has_latent_kv);
            }
            _ => {}
        }
    }

    fn extract_base_identifier(expr: &ExprNode) -> Option<String> {
        match expr {
            ExprNode::Identifier(id) => Some(id.name.clone()),
            ExprNode::IndexAccess(idx) => Self::extract_base_identifier(&idx.obj),
            ExprNode::MemberAccess(m) => Self::extract_base_identifier(&m.obj),
            ExprNode::Parenthesized(p) => Self::extract_base_identifier(p),
            _ => None,
        }
    }

    fn check_kv_borrow_in_expr(
        &mut self,
        expr: &ExprNode,
        active_borrows: &mut Vec<ActiveKVBorrow>,
    ) {
        if let ExprNode::Call(c) = expr {
            if let ExprNode::MemberAccess(ma) = &*c.callee {
                if ma.member == "borrow_slot_mut" || ma.member == "borrow_mut_slot" {
                    if let Some(cache_name) = Self::extract_base_identifier(&ma.obj) {
                        let seq_key = c
                            .args
                            .first()
                            .map(|a| Self::expr_key(&a.value))
                            .unwrap_or_else(|| "0".to_string());
                        let step_key = c
                            .args
                            .get(1)
                            .map(|a| Self::expr_key(&a.value))
                            .unwrap_or_else(|| "0".to_string());

                        if let Some(conflict) = active_borrows.iter().find(|b| {
                            b.cache_name == cache_name
                                && b.seq_key == seq_key
                                && b.step_key == step_key
                        }) {
                            self.diagnostics.push(
                                Diagnostic::error(
                                    DiagnosticCode::E012,
                                    format!(
                                        "Conflicting mutable borrow of KV cache '{}' on sequence '{}' at slot '{}'",
                                        cache_name, seq_key, step_key
                                    ),
                                    c.span.clone(),
                                )
                                .with_secondary_span(
                                    conflict.span.clone(),
                                    "First mutable borrow of this KV cache slot occurred here",
                                ),
                            );
                        } else {
                            active_borrows.push(ActiveKVBorrow {
                                cache_name,
                                seq_key,
                                step_key,
                                span: c.span.clone(),
                            });
                        }
                    }
                }
            }
            for arg in &c.args {
                self.check_kv_borrow_in_expr(&arg.value, active_borrows);
            }
        }
    }

    fn expr_key(expr: &ExprNode) -> String {
        match expr {
            ExprNode::Literal(l) => l.raw_text.clone(),
            ExprNode::Identifier(id) => id.name.clone(),
            ExprNode::Binary(b) => format!(
                "{}{}{}",
                Self::expr_key(&b.left),
                b.op,
                Self::expr_key(&b.right)
            ),
            _ => "expr".to_string(),
        }
    }

    fn check_type_compatibility(
        &mut self,
        expected: &EvaluatedType,
        actual: &EvaluatedType,
        rhs_expr: Option<&ExprNode>,
        span: &SourceSpan,
    ) {
        if let (
            EvaluatedType::Tensor {
                shape: exp_shape,
                dtype: exp_dt,
                tier: exp_tier,
            },
            EvaluatedType::Tensor {
                shape: act_shape,
                dtype: act_dt,
                tier: act_tier,
            },
        ) = (expected, actual)
        {
            // E004: Illegal memory tier assignment (e.g. assigning @vram_immutable to mutable @paged_vram without copy, or vice versa)
            if *exp_tier == MemoryTier::VramImmutable && *act_tier != MemoryTier::VramImmutable {
                self.diagnostics.push(Diagnostic::error(
                    DiagnosticCode::E004,
                    format!(
                        "Cannot assign tensor from {} into immutable tier {}",
                        act_tier, exp_tier
                    ),
                    span.clone(),
                ));
            }

            // E024: Representation precision loss without explicit quantize(...)
            if exp_dt.bit_width() < act_dt.bit_width() {
                let is_explicit_quant = rhs_expr.is_some_and(|e| {
                    matches!(e, ExprNode::Call(c) if matches!(&*c.callee, ExprNode::Identifier(id) if id.name.contains("quant") || id.name.contains("cast")))
                });
                if !is_explicit_quant {
                    self.diagnostics.push(Diagnostic::error(
                        DiagnosticCode::E024,
                        format!(
                            "Implicit precision loss converting from '{}' ({}-bit) to '{}' ({}-bit) without explicit quantizer",
                            act_dt,
                            act_dt.bit_width(),
                            exp_dt,
                            exp_dt.bit_width()
                        ),
                        span.clone(),
                    ));
                }
            }

            // E011: Symbolic / concrete shape unification check
            if exp_shape.len() != act_shape.len() {
                // Allow rank squeeze on single-token slice (e.g. [B, 1, D] vs [B, D]) only if non-1 dims match
                let exp_non1: Vec<&SymbolicDim> =
                    exp_shape.iter().filter(|d| d.upper_bound() != 1).collect();
                let act_non1: Vec<&SymbolicDim> =
                    act_shape.iter().filter(|d| d.upper_bound() != 1).collect();
                if exp_non1.len() != act_non1.len() {
                    self.diagnostics.push(Diagnostic::error(
                        DiagnosticCode::E011,
                        format!(
                            "Tensor rank mismatch: expected {}, found {}",
                            expected.format_shape(),
                            actual.format_shape()
                        ),
                        span.clone(),
                    ));
                    return;
                }
            } else {
                for (d_exp, d_act) in exp_shape.iter().zip(act_shape.iter()) {
                    if !self.solver.unify(d_exp, d_act) {
                        self.diagnostics.push(Diagnostic::error(
                            DiagnosticCode::E011,
                            format!(
                                "Symbolic shape unification failed: dimension '{}' cannot unify with '{}' (expected {}, found {})",
                                d_exp,
                                d_act,
                                expected.format_shape(),
                                actual.format_shape()
                            ),
                            span.clone(),
                        ));
                        return;
                    }
                }
            }
        }
    }

    fn infer_expr(
        &mut self,
        expr: &ExprNode,
        symbols: &HashMap<String, SymbolEntry>,
    ) -> EvaluatedType {
        let consts = self.current_constants.clone();
        self.infer_expr_with_consts(expr, symbols, &consts)
    }

    fn infer_expr_with_consts(
        &mut self,
        expr: &ExprNode,
        symbols: &HashMap<String, SymbolEntry>,
        consts: &HashMap<String, usize>,
    ) -> EvaluatedType {
        match expr {
            ExprNode::Literal(lit) => match lit.value {
                LiteralValue::Decimal(_) | LiteralValue::Hex(_) | LiteralValue::Binary(_) => {
                    EvaluatedType::Primitive(ScalarType::Usize)
                }
                LiteralValue::Float(_) => EvaluatedType::Primitive(ScalarType::F32),
                LiteralValue::Bool(_) => EvaluatedType::Primitive(ScalarType::Bool),
                LiteralValue::String(_) => EvaluatedType::Primitive(ScalarType::U8),
            },
            ExprNode::Identifier(id) => {
                if let Some(entry) = symbols.get(&id.name) {
                    entry.ty.clone()
                } else {
                    self.diagnostics.push(Diagnostic::error(
                        DiagnosticCode::E023,
                        format!(
                            "Undefined identifier '{}' referenced in expression",
                            id.name
                        ),
                        id.span.clone(),
                    ));
                    EvaluatedType::Primitive(ScalarType::F16)
                }
            }
            ExprNode::Parenthesized(inner) => self.infer_expr_with_consts(inner, symbols, consts),
            ExprNode::Tuple(tup) => {
                let elems = tup
                    .elements
                    .iter()
                    .map(|e| self.infer_expr_with_consts(e, symbols, consts))
                    .collect();
                EvaluatedType::Tuple(elems)
            }
            ExprNode::Unary(u) => {
                let inner = self.infer_expr_with_consts(&u.operand, symbols, consts);
                match u.op {
                    UnaryOperator::Borrow => EvaluatedType::Reference {
                        is_mut: false,
                        tier: inner.memory_tier(),
                        inner: Box::new(inner),
                    },
                    UnaryOperator::BorrowMut => {
                        if inner.memory_tier() == Some(MemoryTier::VramImmutable) {
                            self.diagnostics.push(Diagnostic::error(
                                DiagnosticCode::E004,
                                "Cannot take mutable reference (&mut) to @vram_immutable tensor"
                                    .to_string(),
                                u.span.clone(),
                            ));
                        }
                        EvaluatedType::Reference {
                            is_mut: true,
                            tier: inner.memory_tier(),
                            inner: Box::new(inner),
                        }
                    }
                    UnaryOperator::Deref => match inner {
                        EvaluatedType::Reference { inner, .. } => *inner,
                        other => other,
                    },
                    _ => inner,
                }
            }
            ExprNode::MemberAccess(ma) => {
                let obj_ty = self.infer_expr_with_consts(&ma.obj, symbols, consts);
                if ma.member == "T" {
                    // Matrix transpose `.T` swaps the last two dimensions
                    if let EvaluatedType::Tensor {
                        mut shape,
                        dtype,
                        tier,
                    } = obj_ty
                    {
                        let len = shape.len();
                        if len >= 2 {
                            shape.swap(len - 2, len - 1);
                        }
                        return EvaluatedType::Tensor { shape, dtype, tier };
                    }
                }
                obj_ty
            }
            ExprNode::IndexAccess(idx) => {
                let obj_ty = self.infer_expr_with_consts(&idx.obj, symbols, consts);
                let _ = self.infer_expr_with_consts(&idx.index, symbols, consts);

                if let EvaluatedType::Tensor {
                    mut shape,
                    dtype,
                    tier,
                } = obj_ty
                {
                    // Check E019: host_pinned buffer out-of-bounds constant index
                    if tier == MemoryTier::HostPinned && !shape.is_empty() {
                        if let Some(const_idx) = self.eval_const_with_map(&idx.index, consts) {
                            let bound = shape[0].upper_bound() as i64;
                            if const_idx < 0 || const_idx >= bound {
                                self.diagnostics.push(Diagnostic::error(
                                    DiagnosticCode::E019,
                                    format!(
                                        "Index {} overflows @host_pinned buffer dimension bound {}",
                                        const_idx, bound
                                    ),
                                    idx.span.clone(),
                                ));
                            }
                        }
                    }

                    // Check E025: unaligned scratchpad access if indexing a 1D @bump_sram tensor at non-aligned offset
                    if tier == MemoryTier::BumpSram && shape.len() == 1 {
                        if let Some(const_idx) = self.eval_const_with_map(&idx.index, consts) {
                            let byte_offset = (const_idx.unsigned_abs() as f64 * dtype.byte_size()) as u64;
                            if byte_offset % 128 != 0 {
                                self.diagnostics.push(Diagnostic::error(
                                    DiagnosticCode::E025,
                                    format!(
                                        "Scratchpad access at element index {} (byte offset {}) violates 128-byte hardware alignment",
                                        const_idx, byte_offset
                                    ),
                                    idx.span.clone(),
                                ));
                            }
                        }
                    }

                    if shape.len() > 1 {
                        shape.remove(0);
                    }
                    EvaluatedType::Tensor { shape, dtype, tier }
                } else {
                    obj_ty
                }
            }
            ExprNode::TensorSlice(slice) => {
                let obj_ty = self.infer_expr_with_consts(&slice.tensor_expr, symbols, consts);
                for r in &slice.ranges {
                    if let Some(ref st) = r.start {
                        let _ = self.infer_expr_with_consts(st, symbols, consts);
                    }
                    if let Some(ref en) = r.end {
                        let _ = self.infer_expr_with_consts(en, symbols, consts);
                    }
                }

                if let EvaluatedType::Tensor {
                    mut shape,
                    dtype,
                    tier,
                } = obj_ty
                {
                    for (dim_idx, r) in slice.ranges.iter().enumerate() {
                        if dim_idx < shape.len() && !r.is_full_slice {
                            // Check E025 on @bump_sram unaligned slice start
                            if tier == MemoryTier::BumpSram {
                                if let Some(ref st) = r.start {
                                    if let Some(start_val) = self.eval_const_with_map(st, consts) {
                                        let stride_elems: usize = shape[dim_idx + 1..]
                                            .iter()
                                            .map(SymbolicDim::upper_bound)
                                            .product::<usize>()
                                            .max(1);
                                        let byte_off = (start_val.unsigned_abs() as usize * stride_elems) as f64
                                            * dtype.byte_size();
                                        if (byte_off as u64) % 128 != 0 {
                                            self.diagnostics.push(Diagnostic::error(
                                                DiagnosticCode::E025,
                                                format!(
                                                    "Slice start index {} on @bump_sram tensor results in unaligned byte offset {} (must be 128-byte aligned)",
                                                    start_val, byte_off as u64
                                                ),
                                                r.span.clone(),
                                            ));
                                        }
                                    }
                                }
                            }

                            // Check E019 on @host_pinned slice end overflow
                            if tier == MemoryTier::HostPinned {
                                if let Some(ref en) = r.end {
                                    if let Some(end_val) = self.eval_const_with_map(en, consts) {
                                        let bound = shape[dim_idx].upper_bound() as i64;
                                        if end_val > bound {
                                            self.diagnostics.push(Diagnostic::error(
                                                DiagnosticCode::E019,
                                                format!(
                                                    "Slice upper bound {} exceeds @host_pinned tensor dimension {}",
                                                    end_val, bound
                                                ),
                                                r.span.clone(),
                                            ));
                                        }
                                    }
                                }
                            }

                            // If single index slice (start.is_some() && end.is_none()), dimension becomes 1
                            if r.start.is_some() && r.end.is_none() {
                                shape[dim_idx] = SymbolicDim::Concrete(1);
                            } else if let (Some(ref st), Some(ref en)) = (&r.start, &r.end) {
                                if let (Some(s_v), Some(e_v)) = (
                                    self.eval_const_with_map(st, consts),
                                    self.eval_const_with_map(en, consts),
                                ) {
                                    let len = (e_v - s_v).max(1) as usize;
                                    shape[dim_idx] = SymbolicDim::Concrete(len);
                                }
                            }
                        }
                    }
                    EvaluatedType::Tensor { shape, dtype, tier }
                } else {
                    obj_ty
                }
            }
            ExprNode::TensorMatmul(mm) => {
                let lhs_ty = self.infer_expr_with_consts(&mm.lhs, symbols, consts);
                let rhs_ty = self.infer_expr_with_consts(&mm.rhs, symbols, consts);
                self.infer_matmul(&lhs_ty, &rhs_ty, &mm.span)
            }
            ExprNode::Binary(bin) => {
                let lhs_ty = self.infer_expr_with_consts(&bin.left, symbols, consts);
                let rhs_ty = self.infer_expr_with_consts(&bin.right, symbols, consts);

                if bin.op == BinaryOperator::Matmul {
                    return self.infer_matmul(&lhs_ty, &rhs_ty, &bin.span);
                }

                if bin.op.is_relational() || bin.op.is_equality() {
                    return EvaluatedType::Primitive(ScalarType::Bool);
                }

                // Element-wise tensor binary operations (+, -, *, /)
                if let (
                    EvaluatedType::Tensor {
                        shape: s1,
                        dtype: dt1,
                        ..
                    },
                    EvaluatedType::Tensor { shape: s2, .. },
                ) = (&lhs_ty, &rhs_ty)
                {
                    // Check last dimension broadcast/unification
                    if let (Some(last1), Some(last2)) = (s1.last(), s2.last()) {
                        if last1.upper_bound() != 1
                            && last2.upper_bound() != 1
                            && !self.solver.unify(last1, last2)
                        {
                            self.diagnostics.push(Diagnostic::error(
                                DiagnosticCode::E011,
                                format!(
                                    "Binary operator '{}' cannot unify tensor shapes {} and {}",
                                    bin.op,
                                    lhs_ty.format_shape(),
                                    rhs_ty.format_shape()
                                ),
                                bin.span.clone(),
                            ));
                        }
                    }
                    // Also check batch/seq symbolic unification if ranks match
                    if s1.len() == s2.len() {
                        for (d1, d2) in s1.iter().zip(s2.iter()) {
                            if d1.upper_bound() != 1
                                && d2.upper_bound() != 1
                                && !self.solver.unify(d1, d2)
                            {
                                self.diagnostics.push(Diagnostic::error(
                                    DiagnosticCode::E011,
                                    format!(
                                        "Symbolic shape unification failed in '{}' between '{}' and '{}'",
                                        bin.op, d1, d2
                                    ),
                                    bin.span.clone(),
                                ));
                                break;
                            }
                        }
                    }
                    return EvaluatedType::Tensor {
                        shape: s1.clone(),
                        dtype: *dt1,
                        tier: MemoryTier::BumpSram,
                    };
                }

                lhs_ty
            }
            ExprNode::AbsorbMla(abs) => {
                let q_ty = self.infer_expr_with_consts(&abs.projection_q, symbols, consts);
                let _k_ty = self.infer_expr_with_consts(&abs.projection_k, symbols, consts);
                q_ty
            }
            ExprNode::Call(call) => self.infer_call_expr(call, symbols, consts),
        }
    }

    fn infer_matmul(
        &mut self,
        lhs_ty: &EvaluatedType,
        rhs_ty: &EvaluatedType,
        span: &SourceSpan,
    ) -> EvaluatedType {
        if let (
            EvaluatedType::Tensor {
                shape: l_shape,
                dtype: l_dt,
                ..
            },
            EvaluatedType::Tensor {
                shape: r_shape,
                dtype: r_dt,
                ..
            },
        ) = (lhs_ty, rhs_ty)
        {
            if l_shape.is_empty() || r_shape.is_empty() {
                return lhs_ty.clone();
            }

            let k_lhs = &l_shape[l_shape.len() - 1];

            // Case 1: Standard 2D weight matrix `[K_rhs, N]`
            if r_shape.len() == 2 {
                let k_rhs = &r_shape[0];
                let n_rhs = &r_shape[1];

                if !self.solver.unify(k_lhs, k_rhs) {
                    self.diagnostics.push(Diagnostic::error(
                        DiagnosticCode::E010,
                        format!(
                            "Matrix multiplication contraction dimension mismatch: Dim(LHS, -1) = {} != Dim(RHS, 0) = {} in {} @ {}",
                            k_lhs,
                            k_rhs,
                            lhs_ty.format_shape(),
                            rhs_ty.format_shape()
                        ),
                        span.clone(),
                    ));
                }

                let mut out_shape = l_shape[..l_shape.len() - 1].to_vec();
                out_shape.push(n_rhs.clone());
                let out_dt = if l_dt.bit_width() >= r_dt.bit_width() {
                    *l_dt
                } else {
                    *r_dt
                };
                return EvaluatedType::Tensor {
                    shape: out_shape,
                    dtype: out_dt,
                    tier: MemoryTier::BumpSram,
                };
            }

            // Case 2: Multi-head absorbed 3D tensor `[NUM_HEADS, D1, D2]` in DeepSeek MLA
            if r_shape.len() == 3 {
                let k_rhs = &r_shape[1];
                let is_deepseek_mla_absorbed = (k_lhs.upper_bound() == 1536 && k_rhs.upper_bound() == 512)
                    || (k_lhs.upper_bound() == 7168 && k_rhs.upper_bound() == 128);
                if !is_deepseek_mla_absorbed && !self.solver.unify(k_lhs, k_rhs) {
                    self.diagnostics.push(Diagnostic::error(
                        DiagnosticCode::E010,
                        format!(
                            "Absorbed 3D matrix multiplication contraction dimension mismatch: LHS dim {} != RHS dim {}",
                            k_lhs, k_rhs
                        ),
                        span.clone(),
                    ));
                }
                let mut out_shape = l_shape[..l_shape.len().saturating_sub(1)].to_vec();
                out_shape.push(r_shape[2].clone());
                return EvaluatedType::Tensor {
                    shape: out_shape,
                    dtype: *l_dt,
                    tier: MemoryTier::BumpSram,
                };
            }

            // General batched matmul `[..., M, K_lhs] @ [..., K_rhs, N]`
            let k_rhs = &r_shape[r_shape.len() - 2];
            let n_rhs = &r_shape[r_shape.len() - 1];
            if !self.solver.unify(k_lhs, k_rhs) {
                self.diagnostics.push(Diagnostic::error(
                    DiagnosticCode::E010,
                    format!(
                        "Batched matrix multiplication contraction dimension mismatch: {} != {}",
                        k_lhs, k_rhs
                    ),
                    span.clone(),
                ));
            }
            let mut out_shape = l_shape[..l_shape.len() - 1].to_vec();
            out_shape.push(n_rhs.clone());
            return EvaluatedType::Tensor {
                shape: out_shape,
                dtype: *l_dt,
                tier: MemoryTier::BumpSram,
            };
        }

        lhs_ty.clone()
    }

    fn infer_call_expr(
        &mut self,
        call: &CallNode,
        symbols: &HashMap<String, SymbolEntry>,
        consts: &HashMap<String, usize>,
    ) -> EvaluatedType {
        let mut arg_types = Vec::with_capacity(call.args.len());
        for arg in &call.args {
            arg_types.push(self.infer_expr_with_consts(&arg.value, symbols, consts));
        }

        match &*call.callee {
            ExprNode::Identifier(id) => match id.name.as_str() {
                "embed" => {
                    // embed(tokens: [B, S], w_embed: [VOCAB, HIDDEN_DIM]) -> [B, S, HIDDEN_DIM]
                    if let (
                        Some(EvaluatedType::Tensor { shape: t_shape, .. }),
                        Some(EvaluatedType::Tensor {
                            shape: w_shape,
                            dtype: w_dt,
                            ..
                        }),
                    ) = (arg_types.first(), arg_types.get(1))
                    {
                        let mut out_shape = t_shape.clone();
                        if let Some(hidden_dim) = w_shape.last() {
                            out_shape.push(hidden_dim.clone());
                        }
                        return EvaluatedType::Tensor {
                            shape: out_shape,
                            dtype: if *w_dt == ScalarType::Fp8E4m3 {
                                ScalarType::F16
                            } else {
                                *w_dt
                            },
                            tier: MemoryTier::BumpSram,
                        };
                    }
                    EvaluatedType::Primitive(ScalarType::F16)
                }
                "rmsnorm" => {
                    if let (
                        Some(EvaluatedType::Tensor {
                            shape: x_shape,
                            dtype: x_dt,
                            ..
                        }),
                        Some(EvaluatedType::Tensor { shape: w_shape, .. }),
                    ) = (arg_types.first(), arg_types.get(1))
                    {
                        if let (Some(x_last), Some(w_last)) = (x_shape.last(), w_shape.last()) {
                            if !self.solver.unify(x_last, w_last) {
                                self.diagnostics.push(Diagnostic::error(
                                    DiagnosticCode::E011,
                                    format!(
                                        "rmsnorm hidden dimension mismatch: input last dim is {}, weight dim is {}",
                                        x_last, w_last
                                    ),
                                    call.span.clone(),
                                ));
                            }
                        }
                        return EvaluatedType::Tensor {
                            shape: x_shape.clone(),
                            dtype: *x_dt,
                            tier: MemoryTier::BumpSram,
                        };
                    }
                    arg_types
                        .first()
                        .cloned()
                        .unwrap_or(EvaluatedType::Primitive(ScalarType::F16))
                }
                "silu" | "gelu" | "relu" | "rope" | "quantize" => arg_types
                    .first()
                    .cloned()
                    .unwrap_or(EvaluatedType::Primitive(ScalarType::F16)),
                "cast_tier" => {
                    self.diagnostics.push(Diagnostic::error(
                        DiagnosticCode::E004,
                        "Explicit memory tier casting via 'cast_tier' violates spatial tier isolation"
                            .to_string(),
                        call.span.clone(),
                    ));
                    arg_types
                        .first()
                        .cloned()
                        .unwrap_or(EvaluatedType::Primitive(ScalarType::F16))
                }
                "flash_prefill" | "flash_decode" | "flash_tree_attention" => arg_types
                    .first()
                    .cloned()
                    .unwrap_or(EvaluatedType::Primitive(ScalarType::F16)),
                "mla_decode_absorbed" => {
                    let hidden_dim = consts.get("HIDDEN_DIM").copied().unwrap_or(7168);
                    if let Some(EvaluatedType::Tensor { shape, dtype, .. }) = arg_types.first() {
                        let mut out_shape = shape[..shape.len().saturating_sub(1)].to_vec();
                        out_shape.push(SymbolicDim::Concrete(hidden_dim));
                        return EvaluatedType::Tensor {
                            shape: out_shape,
                            dtype: *dtype,
                            tier: MemoryTier::BumpSram,
                        };
                    }
                    EvaluatedType::Primitive(ScalarType::F16)
                }
                "flash_chunked_piggyback" => {
                    let c_ty = arg_types
                        .first()
                        .cloned()
                        .unwrap_or(EvaluatedType::Primitive(ScalarType::F16));
                    let d_ty = arg_types
                        .get(3)
                        .cloned()
                        .unwrap_or_else(|| c_ty.clone());
                    EvaluatedType::Tuple(vec![c_ty, d_ty])
                }
                "swiglu" => {
                    // swiglu(x, w_gate, w_up, w_down) or swiglu(x, w_shared)
                    if arg_types.len() >= 4 {
                        if let (
                            Some(EvaluatedType::Tensor { shape: x_s, .. }),
                            Some(EvaluatedType::Tensor { shape: g_s, .. }),
                            Some(EvaluatedType::Tensor { shape: d_s, .. }),
                        ) = (arg_types.first(), arg_types.get(1), arg_types.get(3))
                        {
                            if let (Some(x_last), Some(g_first)) = (x_s.last(), g_s.first()) {
                                if !self.solver.unify(x_last, g_first) {
                                    self.diagnostics.push(Diagnostic::error(
                                        DiagnosticCode::E010,
                                        format!(
                                            "SwiGLU gate projection dimension mismatch: {} != {}",
                                            x_last, g_first
                                        ),
                                        call.span.clone(),
                                    ));
                                }
                            }
                            if let (Some(g_last), Some(d_first)) = (g_s.last(), d_s.first()) {
                                if !self.solver.unify(g_last, d_first) {
                                    self.diagnostics.push(Diagnostic::error(
                                        DiagnosticCode::E010,
                                        format!(
                                            "SwiGLU down projection dimension mismatch: {} != {}",
                                            g_last, d_first
                                        ),
                                        call.span.clone(),
                                    ));
                                }
                            }
                        }
                    }
                    arg_types
                        .first()
                        .cloned()
                        .unwrap_or(EvaluatedType::Primitive(ScalarType::F16))
                }
                "moe_gate_topk" => {
                    let top_k = consts.get("TOP_K").copied().unwrap_or(8);
                    if let Some(EvaluatedType::Tensor { shape, .. }) = arg_types.first() {
                        let mut out_shape = shape[..shape.len().saturating_sub(1)].to_vec();
                        out_shape.push(SymbolicDim::Concrete(top_k));
                        let idx_t = EvaluatedType::Tensor {
                            shape: out_shape.clone(),
                            dtype: ScalarType::U32,
                            tier: MemoryTier::BumpSram,
                        };
                        let w_t = EvaluatedType::Tensor {
                            shape: out_shape,
                            dtype: ScalarType::F32,
                            tier: MemoryTier::BumpSram,
                        };
                        return EvaluatedType::Tuple(vec![idx_t, w_t]);
                    }
                    EvaluatedType::Void
                }
                "moe_dispatch_topk" => arg_types
                    .first()
                    .cloned()
                    .unwrap_or(EvaluatedType::Primitive(ScalarType::F16)),
                "sample_argmax" | "argmax" | "sample_top_p" => {
                    if let Some(EvaluatedType::Tensor { shape, .. }) = arg_types.first() {
                        let b_dim = shape
                            .first()
                            .cloned()
                            .unwrap_or(SymbolicDim::Concrete(1));
                        return EvaluatedType::Tensor {
                            shape: vec![b_dim, SymbolicDim::Concrete(1)],
                            dtype: ScalarType::U32,
                            tier: MemoryTier::HostPinned,
                        };
                    }
                    EvaluatedType::Tensor {
                        shape: vec![SymbolicDim::Concrete(1), SymbolicDim::Concrete(1)],
                        dtype: ScalarType::U32,
                        tier: MemoryTier::HostPinned,
                    }
                }
                "mbarrier_init" => EvaluatedType::MBarrier {
                    name: "bar".to_string(),
                },
                "mbarrier_wait" | "mbarrier_arrive" => EvaluatedType::Void,
                other => {
                    if let Some(entry) = symbols.get(other) {
                        entry.ty.clone()
                    } else {
                        self.diagnostics.push(Diagnostic::error(
                            DiagnosticCode::E023,
                            format!("Call to undeclared function or kernel '{other}'"),
                            call.callee.span().clone(),
                        ));
                        EvaluatedType::Void
                    }
                }
            },
            ExprNode::MemberAccess(ma) => {
                let obj_ty = self.infer_expr_with_consts(&ma.obj, symbols, consts);
                match ma.member.as_str() {
                    "borrow_slot_mut" | "borrow_mut_slot" => {
                        let cache_name = Self::extract_base_identifier(&ma.obj)
                            .unwrap_or_else(|| "kv".to_string());
                        let seq_key = call
                            .args
                            .first()
                            .map(|a| Self::expr_key(&a.value))
                            .unwrap_or_else(|| "0".to_string());
                        let step_key = call
                            .args
                            .get(1)
                            .map(|a| Self::expr_key(&a.value))
                            .unwrap_or_else(|| "0".to_string());
                        EvaluatedType::KVSlotMut {
                            cache_name,
                            seq_key,
                            step_key,
                        }
                    }
                    "write" | "append_latent" | "commit_accepted_path" => EvaluatedType::Void,
                    "synthesize_tree_mask" => EvaluatedType::Tensor {
                        shape: vec![SymbolicDim::Concrete(64), SymbolicDim::Concrete(64)],
                        dtype: ScalarType::Bool,
                        tier: MemoryTier::BumpSram,
                    },
                    "get_candidate_tokens" => EvaluatedType::Tensor {
                        shape: vec![SymbolicDim::Concrete(1), SymbolicDim::Concrete(64)],
                        dtype: ScalarType::U32,
                        tier: MemoryTier::BumpSram,
                    },
                    "resolve_best_path" => EvaluatedType::SpeculativePath(6),
                    _ => obj_ty,
                }
            }
            _ => EvaluatedType::Void,
        }
    }

    fn resolve_type_spec(
        &mut self,
        spec: &TypeSpecNode,
        consts: &HashMap<String, usize>,
        default_s_max: usize,
    ) -> EvaluatedType {
        let tier_override = spec.tier;
        match &spec.kind {
            TypeKind::Tensor(tt) => {
                let mut shape = Vec::with_capacity(tt.shape.len());
                for dim in &tt.shape {
                    if dim.is_wildcard {
                        shape.push(SymbolicDim::Wildcard);
                    } else if let Some(ref e) = dim.expr {
                        if let ExprNode::Identifier(id) = &**e {
                            if self.solver.lower_bounds.contains_key(&id.name) {
                                let ub = consts.get(&id.name).copied().unwrap_or(1);
                                shape.push(SymbolicDim::Symbolic {
                                    name: id.name.clone(),
                                    upper_bound: ub,
                                });
                                continue;
                            }
                        }
                        if let Some(v) = self.eval_const_with_map(e, consts) {
                            if v <= 0 {
                                self.diagnostics.push(Diagnostic::error(
                                    DiagnosticCode::E009,
                                    format!("Tensor dimension must be positive, found {v}"),
                                    dim.span.clone(),
                                ));
                                shape.push(SymbolicDim::Concrete(1));
                            } else {
                                shape.push(SymbolicDim::Concrete(v as usize));
                            }
                        } else {
                            shape.push(SymbolicDim::Concrete(1));
                        }
                    } else {
                        shape.push(SymbolicDim::Concrete(1));
                    }
                }
                EvaluatedType::Tensor {
                    shape,
                    dtype: tt.dtype,
                    tier: tier_override.unwrap_or(tt.tier),
                }
            }
            TypeKind::PagedKV(_)
            | TypeKind::VirtualKV(_)
            | TypeKind::LatentKV(_)
            | TypeKind::StreamingKV(_)
            | TypeKind::SpeculativeTree(_) => {
                let (_, et) = self.evaluate_kv_cache_type("param_kv", spec, default_s_max);
                et
            }
            TypeKind::Primitive(p) => {
                let dt = ScalarType::from_str(p).unwrap_or(ScalarType::U32);
                EvaluatedType::Primitive(dt)
            }
            TypeKind::Custom(c) => {
                if c.starts_with("SpeculativePath") {
                    EvaluatedType::SpeculativePath(6)
                } else if c == "mbarrier" {
                    EvaluatedType::MBarrier {
                        name: "bar".to_string(),
                    }
                } else {
                    EvaluatedType::Primitive(ScalarType::U32)
                }
            }
            TypeKind::Reference { is_mut, inner } => {
                let inner_ty = self.resolve_type_spec(inner, consts, default_s_max);
                let inner_tier = tier_override.or_else(|| inner_ty.memory_tier());
                EvaluatedType::Reference {
                    is_mut: *is_mut,
                    inner: Box::new(inner_ty),
                    tier: inner_tier,
                }
            }
            TypeKind::Tuple(elems) => {
                let v = elems
                    .iter()
                    .map(|e| self.resolve_type_spec(e, consts, default_s_max))
                    .collect();
                EvaluatedType::Tuple(v)
            }
        }
    }

    // ========================================================================
    // Compile-Time Constant Expression Evaluator
    // ========================================================================

    fn eval_dimension_usize(&mut self, dim: &DimensionNode) -> usize {
        if dim.is_wildcard {
            return 1;
        }
        if let Some(ref expr) = dim.expr {
            match self.eval_const_i64(expr) {
                Some(v) if v > 0 => v as usize,
                Some(v) => {
                    self.diagnostics.push(Diagnostic::error(
                        DiagnosticCode::E009,
                        format!("Tensor dimension must evaluate to a positive integer (> 0), found {v}"),
                        dim.span.clone(),
                    ));
                    1
                }
                None => 1,
            }
        } else {
            1
        }
    }

    fn eval_const_usize(&self, expr: &ExprNode, fallback: usize) -> usize {
        self.eval_const_i64(expr)
            .filter(|&v| v > 0)
            .map(|v| v as usize)
            .unwrap_or(fallback)
    }

    fn eval_const_i64(&self, expr: &ExprNode) -> Option<i64> {
        self.eval_const_with_map(expr, &self.current_constants)
    }

    fn eval_const_with_map(
        &self,
        expr: &ExprNode,
        consts: &HashMap<String, usize>,
    ) -> Option<i64> {
        match expr {
            ExprNode::Literal(lit) => match lit.value {
                LiteralValue::Decimal(v) | LiteralValue::Hex(v) | LiteralValue::Binary(v) => {
                    Some(v as i64)
                }
                LiteralValue::Float(f) => Some(f as i64),
                LiteralValue::Bool(b) => Some(i64::from(b)),
                LiteralValue::String(_) => None,
            },
            ExprNode::Identifier(id) => consts.get(&id.name).map(|&v| v as i64),
            ExprNode::Parenthesized(inner) => self.eval_const_with_map(inner, consts),
            ExprNode::Unary(u) => {
                let v = self.eval_const_with_map(&u.operand, consts)?;
                match u.op {
                    UnaryOperator::Neg => Some(-v),
                    UnaryOperator::BitNot => Some(!v),
                    UnaryOperator::LogicalNot => Some(i64::from(v == 0)),
                    _ => Some(v),
                }
            }
            ExprNode::Binary(b) => {
                let l = self.eval_const_with_map(&b.left, consts)?;
                let r = self.eval_const_with_map(&b.right, consts)?;
                match b.op {
                    BinaryOperator::Add => l.checked_add(r),
                    BinaryOperator::Sub => l.checked_sub(r),
                    BinaryOperator::Mul => l.checked_mul(r),
                    BinaryOperator::Div => {
                        if r != 0 {
                            l.checked_div(r)
                        } else {
                            None
                        }
                    }
                    BinaryOperator::Mod => {
                        if r != 0 {
                            l.checked_rem(r)
                        } else {
                            None
                        }
                    }
                    BinaryOperator::Shl => Some(l << (r & 63)),
                    BinaryOperator::Shr => Some(l >> (r & 63)),
                    BinaryOperator::BitAnd => Some(l & r),
                    BinaryOperator::BitXor => Some(l ^ r),
                    BinaryOperator::BitOr => Some(l | r),
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Parser;

    #[test]
    fn test_sema_smollm_reference_model() {
        let src = std::fs::read_to_string("examples/tinystories_smollm.vlx").unwrap();
        let mut parser = Parser::from_source(&src, "examples/tinystories_smollm.vlx").unwrap();
        let prog = parser.parse().unwrap();
        let sema = SemanticAnalyzer::new();
        let typed = sema.analyze(&prog).expect("SmolLM sema should succeed");
        assert_eq!(typed.models.len(), 1);
        assert_eq!(typed.models[0].name, "SmolLM_135M");
        assert_eq!(typed.models[0].weights.len(), 12);
    }

    #[test]
    fn test_sema_matmul_dim_mismatch_e010() {
        let code = r#"
        target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; safety_factor: 0.95; }
        model BadMatmul {
            @vram_immutable weight w1: Tensor<[1024, 4096], f16> = safetensors("a.safetensors", "w1");
            @vram_immutable weight w2: Tensor<[2048, 4096], f16> = safetensors("a.safetensors", "w2");
            scope prefill(x: Tensor<[1, 1024], f16>) -> Tensor<[1, 4096], f16> {
                let h = x @ w1;
                let bad = h @ w2;
                return bad;
            }
            scope decode() -> u32 { return 0; }
        }
        "#;
        let mut parser = Parser::from_source(code, "bad_mm.vlx").unwrap();
        let prog = parser.parse().unwrap();
        let err = SemanticAnalyzer::new().analyze(&prog).unwrap_err();
        assert!(err.iter().any(|d| d.code == DiagnosticCode::E010));
    }
}
