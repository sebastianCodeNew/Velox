//! Velox Compiler Diagnostic Subsystem (`veloxc::diagnostics`)
//!
//! Formal implementation of all 25 compiler diagnostic codes (E001 through E025),
//! structured diagnostic records, boxed Rustc-grade formatter with visual carets,
//! and batch diagnostic engine.
//!
//! Zero external dependencies (`std` only).

use crate::types::SourceSpan;
use std::fmt;

// ============================================================================
// Severity & Diagnostic Codes (E001 - E025)
// ============================================================================

/// Diagnostic severity classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Severity {
    Error,
    Warning,
    Note,
}

impl Severity {
    /// Returns the canonical string representation.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Error => "ERROR",
            Self::Warning => "WARNING",
            Self::Note => "NOTE",
        }
    }
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// All 25 formal compiler diagnostic codes defined in the Velox Requirements Matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DiagnosticCode {
    E001, // DYNAMIC_HEAP_ALLOCATION_PROHIBITED
    E002, // TARGET_VRAM_BUDGET_EXCEEDED
    E003, // BLOCKING_HOST_SYNC_IN_DECODE
    E004, // ILLEGAL_MEMORY_TIER_CAST
    E005, // INVALID_DIRECT_DMA_PATH
    E006, // SRAM_REFERENCE_ESCAPES_SCOPE
    E007, // UNSUPPORTED_HARDWARE_TARGET
    E008, // MISSING_EXECUTION_SCOPE
    E009, // INVALID_CONST_GENERIC_PARAM
    E010, // TENSOR_MATMUL_DIM_MISMATCH
    E011, // SYMBOLIC_SHAPE_UNIFICATION_FAILURE
    E012, // KV_CACHE_BORROW_CONFLICT
    E013, // SAFETENSORS_KEY_NOT_FOUND
    E014, // SAFETENSORS_DTYPE_MISMATCH
    E015, // BLOCK_SIZE_NOT_POWER_OF_TWO
    E016, // UNBOUNDED_STREAMING_WINDOW
    E017, // SPECULATIVE_TREE_DEPTH_EXCEEDED
    E018, // UNABSORBED_MLA_DECOMPRESSION
    E019, // HOST_PINNED_BUFFER_OVERFLOW
    E020, // ASYNC_TMA_BARRIER_MISUSE
    E021, // NON_CONSTANT_LOOP_BOUND_IN_FIBER
    E022, // DUPLICATE_MODEL_MEMBER_NAME
    E023, // UNDEFINED_IDENTIFIER_REFERENCE
    E024, // REPRESENTATION_PRECISION_LOSS
    E025, // UNALIGNED_SCRATCHPAD_ACCESS
}

/// Array of all 25 diagnostic codes.
pub const ALL_CODES: [DiagnosticCode; 25] = [
    DiagnosticCode::E001,
    DiagnosticCode::E002,
    DiagnosticCode::E003,
    DiagnosticCode::E004,
    DiagnosticCode::E005,
    DiagnosticCode::E006,
    DiagnosticCode::E007,
    DiagnosticCode::E008,
    DiagnosticCode::E009,
    DiagnosticCode::E010,
    DiagnosticCode::E011,
    DiagnosticCode::E012,
    DiagnosticCode::E013,
    DiagnosticCode::E014,
    DiagnosticCode::E015,
    DiagnosticCode::E016,
    DiagnosticCode::E017,
    DiagnosticCode::E018,
    DiagnosticCode::E019,
    DiagnosticCode::E020,
    DiagnosticCode::E021,
    DiagnosticCode::E022,
    DiagnosticCode::E023,
    DiagnosticCode::E024,
    DiagnosticCode::E025,
];

impl DiagnosticCode {
    /// Returns the standard code string (e.g. "E001").
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::E001 => "E001",
            Self::E002 => "E002",
            Self::E003 => "E003",
            Self::E004 => "E004",
            Self::E005 => "E005",
            Self::E006 => "E006",
            Self::E007 => "E007",
            Self::E008 => "E008",
            Self::E009 => "E009",
            Self::E010 => "E010",
            Self::E011 => "E011",
            Self::E012 => "E012",
            Self::E013 => "E013",
            Self::E014 => "E014",
            Self::E015 => "E015",
            Self::E016 => "E016",
            Self::E017 => "E017",
            Self::E018 => "E018",
            Self::E019 => "E019",
            Self::E020 => "E020",
            Self::E021 => "E021",
            Self::E022 => "E022",
            Self::E023 => "E023",
            Self::E024 => "E024",
            Self::E025 => "E025",
        }
    }

    /// Returns the diagnostic formal uppercase identifier.
    #[must_use]
    pub const fn identifier(&self) -> &'static str {
        match self {
            Self::E001 => "DYNAMIC_HEAP_ALLOCATION_PROHIBITED",
            Self::E002 => "TARGET_VRAM_BUDGET_EXCEEDED",
            Self::E003 => "BLOCKING_HOST_SYNC_IN_DECODE",
            Self::E004 => "ILLEGAL_MEMORY_TIER_CAST",
            Self::E005 => "INVALID_DIRECT_DMA_PATH",
            Self::E006 => "SRAM_REFERENCE_ESCAPES_SCOPE",
            Self::E007 => "UNSUPPORTED_HARDWARE_TARGET",
            Self::E008 => "MISSING_EXECUTION_SCOPE",
            Self::E009 => "INVALID_CONST_GENERIC_PARAM",
            Self::E010 => "TENSOR_MATMUL_DIM_MISMATCH",
            Self::E011 => "SYMBOLIC_SHAPE_UNIFICATION_FAILURE",
            Self::E012 => "KV_CACHE_BORROW_CONFLICT",
            Self::E013 => "SAFETENSORS_KEY_NOT_FOUND",
            Self::E014 => "SAFETENSORS_DTYPE_MISMATCH",
            Self::E015 => "BLOCK_SIZE_NOT_POWER_OF_TWO",
            Self::E016 => "UNBOUNDED_STREAMING_WINDOW",
            Self::E017 => "SPECULATIVE_TREE_DEPTH_EXCEEDED",
            Self::E018 => "UNABSORBED_MLA_DECOMPRESSION",
            Self::E019 => "HOST_PINNED_BUFFER_OVERFLOW",
            Self::E020 => "ASYNC_TMA_BARRIER_MISUSE",
            Self::E021 => "NON_CONSTANT_LOOP_BOUND_IN_FIBER",
            Self::E022 => "DUPLICATE_MODEL_MEMBER_NAME",
            Self::E023 => "UNDEFINED_IDENTIFIER_REFERENCE",
            Self::E024 => "REPRESENTATION_PRECISION_LOSS",
            Self::E025 => "UNALIGNED_SCRATCHPAD_ACCESS",
        }
    }

    /// Returns the human-readable diagnostic title.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::E001 => "Dynamic Heap Allocation Prohibited",
            Self::E002 => "Target VRAM Budget Exceeded",
            Self::E003 => "Blocking Host Synchronization in Decode Scope",
            Self::E004 => "Illegal Memory Tier Cast",
            Self::E005 => "Invalid Direct DMA Path",
            Self::E006 => "Shared Memory (SRAM) Reference Escapes Scope",
            Self::E007 => "Unsupported Hardware Target",
            Self::E008 => "Missing Mandatory Execution Scope",
            Self::E009 => "Invalid Const-Generic Parameter",
            Self::E010 => "Tensor Matrix Multiplication Dimension Mismatch",
            Self::E011 => "Symbolic Shape Unification Failure",
            Self::E012 => "Key-Value Cache Borrow Conflict",
            Self::E013 => "Safetensors Key Not Found",
            Self::E014 => "Safetensors Data Type Mismatch",
            Self::E015 => "Block Size Not Power of Two",
            Self::E016 => "Unbounded Streaming Window",
            Self::E017 => "Speculative Tree Depth Exceeded",
            Self::E018 => "Unabsorbed MLA Decompression",
            Self::E019 => "Host-Pinned Buffer Overflow",
            Self::E020 => "Async TMA Barrier Misuse",
            Self::E021 => "Non-Constant Loop Bound in Fiber",
            Self::E022 => "Duplicate Model Member Name",
            Self::E023 => "Undefined Identifier Reference",
            Self::E024 => "Representation Precision Loss",
            Self::E025 => "Unaligned Scratchpad Access",
        }
    }

    /// Returns the architectural invariant category.
    #[must_use]
    pub const fn category(&self) -> &'static str {
        match self {
            Self::E001 => "Invariant 1 Violation",
            Self::E002 => "Invariant 2 Violation",
            Self::E003 => "Invariant 3 Violation",
            Self::E004 => "Memory Tier Violation",
            Self::E005 => "Memory Tier Violation",
            Self::E006 => "Lifetime & Aliasing",
            Self::E007 => "Target Configuration",
            Self::E008 => "Architecture Invariant",
            Self::E009 => "Const-Generics",
            Self::E010 => "Shape Verification",
            Self::E011 => "Shape Verification",
            Self::E012 => "Lifetime & Aliasing",
            Self::E013 => "Weight Binding",
            Self::E014 => "Weight Binding",
            Self::E015 => "KV Cache Architecture",
            Self::E016 => "KV Cache Architecture",
            Self::E017 => "Speculative Decoding",
            Self::E018 => "Performance Invariant",
            Self::E019 => "Memory Tier Violation",
            Self::E020 => "Concurrency & Async",
            Self::E021 => "Invariant 1 Violation",
            Self::E022 => "Grammar & AST",
            Self::E023 => "Frontend Semantics",
            Self::E024 => "Quantization & Dtype",
            Self::E025 => "Hardware Architecture",
        }
    }

    /// Returns formal invariant explanation text.
    #[must_use]
    pub const fn explanation(&self) -> &'static str {
        match self {
            Self::E001 => {
                "Velox enforces Invariant 1: Zero Dynamic Heap Allocation During Inference.\n\
                 All memory must be statically planned via the Velox Static Memory Planner or allocated\n\
                 within designated @bump_sram / @paged_vram storage classes. Calls to heap\n\
                 allocators (malloc, cudaMalloc, calloc, new) are strictly prohibited."
            }
            Self::E002 => {
                "Statically computed memory S_total exceeds the hardware VRAM capacity\n\
                 multiplied by the safety factor (alpha * VRAM_budget).\n\
                 Runtime out-of-memory is mathematically certain without model resizing or offloading."
            }
            Self::E003 => {
                "Velox enforces Invariant 3: Sub-80us Host Event Loop Dispatch.\n\
                 Synchronous device-to-host reads (.item(), .cpu(), cudaStreamSynchronize, cudaMemcpy)\n\
                 stall the CPU pipeline and destroy decoding throughput."
            }
            Self::E004 => {
                "Attempting to cast an immutable tier (@vram_immutable) to a mutable tier\n\
                 (@paged_vram) or between incompatible hardware memory domains."
            }
            Self::E005 => {
                "Initiating a DMA transfer between incompatible memory tiers without\n\
                 intermediate staging (e.g., direct @bump_sram to @host_pinned)."
            }
            Self::E006 => {
                "Memory allocated in @bump_sram resides in thread-block Shared Memory (SMEM).\n\
                 Returning a reference to SMEM across a function or kernel boundary violates\n\
                 the SRAM Escape Invariant. SMEM is discarded at kernel exit."
            }
            Self::E007 => {
                "Declaring a hardware compute capability not supported by the code generator\n\
                 (e.g., SM < 7.0 / pre-Volta architectures lack Tensor Cores)."
            }
            Self::E008 => {
                "Model declarations must define both `prefill` (compute-bound) and\n\
                 `decode` (memory-bound) phase-disjoint execution scopes."
            }
            Self::E009 => {
                "Const-generic parameters must evaluate to positive integer or valid\n\
                 scalar dimension constants."
            }
            Self::E010 => {
                "Inner contraction dimensions must unify in matrix multiplication:\n\
                 Dim(A, -1) == Dim(B, 0)."
            }
            Self::E011 => {
                "Symbolic dimension constraints cannot be satisfied across the operator DAG."
            }
            Self::E012 => {
                "Velox enforces Axiom 1 (Disjoint Borrow Rule). Two overlapping mutable\n\
                 borrows of the same KV cache block/slot cannot coexist simultaneously\n\
                 within the same execution block."
            }
            Self::E013 => {
                "The specified weight tensor identifier was not found in the Safetensors file header."
            }
            Self::E014 => {
                "The data type declared in the Velox model differs from the storage precision\n\
                 in the Safetensors file without an explicit quantizer."
            }
            Self::E015 => {
                "PagedKVCache block size must be a power-of-two (16, 32, or 64) for hardware\n\
                 warp alignment and continuous batching efficiency."
            }
            Self::E016 => {
                "StreamingSinkKVCache requires positive, constant sink and window parameters\n\
                 to guarantee bounded memory footprint."
            }
            Self::E017 => {
                "Candidate speculative token verification tree depth exceeds declared MAX_DEPTH limit."
            }
            Self::E018 => {
                "Materializing full H x D key/value tensors from a LatentKVCache instead of\n\
                 using absorbed projections wastes memory bandwidth."
            }
            Self::E019 => {
                "Attempting to push more tokens into the @host_pinned input ring buffer\n\
                 than allocated capacity."
            }
            Self::E020 => {
                "Issuing `tma_load_async` without an associated hardware `mbarrier`\n\
                 synchronization leads to race conditions."
            }
            Self::E021 => {
                "Using dynamically sized loop bounds inside the token dispatch fiber without\n\
                 static bounds violates deterministic latency guarantees."
            }
            Self::E022 => {
                "Declaring two weights, KV caches, scopes, or helper functions with identical\n\
                 identifiers inside a model."
            }
            Self::E023 => {
                "Referencing a variable, weight, KV cache, or parameter identifier that has\n\
                 not been declared in the current lexical scope."
            }
            Self::E024 => {
                "Implicitly downcasting floating-point precision (e.g. f32 to fp8) without\n\
                 an explicit quantizer operator."
            }
            Self::E025 => {
                "Emitting memory access that violates 128-byte hardware transaction alignment\n\
                 on global VRAM."
            }
        }
    }

    /// Returns deterministic remediation guidance.
    #[must_use]
    pub const fn remediation(&self) -> &'static str {
        match self {
            Self::E001 => {
                "Declare a statically shaped local tensor instead:\n  \
                 `let temp: @bump_sram Tensor<[1, 1024], f16>;`"
            }
            Self::E002 => {
                "Reduce batch size (B_MAX), context length (S_MAX), or enable KV cache\n\
                 compression/quantization."
            }
            Self::E003 => {
                "Use asynchronous DMA transfers or non-blocking event-driven synchronization."
            }
            Self::E004 => {
                "Allocate a mutable buffer in the destination tier and copy the contents explicitly."
            }
            Self::E005 => {
                "Stage data through global VRAM (@paged_vram or intermediate device memory)\n\
                 before transferring across the PCIe bus."
            }
            Self::E006 => {
                "Pass the destination buffer as an argument or commit data to @paged_vram\n\
                 using `tma_store_async` before returning."
            }
            Self::E007 => {
                "Target a device with compute capability 7.0 or higher\n\
                 (e.g., `compute_capability: 8.0;` or `8.6;`)."
            }
            Self::E008 => {
                "Add the missing `scope prefill(...) { ... }` or `scope decode(...) { ... }`\n\
                 definition to the model."
            }
            Self::E009 => {
                "Ensure const-generic parameters are declared with positive integer types\n\
                 (e.g., `const B: usize = 32;`)."
            }
            Self::E010 => {
                "Verify matrix dimensions and transpose operands if necessary."
            }
            Self::E011 => {
                "Ensure tensor shape expressions have consistent dimension symbols across operations."
            }
            Self::E012 => {
                "Ensure slot indices are distinct or serialize mutations sequentially."
            }
            Self::E013 => {
                "Check tensor keys in the .safetensors metadata and ensure exact naming matches."
            }
            Self::E014 => {
                "Update the declared TensorType dtype or specify an explicit quantizer operator."
            }
            Self::E015 => {
                "Change block size parameter to 16, 32, or 64\n\
                 (e.g. `PagedKVCache<NUM_LAYERS, NUM_HEADS, HEAD_DIM, 16, f16>;`)."
            }
            Self::E016 => {
                "Specify constant positive integers for sinks (>= 1) and sliding window (>= 64)."
            }
            Self::E017 => {
                "Prune candidate DAG to fit within declared MAX_DEPTH or increase SpeculativeTreeBlock capacity."
            }
            Self::E018 => {
                "Use absorbed projections (`absorb_mla_q`, `absorb_mla_v`) and call `mla_decode_absorbed`."
            }
            Self::E019 => {
                "Increase ring buffer capacity or throttle request ingestion rate."
            }
            Self::E020 => {
                "Provide an mbarrier token to `tma_load_async(src, dst, mbarrier)`\n\
                 and arrive/wait on the barrier."
            }
            Self::E021 => {
                "Bound loops with compile-time const generics or declare static trip-count bounds."
            }
            Self::E022 => {
                "Rename duplicate members to have unique identifiers within the model."
            }
            Self::E023 => {
                "Declare the identifier before referencing it or verify spelling."
            }
            Self::E024 => {
                "Use an explicit quantizer or scaling function to downcast precision safely."
            }
            Self::E025 => {
                "Ensure tensor dimension offsets and strides align to 128-byte cache line boundaries."
            }
        }
    }

    /// Parses a string representation (e.g. "E001" or "e001") into a `DiagnosticCode`.
    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(val: &str) -> Option<Self> {
        match val.trim().to_ascii_uppercase().as_str() {
            "E001" => Some(Self::E001),
            "E002" => Some(Self::E002),
            "E003" => Some(Self::E003),
            "E004" => Some(Self::E004),
            "E005" => Some(Self::E005),
            "E006" => Some(Self::E006),
            "E007" => Some(Self::E007),
            "E008" => Some(Self::E008),
            "E009" => Some(Self::E009),
            "E010" => Some(Self::E010),
            "E011" => Some(Self::E011),
            "E012" => Some(Self::E012),
            "E013" => Some(Self::E013),
            "E014" => Some(Self::E014),
            "E015" => Some(Self::E015),
            "E016" => Some(Self::E016),
            "E017" => Some(Self::E017),
            "E018" => Some(Self::E018),
            "E019" => Some(Self::E019),
            "E020" => Some(Self::E020),
            "E021" => Some(Self::E021),
            "E022" => Some(Self::E022),
            "E023" => Some(Self::E023),
            "E024" => Some(Self::E024),
            "E025" => Some(Self::E025),
            _ => None,
        }
    }
}

impl std::str::FromStr for DiagnosticCode {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_str(s).ok_or(())
    }
}

impl fmt::Display for DiagnosticCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Static catalog entry metadata for a diagnostic code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiagnosticInfo {
    pub code: DiagnosticCode,
    pub identifier: &'static str,
    pub name: &'static str,
    pub category: &'static str,
    pub default_explanation: &'static str,
    pub default_remediation: &'static str,
}

impl DiagnosticCode {
    /// Returns the complete catalog entry metadata for this code.
    #[must_use]
    pub fn info(&self) -> DiagnosticInfo {
        DiagnosticInfo {
            code: *self,
            identifier: self.identifier(),
            name: self.name(),
            category: self.category(),
            default_explanation: self.explanation(),
            default_remediation: self.remediation(),
        }
    }
}

// ============================================================================
// Diagnostic Record Struct
// ============================================================================

/// Compiler diagnostic record holding code, message, span coordinates,
/// optional hint, notes, secondary spans, and severity.
#[derive(Debug, Clone, PartialEq)]
pub struct Diagnostic {
    pub code: DiagnosticCode,
    pub message: String,
    pub span: SourceSpan,
    pub hint: Option<String>,
    pub notes: Vec<String>,
    pub secondary_spans: Vec<(SourceSpan, String)>,
    pub severity: Severity,
}

impl Diagnostic {
    /// Constructs a new Diagnostic with required fields.
    #[must_use]
    pub fn new(
        code: DiagnosticCode,
        message: impl Into<String>,
        span: SourceSpan,
        severity: Severity,
    ) -> Self {
        Self {
            code,
            message: message.into(),
            span,
            hint: None,
            notes: Vec::new(),
            secondary_spans: Vec::new(),
            severity,
        }
    }

    /// Constructs an ERROR diagnostic.
    #[must_use]
    pub fn error(code: DiagnosticCode, message: impl Into<String>, span: SourceSpan) -> Self {
        Self::new(code, message, span, Severity::Error)
    }

    /// Constructs a WARNING diagnostic.
    #[must_use]
    pub fn warning(code: DiagnosticCode, message: impl Into<String>, span: SourceSpan) -> Self {
        Self::new(code, message, span, Severity::Warning)
    }

    /// Attaches remediation or clarifying hint.
    #[must_use]
    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }

    /// Attaches an additional contextual note.
    #[must_use]
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    /// Attaches a secondary source span pointer with explanation.
    #[must_use]
    pub fn with_secondary_span(mut self, span: SourceSpan, note: impl Into<String>) -> Self {
        self.secondary_spans.push((span, note.into()));
        self
    }

    /// Returns the diagnostic code string representation (e.g. "E001").
    #[inline]
    #[must_use]
    pub fn code_str(&self) -> &'static str {
        self.code.as_str()
    }

    /// Returns the human-readable diagnostic title.
    #[inline]
    #[must_use]
    pub fn name(&self) -> &'static str {
        self.code.name()
    }

    /// Returns the architectural category.
    #[inline]
    #[must_use]
    pub fn category(&self) -> &'static str {
        self.code.category()
    }

    /// Returns formal invariant explanation.
    #[inline]
    #[must_use]
    pub fn explanation(&self) -> &'static str {
        self.code.explanation()
    }

    /// Returns deterministic remediation guidance.
    #[inline]
    #[must_use]
    pub fn remediation(&self) -> &'static str {
        self.code.remediation()
    }
}

// ============================================================================
// Boxed Rustc-Grade Formatter
// ============================================================================

/// Boxed Rustc-grade diagnostic formatter for terminal rendering.
/// Renders an 80-character border, header `[Exxx]: NAME`, `--> file:line:col`,
/// source excerpts with line numbers, visual carets `^^^^`, secondary pointers,
/// architectural invariant explanations, and deterministic remediation guidance.
#[derive(Debug, Clone)]
pub struct DiagnosticFormatter {
    pub use_color: bool,
}

impl Default for DiagnosticFormatter {
    fn default() -> Self {
        Self::new(true)
    }
}

impl DiagnosticFormatter {
    /// Constructs a new formatter with specified ANSI color support.
    #[must_use]
    pub const fn new(use_color: bool) -> Self {
        Self { use_color }
    }

    /// Formats a diagnostic record against the source text.
    #[must_use]
    pub fn format(&self, diag: &Diagnostic, source_text: &str) -> String {
        let (red, yellow, cyan, bold, reset) = if self.use_color {
            (
                "\x1b[1;31m",
                "\x1b[1;33m",
                "\x1b[1;36m",
                "\x1b[1m",
                "\x1b[0m",
            )
        } else {
            ("", "", "", "", "")
        };

        let color = match diag.severity {
            Severity::Error => red,
            Severity::Warning => yellow,
            Severity::Note => cyan,
        };

        let code_str = diag.code.as_str();
        let diag_name = diag.name();
        let border = "=".repeat(80);

        let mut out = Vec::new();
        out.push(border.clone());
        out.push(format!(
            "{color}VELOX COMPILER {} [{code_str}]: {diag_name}{reset}",
            diag.severity.as_str()
        ));
        out.push(border.clone());

        let file_desc = if diag.span.file.is_empty() {
            "<source>"
        } else {
            &diag.span.file
        };

        let loc = diag.span.start;
        out.push(format!(
            "File: {file_desc}, Line {}, Column {}:",
            loc.line, loc.column
        ));
        out.push(format!("  --> {file_desc}:{}:{}", loc.line, loc.column));

        // Source context lines: previous, current, next
        let lines: Vec<&str> = source_text.lines().collect();
        let line_idx = loc.line.saturating_sub(1);
        let col_idx = loc.column.saturating_sub(1);

        if !lines.is_empty() && line_idx < lines.len() {
            let start_line = line_idx.saturating_sub(1);
            let end_line = (line_idx + 2).min(lines.len());

            for (i, line_content) in lines.iter().enumerate().take(end_line).skip(start_line) {
                let line_no = i + 1;
                if i == line_idx {
                    out.push(format!("{color}> {line_no:4} | {reset}{line_content}"));

                    // Compute visual caret length
                    let mut caret_len = 1;
                    if diag.span.end.line == diag.span.start.line
                        && diag.span.end.offset > diag.span.start.offset
                    {
                        caret_len = diag.span.end.offset - diag.span.start.offset;
                    } else if diag.span.end.column > diag.span.start.column {
                        caret_len = diag.span.end.column - diag.span.start.column;
                    }
                    if caret_len == 0 {
                        caret_len = 1;
                    }

                    // "> 142 | " prefix is 8 chars
                    let prefix_len = 8;
                    let indent = " ".repeat(prefix_len + col_idx);
                    let carets = "^".repeat(caret_len);
                    out.push(format!("{color}{indent}{carets}{reset}"));
                    out.push(format!("{color}{indent}{}{reset}", diag.message));
                } else {
                    out.push(format!("  {line_no:4} | {line_content}"));
                }
            }
        }

        // Secondary spans
        for (sec_span, sec_msg) in &diag.secondary_spans {
            let sec_loc = sec_span.start;
            out.push(format!(
                "  Also noted at Line {}, Column {}: {sec_msg}",
                sec_loc.line, sec_loc.column
            ));
        }

        // Additional notes
        for note in &diag.notes {
            out.push(format!("  Note: {note}"));
        }

        // Hint if specified
        if let Some(hint) = &diag.hint {
            out.push(format!("  Hint: {hint}"));
        }

        // Invariant Explanation
        let explanation = diag.explanation();
        if !explanation.is_empty() {
            out.push(String::new());
            out.push(format!("{bold}Explanation:{reset}"));
            for expl_line in explanation.lines() {
                out.push(format!("  {expl_line}"));
            }
        }

        // Remediation
        let remediation = diag.remediation();
        if !remediation.is_empty() {
            out.push(String::new());
            out.push(format!("{cyan}Remediation:{reset}"));
            for rem_line in remediation.lines() {
                out.push(format!("  {rem_line}"));
            }
        }

        out.push(border);
        out.join("\n")
    }
}

// ============================================================================
// Diagnostic Engine
// ============================================================================

/// Diagnostic registry managing emission, filtering, and batch rendering.
#[derive(Debug, Clone)]
pub struct DiagnosticEngine {
    pub source_text: String,
    pub file_path: String,
    pub formatter: DiagnosticFormatter,
    pub diagnostics: Vec<Diagnostic>,
}

impl DiagnosticEngine {
    /// Constructs a new DiagnosticEngine.
    #[must_use]
    pub fn new(source_text: impl Into<String>, file_path: impl Into<String>, use_color: bool) -> Self {
        Self {
            source_text: source_text.into(),
            file_path: file_path.into(),
            formatter: DiagnosticFormatter::new(use_color),
            diagnostics: Vec::new(),
        }
    }

    /// Records an existing diagnostic.
    pub fn emit(&mut self, diag: Diagnostic) {
        self.diagnostics.push(diag);
    }

    /// Emits an ERROR severity diagnostic.
    pub fn emit_error(
        &mut self,
        code: DiagnosticCode,
        message: impl Into<String>,
        span: SourceSpan,
    ) {
        self.diagnostics.push(Diagnostic::error(code, message, span));
    }

    /// Emits a WARNING severity diagnostic.
    pub fn emit_warning(
        &mut self,
        code: DiagnosticCode,
        message: impl Into<String>,
        span: SourceSpan,
    ) {
        self.diagnostics.push(Diagnostic::warning(code, message, span));
    }

    /// Returns `true` if any ERROR severity diagnostic has been recorded.
    #[must_use]
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|d| d.severity == Severity::Error)
    }

    /// Returns the count of ERROR diagnostics.
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Error)
            .count()
    }

    /// Formats all recorded diagnostics with the engine's stored source text.
    #[must_use]
    pub fn format_all(&self) -> String {
        self.format_with_source(&self.source_text)
    }

    /// Formats all diagnostics with a given source text buffer.
    #[must_use]
    pub fn format_with_source(&self, source_text: &str) -> String {
        self.diagnostics
            .iter()
            .map(|d| self.formatter.format(d, source_text))
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_25_codes_metadata_completeness() {
        assert_eq!(ALL_CODES.len(), 25);
        for code in &ALL_CODES {
            assert!(!code.as_str().is_empty());
            assert!(!code.identifier().is_empty());
            assert!(!code.name().is_empty());
            assert!(!code.category().is_empty());
            assert!(!code.explanation().is_empty());
            assert!(!code.remediation().is_empty());
            assert_eq!(DiagnosticCode::from_str(code.as_str()), Some(*code));
        }
    }

    #[test]
    fn test_diagnostic_formatting_plain() {
        let sample = "target hardware { device: \"sm_86\"; }\nmodel Example {\n    let bad_call = malloc(1024);\n}\n";
        let span = SourceSpan::from_coords("example.vlx", 3, 20, 3, 32, 55, 67);
        let diag = Diagnostic::error(
            DiagnosticCode::E001,
            "Dynamic allocation call prohibited",
            span,
        );

        let formatter = DiagnosticFormatter::new(false);
        let rendered = formatter.format(&diag, sample);

        assert!(rendered.contains("================================================================================"));
        assert!(rendered.contains("VELOX COMPILER ERROR [E001]: Dynamic Heap Allocation Prohibited"));
        assert!(rendered.contains("Line 3, Column 20"));
        assert!(rendered.contains("let bad_call = malloc(1024);"));
        assert!(rendered.contains("^^^^^^^^^^^^"));
        assert!(rendered.contains("Explanation:"));
        assert!(rendered.contains("Remediation:"));
        assert!(!rendered.contains("\x1b["));
    }

    #[test]
    fn test_diagnostic_formatting_with_ansi_color() {
        let sample = "scope decode() { let x = 1; }";
        let span = SourceSpan::from_coords("s.vlx", 1, 1, 1, 5, 0, 4);
        let diag = Diagnostic::error(DiagnosticCode::E003, "Blocking call", span);

        let formatter = DiagnosticFormatter::new(true);
        let rendered = formatter.format(&diag, sample);
        assert!(rendered.contains("\x1b[1;31m"));
        assert!(rendered.contains("\x1b[0m"));
    }

    #[test]
    fn test_diagnostic_engine() {
        let mut engine = DiagnosticEngine::new("test source", "test.vlx", false);
        assert!(!engine.has_errors());
        assert_eq!(engine.error_count(), 0);

        let span = SourceSpan::synthetic("test.vlx");
        engine.emit_warning(DiagnosticCode::E024, "Precision loss", span.clone());
        assert!(!engine.has_errors());
        assert_eq!(engine.error_count(), 0);
        assert_eq!(engine.diagnostics.len(), 1);

        engine.emit_error(DiagnosticCode::E001, "Alloc prohibited", span);
        assert!(engine.has_errors());
        assert_eq!(engine.error_count(), 1);
        assert_eq!(engine.diagnostics.len(), 2);

        let formatted = engine.format_all();
        assert!(formatted.contains("[E024]"));
        assert!(formatted.contains("[E001]"));
    }
}
