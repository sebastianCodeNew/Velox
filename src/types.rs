//! Velox Compiler Types (`veloxc::types`)
//!
//! Core type definitions, physical memory tiers, numeric scalar precisions,
//! execution scope kinds, operator hierarchies, and source span tracking primitives
//! for the Velox domain-specific systems programming language.
//!
//! Zero external dependencies (`std` only).

use std::fmt;

// ============================================================================
// Source Coordinate & Span Tracking
// ============================================================================

/// Exact 1-indexed source coordinate in a `.vlx` translation unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct SourceLocation {
    /// 1-indexed line number.
    pub line: usize,
    /// 1-indexed column number (tab expanded to 4-space tab stops).
    pub column: usize,
    /// 0-indexed byte character offset into the source stream.
    pub offset: usize,
}

impl SourceLocation {
    /// Constructs a new `SourceLocation`.
    #[inline]
    #[must_use]
    pub const fn new(line: usize, column: usize, offset: usize) -> Self {
        Self { line, column, offset }
    }

    /// Alias for column coordinate.
    #[inline]
    #[must_use]
    pub const fn col(&self) -> usize {
        self.column
    }
}

impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.line, self.column)
    }
}

/// Source text range bounding an AST node or token for compiler diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct SourceSpan {
    /// Source file path or logical identifier.
    pub file: String,
    /// Start coordinate (inclusive).
    pub start: SourceLocation,
    /// End coordinate (exclusive).
    pub end: SourceLocation,
}

impl SourceSpan {
    /// Constructs a new `SourceSpan`.
    #[must_use]
    pub fn new(file: impl Into<String>, start: SourceLocation, end: SourceLocation) -> Self {
        Self {
            file: file.into(),
            start,
            end,
        }
    }

    /// Constructs a synthetic span for compiler-generated or internal AST nodes.
    #[must_use]
    pub fn synthetic(file: &str) -> Self {
        let loc = SourceLocation::new(0, 0, 0);
        Self {
            file: file.to_string(),
            start: loc,
            end: loc,
        }
    }

    /// Constructs a span from explicit coordinate components.
    #[must_use]
    pub fn from_coords(
        file: &str,
        start_line: usize,
        start_col: usize,
        end_line: usize,
        end_col: usize,
        start_offset: usize,
        end_offset: usize,
    ) -> Self {
        Self {
            file: file.to_string(),
            start: SourceLocation::new(start_line, start_col, start_offset),
            end: SourceLocation::new(end_line, end_col, end_offset),
        }
    }

    /// Merges two spans into a single enclosing span from self.start to other.end.
    #[must_use]
    pub fn merge(&self, other: &SourceSpan) -> Self {
        let file = if !self.file.is_empty() {
            self.file.clone()
        } else {
            other.file.clone()
        };
        Self {
            file,
            start: self.start,
            end: other.end,
        }
    }
}

impl fmt::Display for SourceSpan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let prefix = if self.file.is_empty() {
            String::new()
        } else {
            format!("{}:", self.file)
        };
        write!(
            f,
            "[{}{}:{} -> {}:{}]",
            prefix, self.start.line, self.start.column, self.end.line, self.end.column
        )
    }
}

// ============================================================================
// Physical Memory Tiers
// ============================================================================

/// Physical hardware memory tier qualification.
/// Enforces compile-time spatial safety across heterogeneous device memories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryTier {
    /// Static model weights residing permanently in immutable High Bandwidth Memory (`@vram_immutable`).
    VramImmutable,
    /// Physical block-table allocated Key-Value cache pages in HBM (`@paged_vram`).
    PagedVram,
    /// On-chip fast thread-block scratchpad memory / Shared Memory (`@bump_sram`).
    BumpSram,
    /// Lock-free PCIe DMA-accessible pinned host system RAM (`@host_pinned`).
    HostPinned,
    /// Strictly bounded physical memory limits for guaranteed OOM prevention (`@wired_limit_bounded`).
    WiredLimitBounded,
}

impl MemoryTier {
    /// Returns the canonical decorator string representation.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::VramImmutable => "@vram_immutable",
            Self::PagedVram => "@paged_vram",
            Self::BumpSram => "@bump_sram",
            Self::HostPinned => "@host_pinned",
            Self::WiredLimitBounded => "@wired_limit_bounded",
        }
    }

    /// Parses a memory tier string qualifier or decorator name.
    #[must_use]
    pub fn from_str(val: &str) -> Option<Self> {
        let clean = val.trim();
        let stripped = clean.strip_prefix('@').unwrap_or(clean);
        match stripped {
            "vram_immutable" => Some(Self::VramImmutable),
            "paged_vram" => Some(Self::PagedVram),
            "bump_sram" => Some(Self::BumpSram),
            "host_pinned" => Some(Self::HostPinned),
            "wired_limit_bounded" => Some(Self::WiredLimitBounded),
            _ => None,
        }
    }

    /// Returns `true` if this tier resides in on-device memory (VRAM or SRAM).
    #[inline]
    #[must_use]
    pub const fn is_device_tier(&self) -> bool {
        matches!(self, Self::VramImmutable | Self::PagedVram | Self::BumpSram)
    }

    /// Returns `true` if writes to this tier are forbidden at compile time.
    #[inline]
    #[must_use]
    pub const fn is_immutable(&self) -> bool {
        matches!(self, Self::VramImmutable)
    }
}

impl fmt::Display for MemoryTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ============================================================================
// Scalar Precision Types
// ============================================================================

/// Numeric scalar precision types supported by Velox and modern Tensor Cores.
/// Covers standard IEEE 754, Brain Floating Point, OCP Microscaling FP8, and integers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarType {
    F32,
    F16,
    Bf16,
    Fp8E4m3,
    Fp8E5m2,
    Fp8E8m0,
    Int8,
    Int4,
    U64,
    U32,
    U16,
    U8,
    I64,
    I32,
    I16,
    I8,
    Usize,
    Bool,
}

impl ScalarType {
    /// Returns the canonical string representation.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::Fp8E4m3 => "fp8_e4m3",
            Self::Fp8E5m2 => "fp8_e5m2",
            Self::Fp8E8m0 => "fp8_e8m0",
            Self::Int8 => "int8",
            Self::Int4 => "int4",
            Self::U64 => "u64",
            Self::U32 => "u32",
            Self::U16 => "u16",
            Self::U8 => "u8",
            Self::I64 => "i64",
            Self::I32 => "i32",
            Self::I16 => "i16",
            Self::I8 => "i8",
            Self::Usize => "usize",
            Self::Bool => "bool",
        }
    }

    /// Parses a scalar type string.
    #[must_use]
    pub fn from_str(val: &str) -> Option<Self> {
        match val.trim().to_ascii_lowercase().as_str() {
            "f32" => Some(Self::F32),
            "f16" => Some(Self::F16),
            "bf16" => Some(Self::Bf16),
            "fp8_e4m3" => Some(Self::Fp8E4m3),
            "fp8_e5m2" => Some(Self::Fp8E5m2),
            "fp8_e8m0" => Some(Self::Fp8E8m0),
            "int8" => Some(Self::Int8),
            "int4" => Some(Self::Int4),
            "u64" => Some(Self::U64),
            "u32" => Some(Self::U32),
            "u16" => Some(Self::U16),
            "u8" => Some(Self::U8),
            "i64" => Some(Self::I64),
            "i32" => Some(Self::I32),
            "i16" => Some(Self::I16),
            "i8" => Some(Self::I8),
            "usize" => Some(Self::Usize),
            "bool" => Some(Self::Bool),
            _ => None,
        }
    }

    /// Returns the element byte size (may be fractional for sub-byte types like int4).
    #[must_use]
    pub const fn byte_size(&self) -> f64 {
        match self {
            Self::F32 | Self::U32 | Self::I32 => 4.0,
            Self::F16 | Self::Bf16 | Self::U16 | Self::I16 => 2.0,
            Self::Fp8E4m3
            | Self::Fp8E5m2
            | Self::Fp8E8m0
            | Self::Int8
            | Self::U8
            | Self::I8
            | Self::Bool => 1.0,
            Self::Int4 => 0.5,
            Self::U64 | Self::I64 | Self::Usize => 8.0,
        }
    }

    /// Returns the element bit width.
    #[must_use]
    pub const fn bit_width(&self) -> usize {
        match self {
            Self::Int4 => 4,
            Self::Fp8E4m3
            | Self::Fp8E5m2
            | Self::Fp8E8m0
            | Self::Int8
            | Self::U8
            | Self::I8
            | Self::Bool => 8,
            Self::F16 | Self::Bf16 | Self::U16 | Self::I16 => 16,
            Self::F32 | Self::U32 | Self::I32 => 32,
            Self::U64 | Self::I64 | Self::Usize => 64,
        }
    }

    /// Returns `true` if this type represents a floating point precision.
    #[inline]
    #[must_use]
    pub const fn is_floating_point(&self) -> bool {
        matches!(
            self,
            Self::F32 | Self::F16 | Self::Bf16 | Self::Fp8E4m3 | Self::Fp8E5m2 | Self::Fp8E8m0
        )
    }

    /// Returns `true` if this type represents an integer or index type.
    #[inline]
    #[must_use]
    pub const fn is_integer(&self) -> bool {
        matches!(
            self,
            Self::Int8
                | Self::Int4
                | Self::U64
                | Self::U32
                | Self::U16
                | Self::U8
                | Self::I64
                | Self::I32
                | Self::I16
                | Self::I8
                | Self::Usize
        )
    }
}

impl fmt::Display for ScalarType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ============================================================================
// Execution Scope Kinds
// ============================================================================

/// Phase-disjoint serving execution scopes.
/// Velox models prefill (compute-bound) and decode (memory-bound) as disjoint phases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScopeKind {
    /// Compute-bound prompt evaluation phase.
    Prefill,
    /// Memory-bandwidth-bound single-token generation phase.
    Decode,
    /// Interleaved chunked prefill serving scope.
    ChunkedPrefill,
    /// Speculative tree candidate token verification scope.
    SpeculativeVerify,
}

impl ScopeKind {
    /// Returns the canonical scope kind string.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::ChunkedPrefill => "chunked_prefill",
            Self::SpeculativeVerify => "speculative_verify",
        }
    }

    /// Parses a scope kind identifier.
    #[must_use]
    pub fn from_str(val: &str) -> Option<Self> {
        match val.trim().to_ascii_lowercase().as_str() {
            "prefill" => Some(Self::Prefill),
            "decode" => Some(Self::Decode),
            "chunked_prefill" => Some(Self::ChunkedPrefill),
            "speculative_verify" => Some(Self::SpeculativeVerify),
            _ => None,
        }
    }

    /// Returns `true` if this scope executes prompt ingestion / prefill.
    #[inline]
    #[must_use]
    pub const fn is_prefill_phase(&self) -> bool {
        matches!(self, Self::Prefill | Self::ChunkedPrefill)
    }

    /// Returns `true` if this scope executes token generation / decode.
    #[inline]
    #[must_use]
    pub const fn is_decode_phase(&self) -> bool {
        matches!(self, Self::Decode | Self::SpeculativeVerify)
    }
}

impl fmt::Display for ScopeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ============================================================================
// Key-Value Cache Kinds
// ============================================================================

/// First-class native Key-Value cache architectures in Velox.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KVCacheKind {
    /// Physical page-table backed Key-Value cache (PagedAttention).
    Paged,
    /// CUDA Driver VMM-backed 2MB huge-page virtual memory KV cache.
    VirtualContiguous,
    /// DeepSeek Multi-Head Latent Attention compressed vector cache.
    Latent,
    /// StreamingLLM attention sink and sliding window cache.
    StreamingSink,
    /// Speculative tree verification candidate DAG buffer.
    SpeculativeTree,
}

impl KVCacheKind {
    /// Returns canonical type name.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Paged => "PagedKVCache",
            Self::VirtualContiguous => "VirtualContiguousKV",
            Self::Latent => "LatentKVCache",
            Self::StreamingSink => "StreamingSinkKVCache",
            Self::SpeculativeTree => "SpeculativeTreeBlock",
        }
    }

    /// Parses a KV cache type identifier.
    #[must_use]
    pub fn from_str(val: &str) -> Option<Self> {
        match val.trim() {
            "PagedKVCache" | "paged_kv" => Some(Self::Paged),
            "VirtualContiguousKV" | "virtual_kv" => Some(Self::VirtualContiguous),
            "LatentKVCache" | "latent_kv" => Some(Self::Latent),
            "StreamingSinkKVCache" | "streaming_sink" => Some(Self::StreamingSink),
            "SpeculativeTreeBlock" | "tree_block" => Some(Self::SpeculativeTree),
            _ => None,
        }
    }
}

impl fmt::Display for KVCacheKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ============================================================================
// Operators & Associativity
// ============================================================================

/// Operator associativity direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Associativity {
    Left,
    Right,
}

/// In-place mutation and assignment operators (Precedence Level 14).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssignmentOperator {
    /// Direct assignment (`=`).
    Assign,
    /// Add and assign (`+=`).
    AddAssign,
    /// Subtract and assign (`-=`).
    SubAssign,
    /// Multiply and assign (`*=`).
    MulAssign,
    /// Divide and assign (`/=`).
    DivAssign,
}

impl AssignmentOperator {
    /// Returns symbol string.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Assign => "=",
            Self::AddAssign => "+=",
            Self::SubAssign => "-=",
            Self::MulAssign => "*=",
            Self::DivAssign => "/=",
        }
    }

    /// Parses an assignment operator symbol.
    #[must_use]
    pub fn from_str(val: &str) -> Option<Self> {
        match val.trim() {
            "=" => Some(Self::Assign),
            "+=" => Some(Self::AddAssign),
            "-=" => Some(Self::SubAssign),
            "*=" => Some(Self::MulAssign),
            "/=" => Some(Self::DivAssign),
            _ => None,
        }
    }
}

impl fmt::Display for AssignmentOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Binary expression operators across precedence levels 3 through 13.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOperator {
    Matmul,     // @   (Precedence 3)
    Mul,        // *   (Precedence 4)
    Div,        // /   (Precedence 4)
    Mod,        // %   (Precedence 4)
    Add,        // +   (Precedence 5)
    Sub,        // -   (Precedence 5)
    Shl,        // <<  (Precedence 6)
    Shr,        // >>  (Precedence 6)
    Lt,         // <   (Precedence 7)
    Lte,        // <=  (Precedence 7)
    Gt,         // >   (Precedence 7)
    Gte,        // >=  (Precedence 7)
    Eq,         // ==  (Precedence 8)
    Neq,        // !=  (Precedence 8)
    BitAnd,     // &   (Precedence 9)
    BitXor,     // ^   (Precedence 10)
    BitOr,      // |   (Precedence 11)
    LogicalAnd, // &&  (Precedence 12)
    LogicalOr,  // ||  (Precedence 13)
}

impl BinaryOperator {
    /// Returns symbol string.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Matmul => "@",
            Self::Mul => "*",
            Self::Div => "/",
            Self::Mod => "%",
            Self::Add => "+",
            Self::Sub => "-",
            Self::Shl => "<<",
            Self::Shr => ">>",
            Self::Lt => "<",
            Self::Lte => "<=",
            Self::Gt => ">",
            Self::Gte => ">=",
            Self::Eq => "==",
            Self::Neq => "!=",
            Self::BitAnd => "&",
            Self::BitXor => "^",
            Self::BitOr => "|",
            Self::LogicalAnd => "&&",
            Self::LogicalOr => "||",
        }
    }

    /// Parses a binary operator symbol.
    #[must_use]
    pub fn from_str(val: &str) -> Option<Self> {
        match val.trim() {
            "@" => Some(Self::Matmul),
            "*" => Some(Self::Mul),
            "/" => Some(Self::Div),
            "%" => Some(Self::Mod),
            "+" => Some(Self::Add),
            "-" => Some(Self::Sub),
            "<<" => Some(Self::Shl),
            ">>" => Some(Self::Shr),
            "<" => Some(Self::Lt),
            "<=" => Some(Self::Lte),
            ">" => Some(Self::Gt),
            ">=" => Some(Self::Gte),
            "==" => Some(Self::Eq),
            "!=" => Some(Self::Neq),
            "&" => Some(Self::BitAnd),
            "^" => Some(Self::BitXor),
            "|" => Some(Self::BitOr),
            "&&" => Some(Self::LogicalAnd),
            "||" => Some(Self::LogicalOr),
            _ => None,
        }
    }

    /// Returns formal precedence level (3 = highest binary, 13 = lowest binary).
    #[must_use]
    pub const fn precedence_level(&self) -> usize {
        match self {
            Self::Matmul => 3,
            Self::Mul | Self::Div | Self::Mod => 4,
            Self::Add | Self::Sub => 5,
            Self::Shl | Self::Shr => 6,
            Self::Lt | Self::Lte | Self::Gt | Self::Gte => 7,
            Self::Eq | Self::Neq => 8,
            Self::BitAnd => 9,
            Self::BitXor => 10,
            Self::BitOr => 11,
            Self::LogicalAnd => 12,
            Self::LogicalOr => 13,
        }
    }

    /// Returns Pratt binding powers `(left_bp, right_bp)` for left-associative parsing.
    #[must_use]
    pub const fn binding_powers(&self) -> (u8, u8) {
        match self {
            Self::Matmul => (120, 121),
            Self::Mul | Self::Div | Self::Mod => (110, 111),
            Self::Add | Self::Sub => (100, 101),
            Self::Shl | Self::Shr => (90, 91),
            Self::Lt | Self::Lte | Self::Gt | Self::Gte => (80, 81),
            Self::Eq | Self::Neq => (70, 71),
            Self::BitAnd => (60, 61),
            Self::BitXor => (50, 51),
            Self::BitOr => (40, 41),
            Self::LogicalAnd => (30, 31),
            Self::LogicalOr => (20, 21),
        }
    }

    /// Returns `true` for inequality comparison operators.
    #[inline]
    #[must_use]
    pub const fn is_relational(&self) -> bool {
        matches!(self, Self::Lt | Self::Lte | Self::Gt | Self::Gte)
    }

    /// Returns `true` for equality comparison operators.
    #[inline]
    #[must_use]
    pub const fn is_equality(&self) -> bool {
        matches!(self, Self::Eq | Self::Neq)
    }

    /// Returns `true` for standard arithmetic binary operators.
    #[inline]
    #[must_use]
    pub const fn is_arithmetic(&self) -> bool {
        matches!(
            self,
            Self::Add | Self::Sub | Self::Mul | Self::Div | Self::Mod | Self::Matmul
        )
    }
}

impl fmt::Display for BinaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Unary prefix operators (Precedence Level 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOperator {
    /// Logical NOT (`!`).
    LogicalNot,
    /// Bitwise NOT (`~`).
    BitNot,
    /// Arithmetic negation (`-`).
    Neg,
    /// Dereference (`*`).
    Deref,
    /// Immutable borrow (`&`).
    Borrow,
    /// Mutable borrow (`&mut`).
    BorrowMut,
}

impl UnaryOperator {
    /// Returns symbol string.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::LogicalNot => "!",
            Self::BitNot => "~",
            Self::Neg => "-",
            Self::Deref => "*",
            Self::Borrow => "&",
            Self::BorrowMut => "&mut",
        }
    }

    /// Parses a unary operator symbol.
    #[must_use]
    pub fn from_str(val: &str) -> Option<Self> {
        match val.trim() {
            "!" => Some(Self::LogicalNot),
            "~" => Some(Self::BitNot),
            "-" => Some(Self::Neg),
            "*" => Some(Self::Deref),
            "&" => Some(Self::Borrow),
            "&mut" => Some(Self::BorrowMut),
            _ => None,
        }
    }
}

impl fmt::Display for UnaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ============================================================================
// Hardware Memory Transfer & Offline Transformation Kinds
// ============================================================================

/// Hardware Direct Memory Access transfer direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DmaDirection {
    /// Host to Device DMA transfer (`dma_async_h2d`).
    H2D,
    /// Device to Host DMA transfer (`dma_async_d2h`).
    D2H,
}

impl DmaDirection {
    /// Returns function keyword string.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::H2D => "dma_async_h2d",
            Self::D2H => "dma_async_d2h",
        }
    }

    /// Parses a DMA direction function keyword.
    #[must_use]
    pub fn from_str(val: &str) -> Option<Self> {
        match val.trim() {
            "dma_async_h2d" => Some(Self::H2D),
            "dma_async_d2h" => Some(Self::D2H),
            _ => None,
        }
    }
}

impl fmt::Display for DmaDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Multi-Head Latent Attention offline weight absorption kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AbsorbKind {
    /// Pre-absorbs Key projection matrix into Query projection (`absorb_mla_q`).
    AbsorbMlaQ,
    /// Pre-absorbs Value projection matrix into Output projection (`absorb_mla_v`).
    AbsorbMlaV,
}

impl AbsorbKind {
    /// Returns function keyword string.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::AbsorbMlaQ => "absorb_mla_q",
            Self::AbsorbMlaV => "absorb_mla_v",
        }
    }

    /// Parses an absorption kind function keyword.
    #[must_use]
    pub fn from_str(val: &str) -> Option<Self> {
        match val.trim() {
            "absorb_mla_q" => Some(Self::AbsorbMlaQ),
            "absorb_mla_v" => Some(Self::AbsorbMlaV),
            _ => None,
        }
    }
}

impl fmt::Display for AbsorbKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Literal token value categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LiteralType {
    Decimal,
    Hex,
    Binary,
    Float,
    String,
    Bool,
}

impl LiteralType {
    /// Returns category name string.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Decimal => "decimal",
            Self::Hex => "hex",
            Self::Binary => "binary",
            Self::Float => "float",
            Self::String => "string",
            Self::Bool => "bool",
        }
    }
}

impl fmt::Display for LiteralType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ============================================================================
// Memory Multiplier Constants
// ============================================================================

/// Calculates total bytes given a base integer value and optional memory unit.
///
/// Supported binary byte multipliers:
/// - `KiB`: 1024
/// - `MiB`: 1024^2 (1,048,576)
/// - `GiB`: 1024^3 (1,073,741,824)
/// - `TiB`: 1024^4 (1,099,511,627,776)
///
/// Also supports standard decimal units:
/// - `KB`: 1,000
/// - `MB`: 1,000,000
/// - `GB`: 1,000,000,000
/// - `TB`: 1,000,000,000,000
pub fn parse_memory_capacity(base_value: u64, unit_token: Option<&str>) -> Result<u64, String> {
    let Some(unit) = unit_token else {
        return Ok(base_value);
    };

    let clean = unit.trim_start_matches('*').trim();
    let multiplier: u64 = match clean {
        "KiB" => 1024,
        "MiB" => 1024 * 1024,
        "GiB" => 1024 * 1024 * 1024,
        "TiB" => 1024 * 1024 * 1024 * 1024,
        "KB" => 1000,
        "MB" => 1_000_000,
        "GB" => 1_000_000_000,
        "TB" => 1_000_000_000_000,
        _ => return Err(format!("Unknown memory capacity unit: '{unit}'")),
    };

    base_value
        .checked_mul(multiplier)
        .ok_or_else(|| format!("Memory capacity overflow: {base_value} * {multiplier}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_tier_parsing() {
        assert_eq!(MemoryTier::from_str("@vram_immutable"), Some(MemoryTier::VramImmutable));
        assert_eq!(MemoryTier::from_str("vram_immutable"), Some(MemoryTier::VramImmutable));
        assert_eq!(MemoryTier::from_str("@paged_vram"), Some(MemoryTier::PagedVram));
        assert_eq!(MemoryTier::from_str("@bump_sram"), Some(MemoryTier::BumpSram));
        assert_eq!(MemoryTier::from_str("@host_pinned"), Some(MemoryTier::HostPinned));
        assert_eq!(MemoryTier::from_str("@wired_limit_bounded"), Some(MemoryTier::WiredLimitBounded));
        assert_eq!(MemoryTier::from_str("unknown"), None);

        assert!(MemoryTier::VramImmutable.is_device_tier());
        assert!(MemoryTier::VramImmutable.is_immutable());
        assert!(!MemoryTier::PagedVram.is_immutable());
    }

    #[test]
    fn test_scalar_type_sizes() {
        assert_eq!(ScalarType::F32.byte_size(), 4.0);
        assert_eq!(ScalarType::F16.byte_size(), 2.0);
        assert_eq!(ScalarType::Fp8E4m3.byte_size(), 1.0);
        assert_eq!(ScalarType::Int4.byte_size(), 0.5);
        assert_eq!(ScalarType::Usize.byte_size(), 8.0);

        assert!(ScalarType::F16.is_floating_point());
        assert!(!ScalarType::U32.is_floating_point());
        assert!(ScalarType::I32.is_integer());
    }

    #[test]
    fn test_memory_multipliers() {
        assert_eq!(parse_memory_capacity(4096, Some("MiB")).unwrap(), 4096 * 1024 * 1024);
        assert_eq!(parse_memory_capacity(80, Some("* GiB")).unwrap(), 80 * 1024 * 1024 * 1024);
        assert_eq!(parse_memory_capacity(100, None).unwrap(), 100);
        assert!(parse_memory_capacity(1, Some("invalid")).is_err());
    }
}
