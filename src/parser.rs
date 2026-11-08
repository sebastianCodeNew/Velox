//! Velox Hand-Written Recursive Descent LL(k) Parser (`veloxc::parser`)
//!
//! Implements all 81 ISO/IEC 14977 formal EBNF production rules for the Velox
//! domain-specific language and optimizing compiler frontend.
//! Features a 14-level Pratt Precedence Climbing expression engine with left-associative
//! Tensor Core matrix multiplication (`@`) at Level 3, panic-mode error recovery
//! across statement (`;`) and block (`}`) synchronization barriers, and early
//! architectural invariant diagnostics (E001–E025).
//!
//! Zero external dependencies (`std` only).

use crate::ast::*;
use crate::diagnostics::{Diagnostic, DiagnosticCode, Severity};
use crate::lexer::{LexError, Lexer};
use crate::tokens::{Token, TokenType};
use crate::types::{
    AbsorbKind, AssignmentOperator, BinaryOperator, DmaDirection, MemoryTier, ScalarType,
    ScopeKind, SourceSpan, UnaryOperator,
};
use std::fmt;

/// Raised internally to initiate panic-mode error recovery.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub message: String,
    pub span: SourceSpan,
}

impl ParseError {
    #[must_use]
    pub fn new(message: impl Into<String>, span: SourceSpan) -> Self {
        Self {
            message: message.into(),
            span,
        }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ParseError: {} at {}", self.message, self.span)
    }
}

impl std::error::Error for ParseError {}

/// Pratt parsing binding powers for binary operators: `(left_bp, right_bp)`.
/// Derived from the 14-level operator precedence hierarchy.
#[inline]
#[must_use]
pub fn binary_binding_power(token_type: &TokenType) -> Option<(u8, u8)> {
    match token_type {
        TokenType::At => Some((120, 121)),         // Precedence 3: Matmul @ (Left-assoc)
        TokenType::Star | TokenType::Slash | TokenType::Percent => Some((110, 111)), // Precedence 4: Multiplicative
        TokenType::Plus | TokenType::Minus => Some((100, 101)),     // Precedence 5: Additive
        TokenType::Shl | TokenType::Shr => Some((90, 91)),         // Precedence 6: Bitwise Shifts
        TokenType::Lt | TokenType::LtEq | TokenType::Gt | TokenType::GtEq => Some((80, 81)), // Precedence 7: Relational
        TokenType::EqEq | TokenType::BangEq => Some((70, 71)),     // Precedence 8: Equality
        TokenType::Amp => Some((60, 61)),                          // Precedence 9: Bitwise AND
        TokenType::Caret => Some((50, 51)),                        // Precedence 10: Bitwise XOR
        TokenType::Pipe => Some((40, 41)),                         // Precedence 11: Bitwise OR
        TokenType::AmpAmp => Some((30, 31)),                       // Precedence 12: Logical AND
        TokenType::PipePipe => Some((20, 21)),                     // Precedence 13: Logical OR
        _ => None,
    }
}

/// Recursive descent LL(k) parser with Pratt Precedence Climbing.
pub struct Parser<'a> {
    tokens: Vec<Token>,
    cursor: usize,
    file_path: String,
    source_text: &'a str,
    pub diagnostics: Vec<Diagnostic>,
    in_panic_mode: bool,
}

impl<'a> Parser<'a> {
    /// Constructs a new Parser instance.
    #[must_use]
    pub fn new(tokens: Vec<Token>, file_path: impl Into<String>, source_text: &'a str) -> Self {
        Self {
            tokens,
            cursor: 0,
            file_path: file_path.into(),
            source_text,
            diagnostics: Vec::new(),
            in_panic_mode: false,
        }
    }

    /// Convenience constructor initializing Lexer and Parser together.
    pub fn from_source(source_text: &'a str, file_path: impl Into<String>) -> Result<Self, LexError> {
        let path = file_path.into();
        let tokens = Lexer::tokenize_source(source_text, &path)?;
        Ok(Self::new(tokens, path, source_text))
    }

    /// Returns the source text slice passed to the parser.
    #[inline]
    #[must_use]
    pub fn source_text(&self) -> &'a str {
        self.source_text
    }

    /// Returns the source file path.
    #[inline]
    #[must_use]
    pub fn file_path(&self) -> &str {
        &self.file_path
    }

    // -------------------------------------------------------------------------
    // Cursor & Lookahead Navigation
    // -------------------------------------------------------------------------

    #[inline]
    #[must_use]
    pub fn current(&self) -> &Token {
        if self.cursor >= self.tokens.len() {
            &self.tokens[self.tokens.len() - 1]
        } else {
            &self.tokens[self.cursor]
        }
    }

    #[inline]
    #[must_use]
    pub fn peek(&self, k: usize) -> &Token {
        let idx = self.cursor + k;
        if idx >= self.tokens.len() {
            &self.tokens[self.tokens.len() - 1]
        } else {
            &self.tokens[idx]
        }
    }

    #[inline]
    #[must_use]
    pub fn is_at_end(&self) -> bool {
        matches!(self.current().token_type, TokenType::Eof)
    }

    pub fn advance(&mut self) -> &Token {
        let idx = self.cursor;
        if !self.is_at_end() {
            self.cursor += 1;
        }
        &self.tokens[idx]
    }

    #[inline]
    #[must_use]
    pub fn check(&self, token_type: &TokenType) -> bool {
        if self.is_at_end() {
            return matches!(token_type, TokenType::Eof);
        }
        std::mem::discriminant(&self.current().token_type) == std::mem::discriminant(token_type)
    }

    pub fn match_token(&mut self, token_type: &TokenType) -> bool {
        if self.check(token_type) {
            self.advance();
            true
        } else {
            false
        }
    }

    pub fn consume(
        &mut self,
        expected: TokenType,
        error_msg: &str,
        code: DiagnosticCode,
    ) -> Result<Token, ParseError> {
        if self.check(&expected) {
            Ok(self.advance().clone())
        } else {
            let tok = self.current().clone();
            if !self.in_panic_mode {
                let msg = format!("{error_msg}. Found '{}' instead.", tok.text);
                self.emit_diagnostic(Diagnostic::error(code, msg, tok.span.clone()));
                self.in_panic_mode = true;
            }
            Err(ParseError::new(error_msg, tok.span))
        }
    }

    pub fn emit_diagnostic(&mut self, diag: Diagnostic) {
        self.diagnostics.push(diag);
    }

    /// Discards tokens until reaching a synchronization barrier.
    pub fn synchronize(&mut self) {
        self.in_panic_mode = false;
        if !self.is_at_end() {
            self.advance();
        }
        while !self.is_at_end() {
            match self.current().token_type {
                TokenType::Semicolon => {
                    self.advance();
                    return;
                }
                TokenType::RBrace
                | TokenType::Model
                | TokenType::Target
                | TokenType::Scope
                | TokenType::Weight
                | TokenType::Fn
                | TokenType::Let
                | TokenType::If
                | TokenType::For
                | TokenType::While
                | TokenType::Return
                | TokenType::Yield => {
                    return;
                }
                _ => {
                    self.advance();
                }
            }
        }
    }

    /// Parses an identifier or contextual keyword used as an identifier.
    pub fn parse_identifier_token(&mut self, error_msg: &str) -> Result<Token, ParseError> {
        let tok = self.current().clone();
        match &tok.token_type {
            TokenType::Identifier(_) => {
                self.advance();
                Ok(tok)
            }
            TokenType::Prefill
            | TokenType::Decode
            | TokenType::ChunkedPrefill
            | TokenType::SpeculativeVerify
            | TokenType::Weight
            | TokenType::Tensor
            | TokenType::PagedKVCache
            | TokenType::VirtualContiguousKV
            | TokenType::LatentKVCache
            | TokenType::StreamingSinkKVCache
            | TokenType::SpeculativeTreeBlock => {
                self.advance();
                Ok(tok)
            }
            _ => self.consume(
                TokenType::Identifier(String::new()),
                error_msg,
                DiagnosticCode::E023,
            ),
        }
    }

    // -------------------------------------------------------------------------
    // Public Entry Points
    // -------------------------------------------------------------------------

    /// Parses the token stream into a validated AST, returning error diagnostics on failure.
    pub fn parse(&mut self) -> Result<ProgramNode, Vec<Diagnostic>> {
        let program = self.parse_program();
        self.validate_ast(&program);
        if self.diagnostics.iter().any(|d| d.severity == Severity::Error) {
            Err(self.diagnostics.clone())
        } else {
            Ok(program)
        }
    }

    /// Parses the token stream with panic-mode recovery, always returning an AST if possible.
    pub fn parse_program_safe(&mut self) -> Option<ProgramNode> {
        let program = self.parse_program();
        self.validate_ast(&program);
        Some(program)
    }

    // -------------------------------------------------------------------------
    // Rule 1: Program Unit Root
    // -------------------------------------------------------------------------

    pub fn parse_program(&mut self) -> ProgramNode {
        let start_loc = self.current().span.start;
        let mut imports = Vec::new();
        let mut target_hardware = None;
        let mut models = Vec::new();
        let mut global_bindings = Vec::new();

        // 1. Module Imports
        while matches!(self.current().token_type, TokenType::Import) {
            let prev_cursor = self.cursor;
            match self.parse_import() {
                Ok(imp) => imports.push(imp),
                Err(_) => {
                    self.synchronize();
                    if self.cursor == prev_cursor && !self.is_at_end() {
                        self.advance();
                    }
                }
            }
        }

        // 2. Hardware Target Block
        if matches!(self.current().token_type, TokenType::Target) {
            let prev_cursor = self.cursor;
            match self.parse_target_hardware() {
                Ok(hw) => target_hardware = Some(hw),
                Err(_) => {
                    self.synchronize();
                    if self.cursor == prev_cursor && !self.is_at_end() {
                        self.advance();
                    }
                }
            }
        }

        // 3. Models and Global Declarations
        while !self.is_at_end() {
            let prev_cursor = self.cursor;
            if matches!(self.current().token_type, TokenType::Model) {
                match self.parse_model() {
                    Ok(m) => models.push(m),
                    Err(_) => {
                        self.synchronize();
                        if self.cursor == prev_cursor && !self.is_at_end() {
                            self.advance();
                        }
                    }
                }
            } else if matches!(self.current().token_type, TokenType::Let) {
                match self.parse_let() {
                    Ok(let_stmt) => global_bindings.push(let_stmt),
                    Err(_) => {
                        self.synchronize();
                        if self.cursor == prev_cursor && !self.is_at_end() {
                            self.advance();
                        }
                    }
                }
            } else {
                let tok = self.current().clone();
                if !self.in_panic_mode {
                    self.emit_diagnostic(Diagnostic::error(
                        DiagnosticCode::E023,
                        format!("Unexpected top-level token '{}'.", tok.text),
                        tok.span.clone(),
                    ));
                    self.in_panic_mode = true;
                }
                self.synchronize();
                if self.cursor == prev_cursor && !self.is_at_end() {
                    self.advance();
                }
            }
        }

        let end_loc = self.current().span.end;
        let span = SourceSpan::new(self.file_path.clone(), start_loc, end_loc);

        ProgramNode {
            imports,
            target_hardware,
            model_declarations: models,
            global_bindings,
            span,
        }
    }

    // -------------------------------------------------------------------------
    // Rule 2: Module Imports
    // -------------------------------------------------------------------------

    pub fn parse_import(&mut self) -> Result<ImportNode, ParseError> {
        let import_tok = self.consume(TokenType::Import, "Expected 'import'", DiagnosticCode::E023)?;
        let path_tok = self.consume(
            TokenType::String(String::new()),
            "Expected module path string literal after 'import'",
            DiagnosticCode::E023,
        )?;
        let semi_tok = self.consume(
            TokenType::Semicolon,
            "Expected ';' after import statement",
            DiagnosticCode::E023,
        )?;

        let path = match path_tok.token_type {
            TokenType::String(s) => s,
            _ => path_tok.text,
        };

        let span = import_tok.span.merge(&semi_tok.span);
        Ok(ImportNode::new(path, span))
    }

    // -------------------------------------------------------------------------
    // Rules 3–9: Hardware Target Envelope
    // -------------------------------------------------------------------------

    pub fn parse_target_hardware(&mut self) -> Result<TargetHardwareNode, ParseError> {
        let target_tok = self.consume(TokenType::Target, "Expected 'target'", DiagnosticCode::E007)?;
        self.consume(TokenType::Hardware, "Expected 'hardware' after 'target'", DiagnosticCode::E007)?;
        self.consume(TokenType::LBrace, "Expected '{' opening target hardware block", DiagnosticCode::E007)?;

        let mut device_name = String::new();
        let mut vram_bytes: u64 = 0;
        let mut compute_capability = (8, 0);
        let mut safety_factor = 0.95;

        while !self.check(&TokenType::RBrace) && !self.is_at_end() {
            if self.match_token(&TokenType::Device) {
                self.consume(TokenType::Colon, "Expected ':' after 'device'", DiagnosticCode::E007)?;
                let tok = self.consume(
                    TokenType::String(String::new()),
                    "Expected string literal for device name",
                    DiagnosticCode::E007,
                )?;
                device_name = match tok.token_type {
                    TokenType::String(s) => s,
                    _ => tok.text,
                };
                self.consume(TokenType::Semicolon, "Expected ';' after device property", DiagnosticCode::E007)?;
            } else if self.match_token(&TokenType::VramCapacity) {
                self.consume(TokenType::Colon, "Expected ':' after 'vram_capacity'", DiagnosticCode::E007)?;
                let expr = self.parse_expression(0, true)?;
                vram_bytes = self.eval_const_u64(&expr);
                self.consume(TokenType::Semicolon, "Expected ';' after vram_capacity property", DiagnosticCode::E007)?;
            } else if self.match_token(&TokenType::ComputeCapability) {
                self.consume(TokenType::Colon, "Expected ':' after 'compute_capability'", DiagnosticCode::E007)?;
                let cap_tok = self.current().clone();
                if matches!(cap_tok.token_type, TokenType::Float(_) | TokenType::Decimal(_)) {
                    self.advance();
                    let raw_val: f64 = cap_tok.text.parse().unwrap_or(8.0);
                    let major = raw_val.floor() as u32;
                    let minor = ((raw_val - (major as f64)) * 10.0).round() as u32;
                    compute_capability = (major, minor);
                } else {
                    self.consume(
                        TokenType::Float(0.0),
                        "Expected compute capability float literal (e.g. 8.6)",
                        DiagnosticCode::E007,
                    )?;
                }
                self.consume(TokenType::Semicolon, "Expected ';' after compute_capability property", DiagnosticCode::E007)?;
            } else if self.match_token(&TokenType::SafetyFactor) {
                self.consume(TokenType::Colon, "Expected ':' after 'safety_factor'", DiagnosticCode::E007)?;
                let factor_tok = self.current().clone();
                if matches!(factor_tok.token_type, TokenType::Float(_) | TokenType::Decimal(_)) {
                    self.advance();
                    safety_factor = factor_tok.text.parse().unwrap_or(0.95);
                } else {
                    self.consume(
                        TokenType::Float(0.0),
                        "Expected safety factor float literal (e.g. 0.95)",
                        DiagnosticCode::E007,
                    )?;
                }
                self.consume(TokenType::Semicolon, "Expected ';' after safety_factor property", DiagnosticCode::E007)?;
            } else {
                let tok = self.current().clone();
                self.emit_diagnostic(Diagnostic::error(
                    DiagnosticCode::E007,
                    format!("Unknown target hardware property '{}'.", tok.text),
                    tok.span.clone(),
                ));
                self.advance();
            }
        }

        let close_tok = self.consume(
            TokenType::RBrace,
            "Expected '}' closing target hardware block",
            DiagnosticCode::E007,
        )?;
        let span = target_tok.span.merge(&close_tok.span);

        // Invariant check E007: SM compute capability must be >= 7.0
        if compute_capability.0 < 7 {
            self.emit_diagnostic(Diagnostic::error(
                DiagnosticCode::E007,
                format!(
                    "Compute capability {}.{} is unsupported (requires SM 7.0+).",
                    compute_capability.0, compute_capability.1
                ),
                span.clone(),
            ));
        }

        Ok(TargetHardwareNode::new(
            device_name,
            vram_bytes,
            compute_capability,
            safety_factor,
            span,
        ))
    }

    /// Evaluates a constant integer expression or capacity multiplier at compile time.
    pub fn eval_const_u64(&self, expr: &ExprNode) -> u64 {
        match expr {
            ExprNode::Literal(lit) => match lit.value {
                LiteralValue::Decimal(v) | LiteralValue::Hex(v) | LiteralValue::Binary(v) => v,
                LiteralValue::Float(f) if f >= 0.0 => f as u64,
                _ => 0,
            },
            ExprNode::Identifier(id) => match id.name.as_str() {
                "KiB" => 1024,
                "MiB" => 1024 * 1024,
                "GiB" => 1024 * 1024 * 1024,
                "TiB" => 1024 * 1024 * 1024 * 1024,
                "KB" => 1000,
                "MB" => 1_000_000,
                "GB" => 1_000_000_000,
                "TB" => 1_000_000_000_000,
                _ => 0,
            },
            ExprNode::Binary(bin) => {
                let lhs = self.eval_const_u64(&bin.left);
                let rhs = self.eval_const_u64(&bin.right);
                match bin.op {
                    BinaryOperator::Mul => lhs.saturating_mul(rhs),
                    BinaryOperator::Div => lhs.checked_div(rhs).unwrap_or(0),
                    BinaryOperator::Add => lhs.saturating_add(rhs),
                    BinaryOperator::Sub => lhs.saturating_sub(rhs),
                    _ => 0,
                }
            }
            _ => 0,
        }
    }

    // -------------------------------------------------------------------------
    // Rules 10–13: Model Declarations & Members
    // -------------------------------------------------------------------------

    pub fn parse_model(&mut self) -> Result<ModelDeclarationNode, ParseError> {
        let model_tok = self.consume(TokenType::Model, "Expected 'model'", DiagnosticCode::E022)?;
        let name_tok = self.consume(
            TokenType::Identifier(String::new()),
            "Expected model identifier",
            DiagnosticCode::E022,
        )?;

        let mut generic_params = Vec::new();
        if self.check(&TokenType::Lt) {
            generic_params = self.parse_const_generic_params()?;
        }

        self.consume(TokenType::LBrace, "Expected '{' opening model body", DiagnosticCode::E022)?;

        let mut weights = Vec::new();
        let mut kv_caches = Vec::new();
        let mut scopes = Vec::new();
        let mut functions = Vec::new();

        while !self.check(&TokenType::RBrace) && !self.is_at_end() {
            let prev_cursor = self.cursor;
            if matches!(
                self.current().token_type,
                TokenType::Weight | TokenType::AttrVramImmutable
            ) {
                match self.parse_weight() {
                    Ok(w) => weights.push(w),
                    Err(_) => {
                        self.synchronize();
                        if self.cursor == prev_cursor && !self.is_at_end() {
                            self.advance();
                        }
                    }
                }
            } else if matches!(
                self.current().token_type,
                TokenType::Let | TokenType::AttrPagedVram | TokenType::AttrHostPinned
            ) {
                match self.parse_kv_cache() {
                    Ok(kv) => kv_caches.push(kv),
                    Err(_) => {
                        self.synchronize();
                        if self.cursor == prev_cursor && !self.is_at_end() {
                            self.advance();
                        }
                    }
                }
            } else if matches!(
                self.current().token_type,
                TokenType::Scope | TokenType::AttrKernelFuse | TokenType::AttrWarpSpecialized
            ) {
                match self.parse_scope() {
                    Ok(sc) => scopes.push(sc),
                    Err(_) => {
                        self.synchronize();
                        if self.cursor == prev_cursor && !self.is_at_end() {
                            self.advance();
                        }
                    }
                }
            } else if matches!(self.current().token_type, TokenType::Fn) {
                match self.parse_function() {
                    Ok(f) => functions.push(f),
                    Err(_) => {
                        self.synchronize();
                        if self.cursor == prev_cursor && !self.is_at_end() {
                            self.advance();
                        }
                    }
                }
            } else {
                let tok = self.current().clone();
                if !self.in_panic_mode {
                    self.emit_diagnostic(Diagnostic::error(
                        DiagnosticCode::E023,
                        format!("Unexpected token '{}' in model body.", tok.text),
                        tok.span.clone(),
                    ));
                    self.in_panic_mode = true;
                }
                self.synchronize();
                if self.cursor == prev_cursor && !self.is_at_end() {
                    self.advance();
                }
            }
        }

        let close_tok = self.consume(TokenType::RBrace, "Expected '}' closing model body", DiagnosticCode::E022)?;
        let span = model_tok.span.merge(&close_tok.span);

        let mut model_node = ModelDeclarationNode::new(name_tok.text, span);
        model_node.generic_params = generic_params;
        model_node.weights = weights;
        model_node.kv_caches = kv_caches;
        model_node.scopes = scopes;
        model_node.functions = functions;

        Ok(model_node)
    }

    pub fn parse_const_generic_params(&mut self) -> Result<Vec<ConstGenericParamNode>, ParseError> {
        self.consume(TokenType::Lt, "Expected '<'", DiagnosticCode::E009)?;
        let mut params = Vec::new();

        while !self.check(&TokenType::Gt) && !self.is_at_end() {
            let param = self.parse_const_generic_param()?;
            params.push(param);
            if !self.match_token(&TokenType::Comma) {
                break;
            }
        }

        self.consume(TokenType::Gt, "Expected '>' closing generic parameter list", DiagnosticCode::E009)?;
        Ok(params)
    }

    pub fn parse_const_generic_param(&mut self) -> Result<ConstGenericParamNode, ParseError> {
        let const_tok = self.consume(TokenType::Const, "Expected 'const' in generic parameter", DiagnosticCode::E009)?;
        let name_tok = self.consume(
            TokenType::Identifier(String::new()),
            "Expected parameter identifier",
            DiagnosticCode::E009,
        )?;
        self.consume(TokenType::Colon, "Expected ':' after parameter identifier", DiagnosticCode::E009)?;

        let type_tok = self.advance().clone();
        let mut default_val = None;

        if self.match_token(&TokenType::Eq) {
            let expr = self.parse_expression(0, false)?;
            // Invariant check E009: Negative const-generic parameter triggers E009
            if let ExprNode::Unary(un) = &expr {
                if un.op == UnaryOperator::Neg {
                    let err_span = const_tok.span.merge(expr.span());
                    self.emit_diagnostic(Diagnostic::error(
                        DiagnosticCode::E009,
                        format!(
                            "Negative const-generic parameter '{}' is invalid: const generics must be positive integers.",
                            name_tok.text
                        ),
                        err_span,
                    ));
                }
            }
            default_val = Some(Box::new(expr));
        }

        let span = if let Some(ref def) = default_val {
            const_tok.span.merge(def.span())
        } else {
            const_tok.span.merge(&type_tok.span)
        };

        Ok(ConstGenericParamNode::new(
            name_tok.text,
            type_tok.text,
            default_val,
            span,
        ))
    }

    // -------------------------------------------------------------------------
    // Rules 14–16: Weight Bindings & Safetensors
    // -------------------------------------------------------------------------

    pub fn parse_weight(&mut self) -> Result<WeightDeclarationNode, ParseError> {
        let start_loc = self.current().span.start;
        let tier = MemoryTier::VramImmutable;

        if self.check(&TokenType::AttrVramImmutable) {
            self.advance();
        }

        self.consume(TokenType::Weight, "Expected 'weight'", DiagnosticCode::E004)?;
        let name_tok = self.consume(
            TokenType::Identifier(String::new()),
            "Expected weight identifier",
            DiagnosticCode::E022,
        )?;
        self.consume(TokenType::Colon, "Expected ':' after weight identifier", DiagnosticCode::E004)?;

        let tensor_type = self.parse_tensor_type(tier)?;
        self.consume(TokenType::Eq, "Expected '=' before weight loader binding", DiagnosticCode::E004)?;

        let (binding, file_path, tensor_key) = if self.check(&TokenType::Safetensors) {
            let sf = self.parse_safetensors_binding()?;
            let path = sf.file_path.clone();
            let key = sf.tensor_key.clone();
            (Some(WeightBinding::Safetensors(sf)), Some(path), Some(key))
        } else if self.check(&TokenType::AbsorbMlaQ) || self.check(&TokenType::AbsorbMlaV) {
            let abs = self.parse_absorb_binding()?;
            (Some(WeightBinding::AbsorbMla(abs)), None, None)
        } else {
            let expr = self.parse_expression(0, true)?;
            (Some(WeightBinding::Expr(Box::new(expr))), None, None)
        };

        let semi_tok = self.consume(TokenType::Semicolon, "Expected ';' after weight declaration", DiagnosticCode::E004)?;
        let end_loc = semi_tok.span.end;
        let span = SourceSpan::new(self.file_path.clone(), start_loc, end_loc);

        let mut node = WeightDeclarationNode::new(name_tok.text, span);
        node.tier = tier;
        node.tensor_type = Some(tensor_type);
        node.binding = binding;
        node.file_path = file_path;
        node.tensor_key = tensor_key;

        Ok(node)
    }

    pub fn parse_safetensors_binding(&mut self) -> Result<SafetensorsBindingNode, ParseError> {
        let sf_tok = self.consume(TokenType::Safetensors, "Expected 'safetensors'", DiagnosticCode::E013)?;
        self.consume(TokenType::LParen, "Expected '(' after 'safetensors'", DiagnosticCode::E013)?;

        let file_tok = self.consume(
            TokenType::String(String::new()),
            "Expected Safetensors file path string literal",
            DiagnosticCode::E013,
        )?;
        let file_str = match file_tok.token_type {
            TokenType::String(s) => s,
            _ => file_tok.text,
        };

        self.consume(TokenType::Comma, "Expected ',' separating file path and tensor key", DiagnosticCode::E013)?;

        let key_tok = self.consume(
            TokenType::String(String::new()),
            "Expected tensor key string literal",
            DiagnosticCode::E013,
        )?;
        let key_str = match key_tok.token_type {
            TokenType::String(s) => s,
            _ => key_tok.text,
        };

        let close_tok = self.consume(TokenType::RParen, "Expected ')' closing safetensors binding", DiagnosticCode::E013)?;
        let span = sf_tok.span.merge(&close_tok.span);

        Ok(SafetensorsBindingNode::new(file_str, key_str, span))
    }

    pub fn parse_absorb_binding(&mut self) -> Result<AbsorbMlaNode, ParseError> {
        let op_tok = self.advance().clone();
        let kind = if matches!(op_tok.token_type, TokenType::AbsorbMlaQ) {
            AbsorbKind::AbsorbMlaQ
        } else {
            AbsorbKind::AbsorbMlaV
        };

        self.consume(TokenType::LParen, "Expected '(' after absorption function", DiagnosticCode::E018)?;
        let q_expr = self.parse_expression(0, true)?;
        self.consume(TokenType::Comma, "Expected ',' separating projection operands", DiagnosticCode::E018)?;
        let k_expr = self.parse_expression(0, true)?;
        let close_tok = self.consume(TokenType::RParen, "Expected ')' closing absorption function", DiagnosticCode::E018)?;
        let span = op_tok.span.merge(&close_tok.span);

        Ok(AbsorbMlaNode::new(
            kind,
            Box::new(q_expr),
            Box::new(k_expr),
            span,
        ))
    }

    // -------------------------------------------------------------------------
    // Rule 17: KV Cache Declarations
    // -------------------------------------------------------------------------

    pub fn parse_kv_cache(&mut self) -> Result<KVCacheDeclarationNode, ParseError> {
        let start_loc = self.current().span.start;
        let mut outer_tier = None;
        if self.check(&TokenType::AttrPagedVram) {
            self.advance();
            outer_tier = Some(MemoryTier::PagedVram);
        } else if self.check(&TokenType::AttrHostPinned) {
            self.advance();
            outer_tier = Some(MemoryTier::HostPinned);
        }

        let _let_tok = self.consume(TokenType::Let, "Expected 'let'", DiagnosticCode::E015)?;
        let is_mut = self.match_token(&TokenType::Mut);
        let name_tok = self.parse_identifier_token("Expected KV cache identifier")?;
        self.consume(TokenType::Colon, "Expected ':' after KV cache identifier", DiagnosticCode::E015)?;

        let mut type_spec = self.parse_type_spec()?;
        if let Some(ot) = outer_tier {
            type_spec.tier = Some(ot);
            match &mut type_spec.kind {
                TypeKind::PagedKV(p) => p.tier = ot,
                TypeKind::VirtualKV(v) => v.tier = ot,
                TypeKind::LatentKV(l) => l.tier = ot,
                TypeKind::StreamingKV(s) => s.tier = ot,
                TypeKind::SpeculativeTree(t) => t.tier = ot,
                _ => {}
            }
        }
        let semi_tok = self.consume(TokenType::Semicolon, "Expected ';' after KV cache declaration", DiagnosticCode::E015)?;
        let span = SourceSpan::new(self.file_path.clone(), start_loc, semi_tok.span.end);

        let mut node = KVCacheDeclarationNode::new(name_tok.text, is_mut, span);
        node.cache_type = Some(type_spec);
        Ok(node)
    }

    // -------------------------------------------------------------------------
    // Rules 18–21: Serving Execution Scopes
    // -------------------------------------------------------------------------

    pub fn parse_scope(&mut self) -> Result<ExecutionScopeNode, ParseError> {
        let start_loc = self.current().span.start;
        let mut attributes = Vec::new();

        // Scope Attributes (@kernel_fuse, @warp_specialized)
        while matches!(
            self.current().token_type,
            TokenType::AttrKernelFuse | TokenType::AttrWarpSpecialized
        ) {
            let attr_tok = self.advance().clone();
            let mut params = Vec::new();
            if self.match_token(&TokenType::LParen) {
                while !self.check(&TokenType::RParen) && !self.is_at_end() {
                    let p_name = self.consume(
                        TokenType::Identifier(String::new()),
                        "Expected attribute parameter name",
                        DiagnosticCode::E023,
                    )?;
                    self.consume(TokenType::Eq, "Expected '='", DiagnosticCode::E023)?;
                    let p_val = self.parse_expression(0, true)?;
                    params.push((p_name.text, p_val));
                    if !self.match_token(&TokenType::Comma) {
                        break;
                    }
                }
                self.consume(TokenType::RParen, "Expected ')' closing attribute parameter list", DiagnosticCode::E023)?;
            }
            let mut attr_node = ScopeAttributeNode::new(attr_tok.text, attr_tok.span.clone());
            attr_node.params = params;
            attributes.push(attr_node);
        }

        self.consume(TokenType::Scope, "Expected 'scope'", DiagnosticCode::E008)?;
        let kind_tok = self.advance().clone();

        let mut generic_params = Vec::new();
        if self.check(&TokenType::Lt) {
            generic_params = self.parse_const_generic_params()?;
        }

        self.consume(TokenType::LParen, "Expected '(' opening scope parameter list", DiagnosticCode::E008)?;
        let parameters = self.parse_parameter_list()?;
        self.consume(TokenType::RParen, "Expected ')' closing scope parameter list", DiagnosticCode::E008)?;

        let mut return_type = None;
        if self.match_token(&TokenType::Arrow) {
            return_type = Some(self.parse_type_spec()?);
        }

        let body = self.parse_block()?;
        let end_loc = body.span.end;
        let span = SourceSpan::new(self.file_path.clone(), start_loc, end_loc);

        match kind_tok.token_type {
            TokenType::Prefill => {
                let mut node = PrefillScopeNode::new(kind_tok.text, span);
                node.generic_params = generic_params;
                node.parameters = parameters;
                node.return_type = return_type;
                node.attributes = attributes;
                node.body = body;
                Ok(ExecutionScopeNode::Prefill(node))
            }
            TokenType::Decode => {
                let mut node = DecodeScopeNode::new(kind_tok.text, span);
                node.generic_params = generic_params;
                node.parameters = parameters;
                node.return_type = return_type;
                node.attributes = attributes;
                node.body = body;
                Ok(ExecutionScopeNode::Decode(node))
            }
            TokenType::ChunkedPrefill => {
                let mut node = ChunkedPrefillScopeNode::new(kind_tok.text, span);
                node.generic_params = generic_params;
                node.parameters = parameters;
                node.return_type = return_type;
                node.attributes = attributes;
                node.body = body;
                Ok(ExecutionScopeNode::ChunkedPrefill(node))
            }
            TokenType::SpeculativeVerify => {
                let mut node = SpeculativeVerifyScopeNode::new(kind_tok.text, span);
                node.generic_params = generic_params;
                node.parameters = parameters;
                node.return_type = return_type;
                node.attributes = attributes;
                node.body = body;
                Ok(ExecutionScopeNode::SpeculativeVerify(node))
            }
            _ => {
                // Contextual or identifier scope name
                if kind_tok.text == "prefill" {
                    let mut node = PrefillScopeNode::new(kind_tok.text, span);
                    node.generic_params = generic_params;
                    node.parameters = parameters;
                    node.return_type = return_type;
                    node.attributes = attributes;
                    node.body = body;
                    Ok(ExecutionScopeNode::Prefill(node))
                } else if kind_tok.text == "decode" {
                    let mut node = DecodeScopeNode::new(kind_tok.text, span);
                    node.generic_params = generic_params;
                    node.parameters = parameters;
                    node.return_type = return_type;
                    node.attributes = attributes;
                    node.body = body;
                    Ok(ExecutionScopeNode::Decode(node))
                } else if kind_tok.text == "chunked_prefill" {
                    let mut node = ChunkedPrefillScopeNode::new(kind_tok.text, span);
                    node.generic_params = generic_params;
                    node.parameters = parameters;
                    node.return_type = return_type;
                    node.attributes = attributes;
                    node.body = body;
                    Ok(ExecutionScopeNode::ChunkedPrefill(node))
                } else {
                    let mut node = SpeculativeVerifyScopeNode::new(kind_tok.text, span);
                    node.generic_params = generic_params;
                    node.parameters = parameters;
                    node.return_type = return_type;
                    node.attributes = attributes;
                    node.body = body;
                    Ok(ExecutionScopeNode::SpeculativeVerify(node))
                }
            }
        }
    }

    pub fn parse_function(&mut self) -> Result<FunctionDeclarationNode, ParseError> {
        let fn_tok = self.consume(TokenType::Fn, "Expected 'fn'", DiagnosticCode::E023)?;
        let name_tok = self.consume(
            TokenType::Identifier(String::new()),
            "Expected function name identifier",
            DiagnosticCode::E023,
        )?;

        let mut generic_params = Vec::new();
        if self.check(&TokenType::Lt) {
            generic_params = self.parse_const_generic_params()?;
        }

        self.consume(TokenType::LParen, "Expected '('", DiagnosticCode::E023)?;
        let parameters = self.parse_parameter_list()?;
        self.consume(TokenType::RParen, "Expected ')'", DiagnosticCode::E023)?;

        let mut return_type = None;
        if self.match_token(&TokenType::Arrow) {
            return_type = Some(self.parse_type_spec()?);
        }

        let body = self.parse_block()?;
        let span = fn_tok.span.merge(&body.span);

        let mut node = FunctionDeclarationNode::new(name_tok.text, span);
        node.generic_params = generic_params;
        node.parameters = parameters;
        node.return_type = return_type;
        node.body = body;
        Ok(node)
    }

    pub fn parse_parameter_list(&mut self) -> Result<Vec<ParameterNode>, ParseError> {
        let mut params = Vec::new();
        while !self.check(&TokenType::RParen) && !self.is_at_end() {
            params.push(self.parse_parameter()?);
            if !self.match_token(&TokenType::Comma) {
                break;
            }
        }
        Ok(params)
    }

    pub fn parse_parameter(&mut self) -> Result<ParameterNode, ParseError> {
        let is_mut = self.match_token(&TokenType::Mut);
        let name_tok = self.parse_identifier_token("Expected parameter identifier")?;
        self.consume(TokenType::Colon, "Expected ':' after parameter identifier", DiagnosticCode::E023)?;

        let type_spec = self.parse_type_spec()?;
        let mut default_val = None;
        if self.match_token(&TokenType::Eq) {
            default_val = Some(Box::new(self.parse_expression(0, true)?));
        }

        let span = if let Some(ref def) = default_val {
            name_tok.span.merge(def.span())
        } else {
            name_tok.span.merge(&type_spec.span)
        };

        let mut node = ParameterNode::new(name_tok.text, is_mut, span);
        node.type_spec = Some(type_spec);
        node.default_val = default_val;
        Ok(node)
    }

    // -------------------------------------------------------------------------
    // Rules 22–37: Type System & Memory Tiers
    // -------------------------------------------------------------------------

    pub fn parse_type_spec(&mut self) -> Result<TypeSpecNode, ParseError> {
        let start_loc = self.current().span.start;
        let mut tier = None;

        if self.check(&TokenType::AttrVramImmutable) {
            self.advance();
            tier = Some(MemoryTier::VramImmutable);
        } else if self.check(&TokenType::AttrPagedVram) {
            self.advance();
            tier = Some(MemoryTier::PagedVram);
        } else if self.check(&TokenType::AttrBumpSram) {
            self.advance();
            tier = Some(MemoryTier::BumpSram);
        } else if self.check(&TokenType::AttrHostPinned) {
            self.advance();
            tier = Some(MemoryTier::HostPinned);
        } else if self.check(&TokenType::AttrWiredLimitBounded) {
            self.advance();
            tier = Some(MemoryTier::WiredLimitBounded);
        }

        // Reference modifier: & or &mut
        if self.match_token(&TokenType::Amp) {
            let is_mut = self.match_token(&TokenType::Mut);
            let inner_spec = self.parse_type_spec()?;
            let end_loc = inner_spec.span.end;
            let span = SourceSpan::new(self.file_path.clone(), start_loc, end_loc);
            return Ok(TypeSpecNode::new(
                tier,
                TypeKind::Reference {
                    is_mut,
                    inner: Box::new(inner_spec),
                },
                span,
            ));
        }

        // Parenthesized / Tuple type: (Type1, Type2, ...)
        if self.match_token(&TokenType::LParen) {
            let mut tuple_types = Vec::new();
            while !self.check(&TokenType::RParen) && !self.is_at_end() {
                tuple_types.push(self.parse_type_spec()?);
                if !self.match_token(&TokenType::Comma) {
                    break;
                }
            }
            let close_tok = self.consume(TokenType::RParen, "Expected ')' closing tuple type", DiagnosticCode::E023)?;
            let span = SourceSpan::new(self.file_path.clone(), start_loc, close_tok.span.end);
            return Ok(TypeSpecNode::new(tier, TypeKind::Tuple(tuple_types), span));
        }

        let kind = if self.check(&TokenType::Tensor) {
            TypeKind::Tensor(self.parse_tensor_type(tier.unwrap_or(MemoryTier::VramImmutable))?)
        } else if matches!(
            self.current().token_type,
            TokenType::PagedKVCache
                | TokenType::VirtualContiguousKV
                | TokenType::LatentKVCache
                | TokenType::StreamingSinkKVCache
                | TokenType::SpeculativeTreeBlock
        ) {
            self.parse_kv_type(tier.unwrap_or(MemoryTier::PagedVram))?
        } else {
            self.parse_primitive_or_custom_type()?
        };

        let end_loc = self.current().span.start;
        let span = SourceSpan::new(self.file_path.clone(), start_loc, end_loc);
        Ok(TypeSpecNode::new(tier, kind, span))
    }

    pub fn parse_tensor_type(&mut self, tier: MemoryTier) -> Result<TensorTypeNode, ParseError> {
        let tensor_tok = self.consume(TokenType::Tensor, "Expected 'Tensor'", DiagnosticCode::E010)?;
        self.consume(TokenType::Lt, "Expected '<' after 'Tensor'", DiagnosticCode::E010)?;
        self.consume(TokenType::LBracket, "Expected '[' opening shape dimensions", DiagnosticCode::E010)?;

        let mut shape_dims = Vec::new();
        while !self.check(&TokenType::RBracket) && !self.is_at_end() {
            let dim_start = self.current().span.start;
            if self.match_token(&TokenType::Star) {
                let dim_end = self.current().span.end;
                let span = SourceSpan::new(self.file_path.clone(), dim_start, dim_end);
                shape_dims.push(DimensionNode::new_wildcard(span));
            } else {
                let expr = self.parse_expression(0, false)?;
                let span = expr.span().clone();
                shape_dims.push(DimensionNode::new_expr(Box::new(expr), span));
            }
            if !self.match_token(&TokenType::Comma) {
                break;
            }
        }

        self.consume(TokenType::RBracket, "Expected ']' closing tensor shape dimensions", DiagnosticCode::E010)?;
        self.consume(TokenType::Comma, "Expected ',' separating shape and dtype", DiagnosticCode::E010)?;

        let dtype_tok = self.advance().clone();
        let scalar_dtype = ScalarType::from_str(&dtype_tok.text).unwrap_or(ScalarType::F16);

        let mut block_quant = None;
        if self.match_token(&TokenType::Comma) && self.current().text == "block_scale" {
            self.advance();
            self.consume(TokenType::Eq, "Expected '='", DiagnosticCode::E024)?;
            self.consume(TokenType::LBracket, "Expected '['", DiagnosticCode::E024)?;
            let mut b_dims = Vec::new();
            while !self.check(&TokenType::RBracket) && !self.is_at_end() {
                b_dims.push(self.parse_expression(0, false)?);
                if !self.match_token(&TokenType::Comma) {
                    break;
                }
            }
            let rbr = self.consume(TokenType::RBracket, "Expected ']'", DiagnosticCode::E024)?;
            let bq_span = dtype_tok.span.merge(&rbr.span);
            block_quant = Some(BlockQuantSpecNode::new(b_dims, None, bq_span));
        }

        let close_tok = self.consume(TokenType::Gt, "Expected '>' closing Tensor type", DiagnosticCode::E010)?;
        let span = tensor_tok.span.merge(&close_tok.span);

        let mut node = TensorTypeNode::new(shape_dims, scalar_dtype, tier, span);
        node.block_quant_spec = block_quant;
        Ok(node)
    }

    pub fn parse_kv_type(&mut self, tier: MemoryTier) -> Result<TypeKind, ParseError> {
        let kind_tok = self.advance().clone();
        self.consume(TokenType::Lt, "Expected '<' after KV cache type", DiagnosticCode::E015)?;

        let mut args: Vec<ExprNode> = Vec::new();
        let mut dtypes: Vec<ScalarType> = Vec::new();

        while !self.check(&TokenType::Gt) && !self.is_at_end() {
            if let Some(st) = ScalarType::from_str(&self.current().text) {
                self.advance();
                dtypes.push(st);
            } else {
                args.push(self.parse_expression(0, false)?);
            }
            if !self.match_token(&TokenType::Comma) {
                break;
            }
        }

        let close_tok = self.consume(TokenType::Gt, "Expected '>' closing KV cache type", DiagnosticCode::E015)?;
        let span = kind_tok.span.merge(&close_tok.span);

        match kind_tok.token_type {
            TokenType::PagedKVCache => {
                let layers = if !args.is_empty() {
                    Box::new(args[0].clone())
                } else {
                    Box::new(ExprNode::Literal(LiteralNode::new(
                        LiteralValue::Decimal(1),
                        "1",
                        span.clone(),
                    )))
                };
                let heads = if args.len() > 1 {
                    Box::new(args[1].clone())
                } else {
                    Box::new(ExprNode::Literal(LiteralNode::new(
                        LiteralValue::Decimal(1),
                        "1",
                        span.clone(),
                    )))
                };
                let head_dim = if args.len() > 2 {
                    Box::new(args[2].clone())
                } else {
                    Box::new(ExprNode::Literal(LiteralNode::new(
                        LiteralValue::Decimal(64),
                        "64",
                        span.clone(),
                    )))
                };
                let block_size = if args.len() > 3 {
                    Box::new(args[3].clone())
                } else {
                    Box::new(ExprNode::Literal(LiteralNode::new(
                        LiteralValue::Decimal(16),
                        "16",
                        span.clone(),
                    )))
                };
                let dtype = if !dtypes.is_empty() {
                    dtypes[0]
                } else {
                    ScalarType::F16
                };

                // Invariant check E015: PagedKVCache block size in {16, 32, 64}
                let b_val = self.eval_const_u64(&block_size);
                if b_val != 16 && b_val != 32 && b_val != 64 {
                    self.emit_diagnostic(Diagnostic::error(
                        DiagnosticCode::E015,
                        format!("PagedKVCache block size {b_val} is not a power-of-two in {{16, 32, 64}}."),
                        block_size.span().clone(),
                    ));
                }

                Ok(TypeKind::PagedKV(PagedKVTypeNode::new(
                    layers, heads, head_dim, block_size, dtype, tier, span,
                )))
            }
            TokenType::VirtualContiguousKV => {
                let layers = if !args.is_empty() {
                    Box::new(args[0].clone())
                } else {
                    Box::new(ExprNode::Literal(LiteralNode::new(
                        LiteralValue::Decimal(1),
                        "1",
                        span.clone(),
                    )))
                };
                let heads = if args.len() > 1 {
                    Box::new(args[1].clone())
                } else {
                    Box::new(ExprNode::Literal(LiteralNode::new(
                        LiteralValue::Decimal(1),
                        "1",
                        span.clone(),
                    )))
                };
                let head_dim = if args.len() > 2 {
                    Box::new(args[2].clone())
                } else {
                    Box::new(ExprNode::Literal(LiteralNode::new(
                        LiteralValue::Decimal(64),
                        "64",
                        span.clone(),
                    )))
                };
                let max_seq_len = if args.len() > 3 {
                    Box::new(args[3].clone())
                } else {
                    Box::new(ExprNode::Literal(LiteralNode::new(
                        LiteralValue::Decimal(4096),
                        "4096",
                        span.clone(),
                    )))
                };
                let physical_page_size = Box::new(ExprNode::Literal(LiteralNode::new(
                    LiteralValue::Decimal(2_097_152),
                    "2097152",
                    span.clone(),
                )));
                let dtype = if !dtypes.is_empty() {
                    dtypes[0]
                } else {
                    ScalarType::F16
                };

                Ok(TypeKind::VirtualKV(VirtualKVTypeNode::new(
                    layers,
                    heads,
                    head_dim,
                    max_seq_len,
                    physical_page_size,
                    dtype,
                    tier,
                    span,
                )))
            }
            TokenType::LatentKVCache => {
                let layers = if !args.is_empty() {
                    Box::new(args[0].clone())
                } else {
                    Box::new(ExprNode::Literal(LiteralNode::new(
                        LiteralValue::Decimal(1),
                        "1",
                        span.clone(),
                    )))
                };
                let latent_dim = if args.len() > 1 {
                    Box::new(args[1].clone())
                } else {
                    Box::new(ExprNode::Literal(LiteralNode::new(
                        LiteralValue::Decimal(512),
                        "512",
                        span.clone(),
                    )))
                };
                let rope_dim = if args.len() > 2 {
                    Box::new(args[2].clone())
                } else {
                    Box::new(ExprNode::Literal(LiteralNode::new(
                        LiteralValue::Decimal(64),
                        "64",
                        span.clone(),
                    )))
                };
                let dtype = if !dtypes.is_empty() {
                    dtypes[0]
                } else {
                    ScalarType::Fp8E4m3
                };

                Ok(TypeKind::LatentKV(LatentKVTypeNode::new(
                    layers, latent_dim, rope_dim, dtype, tier, span,
                )))
            }
            TokenType::StreamingSinkKVCache => {
                let layers = if !args.is_empty() {
                    Box::new(args[0].clone())
                } else {
                    Box::new(ExprNode::Literal(LiteralNode::new(
                        LiteralValue::Decimal(1),
                        "1",
                        span.clone(),
                    )))
                };
                let heads = if args.len() > 1 {
                    Box::new(args[1].clone())
                } else {
                    Box::new(ExprNode::Literal(LiteralNode::new(
                        LiteralValue::Decimal(1),
                        "1",
                        span.clone(),
                    )))
                };
                let head_dim = if args.len() > 2 {
                    Box::new(args[2].clone())
                } else {
                    Box::new(ExprNode::Literal(LiteralNode::new(
                        LiteralValue::Decimal(64),
                        "64",
                        span.clone(),
                    )))
                };
                let sinks = if args.len() > 3 {
                    Box::new(args[3].clone())
                } else {
                    Box::new(ExprNode::Literal(LiteralNode::new(
                        LiteralValue::Decimal(4),
                        "4",
                        span.clone(),
                    )))
                };
                let window = if args.len() > 4 {
                    Box::new(args[4].clone())
                } else {
                    Box::new(ExprNode::Literal(LiteralNode::new(
                        LiteralValue::Decimal(256),
                        "256",
                        span.clone(),
                    )))
                };
                let dtype = if !dtypes.is_empty() {
                    dtypes[0]
                } else {
                    ScalarType::F16
                };

                // Invariant check E016: sinks and window must be constant positive integers (> 0)
                let s_val = self.eval_const_u64(&sinks);
                let w_val = self.eval_const_u64(&window);
                if s_val == 0 || w_val == 0 {
                    self.emit_diagnostic(Diagnostic::error(
                        DiagnosticCode::E016,
                        "StreamingSinkKVCache sinks and window must be constant positive integers.",
                        span.clone(),
                    ));
                }

                Ok(TypeKind::StreamingKV(StreamingKVTypeNode::new(
                    layers, heads, head_dim, sinks, window, dtype, tier, span,
                )))
            }
            TokenType::SpeculativeTreeBlock => {
                let max_nodes = if !args.is_empty() {
                    Box::new(args[0].clone())
                } else {
                    Box::new(ExprNode::Literal(LiteralNode::new(
                        LiteralValue::Decimal(64),
                        "64",
                        span.clone(),
                    )))
                };
                let max_depth = if args.len() > 1 {
                    Box::new(args[1].clone())
                } else {
                    Box::new(ExprNode::Literal(LiteralNode::new(
                        LiteralValue::Decimal(6),
                        "6",
                        span.clone(),
                    )))
                };
                let head_dim = if args.len() > 2 {
                    Box::new(args[2].clone())
                } else {
                    Box::new(ExprNode::Literal(LiteralNode::new(
                        LiteralValue::Decimal(64),
                        "64",
                        span.clone(),
                    )))
                };
                let dtype = if !dtypes.is_empty() {
                    dtypes[0]
                } else {
                    ScalarType::F16
                };

                Ok(TypeKind::SpeculativeTree(SpeculativeTreeTypeNode::new(
                    max_nodes,
                    max_depth,
                    head_dim,
                    dtype,
                    MemoryTier::BumpSram,
                    span,
                )))
            }
            _ => Ok(TypeKind::Custom(kind_tok.text)),
        }
    }

    pub fn parse_primitive_or_custom_type(&mut self) -> Result<TypeKind, ParseError> {
        let tok = self.advance().clone();
        if ScalarType::from_str(&tok.text).is_some() {
            return Ok(TypeKind::Primitive(tok.text));
        }

        // Generic custom type (e.g. SpeculativePath<6>)
        let mut type_name = tok.text;
        if self.match_token(&TokenType::Lt) {
            let mut params = Vec::new();
            while !self.check(&TokenType::Gt) && !self.is_at_end() {
                let p = self.advance().clone();
                params.push(p.text);
                if !self.match_token(&TokenType::Comma) {
                    break;
                }
            }
            self.consume(TokenType::Gt, "Expected '>' closing generic custom type", DiagnosticCode::E023)?;
            type_name = format!("{}<{}>", type_name, params.join(", "));
        }

        Ok(TypeKind::Custom(type_name))
    }

    // -------------------------------------------------------------------------
    // Rules 43–59: Statements & Imperative Logic
    // -------------------------------------------------------------------------

    pub fn parse_block(&mut self) -> Result<BlockStatementNode, ParseError> {
        let lbrace = self.consume(TokenType::LBrace, "Expected '{' opening block statement", DiagnosticCode::E023)?;
        let mut stmts = Vec::new();

        while !self.check(&TokenType::RBrace) && !self.is_at_end() {
            let prev_cursor = self.cursor;
            match self.parse_statement() {
                Ok(stmt) => stmts.push(stmt),
                Err(_) => {
                    self.synchronize();
                    if self.cursor == prev_cursor && !self.is_at_end() {
                        self.advance();
                    }
                }
            }
        }

        let rbrace = self.consume(TokenType::RBrace, "Expected '}' closing block statement", DiagnosticCode::E023)?;
        let span = lbrace.span.merge(&rbrace.span);
        let mut node = BlockStatementNode::new(span);
        node.statements = stmts;
        Ok(node)
    }

    pub fn parse_statement(&mut self) -> Result<StmtNode, ParseError> {
        if matches!(self.current().token_type, TokenType::Let) {
            return Ok(StmtNode::Let(self.parse_let()?));
        }
        if matches!(self.current().token_type, TokenType::If) {
            return Ok(StmtNode::If(self.parse_if()?));
        }
        if matches!(self.current().token_type, TokenType::For) {
            return Ok(StmtNode::For(self.parse_for()?));
        }
        if matches!(self.current().token_type, TokenType::While) {
            return Ok(StmtNode::While(self.parse_while()?));
        }
        if matches!(self.current().token_type, TokenType::Return) {
            return Ok(StmtNode::Return(self.parse_return()?));
        }
        if matches!(self.current().token_type, TokenType::Yield) {
            return Ok(StmtNode::Yield(self.parse_yield()?));
        }
        if matches!(
            self.current().token_type,
            TokenType::TmaLoadAsync | TokenType::TmaStoreAsync
        ) {
            return self.parse_async_tma();
        }
        if matches!(
            self.current().token_type,
            TokenType::DmaAsyncH2D | TokenType::DmaAsyncD2H
        ) {
            return Ok(StmtNode::AsyncDmaTransfer(self.parse_async_dma()?));
        }

        // Expression statement or In-place mutation assignment
        let expr = self.parse_expression(0, true)?;

        // Assignment operators: =, +=, -=, *=, /=
        let op = match self.current().token_type {
            TokenType::Eq => Some(AssignmentOperator::Assign),
            TokenType::PlusEq => Some(AssignmentOperator::AddAssign),
            TokenType::MinusEq => Some(AssignmentOperator::SubAssign),
            TokenType::StarEq => Some(AssignmentOperator::MulAssign),
            TokenType::SlashEq => Some(AssignmentOperator::DivAssign),
            _ => None,
        };

        if let Some(assign_op) = op {
            self.advance();
            let rhs = self.parse_expression(0, true)?;
            let semi = self.consume(TokenType::Semicolon, "Expected ';' after assignment", DiagnosticCode::E023)?;
            let span = expr.span().merge(&semi.span);
            return Ok(StmtNode::Assignment(AssignmentNode::new(
                Box::new(expr),
                assign_op,
                Box::new(rhs),
                span,
            )));
        }

        let semi = self.consume(TokenType::Semicolon, "Expected ';' after statement", DiagnosticCode::E023)?;
        let span = expr.span().merge(&semi.span);
        Ok(StmtNode::Expression(ExpressionStmtNode::new(
            Box::new(expr),
            span,
        )))
    }

    pub fn parse_let(&mut self) -> Result<LetBindingNode, ParseError> {
        let let_tok = self.consume(TokenType::Let, "Expected 'let'", DiagnosticCode::E023)?;
        let is_mut = self.match_token(&TokenType::Mut);

        // Tuple destructuring pattern: let (a, b) = ...
        if self.match_token(&TokenType::LParen) {
            let mut pat_names = Vec::new();
            while !self.check(&TokenType::RParen) && !self.is_at_end() {
                let id = self.parse_identifier_token("Expected identifier in destructuring pattern")?;
                pat_names.push(id.text);
                if !self.match_token(&TokenType::Comma) {
                    break;
                }
            }
            self.consume(TokenType::RParen, "Expected ')' closing destructuring pattern", DiagnosticCode::E023)?;
            self.consume(TokenType::Eq, "Expected '=' after destructuring pattern", DiagnosticCode::E023)?;
            let init_expr = self.parse_expression(0, true)?;
            let semi = self.consume(TokenType::Semicolon, "Expected ';' after let statement", DiagnosticCode::E023)?;
            let span = let_tok.span.merge(&semi.span);

            let mut node = LetBindingNode::new(format!("({})", pat_names.join(", ")), is_mut, span);
            node.init_expr = Some(Box::new(init_expr));
            node.tuple_pattern = Some(pat_names);
            return Ok(node);
        }

        let name_tok = self.parse_identifier_token("Expected identifier after 'let'")?;
        let mut type_annot = None;
        if self.match_token(&TokenType::Colon) {
            type_annot = Some(self.parse_type_spec()?);
        }

        let mut init_expr = None;
        if self.match_token(&TokenType::Eq) {
            init_expr = Some(Box::new(self.parse_expression(0, true)?));
        }

        let semi = self.consume(TokenType::Semicolon, "Expected ';' after let statement", DiagnosticCode::E023)?;
        let span = let_tok.span.merge(&semi.span);

        let mut node = LetBindingNode::new(name_tok.text, is_mut, span);
        node.type_annot = type_annot;
        node.init_expr = init_expr;
        Ok(node)
    }

    pub fn parse_if(&mut self) -> Result<IfNode, ParseError> {
        let if_tok = self.consume(TokenType::If, "Expected 'if'", DiagnosticCode::E023)?;
        self.consume(TokenType::LParen, "Expected '(' after 'if'", DiagnosticCode::E023)?;
        let cond = self.parse_expression(0, true)?;
        self.consume(TokenType::RParen, "Expected ')' after if condition", DiagnosticCode::E023)?;

        let then_branch = self.parse_block()?;
        let mut else_branch = None;

        if self.match_token(&TokenType::Else) {
            if self.check(&TokenType::If) {
                let elif = self.parse_if()?;
                else_branch = Some(Box::new(StmtNode::If(elif)));
            } else {
                let blk = self.parse_block()?;
                else_branch = Some(Box::new(StmtNode::Block(blk)));
            }
        }

        let span = if let Some(ref eb) = else_branch {
            if_tok.span.merge(eb.span())
        } else {
            if_tok.span.merge(&then_branch.span)
        };

        Ok(IfNode::new(Box::new(cond), then_branch, else_branch, span))
    }

    pub fn parse_for(&mut self) -> Result<ForNode, ParseError> {
        let for_tok = self.consume(TokenType::For, "Expected 'for'", DiagnosticCode::E023)?;
        let var_tok = self.parse_identifier_token("Expected loop variable identifier")?;
        self.consume(TokenType::In, "Expected 'in' after for variable", DiagnosticCode::E023)?;

        let start = self.parse_expression(0, true)?;
        self.consume(TokenType::DotDot, "Expected '..' in range loop", DiagnosticCode::E023)?;
        let end = self.parse_expression(0, true)?;

        let body = self.parse_block()?;
        let span = for_tok.span.merge(&body.span);

        Ok(ForNode::new(
            var_tok.text,
            Box::new(start),
            Box::new(end),
            body,
            span,
        ))
    }

    pub fn parse_while(&mut self) -> Result<WhileNode, ParseError> {
        let while_tok = self.consume(TokenType::While, "Expected 'while'", DiagnosticCode::E023)?;
        self.consume(TokenType::LParen, "Expected '(' after 'while'", DiagnosticCode::E023)?;
        let cond = self.parse_expression(0, true)?;
        self.consume(TokenType::RParen, "Expected ')' after while condition", DiagnosticCode::E023)?;

        let body = self.parse_block()?;
        let span = while_tok.span.merge(&body.span);

        Ok(WhileNode::new(Box::new(cond), body, span))
    }

    pub fn parse_return(&mut self) -> Result<ReturnNode, ParseError> {
        let ret_tok = self.consume(TokenType::Return, "Expected 'return'", DiagnosticCode::E023)?;
        let mut val = None;
        if !self.check(&TokenType::Semicolon) {
            val = Some(Box::new(self.parse_expression(0, true)?));
        }
        let semi = self.consume(TokenType::Semicolon, "Expected ';' after return", DiagnosticCode::E023)?;
        let span = ret_tok.span.merge(&semi.span);
        Ok(ReturnNode::new(val, span))
    }

    pub fn parse_yield(&mut self) -> Result<YieldNode, ParseError> {
        let yield_tok = self.consume(TokenType::Yield, "Expected 'yield'", DiagnosticCode::E023)?;
        let mut val = None;
        if self.match_token(&TokenType::LParen) {
            if !self.check(&TokenType::RParen) {
                val = Some(Box::new(self.parse_expression(0, true)?));
            }
            self.consume(TokenType::RParen, "Expected ')' after yield", DiagnosticCode::E023)?;
        } else if !self.check(&TokenType::Semicolon) {
            val = Some(Box::new(self.parse_expression(0, true)?));
        }
        let semi = self.consume(TokenType::Semicolon, "Expected ';' after yield", DiagnosticCode::E023)?;
        let span = yield_tok.span.merge(&semi.span);
        Ok(YieldNode::new(val, span))
    }

    pub fn parse_async_tma(&mut self) -> Result<StmtNode, ParseError> {
        let op_tok = self.advance().clone();
        self.consume(TokenType::LParen, "Expected '(' after TMA transfer keyword", DiagnosticCode::E020)?;
        let src = self.parse_expression(0, true)?;
        self.consume(TokenType::Comma, "Expected ','", DiagnosticCode::E020)?;
        let dst = self.parse_expression(0, true)?;

        if matches!(op_tok.token_type, TokenType::TmaLoadAsync) {
            self.consume(TokenType::Comma, "Expected ',' before mbarrier token", DiagnosticCode::E020)?;
            let mbarrier = self.parse_expression(0, true)?;
            self.consume(TokenType::RParen, "Expected ')'", DiagnosticCode::E020)?;
            let semi = self.consume(TokenType::Semicolon, "Expected ';'", DiagnosticCode::E020)?;
            let span = op_tok.span.merge(&semi.span);
            Ok(StmtNode::TmaLoad(TmaLoadNode::new(
                Box::new(src),
                Box::new(dst),
                Box::new(mbarrier),
                span,
            )))
        } else {
            self.consume(TokenType::RParen, "Expected ')'", DiagnosticCode::E020)?;
            let semi = self.consume(TokenType::Semicolon, "Expected ';'", DiagnosticCode::E020)?;
            let span = op_tok.span.merge(&semi.span);
            Ok(StmtNode::TmaStore(TmaStoreNode::new(
                Box::new(src),
                Box::new(dst),
                span,
            )))
        }
    }

    pub fn parse_async_dma(&mut self) -> Result<AsyncDmaTransferNode, ParseError> {
        let op_tok = self.advance().clone();
        let direction = if matches!(op_tok.token_type, TokenType::DmaAsyncH2D) {
            DmaDirection::H2D
        } else {
            DmaDirection::D2H
        };

        self.consume(TokenType::LParen, "Expected '(' after DMA transfer keyword", DiagnosticCode::E005)?;
        let src = self.parse_expression(0, true)?;
        self.consume(TokenType::Comma, "Expected ','", DiagnosticCode::E005)?;
        let dst = self.parse_expression(0, true)?;

        let mut signal = None;
        if self.match_token(&TokenType::Comma) {
            signal = Some(Box::new(self.parse_expression(0, true)?));
        }

        self.consume(TokenType::RParen, "Expected ')'", DiagnosticCode::E005)?;
        let semi = self.consume(TokenType::Semicolon, "Expected ';'", DiagnosticCode::E005)?;
        let span = op_tok.span.merge(&semi.span);

        Ok(AsyncDmaTransferNode::new(
            direction,
            Box::new(src),
            Box::new(dst),
            signal,
            span,
        ))
    }

    // -------------------------------------------------------------------------
    // Rules 60–72: 14-Level Pratt Precedence Climbing Engine
    // -------------------------------------------------------------------------

    pub fn parse_expression(&mut self, min_bp: u8, allow_gt: bool) -> Result<ExprNode, ParseError> {
        let mut lhs = self.parse_unary_or_primary()?;

        loop {
            let tok = self.current().clone();

            // When parsing inside angle brackets, suppress GT as a binary operator
            if !allow_gt && matches!(tok.token_type, TokenType::Gt | TokenType::GtEq | TokenType::Shr) {
                break;
            }

            let bp = match binary_binding_power(&tok.token_type) {
                Some(b) => b,
                None => break,
            };

            if bp.0 < min_bp {
                break;
            }

            self.advance();
            let rhs = self.parse_expression(bp.1, allow_gt)?;
            let span = lhs.span().merge(rhs.span());

            if matches!(tok.token_type, TokenType::At) {
                lhs = ExprNode::TensorMatmul(TensorMatmulNode::new(
                    Box::new(lhs),
                    Box::new(rhs),
                    span,
                ));
            } else {
                let bin_op = match tok.token_type {
                    TokenType::Plus => BinaryOperator::Add,
                    TokenType::Minus => BinaryOperator::Sub,
                    TokenType::Star => BinaryOperator::Mul,
                    TokenType::Slash => BinaryOperator::Div,
                    TokenType::Percent => BinaryOperator::Mod,
                    TokenType::Shl => BinaryOperator::Shl,
                    TokenType::Shr => BinaryOperator::Shr,
                    TokenType::Lt => BinaryOperator::Lt,
                    TokenType::LtEq => BinaryOperator::Lte,
                    TokenType::Gt => BinaryOperator::Gt,
                    TokenType::GtEq => BinaryOperator::Gte,
                    TokenType::EqEq => BinaryOperator::Eq,
                    TokenType::BangEq => BinaryOperator::Neq,
                    TokenType::Amp => BinaryOperator::BitAnd,
                    TokenType::Caret => BinaryOperator::BitXor,
                    TokenType::Pipe => BinaryOperator::BitOr,
                    TokenType::AmpAmp => BinaryOperator::LogicalAnd,
                    TokenType::PipePipe => BinaryOperator::LogicalOr,
                    _ => BinaryOperator::Add,
                };
                lhs = ExprNode::Binary(BinaryOpNode::new(
                    Box::new(lhs),
                    bin_op,
                    Box::new(rhs),
                    span,
                ));
            }
        }

        Ok(lhs)
    }

    pub fn parse_unary_or_primary(&mut self) -> Result<ExprNode, ParseError> {
        // Level 2: Unary Prefix Operators (!, ~, -, *, &, &mut)
        if matches!(
            self.current().token_type,
            TokenType::Bang | TokenType::Tilde | TokenType::Minus | TokenType::Star
        ) {
            let op_tok = self.advance().clone();
            let un_op = match op_tok.token_type {
                TokenType::Bang => UnaryOperator::LogicalNot,
                TokenType::Tilde => UnaryOperator::BitNot,
                TokenType::Minus => UnaryOperator::Neg,
                TokenType::Star => UnaryOperator::Deref,
                _ => UnaryOperator::LogicalNot,
            };
            let operand = self.parse_unary_or_primary()?;
            let span = op_tok.span.merge(operand.span());
            return Ok(ExprNode::Unary(UnaryOpNode::new(
                un_op,
                Box::new(operand),
                span,
            )));
        }

        if matches!(self.current().token_type, TokenType::Amp) {
            let amp_tok = self.advance().clone();
            let un_op = if self.match_token(&TokenType::Mut) {
                UnaryOperator::BorrowMut
            } else {
                UnaryOperator::Borrow
            };
            let operand = self.parse_unary_or_primary()?;
            let span = amp_tok.span.merge(operand.span());
            return Ok(ExprNode::Unary(UnaryOpNode::new(
                un_op,
                Box::new(operand),
                span,
            )));
        }

        let primary = self.parse_primary()?;
        self.parse_postfix(primary)
    }

    pub fn parse_primary(&mut self) -> Result<ExprNode, ParseError> {
        let tok = self.current().clone();

        match &tok.token_type {
            TokenType::Decimal(v) => {
                self.advance();
                Ok(ExprNode::Literal(LiteralNode::new(
                    LiteralValue::Decimal(*v),
                    tok.text,
                    tok.span,
                )))
            }
            TokenType::Hex(v) => {
                self.advance();
                Ok(ExprNode::Literal(LiteralNode::new(
                    LiteralValue::Hex(*v),
                    tok.text,
                    tok.span,
                )))
            }
            TokenType::Binary(v) => {
                self.advance();
                Ok(ExprNode::Literal(LiteralNode::new(
                    LiteralValue::Binary(*v),
                    tok.text,
                    tok.span,
                )))
            }
            TokenType::Float(f) => {
                self.advance();
                Ok(ExprNode::Literal(LiteralNode::new(
                    LiteralValue::Float(*f),
                    tok.text,
                    tok.span,
                )))
            }
            TokenType::String(s) => {
                self.advance();
                Ok(ExprNode::Literal(LiteralNode::new(
                    LiteralValue::String(s.clone()),
                    tok.text,
                    tok.span,
                )))
            }
            TokenType::Bool(b) => {
                self.advance();
                Ok(ExprNode::Literal(LiteralNode::new(
                    LiteralValue::Bool(*b),
                    tok.text,
                    tok.span,
                )))
            }
            TokenType::UnitKiB
            | TokenType::UnitMiB
            | TokenType::UnitGiB
            | TokenType::UnitTiB
            | TokenType::UnitKB
            | TokenType::UnitMB
            | TokenType::UnitGB
            | TokenType::UnitTB => {
                self.advance();
                let mult = match tok.text.as_str() {
                    "KiB" => 1024,
                    "MiB" => 1024 * 1024,
                    "GiB" => 1024 * 1024 * 1024,
                    "TiB" => 1024 * 1024 * 1024 * 1024,
                    "KB" => 1000,
                    "MB" => 1_000_000,
                    "GB" => 1_000_000_000,
                    "TB" => 1_000_000_000_000,
                    _ => 1,
                };
                Ok(ExprNode::Literal(LiteralNode::new(
                    LiteralValue::Decimal(mult),
                    tok.text,
                    tok.span,
                )))
            }
            TokenType::AbsorbMlaQ | TokenType::AbsorbMlaV => {
                let abs = self.parse_absorb_binding()?;
                Ok(ExprNode::AbsorbMla(abs))
            }
            TokenType::LParen => {
                let lparen_tok = self.advance().clone();
                let first_expr = self.parse_expression(0, true)?;

                if self.match_token(&TokenType::Comma) {
                    // Tuple expression: (a, b, ...)
                    let mut elements = vec![first_expr];
                    while !self.check(&TokenType::RParen) && !self.is_at_end() {
                        elements.push(self.parse_expression(0, true)?);
                        if !self.match_token(&TokenType::Comma) {
                            break;
                        }
                    }
                    let rparen_tok = self.consume(TokenType::RParen, "Expected ')' closing tuple", DiagnosticCode::E023)?;
                    let span = lparen_tok.span.merge(&rparen_tok.span);
                    Ok(ExprNode::Tuple(TupleNode::new(elements, span)))
                } else {
                    self.consume(TokenType::RParen, "Expected ')' closing expression", DiagnosticCode::E023)?;
                    Ok(ExprNode::Parenthesized(Box::new(first_expr)))
                }
            }
            TokenType::Identifier(name) => {
                self.advance();
                Ok(ExprNode::Identifier(IdentifierNode::new(name.clone(), tok.span)))
            }
            TokenType::Prefill
            | TokenType::Decode
            | TokenType::ChunkedPrefill
            | TokenType::SpeculativeVerify
            | TokenType::Weight
            | TokenType::Tensor => {
                self.advance();
                Ok(ExprNode::Identifier(IdentifierNode::new(tok.text.clone(), tok.span)))
            }
            _ => {
                if !self.in_panic_mode {
                    self.emit_diagnostic(Diagnostic::error(
                        DiagnosticCode::E023,
                        format!("Expected expression, found '{}'.", tok.text),
                        tok.span.clone(),
                    ));
                    self.in_panic_mode = true;
                }
                Err(ParseError::new(format!("Unexpected token '{}'", tok.text), tok.span))
            }
        }
    }

    // -------------------------------------------------------------------------
    // Rules 73–78: Postfix Operations (Member, Call, Index, Slice)
    // -------------------------------------------------------------------------

    pub fn parse_postfix(&mut self, mut expr: ExprNode) -> Result<ExprNode, ParseError> {
        loop {
            // Member Access: expr.field
            if self.match_token(&TokenType::Dot) {
                let member_tok = self.advance().clone();
                let span = expr.span().merge(&member_tok.span);
                expr = ExprNode::MemberAccess(MemberAccessNode::new(
                    Box::new(expr),
                    member_tok.text,
                    span,
                ));
                continue;
            }

            // Function / Kernel Call: expr(arg1, arg2, name = val)
            if self.match_token(&TokenType::LParen) {
                let args = self.parse_argument_list()?;
                let rparen = self.consume(TokenType::RParen, "Expected ')' closing function call", DiagnosticCode::E023)?;
                let span = expr.span().merge(&rparen.span);
                expr = ExprNode::Call(CallNode::new(Box::new(expr), args, span));
                continue;
            }

            // Array Index or Slice: expr[idx] or expr[start..end] or expr[:, S - 1, :]
            if self.match_token(&TokenType::LBracket) {
                expr = self.parse_bracket_access(expr)?;
                continue;
            }

            break;
        }

        Ok(expr)
    }

    pub fn parse_argument_list(&mut self) -> Result<Vec<ArgumentNode>, ParseError> {
        let mut args = Vec::new();
        while !self.check(&TokenType::RParen) && !self.is_at_end() {
            // Check for named keyword argument: identifier = expr
            if matches!(self.current().token_type, TokenType::Identifier(_))
                && matches!(self.peek(1).token_type, TokenType::Eq)
            {
                let name_tok = self.advance().clone();
                self.advance(); // consume '='
                let val_expr = self.parse_expression(0, true)?;
                let span = name_tok.span.merge(val_expr.span());
                args.push(ArgumentNode::new(
                    Some(name_tok.text),
                    Box::new(val_expr),
                    span,
                ));
            } else {
                let val_expr = self.parse_expression(0, true)?;
                let span = val_expr.span().clone();
                args.push(ArgumentNode::new(None, Box::new(val_expr), span));
            }

            if !self.match_token(&TokenType::Comma) {
                break;
            }
        }
        Ok(args)
    }

    pub fn parse_bracket_access(&mut self, target: ExprNode) -> Result<ExprNode, ParseError> {
        let mut dim_ranges = Vec::new();
        let mut is_multidimensional = false;
        let mut first_index_expr = None;

        while !self.check(&TokenType::RBracket) && !self.is_at_end() {
            let loc = self.current().span.start;

            if self.match_token(&TokenType::Colon) {
                // Full slice colon: ':'
                is_multidimensional = true;
                let span = SourceSpan::new(self.file_path.clone(), loc, loc);
                dim_ranges.push(SliceRangeNode::new(None, None, true, span));
            } else if self.match_token(&TokenType::DotDot) {
                // Range starting with '..' e.g. ..end
                is_multidimensional = true;
                let end_expr = if self.check(&TokenType::Comma) || self.check(&TokenType::RBracket) {
                    None
                } else {
                    Some(Box::new(self.parse_expression(0, true)?))
                };
                let span = if let Some(ref e) = end_expr {
                    e.span().clone()
                } else {
                    SourceSpan::new(self.file_path.clone(), loc, loc)
                };
                dim_ranges.push(SliceRangeNode::new(None, end_expr, false, span));
            } else {
                let first_expr = self.parse_expression(0, true)?;
                if self.match_token(&TokenType::DotDot) {
                    // Range start..end or start..
                    is_multidimensional = true;
                    let end_expr = if self.check(&TokenType::Comma) || self.check(&TokenType::RBracket) {
                        None
                    } else {
                        Some(Box::new(self.parse_expression(0, true)?))
                    };
                    let span = if let Some(ref e) = end_expr {
                        first_expr.span().merge(e.span())
                    } else {
                        first_expr.span().clone()
                    };
                    dim_ranges.push(SliceRangeNode::new(
                        Some(Box::new(first_expr)),
                        end_expr,
                        false,
                        span,
                    ));
                } else {
                    if dim_ranges.is_empty() && !is_multidimensional {
                        first_index_expr = Some(first_expr.clone());
                    }
                    let span = first_expr.span().clone();
                    dim_ranges.push(SliceRangeNode::new(
                        Some(Box::new(first_expr)),
                        None,
                        false,
                        span,
                    ));
                }
            }

            if self.match_token(&TokenType::Comma) {
                is_multidimensional = true;
            } else {
                break;
            }
        }

        let rbracket = self.consume(TokenType::RBracket, "Expected ']' closing bracket access", DiagnosticCode::E023)?;
        let span = target.span().merge(&rbracket.span);

        if is_multidimensional || dim_ranges.len() > 1 {
            Ok(ExprNode::TensorSlice(TensorSliceNode::new(
                Box::new(target),
                dim_ranges,
                span,
            )))
        } else if let Some(idx) = first_index_expr {
            Ok(ExprNode::IndexAccess(IndexAccessNode::new(
                Box::new(target),
                Box::new(idx),
                span,
            )))
        } else {
            Ok(ExprNode::TensorSlice(TensorSliceNode::new(
                Box::new(target),
                dim_ranges,
                span,
            )))
        }
    }

    // -------------------------------------------------------------------------
    // Frontend Semantic & Architectural Invariant Validation (E001–E025)
    // -------------------------------------------------------------------------

    pub fn validate_ast(&mut self, program: &ProgramNode) {
        for model in &program.model_declarations {
            // 1. E008: Missing mandatory prefill or decode scope
            let has_prefill = model.scopes.iter().any(|s| {
                matches!(
                    s.kind(),
                    ScopeKind::Prefill | ScopeKind::ChunkedPrefill
                )
            });
            let has_decode = model.scopes.iter().any(|s| {
                matches!(
                    s.kind(),
                    ScopeKind::Decode | ScopeKind::SpeculativeVerify
                )
            });

            if !has_prefill || !has_decode {
                self.emit_diagnostic(Diagnostic::error(
                    DiagnosticCode::E008,
                    format!(
                        "Model '{}' omits mandatory execution scope ('prefill' or 'decode').",
                        model.name
                    ),
                    model.span.clone(),
                ));
            }

            // 2. E022: Duplicate member names
            let mut seen_members: Vec<(&str, &SourceSpan)> = Vec::new();

            for w in &model.weights {
                if let Some((_, prev_span)) = seen_members.iter().find(|(name, _)| *name == w.name) {
                    let mut diag = Diagnostic::error(
                        DiagnosticCode::E022,
                        format!("Duplicate member '{}' declared in model '{}'.", w.name, model.name),
                        w.span.clone(),
                    );
                    diag = diag.with_secondary_span(
                        (*prev_span).clone(),
                        format!("Previous definition of '{}' was here", w.name),
                    );
                    self.emit_diagnostic(diag);
                } else {
                    seen_members.push((&w.name, &w.span));
                }
            }

            for kv in &model.kv_caches {
                if let Some((_, prev_span)) = seen_members.iter().find(|(name, _)| *name == kv.name) {
                    let mut diag = Diagnostic::error(
                        DiagnosticCode::E022,
                        format!("Duplicate member '{}' declared in model '{}'.", kv.name, model.name),
                        kv.span.clone(),
                    );
                    diag = diag.with_secondary_span(
                        (*prev_span).clone(),
                        format!("Previous definition of '{}' was here", kv.name),
                    );
                    self.emit_diagnostic(diag);
                } else {
                    seen_members.push((&kv.name, &kv.span));
                }
            }

            // 3. Inspect scopes for Invariant 1 (E001) & Invariant 3 (E003)
            for scope in &model.scopes {
                let (scope_kind, body) = match scope {
                    ExecutionScopeNode::Prefill(n) => (ScopeKind::Prefill, &n.body),
                    ExecutionScopeNode::Decode(n) => (ScopeKind::Decode, &n.body),
                    ExecutionScopeNode::ChunkedPrefill(n) => (ScopeKind::ChunkedPrefill, &n.body),
                    ExecutionScopeNode::SpeculativeVerify(n) => (ScopeKind::SpeculativeVerify, &n.body),
                };
                for stmt in &body.statements {
                    self.validate_stmt_invariants(stmt, scope_kind);
                }
            }
        }
    }

    fn validate_stmt_invariants(&mut self, stmt: &StmtNode, scope_kind: ScopeKind) {
        match stmt {
            StmtNode::Let(n) => {
                if let Some(ref expr) = n.init_expr {
                    self.validate_expr_invariants(expr, scope_kind);
                }
            }
            StmtNode::Assignment(n) => {
                self.validate_expr_invariants(&n.lhs, scope_kind);
                self.validate_expr_invariants(&n.rhs, scope_kind);
            }
            StmtNode::If(n) => {
                self.validate_expr_invariants(&n.condition, scope_kind);
                for s in &n.then_branch.statements {
                    self.validate_stmt_invariants(s, scope_kind);
                }
                if let Some(ref eb) = n.else_branch {
                    self.validate_stmt_invariants(eb, scope_kind);
                }
            }
            StmtNode::For(n) => {
                self.validate_expr_invariants(&n.start, scope_kind);
                self.validate_expr_invariants(&n.end, scope_kind);
                for s in &n.body.statements {
                    self.validate_stmt_invariants(s, scope_kind);
                }
            }
            StmtNode::While(n) => {
                self.validate_expr_invariants(&n.condition, scope_kind);
                for s in &n.body.statements {
                    self.validate_stmt_invariants(s, scope_kind);
                }
            }
            StmtNode::Return(n) => {
                if let Some(ref expr) = n.value {
                    self.validate_expr_invariants(expr, scope_kind);
                }
            }
            StmtNode::Yield(n) => {
                if let Some(ref expr) = n.value {
                    self.validate_expr_invariants(expr, scope_kind);
                }
            }
            StmtNode::Block(n) => {
                for s in &n.statements {
                    self.validate_stmt_invariants(s, scope_kind);
                }
            }
            StmtNode::Expression(n) => {
                self.validate_expr_invariants(&n.expr, scope_kind);
            }
            StmtNode::TmaLoad(n) => {
                self.validate_expr_invariants(&n.source, scope_kind);
                self.validate_expr_invariants(&n.destination, scope_kind);
                self.validate_expr_invariants(&n.mbarrier, scope_kind);
            }
            StmtNode::TmaStore(n) => {
                self.validate_expr_invariants(&n.source, scope_kind);
                self.validate_expr_invariants(&n.destination, scope_kind);
            }
            StmtNode::AsyncDmaTransfer(n) => {
                self.validate_expr_invariants(&n.source, scope_kind);
                self.validate_expr_invariants(&n.destination, scope_kind);
                if let Some(ref sig) = n.signal {
                    self.validate_expr_invariants(sig, scope_kind);
                }
            }
        }
    }

    fn validate_expr_invariants(&mut self, expr: &ExprNode, scope_kind: ScopeKind) {
        match expr {
            ExprNode::Call(call) => {
                let mut callee_name = "";
                if let ExprNode::Identifier(id) = &*call.callee {
                    callee_name = &id.name;
                } else if let ExprNode::MemberAccess(m) = &*call.callee {
                    callee_name = &m.member;
                }

                // Invariant 1 (E001): Prohibited dynamic allocators
                if matches!(
                    callee_name,
                    "malloc" | "free" | "cudaMalloc" | "cudaMallocAsync" | "calloc" | "new"
                ) {
                    self.emit_diagnostic(Diagnostic::error(
                        DiagnosticCode::E001,
                        format!("Dynamic heap allocation '{callee_name}' prohibited inside execution scope."),
                        call.span.clone(),
                    ));
                }

                // Invariant 3 (E003): Blocking host synchronization in decode
                if scope_kind.is_decode_phase()
                    && matches!(
                        callee_name,
                        "item" | "cpu" | "cudaStreamSynchronize" | "cudaMemcpy"
                    )
                {
                    self.emit_diagnostic(Diagnostic::error(
                        DiagnosticCode::E003,
                        format!("Blocking host synchronization call '{callee_name}' prohibited in decode loop."),
                        call.span.clone(),
                    ));
                }

                self.validate_expr_invariants(&call.callee, scope_kind);
                for arg in &call.args {
                    self.validate_expr_invariants(&arg.value, scope_kind);
                }
            }
            ExprNode::MemberAccess(m) => {
                // Invariant 3 (E003): Member access e.g. token.item()
                if scope_kind.is_decode_phase() && (m.member == "item" || m.member == "cpu") {
                    self.emit_diagnostic(Diagnostic::error(
                        DiagnosticCode::E003,
                        format!("Blocking host synchronization '{}' prohibited in decode loop.", m.member),
                        m.span.clone(),
                    ));
                }
                self.validate_expr_invariants(&m.obj, scope_kind);
            }
            ExprNode::Binary(bin) => {
                self.validate_expr_invariants(&bin.left, scope_kind);
                self.validate_expr_invariants(&bin.right, scope_kind);
            }
            ExprNode::TensorMatmul(mat) => {
                self.validate_expr_invariants(&mat.lhs, scope_kind);
                self.validate_expr_invariants(&mat.rhs, scope_kind);
            }
            ExprNode::Unary(un) => {
                self.validate_expr_invariants(&un.operand, scope_kind);
            }
            ExprNode::IndexAccess(idx) => {
                self.validate_expr_invariants(&idx.obj, scope_kind);
                self.validate_expr_invariants(&idx.index, scope_kind);
            }
            ExprNode::TensorSlice(sl) => {
                self.validate_expr_invariants(&sl.tensor_expr, scope_kind);
                for r in &sl.ranges {
                    if let Some(ref st) = r.start {
                        self.validate_expr_invariants(st, scope_kind);
                    }
                    if let Some(ref ed) = r.end {
                        self.validate_expr_invariants(ed, scope_kind);
                    }
                }
            }
            ExprNode::Tuple(tup) => {
                for el in &tup.elements {
                    self.validate_expr_invariants(el, scope_kind);
                }
            }
            ExprNode::Parenthesized(inner) => {
                self.validate_expr_invariants(inner, scope_kind);
            }
            ExprNode::AbsorbMla(abs) => {
                self.validate_expr_invariants(&abs.projection_q, scope_kind);
                self.validate_expr_invariants(&abs.projection_k, scope_kind);
            }
            ExprNode::Identifier(_) | ExprNode::Literal(_) => {}
        }
    }
}
