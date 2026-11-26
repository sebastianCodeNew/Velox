//! Velox Standalone Compiler Frontend CLI (`veloxc::cli`)
//!
//! Pure `std` command-line interface processor, AST text pretty printer,
//! token stream formatter, and deterministic exit code driver.
//!
//! Zero external dependencies (`std` only).

use crate::ast::*;
use crate::codegen::CodeGenerator;
use crate::diagnostics::{Diagnostic, DiagnosticCode, DiagnosticFormatter, Severity};
use crate::ir::lower_program_to_vir;
use crate::lexer::Lexer;
use crate::memory_planner::StaticMemoryPlanner;
use crate::optimizer::optimize_vir_module;
use crate::parser::Parser;
use crate::runtime::{TransformerConfig, VeloxStaticExecutionEngine};
use crate::sampling::{SamplerSession, SamplingParams};
use crate::sema::SemanticAnalyzer;
use crate::tokens::{Token, TokenType};
use crate::types::AbsorbKind;
use crate::tokenizer::{StreamingDecoder, Tokenizer};
use std::fmt::Write as FmtWrite;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

// ============================================================================
// Deterministic Exit Code Constants
// ============================================================================

/// Exit code 0: Successful compilation, syntax checking, or help/version display.
pub const EXIT_SUCCESS: i32 = 0;

/// Exit code 1: Compiler lexical, syntax, or architectural diagnostic error (E001–E025).
pub const EXIT_COMPILATION_ERROR: i32 = 1;

/// Exit code 2: File I/O error (input file missing, unreadable, directory, or unwritable output).
pub const EXIT_IO_ERROR: i32 = 2;

/// Exit code 3: Command-line usage error (missing input file, unknown option, invalid format).
pub const EXIT_USAGE_ERROR: i32 = 3;

// ============================================================================
// CLI Configuration & Argument Parsing
// ============================================================================

/// Output format for AST serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Text,
    Json,
}

impl OutputFormat {
    #[must_use]
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "text" => Some(Self::Text),
            "json" => Some(Self::Json),
            _ => None,
        }
    }
}

/// Execution backend for model generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionBackend {
    Auto,
    Cuda,
    C99,
    Rust,
}

impl ExecutionBackend {
    #[must_use]
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "cuda" => Some(Self::Cuda),
            "c99" => Some(Self::C99),
            "rust" => Some(Self::Rust),
            _ => None,
        }
    }

    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cuda => "cuda",
            Self::C99 => "c99",
            Self::Rust => "rust",
        }
    }
}

/// Speculative decoding options forwarded to the standalone CUDA runner.
/// Semantics mirror the runner's own argv exactly (pure passthrough):
/// `--speculative` enables prompt-lookup drafting; `--draft-len` sizes it and
/// does NOT enable it on its own; `--spec-ema` sets the adaptive EMA floor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpecOptions {
    /// Enable prompt-lookup speculative decoding in the external runner.
    pub speculative: bool,
    /// Maximum draft length K in 1..=7 (K+1 tokens verified per step).
    pub draft_len: Option<u32>,
    /// Adaptive acceptance EMA floor in 0.0..=1.0 below which drafting pauses.
    pub spec_ema: Option<f32>,
    /// Use lossless stochastic verification (accept with probability p(draft)).
    pub stochastic: bool,
    /// Seed for the per-step stochastic verification nonces.
    pub seed: Option<u64>,
}

impl Default for SpecOptions {
    fn default() -> Self {
        Self {
            speculative: false,
            draft_len: None,
            spec_ema: None,
            stochastic: false,
            seed: None,
        }
    }
}

/// Parsed command-line flags and positional options.
#[derive(Debug, Clone, PartialEq)]
pub struct CliArgs {
    pub input_file: String,
    pub check: bool,
    pub dump_tokens: bool,
    pub dump_ast: bool,
    pub prove_vram: bool,
    pub emit_c: bool,
    pub emit_cuda: bool,
    pub emit_ir: bool,
    pub run_model: bool,
    pub chat: bool,
    /// Start the OpenAI-compatible HTTP server instead of a one-shot run.
    pub serve: bool,
    /// Bind address for `--serve` (`host:port`).
    pub host: Option<String>,
    pub backend: ExecutionBackend,
    pub weights_path: Option<String>,
    pub tokenizer_path: Option<String>,
    pub prompt_text: Option<String>,
    pub prompt_tokens: Vec<u32>,
    pub steps: usize,
    pub spec: SpecOptions,
    pub benchmark: bool,
    pub warmup: usize,
    pub threads: Option<usize>,
    pub format: OutputFormat,
    pub no_color: bool,
    pub output: Option<String>,
    pub verbose: bool,
    /// Sampling temperature; `<= 0.0` selects greedy argmax.
    pub temperature: f32,
    /// Keep only the highest-logit `top_k` tokens; `0` disables.
    pub top_k: usize,
    /// Nucleus probability mass in `(0, 1]`.
    pub top_p: f32,
    /// Minimum token probability relative to the top token.
    pub min_p: f32,
    /// Repetition penalty divisor (`> 0`); `1.0` disables.
    pub repetition_penalty: f32,
    /// Additive frequency penalty.
    pub frequency_penalty: f32,
    /// Additive one-time presence penalty.
    pub presence_penalty: f32,
    /// Deterministic seed for stochastic sampling.
    pub seed: u64,
    /// Text-level stop strings; generation ends at the earliest match.
    pub stop_strings: Vec<String>,
    /// Suppress the model's `?` reasoning block in chat output.
    pub hide_thinking: bool,
}

impl CliArgs {
    /// Builds the validated sampler configuration for this invocation.
    #[must_use]
    pub fn sampling_params(&self) -> SamplingParams {
        SamplingParams {
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            min_p: self.min_p,
            repetition_penalty: self.repetition_penalty,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            seed: self.seed,
        }
    }
}

impl Default for CliArgs {
    fn default() -> Self {
        Self {
            input_file: String::new(),
            check: false,
            dump_tokens: false,
            dump_ast: false,
            prove_vram: false,
            emit_c: false,
            emit_cuda: false,
            emit_ir: false,
            run_model: false,
            chat: false,
            serve: false,
            host: None,
            backend: ExecutionBackend::Auto,
            weights_path: None,
            tokenizer_path: None,
            prompt_text: None,
            prompt_tokens: vec![1, 504, 346],
            steps: 32,
            spec: SpecOptions::default(),
            benchmark: false,
            warmup: 0,
            threads: None,
            format: OutputFormat::Text,
            no_color: false,
            output: None,
            verbose: false,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: 0x5EED_0F_5EED_0F5E,
            stop_strings: Vec::new(),
            hide_thinking: true,
        }
    }
}

/// CLI action resolved from argument stream.
#[derive(Debug, Clone, PartialEq)]
pub enum CliAction {
    Help,
    Version,
    Run(CliArgs),
}

/// Parses CLI arguments without mutating global state.
pub fn parse_cli_args(raw_args: &[String]) -> Result<CliAction, String> {
    // If the first argument is the binary path (ends with veloxc or veloxc.exe), strip it.
    let args: &[String] = if !raw_args.is_empty() {
        let first = raw_args[0].to_ascii_lowercase();
        if first.ends_with("veloxc.exe") || first.ends_with("veloxc") || first == "veloxc" {
            &raw_args[1..]
        } else {
            raw_args
        }
    } else {
        raw_args
    };

    // 1. Early check for help
    for arg in args {
        if arg == "-h" || arg == "--help" {
            return Ok(CliAction::Help);
        }
    }

    // 2. Early check for version
    if args.len() == 1 && (args[0] == "-v" || args[0] == "-V" || args[0] == "--version") {
        return Ok(CliAction::Version);
    }
    for arg in args {
        if arg == "--version" {
            return Ok(CliAction::Version);
        }
    }

    let mut cli_args = CliArgs::default();
    let mut positional_files: Vec<String> = Vec::new();
    let mut idx = 0;

    while idx < args.len() {
        let arg = &args[idx];

        if arg == "--check" {
            cli_args.check = true;
            idx += 1;
        } else if arg == "--dump-tokens" {
            cli_args.dump_tokens = true;
            idx += 1;
        } else if arg == "--dump-ast" {
            cli_args.dump_ast = true;
            idx += 1;
        } else if arg == "--prove-vram" {
            cli_args.prove_vram = true;
            idx += 1;
        } else if arg == "--emit-c" {
            cli_args.emit_c = true;
            idx += 1;
        } else if arg == "--emit-cuda" {
            cli_args.emit_cuda = true;
            idx += 1;
        } else if arg == "--emit-ir" {
            cli_args.emit_ir = true;
            idx += 1;
        } else if arg == "--run" {
            cli_args.run_model = true;
            // The CUDA/C99 runners have no SmolLM implementation. For the
            // CPU reference architecture, --run without an explicit backend
            // must stay on Rust; Auto still upgrades K2 to CUDA.
            if cli_args.backend == ExecutionBackend::Auto {
                cli_args.backend = ExecutionBackend::Rust;
            }
            idx += 1;
        } else if arg == "--chat" {
            cli_args.chat = true;
            cli_args.run_model = true;
            idx += 1;
        } else if arg == "--serve" {
            cli_args.serve = true;
            idx += 1;
        } else if arg == "--host" {
            if idx + 1 >= args.len() {
                return Err("error: flag '--host' requires an address (host:port)".to_string());
            }
            cli_args.host = Some(args[idx + 1].clone());
            idx += 2;
        } else if let Some(host) = arg.strip_prefix("--host=") {
            cli_args.host = Some(host.to_string());
            idx += 1;
        } else if arg == "--no-color" {
            cli_args.no_color = true;
            idx += 1;
        } else if arg == "-v" || arg == "--verbose" {
            cli_args.verbose = true;
            idx += 1;
        } else if arg == "-V" {
            return Ok(CliAction::Version);
        } else if arg == "--backend" {
            if idx + 1 >= args.len() {
                return Err("error: flag '--backend' requires an argument (cuda, c99, rust, auto)".to_string());
            }
            let choice = &args[idx + 1];
            match ExecutionBackend::from_str_opt(choice) {
                Some(b) => cli_args.backend = b,
                None => {
                    return Err(format!(
                        "error: argument --backend: invalid choice: '{choice}' (choose from 'cuda', 'c99', 'rust', 'auto')"
                    ));
                }
            }
            idx += 2;
        } else if let Some(choice) = arg.strip_prefix("--backend=") {
            match ExecutionBackend::from_str_opt(choice) {
                Some(b) => cli_args.backend = b,
                None => {
                    return Err(format!(
                        "error: argument --backend: invalid choice: '{choice}' (choose from 'cuda', 'c99', 'rust', 'auto')"
                    ));
                }
            }
            idx += 1;
        } else if arg == "--prompt" {
            if idx + 1 >= args.len() {
                return Err("error: flag '--prompt' requires a string argument".to_string());
            }
            cli_args.prompt_text = Some(args[idx + 1].clone());
            idx += 2;
        } else if let Some(p_text) = arg.strip_prefix("--prompt=") {
            cli_args.prompt_text = Some(p_text.to_string());
            idx += 1;
        } else if arg == "--tokenizer" {
            if idx + 1 >= args.len() {
                return Err("error: flag '--tokenizer' requires a path argument".to_string());
            }
            cli_args.tokenizer_path = Some(args[idx + 1].clone());
            idx += 2;
        } else if let Some(t_path) = arg.strip_prefix("--tokenizer=") {
            cli_args.tokenizer_path = Some(t_path.to_string());
            idx += 1;
        } else if arg == "--benchmark" {
            cli_args.benchmark = true;
            idx += 1;
        } else if arg == "--warmup" {
            if idx + 1 >= args.len() {
                return Err("error: flag '--warmup' requires an integer argument".to_string());
            }
            cli_args.warmup = args[idx + 1]
                .parse::<usize>()
                .map_err(|_| format!("error: invalid integer for --warmup: '{}'", args[idx + 1]))?;
            idx += 2;
        } else if let Some(w_str) = arg.strip_prefix("--warmup=") {
            cli_args.warmup = w_str
                .parse::<usize>()
                .map_err(|_| format!("error: invalid integer for --warmup: '{w_str}'"))?;
            idx += 1;
        } else if arg == "--threads" {
            if idx + 1 >= args.len() {
                return Err("error: flag '--threads' requires an integer argument".to_string());
            }
            cli_args.threads = Some(args[idx + 1]
                .parse::<usize>()
                .map_err(|_| format!("error: invalid integer for --threads: '{}'", args[idx + 1]))?);
            idx += 2;
        } else if let Some(t_str) = arg.strip_prefix("--threads=") {
            cli_args.threads = Some(t_str
                .parse::<usize>()
                .map_err(|_| format!("error: invalid integer for --threads: '{t_str}'"))?);
            idx += 1;
        } else if arg == "--weights" {
            if idx + 1 >= args.len() {
                return Err("error: flag '--weights' requires an argument".to_string());
            }
            cli_args.weights_path = Some(args[idx + 1].clone());
            idx += 2;
        } else if let Some(w_path) = arg.strip_prefix("--weights=") {
            cli_args.weights_path = Some(w_path.to_string());
            idx += 1;
        } else if arg == "--tokens" || arg == "--prompt-tokens" {
            if idx + 1 >= args.len() {
                return Err(format!("error: flag '{arg}' requires a comma-separated token list"));
            }
            cli_args.prompt_tokens = parse_token_list(&args[idx + 1])?;
            idx += 2;
        } else if let Some(tok_str) = arg.strip_prefix("--tokens=") {
            cli_args.prompt_tokens = parse_token_list(tok_str)?;
            idx += 1;
        } else if let Some(tok_str) = arg.strip_prefix("--prompt-tokens=") {
            cli_args.prompt_tokens = parse_token_list(tok_str)?;
            idx += 1;
        } else if arg == "--steps" {
            if idx + 1 >= args.len() {
                return Err("error: flag '--steps' requires an integer argument".to_string());
            }
            cli_args.steps = args[idx + 1]
                .parse::<usize>()
                .map_err(|_| format!("error: invalid integer for --steps: '{}'", args[idx + 1]))?;
            idx += 2;
        } else if let Some(steps_str) = arg.strip_prefix("--steps=") {
            cli_args.steps = steps_str
                .parse::<usize>()
                .map_err(|_| format!("error: invalid integer for --steps: '{steps_str}'"))?;
            idx += 1;
        } else if arg == "--speculative" {
            cli_args.spec.speculative = true;
            // Optional numeric draft width, mirroring the runner's argv:
            // "--speculative K". Consume the next token only when it parses
            // as an integer so "veloxc --speculative model.vlx" still treats
            // model.vlx as the required positional input file.
            if idx + 1 < args.len() && !args[idx + 1].starts_with('-') {
                if let Ok(k) = args[idx + 1].parse::<u32>() {
                    if (1u32..=7u32).contains(&k) {
                        cli_args.spec.draft_len = Some(k);
                        idx += 1;
                    } else {
                        return Err(format!(
                            "error: draft length for --speculative must be in the range 1-7: {k}"
                        ));
                    }
                }
            }
            idx += 1;
        } else if let Some(k_str) = arg.strip_prefix("--speculative=") {
            match k_str.parse::<u32>() {
                Ok(k) if (1u32..=7u32).contains(&k) => {
                    cli_args.spec.speculative = true;
                    cli_args.spec.draft_len = Some(k);
                }
                Ok(k) => {
                    return Err(format!(
                        "error: draft length for --speculative must be in the range 1-7: {k}"
                    ));
                }
                Err(_) => {
                    return Err(format!("error: invalid draft length for --speculative: '{k_str}'"));
                }
            }
            idx += 1;
        } else if arg == "--draft-len" {
            if idx + 1 >= args.len() {
                return Err("error: flag '--draft-len' requires an integer argument (1-7)".to_string());
            }
            match args[idx + 1].parse::<u32>() {
                Ok(k) if (1u32..=7u32).contains(&k) => {
                    cli_args.spec.draft_len = Some(k);
                }
                Ok(k) => {
                    return Err(format!("error: draft length must be in the range 1-7: {k}"));
                }
                Err(_) => {
                    return Err(format!(
                        "error: invalid integer for --draft-len: '{}'",
                        args[idx + 1]
                    ));
                }
            }
            idx += 2;
        } else if let Some(k_str) = arg.strip_prefix("--draft-len=") {
            match k_str.parse::<u32>() {
                Ok(k) if (1u32..=7u32).contains(&k) => {
                    cli_args.spec.draft_len = Some(k);
                }
                Ok(k) => {
                    return Err(format!("error: draft length must be in the range 1-7: {k}"));
                }
                Err(_) => {
                    return Err(format!("error: invalid integer for --draft-len: '{k_str}'"));
                }
            }
            idx += 1;
        } else if arg == "--spec-ema" {
            if idx + 1 >= args.len() {
                return Err("error: flag '--spec-ema' requires a numeric argument (0.0-1.0)".to_string());
            }
            let e_str = &args[idx + 1];
            match e_str.parse::<f32>() {
                Ok(e) if e.is_finite() && (0.0f32..=1.0f32).contains(&e) => {
                    cli_args.spec.spec_ema = Some(e);
                }
                Ok(e) => {
                    return Err(format!("error: EMA floor for --spec-ema must be in 0.0-1.0: {e}"));
                }
                Err(_) => {
                    return Err(format!("error: invalid float for --spec-ema: '{e_str}'"));
                }
            }
            idx += 2;
        } else if let Some(e_str) = arg.strip_prefix("--spec-ema=") {
            match e_str.parse::<f32>() {
                Ok(e) if e.is_finite() && (0.0f32..=1.0f32).contains(&e) => {
                    cli_args.spec.spec_ema = Some(e);
                }
                Ok(e) => {
                    return Err(format!("error: EMA floor for --spec-ema must be in 0.0-1.0: {e}"));
                }
                Err(_) => {
                    return Err(format!("error: invalid float for --spec-ema: '{e_str}'"));
                }
            }
            idx += 1;
        } else if arg == "--temperature" {
            cli_args.temperature = parse_f32_arg(args, idx, arg, "temperature")?;
            idx += 2;
        } else if let Some(v) = arg.strip_prefix("--temperature=") {
            cli_args.temperature = parse_f32_value(v, "temperature")?;
            idx += 1;
        } else if arg == "--top-k" {
            if idx + 1 >= args.len() {
                return Err("error: flag '--top-k' requires an integer argument".to_string());
            }
            cli_args.top_k = parse_usize_value(&args[idx + 1], "top-k")?;
            idx += 2;
        } else if let Some(v) = arg.strip_prefix("--top-k=") {
            cli_args.top_k = parse_usize_value(v, "top-k")?;
            idx += 1;
        } else if arg == "--top-p" {
            cli_args.top_p = parse_f32_arg(args, idx, arg, "top-p")?;
            idx += 2;
        } else if let Some(v) = arg.strip_prefix("--top-p=") {
            cli_args.top_p = parse_f32_value(v, "top-p")?;
            idx += 1;
        } else if arg == "--min-p" {
            cli_args.min_p = parse_f32_arg(args, idx, arg, "min-p")?;
            idx += 2;
        } else if let Some(v) = arg.strip_prefix("--min-p=") {
            cli_args.min_p = parse_f32_value(v, "min-p")?;
            idx += 1;
        } else if arg == "--repetition-penalty" {
            cli_args.repetition_penalty = parse_f32_arg(args, idx, arg, "repetition-penalty")?;
            idx += 2;
        } else if let Some(v) = arg.strip_prefix("--repetition-penalty=") {
            cli_args.repetition_penalty = parse_f32_value(v, "repetition-penalty")?;
            idx += 1;
        } else if arg == "--frequency-penalty" {
            cli_args.frequency_penalty = parse_f32_arg(args, idx, arg, "frequency-penalty")?;
            idx += 2;
        } else if let Some(v) = arg.strip_prefix("--frequency-penalty=") {
            cli_args.frequency_penalty = parse_f32_value(v, "frequency-penalty")?;
            idx += 1;
        } else if arg == "--presence-penalty" {
            cli_args.presence_penalty = parse_f32_arg(args, idx, arg, "presence-penalty")?;
            idx += 2;
        } else if let Some(v) = arg.strip_prefix("--presence-penalty=") {
            cli_args.presence_penalty = parse_f32_value(v, "presence-penalty")?;
            idx += 1;
        } else if arg == "--seed" {
            if idx + 1 >= args.len() {
                return Err("error: flag '--seed' requires an unsigned 64-bit integer".to_string());
            }
            cli_args.seed = args[idx + 1]
                .parse::<u64>()
                .map_err(|_| format!("error: invalid integer for --seed: '{}'", args[idx + 1]))?;
            idx += 2;
        } else if let Some(v) = arg.strip_prefix("--seed=") {
            cli_args.seed = v
                .parse::<u64>()
                .map_err(|_| format!("error: invalid integer for --seed: '{v}'"))?;
            idx += 1;
        } else if arg == "--spec-stochastic" {
            cli_args.spec.speculative = true;
            cli_args.spec.stochastic = true;
            idx += 1;
        } else if arg == "--spec-seed" {
            if idx + 1 >= args.len() {
                return Err("error: flag '--spec-seed' requires an unsigned 64-bit integer".to_string());
            }
            cli_args.spec.seed = Some(
                args[idx + 1]
                    .parse::<u64>()
                    .map_err(|_| format!("error: invalid integer for --spec-seed: '{}'", args[idx + 1]))?,
            );
            idx += 2;
        } else if let Some(v) = arg.strip_prefix("--spec-seed=") {
            cli_args.spec.seed = Some(
                v.parse::<u64>()
                    .map_err(|_| format!("error: invalid integer for --spec-seed: '{v}'"))?,
            );
            idx += 1;
        } else if arg == "--show-thinking" {
            cli_args.hide_thinking = false;
            idx += 1;
        } else if arg == "--stop" {
            if idx + 1 >= args.len() {
                return Err("error: flag '--stop' requires a string argument".to_string());
            }
            cli_args.stop_strings.push(args[idx + 1].clone());
            idx += 2;
        } else if let Some(v) = arg.strip_prefix("--stop=") {
            cli_args.stop_strings.push(v.to_string());
            idx += 1;
        } else if arg == "--format" {
            if idx + 1 >= args.len() {
                return Err("error: flag '--format' requires an argument (choose from 'text', 'json')".to_string());
            }
            let choice = &args[idx + 1];
            match OutputFormat::from_str_opt(choice) {
                Some(fmt) => cli_args.format = fmt,
                None => {
                    return Err(format!(
                        "error: argument --format: invalid choice: '{choice}' (choose from 'text', 'json')"
                    ));
                }
            }
            idx += 2;
        } else if let Some(choice) = arg.strip_prefix("--format=") {
            match OutputFormat::from_str_opt(choice) {
                Some(fmt) => cli_args.format = fmt,
                None => {
                    return Err(format!(
                        "error: argument --format: invalid choice: '{choice}' (choose from 'text', 'json')"
                    ));
                }
            }
            idx += 1;
        } else if arg == "-o" || arg == "--output" {
            if idx + 1 >= args.len() {
                return Err(format!("error: flag '{arg}' requires an argument"));
            }
            cli_args.output = Some(args[idx + 1].clone());
            idx += 2;
        } else if let Some(out_path) = arg.strip_prefix("-o=") {
            cli_args.output = Some(out_path.to_string());
            idx += 1;
        } else if let Some(out_path) = arg.strip_prefix("--output=") {
            cli_args.output = Some(out_path.to_string());
            idx += 1;
        } else if arg.starts_with('-') {
            return Err(format!("error: unrecognized arguments: {arg}"));
        } else {
            positional_files.push(arg.clone());
            idx += 1;
        }
    }

    if positional_files.is_empty() {
        if cli_args.run_model {
            let default_model = "examples/tinystories_smollm.vlx";
            if Path::new(default_model).exists() {
                cli_args.input_file = default_model.to_string();
            } else {
                return Err("error: the following arguments are required: input_file".to_string());
            }
        } else {
            return Err("error: the following arguments are required: input_file".to_string());
        }
    } else if positional_files.len() > 1 {
        let extra = &positional_files[1..];
        return Err(format!("error: unrecognized arguments: {}", extra.join(" ")));
    } else {
        cli_args.input_file = positional_files.remove(0);
    }
    Ok(CliAction::Run(cli_args))
}

/// Strips the model's `?` reasoning block, returning only the final answer.
///
/// If the block was never closed (generation hit the cap mid-thought) the
/// reasoning is discarded and a short marker is returned instead, so a
/// truncated think never dumps raw chain-of-thought into the chat.
fn strip_thinking(text: &str) -> String {
    const OPEN: &str = "?";
    const CLOSE: &str = "?";
    if let Some(start) = text.find(OPEN) {
        let after = &text[start + OPEN.len()..];
        if let Some(end) = after.find(CLOSE) {
            return after[end + CLOSE.len()..].trim().to_string();
        }
        return "[thinking truncated ? raise --steps]".to_string();
    }
    text.trim().to_string()
}

/// Returns the byte offset of the earliest stop-string match in `text`.
fn find_stop_string(text: &str, stops: &[String]) -> Option<usize> {
    stops
        .iter()
        .filter(|stop| !stop.is_empty())
        .filter_map(|stop| text.find(stop.as_str()))
        .min()
}

fn parse_f32_arg(args: &[String], idx: usize, flag: &str, field: &str) -> Result<f32, String> {
    if idx + 1 >= args.len() {
        return Err(format!("error: flag '{flag}' requires a numeric argument"));
    }
    parse_f32_value(&args[idx + 1], field)
}

fn parse_f32_value(raw: &str, field: &str) -> Result<f32, String> {
    raw.parse::<f32>()
        .map_err(|_| format!("error: invalid float for --{field}: '{raw}'"))
}

fn parse_usize_value(raw: &str, field: &str) -> Result<usize, String> {
    raw.parse::<usize>()
        .map_err(|_| format!("error: invalid integer for --{field}: '{raw}'"))
}

fn parse_token_list(raw: &str) -> Result<Vec<u32>, String> {
    let mut out = Vec::new();
    for part in raw.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        let id = trimmed
            .parse::<u32>()
            .map_err(|_| format!("error: invalid token ID '{trimmed}' in --tokens"))?;
        out.push(id);
    }
    if out.is_empty() {
        return Err("error: --tokens list cannot be empty".to_string());
    }
    Ok(out)
}

/// Emits the standard Velox CLI usage manual.
pub fn format_help_manual() -> &'static str {
    "\
usage: veloxc [options] <file.vlx>

Velox Native Optimizing Compiler & Zero-Copy Runtime (veloxc 0.1.0)
Domain-specific systems language compiler for LLM serving and inference.

positional arguments:
  input_file             Path to the input Velox (.vlx) source file.

options:
  -h, --help             Show this help message and exit.
  -v, --version          Show compiler frontend version.
  --check                Perform syntax, semantic, and VRAM budget validation.
  --prove-vram           Emit Compile-Time VRAM Feasibility Proof Certificate (.h).
  --emit-ir              Lower to Velox IR (VIR), run SOTA passes, and dump SSA IR.
  --emit-c               Emit standalone zero-allocation ANSI C99 source (.c).
  --emit-cuda            Emit standalone zero-allocation NVIDIA CUDA source (.cu).
  --run                  Execute model natively using zero-copy safetensors mmap.
  --chat                 Launch interactive native coding REPL (100% zero-Python).
  --serve                Start the OpenAI-compatible HTTP server (default 127.0.0.1:8080).
  --host <host:port>     Bind address for --serve (default 127.0.0.1:8080).
  --backend <backend>    Execution backend: cuda, c99, rust, auto (default: auto; K2 uses CUDA when available).
  --prompt \"<text>\"      Input text prompt (encoded via native BPE tokenizer).
  --tokens <ids>         Comma-separated input prompt token IDs for --run.
  --prompt-tokens <ids>  Alias for --tokens.
  --steps <N>            Number of tokens to generate during --run (default: 32).
  --temperature <t>      Sampling temperature; <= 0 selects greedy argmax (default: 0).
  --top-k <N>            Keep only the N highest-logit tokens; 0 disables (default: 0).
  --top-p <p>            Nucleus probability mass in (0, 1] (default: 1.0).
  --min-p <p>            Minimum token probability relative to the top token (default: 0.0).
  --repetition-penalty <r> Divide logits of repeated tokens by r > 0 (default: 1.0).
  --frequency-penalty <f> Subtract f * occurrence_count from repeated logits (default: 0).
  --presence-penalty <p> Subtract p once for any repeated token (default: 0).
  --seed <N>             Deterministic sampling seed as u64 (default: 0x5EED0F5EED0F5E).
  --stop <text>          Stop generation at this text (repeatable).
  --show-thinking        Show the model's ? reasoning block (hidden by default).
  --speculative [K]      Enable prompt-lookup speculative decoding (CUDA runner; optional draft width, default K=3).
  --draft-len <K>        Set the max speculative draft length 1-7 (used with --speculative).
  --spec-ema <x>         Adaptive drafting: EMA acceptance floor 0.0-1.0 below which drafting pauses (default: 0.20).
  --spec-stochastic      Lossless stochastic verification: accept a draft with probability p(draft).
  --spec-seed <N>        Seed for stochastic verification nonces (default: 0x5EED0F5EED0F5E).
  --weights <path>       Override path to .safetensors weights file for --run.
  --tokenizer <path>     Path to tokenizer.json (default: tokenizer.json).
  --benchmark            Print detailed inference latency, throughput, and allocations.
  --warmup <N>           Number of warmup iterations before benchmark timing.
  --threads <N>          Number of threads for CPU execution (sets OMP_NUM_THREADS).
  --dump-ast             Parse source and serialize AST to stdout or output file.
  --dump-tokens          Tokenize source and print token stream to stdout.
  --format {text,json}   Output format for AST serialization (default: text).
  --no-color             Disable ANSI color escape sequences in diagnostics.
  -o, --output <path>    Write generated artifact to the specified output file.
  --verbose              Enable verbose validation logs.
"
}

/// Locates a pre-compiled runner executable across candidate directory locations.
pub fn find_runner_binary(binary_name: &str) -> Option<PathBuf> {
    // 1. Direct path in src/runners
    let p1 = Path::new("src").join("runners").join(binary_name);
    if p1.exists() {
        return Some(p1);
    }
    // 2. Direct path in current working directory
    let p2 = Path::new(binary_name);
    if p2.exists() {
        return Some(p2.to_path_buf());
    }
    // 3. Executable-relative paths
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(parent) = exe_path.parent() {
            let p3 = parent.join(binary_name);
            if p3.exists() {
                return Some(p3);
            }
            let p4 = parent.join("src").join("runners").join(binary_name);
            if p4.exists() {
                return Some(p4);
            }
            if let Some(grandparent) = parent.parent() {
                let p5 = grandparent.join(binary_name);
                if p5.exists() {
                    return Some(p5);
                }
                let p6 = grandparent.join("src").join("runners").join(binary_name);
                if p6.exists() {
                    return Some(p6);
                }
                if let Some(great_grandparent) = grandparent.parent() {
                    let p7 = great_grandparent.join(binary_name);
                    if p7.exists() {
                        return Some(p7);
                    }
                    let p8 = great_grandparent.join("src").join("runners").join(binary_name);
                    if p8.exists() {
                        return Some(p8);
                    }
                }
            }
        }
    }
    None
}

/// A long-lived CUDA runner that keeps the checkpoint resident between turns.
///
/// The one-shot path re-reads ~2.9 GB of weights per message. In persistent
/// mode the runner loads once and only clears the KV pool between prompts, so
/// every turn after the first costs a prefill instead of a model load.
struct PersistentRunner {
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    reader: std::io::BufReader<std::process::ChildStdout>,
}

impl PersistentRunner {
    #[allow(clippy::too_many_arguments)]
    fn spawn(
        exe: &Path,
        weights: &str,
        steps: usize,
        stop_tokens: &[u32],
        temperature: f32,
        top_k: usize,
        min_p: f32,
        seed: u64,
    ) -> Result<Self, String> {
        use std::io::BufRead;
        let mut child = Command::new(exe)
            .arg("--weights")
            .arg(weights)
            .arg("--serve-stdin")
            .arg("--steps")
            .arg(steps.to_string())
            .arg("--stop-tokens")
            .arg(
                stop_tokens
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
            )
            .arg("--temperature")
            .arg(temperature.to_string())
            .arg("--top-k")
            .arg(top_k.to_string())
            .arg("--min-p")
            .arg(min_p.to_string())
            .arg("--sample-seed")
            .arg(seed.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|err| format!("failed to spawn persistent runner: {err}"))?;
        let stdin = child.stdin.take().ok_or("runner stdin unavailable")?;
        let stdout = child.stdout.take().ok_or("runner stdout unavailable")?;
        let _ = BufRead::fill_buf(&mut std::io::BufReader::new(std::io::empty()));
        Ok(Self {
            child,
            stdin,
            reader: std::io::BufReader::new(stdout),
        })
    }

    /// Sends one prompt (token IDs) and returns the generated token IDs.
    fn generate(&mut self, tokens: &[u32]) -> Result<Vec<u32>, String> {
        use std::io::BufRead;
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        let line = tokens
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        writeln!(self.stdin, "{line}").map_err(|err| format!("runner stdin write failed: {err}"))?;
        self.stdin
            .flush()
            .map_err(|err| format!("runner stdin flush failed: {err}"))?;

        let mut reply = String::new();
        let read = self
            .reader
            .read_line(&mut reply)
            .map_err(|err| format!("runner stdout read failed: {err}"))?;
        if read == 0 {
            return Err("runner closed its output stream".to_string());
        }
        Ok(reply
            .trim()
            .split(',')
            .filter_map(|part| part.trim().parse::<u32>().ok())
            .collect())
    }
}

impl Drop for PersistentRunner {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Dispatches execution to a standalone compiled C99 or CUDA runner process,
/// streaming tokens to stdout and parsing the output report.
pub fn run_external_runner(
    runner_path: &Path,
    weights_path: &str,
    prompt_tokens: &[u32],
    steps: usize,
    spec: SpecOptions,
    threads_opt: Option<usize>,
    mut streaming_decoder_opt: Option<StreamingDecoder>,
    verbose: bool,
    benchmark: bool,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> Result<i32, i32> {
    let mut cmd = Command::new(runner_path);
    cmd.arg("--weights").arg(weights_path);

    let tokens_str = prompt_tokens
        .iter()
        .map(|t| t.to_string())
        .collect::<Vec<_>>()
        .join(",");
    cmd.arg("--tokens").arg(&tokens_str);
    cmd.arg("--steps").arg(steps.to_string());
    cmd.arg("--verbose");

    // Speculative-decoding passthrough (mirrors the runner's raw argv
    // semantics: --speculative enables, --draft-len sizes, --spec-ema floors).
    if spec.speculative {
        cmd.arg("--speculative");
    }
    if let Some(draft_len) = spec.draft_len {
        cmd.arg("--draft-len").arg(draft_len.to_string());
    }
    if let Some(spec_ema) = spec.spec_ema {
        cmd.arg("--spec-ema").arg(spec_ema.to_string());
    }
    if spec.stochastic {
        cmd.arg("--spec-stochastic");
    }
    if let Some(seed) = spec.seed {
        cmd.arg("--spec-seed").arg(seed.to_string());
    }

    if runner_path.to_string_lossy().contains("k2") {
        cmd.arg("--context-len").arg("16384");
    }

    if let Some(threads) = threads_opt {
        cmd.env("OMP_NUM_THREADS", threads.to_string());
    }

    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(err) => {
            let _ = writeln!(
                stderr,
                "veloxc: failed to execute runner '{:?}': {}",
                runner_path, err
            );
            return Err(EXIT_IO_ERROR);
        }
    };

    let mut captured_lines = Vec::new();
    let mut generated_tokens = Vec::new();
    let mut streamed_any = false;

    if let Some(child_stdout) = child.stdout.take() {
        use std::io::BufRead;
        let reader = std::io::BufReader::new(child_stdout);
        for line_res in reader.lines() {
            let line = match line_res {
                Ok(l) => l,
                Err(_) => break,
            };

            // Detect step tokens: "[Step <S>] Token: <ID>"
            if let Some(step_part) = line.strip_prefix("[Step ") {
                if let Some(idx_token) = step_part.find("] Token: ") {
                    let tok_str = step_part[idx_token + 9..].trim();
                    if let Ok(tok_id) = tok_str.parse::<u32>() {
                        generated_tokens.push(tok_id);
                        if let Some(ref mut dec) = streaming_decoder_opt {
                            let text_chunk = dec.decode_token(tok_id);
                            let _ = write!(stdout, "{text_chunk}");
                            let _ = stdout.flush();
                            streamed_any = true;
                        } else if verbose {
                            let _ = writeln!(stdout, "{line}");
                            let _ = stdout.flush();
                        }
                        continue;
                    }
                }
            }

            captured_lines.push(line);
        }
    }

    let status = match child.wait() {
        Ok(s) => s,
        Err(err) => {
            let _ = writeln!(stderr, "veloxc: error waiting for runner process: {}", err);
            return Err(EXIT_IO_ERROR);
        }
    };

    if let Some(ref mut dec) = streaming_decoder_opt {
        let flush_chunk = dec.flush();
        if !flush_chunk.is_empty() {
            let _ = write!(stdout, "{flush_chunk}");
            let _ = stdout.flush();
            streamed_any = true;
        }
        if streamed_any {
            let _ = writeln!(stdout);
        }
    }

    // Print captured execution / benchmark report from runner
    if benchmark || verbose || streaming_decoder_opt.is_none() {
        for line in &captured_lines {
            let _ = writeln!(stdout, "{line}");
        }
    }

    if status.success() {
        Ok(EXIT_SUCCESS)
    } else {
        Err(status.code().unwrap_or(EXIT_COMPILATION_ERROR))
    }
}

// ============================================================================
// Token Stream Formatter
// ============================================================================

/// Maps any `TokenType` to a uppercase categorical classification string.
#[must_use]
pub fn token_type_to_kind_str(t: &TokenType) -> &'static str {
    match t {
        TokenType::Eof => "EOF",
        TokenType::Error(_) => "ERROR",
        TokenType::Identifier(_) => "IDENTIFIER",
        TokenType::Decimal(_) => "DECIMAL_LITERAL",
        TokenType::Hex(_) => "HEX_LITERAL",
        TokenType::Binary(_) => "BIN_LITERAL",
        TokenType::Float(_) => "FLOAT_LITERAL",
        TokenType::String(_) => "STRING_LITERAL",
        TokenType::Bool(_) => "BOOLEAN_LITERAL",

        TokenType::Target => "KW_TARGET",
        TokenType::Hardware => "KW_HARDWARE",
        TokenType::Device => "KW_DEVICE",
        TokenType::VramCapacity => "KW_VRAM_CAPACITY",
        TokenType::ComputeCapability => "KW_COMPUTE_CAPABILITY",
        TokenType::SafetyFactor => "KW_SAFETY_FACTOR",
        TokenType::Model => "KW_MODEL",
        TokenType::Weight => "KW_WEIGHT",
        TokenType::Tensor => "KW_TENSOR",
        TokenType::Safetensors => "KW_SAFETENSORS",
        TokenType::Mmap => "KW_MMAP",
        TokenType::Import => "KW_IMPORT",
        TokenType::As => "KW_AS",
        TokenType::Const => "KW_CONST",

        TokenType::Scope => "KW_SCOPE",
        TokenType::Prefill => "KW_PREFILL",
        TokenType::Decode => "KW_DECODE",
        TokenType::ChunkedPrefill => "KW_CHUNKED_PREFILL",
        TokenType::SpeculativeVerify => "KW_SPECULATIVE_VERIFY",

        TokenType::PagedKVCache => "KW_PAGED_KV_CACHE",
        TokenType::VirtualContiguousKV => "KW_VIRTUAL_CONTIGUOUS_KV",
        TokenType::LatentKVCache => "KW_LATENT_KV_CACHE",
        TokenType::StreamingSinkKVCache => "KW_STREAMING_SINK_KV_CACHE",
        TokenType::SpeculativeTreeBlock => "KW_SPECULATIVE_TREE_BLOCK",

        TokenType::Let => "KW_LET",
        TokenType::Mut => "KW_MUT",
        TokenType::Fn => "KW_FN",
        TokenType::Kernel => "KW_KERNEL",
        TokenType::Return => "KW_RETURN",
        TokenType::If => "KW_IF",
        TokenType::Else => "KW_ELSE",
        TokenType::For => "KW_FOR",
        TokenType::In => "KW_IN",
        TokenType::While => "KW_WHILE",
        TokenType::Yield => "KW_YIELD",
        TokenType::Null => "KW_NULL",

        TokenType::AbsorbMlaQ => "KW_ABSORB_MLA_Q",
        TokenType::AbsorbMlaV => "KW_ABSORB_MLA_V",
        TokenType::DmaAsyncH2D => "KW_DMA_ASYNC_H2D",
        TokenType::DmaAsyncD2H => "KW_DMA_ASYNC_D2H",
        TokenType::TmaLoadAsync => "KW_TMA_LOAD_ASYNC",
        TokenType::TmaStoreAsync => "KW_TMA_STORE_ASYNC",
        TokenType::Mbarrier => "KW_MBARRIER",

        TokenType::AttrVramImmutable => "ATTR_VRAM_IMMUTABLE",
        TokenType::AttrPagedVram => "ATTR_PAGED_VRAM",
        TokenType::AttrBumpSram => "ATTR_BUMP_SRAM",
        TokenType::AttrHostPinned => "ATTR_HOST_PINNED",
        TokenType::AttrWiredLimitBounded => "ATTR_WIRED_LIMIT_BOUNDED",
        TokenType::AttrWarpSpecialized => "ATTR_WARP_SPECIALIZED",
        TokenType::AttrKernelFuse => "ATTR_KERNEL_FUSE",

        TokenType::TypeF32 => "TYPE_F32",
        TokenType::TypeF16 => "TYPE_F16",
        TokenType::TypeBf16 => "TYPE_BF16",
        TokenType::TypeFp8E4m3 => "TYPE_FP8_E4M3",
        TokenType::TypeFp8E5m2 => "TYPE_FP8_E5M2",
        TokenType::TypeFp8E8m0 => "TYPE_FP8_E8M0",
        TokenType::TypeInt8 => "TYPE_INT8",
        TokenType::TypeInt4 => "TYPE_INT4",
        TokenType::TypeU64 => "TYPE_U64",
        TokenType::TypeU32 => "TYPE_U32",
        TokenType::TypeU16 => "TYPE_U16",
        TokenType::TypeU8 => "TYPE_U8",
        TokenType::TypeI64 => "TYPE_I64",
        TokenType::TypeI32 => "TYPE_I32",
        TokenType::TypeI16 => "TYPE_I16",
        TokenType::TypeI8 => "TYPE_I8",
        TokenType::TypeBool => "TYPE_BOOL",
        TokenType::TypeUsize => "TYPE_USIZE",

        TokenType::UnitKiB => "UNIT_KIB",
        TokenType::UnitMiB => "UNIT_MIB",
        TokenType::UnitGiB => "UNIT_GIB",
        TokenType::UnitTiB => "UNIT_TIB",
        TokenType::UnitKB => "UNIT_KB",
        TokenType::UnitMB => "UNIT_MB",
        TokenType::UnitGB => "UNIT_GB",
        TokenType::UnitTB => "UNIT_TB",

        TokenType::At => "AT",
        TokenType::Plus => "PLUS",
        TokenType::Minus => "MINUS",
        TokenType::Star => "STAR",
        TokenType::Slash => "SLASH",
        TokenType::Percent => "PERCENT",
        TokenType::Shl => "SHL",
        TokenType::Shr => "SHR",
        TokenType::Amp => "AMP",
        TokenType::Caret => "CARET",
        TokenType::Pipe => "PIPE",
        TokenType::Tilde => "TILDE",
        TokenType::Bang => "BANG",
        TokenType::AmpAmp => "AMP_AMP",
        TokenType::PipePipe => "PIPE_PIPE",
        TokenType::Eq => "EQ",
        TokenType::EqEq => "EQ_EQ",
        TokenType::BangEq => "BANG_EQ",
        TokenType::Lt => "LT",
        TokenType::LtEq => "LT_EQ",
        TokenType::Gt => "GT",
        TokenType::GtEq => "GT_EQ",
        TokenType::PlusEq => "PLUS_EQ",
        TokenType::MinusEq => "MINUS_EQ",
        TokenType::StarEq => "STAR_EQ",
        TokenType::SlashEq => "SLASH_EQ",
        TokenType::Arrow => "ARROW",
        TokenType::FatArrow => "FAT_ARROW",
        TokenType::ColonColon => "COLON_COLON",

        TokenType::Dot => "DOT",
        TokenType::DotDot => "DOT_DOT",
        TokenType::Comma => "COMMA",
        TokenType::Colon => "COLON",
        TokenType::Semicolon => "SEMICOLON",
        TokenType::LParen => "LPAREN",
        TokenType::RParen => "RPAREN",
        TokenType::LBracket => "LBRACKET",
        TokenType::RBracket => "RBRACKET",
        TokenType::LBrace => "LBRACE",
        TokenType::RBrace => "RBRACE",
    }
}

/// Formats a token slice into a structured token stream dump.
#[must_use]
pub fn format_token_stream(tokens: &[Token]) -> String {
    let mut out = String::new();
    for tok in tokens {
        let loc = format!("{}:{}", tok.span.start.line, tok.span.start.column);
        let kind = token_type_to_kind_str(&tok.token_type);
        let _ = writeln!(out, "[{:>8}] {:<28} {:?}", loc, kind, tok.text);
    }
    out
}

// ============================================================================
// Text AST Pretty Printer
// ============================================================================

/// Pretty printer for formatting parsed Velox AST hierarchies into indented text.
#[derive(Debug, Clone)]
pub struct ASTPrettyPrinter {
    pub indent_size: usize,
}

impl Default for ASTPrettyPrinter {
    fn default() -> Self {
        Self::new()
    }
}

impl ASTPrettyPrinter {
    #[must_use]
    pub fn new() -> Self {
        Self { indent_size: 2 }
    }

    #[must_use]
    pub fn with_indent_size(indent_size: usize) -> Self {
        Self { indent_size }
    }

    #[must_use]
    pub fn print(&self, program: &ProgramNode) -> String {
        let mut out = String::new();
        self.format_program(program, 0, &mut out);
        out
    }

    fn indent(&self, level: usize) -> String {
        " ".repeat(level * self.indent_size)
    }

    fn format_program(&self, node: &ProgramNode, depth: usize, out: &mut String) {
        let ind = self.indent(depth);
        let _ = writeln!(out, "{}ProgramNode ({})", ind, node.span);

        // Target hardware
        if let Some(ref hw) = node.target_hardware {
            self.format_target_hardware(hw, depth + 1, out);
        }

        // Imports
        if !node.imports.is_empty() {
            let _ = writeln!(out, "{}imports: [", self.indent(depth + 1));
            for imp in &node.imports {
                let _ = writeln!(
                    out,
                    "{}ImportNode path={:?} ({})",
                    self.indent(depth + 2),
                    imp.path,
                    imp.span
                );
            }
            let _ = writeln!(out, "{}]", self.indent(depth + 1));
        }

        // Models
        if !node.model_declarations.is_empty() {
            let _ = writeln!(out, "{}model_declarations: [", self.indent(depth + 1));
            for m in &node.model_declarations {
                self.format_model_declaration(m, depth + 2, out);
            }
            let _ = writeln!(out, "{}]", self.indent(depth + 1));
        }

        // Global bindings
        if !node.global_bindings.is_empty() {
            let _ = writeln!(out, "{}global_bindings: [", self.indent(depth + 1));
            for b in &node.global_bindings {
                self.format_let_binding(b, depth + 2, out);
            }
            let _ = writeln!(out, "{}]", self.indent(depth + 1));
        }
    }

    fn format_target_hardware(&self, node: &TargetHardwareNode, depth: usize, out: &mut String) {
        let ind = self.indent(depth);
        let _ = writeln!(
            out,
            "{}target_hardware: TargetHardwareNode (device: {:?}, vram: {} bytes, sm: ({}, {}), safety: {}) ({})",
            ind,
            node.device_name,
            node.vram_capacity_bytes,
            node.compute_capability.0,
            node.compute_capability.1,
            node.safety_factor,
            node.span
        );
    }

    fn format_model_declaration(&self, node: &ModelDeclarationNode, depth: usize, out: &mut String) {
        let ind = self.indent(depth);
        let _ = writeln!(out, "{}ModelDeclarationNode {:?} ({})", ind, node.name, node.span);

        // Const generics
        if !node.generic_params.is_empty() {
            let _ = writeln!(out, "{}const_generics: [", self.indent(depth + 1));
            for cg in &node.generic_params {
                let _ = writeln!(
                    out,
                    "{}ConstGenericParamNode name={:?} type={:?} ({})",
                    self.indent(depth + 2),
                    cg.name,
                    cg.type_name,
                    cg.span
                );
            }
            let _ = writeln!(out, "{}]", self.indent(depth + 1));
        }

        // Weights
        if !node.weights.is_empty() {
            let _ = writeln!(out, "{}weights: [", self.indent(depth + 1));
            for w in &node.weights {
                self.format_weight_declaration(w, depth + 2, out);
            }
            let _ = writeln!(out, "{}]", self.indent(depth + 1));
        }

        // KV Caches
        if !node.kv_caches.is_empty() {
            let _ = writeln!(out, "{}kv_caches: [", self.indent(depth + 1));
            for kv in &node.kv_caches {
                self.format_kv_cache_declaration(kv, depth + 2, out);
            }
            let _ = writeln!(out, "{}]", self.indent(depth + 1));
        }

        // Scopes
        if !node.scopes.is_empty() {
            let _ = writeln!(out, "{}scopes: [", self.indent(depth + 1));
            for s in &node.scopes {
                self.format_execution_scope(s, depth + 2, out);
            }
            let _ = writeln!(out, "{}]", self.indent(depth + 1));
        }
    }

    fn format_weight_declaration(&self, node: &WeightDeclarationNode, depth: usize, out: &mut String) {
        let ind = self.indent(depth);
        let binding_str = match &node.binding {
            Some(WeightBinding::Safetensors(s)) => format!("safetensors({:?}, key={:?})", s.file_path, s.tensor_key),
            Some(WeightBinding::AbsorbMla(a)) => match a.kind {
                AbsorbKind::AbsorbMlaQ => "absorb_mla_q(...)".to_string(),
                AbsorbKind::AbsorbMlaV => "absorb_mla_v(...)".to_string(),
            },
            Some(WeightBinding::Expr(_)) => "expr(...)".to_string(),
            None => "unbound".to_string(),
        };
        let _ = writeln!(
            out,
            "{}WeightDeclarationNode name={:?} binding={} ({})",
            ind, node.name, binding_str, node.span
        );
    }

    fn format_kv_cache_declaration(&self, node: &KVCacheDeclarationNode, depth: usize, out: &mut String) {
        let ind = self.indent(depth);
        let type_desc = match &node.cache_type {
            Some(spec) => match &spec.kind {
                TypeKind::PagedKV(_) => "PagedKVCache",
                TypeKind::VirtualKV(_) => "VirtualContiguousKV",
                TypeKind::LatentKV(_) => "LatentKVCache",
                TypeKind::StreamingKV(_) => "StreamingSinkKVCache",
                TypeKind::SpeculativeTree(_) => "SpeculativeTreeBlock",
                TypeKind::Tensor(_) => "Tensor",
                TypeKind::Primitive(s) => s.as_str(),
                TypeKind::Custom(s) => s.as_str(),
                TypeKind::Reference { .. } => "Reference",
                TypeKind::Tuple(_) => "Tuple",
            },
            None => "unspecified",
        };
        let _ = writeln!(
            out,
            "{}KVCacheDeclarationNode name={:?} type={} ({})",
            ind, node.name, type_desc, node.span
        );
    }

    fn format_execution_scope(&self, node: &ExecutionScopeNode, depth: usize, out: &mut String) {
        let ind = self.indent(depth);
        match node {
            ExecutionScopeNode::Prefill(n) => {
                let _ = writeln!(
                    out,
                    "{}PrefillScopeNode name={:?} params={} stmts={} ({})",
                    ind,
                    n.name,
                    n.parameters.len(),
                    n.body.statements.len(),
                    n.span
                );
            }
            ExecutionScopeNode::Decode(n) => {
                let _ = writeln!(
                    out,
                    "{}DecodeScopeNode name={:?} params={} stmts={} ({})",
                    ind,
                    n.name,
                    n.parameters.len(),
                    n.body.statements.len(),
                    n.span
                );
            }
            ExecutionScopeNode::ChunkedPrefill(n) => {
                let _ = writeln!(
                    out,
                    "{}ChunkedPrefillScopeNode name={:?} params={} stmts={} ({})",
                    ind,
                    n.name,
                    n.parameters.len(),
                    n.body.statements.len(),
                    n.span
                );
            }
            ExecutionScopeNode::SpeculativeVerify(n) => {
                let _ = writeln!(
                    out,
                    "{}SpeculativeVerifyScopeNode name={:?} params={} stmts={} ({})",
                    ind,
                    n.name,
                    n.parameters.len(),
                    n.body.statements.len(),
                    n.span
                );
            }
        }
    }

    fn format_let_binding(&self, node: &LetBindingNode, depth: usize, out: &mut String) {
        let ind = self.indent(depth);
        let _ = writeln!(
            out,
            "{}LetBindingNode name={:?} is_mut={} ({})",
            ind, node.name, node.is_mut, node.span
        );
    }
}

// ============================================================================
// Core CLI Runner with Custom IO Streams
// ============================================================================

/// Runs the CLI driver writing to arbitrary standard output and error sinks.
pub fn run_cli_with_io(
    args: &[String],
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> i32 {
    let action = match parse_cli_args(args) {
        Ok(act) => act,
        Err(err_msg) => {
            let _ = writeln!(stderr, "veloxc: {err_msg}");
            return EXIT_USAGE_ERROR;
        }
    };

    match action {
        CliAction::Help => {
            let _ = write!(stdout, "{}", format_help_manual());
            EXIT_SUCCESS
        }
        CliAction::Version => {
            let _ = writeln!(stdout, "veloxc {}", crate::VERSION);
            EXIT_SUCCESS
        }
        CliAction::Run(cli_args) => execute_compilation(&cli_args, stdout, stderr),
    }
}

/// Main entry point for binary execution using process standard streams.
#[must_use]
pub fn run_cli(args: &[String]) -> i32 {
    let mut stdout = std::io::stdout();
    let mut stderr = std::io::stderr();
    run_cli_with_io(args, &mut stdout, &mut stderr)
}

fn execute_server(cli_args: &CliArgs, stdout: &mut dyn Write, stderr: &mut dyn Write) -> i32 {
    let weights_file = cli_args
        .weights_path
        .clone()
        .unwrap_or_else(|| "smollm-135m.safetensors".to_string());
    let tokenizer_path = cli_args
        .tokenizer_path
        .clone()
        .unwrap_or_else(|| "tokenizer.json".to_string());
    let host = cli_args
        .host
        .clone()
        .unwrap_or_else(|| crate::server::DEFAULT_BIND_ADDR.to_string());

    let _ = writeln!(
        stdout,
        "velox server: model={}, weights={}, tokenizer={}, bind={}",
        cli_args.input_file, weights_file, tokenizer_path, host
    );
    let _ = stdout.flush();

    let config = crate::server::ServerConfig {
        bind_addr: host,
        weights_path: weights_file,
        tokenizer_path,
        model_file: cli_args.input_file.clone(),
        max_seq_len: 2048,
        worker_threads: cli_args.threads.unwrap_or(4).max(1),
        default_max_tokens: cli_args.steps.clamp(1, crate::server::MAX_REQUEST_TOKENS),
    };

    match crate::server::serve(config) {
        Ok(()) => EXIT_SUCCESS,
        Err(err) => {
            let _ = writeln!(stderr, "veloxc: server error: {err}");
            EXIT_IO_ERROR
        }
    }
}

fn execute_compilation(
    cli_args: &CliArgs,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> i32 {
    if cli_args.serve {
        return execute_server(cli_args, stdout, stderr);
    }

    let input_path = Path::new(&cli_args.input_file);

    // 1. Verify existence (I/O check)
    if !input_path.exists() {
        let _ = writeln!(
            stderr,
            "veloxc: I/O error: input file not found: {}",
            cli_args.input_file
        );
        return EXIT_IO_ERROR;
    }

    // 2. Verify is file (I/O check)
    if !input_path.is_file() {
        let _ = writeln!(
            stderr,
            "veloxc: I/O error: path is not a regular file: {}",
            cli_args.input_file
        );
        return EXIT_IO_ERROR;
    }

    // 3. Read content (I/O check)
    let source_text = match fs::read_to_string(input_path) {
        Ok(text) => text,
        Err(err) => {
            let _ = writeln!(
                stderr,
                "veloxc: I/O error: could not read input file '{}': {}",
                cli_args.input_file, err
            );
            return EXIT_IO_ERROR;
        }
    };

    let formatter = DiagnosticFormatter::new(!cli_args.no_color);

    // 4. Tokenization & Lexical Analysis
    let tokens = match Lexer::tokenize_source(&source_text, &cli_args.input_file) {
        Ok(toks) => toks,
        Err(lex_err) => {
            let diag = Diagnostic::error(
                DiagnosticCode::E023,
                format!("Unterminated string literal or lexical error: {}", lex_err.message),
                lex_err.span,
            );
            let _ = writeln!(stderr, "{}", formatter.format(&diag, &source_text));
            let _ = writeln!(stderr, "veloxc: lexical error: {}", lex_err.message);
            return EXIT_COMPILATION_ERROR;
        }
    };

    // Check for error tokens
    let error_tokens: Vec<&Token> = tokens.iter().filter(|t| t.is_error()).collect();
    if !error_tokens.is_empty() {
        for err_tok in error_tokens {
            let diag = Diagnostic::error(
                DiagnosticCode::E023,
                format!("Unrecognized lexical token '{}'", err_tok.text),
                err_tok.span.clone(),
            );
            let _ = writeln!(stderr, "{}", formatter.format(&diag, &source_text));
        }
        return EXIT_COMPILATION_ERROR;
    }

    // 5. Dump Tokens Mode
    if cli_args.dump_tokens {
        let formatted = format_token_stream(&tokens);
        if let Err(io_code) = write_output_text(&formatted, cli_args.output.as_deref(), stdout, stderr) {
            return io_code;
        }
        return EXIT_SUCCESS;
    }

    // 6. Parsing & Invariant Validation
    let mut parser = Parser::new(tokens, &cli_args.input_file, &source_text);
    let ast_opt = parser.parse_program_safe();

    let has_errors = parser
        .diagnostics
        .iter()
        .any(|d| d.severity == Severity::Error);

    if has_errors || ast_opt.is_none() {
        for diag in &parser.diagnostics {
            let _ = writeln!(stderr, "{}", formatter.format(diag, &source_text));
        }
        return EXIT_COMPILATION_ERROR;
    }

    let program = ast_opt.expect("Validated AST must exist when no error diagnostics emitted");

    // 7. AST Serialization Mode (--dump-ast, or output file specified when no backend flag is active)
    let has_backend_mode = cli_args.prove_vram
        || cli_args.emit_c
        || cli_args.emit_cuda
        || cli_args.emit_ir
        || cli_args.run_model;
    let should_dump_ast = cli_args.dump_ast || (cli_args.output.is_some() && !has_backend_mode);
    if should_dump_ast {
        let content = match cli_args.format {
            OutputFormat::Json => program.to_json(),
            OutputFormat::Text => ASTPrettyPrinter::new().print(&program),
        };
        if let Err(io_code) = write_output_text(&content, cli_args.output.as_deref(), stdout, stderr) {
            return io_code;
        }
        return EXIT_SUCCESS;
    }

    // 8. Step 6A: Semantic Analysis, Shape Inference & Symbolic Unification
    let sema = SemanticAnalyzer::new();
    let typed_program = match sema.analyze(&program) {
        Ok(tp) => tp,
        Err(diags) => {
            for diag in diags {
                let _ = writeln!(stderr, "{}", formatter.format(&diag, &source_text));
            }
            return EXIT_COMPILATION_ERROR;
        }
    };

    // 9. Step 6B: Static Memory Planning, Interval Graph Coloring & VRAM Feasibility Proof
    let planner = StaticMemoryPlanner::new();
    let memory_plans = match planner.plan_program(&typed_program) {
        Ok(plans) => plans,
        Err(diags) => {
            for diag in diags {
                let _ = writeln!(stderr, "{}", formatter.format(&diag, &source_text));
            }
            return EXIT_COMPILATION_ERROR;
        }
    };

    // 10. Mode: --prove-vram
    if cli_args.prove_vram {
        let mut out_buf = String::new();
        for plan in &memory_plans {
            out_buf.push_str(&plan.format_proof_certificate());
        }
        if let Err(io_code) = write_output_text(&out_buf, cli_args.output.as_deref(), stdout, stderr) {
            return io_code;
        }
        return EXIT_SUCCESS;
    }

    // 11. Step 7: Lowering to VIR, Optimization Passes & Code Generation (--emit-ir, --emit-c, --emit-cuda)
    if cli_args.emit_ir || cli_args.emit_c || cli_args.emit_cuda {
        let mut vir_modules = lower_program_to_vir(&typed_program, &memory_plans);
        for vir_mod in &mut vir_modules {
            optimize_vir_module(vir_mod);
        }

        let mut out_buf = String::new();
        for vir_mod in &vir_modules {
            if cli_args.emit_ir {
                out_buf.push_str(&vir_mod.format_ir());
            } else if cli_args.emit_c {
                out_buf.push_str(&CodeGenerator::emit_c99(vir_mod));
            } else if cli_args.emit_cuda {
                out_buf.push_str(&CodeGenerator::emit_cuda(vir_mod));
            }
        }

        if let Err(io_code) = write_output_text(&out_buf, cli_args.output.as_deref(), stdout, stderr) {
            return io_code;
        }
        return EXIT_SUCCESS;
    }

    // 12. Steps 8 & 9: Zero-Copy Native Runtime Execution (--run)
    // 12. Steps 8, 9 & 17: Zero-Copy Native Multi-Backend Serving Execution (--run)
    if cli_args.run_model {
        let Some(first_model) = typed_program.models.first() else {
            let _ = writeln!(stderr, "veloxc: runtime error: no model declared in {}", cli_args.input_file);
            return EXIT_COMPILATION_ERROR;
        };

        let is_k2 = first_model.name.contains("K2") || first_model.name.contains("Horizon");

        let weights_file = if let Some(ref w_override) = cli_args.weights_path {
            w_override.clone()
        } else if is_k2 && Path::new("weights/k2_horizon_q4_k_m.gguf").exists() {
            "weights/k2_horizon_q4_k_m.gguf".to_string()
        } else if let Some(first_w) = first_model.weights.first() {
            first_w
                .file_path
                .clone()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| if is_k2 { "k2-horizon-3.7b-int4.safetensors".to_string() } else { "smollm-135m.safetensors".to_string() })
        } else if is_k2 {
            "k2-horizon-3.7b-int4.safetensors".to_string()
        } else {
            "smollm-135m.safetensors".to_string()
        };

        // Resolve tokenizer path
        let tokenizer_path = if let Some(ref t_path) = cli_args.tokenizer_path {
            Some(PathBuf::from(t_path))
        } else if is_k2 && Path::new("k2_tokenizer.json").exists() {
            Some(PathBuf::from("k2_tokenizer.json"))
        } else if Path::new("tokenizer.json").exists() {
            Some(PathBuf::from("tokenizer.json"))
        } else {
            let candidate = Path::new(&weights_file).with_file_name("tokenizer.json");
            if candidate.exists() {
                Some(candidate)
            } else {
                None
            }
        };

        let tokenizer_opt = if let Some(ref t_path) = tokenizer_path {
            Tokenizer::from_file(t_path).ok()
        } else {
            None
        };

        // Resolve prompt token sequence
        // Resolve execution backend and model family
        let is_k2 = first_model.name.contains("K2") || first_model.name.contains("Horizon");

        let prompt_tokens = if let Some(ref prompt_text) = cli_args.prompt_text {
            if let Some(ref tok) = tokenizer_opt {
                let mut enc = tok.encode(prompt_text);
                let bos_tok = if is_k2 { 0u32 } else { 1u32 };
                // Prepend model-appropriate BOS token (0 for K2, 1 for LLaMA/SmolLM) if not present
                if !enc.starts_with(&[bos_tok]) && (!prompt_text.starts_with("<|") || is_k2) {
                    enc.insert(0, bos_tok);
                }
                enc
            } else {
                let _ = writeln!(
                    stderr,
                    "veloxc: runtime error: tokenizer.json not found (required to encode --prompt). Specify --tokenizer <path> or use --tokens <id1,id2...>."
                );
                return EXIT_IO_ERROR;
            }
        } else {
            cli_args.prompt_tokens.clone()
        };

        let effective_backend = match cli_args.backend {
            ExecutionBackend::Auto => {
                let cuda_target = if is_k2 { "velox_cuda_runner_k2.exe" } else { "velox_cuda_runner.exe" };
                if find_runner_binary(cuda_target).is_some() {
                    ExecutionBackend::Cuda
                } else if find_runner_binary("velox_c99_runner.exe").is_some() {
                    ExecutionBackend::C99
                } else {
                    ExecutionBackend::Rust
                }
            }
            other => other,
        };

        let runner_name = if effective_backend == ExecutionBackend::Cuda {
            if is_k2 {
                "velox_cuda_runner_k2.exe"
            } else {
                "velox_cuda_runner.exe"
            }
        } else {
            "velox_c99_runner.exe"
        };

        // Native Zero-Python Interactive Coding REPL (--chat)
        if cli_args.chat {
            let Some(ref tok) = tokenizer_opt else {
                let _ = writeln!(stderr, "veloxc: chat error: tokenizer file required. Specify --tokenizer <path>.");
                return EXIT_IO_ERROR;
            };

            let _ = writeln!(stdout, "\x1b[1;36m================================================================================\x1b[0m");
            let _ = writeln!(stdout, "\x1b[1;36m       VELOX NATIVE BARE-METAL CODING REPL (ZERO-PYTHON PURE SILICON)           \x1b[0m");
            let _ = writeln!(stdout, "\x1b[1;36m================================================================================\x1b[0m");
            let _ = writeln!(stdout, "  Target Model   : {} (100% Zero-Dynamic-Allocation DAG)", first_model.name);
            let _ = writeln!(stdout, "  Active Backend : {:?}", effective_backend);
            let _ = writeln!(stdout, "  Runtime Engine : 100% Pure Native Velox (0 Python, 0 Ollama, 0 PyTorch)");
            let _ = writeln!(stdout, "\x1b[36m--------------------------------------------------------------------------------\x1b[0m");
            let _ = writeln!(stdout, "  Commands: /save <file>, /clear, /exit");
            let _ = writeln!(stdout, "\x1b[1;36m================================================================================\x1b[0m\n");

            let runner_path = if effective_backend == ExecutionBackend::Cuda || effective_backend == ExecutionBackend::C99 {
                match find_runner_binary(runner_name) {
                    Some(p) => Some(p),
                    None => {
                        let _ = writeln!(stderr, "veloxc: runner executable '{}' not found", runner_name);
                        return EXIT_IO_ERROR;
                    }
                }
            } else {
                None
            };

            let mut latest_response = String::new();

            // Keep one runner alive for the whole session when the backend
            // supports it; fall back to the one-shot path otherwise.
            // K2 is a reasoning model: it narrates a long thinking block before
            // answering, so a small step budget truncates mid-sentence. Stop
            // tokens make a generous budget safe -- generation ends exactly at
            // the turn boundary instead of running to the cap.
            let chat_steps = cli_args.steps.max(512);
            let stop_ids: Vec<u32> = tok
                .special_tokens
                .iter()
                .filter(|(text, _)| text.contains("im_end") || text.contains("endoftext"))
                .map(|(_, id)| *id)
                .collect();

            // Greedy argmax makes a reasoning model loop inside its thinking
            // block, so chat defaults to sampling. Explicit flags still win.
            let chat_temp = if cli_args.temperature > 1e-5 { cli_args.temperature } else { 0.7 };
            let chat_top_k = if cli_args.top_k > 0 { cli_args.top_k } else { 40 };
            let chat_min_p = if cli_args.min_p > 0.0 { cli_args.min_p } else { 0.05 };

            let mut persistent: Option<PersistentRunner> = match (effective_backend, runner_path.as_ref()) {
                (ExecutionBackend::Cuda, Some(path)) => PersistentRunner::spawn(
                    path,
                    &weights_file,
                    chat_steps,
                    &stop_ids,
                    chat_temp,
                    chat_top_k,
                    chat_min_p,
                    cli_args.seed,
                )
                .ok(),
                _ => None,
            };
            if persistent.is_none() && effective_backend == ExecutionBackend::Cuda {
                let _ = writeln!(
                    stderr,
                    "veloxc: persistent runner unavailable, falling back to one process per turn"
                );
            }

            loop {
                let _ = write!(stdout, "\x1b[1;36mvelox[{}]> \x1b[0m", first_model.name);
                let _ = stdout.flush();

                let mut input = String::new();
                if std::io::stdin().read_line(&mut input).is_err() || input.is_empty() {
                    let _ = writeln!(stdout, "\n\x1b[33m[*] Exiting Velox REPL. Goodbye!\x1b[0m");
                    break;
                }

                let trimmed = input.trim();
                if trimmed.is_empty() {
                    continue;
                }

                if trimmed == "/exit" || trimmed == "/quit" {
                    let _ = writeln!(stdout, "\x1b[33m[*] Exiting Velox REPL. Goodbye!\x1b[0m");
                    break;
                } else if trimmed == "/clear" {
                    latest_response.clear();
                    let _ = writeln!(stdout, "\x1b[32m[+] Context cleared.\x1b[0m");
                    continue;
                } else if trimmed.starts_with("/save ") {
                    let path = trimmed[6..].trim();
                    if latest_response.is_empty() {
                        let _ = writeln!(stderr, "\x1b[31m[-] No response generated yet to save.\x1b[0m");
                    } else {
                        match fs::write(path, &latest_response) {
                            Ok(_) => {
                                let _ = writeln!(stdout, "\x1b[32m[+] Response saved to '{}' ({} bytes)\x1b[0m", path, latest_response.len());
                            }
                            Err(e) => {
                                let _ = writeln!(stderr, "\x1b[31m[-] Failed to write file: {}\x1b[0m", e);
                            }
                        }
                    }
                    continue;
                }

                let prompt_str = if trimmed.starts_with("def ") || trimmed.starts_with("fn ") || trimmed.starts_with("class ") || trimmed.starts_with("struct ") {
                    trimmed.to_string()
                } else if is_k2 {
                    format!("<|ifm|im_start|>user\n{}<|ifm|im_end|>\n<|ifm|im_start|>assistant\n<ifm|think>\n", trimmed)
                } else {
                    format!("USER: {}\nASSISTANT: ", trimmed)
                };

                let mut turn_tokens = tok.encode(&prompt_str);
                let bos_tok = if is_k2 { 0u32 } else { 1u32 };
                if !turn_tokens.starts_with(&[bos_tok]) && (!prompt_str.starts_with("<|") || is_k2) {
                    turn_tokens.insert(0, bos_tok);
                }

                let _ = write!(stdout, "\n\x1b[1;32mvelox-assistant> \x1b[0m");
                let _ = stdout.flush();

                if let Some(ref mut session) = persistent {
                    match session.generate(&turn_tokens) {
                        Ok(ids) => {
                            // Decode incrementally so a stop string can cut the
                            // reply at the exact match, mirroring llama.cpp's
                            // text-level stop handling.
                            let mut decoder = tok.streaming_decoder();
                            let mut text = String::new();
                            let mut stopped = false;
                            for id in &ids {
                                text.push_str(&decoder.decode_token(*id));
                                if find_stop_string(&text, &cli_args.stop_strings).is_some() {
                                    stopped = true;
                                    break;
                                }
                            }
                            if !stopped {
                                text.push_str(&decoder.flush());
                            } else if let Some(pos) =
                                find_stop_string(&text, &cli_args.stop_strings)
                            {
                                text.truncate(pos);
                            }
                            latest_response = if cli_args.hide_thinking {
                                strip_thinking(&text)
                            } else {
                                text
                            };
                            let _ = write!(stdout, "{}", latest_response);
                            let _ = stdout.flush();
                        }
                        Err(err) => {
                            let _ = writeln!(stderr, "veloxc: persistent runner error: {err}");
                        }
                    }
                } else if let Some(ref r_path) = runner_path {
                    let streaming_decoder = tok.streaming_decoder();
                    let _ = run_external_runner(
                        r_path,
                        &weights_file,
                        &turn_tokens,
                        cli_args.steps,
                        cli_args.spec,
                        cli_args.threads,
                        Some(streaming_decoder),
                        false,
                        false,
                        stdout,
                        stderr,
                    );
                }
                let _ = writeln!(stdout);
            }
            return EXIT_SUCCESS;
        }

        // Multi-backend dispatch: External Standalone Runner (CUDA / C99)
        if effective_backend == ExecutionBackend::Cuda || effective_backend == ExecutionBackend::C99 {
            // Speculative drafting is only implemented in the K2 CUDA runner's
            // verify path; warn instead of silently swallowing the flags.
            if cli_args.spec.speculative && is_k2 && effective_backend != ExecutionBackend::Cuda {
                let _ = writeln!(
                    stderr,
                    "veloxc: warning: --speculative requires the CUDA backend; the {:?} backend ignores it.",
                    effective_backend
                );
            }
            if !cli_args.sampling_params().is_greedy() {
                let _ = writeln!(
                    stderr,
                    "veloxc: warning: sampling flags require --backend rust; the {:?} backend uses greedy argmax.",
                    effective_backend
                );
            }
            let runner_path = match find_runner_binary(runner_name) {
                Some(p) => p,
                None => {
                    let _ = writeln!(
                        stderr,
                        "veloxc: runtime error: runner executable '{}' not found. Please compile runners using src/runners/build_runners.bat or use --backend rust.",
                        runner_name
                    );
                    return EXIT_IO_ERROR;
                }
            };

            let streaming_decoder_opt = tokenizer_opt.as_ref().map(|t| t.streaming_decoder());

            match run_external_runner(
                &runner_path,
                &weights_file,
                &prompt_tokens,
                cli_args.steps,
                cli_args.spec,
                cli_args.threads,
                streaming_decoder_opt,
                cli_args.verbose,
                cli_args.benchmark,
                stdout,
                stderr,
            ) {
                Ok(code) => return code,
                Err(code) => return code,
            }
        }

        // Native In-Process Rust Execution Engine
        let max_seq_required = (prompt_tokens.len() + cli_args.steps + 16).max(64);
        let config = TransformerConfig::from_typed_model(first_model, max_seq_required);

        let mut engine = match VeloxStaticExecutionEngine::load_safetensors(&weights_file, config) {
            Ok(eng) => eng,
            Err(err) => {
                let _ = writeln!(
                    stderr,
                    "veloxc: runtime error loading safetensors '{}': {}",
                    weights_file, err
                );
                return EXIT_IO_ERROR;
            }
        };

        // Warmup iterations if requested. A session is built once so the
        // sampler scratch and history allocations are excluded from the
        // measured generation loop.
        let sampling_params = cli_args.sampling_params();
        if let Err(err) = sampling_params.validate() {
            let _ = writeln!(stderr, "veloxc: invalid sampling configuration: {err}");
            return EXIT_USAGE_ERROR;
        }
        let history_capacity = prompt_tokens.len().saturating_add(cli_args.steps);
        let mut sampler_session = match SamplerSession::new(
            sampling_params,
            engine.config.vocab_size,
            history_capacity,
        ) {
            Ok(session) => session,
            Err(err) => {
                let _ = writeln!(stderr, "veloxc: failed to initialize sampler: {err}");
                return EXIT_USAGE_ERROR;
            }
        };

        if cli_args.warmup > 0 {
            let mut warmup_buf = vec![0u32; cli_args.steps];
            for _ in 0..cli_args.warmup {
                engine.generate_sampled_streaming_with_session(
                    &prompt_tokens,
                    cli_args.steps,
                    &mut sampler_session,
                    &mut warmup_buf,
                    |_, _| {},
                );
            }
        }

        let mut generated_tokens = vec![0u32; cli_args.steps];
        let t_start = std::time::Instant::now();
        let mut t_prefill_stop = None;

        let (written, inference_heap_allocs) = crate::alloc_counter::count_allocations(|| {
            engine.generate_sampled_streaming_with_session(
                &prompt_tokens,
                cli_args.steps,
                &mut sampler_session,
                &mut generated_tokens,
                |step, _tok_id| {
                    if step == 0 {
                        t_prefill_stop = Some(std::time::Instant::now());
                    }
                },
            )
        });
        generated_tokens.truncate(written);
        let t_end = std::time::Instant::now();

        // Real-time streaming output to stdout
        if let Some(ref tok) = tokenizer_opt {
            let mut dec = tok.streaming_decoder();
            for &tok_id in &generated_tokens {
                let text_chunk = dec.decode_token(tok_id);
                let _ = write!(stdout, "{text_chunk}");
                let _ = stdout.flush();
            }
            let flush_chunk = dec.flush();
            if !flush_chunk.is_empty() {
                let _ = write!(stdout, "{flush_chunk}");
                let _ = stdout.flush();
            }
            let _ = writeln!(stdout);
        } else if cli_args.verbose {
            for (step, &tok_id) in generated_tokens.iter().enumerate() {
                let _ = writeln!(stdout, "[Step {step}] Token: {tok_id}");
                let _ = stdout.flush();
            }
        }

        let prefill_ms = t_prefill_stop
            .unwrap_or(t_end)
            .duration_since(t_start)
            .as_secs_f64()
            * 1000.0;
        let total_ms = t_end.duration_since(t_start).as_secs_f64() * 1000.0;
        let decode_ms = if total_ms > prefill_ms { total_ms - prefill_ms } else { 0.0 };
        let decode_tok_per_sec = if cli_args.steps > 1 && decode_ms > 0.0 {
            ((cli_args.steps - 1) as f64 * 1000.0) / decode_ms
        } else {
            0.0
        };

        let mut full_sequence = Vec::with_capacity(prompt_tokens.len() + generated_tokens.len());
        full_sequence.extend_from_slice(&prompt_tokens);
        full_sequence.extend_from_slice(&generated_tokens);

        // Find top-5 logits from final token step
        let mut top5: Vec<(u32, f32)> = engine
            .logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as u32, v))
            .collect();
        top5.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        top5.truncate(5);

        const EXPECTED_32: [u32; 32] = [
            359, 253, 4028, 282, 260, 1773, 30, 198, 26501, 368, 281, 260, 1773, 198, 20255, 3726,
            288, 9517, 253, 12301, 198, 19161, 354, 12301, 198, 19161, 354, 12301, 198, 2683, 416, 722,
        ];
        let match_4 = prompt_tokens == [1, 504, 346]
            && generated_tokens.len() >= 4
            && generated_tokens[..4] == EXPECTED_32[..4];
        let match_32 = prompt_tokens == [1, 504, 346]
            && generated_tokens.len() >= 32
            && generated_tokens[..32] == EXPECTED_32[..32];

        let mut report = String::new();
        let _ = writeln!(report, "================================================================================");
        if cli_args.benchmark {
            let _ = writeln!(report, "VELOX SOTA ROOFLINE INFERENCE BENCHMARK REPORT");
        } else {
            let _ = writeln!(report, "VELOX ZERO-COPY NATIVE RUNTIME EXECUTION REPORT");
        }
        let _ = writeln!(report, "================================================================================");
        let _ = writeln!(report, "Model                  : {}", first_model.name);
        let _ = writeln!(report, "Backend                : Native Rust AVX2/FMA3 on Host CPU");
        let _ = writeln!(report, "Safetensors Weights    : {} (Zero-Copy Win32 Memory Mapped)", weights_file);
        let _ = writeln!(
            report,
            "Prompt Tokens ({})      : {:?}",
            prompt_tokens.len(),
            prompt_tokens
        );
        let _ = writeln!(
            report,
            "Generated Tokens ({})   : {:?}",
            generated_tokens.len(),
            generated_tokens
        );
        let _ = writeln!(report, "Full Token Sequence    : {:?}", full_sequence);
        if let Some(ref tok) = tokenizer_opt {
            let decoded_text = tok.decode(&generated_tokens);
            let _ = writeln!(report, "Decoded Output Text    : {:?}", decoded_text);
        }
        if cli_args.benchmark {
            let _ = writeln!(report, "TTFT (Prefill Latency) : {:.3} ms", prefill_ms);
            if cli_args.steps > 1 {
                let _ = writeln!(
                    report,
                    "Decode Latency ({} tok): {:.3} ms ({:.2} tok/s, {:.3} ms/tok)",
                    cli_args.steps - 1,
                    decode_ms,
                    decode_tok_per_sec,
                    decode_ms / (cli_args.steps - 1) as f64
                );
            }
            let _ = writeln!(report, "Total Inference Time   : {:.3} ms", total_ms);
        }
        let _ = writeln!(
            report,
            "Dynamic Heap Allocs    : {} (measured by Velox counting allocator over prefill + decode)",
            inference_heap_allocs
        );
        let _ = writeln!(report, "Final Step Top-5 Logits:");
        for (rank, (tok_id, logit_val)) in top5.iter().enumerate() {
            let _ = writeln!(report, "  #{}: token_id={} logit={:.6}", rank + 1, tok_id, logit_val);
        }
        if cli_args.steps >= 4 {
            let parity_label = if cli_args.sampling_params().is_greedy() {
                if match_4 { "PASSED [359, 253, 4028, 282]" } else { "FAILED" }
            } else {
                "SKIPPED (stochastic sampling)"
            };
            let _ = writeln!(report, "Parity 4-token check   : {parity_label}");
        }
        if cli_args.steps >= 32 {
            let parity_label = if cli_args.sampling_params().is_greedy() {
                if match_32 { "PASSED (Exact match with Velox Rust CPU reference)" } else { "FAILED" }
            } else {
                "SKIPPED (stochastic sampling)"
            };
            let _ = writeln!(
                report,
                "Parity 32-token check  : {parity_label}"
            );
        }
        let _ = writeln!(report, "================================================================================");

        if let Err(io_code) = write_output_text(&report, cli_args.output.as_deref(), stdout, stderr) {
            return io_code;
        }
        return EXIT_SUCCESS;
    }

    // 13. Check Mode (--check, verbose, or default clean validation)
    if cli_args.verbose || !cli_args.check {
        let _ = writeln!(
            stdout,
            "veloxc: validation successful for {} (0 errors).",
            cli_args.input_file
        );
    }

    EXIT_SUCCESS
}

fn write_output_text(
    content: &str,
    output_path_opt: Option<&str>,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> Result<(), i32> {
    if let Some(out_path) = output_path_opt {
        let p = Path::new(out_path);
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() && !parent.exists() {
                if let Err(err) = fs::create_dir_all(parent) {
                    let _ = writeln!(
                        stderr,
                        "veloxc: I/O error: could not write output to '{}': {}",
                        out_path, err
                    );
                    return Err(EXIT_IO_ERROR);
                }
            }
        }
        if let Err(err) = fs::write(p, content) {
            let _ = writeln!(
                stderr,
                "veloxc: I/O error: could not write output to '{}': {}",
                out_path, err
            );
            return Err(EXIT_IO_ERROR);
        }
    } else {
        let _ = write!(stdout, "{content}");
    }
    Ok(())
}

// ============================================================================
// Unit Tests for CLI Arguments & Formatting
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_help_options() {
        assert_eq!(
            parse_cli_args(&["-h".to_string()]),
            Ok(CliAction::Help)
        );
        assert_eq!(
            parse_cli_args(&["--help".to_string()]),
            Ok(CliAction::Help)
        );
    }

    #[test]
    fn test_parse_version_options() {
        assert_eq!(
            parse_cli_args(&["-v".to_string()]),
            Ok(CliAction::Version)
        );
        assert_eq!(
            parse_cli_args(&["--version".to_string()]),
            Ok(CliAction::Version)
        );
    }

    #[test]
    fn test_parse_valid_flags() {
        let args = vec![
            "--check".to_string(),
            "--no-color".to_string(),
            "-v".to_string(),
            "model.vlx".to_string(),
        ];
        match parse_cli_args(&args) {
            Ok(CliAction::Run(cli)) => {
                assert_eq!(cli.input_file, "model.vlx");
                assert!(cli.check);
                assert!(cli.no_color);
                assert!(cli.verbose);
            }
            other => panic!("Unexpected action: {other:?}"),
        }
    }

    #[test]
    fn test_parse_dump_ast_with_format_and_output() {
        let args = vec![
            "--dump-ast".to_string(),
            "--format".to_string(),
            "json".to_string(),
            "-o".to_string(),
            "out.json".to_string(),
            "script.vlx".to_string(),
        ];
        match parse_cli_args(&args) {
            Ok(CliAction::Run(cli)) => {
                assert!(cli.dump_ast);
                assert_eq!(cli.format, OutputFormat::Json);
                assert_eq!(cli.output, Some("out.json".to_string()));
                assert_eq!(cli.input_file, "script.vlx");
            }
            other => panic!("Unexpected action: {other:?}"),
        }
    }

    #[test]
    fn test_parse_invalid_choice_format() {
        let args = vec!["--format".to_string(), "yaml".to_string(), "test.vlx".to_string()];
        assert!(parse_cli_args(&args).is_err());
    }

    #[test]
    fn test_parse_missing_input_file() {
        assert!(parse_cli_args(&[]).is_err());
    }

    #[test]
    fn test_parse_backend_options() {
        let args_cuda = vec!["--run".to_string(), "--backend".to_string(), "cuda".to_string(), "model.vlx".to_string()];
        match parse_cli_args(&args_cuda) {
            Ok(CliAction::Run(cli)) => {
                assert!(cli.run_model);
                assert_eq!(cli.backend, ExecutionBackend::Cuda);
            }
            other => panic!("Unexpected action: {other:?}"),
        }

        let args_c99 = vec!["--run".to_string(), "--backend=c99".to_string(), "model.vlx".to_string()];
        match parse_cli_args(&args_c99) {
            Ok(CliAction::Run(cli)) => {
                assert!(cli.run_model);
                assert_eq!(cli.backend, ExecutionBackend::C99);
            }
            other => panic!("Unexpected action: {other:?}"),
        }

        let args_rust = vec!["--run".to_string(), "--backend".to_string(), "rust".to_string(), "model.vlx".to_string()];
        match parse_cli_args(&args_rust) {
            Ok(CliAction::Run(cli)) => {
                assert!(cli.run_model);
                assert_eq!(cli.backend, ExecutionBackend::Rust);
            }
            other => panic!("Unexpected action: {other:?}"),
        }
    }

    #[test]
    fn test_parse_prompt_and_benchmark_options() {
        let args = vec![
            "--run".to_string(),
            "--backend".to_string(),
            "rust".to_string(),
            "--prompt".to_string(),
            "The you".to_string(),
            "--steps".to_string(),
            "4".to_string(),
            "--benchmark".to_string(),
            "--warmup".to_string(),
            "1".to_string(),
            "--threads".to_string(),
            "8".to_string(),
            "examples/tinystories_smollm.vlx".to_string(),
        ];
        match parse_cli_args(&args) {
            Ok(CliAction::Run(cli)) => {
                assert!(cli.run_model);
                assert_eq!(cli.backend, ExecutionBackend::Rust);
                assert_eq!(cli.prompt_text, Some("The you".to_string()));
                assert_eq!(cli.steps, 4);
                assert!(cli.benchmark);
                assert_eq!(cli.warmup, 1);
                assert_eq!(cli.threads, Some(8));
                assert_eq!(cli.input_file, "examples/tinystories_smollm.vlx");
            }
            other => panic!("Unexpected action: {other:?}"),
        }
    }

    #[test]
    fn test_parse_speculative_options() {
        let args = vec![
            "--run".to_string(),
            "--backend".to_string(),
            "cuda".to_string(),
            "--speculative".to_string(),
            "--spec-ema".to_string(),
            "0.35".to_string(),
            "model.vlx".to_string(),
        ];
        match parse_cli_args(&args) {
            Ok(CliAction::Run(cli)) => {
                assert!(cli.spec.speculative);
                assert_eq!(cli.spec.draft_len, None);
                assert_eq!(cli.spec.spec_ema, Some(0.35));
            }
            other => panic!("Unexpected action: {other:?}"),
        }

        let args_k = vec![
            "--run".to_string(),
            "--backend=cuda".to_string(),
            "--speculative=5".to_string(),
            "model.vlx".to_string(),
        ];
        match parse_cli_args(&args_k) {
            Ok(CliAction::Run(cli)) => {
                assert!(cli.spec.speculative);
                assert_eq!(cli.spec.draft_len, Some(5));
                assert_eq!(cli.spec.spec_ema, None);
            }
            other => panic!("Unexpected action: {other:?}"),
        }

        // --draft-len is a pure width selector and does not enable drafting.
        let args_len = vec![
            "--run".to_string(),
            "--backend=cuda".to_string(),
            "--draft-len".to_string(),
            "7".to_string(),
            "model.vlx".to_string(),
        ];
        match parse_cli_args(&args_len) {
            Ok(CliAction::Run(cli)) => {
                assert!(!cli.spec.speculative);
                assert_eq!(cli.spec.draft_len, Some(7));
            }
            other => panic!("Unexpected action: {other:?}"),
        }

        // Validation boundaries: draft width, EMA floor, and error paths.
        assert!(parse_cli_args(&["--run".to_string(), "--speculative".to_string(), "0".to_string(), "model.vlx".to_string()]).is_err());
        assert!(parse_cli_args(&["--run".to_string(), "--speculative".to_string(), "8".to_string(), "model.vlx".to_string()]).is_err());
        assert!(parse_cli_args(&["--run".to_string(), "--draft-len".to_string(), "9".to_string(), "model.vlx".to_string()]).is_err());
        assert!(parse_cli_args(&["--run".to_string(), "--draft-len".to_string(), "model.vlx".to_string()]).is_err());
        assert!(parse_cli_args(&["--run".to_string(), "--spec-ema".to_string(), "1.5".to_string(), "model.vlx".to_string()]).is_err());
        assert!(parse_cli_args(&["--run".to_string(), "--spec-ema".to_string(), "-0.1".to_string(), "model.vlx".to_string()]).is_err());
        assert!(parse_cli_args(&["--run".to_string(), "--spec-ema".to_string(), "abc".to_string(), "model.vlx".to_string()]).is_err());
        assert!(parse_cli_args(&["--run".to_string(), "--spec-ema".to_string(), "model.vlx".to_string()]).is_err());

        // Boundary successes: K=1, K=7, EMA floors 0.0 and 1.0.
        let args_bounds = vec![
            "--run".to_string(),
            "--backend=cuda".to_string(),
            "--speculative".to_string(),
            "1".to_string(),
            "--draft-len=7".to_string(),
            "--spec-ema=1.0".to_string(),
            "model.vlx".to_string(),
        ];
        match parse_cli_args(&args_bounds) {
            Ok(CliAction::Run(cli)) => {
                assert!(cli.spec.speculative);
                assert_eq!(cli.spec.draft_len, Some(7));
                assert_eq!(cli.spec.spec_ema, Some(1.0));
            }
            other => panic!("Unexpected action: {other:?}"),
        }

        let args_zero = vec![
            "--run".to_string(),
            "--backend=cuda".to_string(),
            "--spec-ema=0.0".to_string(),
            "model.vlx".to_string(),
        ];
        match parse_cli_args(&args_zero) {
            Ok(CliAction::Run(cli)) => {
                assert_eq!(cli.spec.spec_ema, Some(0.0));
            }
            other => panic!("Unexpected action: {other:?}"),
        }

        // Optional numeric is not consumed when the next token is the
        // positional input file or another flag.
        let args_fallthrough = vec![
            "--run".to_string(),
            "--backend=cuda".to_string(),
            "--speculative".to_string(),
            "model.vlx".to_string(),
        ];
        match parse_cli_args(&args_fallthrough) {
            Ok(CliAction::Run(cli)) => {
                assert!(cli.spec.speculative);
                assert_eq!(cli.spec.draft_len, None);
                assert_eq!(cli.input_file, "model.vlx");
            }
            other => panic!("Unexpected action: {other:?}"),
        }

        let args_flag_next = vec![
            "--run".to_string(),
            "--backend=cuda".to_string(),
            "--speculative".to_string(),
            "--spec-ema".to_string(),
            "0.2".to_string(),
            "model.vlx".to_string(),
        ];
        match parse_cli_args(&args_flag_next) {
            Ok(CliAction::Run(cli)) => {
                assert!(cli.spec.speculative);
                assert_eq!(cli.spec.draft_len, None);
                assert_eq!(cli.spec.spec_ema, Some(0.2));
            }
            other => panic!("Unexpected action: {other:?}"),
        }
    }
}
