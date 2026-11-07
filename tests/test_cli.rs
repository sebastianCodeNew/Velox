//! Integration test suite for the Velox standalone compiler frontend CLI (`tests/test_cli.rs`)
//!
//! Verifies:
//! - Commands and flags:
//!   - `--version` / `-v`: prints compiler version and exits with EXIT_SUCCESS (0).
//!   - `--help` / `-h`: prints usage manual and exits with EXIT_SUCCESS (0).
//!   - `--check`: validates syntax and semantics without dumping AST.
//!   - `--dump-ast --format text`: pretty-printed hierarchical AST serialization.
//!   - `--dump-ast --format json`: machine-readable pure-Rust AST JSON export.
//!   - `--dump-tokens`: token stream formatting with coordinates and spans.
//!   - `-o` / `--output`: writing AST or token stream to target file.
//!   - `--no-color`: suppressing ANSI escape color codes in diagnostics.
//!   - `-v` / `--verbose`: verbose validation logs.
//! - Deterministic exit codes:
//!   - 0: EXIT_SUCCESS on valid compilation / checking / help / version.
//!   - 1: EXIT_COMPILATION_ERROR on syntax, lexical, or semantic invariant failure (E001–E025).
//!   - 2: EXIT_IO_ERROR on missing input file, directory as file, or unwritable output.
//!   - 3: EXIT_USAGE_ERROR on invalid options, missing mandatory arguments, or bad formats.
//! - Subprocess execution of the native `veloxc.exe` binary.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use veloxc::cli::{
    run_cli_with_io, EXIT_COMPILATION_ERROR, EXIT_IO_ERROR, EXIT_SUCCESS, EXIT_USAGE_ERROR,
};

const MINIMAL_VALID_VLX: &str = r#"
target hardware {
    device: "sm_86";
    vram_capacity: 4096 * MiB;
    compute_capability: 8.6;
    safety_factor: 0.95;
}

model MinimalModel {
    scope prefill() -> u32 {
        return 0;
    }
    scope decode() -> u32 {
        return 0;
    }
}
"#;

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(name: &str) -> Self {
        let unique = format!(
            "velox_cli_test_{}_{}_{}",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let path = std::env::temp_dir().join(unique);
        fs::create_dir_all(&path).expect("Failed to create temp dir");
        Self { path }
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn file_path(&self, filename: &str) -> PathBuf {
        self.path.join(filename)
    }

    fn write_file(&self, filename: &str, content: &str) -> PathBuf {
        let p = self.file_path(filename);
        fs::write(&p, content).expect("Failed to write temp file");
        p
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn call_cli(args: &[&str]) -> (i32, String, String) {
    let owned_args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let mut stdout_buf = Vec::new();
    let mut stderr_buf = Vec::new();
    let code = run_cli_with_io(&owned_args, &mut stdout_buf, &mut stderr_buf);
    let stdout = String::from_utf8_lossy(&stdout_buf).to_string();
    let stderr = String::from_utf8_lossy(&stderr_buf).to_string();
    (code, stdout, stderr)
}

fn find_binary_path() -> PathBuf {
    let debug_path = Path::new("target").join("debug").join("veloxc.exe");
    if debug_path.exists() {
        return debug_path;
    }
    let release_path = Path::new("target").join("release").join("veloxc.exe");
    if release_path.exists() {
        return release_path;
    }
    let root_path = Path::new("veloxc.exe");
    if root_path.exists() {
        return root_path.to_path_buf();
    }
    panic!("veloxc.exe binary not found for subprocess testing");
}

// ============================================================================
// 1. Version & Basic Help Options
// ============================================================================

#[test]
fn test_cli_version_direct() {
    let (code, stdout, _) = call_cli(&["--version"]);
    assert_eq!(code, EXIT_SUCCESS);
    assert!(stdout.contains(&format!("veloxc {}", veloxc::VERSION)));

    let (code_v, stdout_v, _) = call_cli(&["-v"]);
    assert_eq!(code_v, EXIT_SUCCESS);
    assert!(stdout_v.contains(&format!("veloxc {}", veloxc::VERSION)));
}

#[test]
fn test_cli_help_direct() {
    let (code, stdout, _) = call_cli(&["--help"]);
    assert_eq!(code, EXIT_SUCCESS);
    assert!(stdout.contains("usage: veloxc"));
    assert!(stdout.contains("--dump-ast"));
    assert!(stdout.contains("--check"));

    let (code_h, stdout_h, _) = call_cli(&["-h"]);
    assert_eq!(code_h, EXIT_SUCCESS);
    assert!(stdout_h.contains("usage: veloxc"));
}

// ============================================================================
// 2. Exit Code 3: CLI Usage Errors (Invalid options, missing arguments)
// ============================================================================

#[test]
fn test_cli_missing_input_file_direct() {
    let (code, _, stderr) = call_cli(&[]);
    assert_eq!(code, EXIT_USAGE_ERROR);
    assert!(stderr.contains("error: the following arguments are required: input_file"));
}

#[test]
fn test_cli_invalid_option_direct() {
    let (code, _, stderr) = call_cli(&["--non-existent-option"]);
    assert_eq!(code, EXIT_USAGE_ERROR);
    assert!(stderr.contains("error: unrecognized arguments"));
}

#[test]
fn test_cli_invalid_format_choice_direct() {
    let temp = TempDir::new("invalid_fmt");
    let file = temp.write_file("test.vlx", MINIMAL_VALID_VLX);
    let (code, _, stderr) = call_cli(&["--format", "yaml", file.to_str().unwrap()]);
    assert_eq!(code, EXIT_USAGE_ERROR);
    assert!(stderr.to_lowercase().contains("invalid choice"));
}

#[test]
fn test_cli_missing_output_arg_direct() {
    let (code, _, stderr) = call_cli(&["-o"]);
    assert_eq!(code, EXIT_USAGE_ERROR);
    assert!(stderr.contains("flag '-o' requires an argument"));
}

// ============================================================================
// 3. Exit Code 2: File I/O Errors (Non-existent files, unreadable files)
// ============================================================================

#[test]
fn test_cli_non_existent_file_direct() {
    let (code, _, stderr) = call_cli(&["this_file_does_not_exist_98765.vlx"]);
    assert_eq!(code, EXIT_IO_ERROR);
    assert!(stderr.to_lowercase().contains("input file not found"));
}

#[test]
fn test_cli_directory_given_as_input_file_direct() {
    let temp = TempDir::new("dir_as_file");
    let (code, _, stderr) = call_cli(&[temp.path().to_str().unwrap()]);
    assert_eq!(code, EXIT_IO_ERROR);
    assert!(stderr.to_lowercase().contains("not a regular file"));
}

#[test]
fn test_cli_unwritable_output_path_direct() {
    let temp = TempDir::new("unwritable_out");
    let vlx_file = temp.write_file("valid.vlx", MINIMAL_VALID_VLX);

    // Creating an existing directory with the output path causes write failure
    let unwritable_dir = temp.file_path("unwritable_dir");
    fs::create_dir_all(&unwritable_dir).unwrap();

    let (code, _, stderr) = call_cli(&[
        "--dump-ast",
        vlx_file.to_str().unwrap(),
        "-o",
        unwritable_dir.to_str().unwrap(),
    ]);
    assert_eq!(code, EXIT_IO_ERROR);
    assert!(stderr.to_lowercase().contains("could not write output"));
}

// ============================================================================
// 4. Exit Code 1: Compilation Errors (Syntax, Lexer, Invariants)
// ============================================================================

#[test]
fn test_cli_syntax_error_direct() {
    let temp = TempDir::new("bad_syntax");
    let bad_file = temp.write_file("bad_syntax.vlx", "target hardware { let x = ; }");

    let (code, _, stderr) = call_cli(&[bad_file.to_str().unwrap()]);
    assert_eq!(code, EXIT_COMPILATION_ERROR);
    assert!(stderr.contains("VELOX COMPILER ERROR"));
}

#[test]
fn test_cli_lexer_error_direct() {
    let temp = TempDir::new("bad_lexer");
    let bad_file = temp.write_file("bad_lexer.vlx", "\"unterminated string literal\n");

    let (code, _, stderr) = call_cli(&[bad_file.to_str().unwrap()]);
    assert_eq!(code, EXIT_COMPILATION_ERROR);
    assert!(stderr.to_lowercase().contains("unterminated string literal"));
}

#[test]
fn test_cli_invariant_error_missing_decode_direct() {
    let temp = TempDir::new("missing_decode");
    let bad_file = temp.write_file(
        "missing_decode.vlx",
        r#"
        target hardware { device: "sm_86"; vram_capacity: 4096 * MiB; compute_capability: 8.6; safety_factor: 0.95; }
        model Incomplete {
            scope prefill() -> u32 { return 0; }
            // missing decode scope triggers E008
        }
        "#,
    );

    let (code, _, stderr) = call_cli(&[bad_file.to_str().unwrap()]);
    assert_eq!(code, EXIT_COMPILATION_ERROR);
    assert!(stderr.contains("E008"));
}

// ============================================================================
// 5. Exit Code 0: Validation & Checking (--check, default)
// ============================================================================

#[test]
fn test_cli_check_valid_file_direct() {
    let temp = TempDir::new("check_valid");
    let file = temp.write_file("minimal.vlx", MINIMAL_VALID_VLX);

    let (code, _, stderr) = call_cli(&["--check", file.to_str().unwrap()]);
    assert_eq!(code, EXIT_SUCCESS);
    assert_eq!(stderr, "");
}

#[test]
fn test_cli_check_valid_file_verbose_direct() {
    let temp = TempDir::new("check_verbose");
    let file = temp.write_file("minimal.vlx", MINIMAL_VALID_VLX);

    let (code, stdout, _) = call_cli(&["--check", "-v", file.to_str().unwrap()]);
    assert_eq!(code, EXIT_SUCCESS);
    assert!(stdout.to_lowercase().contains("validation successful"));
}

#[test]
fn test_cli_default_run_on_reference_models() {
    let models = [
        "examples/tinystories_smollm.vlx",
        "examples/llama3_8b.vlx",
        "examples/qwen2_5_7b.vlx",
        "examples/deepseek_v3_mla.vlx",
        "examples/nanbeige_33b.vlx",
    ];

    for m in models {
        let (code, stdout, stderr) = call_cli(&[m]);
        assert_eq!(code, EXIT_SUCCESS, "Model failed in default CLI run: {m}");
        assert_eq!(stderr, "", "Stderr should be empty for clean model {m}");
        assert!(
            stdout.to_lowercase().contains("validation successful"),
            "Stdout should report success for {m}"
        );
    }
}

// ============================================================================
// 6. AST Dumping (--dump-ast, --format text/json, -o)
// ============================================================================

#[test]
fn test_cli_dump_ast_text_direct() {
    let temp = TempDir::new("dump_text");
    let file = temp.write_file("minimal.vlx", MINIMAL_VALID_VLX);

    let (code, stdout, stderr) = call_cli(&["--dump-ast", "--format", "text", file.to_str().unwrap()]);
    assert_eq!(code, EXIT_SUCCESS);
    assert_eq!(stderr, "");
    assert!(stdout.contains("ProgramNode"));
    assert!(stdout.contains("TargetHardwareNode"));
    assert!(stdout.contains("MinimalModel"));
}

#[test]
fn test_cli_dump_ast_json_direct() {
    let temp = TempDir::new("dump_json");
    let file = temp.write_file("minimal.vlx", MINIMAL_VALID_VLX);

    let (code, stdout, stderr) = call_cli(&["--dump-ast", "--format", "json", file.to_str().unwrap()]);
    assert_eq!(code, EXIT_SUCCESS);
    assert_eq!(stderr, "");
    assert!(stdout.contains("\"node_type\": \"ProgramNode\""));
    assert!(stdout.contains("\"device_name\": \"sm_86\""));
    assert!(stdout.contains("MinimalModel"));
}

#[test]
fn test_cli_dump_ast_to_output_file_direct() {
    let temp = TempDir::new("dump_out_file");
    let file = temp.write_file("minimal.vlx", MINIMAL_VALID_VLX);
    let out_file = temp.file_path("ast_output.json");

    let (code, _, stderr) = call_cli(&[
        "--dump-ast",
        "--format",
        "json",
        "-o",
        out_file.to_str().unwrap(),
        file.to_str().unwrap(),
    ]);
    assert_eq!(code, EXIT_SUCCESS);
    assert_eq!(stderr, "");
    assert!(out_file.exists());

    let content = fs::read_to_string(&out_file).unwrap();
    assert!(content.contains("\"node_type\": \"ProgramNode\""));
    assert!(content.contains("MinimalModel"));
}

// ============================================================================
// 7. Token Dumping (--dump-tokens, -o)
// ============================================================================

#[test]
fn test_cli_dump_tokens_direct() {
    let temp = TempDir::new("dump_tok");
    let file = temp.write_file("tokens_test.vlx", "let x = 42;");

    let (code, stdout, stderr) = call_cli(&["--dump-tokens", file.to_str().unwrap()]);
    assert_eq!(code, EXIT_SUCCESS);
    assert_eq!(stderr, "");
    assert!(stdout.contains("KW_LET"));
    assert!(stdout.contains("IDENTIFIER"));
    assert!(stdout.contains("DECIMAL_LITERAL"));
    assert!(stdout.contains("SEMICOLON"));
    assert!(stdout.contains("EOF"));
}

#[test]
fn test_cli_dump_tokens_to_output_file_direct() {
    let temp = TempDir::new("dump_tok_file");
    let file = temp.write_file(
        "tokens_test.vlx",
        "model Tiny { scope prefill() -> u32 { return 0; } scope decode() -> u32 { return 0; } }",
    );
    let tokens_out = temp.file_path("tokens.txt");

    let (code, _, stderr) = call_cli(&[
        "--dump-tokens",
        "-o",
        tokens_out.to_str().unwrap(),
        file.to_str().unwrap(),
    ]);
    assert_eq!(code, EXIT_SUCCESS);
    assert_eq!(stderr, "");
    assert!(tokens_out.exists());

    let content = fs::read_to_string(&tokens_out).unwrap();
    assert!(content.contains("KW_MODEL"));
    assert!(content.contains("KW_PREFILL"));
    assert!(content.contains("KW_DECODE"));
}

// ============================================================================
// 8. ANSI Color Suppression (--no-color)
// ============================================================================

#[test]
fn test_cli_no_color_flag_suppresses_ansi_direct() {
    let temp = TempDir::new("ansi_test");
    let bad_file = temp.write_file(
        "bad.vlx",
        "model Bad { scope prefill() -> u32 { let a = ; } scope decode() -> u32 { return 0; } }",
    );

    // With color: contains ANSI escape code "\x1b["
    let (code_color, _, stderr_color) = call_cli(&[bad_file.to_str().unwrap()]);
    assert_eq!(code_color, EXIT_COMPILATION_ERROR);
    assert!(stderr_color.contains("\x1b["));

    // With --no-color: ANSI escape code "\x1b[" is suppressed
    let (code_no_color, _, stderr_no_color) = call_cli(&["--no-color", bad_file.to_str().unwrap()]);
    assert_eq!(code_no_color, EXIT_COMPILATION_ERROR);
    assert!(!stderr_no_color.contains("\x1b["));
    assert!(stderr_no_color.contains("VELOX COMPILER ERROR"));
}

// ============================================================================
// 9. Subprocess Native Binary Tests
// ============================================================================

#[test]
fn test_cli_subprocess_version() {
    let binary = find_binary_path();
    let output = Command::new(&binary)
        .arg("--version")
        .output()
        .expect("Failed to execute veloxc binary");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains(&format!("veloxc {}", veloxc::VERSION)));
}

#[test]
fn test_cli_subprocess_reference_models() {
    let binary = find_binary_path();
    let models = [
        "examples/tinystories_smollm.vlx",
        "examples/llama3_8b.vlx",
        "examples/qwen2_5_7b.vlx",
        "examples/deepseek_v3_mla.vlx",
        "examples/nanbeige_33b.vlx",
    ];

    for m in models {
        let output = Command::new(&binary)
            .arg(m)
            .output()
            .unwrap_or_else(|e| panic!("Failed to run binary on {m}: {e}"));

        assert_eq!(
            output.status.code(),
            Some(0),
            "Subprocess execution of {m} failed with exit code {:?}\nstderr: {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
