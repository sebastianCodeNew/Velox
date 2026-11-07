//! Velox Compiler Frontend CLI (`veloxc`)
//!
//! Native optimizing compiler frontend for the Velox LLM serving language.
//! Pure `std` implementation with zero external dependencies.

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let exit_code = veloxc::cli::run_cli(&args);
    std::process::exit(exit_code);
}
