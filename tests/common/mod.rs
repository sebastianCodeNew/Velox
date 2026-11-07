//! Shared helpers for Velox integration tests.

use std::path::Path;

/// Relative path (from the crate root, which is the cargo test working directory) of the
/// real open-weights checkpoint used by the end-to-end runtime tests.
pub const SMOLLM_WEIGHTS: &str = "smollm-135m.safetensors";

/// Environment variable that explicitly opts in to skipping real-weight tests when the
/// checkpoint is absent (e.g. on CI machines without the 538 MB file).
pub const ALLOW_MISSING_ENV: &str = "VELOX_ALLOW_MISSING_WEIGHTS";

/// Returns the SmolLM-135M checkpoint path, or fails the test loudly if it is missing.
///
/// A missing checkpoint must never silently turn a real-model test into a vacuous pass.
/// Set `VELOX_ALLOW_MISSING_WEIGHTS=1` to deliberately skip (returns `None`).
#[must_use]
pub fn require_smollm_weights() -> Option<&'static Path> {
    let path = Path::new(SMOLLM_WEIGHTS);
    if path.exists() {
        return Some(path);
    }
    if std::env::var(ALLOW_MISSING_ENV).is_ok_and(|v| v == "1") {
        eprintln!("SKIPPED: {SMOLLM_WEIGHTS} not found and {ALLOW_MISSING_ENV}=1");
        return None;
    }
    panic!(
        "{SMOLLM_WEIGHTS} not found in the crate root. Place the checkpoint there, or set \
         {ALLOW_MISSING_ENV}=1 to explicitly skip real-weight tests."
    );
}
