"""
Empirical Adversarial Challenge Test Suite for Velox Milestone 5 (Step 15)
Vectorized FP16 (__half2) and Marlin INT4 (w4a16) Execution
Author: challenger_m5_1 (Empirical Challenger)

Challenge 1: FP16 Parity Stability & Repeatability (3 Consecutive Runs)
Challenge 2: FP16 Throughput (>350 tok/s) & VRAM Footprint (<=360 MiB)
Challenge 3: Edge Cases: --steps 1, 2, 4, 64
Challenge 4: Eager Fallback (--no-cuda-graph) FP16 Parity
Challenge 5: Verbose Logging Mode FP16 Stability
Challenge 6: Source Code Audit (128-Bit uint4 loads, __half22float2, PTX lop3.b32)
Challenge 7: Marlin INT4 (w4a16) Execution, VRAM Footprint & Accuracy
"""

import os
import re
import subprocess
import time
from pathlib import Path
import pytest

PROJECT_ROOT = Path(r"C:\Sebastian\Project\Velox").resolve()
CUDA_RUNNER_EXE = PROJECT_ROOT / "src" / "runners" / "velox_cuda_runner.exe"
CUDA_RUNNER_CU = PROJECT_ROOT / "src" / "runners" / "velox_cuda_runner.cu"
TINYSTORIES_CU = PROJECT_ROOT / "src" / "runners" / "tinystories_smollm.cu"
CODEGEN_RS = PROJECT_ROOT / "src" / "codegen.rs"
WEIGHTS_PATH = PROJECT_ROOT / "smollm-135m.safetensors"

EXPECTED_32_TOKENS = [
    359, 253, 4028, 282, 260, 1773, 30, 198,
    26501, 368, 281, 260, 1773, 198, 20255, 3726,
    288, 9517, 253, 12301, 198, 19161, 354, 12301,
    198, 19161, 354, 12301, 198, 2683, 416, 722
]


def run_runner(args: list[str], timeout: int = 40) -> tuple[int, str, str]:
    cmd = [str(CUDA_RUNNER_EXE)] + args
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=timeout)
    return proc.returncode, proc.stdout, proc.stderr


def parse_tokens(stdout: str) -> list[int]:
    m = re.search(r"Generated Tokens \(\d+\)\s*:\s*\[(.*?)\]", stdout)
    if not m:
        return []
    token_str = m.group(1).strip()
    if not token_str:
        return []
    return [int(t.strip()) for t in token_str.split(",") if t.strip()]


def parse_vram_mib(stdout: str) -> float:
    m = re.search(r"Total VRAM Allocated\s*:\s*([\d\.]+)\s*MiB", stdout)
    assert m is not None, f"Could not find Total VRAM Allocated in:\n{stdout}"
    return float(m.group(1))


def parse_decode_throughput(stdout: str) -> float:
    m = re.search(r"Decode Latency \(\d+\s*tok\):\s*[\d\.]+\s*ms\s*\(([\d\.]+)\s*tok/s", stdout)
    assert m is not None, f"Could not find Decode Latency throughput in:\n{stdout}"
    return float(m.group(1))


# ==============================================================================
# Challenge 1: FP16 Parity Stability & Repeatability (3 Consecutive Runs)
# ==============================================================================

def test_challenge_1_fp16_parity_stability_repeatability_3_runs():
    """Verify that 3 consecutive runs of FP16 inference produce 100% 32-token bit-exact parity."""
    assert CUDA_RUNNER_EXE.exists(), f"Missing binary: {CUDA_RUNNER_EXE}"
    assert WEIGHTS_PATH.exists(), f"Missing weights: {WEIGHTS_PATH}"

    runs_tokens = []
    runs_throughput = []
    runs_vram = []

    for run_idx in range(1, 4):
        rc, stdout, stderr = run_runner(["--weights", str(WEIGHTS_PATH), "--steps", "32", "--quantize", "f16"])
        assert rc == 0, f"Run {run_idx} failed with exit code {rc}:\n{stderr}\n{stdout}"

        tokens = parse_tokens(stdout)
        assert len(tokens) == 32, f"Run {run_idx}: Expected 32 tokens, got {len(tokens)}"
        assert tokens == EXPECTED_32_TOKENS, f"Run {run_idx}: Token mismatch:\nGot:      {tokens}\nExpected: {EXPECTED_32_TOKENS}"

        assert "Parity 32-token check  : PASSED" in stdout, f"Run {run_idx}: Parity check line missing or failed"
        assert "Dynamic Heap Allocs    : 0" in stdout, f"Run {run_idx}: Non-zero dynamic allocations detected"

        vram = parse_vram_mib(stdout)
        throughput = parse_decode_throughput(stdout)

        runs_tokens.append(tokens)
        runs_throughput.append(throughput)
        runs_vram.append(vram)
        print(f"\n[Run {run_idx}] Throughput: {throughput:.2f} tok/s, VRAM: {vram:.2f} MiB")

    # Invariance check: tokens across all 3 runs must be identical
    assert runs_tokens[0] == runs_tokens[1] == runs_tokens[2], "Tokens drifted between consecutive runs!"


# ==============================================================================
# Challenge 2: FP16 Speedup & VRAM Footprint
# ==============================================================================

def test_challenge_2_fp16_speedup_and_vram_footprint():
    """Verify decode throughput > 350 tok/s and VRAM footprint <= 360 MiB."""
    rc, stdout, stderr = run_runner(["--weights", str(WEIGHTS_PATH), "--steps", "32", "--quantize", "f16"])
    assert rc == 0, f"Execution failed:\n{stderr}\n{stdout}"

    vram = parse_vram_mib(stdout)
    throughput = parse_decode_throughput(stdout)

    assert vram <= 360.0, f"VRAM footprint {vram:.2f} MiB exceeded threshold 360.0 MiB"
    assert throughput >= 350.0, f"Decode throughput {throughput:.2f} tok/s below required 350.0 tok/s"
    print(f"\n[Challenge 2 Verified] VRAM: {vram:.2f} MiB (<=360 MiB), Throughput: {throughput:.2f} tok/s (>350 tok/s)")


# ==============================================================================
# Challenge 3: Edge Cases: --steps 1, 2, 4, 64
# ==============================================================================

@pytest.mark.parametrize("steps", [1, 2, 4, 64])
def test_challenge_3_edge_cases_steps_bounds(steps: int):
    """Test boundary edge cases for step count with FP16 quantization."""
    rc, stdout, stderr = run_runner(["--weights", str(WEIGHTS_PATH), "--steps", str(steps), "--quantize", "f16"])
    assert rc == 0, f"Steps {steps} failed with code {rc}:\n{stderr}\n{stdout}"

    tokens = parse_tokens(stdout)
    assert len(tokens) == steps, f"Expected {steps} tokens, got {len(tokens)}"

    # Check prefix against EXPECTED_32_TOKENS
    for i in range(min(steps, 32)):
        assert tokens[i] == EXPECTED_32_TOKENS[i], f"Step {i} mismatch: got {tokens[i]}, expected {EXPECTED_32_TOKENS[i]}"

    assert "Dynamic Heap Allocs    : 0" in stdout


# ==============================================================================
# Challenge 4: Eager Fallback (--no-cuda-graph) FP16 Parity
# ==============================================================================

def test_challenge_4_fp16_eager_fallback():
    """Verify FP16 produces 100% bit-exact parity under eager execution (--no-cuda-graph)."""
    rc, stdout, stderr = run_runner(["--weights", str(WEIGHTS_PATH), "--steps", "32", "--quantize", "f16", "--no-cuda-graph"])
    assert rc == 0, f"Eager execution failed:\n{stderr}\n{stdout}"

    tokens = parse_tokens(stdout)
    assert tokens == EXPECTED_32_TOKENS, f"Eager FP16 token mismatch:\nGot:      {tokens}\nExpected: {EXPECTED_32_TOKENS}"
    assert "Parity 32-token check  : PASSED" in stdout
    assert "Dynamic Heap Allocs    : 0" in stdout


# ==============================================================================
# Challenge 5: Verbose Logging Mode FP16 Stability
# ==============================================================================

def test_challenge_5_fp16_verbose_mode():
    """Verify --verbose mode logs correctly during FP16 inference without memory corruption."""
    rc, stdout, stderr = run_runner(["--weights", str(WEIGHTS_PATH), "--steps", "4", "--quantize", "f16", "--verbose"])
    assert rc == 0, f"Verbose execution failed:\n{stderr}\n{stdout}"

    assert "[Step 1] Token: 253" in stdout
    assert "[Step 2] Token: 4028" in stdout
    assert "[Step 3] Token: 282" in stdout
    assert "Parity 4-token check   : PASSED [359, 253, 4028, 282]" in stdout


# ==============================================================================
# Challenge 6: Source Code Audit (Vectorized uint4 FP16 & PTX LOP3 INT4)
# ==============================================================================

def test_challenge_6_source_audit_vectorized_instructions():
    """Verify source code contains 128-bit uint4 vectorized loads, __half22float2, and inline PTX lop3."""
    assert CODEGEN_RS.exists()
    assert TINYSTORIES_CU.exists()

    codegen_code = CODEGEN_RS.read_text(encoding="utf-8")
    tinystories_code = TINYSTORIES_CU.read_text(encoding="utf-8")

    # In codegen.rs (emitted string with escaped quotes):
    assert "uint4 w_u4 = __ldg(reinterpret_cast<const uint4*>" in codegen_code
    assert "__half22float2(velox_u32_as_half2(" in codegen_code
    assert "lop3.b32 %0, %1, %2, %3, 0xEA;" in codegen_code

    # In tinystories_smollm.cu (compiled CUDA source):
    assert "uint4 w_u4 = __ldg(reinterpret_cast<const uint4*>" in tinystories_code
    assert "__half22float2(velox_u32_as_half2(" in tinystories_code
    assert 'asm("lop3.b32 %0, %1, %2, %3, 0xEA;"' in tinystories_code


# ==============================================================================
# Challenge 7: Marlin INT4 (w4a16) Execution, VRAM Footprint & Accuracy
# ==============================================================================

def test_challenge_7_marlin_int4_execution_and_accuracy():
    """Verify Marlin INT4 runs with zero heap allocations, VRAM <= 170 MiB, and logit error <= 0.03."""
    rc, stdout, stderr = run_runner(["--weights", str(WEIGHTS_PATH), "--steps", "4", "--quantize", "q4_0"])
    assert rc == 0, f"INT4 execution failed:\n{stderr}\n{stdout}"

    vram = parse_vram_mib(stdout)
    assert vram <= 170.0, f"INT4 VRAM {vram:.2f} MiB exceeded 170.0 MiB"

    assert "Dynamic Heap Allocs    : 0" in stdout
    assert "Accuracy Metric Check  : PASSED" in stdout
    assert "Parity 4-token check   : DIVERGED" in stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
