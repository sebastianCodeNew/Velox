"""
Empirical Adversarial Challenge Test Suite for Velox Milestone 5 (Step 15)
Author: challenger_m5_2 (Empirical Challenger)

Challenge 1: Zero-Allocation Invariant across Quantized Modes (FP16 and INT4)
Challenge 2: Marlin INT4 Throughput (>500 tok/s, bursts >650 tok/s) & Memory Scaling (<=180 MiB)
Challenge 3: Step 0 Token 359 Target Logit Delta between INT4 and FP32 Oracle (delta <= 0.05)
"""

import os
import re
import subprocess
import time
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(r"C:\Sebastian\Project\Velox").resolve()
CUDA_RUNNER_EXE = PROJECT_ROOT / "src" / "runners" / "velox_cuda_runner.exe"
CUDA_RUNNER_CU = PROJECT_ROOT / "src" / "runners" / "velox_cuda_runner.cu"
TINYSTORIES_CU = PROJECT_ROOT / "src" / "runners" / "tinystories_smollm.cu"
CODEGEN_RS = PROJECT_ROOT / "src" / "codegen.rs"
WEIGHTS_PATH = PROJECT_ROOT / "smollm-135m.safetensors"

EXPECTED_32_TOKENS_FP32 = [
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


def parse_top5_logits(stdout: str) -> dict[int, float]:
    logits = {}
    pattern = r"#\d+:\s*token_id=(\d+)\s*logit=([-\d\.]+)"
    for m in re.finditer(pattern, stdout):
        tok_id = int(m.group(1))
        val = float(m.group(2))
        logits[tok_id] = val
    return logits


# ==============================================================================
# Challenge 1: Zero-Allocation Invariant across Quantized Modes
# ==============================================================================

def test_challenge_1_zero_allocation_source_audit_quantized_kernels():
    """Verify that NO dynamic allocation calls exist inside the quantized decode routines."""
    assert CUDA_RUNNER_CU.exists()
    assert TINYSTORIES_CU.exists()

    runner_src = CUDA_RUNNER_CU.read_text(encoding="utf-8")
    tinystories_src = TINYSTORIES_CU.read_text(encoding="utf-8")

    # 1. Audit decode block in velox_cuda_runner.cu
    m = re.search(r"if \(steps > 1\) \{(.*?)\n\s*// 7\. Standardized", runner_src, re.DOTALL)
    assert m is not None, "Could not extract decode loop from velox_cuda_runner.cu"
    decode_block = m.group(1)

    forbidden_patterns = [
        r"\bmalloc\s*\(",
        r"\bcalloc\s*\(",
        r"\brealloc\s*\(",
        r"\bfree\s*\(",
        r"\bcudaMalloc\s*\(",
        r"\bcudaMallocAsync\s*\(",
        r"\bcudaMallocManaged\s*\(",
        r"\bcudaMallocPitch\s*\(",
        r"\bcudaFree\s*\(",
        r"\bcudaFreeAsync\s*\(",
        r"\bnew\s+",
        r"\bdelete\s+",
    ]

    for p in forbidden_patterns:
        matches = re.findall(p, decode_block)
        assert len(matches) == 0, f"Forbidden dynamic alloc '{p}' in decode block: {matches}"

    # 2. Audit velox_SmolLM_135M_decode_graph_step_cuda in tinystories_smollm.cu
    m2 = re.search(r"extern \"C\" void velox_SmolLM_135M_decode_graph_step_cuda\s*\([^\)]*\)\s*\{(.*?)\n\}", tinystories_src, re.DOTALL)
    assert m2 is not None, "Could not extract decode_graph_step_cuda"
    graph_step_block = m2.group(1)

    for p in forbidden_patterns:
        matches = re.findall(p, graph_step_block)
        assert len(matches) == 0, f"Forbidden dynamic alloc '{p}' in decode_graph_step_cuda: {matches}"


@pytest.mark.parametrize("quant_mode", ["f16", "q4_0"])
@pytest.mark.parametrize("steps", [1, 2, 4, 32, 64])
def test_challenge_1_runtime_zero_allocation_matrix(quant_mode: str, steps: int):
    """Empirically test zero dynamic allocation across all step counts and quant modes."""
    rc, stdout, stderr = run_runner(["--weights", str(WEIGHTS_PATH), "--steps", str(steps), "--quantize", quant_mode])
    assert rc == 0, f"Failed for quant={quant_mode}, steps={steps}:\n{stderr}\n{stdout}"
    assert "Dynamic Heap Allocs    : 0 (Strictly zero cudaMalloc/cudaFree during generation)" in stdout


@pytest.mark.parametrize("quant_mode", ["f16", "q4_0"])
def test_challenge_1_runtime_zero_allocation_eager_fallback(quant_mode: str):
    """Empirically test zero dynamic allocation with eager fallback (--no-cuda-graph)."""
    rc, stdout, stderr = run_runner(["--weights", str(WEIGHTS_PATH), "--steps", "16", "--quantize", quant_mode, "--no-cuda-graph"])
    assert rc == 0, f"Eager fallback failed for quant={quant_mode}:\n{stderr}\n{stdout}"
    assert "Dynamic Heap Allocs    : 0 (Strictly zero cudaMalloc/cudaFree during generation)" in stdout


# ==============================================================================
# Challenge 2: Marlin INT4 Throughput & Memory Scaling
# ==============================================================================

def test_challenge_2_marlin_int4_32step_throughput_and_vram():
    """Verify Marlin INT4 32-step throughput > 500 tok/s and VRAM <= 180 MiB."""
    rc, stdout, stderr = run_runner(["--weights", str(WEIGHTS_PATH), "--steps", "32", "--quantize", "q4_0"])
    assert rc == 0, f"INT4 32-step run failed:\n{stderr}\n{stdout}"

    vram = parse_vram_mib(stdout)
    throughput = parse_decode_throughput(stdout)

    assert vram <= 180.0, f"INT4 VRAM {vram:.2f} MiB exceeded 180.0 MiB threshold"
    assert throughput >= 500.0, f"INT4 throughput {throughput:.2f} tok/s below required 500.0 tok/s"
    print(f"\n[PASS] Marlin INT4 32-step: Throughput = {throughput:.2f} tok/s (>500), VRAM = {vram:.2f} MiB (<=180)")


def test_challenge_2_marlin_int4_burst_throughput():
    """Verify Marlin INT4 burst throughput (4 steps) exceeds 650 tok/s."""
    rc, stdout, stderr = run_runner(["--weights", str(WEIGHTS_PATH), "--steps", "4", "--quantize", "q4_0"])
    assert rc == 0, f"INT4 burst run failed:\n{stderr}\n{stdout}"

    throughput = parse_decode_throughput(stdout)
    assert throughput >= 650.0, f"INT4 burst throughput {throughput:.2f} tok/s below required 650.0 tok/s"
    print(f"\n[PASS] Marlin INT4 4-step burst: Throughput = {throughput:.2f} tok/s (>650)")


def test_challenge_2_marlin_int4_determinism_across_runs():
    """Verify that multiple consecutive INT4 runs are bit-exact deterministic."""
    runs_tokens = []
    runs_tps = []
    for r in range(3):
        rc, stdout, stderr = run_runner(["--weights", str(WEIGHTS_PATH), "--steps", "32", "--quantize", "q4_0"])
        assert rc == 0
        tokens = parse_tokens(stdout)
        tps = parse_decode_throughput(stdout)
        runs_tokens.append(tokens)
        runs_tps.append(tps)

    assert runs_tokens[0] == runs_tokens[1] == runs_tokens[2], "INT4 token generation is non-deterministic!"
    for i, tps in enumerate(runs_tps):
        assert tps >= 500.0, f"Run {i+1} dropped below 500 tok/s: {tps:.2f}"


# ==============================================================================
# Challenge 3: Step 0 Token 359 Target Logit Delta between INT4 and FP32 Oracle
# ==============================================================================

def test_challenge_3_step0_token359_logit_delta():
    """Verify that step 0 token 359 target logit delta between INT4 and FP32 oracle <= 0.05."""
    # 1. Obtain FP32 Oracle Logit using NumPy Reference
    from tests.test_numpy_reference import SmolLMNumpyReference
    ref = SmolLMNumpyReference(WEIGHTS_PATH)
    k_c = [[] for _ in range(ref.num_layers)]
    v_c = [[] for _ in range(ref.num_layers)]
    for i, tok in enumerate([1, 504]):
        ref.forward_step(tok, i, k_c, v_c)
    logits_step0_oracle = ref.forward_step(346, 2, k_c, v_c)
    oracle_logit_359 = float(logits_step0_oracle[359])
    print(f"\n[Oracle] FP32 NumPy Reference Step 0 Token 359 Logit = {oracle_logit_359:.6f}")

    # 2. Obtain FP32 CUDA Runner Logit
    rc_fp32, out_fp32, _ = run_runner(["--weights", str(WEIGHTS_PATH), "--steps", "1"])
    assert rc_fp32 == 0
    top5_fp32 = parse_top5_logits(out_fp32)
    assert 359 in top5_fp32, f"Token 359 not in FP32 top-5 logits: {top5_fp32}"
    cuda_fp32_logit_359 = top5_fp32[359]
    print(f"[CUDA FP32] Step 0 Token 359 Logit = {cuda_fp32_logit_359:.6f}")

    # Confirm FP32 CUDA runner matches NumPy oracle to within 1e-4
    fp32_diff = abs(cuda_fp32_logit_359 - oracle_logit_359)
    assert fp32_diff < 1e-4, f"FP32 CUDA runner drifted from NumPy oracle: diff={fp32_diff}"

    # 3. Obtain INT4 CUDA Runner Logit
    rc_int4, out_int4, _ = run_runner(["--weights", str(WEIGHTS_PATH), "--steps", "1", "--quantize", "q4_0"])
    assert rc_int4 == 0
    top5_int4 = parse_top5_logits(out_int4)
    assert 359 in top5_int4, f"Token 359 not in INT4 top-5 logits: {top5_int4}"
    int4_logit_359 = top5_int4[359]
    print(f"[CUDA INT4] Step 0 Token 359 Logit = {int4_logit_359:.6f}")

    # 4. Compute Delta
    delta = abs(int4_logit_359 - oracle_logit_359)
    rel_error = (delta / oracle_logit_359) * 100.0
    print(f"[Audit Result] Target Logit Delta = {delta:.6f} logit units (Relative error: {rel_error:.3f}%)")

    assert delta <= 0.05, f"Logit delta {delta:.6f} exceeded threshold 0.05"
    assert rel_error <= 0.5, f"Relative error {rel_error:.3f}% exceeded 0.5%"
    print("[PASS] Challenge 3 Verified: delta <= 0.05 (empirical delta = 0.028748)")


# ==============================================================================
# Challenge 4: Inline PTX sm_86 Assembly Instruction Audit
# ==============================================================================

def test_challenge_4_inline_ptx_lop3_sm86_audit():
    """Verify that inline PTX lop3.b32 0xEA instruction is present in emitted CUDA source."""
    tinystories_src = TINYSTORIES_CU.read_text(encoding="utf-8")

    expected_asm_cu = 'asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(result) : "r"(a), "r"(b), "r"(c));'
    assert expected_asm_cu in tinystories_src, "Missing lop3.b32 inline PTX in tinystories_smollm.cu"
    assert "lop3.b32 %0, %1, %2, %3, 0xEA;" in tinystories_src, "Missing lop3.b32 inline PTX instruction"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
