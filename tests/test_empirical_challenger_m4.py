"""
Empirical Adversarial Challenge Test Suite for Velox Milestone 4 (Step 14)
CUDA Graph Capture and Device-Resident Decode State
Author: challenger_m4_2 (Empirical Challenger)

Challenge 1: Zero-Allocation Invariant Verification
Challenge 2: Ring Buffer & Boundary Stress (--steps 1, 2, 4, 64)
Challenge 3: Invariant Pointer & Host-Device Synchronization Check
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path
import pytest

PROJECT_ROOT = Path(r"C:\Sebastian\Project\Velox").resolve()
CUDA_RUNNER_CU = PROJECT_ROOT / "src" / "runners" / "velox_cuda_runner.cu"
TINYSTORIES_CU = PROJECT_ROOT / "src" / "runners" / "tinystories_smollm.cu"
CODEGEN_RS = PROJECT_ROOT / "src" / "codegen.rs"
CUDA_RUNNER_EXE = PROJECT_ROOT / "src" / "runners" / "velox_cuda_runner.exe"
WEIGHTS_PATH = PROJECT_ROOT / "smollm-135m.safetensors"

EXPECTED_32_TOKENS = [
    359, 253, 4028, 282, 260, 1773, 30, 198,
    26501, 368, 281, 260, 1773, 198, 20255, 3726,
    288, 9517, 253, 12301, 198, 19161, 354, 12301,
    198, 19161, 354, 12301, 198, 2683, 416, 722
]

# ==============================================================================
# Challenge 1: Zero-Allocation Invariant Verification
# ==============================================================================

def test_challenge_1_zero_allocation_in_decode_loop_source_audit():
    """Verify that NO dynamic allocation calls exist inside the decode loop or kernels."""
    assert CUDA_RUNNER_CU.exists(), f"Missing {CUDA_RUNNER_CU}"
    assert TINYSTORIES_CU.exists(), f"Missing {TINYSTORIES_CU}"

    runner_src = CUDA_RUNNER_CU.read_text(encoding="utf-8")
    tinystories_src = TINYSTORIES_CU.read_text(encoding="utf-8")

    # 1. Extract the decode loop in velox_cuda_runner.cu
    # Look for: if (steps > 1) { ... } before Standardized Velox Execution Report
    m = re.search(r"if \(steps > 1\) \{(.*?)\n\s*// 7\. Standardized", runner_src, re.DOTALL)
    assert m is not None, "Could not extract decode block from velox_cuda_runner.cu"
    decode_block = m.group(1)

    forbidden_alloc_patterns = [
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

    for pattern in forbidden_alloc_patterns:
        matches = re.findall(pattern, decode_block)
        assert len(matches) == 0, f"Found forbidden allocation pattern '{pattern}' in decode loop: {matches}"

    # 2. Inspect velox_SmolLM_135M_decode_graph_step_cuda in tinystories_smollm.cu
    m2 = re.search(r"extern \"C\" void velox_SmolLM_135M_decode_graph_step_cuda\s*\([^\)]*\)\s*\{(.*?)\n\}", tinystories_src, re.DOTALL)
    assert m2 is not None, "Could not extract velox_SmolLM_135M_decode_graph_step_cuda from tinystories_smollm.cu"
    graph_step_block = m2.group(1)

    for pattern in forbidden_alloc_patterns:
        matches = re.findall(pattern, graph_step_block)
        assert len(matches) == 0, f"Found forbidden allocation pattern '{pattern}' in velox_SmolLM_135M_decode_graph_step_cuda: {matches}"


def test_challenge_1_runtime_zero_allocation_claim():
    """Verify runtime report explicitly reports 0 Dynamic Heap Allocs across all step lengths."""
    for steps in [1, 2, 4, 32, 64]:
        cmd = [str(CUDA_RUNNER_EXE), "--weights", str(WEIGHTS_PATH), "--steps", str(steps)]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=30)
        assert proc.returncode == 0, f"Failed for --steps {steps}: {proc.stderr}"
        assert "Dynamic Heap Allocs    : 0 (Strictly zero cudaMalloc/cudaFree during generation)" in proc.stdout


# ==============================================================================
# Challenge 2: Ring Buffer & Boundary Stress
# ==============================================================================

@pytest.mark.parametrize("steps", [1, 2, 4, 64])
def test_challenge_2_ring_buffer_boundaries_cuda_graph(steps: int):
    """Test ring buffer boundary execution with CUDA graph: --steps 1, 2, 4, 64."""
    cmd = [
        str(CUDA_RUNNER_EXE),
        "--weights", str(WEIGHTS_PATH),
        "--tokens", "1,504,346",
        "--steps", str(steps),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=30)
    assert proc.returncode == 0, f"Execution failed for steps={steps} (rc={proc.returncode}):\n{proc.stderr}\n{proc.stdout}"

    # Parse Generated Tokens
    m = re.search(r"Generated Tokens \(\d+\)\s*:\s*\[(.*?)\]", proc.stdout)
    assert m is not None, f"Could not find Generated Tokens in output for steps={steps}"
    token_str = m.group(1).strip()
    if token_str:
        tokens = [int(t.strip()) for t in token_str.split(",") if t.strip()]
    else:
        tokens = []

    assert len(tokens) == steps, f"Expected {steps} generated tokens, got {len(tokens)}"

    # Check prefix matching against EXPECTED_32_TOKENS
    for i in range(min(steps, 32)):
        assert tokens[i] == EXPECTED_32_TOKENS[i], f"Token mismatch at index {i}: got {tokens[i]}, expected {EXPECTED_32_TOKENS[i]}"

    if steps >= 4:
        assert "Parity 4-token check   : PASSED [359, 253, 4028, 282]" in proc.stdout
    else:
        assert f"Parity 4-token check   : SKIPPED (fewer than 4 steps requested: {steps})" in proc.stdout

    if steps >= 32:
        assert "Parity 32-token check  : PASSED" in proc.stdout
    else:
        assert f"Parity 32-token check  : SKIPPED (fewer than 32 steps requested: {steps})" in proc.stdout


@pytest.mark.parametrize("steps", [1, 2, 4, 64])
def test_challenge_2_ring_buffer_boundaries_eager_fallback(steps: int):
    """Test eager fallback (--no-cuda-graph) across step bounds for parity and stability."""
    cmd = [
        str(CUDA_RUNNER_EXE),
        "--weights", str(WEIGHTS_PATH),
        "--tokens", "1,504,346",
        "--steps", str(steps),
        "--no-cuda-graph",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=30)
    assert proc.returncode == 0, f"Eager execution failed for steps={steps} (rc={proc.returncode}):\n{proc.stderr}\n{proc.stdout}"

    m = re.search(r"Generated Tokens \(\d+\)\s*:\s*\[(.*?)\]", proc.stdout)
    assert m is not None
    tokens = [int(t.strip()) for t in m.group(1).split(",") if t.strip()]
    assert len(tokens) == steps

    for i in range(min(steps, 32)):
        assert tokens[i] == EXPECTED_32_TOKENS[i], f"Eager token mismatch at index {i}: got {tokens[i]}, expected {EXPECTED_32_TOKENS[i]}"


def test_challenge_2_verbose_mode_edge_cases():
    """Verify --verbose mode logs each step without crashing or breaking ring buffer index."""
    cmd = [
        str(CUDA_RUNNER_EXE),
        "--weights", str(WEIGHTS_PATH),
        "--tokens", "1,504,346",
        "--steps", "4",
        "--verbose",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=30)
    assert proc.returncode == 0
    assert "[Step 1] Token: 253" in proc.stdout
    assert "[Step 2] Token: 4028" in proc.stdout
    assert "[Step 3] Token: 282" in proc.stdout


# ==============================================================================
# Challenge 3: Invariant Pointer & Host-Device Synchronization Check
# ==============================================================================

def test_challenge_3_synchronization_invariance_audit():
    """Confirm that during non-verbose decode (steps <= 1024), CPU executes only cudaGraphLaunch."""
    runner_src = CUDA_RUNNER_CU.read_text(encoding="utf-8")

    # Locate the non-verbose fast-path loop:
    # if (!verbose && steps <= VELOX_HOST_RING_SLOTS) { ... }
    m = re.search(r"if \(!verbose && steps <= VELOX_HOST_RING_SLOTS\)\s*\{(.*?)\}\s*else", runner_src, re.DOTALL)
    assert m is not None, "Could not find fast-path branch in velox_cuda_runner.cu"
    fast_path_body = m.group(1)

    # Extract the loop inside fast_path_body
    m_loop = re.search(r"for\s*\([^)]*\)\s*\{(.*?)\}", fast_path_body, re.DOTALL)
    assert m_loop is not None, "Could not find for-loop inside fast-path branch"
    loop_body = m_loop.group(1).strip()

    # The loop body MUST contain cudaGraphLaunch
    assert "cudaGraphLaunch(decode_graph_exec, stream);" in loop_body

    # The loop body MUST NOT contain any synchronization or host-device transfers
    forbidden_sync_calls = [
        "cudaStreamSynchronize",
        "cudaDeviceSynchronize",
        "cudaMemcpy",
        "cudaMemcpyAsync",
        "cudaEventSynchronize",
        "cudaEventElapsedTime",
    ]
    for sync_call in forbidden_sync_calls:
        assert sync_call not in loop_body, f"Found forbidden sync call '{sync_call}' in fast-path loop body:\n{loop_body}"

    # Confirm that cudaStreamSynchronize occurs strictly AFTER the loop completes
    after_loop = fast_path_body[m_loop.end():]
    assert "cudaStreamSynchronize(stream);" in after_loop, "Expected cudaStreamSynchronize after fast-path loop"


def test_challenge_3_dispatch_latency_overhead():
    """Empirical proof of zero-sync dispatch: verify CPU launch rate does not stall."""
    # Run 64 steps and measure total decode time
    cmd = [
        str(CUDA_RUNNER_EXE),
        "--weights", str(WEIGHTS_PATH),
        "--tokens", "1,504,346",
        "--steps", "64",
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=30)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    assert proc.returncode == 0

    # Parse decode throughput
    m = re.search(r"Decode Latency \(63 tok\):\s*([\d\.]+)\s*ms\s*\(([\d\.]+)\s*tok/s", proc.stdout)
    assert m is not None, "Could not parse Decode Latency"
    decode_ms = float(m.group(1))
    tok_per_sec = float(m.group(2))

    # RTX 3050 should achieve > 200 tok/s with CUDA graph replay (vs ~14 tok/s prior to warp-gemv & graphs)
    assert tok_per_sec > 200.0, f"Expected > 200 tok/s, got {tok_per_sec:.2f} tok/s"
    print(f"\n[PASS] Empirical decode throughput: {tok_per_sec:.2f} tok/s ({decode_ms:.3f} ms for 63 tokens)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
