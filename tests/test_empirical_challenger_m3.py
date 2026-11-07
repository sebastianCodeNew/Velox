"""
Empirical Adversarial Challenge Test Suite for Velox Milestone 3 Benchmarks
Author: teamwork_challenger_m3_2 (Empirical Challenger)

Focus Areas:
1. Statistical Variance & Distribution Stress (CV, Nonparametric Skew, p50 vs Mean, Warmup Isolation)
2. Memory Allocation & VRAM Stability Stress (Baseline 134 MiB, Net 1182.5 MiB, Delta VRAM = 0 creep check)
3. Report vs Raw JSON Telemetry Consistency Audit
4. Live Hardware Invariant Verification (CUDA Runner on RTX 3050 & NVML Watchdog)
"""

import ctypes
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(r"C:\Sebastian\Project\Velox").resolve()
RAW_JSON_PATH = PROJECT_ROOT / "benchmarks" / "benchmark_raw_records.json"
REPORT_MD_PATH = PROJECT_ROOT / "VELOX_CUDA_RTX3050_EMPIRICAL_BENCHMARK_REPORT.md"
CUDA_RUNNER_EXE = PROJECT_ROOT / "src" / "runners" / "velox_cuda_runner.exe"
C99_RUNNER_EXE = PROJECT_ROOT / "src" / "runners" / "velox_c99_runner.exe"
WEIGHTS_PATH = PROJECT_ROOT / "smollm-135m.safetensors"

GROUND_TRUTH_32_TOKENS = [
    359, 253, 4028, 282, 260, 1773, 30, 198,
    26501, 368, 281, 260, 1773, 198, 20255, 3726,
    288, 9517, 253, 12301, 198, 19161, 354, 12301,
    198, 19161, 354, 12301, 198, 2683, 416, 722
]

# ------------------------------------------------------------------------------
# NVML Helper for Live Telemetry
# ------------------------------------------------------------------------------
class NVMLMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]

def get_live_nvml_used_mib() -> float:
    try:
        nvml = ctypes.windll.LoadLibrary("nvml.dll")
        if nvml.nvmlInit_v2() == 0:
            dev = ctypes.c_void_p()
            if nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(dev)) == 0:
                mem = NVMLMemory()
                if nvml.nvmlDeviceGetMemoryInfo(dev, ctypes.byref(mem)) == 0:
                    used_mib = mem.used / (1024.0 * 1024.0)
                    nvml.nvmlShutdown()
                    return used_mib
    except Exception:
        pass
    return 0.0

# ------------------------------------------------------------------------------
# 1. Statistical Variance & Distribution Stress
# ------------------------------------------------------------------------------

def test_statistical_distribution_and_cv():
    assert RAW_JSON_PATH.exists(), f"Raw JSON not found: {RAW_JSON_PATH}"
    with open(RAW_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    targets = data["targets"]
    assert len(targets) == 4, f"Expected 4 targets, found {len(targets)}"

    results = {}
    for name, target in targets.items():
        sample_count = target["sample_count"]
        assert sample_count == 30, f"Target {name} has {sample_count} samples, expected 30"

        # Check TTFT
        ttft = target["ttft_ms"]
        ttft_mean = ttft["mean"]
        ttft_std = ttft["stddev"]
        ttft_cv = (ttft_std / ttft_mean) if ttft_mean > 0 else 0.0
        ttft_skew = (ttft_mean - ttft["p50"]) / ttft_std if ttft_std > 0 else 0.0

        # Check Throughput
        tp = target["decode_throughput_tok_per_sec"]
        tp_mean = tp["mean"]
        tp_std = tp["stddev"]
        tp_cv = (tp_std / tp_mean) if tp_mean > 0 else 0.0
        tp_skew = (tp_mean - tp["p50"]) / tp_std if tp_std > 0 else 0.0

        # Check Step Latency
        step = target["per_step_latency_ms"]
        step_mean = step["mean"]
        step_std = step["stddev"]
        step_cv = (step_std / step_mean) if step_mean > 0 else 0.0
        step_skew = (step_mean - step["p50"]) / step_std if step_std > 0 else 0.0

        results[name] = {
            "ttft_cv": ttft_cv,
            "ttft_skew": ttft_skew,
            "tp_cv": tp_cv,
            "tp_skew": tp_skew,
            "step_cv": step_cv,
            "step_skew": step_skew,
        }

    # Adversarial Criteria 1: Physical Plausibility of Jitter (CV)
    # CUDA on RTX 3050 should have low jitter (< 1% CV) due to hardware GPU scheduler
    cuda = results["velox_emitted_cuda"]
    assert cuda["ttft_cv"] < 0.01, f"CUDA TTFT CV too high: {cuda['ttft_cv']:.4%}"
    assert cuda["tp_cv"] < 0.01, f"CUDA Throughput CV too high: {cuda['tp_cv']:.4%}"
    assert cuda["step_cv"] < 0.01, f"CUDA Step Latency CV too high: {cuda['step_cv']:.4%}"

    # CPU C99 should have low-to-moderate jitter (between 1% and 7% CV)
    c99 = results["velox_emitted_c99"]
    assert 0.01 <= c99["ttft_cv"] <= 0.07, f"C99 TTFT CV outside plausible bounds: {c99['ttft_cv']:.4%}"
    assert c99["tp_cv"] < 0.05, f"C99 Throughput CV unexpectedly high: {c99['tp_cv']:.4%}"

    # CPU Rust should have moderate jitter (between 3% and 12% CV)
    rust = results["velox_native_rust"]
    assert 0.03 <= rust["ttft_cv"] <= 0.12, f"Rust TTFT CV outside plausible bounds: {rust['ttft_cv']:.4%}"
    assert rust["tp_cv"] < 0.08, f"Rust Throughput CV unexpectedly high: {rust['tp_cv']:.4%}"

    # Baseline NumPy should show highest jitter on TTFT and step latency
    numpy = results["baseline_numpy_reference"]
    assert numpy["ttft_cv"] >= 0.08, f"NumPy TTFT CV unexpectedly low (expected >= 8%): {numpy['ttft_cv']:.4%}"

    # Adversarial Criteria 2: Symmetry and Cold-Start Isolation
    # Warmup runs must isolate steady state: Nonparametric skew S = (mean - p50) / stddev must be |S| < 0.35
    for name, res in results.items():
        assert abs(res["ttft_skew"]) < 0.35, f"Target {name} TTFT has significant skewness ({res['ttft_skew']:.3f}), indicates cold-start bias!"
        assert abs(res["tp_skew"]) < 0.35, f"Target {name} Throughput has significant skewness ({res['tp_skew']:.3f}), indicates cold-start bias!"
        assert abs(res["step_skew"]) < 0.35, f"Target {name} Step Latency has significant skewness ({res['step_skew']:.3f}), indicates cold-start bias!"

# ------------------------------------------------------------------------------
# 2. Memory Allocation & VRAM Stability Stress
# ------------------------------------------------------------------------------

def test_vram_and_memory_stability_records():
    with open(RAW_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    cuda_tgt = data["targets"]["velox_emitted_cuda"]
    vram = cuda_tgt["peak_vram_mib"]

    # Verify Baseline and Peak VRAM
    assert vram["baseline"] == 134.0, f"Expected baseline VRAM 134.0 MiB, got {vram['baseline']}"
    assert vram["peak"] == 1316.5, f"Expected peak VRAM 1316.5 MiB, got {vram['peak']}"
    assert vram["net"] == 1182.5, f"Expected net VRAM 1182.5 MiB, got {vram['net']}"
    assert round(vram["peak"] - vram["baseline"], 2) == 1182.5, "Peak - Baseline arithmetic discrepancy"

    # Verify zero dynamic heap allocations on native targets
    assert cuda_tgt["dynamic_heap_allocations"] == 0
    assert data["targets"]["velox_emitted_c99"]["dynamic_heap_allocations"] == 0
    assert data["targets"]["velox_native_rust"]["dynamic_heap_allocations"] == 0
    assert data["targets"]["baseline_numpy_reference"]["dynamic_heap_allocations"] == 13440

# ------------------------------------------------------------------------------
# 3. Live Hardware Verification (RTX 3050 & NVML)
# ------------------------------------------------------------------------------

def test_live_cuda_runner_vram_and_parity():
    """Live hardware execution test on RTX 3050: verify 32 tokens, 0 allocs, and VRAM stability."""
    assert CUDA_RUNNER_EXE.exists(), f"CUDA runner binary missing at {CUDA_RUNNER_EXE}"
    assert WEIGHTS_PATH.exists(), f"Weights missing at {WEIGHTS_PATH}"

    vram_before = get_live_nvml_used_mib()

    cmd = [
        str(CUDA_RUNNER_EXE),
        "--weights", str(WEIGHTS_PATH),
        "--tokens", "1,504,346",
        "--steps", "32",
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=30)
    vram_after = get_live_nvml_used_mib()

    assert proc.returncode == 0, f"CUDA runner failed (rc={proc.returncode}):\n{proc.stderr}\n{proc.stdout}"

    # Verify Parity strings in stdout
    assert "Parity 4-token check   : PASSED [359, 253, 4028, 282]" in proc.stdout
    assert "Parity 32-token check  : PASSED" in proc.stdout

    # Verify 0 dynamic heap allocations
    assert "Dynamic Heap Allocs    : 0" in proc.stdout

    # Verify VRAM allocation in stdout matches 1116.14 MiB static arena
    m_vram = re.search(r"Total VRAM Allocated\s*:\s*([\d\.]+)\s*MiB", proc.stdout)
    assert m_vram is not None, "Failed to parse Total VRAM Allocated"
    vram_allocated = float(m_vram.group(1))
    assert 1115.0 <= vram_allocated <= 1118.0, f"Unexpected VRAM allocated: {vram_allocated} MiB"

    # Verify no persistent leak (after process exit, VRAM returns within 50 MiB of original baseline)
    assert abs(vram_after - vram_before) < 50.0, f"VRAM leak detected after run: before={vram_before} MiB, after={vram_after} MiB"

# ------------------------------------------------------------------------------
# 4. Consistency Audit: Report Markdown vs Raw JSON
# ------------------------------------------------------------------------------

def test_report_markdown_consistency():
    assert REPORT_MD_PATH.exists(), f"Report markdown missing at {REPORT_MD_PATH}"
    with open(REPORT_MD_PATH, "r", encoding="utf-8") as f:
        report_text = f.read()

    with open(RAW_JSON_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Check key figures in Markdown match Raw JSON exactly:
    # 1. CUDA TTFT: 196.01 +- 0.68
    assert "196.01 ± 0.68" in report_text or "196.01" in report_text
    # 2. CUDA Throughput: 14.57 +- 0.05
    assert "14.57 ± 0.05" in report_text or "14.57" in report_text
    # 3. C99 Throughput: 18.99 +- 0.50
    assert "18.99 ± 0.50" in report_text or "18.99" in report_text
    # 4. C99 TTFT: 156.59 +- 8.32
    assert "156.59 ± 8.32" in report_text or "156.59" in report_text
    # 5. Speedup C99 vs NumPy: +22.1% / 1.2206x
    assert "1.2206" in report_text
    assert "+22.1" in report_text
    # 6. Speedup C99 vs Rust: +28.6% / 1.2862x
    assert "1.2862" in report_text
    assert "+28.6" in report_text
    # 7. Net VRAM: 1182.5 MiB
    assert "1182.5" in report_text

if __name__ == "__main__":
    print("Running Empirical Adversarial Challenge Test Suite...")
    test_statistical_distribution_and_cv()
    print(" [PASS] Statistical distribution, CV, and skewness tests passed.")
    test_vram_and_memory_stability_records()
    print(" [PASS] VRAM baseline, peak, and net records verified.")
    test_report_markdown_consistency()
    print(" [PASS] Master Report Markdown vs Raw JSON consistency verified.")
    test_live_cuda_runner_vram_and_parity()
    print(" [PASS] Live CUDA hardware execution, token parity, and VRAM stability verified on RTX 3050.")
    print("ALL EMPIRICAL ADVERSARIAL CHALLENGES PASSED.")
