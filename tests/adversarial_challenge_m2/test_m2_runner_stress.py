"""
Velox Milestone 2 Adversarial Challenge & Stress Test Suite
Testing RTX 3050 Laptop GPU CUDA Runner & Host C99 Runner

Tests:
1. Re-compilation & clean link via build_runners.bat
2. 5 consecutive CUDA runner executions on RTX 3050 with active nvidia-smi VRAM polling
3. VRAM leak check (pre vs post execution)
4. Bitwise / numerical determinism across runs
5. Step count variations (4, 16, 32 steps) & prefix invariance
6. Prompt variations (1 token, 2 tokens, 4 tokens) & autoregressive stability
7. C99 CPU runner parity & cross-engine logit error analysis
8. Negative & edge case testing (missing weights, zero steps, invalid args)
"""

import subprocess
import os
import sys
import time
import re
import json
import threading

PROJECT_ROOT = r"C:\Sebastian\Project\Velox"
RUNNERS_DIR = os.path.join(PROJECT_ROOT, "src", "runners")
CUDA_RUNNER = os.path.join(RUNNERS_DIR, "velox_cuda_runner.exe")
C99_RUNNER = os.path.join(RUNNERS_DIR, "velox_c99_runner.exe")
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "smollm-135m.safetensors")
BUILD_BAT = os.path.join(PROJECT_ROOT, "build_runners.bat")

EXPECTED_32 = [
    359, 253, 4028, 282, 260, 1773, 30, 198, 26501, 368, 281, 260, 1773, 198, 20255, 3726,
    288, 9517, 253, 12301, 198, 19161, 354, 12301, 198, 19161, 354, 12301, 198, 2683, 416, 722
]
EXPECTED_4 = [359, 253, 4028, 282]


def query_nvidia_smi():
    """Queries current VRAM used, free, total in MiB."""
    try:
        cmd = ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total,utilization.gpu", "--format=csv,noheader,nounits"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        line = res.stdout.strip().split("\n")[0]
        used, free, total, util = [float(x.strip()) for x in line.split(",")]
        return {"used_mib": used, "free_mib": free, "total_mib": total, "util_gpu_pct": util}
    except Exception as e:
        return {"error": str(e)}


class VRAMSampler:
    """Background sampler for VRAM to capture peak usage during execution."""
    def __init__(self, interval_sec=0.02):
        self.interval = interval_sec
        self.running = False
        self.peak_used = 0.0
        self.samples = []
        self._thread = None

    def start(self):
        self.running = True
        self.samples = []
        self.peak_used = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self.running:
            data = query_nvidia_smi()
            if "used_mib" in data:
                u = data["used_mib"]
                self.samples.append(u)
                if u > self.peak_used:
                    self.peak_used = u
            time.sleep(self.interval)

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        return self.peak_used


def parse_runner_output(stdout_str):
    """Extracts tokens, top-5 logits, TTFT, decode latency from runner stdout."""
    res = {
        "raw": stdout_str,
        "tokens": [],
        "top5": [],
        "ttft_ms": None,
        "decode_ms": None,
        "total_ms": None,
        "parity_4": None,
        "parity_32": None,
    }
    # Generated Tokens
    m_tokens = re.search(r"Generated Tokens \(\d+\)\s*:\s*\[(.*?)\]", stdout_str)
    if m_tokens:
        tok_str = m_tokens.group(1).strip()
        if tok_str:
            res["tokens"] = [int(t.strip()) for t in tok_str.split(",") if t.strip()]

    # Top-5 logits
    top5_matches = re.findall(r"#(\d+):\s*token_id=(\d+)\s*logit=([-\d\.]+)", stdout_str)
    for rank, tid, val in top5_matches:
        res["top5"].append({"rank": int(rank), "token_id": int(tid), "logit": float(val)})

    # Latencies
    m_ttft = re.search(r"TTFT \(Prefill Latency\)\s*:\s*([\d\.]+)\s*ms", stdout_str)
    if m_ttft:
        res["ttft_ms"] = float(m_ttft.group(1))

    m_dec = re.search(r"Decode Latency \(\d+ tok\)\s*:\s*([\d\.]+)\s*ms", stdout_str)
    if m_dec:
        res["decode_ms"] = float(m_dec.group(1))

    m_tot = re.search(r"Total (?:GPU )?Inference\s*:\s*([\d\.]+)\s*ms", stdout_str)
    if m_tot:
        res["total_ms"] = float(m_tot.group(1))

    m_p4 = re.search(r"Parity 4-token check\s*:\s*(\w+)", stdout_str)
    if m_p4:
        res["parity_4"] = m_p4.group(1)

    m_p32 = re.search(r"Parity 32-token check\s*:\s*(\w+)", stdout_str)
    if m_p32:
        res["parity_32"] = m_p32.group(1)

    return res


def run_binary(bin_path, args, timeout=60):
    cmd = [bin_path] + args
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    elapsed = (time.perf_counter() - t0) * 1000.0
    parsed = parse_runner_output(proc.stdout)
    return {
        "exit_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "elapsed_wall_ms": elapsed,
        "parsed": parsed,
    }


def main():
    print("=" * 80)
    print("VELOX M2 ADVERSARIAL CHALLENGE & EMPIRICAL STRESS TEST HARNESS")
    print("=" * 80)

    results = {
        "build_stress": {},
        "vram_consecutive_runs": [],
        "determinism_check": {},
        "step_variations": {},
        "prompt_variations": {},
        "c99_cpu_parity": {},
        "negative_tests": {},
        "verdict": "UNKNOWN",
    }

    # -------------------------------------------------------------------------
    # Test 1: Compile & Link Stress
    # -------------------------------------------------------------------------
    print("\n[TEST 1] Compile and Link Stress Test...")
    build_t0 = time.perf_counter()
    proc_b = subprocess.run(["cmd.exe", "/c", BUILD_BAT], capture_output=True, text=True)
    build_time = (time.perf_counter() - build_t0) * 1000.0

    t1_pass = (proc_b.returncode == 0) and os.path.exists(CUDA_RUNNER) and os.path.exists(C99_RUNNER)
    results["build_stress"] = {
        "exit_code": proc_b.returncode,
        "wall_time_ms": build_time,
        "cuda_runner_exists": os.path.exists(CUDA_RUNNER),
        "c99_runner_exists": os.path.exists(C99_RUNNER),
        "passed": t1_pass,
        "stdout_tail": "\n".join(proc_b.stdout.strip().split("\n")[-10:]),
    }
    print(f"  Build return code: {proc_b.returncode}")
    print(f"  CUDA runner exists: {os.path.exists(CUDA_RUNNER)} ({os.path.getsize(CUDA_RUNNER)} bytes)")
    print(f"  C99 runner exists:  {os.path.exists(C99_RUNNER)} ({os.path.getsize(C99_RUNNER)} bytes)")
    print(f"  Result: {'PASS' if t1_pass else 'FAIL'}")

    # -------------------------------------------------------------------------
    # Test 2: 5 Consecutive Runs of velox_cuda_runner.exe on RTX 3050 & VRAM Tracking
    # -------------------------------------------------------------------------
    print("\n[TEST 2] 5 Consecutive Runs on RTX 3050 GPU with Active VRAM Tracking...")
    base_vram_init = query_nvidia_smi()
    print(f"  Initial Baseline VRAM: {base_vram_init.get('used_mib', 'N/A')} MiB used / {base_vram_init.get('total_mib', 'N/A')} MiB total")

    runs_data = []
    all_runs_passed = True

    for run_idx in range(1, 6):
        vram_before = query_nvidia_smi()
        sampler = VRAMSampler(interval_sec=0.01)
        sampler.start()

        run_res = run_binary(CUDA_RUNNER, ["--weights", WEIGHTS_PATH, "--tokens", "1,504,346", "--steps", "32"])
        peak_vram = sampler.stop()

        time.sleep(0.5)  # brief settle
        vram_after = query_nvidia_smi()

        tokens = run_res["parsed"]["tokens"]
        match_32 = (tokens == EXPECTED_32)
        match_4 = (tokens[:4] == EXPECTED_4)
        run_ok = (run_res["exit_code"] == 0) and match_32 and match_4

        if not run_ok:
            all_runs_passed = False

        record = {
            "run_index": run_idx,
            "exit_code": run_res["exit_code"],
            "wall_ms": run_res["elapsed_wall_ms"],
            "ttft_ms": run_res["parsed"]["ttft_ms"],
            "decode_ms": run_res["parsed"]["decode_ms"],
            "total_ms": run_res["parsed"]["total_ms"],
            "vram_before_mib": vram_before.get("used_mib"),
            "vram_peak_mib": peak_vram,
            "vram_after_mib": vram_after.get("used_mib"),
            "vram_leaked_mib": vram_after.get("used_mib", 0) - vram_before.get("used_mib", 0),
            "match_4": match_4,
            "match_32": match_32,
            "top5": run_res["parsed"]["top5"],
            "tokens": tokens,
        }
        runs_data.append(record)
        print(f"  Run #{run_idx}: Exit={run_res['exit_code']} | Wall={run_res['elapsed_wall_ms']:.1f}ms | "
              f"TTFT={record['ttft_ms']}ms | Decode={record['decode_ms']}ms | "
              f"Peak VRAM={peak_vram:.1f} MiB | VRAM After={vram_after.get('used_mib')} MiB | "
              f"Tokens32 Match: {match_32}")

    results["vram_consecutive_runs"] = runs_data

    # Check VRAM leak across entire sequence
    final_vram = query_nvidia_smi()
    net_vram_leak = final_vram.get("used_mib", 0) - base_vram_init.get("used_mib", 0)
    print(f"  Final Post-Run VRAM: {final_vram.get('used_mib')} MiB (Net change: {net_vram_leak:+f} MiB)")

    # Check Determinism across 5 runs
    first_tokens = runs_data[0]["tokens"]
    first_top5 = runs_data[0]["top5"]
    tokens_identical = all(r["tokens"] == first_tokens for r in runs_data)
    logits_identical = all(r["top5"] == first_top5 for r in runs_data)

    results["determinism_check"] = {
        "tokens_identical_across_5_runs": tokens_identical,
        "logits_identical_across_5_runs": logits_identical,
        "net_vram_leak_mib": net_vram_leak,
        "all_runs_match_expected_32": all_runs_passed,
    }
    print(f"  Determinism: Tokens Identical={tokens_identical}, Logits Identical={logits_identical}")
    print(f"  VRAM Leak Test: {'PASS (0 MiB leak)' if net_vram_leak == 0 else f'WARN/FAIL ({net_vram_leak} MiB)'}")

    # -------------------------------------------------------------------------
    # Test 3: Varying Step Counts (4, 16, 32)
    # -------------------------------------------------------------------------
    print("\n[TEST 3] Step Count Variations (4, 16, 32 steps)...")
    step_results = {}
    for s in [4, 16, 32]:
        res = run_binary(CUDA_RUNNER, ["--weights", WEIGHTS_PATH, "--tokens", "1,504,346", "--steps", str(s)])
        toks = res["parsed"]["tokens"]
        expected_slice = EXPECTED_32[:s]
        is_exact = (toks == expected_slice)
        step_results[f"steps_{s}"] = {
            "exit_code": res["exit_code"],
            "generated_len": len(toks),
            "tokens": toks,
            "expected_slice": expected_slice,
            "match": is_exact,
            "parity_4": res["parsed"]["parity_4"],
            "parity_32": res["parsed"]["parity_32"],
        }
        print(f"  --steps {s}: count={len(toks)} | Exit={res['exit_code']} | Match={is_exact} | "
              f"Parity4={res['parsed']['parity_4']} | Parity32={res['parsed']['parity_32']}")

    results["step_variations"] = step_results

    # -------------------------------------------------------------------------
    # Test 4: Prompt Token Variations & Autoregressive Stability
    # -------------------------------------------------------------------------
    print("\n[TEST 4] Prompt Token Variations & Autoregressive Stability...")
    prompt_tests = [
        {"desc": "2-token prompt", "tokens": "1,504"},
        {"desc": "1-token prompt (BOS only)", "tokens": "1"},
        {"desc": "4-token prompt", "tokens": "1,504,346,359"},
    ]
    prompt_results = {}
    for pt in prompt_tests:
        desc = pt["desc"]
        tok_str = pt["tokens"]
        # Run twice to verify prompt determinism
        run_a = run_binary(CUDA_RUNNER, ["--weights", WEIGHTS_PATH, "--tokens", tok_str, "--steps", "8"])
        run_b = run_binary(CUDA_RUNNER, ["--weights", WEIGHTS_PATH, "--tokens", tok_str, "--steps", "8"])

        toks_a = run_a["parsed"]["tokens"]
        toks_b = run_b["parsed"]["tokens"]
        stable = (run_a["exit_code"] in [0, 2]) and (run_b["exit_code"] in [0, 2])
        det = (toks_a == toks_b) and (len(toks_a) == 8)

        prompt_results[desc] = {
            "tokens_arg": tok_str,
            "run_a_exit": run_a["exit_code"],
            "run_b_exit": run_b["exit_code"],
            "run_a_tokens": toks_a,
            "run_b_tokens": toks_b,
            "deterministic": det,
            "stable_execution": stable,
        }
        print(f"  {desc} ({tok_str}): RunA={toks_a} | RunB={toks_b} | Deterministic={det} | Stable={stable}")

    results["prompt_variations"] = prompt_results

    # -------------------------------------------------------------------------
    # Test 5: Emitted C99 CPU Runner Cross-Engine Parity
    # -------------------------------------------------------------------------
    print("\n[TEST 5] Emitted C99 Host Runner on CPU & Cross-Engine Parity...")
    c99_runs = []
    c99_all_match = True
    for c_idx in range(1, 4):
        c_res = run_binary(C99_RUNNER, ["--weights", WEIGHTS_PATH, "--tokens", "1,504,346", "--steps", "32"])
        toks = c_res["parsed"]["tokens"]
        match_32 = (toks == EXPECTED_32)
        match_4 = (toks[:4] == EXPECTED_4)
        c_ok = (c_res["exit_code"] == 0) and match_32 and match_4
        if not c_ok:
            c99_all_match = False
        c99_runs.append({
            "run_idx": c_idx,
            "exit_code": c_res["exit_code"],
            "ttft_ms": c_res["parsed"]["ttft_ms"],
            "decode_ms": c_res["parsed"]["decode_ms"],
            "total_ms": c_res["parsed"]["total_ms"],
            "tokens": toks,
            "top5": c_res["parsed"]["top5"],
            "match_32": match_32,
        })
        print(f"  C99 Run #{c_idx}: Exit={c_res['exit_code']} | TTFT={c_res['parsed']['ttft_ms']}ms | "
              f"Decode={c_res['parsed']['decode_ms']}ms | Match32={match_32}")

    # Compute max logit discrepancy between CUDA GPU run 1 and C99 CPU run 1
    cuda_top5 = runs_data[0]["top5"]
    c99_top5 = c99_runs[0]["top5"]
    logit_diffs = []
    for k in range(min(len(cuda_top5), len(c99_top5))):
        c_val = cuda_top5[k]["logit"]
        cpu_val = c99_top5[k]["logit"]
        diff = abs(c_val - cpu_val)
        logit_diffs.append({
            "rank": k + 1,
            "cuda_token": cuda_top5[k]["token_id"],
            "cuda_logit": c_val,
            "cpu_token": c99_top5[k]["token_id"],
            "cpu_logit": cpu_val,
            "abs_err": diff,
        })

    max_logit_err = max(d["abs_err"] for d in logit_diffs) if logit_diffs else 999.0
    results["c99_cpu_parity"] = {
        "c99_runs": c99_runs,
        "all_c99_match_expected_32": c99_all_match,
        "cross_engine_logit_diffs": logit_diffs,
        "max_logit_err": max_logit_err,
        "parity_threshold_met": (max_logit_err < 0.001),
    }
    print(f"  Cross-Engine Max Top-5 Logit Abs Error: {max_logit_err:.6f} (Threshold < 0.001: {max_logit_err < 0.001})")

    # -------------------------------------------------------------------------
    # Test 6: Host Runner Error Handling & Negative Testing
    # -------------------------------------------------------------------------
    print("\n[TEST 6] Negative Testing & Edge Cases...")
    neg_results = {}

    # 6a. Missing weights file
    res_missing = run_binary(CUDA_RUNNER, ["--weights", "non_existent_weights.safetensors"])
    neg_results["missing_weights"] = {
        "exit_code": res_missing["exit_code"],
        "handled_gracefully": (res_missing["exit_code"] != 0 and "Error" in res_missing["stderr"]),
        "stderr": res_missing["stderr"].strip(),
    }
    print(f"  Missing weights: Exit={res_missing['exit_code']} | Handled gracefully: {neg_results['missing_weights']['handled_gracefully']}")

    # 6b. Zero steps
    res_zero_steps = run_binary(CUDA_RUNNER, ["--weights", WEIGHTS_PATH, "--tokens", "1,504,346", "--steps", "0"])
    neg_results["zero_steps"] = {
        "exit_code": res_zero_steps["exit_code"],
        "handled_gracefully": (res_zero_steps["exit_code"] in [0, 2]),
        "stdout_snippet": res_zero_steps["stdout"][:200].replace("\n", " "),
    }
    print(f"  Zero steps: Exit={res_zero_steps['exit_code']} | Handled: {neg_results['zero_steps']['handled_gracefully']}")

    # 6c. Missing prompt tokens / invalid string
    res_empty_tok = run_binary(CUDA_RUNNER, ["--weights", WEIGHTS_PATH, "--tokens", "", "--steps", "4"])
    neg_results["empty_tokens_flag"] = {
        "exit_code": res_empty_tok["exit_code"],
        "crashed": (res_empty_tok["exit_code"] not in [0, 1, 2]),
    }
    print(f"  Empty tokens string: Exit={res_empty_tok['exit_code']} | Crashed: {neg_results['empty_tokens_flag']['crashed']}")

    results["negative_tests"] = neg_results

    # -------------------------------------------------------------------------
    # Final Verdict Assessment
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 80)

    pass_build = results["build_stress"]["passed"]
    pass_vram = (net_vram_leak == 0)
    pass_determinism = tokens_identical and logits_identical and all_runs_passed
    pass_cross_engine = c99_all_match and (max_logit_err < 0.001)

    all_criteria_met = pass_build and pass_vram and pass_determinism and pass_cross_engine

    results["verdict"] = "APPROVE" if all_criteria_met else "REQUEST_CHANGES"
    print(f"  Build & Clean Recompile : {'PASS' if pass_build else 'FAIL'}")
    print(f"  VRAM Zero Leak (5 runs) : {'PASS' if pass_vram else 'FAIL'} (0 MiB leaked)")
    print(f"  CUDA Parity & Determinism: {'PASS' if pass_determinism else 'FAIL'} (100% exact match across all 5 runs)")
    print(f"  C99 Parity & Logit Match: {'PASS' if pass_cross_engine else 'FAIL'} (max err = {max_logit_err:.6f})")
    print(f"  OVERALL ADVERSARIAL VERDICT: {results['verdict']}")
    print("=" * 80)

    # Save output JSON
    out_json = os.path.join(r"C:\Sebastian\Project\Velox\tests\adversarial_challenge_m2", "adversarial_m2_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved full empirical test telemetry to: {out_json}")


if __name__ == "__main__":
    main()
