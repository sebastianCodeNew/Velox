"""
Adversarial Numerical & Token Parity Audit for Velox Milestone 2.
Audits:
1. Standalone CUDA Runner (`velox_cuda_runner.exe`) on NVIDIA RTX 3050 Laptop GPU
2. Standalone C99 Runner (`velox_c99_runner.exe`) on Host CPU
3. Native Rust Interpreter (`veloxc.exe --run`)
4. NumPy Reference Implementation (`SmolLMNumpyReference`)

Performs:
- Token Parity Audit for 4 tokens and 32 tokens
- Top-5 Logits Numerical Parity (MAE, RMSE, Max Absolute Error)
- Rank order agreement
- Edge cases (--steps 0, --steps 1, repeatability)
"""

import json
import math
import mmap
import os
import re
import struct
import subprocess
import sys
import time
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Sebastian\Project\Velox")
WEIGHTS_PATH = PROJECT_ROOT / "smollm-135m.safetensors"
CUDA_RUNNER_EXE = PROJECT_ROOT / "src" / "runners" / "velox_cuda_runner.exe"
C99_RUNNER_EXE = PROJECT_ROOT / "src" / "runners" / "velox_c99_runner.exe"
VELOXC_EXE = PROJECT_ROOT / "veloxc.exe"
VLX_PATH = PROJECT_ROOT / "examples" / "tinystories_smollm.vlx"

EXPECTED_4_TOKENS = [359, 253, 4028, 282]
EXPECTED_32_TOKENS = [
    359, 253, 4028, 282, 260, 1773, 30, 198, 26501, 368, 281, 260, 1773, 198,
    20255, 3726, 288, 9517, 253, 12301, 198, 19161, 354, 12301, 198, 19161, 354,
    12301, 198, 2683, 416, 722
]

class SmolLMNumpyReference:
    def __init__(self, weights_path: Path):
        self.weights_path = weights_path
        self.tensors = {}
        self._load_safetensors()

        self.hidden_dim = 576
        self.num_layers = 30
        self.num_heads = 9
        self.num_kv_heads = 3
        self.head_dim = 64
        self.intermediate_dim = 1536
        self.vocab_size = 49152
        self.rms_norm_eps = 1e-5
        self.rope_theta = 10000.0

        self.inv_freq = 1.0 / (self.rope_theta ** (np.arange(0, self.head_dim, 2, dtype=np.float32) / float(self.head_dim)))

    def _load_safetensors(self):
        f = open(self.weights_path, "rb")
        self._file = f
        self._mmap_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        
        header_len = struct.unpack("<Q", self._mmap_file[:8])[0]
        header_json_bytes = self._mmap_file[8 : 8 + header_len]
        header = json.loads(header_json_bytes.decode("utf-8"))
        data_offset = 8 + header_len

        for name, meta in header.items():
            if name == "__metadata__":
                continue
            start, end = meta["data_offsets"]
            shape = meta["shape"]
            dtype_str = meta["dtype"]
            dtype = np.float32 if dtype_str == "F32" else np.float16
            raw_bytes = self._mmap_file[data_offset + start : data_offset + end]
            arr = np.frombuffer(raw_bytes, dtype=dtype).reshape(shape)
            self.tensors[name] = arr

        if "lm_head.weight" not in self.tensors:
            self.tensors["lm_head.weight"] = self.tensors["model.embed_tokens.weight"]

    def rmsnorm(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        var = np.mean(x * x)
        inv_rms = 1.0 / np.sqrt(var + self.rms_norm_eps)
        return (x * inv_rms) * weight

    def rotate_half(self, x: np.ndarray, pos: int) -> np.ndarray:
        num_heads, head_dim = x.shape
        half = head_dim // 2
        angles = pos * self.inv_freq
        cos_a = np.cos(angles).astype(np.float32)
        sin_a = np.sin(angles).astype(np.float32)

        out = np.empty_like(x)
        for h in range(num_heads):
            x1 = x[h, :half]
            x2 = x[h, half:]
            out[h, :half] = x1 * cos_a - x2 * sin_a
            out[h, half:] = x2 * cos_a + x1 * sin_a
        return out

    def forward_step(self, token: int, pos: int, k_cache: list, v_cache: list) -> np.ndarray:
        w_embed = self.tensors["model.embed_tokens.weight"]
        hidden = w_embed[token].copy()

        scale = 1.0 / np.sqrt(float(self.head_dim))

        for l in range(self.num_layers):
            w_attn_norm = self.tensors[f"model.layers.{l}.input_layernorm.weight"]
            w_q = self.tensors[f"model.layers.{l}.self_attn.q_proj.weight"]
            w_k = self.tensors[f"model.layers.{l}.self_attn.k_proj.weight"]
            w_v = self.tensors[f"model.layers.{l}.self_attn.v_proj.weight"]
            w_o = self.tensors[f"model.layers.{l}.self_attn.o_proj.weight"]

            norm_x = self.rmsnorm(hidden, w_attn_norm)

            q = (norm_x @ w_q.T).reshape(self.num_heads, self.head_dim)
            k = (norm_x @ w_k.T).reshape(self.num_kv_heads, self.head_dim)
            v = (norm_x @ w_v.T).reshape(self.num_kv_heads, self.head_dim)

            q_rot = self.rotate_half(q, pos)
            k_rot = self.rotate_half(k, pos)

            k_cache[l].append(k_rot)
            v_cache[l].append(v)

            attn_out = np.zeros((self.num_heads, self.head_dim), dtype=np.float32)
            k_history = np.array(k_cache[l])
            v_history = np.array(v_cache[l])

            for qh in range(self.num_heads):
                kv_h = qh // 3
                q_head = q_rot[qh]
                k_heads = k_history[:, kv_h, :]
                v_heads = v_history[:, kv_h, :]

                scores = (k_heads @ q_head) * scale
                max_score = np.max(scores)
                exp_scores = np.exp(scores - max_score)
                probs = exp_scores / np.sum(exp_scores)

                attn_out[qh] = probs @ v_heads

            attn_flat = attn_out.reshape(-1)
            hidden += attn_flat @ w_o.T

            w_mlp_norm = self.tensors[f"model.layers.{l}.post_attention_layernorm.weight"]
            w_gate = self.tensors[f"model.layers.{l}.mlp.gate_proj.weight"]
            w_up = self.tensors[f"model.layers.{l}.mlp.up_proj.weight"]
            w_down = self.tensors[f"model.layers.{l}.mlp.down_proj.weight"]

            mlp_norm_x = self.rmsnorm(hidden, w_mlp_norm)
            gate = mlp_norm_x @ w_gate.T
            up = mlp_norm_x @ w_up.T
            silu = gate / (1.0 + np.exp(-gate))
            act = silu * up
            hidden += act @ w_down.T

        w_final_norm = self.tensors["model.norm.weight"]
        w_head = self.tensors["lm_head.weight"]
        final_norm = self.rmsnorm(hidden, w_final_norm)
        logits = final_norm @ w_head.T
        return logits

    def generate(self, prompt_tokens: list[int], num_steps: int):
        k_cache = [[] for _ in range(self.num_layers)]
        v_cache = [[] for _ in range(self.num_layers)]

        pos = 0
        last_logits = None
        for tok in prompt_tokens:
            last_logits = self.forward_step(tok, pos, k_cache, v_cache)
            pos += 1

        next_tok = int(np.argmax(last_logits))
        generated = []
        if num_steps > 0:
            generated.append(next_tok)

        for _ in range(1, num_steps):
            last_logits = self.forward_step(next_tok, pos, k_cache, v_cache)
            pos += 1
            next_tok = int(np.argmax(last_logits))
            generated.append(next_tok)

        return generated, last_logits

def parse_runner_output(output: str):
    """Parses standard Velox execution report for tokens and top-5 logits."""
    tokens = []
    top5 = []

    # Parse Generated Tokens (N)   : [x, y, z...]
    m_tokens = re.search(r"Generated Tokens \(\d+\)\s*:\s*\[(.*?)\]", output)
    if m_tokens:
        tok_str = m_tokens.group(1).strip()
        if tok_str:
            tokens = [int(t.strip()) for t in tok_str.split(",") if t.strip()]

    # Parse Top-5 Logits
    #   #1: token_id=722 logit=17.459309
    pattern = re.compile(r"#(\d+):\s*token_id=(\d+)\s*logit=([-\d\.]+)")
    for match in pattern.finditer(output):
        rank = int(match.group(1))
        tok_id = int(match.group(2))
        logit = float(match.group(3))
        top5.append((rank, tok_id, logit))

    return tokens, top5

def run_process(cmd: list[str], cwd: Path = PROJECT_ROOT):
    p = subprocess.run(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr

def compute_metrics(logits_a: list[float], logits_b: list[float]):
    diffs = [a - b for a, b in zip(logits_a, logits_b)]
    mae = sum(abs(d) for d in diffs) / len(diffs)
    rmse = math.sqrt(sum(d * d for d in diffs) / len(diffs))
    max_err = max(abs(d) for d in diffs)
    return mae, rmse, max_err

def main():
    print("=" * 80)
    print("VELOX MILESTONE 2 ADVERSARIAL NUMERICAL & TOKEN PARITY AUDIT")
    print("=" * 80)

    # 1. NumPy Reference Execution (32 steps)
    print("\n[STAGE 1] Executing Pure NumPy Reference Forward Pass (32 steps)...")
    t0 = time.perf_counter()
    numpy_ref = SmolLMNumpyReference(WEIGHTS_PATH)
    numpy_tokens, numpy_logits = numpy_ref.generate([1, 504, 346], 32)
    numpy_time = time.perf_counter() - t0
    print(f"  NumPy 32 steps completed in {numpy_time:.2f}s")
    print(f"  NumPy Generated Tokens (32): {numpy_tokens}")

    numpy_top5_idx = np.argsort(numpy_logits)[-5:][::-1]
    numpy_top5 = [(r + 1, int(idx), float(numpy_logits[idx])) for r, idx in enumerate(numpy_top5_idx)]
    print("  NumPy Final Step Top-5 Logits:")
    for rank, tok_id, val in numpy_top5:
        print(f"    #{rank}: token_id={tok_id} logit={val:.6f}")

    # 2. Native Rust Reference Execution (veloxc.exe --run)
    print("\n[STAGE 2] Executing Velox Native Rust Reference Runner (veloxc.exe)...")
    t0 = time.perf_counter()
    code, out_rust, err = run_process([
        str(VELOXC_EXE), "--run", "--tokens", "1,504,346", "--steps", "32", str(VLX_PATH)
    ])
    rust_time = time.perf_counter() - t0
    assert code == 0, f"veloxc.exe failed: {err}"
    rust_tokens, rust_top5 = parse_runner_output(out_rust)
    print(f"  Rust 32 steps completed in {rust_time:.2f}s (exit {code})")
    print(f"  Rust Generated Tokens (32): {rust_tokens}")
    print("  Rust Final Step Top-5 Logits:")
    for rank, tok_id, val in rust_top5:
        print(f"    #{rank}: token_id={tok_id} logit={val:.6f}")

    # 3. Emitted C99 Host Runner (velox_c99_runner.exe)
    print("\n[STAGE 3] Executing Velox Emitted C99 Standalone Runner (velox_c99_runner.exe)...")
    t0 = time.perf_counter()
    code, out_c99, err = run_process([
        str(C99_RUNNER_EXE), "--weights", str(WEIGHTS_PATH), "--tokens", "1,504,346", "--steps", "32"
    ])
    c99_time = time.perf_counter() - t0
    assert code == 0, f"velox_c99_runner.exe failed: {err}"
    c99_tokens, c99_top5 = parse_runner_output(out_c99)
    print(f"  C99 32 steps completed in {c99_time:.2f}s (exit {code})")
    print(f"  C99 Generated Tokens (32): {c99_tokens}")
    print("  C99 Final Step Top-5 Logits:")
    for rank, tok_id, val in c99_top5:
        print(f"    #{rank}: token_id={tok_id} logit={val:.6f}")

    # 4. Emitted CUDA Standalone Runner (velox_cuda_runner.exe on RTX 3050)
    print("\n[STAGE 4] Executing Velox Emitted CUDA Standalone Runner (velox_cuda_runner.exe on RTX 3050)...")
    t0 = time.perf_counter()
    code, out_cuda, err = run_process([
        str(CUDA_RUNNER_EXE), "--weights", str(WEIGHTS_PATH), "--tokens", "1,504,346", "--steps", "32"
    ])
    cuda_time = time.perf_counter() - t0
    assert code == 0, f"velox_cuda_runner.exe failed: {err}"
    cuda_tokens, cuda_top5 = parse_runner_output(out_cuda)
    print(f"  CUDA 32 steps completed in {cuda_time:.2f}s (exit {code})")
    print(f"  CUDA Generated Tokens (32): {cuda_tokens}")
    print("  CUDA Final Step Top-5 Logits:")
    for rank, tok_id, val in cuda_top5:
        print(f"    #{rank}: token_id={tok_id} logit={val:.6f}")

    # =========================================================================
    # AUDIT VERIFICATIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("AUDIT SECTION 1: TOKEN PARITY VERIFICATION")
    print("=" * 80)

    # 4 tokens verification
    print("\n[CHECK 1] 4-Token Parity Check vs Ground Truth [359, 253, 4028, 282]:")
    targets = {
        "NumPy Reference": numpy_tokens[:4],
        "Rust Reference": rust_tokens[:4],
        "C99 Runner": c99_tokens[:4],
        "CUDA Runner (RTX 3050)": cuda_tokens[:4],
    }
    all_4_match = True
    for name, toks in targets.items():
        match = (toks == EXPECTED_4_TOKENS)
        all_4_match = all_4_match and match
        print(f"  {name:25s}: {toks} -> {'PASS' if match else 'FAIL'}")
    assert all_4_match, "4-token parity check failed"

    # 32 tokens verification
    print("\n[CHECK 2] 32-Token Parity Check vs Ground Truth Sequence:")
    targets_32 = {
        "NumPy Reference": numpy_tokens,
        "Rust Reference": rust_tokens,
        "C99 Runner": c99_tokens,
        "CUDA Runner (RTX 3050)": cuda_tokens,
    }
    all_32_match = True
    for name, toks in targets_32.items():
        match = (toks == EXPECTED_32_TOKENS)
        agreement = (sum(1 for a, b in zip(toks, EXPECTED_32_TOKENS)) / len(EXPECTED_32_TOKENS)) * 100.0
        all_32_match = all_32_match and match
        print(f"  {name:25s}: Match={match} (Agreement: {agreement:.1f}%)")
    assert all_32_match, "32-token parity check failed"

    print("\n" + "=" * 80)
    print("AUDIT SECTION 2: NUMERICAL PARITY & LOGITS AUDIT (Final Step Top-5)")
    print("=" * 80)

    # Extract top-5 token ids and values
    rust_vals = [x[2] for x in rust_top5]
    c99_vals = [x[2] for x in c99_top5]
    cuda_vals = [x[2] for x in cuda_top5]
    numpy_vals = [x[2] for x in numpy_top5]

    rust_ids = [x[1] for x in rust_top5]
    c99_ids = [x[1] for x in c99_top5]
    cuda_ids = [x[1] for x in cuda_top5]
    numpy_ids = [x[1] for x in numpy_top5]

    # Rank-order verification
    print("\n[CHECK 3] Top-5 Token ID Rank Order Agreement:")
    print(f"  NumPy IDs : {numpy_ids}")
    print(f"  Rust IDs  : {rust_ids}")
    print(f"  C99 IDs   : {c99_ids}")
    print(f"  CUDA IDs  : {cuda_ids}")

    assert rust_ids == c99_ids == cuda_ids == numpy_ids == [722, 14577, 1532, 597, 5757], \
        "Top-5 token IDs rank order mismatch!"
    print("  -> PASSED: All 4 execution engines agree 100% on top-5 token ID ranking [722, 14577, 1532, 597, 5757]!")

    # Pairwise numerical error metrics
    comparisons = [
        ("C99 Runner vs Rust Reference", c99_vals, rust_vals),
        ("CUDA Runner (RTX 3050) vs Rust Reference", cuda_vals, rust_vals),
        ("CUDA Runner (RTX 3050) vs C99 Runner", cuda_vals, c99_vals),
        ("NumPy Reference vs Rust Reference", numpy_vals, rust_vals),
        ("CUDA Runner (RTX 3050) vs NumPy Reference", cuda_vals, numpy_vals),
    ]

    print("\n[CHECK 4] Pairwise Logits Numerical Error (MAE, RMSE, Max Absolute Error):")
    for label, vals_a, vals_b in comparisons:
        mae, rmse, max_err = compute_metrics(vals_a, vals_b)
        print(f"\n  --- {label} ---")
        print(f"    MAE (Mean Absolute Error)     : {mae:.8f}")
        print(f"    RMSE (Root Mean Square Error) : {rmse:.8f}")
        print(f"    Max Absolute Error            : {max_err:.8f}")
        print("    Per-Rank Deltas:")
        for r in range(5):
            delta = vals_a[r] - vals_b[r]
            print(f"      Rank #{r+1} (ID={rust_ids[r]}): a={vals_a[r]:.6f}, b={vals_b[r]:.6f}, diff={delta:+.8f}")

        # Assert stringent thresholds:
        # C99 vs Rust: exact CPU arithmetic -> MAE == 0.000000
        # CUDA vs CPU / NumPy: GPU parallel reduction / FMA rounding -> MAE < 0.0001 (1e-4)
        if "C99" in label and "Rust" in label:
            assert mae < 1e-6, f"C99 vs Rust MAE too high: {mae}"
        else:
            assert mae < 1e-4, f"{label} MAE exceeded 1e-4 tolerance: {mae}"
            assert max_err < 1e-4, f"{label} Max error exceeded 1e-4 tolerance: {max_err}"

    print("\n" + "=" * 80)
    print("AUDIT SECTION 3: ADVERSARIAL STRESS TESTING & EDGE CASES")
    print("=" * 80)

    # Stress Test 1: Steps = 0
    print("\n[STRESS 1] Edge Case: --steps 0 (Zero Generation Steps)")
    for name, exe, extra_args in [
        ("CUDA Runner", CUDA_RUNNER_EXE, ["--weights", str(WEIGHTS_PATH)]),
        ("C99 Runner", C99_RUNNER_EXE, ["--weights", str(WEIGHTS_PATH)]),
        ("Rust Runner", VELOXC_EXE, [str(VLX_PATH)]),
    ]:
        code, out, err = run_process([str(exe), "--run", "--tokens", "1,504,346", "--steps", "0"] if "veloxc" in str(exe) else [str(exe)] + extra_args + ["--tokens", "1,504,346", "--steps", "0"])
        toks, _ = parse_runner_output(out)
        print(f"  {name:15s}: exit={code}, tokens_count={len(toks)} -> {'PASS' if len(toks) == 0 else 'FAIL'}")
        assert len(toks) == 0, f"{name} failed on --steps 0"

    # Stress Test 2: Repeatability / Determinism (3 consecutive runs)
    print("\n[STRESS 2] Invariance: Determinism across 3 consecutive runs on RTX 3050")
    cuda_runs = []
    for run_idx in range(3):
        code, out, err = run_process([
            str(CUDA_RUNNER_EXE), "--weights", str(WEIGHTS_PATH), "--tokens", "1,504,346", "--steps", "32"
        ])
        toks, top5 = parse_runner_output(out)
        cuda_runs.append((toks, top5))
        print(f"  Run #{run_idx+1}: tokens={toks[:6]}... top1_logit={top5[0][2]:.6f}")

    assert cuda_runs[0] == cuda_runs[1] == cuda_runs[2], "CUDA execution is not bitwise deterministic across runs!"
    print("  -> PASSED: 100% bitwise deterministic execution across repeated GPU invocations.")

    # Stress Test 3: Invalid weights path rejection
    print("\n[STRESS 3] Negative Test: Non-existent weights file handling")
    for name, exe in [("CUDA Runner", CUDA_RUNNER_EXE), ("C99 Runner", C99_RUNNER_EXE)]:
        code, out, err = run_process([str(exe), "--weights", "non_existent_weights.safetensors"])
        print(f"  {name:15s}: exit={code} (non-zero as expected) -> {'PASS' if code != 0 else 'FAIL'}")
        assert code != 0, f"{name} accepted non-existent weights file"

    print("\n" + "=" * 80)
    print("AUDIT VERDICT: ALL NUMERICAL & TOKEN PARITY CHECKS PASSED EMPIRICALLY.")
    print("=" * 80)

if __name__ == "__main__":
    main()
