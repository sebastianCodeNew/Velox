"""
Velox Silicon Bridge: Subprocess Execution & Streaming Interface
Connects Python Chat Frontends directly to Native C99/CUDA Bare-Metal Binaries.
Strict Invariants: Zero external serving frameworks (0 Ollama, 0 llama.cpp, 0 PyTorch).
"""

import os
import re
import subprocess
import time
from typing import Callable, Dict, Generator, List, Optional, Tuple

from src.runners.velox_tokenizer import VeloxTokenizer


class VeloxBridge:
    def __init__(self, model_name: str = "smollm", project_root: Optional[str] = None):
        self.model_name = model_name
        self.project_root = project_root or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

        # Setup paths
        if model_name == "k2" or model_name == "k2_horizon":
            self.model_name = "k2_horizon"
            self.runner_exe = os.path.join(self.project_root, "src", "runners", "velox_cuda_runner_k2.exe")
            self.weights_path = os.path.join(self.project_root, "k2-horizon-3.7b-int4.safetensors")
            self.tokenizer_path = os.path.join(self.project_root, "k2_tokenizer.json")
            self.stop_tokens = {1, 250000, 250001, 250002} # <|ifm|endoftext|>
        else:
            self.model_name = "smollm"
            self.runner_exe = os.path.join(self.project_root, "src", "runners", "velox_cuda_runner.exe")
            self.weights_path = os.path.join(self.project_root, "smollm-135m.safetensors")
            self.tokenizer_path = os.path.join(self.project_root, "tokenizer.json")
            self.stop_tokens = {0, 2} # <|endoftext|>, <|im_end|>

        # Load tokenizer
        if not os.path.exists(self.tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found: {self.tokenizer_path}")
        self.tokenizer = VeloxTokenizer(self.tokenizer_path)

    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        context_len: int = 4096,
        on_token: Optional[Callable[[str, int], None]] = None,
    ) -> Generator[Tuple[str, int], None, Dict[str, any]]:
        """
        Executes native Velox CUDA runner with prompt tokens.
        Streams decoded text tokens live as the GPU generates them.
        Returns hardware and inference metrics at completion.
        """
        prompt_tokens = self.tokenizer.encode(prompt)
        tokens_str = ",".join(str(t) for t in prompt_tokens)

        cmd = [
            self.runner_exe,
            "--tokens", tokens_str,
            "--steps", str(max_tokens),
            "--verbose",
        ]
        if os.path.exists(self.weights_path):
            cmd.extend(["--weights", self.weights_path])
        if self.model_name == "k2_horizon":
            cmd.extend(["--context-len", str(context_len)])

        # Run process
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=self.project_root,
        )

        token_regex = re.compile(r"\[Step \d+\] (?:Prompt Token: \d+ -> Next: (\d+)|Token: (\d+))")
        metrics: Dict[str, any] = {
            "model": self.model_name,
            "prompt_tokens_len": len(prompt_tokens),
            "generated_tokens": [],
            "generated_text": "",
            "ttft_ms": 0.0,
            "decode_ms": 0.0,
            "tok_per_sec": 0.0,
            "vram_allocated_mb": 0.0,
            "kv_cache_desc": "",
            "total_inference_ms": 0.0,
        }

        generated_token_ids: List[int] = []

        for line in proc.stdout:
            line_str = line.strip()
            match = token_regex.search(line_str)
            if match:
                tok_id = int(match.group(1) or match.group(2))
                if tok_id in self.stop_tokens:
                    # Encountered stop token
                    break
                generated_token_ids.append(tok_id)
                tok_text = self.tokenizer.decode([tok_id])
                if on_token:
                    on_token(tok_text, tok_id)
                yield (tok_text, tok_id)

            # Parse metrics from execution report
            if "TTFT (Prefill Latency)" in line_str:
                m = re.search(r":\s*([0-9.]+)\s*ms", line_str)
                if m:
                    metrics["ttft_ms"] = float(m.group(1))
            elif "Decode Latency" in line_str:
                m = re.search(r"\(([0-9.]+)\s*tok/s", line_str)
                if m:
                    metrics["tok_per_sec"] = float(m.group(1))
                m_lat = re.search(r":\s*([0-9.]+)\s*ms", line_str)
                if m_lat:
                    metrics["decode_ms"] = float(m_lat.group(1))
            elif "Device VRAM Allocated" in line_str or "Total VRAM Allocated" in line_str:
                m = re.search(r":\s*([0-9.]+)\s*MiB", line_str)
                if m:
                    metrics["vram_allocated_mb"] = float(m.group(1))
            elif "KV Cache Allocation" in line_str:
                metrics["kv_cache_desc"] = line_str.split(":", 1)[1].strip()
            elif "Total GPU Inference" in line_str:
                m = re.search(r":\s*([0-9.]+)\s*ms", line_str)
                if m:
                    metrics["total_inference_ms"] = float(m.group(1))

        proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except Exception:
            pass

        metrics["generated_tokens"] = generated_token_ids
        metrics["generated_text"] = self.tokenizer.decode(generated_token_ids)
        return metrics
