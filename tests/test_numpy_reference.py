"""
NumPy Reference Implementation for SmolLM-135M.
Used by Velox adversarial challenge to audit numerical parity.
"""

import json
import mmap
import os
import struct
import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Sebastian\Project\Velox")
WEIGHTS_PATH = PROJECT_ROOT / "smollm-135m.safetensors"

class SmolLMNumpyReference:
    def __init__(self, weights_path: Path):
        self.weights_path = weights_path
        self._mmap_file = None
        self.tensors = {}
        self._load_safetensors()

        # Architecture constants
        self.hidden_dim = 576
        self.num_layers = 30
        self.num_heads = 9
        self.num_kv_heads = 3
        self.head_dim = 64
        self.intermediate_dim = 1536
        self.vocab_size = 49152
        self.rms_norm_eps = 1e-5
        self.rope_theta = 10000.0

        # Precompute RoPE inv_freq
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
            if dtype_str == "F32":
                dtype = np.float32
            elif dtype_str == "F16":
                dtype = np.float16
            else:
                dtype = np.float32
            
            raw_bytes = self._mmap_file[data_offset + start : data_offset + end]
            arr = np.frombuffer(raw_bytes, dtype=dtype).reshape(shape)
            self.tensors[name] = arr

        # Tied lm_head
        if "lm_head.weight" not in self.tensors:
            self.tensors["lm_head.weight"] = self.tensors["model.embed_tokens.weight"]

    def rmsnorm(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        # x: [dim], weight: [dim]
        # inv_rms = 1.0 / sqrt(mean(x^2) + eps)
        # matches velox runtime: mean(x^2), inv_rms = 1.0 / sqrt(mean + eps)
        var = np.mean(x * x)
        inv_rms = 1.0 / np.sqrt(var + self.rms_norm_eps)
        return (x * inv_rms) * weight

    def rotate_half(self, x: np.ndarray, pos: int) -> np.ndarray:
        # x: [num_heads, head_dim]
        # rotate_half: pair (x[j], x[j + half])
        # x1 * cos - x2 * sin, x2 * cos + x1 * sin
        num_heads, head_dim = x.shape
        half = head_dim // 2
        angles = pos * self.inv_freq # shape [half]
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
        """Runs a single forward step, appending to k_cache and v_cache, returns logits [vocab_size]."""
        w_embed = self.tensors["model.embed_tokens.weight"]
        hidden = w_embed[token].copy() # [hidden_dim]

        scale = 1.0 / np.sqrt(float(self.head_dim))

        for l in range(self.num_layers):
            w_attn_norm = self.tensors[f"model.layers.{l}.input_layernorm.weight"]
            w_q = self.tensors[f"model.layers.{l}.self_attn.q_proj.weight"]
            w_k = self.tensors[f"model.layers.{l}.self_attn.k_proj.weight"]
            w_v = self.tensors[f"model.layers.{l}.self_attn.v_proj.weight"]
            w_o = self.tensors[f"model.layers.{l}.self_attn.o_proj.weight"]

            norm_x = self.rmsnorm(hidden, w_attn_norm)

            # Matmul: y = norm_x @ W.T
            q = norm_x @ w_q.T # [576]
            k = norm_x @ w_k.T # [192]
            v = norm_x @ w_v.T # [192]

            q = q.reshape(self.num_heads, self.head_dim)
            k = k.reshape(self.num_kv_heads, self.head_dim)
            v = v.reshape(self.num_kv_heads, self.head_dim)

            # RoPE
            q_rot = self.rotate_half(q, pos)
            k_rot = self.rotate_half(k, pos)

            # Cache K and V
            k_cache[l].append(k_rot)
            v_cache[l].append(v)

            # Attention: GQA
            # For each Q head qh in 0..8, kv head = qh // 3
            attn_out = np.zeros((self.num_heads, self.head_dim), dtype=np.float32)
            seq_len = pos + 1

            # Stack cached keys & values: [seq_len, num_kv_heads, head_dim]
            k_history = np.array(k_cache[l]) # [seq_len, num_kv_heads, head_dim]
            v_history = np.array(v_cache[l]) # [seq_len, num_kv_heads, head_dim]

            for qh in range(self.num_heads):
                kv_h = qh // 3
                q_head = q_rot[qh] # [64]
                k_heads = k_history[:, kv_h, :] # [seq_len, 64]
                v_heads = v_history[:, kv_h, :] # [seq_len, 64]

                # scores = (q @ k.T) * scale
                scores = (k_heads @ q_head) * scale # [seq_len]
                max_score = np.max(scores)
                exp_scores = np.exp(scores - max_score)
                probs = exp_scores / np.sum(exp_scores) # [seq_len]

                attn_out[qh] = probs @ v_heads

            attn_flat = attn_out.reshape(-1) # [576]
            hidden += attn_flat @ w_o.T

            # MLP SwiGLU
            w_mlp_norm = self.tensors[f"model.layers.{l}.post_attention_layernorm.weight"]
            w_gate = self.tensors[f"model.layers.{l}.mlp.gate_proj.weight"]
            w_up = self.tensors[f"model.layers.{l}.mlp.up_proj.weight"]
            w_down = self.tensors[f"model.layers.{l}.mlp.down_proj.weight"]

            mlp_norm_x = self.rmsnorm(hidden, w_mlp_norm)
            gate = mlp_norm_x @ w_gate.T
            up = mlp_norm_x @ w_up.T
            # silu(gate) = gate / (1 + exp(-gate))
            silu = gate / (1.0 + np.exp(-gate))
            act = silu * up
            hidden += act @ w_down.T

        # Final Norm + Tied LM Head
        w_final_norm = self.tensors["model.norm.weight"]
        w_head = self.tensors["lm_head.weight"]
        final_norm = self.rmsnorm(hidden, w_final_norm)
        logits = final_norm @ w_head.T
        return logits

    def generate(self, prompt_tokens: list[int], num_steps: int):
        k_cache = [[] for _ in range(self.num_layers)]
        v_cache = [[] for _ in range(self.num_layers)]

        # Prefill
        pos = 0
        last_logits = None
        for tok in prompt_tokens:
            last_logits = self.forward_step(tok, pos, k_cache, v_cache)
            pos += 1

        next_tok = int(np.argmax(last_logits))
        generated = []
        if num_steps > 0:
            generated.append(next_tok)

        # Decode
        for _ in range(1, num_steps):
            last_logits = self.forward_step(next_tok, pos, k_cache, v_cache)
            pos += 1
            next_tok = int(np.argmax(last_logits))
            generated.append(next_tok)

        return generated, last_logits

if __name__ == "__main__":
    ref = SmolLMNumpyReference(WEIGHTS_PATH)
    print("Loaded SmolLMNumpyReference, generating 4 tokens...")
    tokens, logits = ref.generate([1, 504, 346], 4)
    print("Generated tokens:", tokens)
    top5_idx = np.argsort(logits)[-5:][::-1]
    print("Final top-5 logits:")
    for rank, idx in enumerate(top5_idx, 1):
        print(f"  #{rank}: token_id={idx} logit={logits[idx]:.6f}")
