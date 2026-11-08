"""
Velox High-Performance Zero-Dependency BPE Tokenizer
Supports standard HuggingFace tokenizer.json (GPT-2 style Byte-Pair Encoding)
Used by SmolLM-135M and K2-Horizon-3.7B
"""

import json
import re
from typing import Dict, List, Optional, Tuple, Set


def bytes_to_unicode() -> Dict[int, str]:
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab.
    The standard map preserves all byte values cleanly.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class VeloxTokenizer:
    def __init__(self, tokenizer_json_path: str):
        self.tokenizer_path = tokenizer_json_path
        with open(tokenizer_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # Vocab: token string -> token id
        model_data = data.get("model", {})
        self.vocab: Dict[str, int] = model_data.get("vocab", {})
        self.id_to_vocab: Dict[int, str] = {v: k for k, v in self.vocab.items()}

        # Added tokens (special tokens like <|im_start|>, <|im_end|>, etc.)
        self.special_tokens: Dict[str, int] = {}
        for item in data.get("added_tokens", []):
            tok_id = item["id"]
            content = item["content"]
            self.special_tokens[content] = tok_id
            self.vocab[content] = tok_id
            self.id_to_vocab[tok_id] = content

        # Merges
        merges = model_data.get("merges", [])
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}
        for idx, merge in enumerate(merges):
            if isinstance(merge, str):
                parts = merge.split(" ")
                if len(parts) == 2:
                    self.bpe_ranks[(parts[0], parts[1])] = idx
            elif isinstance(merge, list) and len(merge) == 2:
                self.bpe_ranks[(merge[0], merge[1])] = idx

        self.cache: Dict[str, List[str]] = {}
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d|[^\s]+|\s+""",
            re.IGNORECASE,
        )

        # Regex pattern for special tokens
        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens.keys(), key=lambda s: len(s), reverse=True)
            escaped_specials = [re.escape(s) for s in sorted_specials]
            self.special_pat = re.compile("(" + "|".join(escaped_specials) + ")")
        else:
            self.special_pat = None

    def get_pairs(self, word: Tuple[str, ...]) -> Set[Tuple[str, str]]:
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def bpe(self, token: str) -> List[str]:
        if token in self.cache:
            return self.cache[token]

        word = tuple(token)
        pairs = self.get_pairs(word)
        if not pairs:
            return [token]

        while True:
            # Find the pair with the lowest rank
            min_pair = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if min_pair not in self.bpe_ranks:
                break
            first, second = min_pair
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            else:
                pairs = self.get_pairs(word)

        result = list(word)
        self.cache[token] = result
        return result

    def encode(self, text: str) -> List[int]:
        """Encode string to list of token IDs."""
        tokens: List[int] = []

        # If special tokens exist, split on them first
        if self.special_pat:
            parts = self.special_pat.split(text)
        else:
            parts = [text]

        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                tokens.append(self.special_tokens[part])
                continue

            # Standard BPE encoding for text chunks
            # Byte-encode characters
            # Python standard re doesn't support \p{L}, fallback to regex pattern:
            words = re.findall(r"""'s|'t|'re|'ve|'m|'ll|'d|[^\r\n\s]+|\s+""", part)
            if not words:
                words = [part]

            for word in words:
                # Map each byte to unicode
                byte_encoded = "".join(self.byte_encoder[b] for b in word.encode("utf-8"))
                for bpe_token in self.bpe(byte_encoded):
                    if bpe_token in self.vocab:
                        tokens.append(self.vocab[bpe_token])
                    elif bpe_token in self.special_tokens:
                        tokens.append(self.special_tokens[bpe_token])
                    else:
                        # Character level fallback
                        for ch in bpe_token:
                            if ch in self.vocab:
                                tokens.append(self.vocab[ch])

        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decode list of token IDs back into string."""
        raw_bytes = bytearray()
        for tid in token_ids:
            if tid in self.id_to_vocab:
                token_str = self.id_to_vocab[tid]
                if skip_special_tokens and token_str in self.special_tokens:
                    continue
                # If special token, output as utf-8 directly
                if token_str in self.special_tokens:
                    raw_bytes.extend(token_str.encode("utf-8", errors="replace"))
                else:
                    for char in token_str:
                        if char in self.byte_decoder:
                            raw_bytes.append(self.byte_decoder[char])
                        else:
                            raw_bytes.extend(char.encode("utf-8", errors="replace"))
        return raw_bytes.decode("utf-8", errors="replace")


if __name__ == "__main__":
    # Test SmolLM tokenizer
    import os
    if os.path.exists("tokenizer.json"):
        tok = VeloxTokenizer("tokenizer.json")
        sample = "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)"
        encoded = tok.encode(sample)
        decoded = tok.decode(encoded)
        print("SmolLM Vocab size:", len(tok.vocab))
        print("Sample encoded length:", len(encoded))
        print("Roundtrip match:", sample == decoded)
