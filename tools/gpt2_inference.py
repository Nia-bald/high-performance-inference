#!/usr/bin/env python3
"""
GPT-2 inference from raw weights (gpt2_weights.bin) + BPE tokenizer (vocab.json, merges.txt).

Usage:
    python tools/gpt2_inference.py --prompt "Alan Turing was a"
    python tools/gpt2_inference.py --prompt "The meaning of life is" --max_tokens 50 --temperature 0.8
"""

import argparse
import json
import math
import os
import struct
import sys
import time
import re
import numpy as np


# ─────────────────────────── GPT-2 Hyperparameters ───────────────────────────
VOCAB_SIZE  = 50257
MAX_SEQ_LEN = 1024
D_MODEL     = 768
NUM_HEADS   = 12
NUM_LAYERS  = 12
D_FF        = 3072  # 4 * D_MODEL


# ─────────────────────────────── BPE Tokenizer ───────────────────────────────

def bytes_to_unicode():
    """Returns mapping from byte values to unicode chars, matching GPT-2's encoder."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))


class GPT2Tokenizer:
    """Minimal GPT-2 BPE tokenizer using vocab.json + merges.txt."""

    def __init__(self, vocab_path: str, merges_path: str):
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.encoder = json.load(f)
        self.decoder = {v: k for k, v in self.encoder.items()}

        with open(merges_path, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
        # Skip header line if present
        merge_lines = lines[1:] if lines and lines[0].startswith("#") else lines
        merges = [tuple(line.split()) for line in merge_lines if line.strip() and len(line.split()) == 2]
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # Pattern to split text into tokens (GPT-2 style)
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+""",
        )

    def _get_pairs(self, word):
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def _bpe(self, token: str) -> str:
        word = tuple(token)
        pairs = self._get_pairs(word)
        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = self._get_pairs(word)

        return " ".join(word)

    def encode(self, text: str) -> list[int]:
        tokens = []
        for match in re.findall(self.pat, text):
            # Encode each byte to the unicode representation
            encoded = "".join(self.byte_encoder[b] for b in match.encode("utf-8"))
            bpe_tokens = self._bpe(encoded).split(" ")
            tokens.extend(self.encoder[t] for t in bpe_tokens)
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        text = "".join(self.decoder.get(t, "") for t in token_ids)
        return bytearray(self.byte_decoder[c] for c in text).decode("utf-8", errors="replace")


# ──────────────────────────── Weight Loading ─────────────────────────────────

def load_gpt2_weights(path: str):
    """Load all GPT-2 weights from a flat binary file (float32).

    Returns a dict with all weight tensors as numpy arrays.
    """
    print(f"Loading weights from {path} ...", end=" ", flush=True)
    start = time.perf_counter()

    data = np.fromfile(path, dtype=np.float32)
    offset = 0

    def read(shape):
        nonlocal offset
        size = 1
        for s in shape:
            size *= s
        tensor = data[offset : offset + size].reshape(shape).copy()
        offset += size
        return tensor

    weights = {}

    # 1. Embeddings
    weights["token_embed"] = read((VOCAB_SIZE, D_MODEL))      # [V, D]
    weights["pos_embed"]   = read((MAX_SEQ_LEN, D_MODEL))     # [MaxSeq, D]

    # 2. Transformer layers
    layers = []
    for i in range(NUM_LAYERS):
        lw = {}
        # Attention LayerNorm
        lw["attn_norm_gamma"] = read((D_MODEL,))
        lw["attn_norm_beta"]  = read((D_MODEL,))
        # Attention Q/K/V/O
        lw["W_q"] = read((D_MODEL, D_MODEL))
        lw["W_k"] = read((D_MODEL, D_MODEL))
        lw["W_v"] = read((D_MODEL, D_MODEL))
        lw["W_o"] = read((D_MODEL, D_MODEL))
        # FFN LayerNorm
        lw["ffn_norm_gamma"] = read((D_MODEL,))
        lw["ffn_norm_beta"]  = read((D_MODEL,))
        # FFN Up / Down
        lw["W_up"]   = read((D_MODEL, D_FF))
        lw["b_up"]   = read((D_FF,))
        lw["W_down"] = read((D_FF, D_MODEL))
        lw["b_down"] = read((D_MODEL,))
        layers.append(lw)
    weights["layers"] = layers

    # 3. Final LayerNorm
    weights["final_norm_gamma"] = read((D_MODEL,))
    weights["final_norm_beta"]  = read((D_MODEL,))

    # 4. LM Head  – HF GPT-2 exports this as [Vocab, D_model], we need [D, V] for matmul
    lm_head_raw = read((VOCAB_SIZE, D_MODEL))    # stored as [V, D]
    weights["lm_head"] = lm_head_raw.T.copy()    # -> [D, V]

    elapsed = time.perf_counter() - start
    print(f"done ({elapsed:.2f}s, {offset:,} floats)")
    return weights


# ──────────────────────── NumPy Forward Pass ─────────────────────────────────

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var  = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def gelu(x):
    """Approximate GELU matching PyTorch's default."""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def self_attention(x, W_q, W_k, W_v, W_o, num_heads):
    """Multi-head causal self-attention.  x: (B, T, D)"""
    B, T, D = x.shape
    head_dim = D // num_heads

    Q = x @ W_q  # (B, T, D)
    K = x @ W_k
    V = x @ W_v

    # Reshape to (B, H, T, hd)
    Q = Q.reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)

    scores = (Q @ K.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)

    # Causal mask
    mask = np.triu(np.ones((T, T), dtype=np.float32), k=1)
    scores = scores + mask * (-1e9)

    attn_weights = softmax(scores, axis=-1)
    context = attn_weights @ V  # (B, H, T, hd)

    # Concat heads
    context = context.transpose(0, 2, 1, 3).reshape(B, T, D)
    return context @ W_o


def feed_forward(x, W_up, b_up, W_down, b_down):
    hidden = gelu(x @ W_up + b_up)
    return hidden @ W_down + b_down


def transformer_block(x, lw, num_heads):
    # Pre-norm attention
    norm1 = layer_norm(x, lw["attn_norm_gamma"], lw["attn_norm_beta"])
    attn_out = self_attention(norm1, lw["W_q"], lw["W_k"], lw["W_v"], lw["W_o"], num_heads)
    res1 = x + attn_out

    # Pre-norm FFN
    norm2 = layer_norm(res1, lw["ffn_norm_gamma"], lw["ffn_norm_beta"])
    ffn_out = feed_forward(norm2, lw["W_up"], lw["b_up"], lw["W_down"], lw["b_down"])
    return res1 + ffn_out


def transformer_forward(token_ids, weights):
    """Full GPT-2 forward pass.  token_ids: list[int] of length T.  Returns logits (T, V)."""
    T = len(token_ids)
    ids = np.array(token_ids, dtype=np.int64)

    # Embedding
    x = weights["token_embed"][ids] + weights["pos_embed"][:T]  # (T, D)
    x = x[np.newaxis, :, :]  # (1, T, D)

    # Blocks
    for lw in weights["layers"]:
        x = transformer_block(x, lw, NUM_HEADS)

    # Final norm
    x = layer_norm(x, weights["final_norm_gamma"], weights["final_norm_beta"])

    # LM head: (1, T, D) @ (D, V) -> (1, T, V)
    logits = x @ weights["lm_head"]
    return logits[0]  # (T, V)


# ──────────────────────── Sampling / Generation ──────────────────────────────

def sample_top_k(logits, temperature=1.0, top_k=40):
    """Sample from top-k filtered logits with temperature."""
    if temperature == 0:
        return int(np.argmax(logits))

    logits = logits / temperature

    if top_k > 0:
        top_k = min(top_k, len(logits))
        indices = np.argpartition(logits, -top_k)[-top_k:]
        top_logits = logits[indices]
        probs = softmax(top_logits)
        choice = np.random.choice(len(top_logits), p=probs)
        return int(indices[choice])
    else:
        probs = softmax(logits)
        return int(np.random.choice(len(probs), p=probs))


def generate(prompt: str, tokenizer: GPT2Tokenizer, weights: dict,
             max_tokens: int = 30, temperature: float = 0.8, top_k: int = 40):
    """Autoregressive generation from a text prompt."""
    token_ids = tokenizer.encode(prompt)
    print(f"\nPrompt tokens: {token_ids}  ({len(token_ids)} tokens)")
    print(f"Generating up to {max_tokens} tokens  (temp={temperature}, top_k={top_k})\n")
    print(prompt, end="", flush=True)

    for step in range(max_tokens):
        # Truncate to max context
        context = token_ids[-MAX_SEQ_LEN:]

        t0 = time.perf_counter()
        logits = transformer_forward(context, weights)
        elapsed = time.perf_counter() - t0

        # Only use the last token's logits
        next_logits = logits[-1]
        next_id = sample_top_k(next_logits, temperature=temperature, top_k=top_k)
        token_ids.append(next_id)

        # Decode just the new token
        new_text = tokenizer.decode([next_id])
        print(new_text, end="", flush=True)

        # Print timing every 5 tokens
        if (step + 1) % 5 == 0:
            print(f"  [{elapsed*1000:.0f}ms/tok]", end="", flush=True)

    print("\n")
    return tokenizer.decode(token_ids)


# ───────────────────────────────── Main ──────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GPT-2 inference from raw weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python tools/gpt2_inference.py --prompt "Alan Turing was a"
  python tools/gpt2_inference.py --prompt "The capital of France" --max_tokens 20 --temperature 0.0
  python tools/gpt2_inference.py --prompt "Once upon a time" --temperature 1.0 --top_k 50
""",
    )
    parser.add_argument("--prompt", type=str, default="Alan Turing was a",
                        help="Input text prompt")
    parser.add_argument("--max_tokens", type=int, default=30,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (0 = greedy)")
    parser.add_argument("--top_k", type=int, default=40,
                        help="Top-k sampling (0 = disabled)")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to gpt2_weights.bin")
    parser.add_argument("--vocab", type=str, default=None,
                        help="Path to vocab.json")
    parser.add_argument("--merges", type=str, default=None,
                        help="Path to merges.txt")
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_path = args.weights or os.path.join(project_root, "gpt2_weights.bin")
    vocab_path   = args.vocab   or os.path.join(project_root, "vocab.json")
    merges_path  = args.merges  or os.path.join(project_root, "merges.txt")

    # Validate files exist
    for name, path in [("weights", weights_path), ("vocab", vocab_path), ("merges", merges_path)]:
        if not os.path.isfile(path):
            print(f"Error: {name} file not found at {path}", file=sys.stderr)
            sys.exit(1)

    # Load tokenizer
    print("Loading tokenizer ...", end=" ", flush=True)
    tokenizer = GPT2Tokenizer(vocab_path, merges_path)
    print("done")

    # Load weights
    weights = load_gpt2_weights(weights_path)

    # Generate
    generate(args.prompt, tokenizer, weights,
             max_tokens=args.max_tokens,
             temperature=args.temperature,
             top_k=args.top_k)


if __name__ == "__main__":
    main()
