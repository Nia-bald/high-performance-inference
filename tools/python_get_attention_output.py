import torch
import torch.nn.functional as F
import math
import time
import sys
import json

def torch_mha(x, W_q, W_k, W_v, W_o, num_heads, verbose=False):
    """
    Multi-head self-attention (causal) reference implementation.

    Args:
        x:  (B, T, D) input tensor
        W_q, W_k, W_v, W_o:  (D, D) weight matrices  (or custom dims for GQA)
        num_heads: number of attention heads
        verbose: print step-by-step logging

    Returns:
        output: (B, T, D) tensor
    """
    B, T, D = x.shape
    head_dim = D // num_heads

    if verbose:
        print("=" * 50)
        print("🚀 STARTING ATTENTION BLOCK (EXTREME LOGGING)")
        print("=" * 50)
        print(f"[INIT] Batch: {B}, SeqLen: {T}, Dim: {D}, Heads: {num_heads}, HeadDim: {head_dim}\n")
        print("-" * 30 + "\nSTEP 0: Initial Inputs and Weights")
        print(f"Input x | Shape {x.shape}:\n{x}\n")
        print(f"W_q | Shape {W_q.shape}:\n{W_q}\n")
        print(f"W_k | Shape {W_k.shape}:\n{W_k}\n")
        print(f"W_v | Shape {W_v.shape}:\n{W_v}\n")
        print(f"W_o | Shape {W_o.shape}:\n{W_o}\n")

    # --- 1. Compute Q, K, V ---
    if verbose:
        print("-" * 30 + "\nSTEP 1: Compute Linear Projections (Q, K, V)")
    Q = x @ W_q
    K = x @ W_k
    V = x @ W_v
    if verbose:
        print(f"Q (x @ W_q) | Shape {Q.shape}:\n{Q}")
        print(f"K (x @ W_k) | Shape {K.shape}:\n{K}")
        print(f"V (x @ W_v) | Shape {V.shape}:\n{V}\n")

    # --- 2. Reshape and Permute ---
    if verbose:
        print("-" * 30 + "\nSTEP 2: Reshape and Permute for Multi-Head")
    Q_heads = Q.view(B, T, num_heads, head_dim).transpose(1, 2)
    K_heads = K.view(B, T, num_heads, head_dim).transpose(1, 2)
    V_heads = V.view(B, T, num_heads, head_dim).transpose(1, 2)
    if verbose:
        print(f"Q_heads | Shape {Q_heads.shape}:\n{Q_heads}")
        print(f"K_heads | Shape {K_heads.shape}:\n{K_heads}")
        print(f"V_heads | Shape {V_heads.shape}:\n{V_heads}\n")

    # --- 3. Scaled Dot-Product ---
    if verbose:
        print("-" * 30 + "\nSTEP 3: Scaled Dot-Product (Q @ K^T / sqrt(head_dim))")
    K_T = K_heads.transpose(-2, -1)
    scale_factor = math.sqrt(head_dim)
    raw_scores = Q_heads @ K_T
    scores = raw_scores / scale_factor
    if verbose:
        print(f"K_T (Transposed K) | Shape {K_T.shape}:\n{K_T}")
        print(f"Raw Scores (Q @ K_T) | Shape {raw_scores.shape}:\n{raw_scores}")
        print(f"Scaled Scores (Raw / {scale_factor:.4f}) | Shape {scores.shape}:\n{scores}\n")

    # --- 4. Apply Causal Mask ---
    if verbose:
        print("-" * 30 + "\nSTEP 4: Apply Causal Mask")
    mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
    scores.masked_fill_(mask, float('-inf'))
    if verbose:
        print(f"Boolean Causal Mask | Shape {mask.shape}:\n{mask}")
        print(f"Masked Scores | Shape {scores.shape}:\n{scores}\n")

    # --- 5. Softmax ---
    if verbose:
        print("-" * 30 + "\nSTEP 5: Softmax (Attention Weights)")
    attn_weights = F.softmax(scores, dim=-1)
    if verbose:
        print(f"Attention Weights | Shape {attn_weights.shape}:\n{attn_weights}\n")

    # --- 6. Apply to V ---
    if verbose:
        print("-" * 30 + "\nSTEP 6: Apply Attention Weights to V (Context per head)")
    context_heads = attn_weights @ V_heads
    if verbose:
        print(f"Context per head | Shape {context_heads.shape}:\n{context_heads}\n")

    # --- 7. Concatenate Heads ---
    if verbose:
        print("-" * 30 + "\nSTEP 7: Concatenate Heads")
    context_transposed = context_heads.transpose(1, 2).contiguous()
    context_concat = context_transposed.view(B, T, D)
    if verbose:
        print(f"Context Transposed | Shape {context_transposed.shape}:\n{context_transposed}")
        print(f"Context Concatenated | Shape {context_concat.shape}:\n{context_concat}\n")

    # --- 8. Output Projection ---
    if verbose:
        print("-" * 30 + "\nSTEP 8: Final Output Projection (Context @ W_o)")
    output = context_concat @ W_o
    if verbose:
        print(f"Final Output | Shape {output.shape}:\n{output}\n")
        print("=" * 50)
        print("🏁 ATTENTION BLOCK COMPLETE")
        print("=" * 50)

    return output


# ---------------------------------------------------------------------------
# Generation mode: called by the CUDA test to produce ground-truth output.
#
# Usage:
#   python python_get_attention_output.py --generate \
#       --d_model 64 --num_heads 2 --batch_size 8 --seq_len 2 --seed 42
#
# Output (to stdout, parseable by the C++ test):
#   LINE 1:  <python_time_ms>
#   LINE 2:  <float> <float> ...   (input,  B*T*D  floats)
#   LINE 3:  <float> <float> ...   (W_q,    D*D    floats, row-major)
#   LINE 4:  <float> <float> ...   (W_k,    D*D    floats)
#   LINE 5:  <float> <float> ...   (W_v,    D*D    floats)
#   LINE 6:  <float> <float> ...   (W_o,    D*D    floats)
#   LINE 7:  <float> <float> ...   (expected output, B*T*D floats)
# ---------------------------------------------------------------------------

def _floats_to_line(tensor: torch.Tensor) -> str:
    """Flatten tensor and return space-separated string of floats."""
    flat = tensor.detach().cpu().contiguous().view(-1).tolist()
    return " ".join(f"{v:.8g}" for v in flat)


def generate_mode(d_model: int, num_heads: int, batch_size: int, seq_len: int, seed: int):
    """Generate random inputs/weights, compute attention, print results."""
    torch.manual_seed(seed)

    with torch.no_grad():
        # Random inputs and weights (small magnitude to keep floats well-behaved)
        x   = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32)
        W_q = torch.randn(d_model, d_model, dtype=torch.float32) * 0.1
        W_k = torch.randn(d_model, d_model, dtype=torch.float32) * 0.1
        W_v = torch.randn(d_model, d_model, dtype=torch.float32) * 0.1
        W_o = torch.randn(d_model, d_model, dtype=torch.float32) * 0.1

        # Time the Python computation
        start = time.perf_counter()
        output = torch_mha(x, W_q, W_k, W_v, W_o, num_heads, verbose=False)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        # Print results (one item per line, easy to parse in C++)
        print(f"{elapsed_ms:.6f}")           # line 1 – python time in ms
        print(_floats_to_line(x))            # line 2 – input
        print(_floats_to_line(W_q))          # line 3 – W_q
        print(_floats_to_line(W_k))          # line 4 – W_k
        print(_floats_to_line(W_v))          # line 5 – W_v
        print(_floats_to_line(W_o))          # line 6 – W_o
        print(_floats_to_line(output))       # line 7 – expected output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch multi-head attention reference")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable step-by-step logging (interactive mode)")
    parser.add_argument("--generate", action="store_true",
                        help="Generate random test data and expected output for CUDA test")

    # Parameters for --generate mode
    parser.add_argument("--d_model",    type=int, default=64)
    parser.add_argument("--num_heads",  type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len",    type=int, default=2)
    parser.add_argument("--seed",       type=int, default=42)

    args = parser.parse_args()

    if args.generate:
        # Machine-readable output for the CUDA test
        generate_mode(args.d_model, args.num_heads, args.batch_size, args.seq_len, args.seed)
    else:
        # Interactive / verbose mode (legacy behaviour)
        if args.verbose:
            torch.set_printoptions(profile="full")

        with torch.no_grad():
            H = 2    # Number of heads
            D = 64   # d_model
            B = 8    # Batch
            T = 2    # Sequence Length

            x   = torch.linspace(1.0, float(B * T * D), steps=B * T * D, dtype=torch.float32).view(B, T, D)
            W_q = torch.linspace(1.0, float(D * D), steps=D * D, dtype=torch.float32).view(D, D)
            W_k = torch.linspace(1.0, float(D * D), steps=D * D, dtype=torch.float32).view(D, D)
            W_v = torch.linspace(1.0, float(D * D), steps=D * D, dtype=torch.float32).view(D, D)
            W_o = torch.linspace(1.0, float(D * D), steps=D * D, dtype=torch.float32).view(D, D)

            start = time.perf_counter()
            final_out = torch_mha(x, W_q, W_k, W_v, W_o, H, verbose=args.verbose)
            elapsed_ms = (time.perf_counter() - start) * 1000
            print(f"⏱️  Time taken: {elapsed_ms:.2f} ms")
