import torch
import torch.nn.functional as F
import math
import time

def torch_mha_extreme_logging(x, W_q, W_k, W_v, W_o, num_heads, verbose=False):
    B, T, D = x.shape
    head_dim = D // num_heads

    if verbose:
        print("="*50)
        print("🚀 STARTING ATTENTION BLOCK (EXTREME LOGGING)")
        print("="*50)
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
        print("="*50)
        print("🏁 ATTENTION BLOCK COMPLETE")
        print("="*50)

    return output

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable step-by-step logging")
    args = parser.parse_args()

    if args.verbose:
        torch.set_printoptions(profile="full")

    with torch.no_grad():
        H = 2   # Number of heads
        D = 64  # d_model
        B = 8   # Batch
        T = 2   # Sequence Length

        # ---------------------------------------------------------
        # Deterministic Matrix Generation (Acts exactly like hardcoding)
        # ---------------------------------------------------------

        # Input (B=1, T=2, D=32) -> 64 elements
        x = torch.linspace(1.0, float(B*T*D), steps=B*T*D, dtype=torch.float32).view(B, T, D)

        # Weight Matrices (D=32, D=32) -> 1024 elements each
        W_q = torch.linspace(1.0, float(D*D), steps=D*D, dtype=torch.float32).view(D, D)
        W_k = torch.linspace(1.0, float(D*D), steps=D*D, dtype=torch.float32).view(D, D)
        W_v = torch.linspace(1.0, float(D*D), steps=D*D, dtype=torch.float32).view(D, D)

        # For W_o, creating a true Identity matrix (pass-through)
        # so the output directly reflects the concatenated head contexts.
        W_o = torch.linspace(1.0, float(D*D), steps=D*D, dtype=torch.float32).view(D, D)

        # Run
        start = time.perf_counter()
        final_out = torch_mha_extreme_logging(x, W_q, W_k, W_v, W_o, H, verbose=args.verbose)
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"⏱️  Time taken: {elapsed_ms:.2f} ms")

