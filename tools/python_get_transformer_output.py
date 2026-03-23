import torch
import torch.nn.functional as F
import math
import time
import argparse


def layer_norm(x, gamma, beta, eps=1e-5):
    """LayerNorm matching the CUDA kernel (hardcoded epsilon of 1e-5)."""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return gamma * (x - mean) / torch.sqrt(var + eps) + beta


def self_attention(x, W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o, num_heads):
    """
    Multi-head causal self-attention (matching the CUDA SelfAttention::forward).
    
    x:      (B, T, D)
    W_q/k/v/o:  (D, D)
    b_q/k/v/o:  (D,)
    """
    B, T, D = x.shape
    head_dim = D // num_heads

    Q = (x @ W_q) + b_q   # (B, T, D)
    K = (x @ W_k) + b_k
    V = (x @ W_v) + b_v

    # Reshape to (B, H, T, head_dim)
    Q = Q.view(B, T, num_heads, head_dim).transpose(1, 2)
    K = K.view(B, T, num_heads, head_dim).transpose(1, 2)
    V = V.view(B, T, num_heads, head_dim).transpose(1, 2)

    # Scaled dot-product attention
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(head_dim)

    # Causal mask
    mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
    scores.masked_fill_(mask, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)
    context = attn_weights @ V  # (B, H, T, head_dim)

    # Concat heads
    context = context.transpose(1, 2).contiguous().view(B, T, D)

    # Output projection
    return (context @ W_o) + b_o


def feed_forward(x, W_up, b_up, W_down, b_down):
    """FFN matching the CUDA FeedForward::forward (GELU activation)."""
    hidden = x @ W_up
    hidden = F.gelu(hidden + b_up)
    output = hidden @ W_down
    return output + b_down


def transformer_block(x, attn_norm_gamma, attn_norm_beta,
                      W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o,
                      ffn_norm_gamma, ffn_norm_beta,
                      W_up, b_up, W_down, b_down,
                      num_heads):
    """
    Pre-norm transformer block matching TransformerBlock::forward.
    
    1. residual_1 = x + Attention(LayerNorm(x))
    2. output     = residual_1 + FFN(LayerNorm(residual_1))
    """
    # Attention path
    norm1 = layer_norm(x, attn_norm_gamma, attn_norm_beta)
    attn_out = self_attention(norm1, W_q, b_q, W_k, b_k, W_v, b_v, W_o, b_o, num_heads)
    res1 = x + attn_out

    # FFN path
    norm2 = layer_norm(res1, ffn_norm_gamma, ffn_norm_beta)
    ffn_out = feed_forward(norm2, W_up, b_up, W_down, b_down)
    return res1 + ffn_out


def transformer_forward(token_ids, token_embed, pos_embed,
                        layer_weights, final_norm_gamma, final_norm_beta,
                        lm_head, num_heads):
    """
    Full transformer forward pass matching Transformer::forward.
    
    token_ids:   (B, T) integer tensor
    token_embed: (V, D)
    pos_embed:   (MaxSeq, D)
    layer_weights: list of dicts, one per layer
    final_norm_gamma/beta: (D,)
    lm_head:     (D, V)
    
    Returns logits: (B, T, V)
    """
    B, T = token_ids.shape
    D = token_embed.shape[1]

    # 1. Embedding lookup: token + positional
    x = token_embed[token_ids] + pos_embed[:T].unsqueeze(0)  # (B, T, D)

    # 2. Transformer blocks
    for lw in layer_weights:
        x = transformer_block(
            x,
            lw['attn_norm_gamma'], lw['attn_norm_beta'],
            lw['W_q'], lw['b_q'], lw['W_k'], lw['b_k'], lw['W_v'], lw['b_v'], lw['W_o'], lw['b_o'],
            lw['ffn_norm_gamma'], lw['ffn_norm_beta'],
            lw['W_up'], lw['b_up'], lw['W_down'], lw['b_down'],
            num_heads
        )

    # 3. Final LayerNorm
    x = layer_norm(x, final_norm_gamma, final_norm_beta)

    # 4. LM Head (linear projection, no bias)
    # x: (B*T, D) @ lm_head: (D, V) -> (B*T, V)
    logits = x.view(B * T, D) @ lm_head
    logits = logits.view(B, T, -1)

    return logits


def _floats_to_line(tensor: torch.Tensor) -> str:
    """Flatten tensor and return space-separated string of floats."""
    flat = tensor.detach().cpu().contiguous().view(-1).tolist()
    return " ".join(f"{v:.8g}" for v in flat)


def _ints_to_line(tensor: torch.Tensor) -> str:
    """Flatten integer tensor and return space-separated string."""
    flat = tensor.detach().cpu().contiguous().view(-1).tolist()
    return " ".join(str(int(v)) for v in flat)


def generate_mode(vocab_size, max_seq_len, d_model, num_heads, num_layers, d_ff,
                  batch_size, seq_len, seed):
    """Generate random inputs/weights, compute full transformer, print results."""
    torch.manual_seed(seed)

    with torch.no_grad():
        # Random token IDs
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Embedding tables (small magnitude)
        token_embed = torch.randn(vocab_size, d_model, dtype=torch.float32) * 0.1
        pos_embed = torch.randn(max_seq_len, d_model, dtype=torch.float32) * 0.1

        # Per-layer weights
        layer_weights = []
        for _ in range(num_layers):
            lw = {}
            # Attention norm
            lw['attn_norm_gamma'] = torch.ones(d_model, dtype=torch.float32)
            lw['attn_norm_beta'] = torch.zeros(d_model, dtype=torch.float32)
            # Attention projections
            lw['W_q'] = torch.randn(d_model, d_model, dtype=torch.float32) * 0.1
            lw['b_q'] = torch.randn(d_model, dtype=torch.float32) * 0.1
            lw['W_k'] = torch.randn(d_model, d_model, dtype=torch.float32) * 0.1
            lw['b_k'] = torch.randn(d_model, dtype=torch.float32) * 0.1
            lw['W_v'] = torch.randn(d_model, d_model, dtype=torch.float32) * 0.1
            lw['b_v'] = torch.randn(d_model, dtype=torch.float32) * 0.1
            lw['W_o'] = torch.randn(d_model, d_model, dtype=torch.float32) * 0.1
            lw['b_o'] = torch.randn(d_model, dtype=torch.float32) * 0.1
            # FFN norm
            lw['ffn_norm_gamma'] = torch.ones(d_model, dtype=torch.float32)
            lw['ffn_norm_beta'] = torch.zeros(d_model, dtype=torch.float32)
            # FFN weights
            lw['W_up'] = torch.randn(d_model, d_ff, dtype=torch.float32) * 0.1
            lw['b_up'] = torch.randn(d_ff, dtype=torch.float32) * 0.1
            lw['W_down'] = torch.randn(d_ff, d_model, dtype=torch.float32) * 0.1
            lw['b_down'] = torch.randn(d_model, dtype=torch.float32) * 0.1
            layer_weights.append(lw)

        # Final norm
        final_norm_gamma = torch.ones(d_model, dtype=torch.float32)
        final_norm_beta = torch.zeros(d_model, dtype=torch.float32)

        # LM Head
        lm_head = torch.randn(d_model, vocab_size, dtype=torch.float32) * 0.1

        # Run forward pass
        start = time.perf_counter()
        logits = transformer_forward(
            token_ids, token_embed, pos_embed,
            layer_weights, final_norm_gamma, final_norm_beta,
            lm_head, num_heads
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        # --- Output (one item per line, parsed by C++ test) ---
        # LINE 1: python time
        print(f"{elapsed_ms:.6f}")

        # LINE 2: token_ids (integers)
        print(_ints_to_line(token_ids))

        # LINE 3: token_embedding_table [Vocab, d_model]
        print(_floats_to_line(token_embed))

        # LINE 4: pos_embedding_table [MaxSeq, d_model]
        print(_floats_to_line(pos_embed))

        # Per layer (16 lines each):
        for lw in layer_weights:
            print(_floats_to_line(lw['attn_norm_gamma']))   # attn_norm gamma
            print(_floats_to_line(lw['attn_norm_beta']))    # attn_norm beta
            print(_floats_to_line(lw['W_q']))               # W_q
            print(_floats_to_line(lw['b_q']))               # b_q
            print(_floats_to_line(lw['W_k']))               # W_k
            print(_floats_to_line(lw['b_k']))               # b_k
            print(_floats_to_line(lw['W_v']))               # W_v
            print(_floats_to_line(lw['b_v']))               # b_v
            print(_floats_to_line(lw['W_o']))               # W_o
            print(_floats_to_line(lw['b_o']))               # b_o
            print(_floats_to_line(lw['ffn_norm_gamma']))    # ffn_norm gamma
            print(_floats_to_line(lw['ffn_norm_beta']))     # ffn_norm beta
            print(_floats_to_line(lw['W_up']))              # W_up
            print(_floats_to_line(lw['b_up']))              # b_up
            print(_floats_to_line(lw['W_down']))            # W_down
            print(_floats_to_line(lw['b_down']))            # b_down

        # Final norm weights
        print(_floats_to_line(final_norm_gamma))
        print(_floats_to_line(final_norm_beta))

        # LM Head
        print(_floats_to_line(lm_head))

        # Expected logits
        print(_floats_to_line(logits))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch transformer reference")
    parser.add_argument("--generate", action="store_true",
                        help="Generate random test data and expected output for CUDA test")

    parser.add_argument("--vocab_size",   type=int, default=128)
    parser.add_argument("--max_seq_len",  type=int, default=16)
    parser.add_argument("--d_model",      type=int, default=64)
    parser.add_argument("--num_heads",    type=int, default=2)
    parser.add_argument("--num_layers",   type=int, default=2)
    parser.add_argument("--d_ff",         type=int, default=256)
    parser.add_argument("--batch_size",   type=int, default=2)
    parser.add_argument("--seq_len",      type=int, default=4)
    parser.add_argument("--seed",         type=int, default=42)

    args = parser.parse_args()

    if args.generate:
        generate_mode(args.vocab_size, args.max_seq_len, args.d_model, args.num_heads,
                      args.num_layers, args.d_ff, args.batch_size, args.seq_len, args.seed)
    else:
        # Quick interactive demo
        with torch.no_grad():
            V, MS, D, H, L, FF, B, T = 128, 16, 64, 2, 2, 256, 2, 4
            token_ids = torch.randint(0, V, (B, T))
            token_embed = torch.randn(V, D) * 0.1
            pos_embed = torch.randn(MS, D) * 0.1
            layer_weights = []
            for _ in range(L):
                lw = {
                    'attn_norm_gamma': torch.ones(D), 'attn_norm_beta': torch.zeros(D),
                    'W_q': torch.randn(D, D) * 0.1, 'b_q': torch.randn(D) * 0.1,
                    'W_k': torch.randn(D, D) * 0.1, 'b_k': torch.randn(D) * 0.1,
                    'W_v': torch.randn(D, D) * 0.1, 'b_v': torch.randn(D) * 0.1,
                    'W_o': torch.randn(D, D) * 0.1, 'b_o': torch.randn(D) * 0.1,
                    'ffn_norm_gamma': torch.ones(D), 'ffn_norm_beta': torch.zeros(D),
                    'W_up': torch.randn(D, FF) * 0.1, 'b_up': torch.randn(FF) * 0.1,
                    'W_down': torch.randn(FF, D) * 0.1, 'b_down': torch.randn(D) * 0.1,
                }
                layer_weights.append(lw)
            fn_g = torch.ones(D)
            fn_b = torch.zeros(D)
            head = torch.randn(D, V) * 0.1

            start = time.perf_counter()
            logits = transformer_forward(token_ids, token_embed, pos_embed,
                                         layer_weights, fn_g, fn_b, head, H)
            elapsed_ms = (time.perf_counter() - start) * 1000
            print(f"Output shape: {logits.shape}")
            print(f"⏱️  Time taken: {elapsed_ms:.2f} ms")
