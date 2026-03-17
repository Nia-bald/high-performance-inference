import torch
import torch.nn.functional as F
import time
import argparse


def torch_feed_forward(x, W_up, b_up, W_down, b_down):
    """
    Feed-forward network reference implementation.

    Steps (matching the CUDA FeedForward::forward):
      1. hidden = x @ W_up           (Up Projection)
      2. hidden = GELU(hidden + b_up) (Fused Bias + GELU)
      3. output = hidden @ W_down     (Down Projection)
      4. output = output + b_down     (Final Bias Add)

    Args:
        x:      (B, T, d_model)
        W_up:   (d_model, d_ff)
        b_up:   (d_ff,)
        W_down: (d_ff, d_model)
        b_down: (d_model,)

    Returns:
        output: (B, T, d_model)
    """
    # Step 1: Up projection
    hidden = x @ W_up

    # Step 2: Bias + GELU
    hidden = F.gelu(hidden + b_up)

    # Step 3: Down projection
    output = hidden @ W_down

    # Step 4: Final bias
    output = output + b_down

    return output


def _floats_to_line(tensor: torch.Tensor) -> str:
    """Flatten tensor and return space-separated string of floats."""
    flat = tensor.detach().cpu().contiguous().view(-1).tolist()
    return " ".join(f"{v:.8g}" for v in flat)


def generate_mode(d_model: int, d_ff: int, batch_size: int, seq_len: int, seed: int):
    """Generate random inputs/weights, compute FFN, print results."""
    torch.manual_seed(seed)

    with torch.no_grad():
        # Random inputs and weights (small magnitude to keep floats well-behaved)
        x      = torch.randn(batch_size, seq_len, d_model, dtype=torch.float32)
        W_up   = torch.randn(d_model, d_ff, dtype=torch.float32) * 0.1
        b_up   = torch.randn(d_ff, dtype=torch.float32) * 0.1
        W_down = torch.randn(d_ff, d_model, dtype=torch.float32) * 0.1
        b_down = torch.randn(d_model, dtype=torch.float32) * 0.1

        # Time the Python computation
        start = time.perf_counter()
        output = torch_feed_forward(x, W_up, b_up, W_down, b_down)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        # Print results (one item per line, easy to parse in C++)
        # LINE 1: python time in ms
        # LINE 2: input        (B*T*d_model floats)
        # LINE 3: W_up         (d_model*d_ff floats, row-major)
        # LINE 4: b_up         (d_ff floats)
        # LINE 5: W_down       (d_ff*d_model floats, row-major)
        # LINE 6: b_down       (d_model floats)
        # LINE 7: expected     (B*T*d_model floats)
        print(f"{elapsed_ms:.6f}")
        print(_floats_to_line(x))
        print(_floats_to_line(W_up))
        print(_floats_to_line(b_up))
        print(_floats_to_line(W_down))
        print(_floats_to_line(b_down))
        print(_floats_to_line(output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch feed-forward reference")
    parser.add_argument("--generate", action="store_true",
                        help="Generate random test data and expected output for CUDA test")

    # Parameters for --generate mode
    parser.add_argument("--d_model",    type=int, default=768)
    parser.add_argument("--d_ff",       type=int, default=3072)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len",    type=int, default=16)
    parser.add_argument("--seed",       type=int, default=42)

    args = parser.parse_args()

    if args.generate:
        generate_mode(args.d_model, args.d_ff, args.batch_size, args.seq_len, args.seed)
    else:
        # Quick interactive demo
        with torch.no_grad():
            D, F_DIM, B, T = 768, 3072, 2, 4
            x      = torch.randn(B, T, D)
            W_up   = torch.randn(D, F_DIM) * 0.1
            b_up   = torch.randn(F_DIM) * 0.1
            W_down = torch.randn(F_DIM, D) * 0.1
            b_down = torch.randn(D) * 0.1

            start = time.perf_counter()
            out = torch_feed_forward(x, W_up, b_up, W_down, b_down)
            elapsed_ms = (time.perf_counter() - start) * 1000
            print(f"Output shape: {out.shape}")
            print(f"⏱️  Time taken: {elapsed_ms:.2f} ms")
