"""
Test B=2, D=64, H=2, T=2 — the failing case.
Verify at which stage the mismatch appears.
"""
import torch, torch.nn.functional as F, math

torch.manual_seed(42)
torch.set_printoptions(precision=6, linewidth=200)

B, T, D, H = 2, 2, 64, 2
hd = D // H

with torch.no_grad():
    x   = torch.randn(B, T, D)
    W_q = torch.randn(D, D) * 0.1
    W_k = torch.randn(D, D) * 0.1
    W_v = torch.randn(D, D) * 0.1
    W_o = torch.randn(D, D) * 0.1

    Q = (x @ W_q).view(B*T, D)
    K = (x @ W_k).view(B*T, D)
    V = (x @ W_v).view(B*T, D)

    K_global_T = K.T.contiguous()

    # batched_gemm: Q[B*T, D] x K_T[D, B*T] -> C[B*T, H*T]
    sA, sB, sK = T, T, hd  # 2, 2, 32
    M, N, Kv = B*T, B*T, D  # 4, 4, 64
    total_cols = sB * (Kv // sK)  # 2 * 2 = 4

    C = torch.zeros(M, total_cols)
    for r in range(M):
        for c in range(total_cols):
            Bi = r // sA; Bj = c // sB
            v = 0.0
            for k in range(sK):
                a_val = Q[r, Bj*sK + k].item()
                b_row = Bj*sK + k
                b_col = sB*(Bi - Bj) + c
                b_val = K_global_T[b_row, b_col].item() if 0 <= b_col < N else 0.0
                v += a_val * b_val
            C[r, c] = v

    # PyTorch reference
    Q_h = Q.view(B, T, H, hd).transpose(1, 2)
    K_h = K.view(B, T, H, hd).transpose(1, 2)
    V_h = V.view(B, T, H, hd).transpose(1, 2)
    scale = math.sqrt(hd)

    pt_raw_scores = Q_h @ K_h.transpose(-2, -1)

    # Compare scores
    print("=== ATTENTION SCORES COMPARISON ===")
    all_ok = True
    for r in range(M):
        bi = r // T; si = r % T
        for c in range(total_cols):
            h = c // T; sj = c % T
            cv = C[r, c].item()
            pv = pt_raw_scores[bi, h, si, sj].item()
            ok = abs(cv - pv) < 1e-3
            if not ok:
                print(f"  ✗ C[{r},{c}] b={bi},h={h},i={si},j={sj}: CUDA={cv:.6f} PT={pv:.6f} diff={abs(cv-pv):.6f}")
                all_ok = False
    if all_ok: print("  All 16 scores match ✓")

    # Scale + Mask
    C_scaled = C / scale
    rows_m, cols_m, sr, sc = B*T, H*T, T, T
    C_masked = C_scaled.clone()
    for r in range(rows_m):
        for c in range(cols_m):
            LHS = r + (c // sc) * sc
            RHS = c + (r // sr) * sr
            if LHS < RHS:
                C_masked.view(-1)[r * cols_m + c] = float('-inf')

    pt_scores = pt_raw_scores / scale
    causal = torch.triu(torch.ones(T, T), diagonal=1).bool()
    pt_scores.masked_fill_(causal, float('-inf'))

    # Compare mask
    print("\n=== MASK COMPARISON ===")
    all_ok = True
    for r in range(rows_m):
        bi = r // T; si = r % T
        for c in range(cols_m):
            h = c // T; sj = c % T
            cv = C_masked[r, c].item()
            pv = pt_scores[bi, h, si, sj].item()
            ok = (abs(cv - pv) < 1e-3) or (cv == float('-inf') and pv == float('-inf'))
            if not ok:
                print(f"  ✗ mask[{r},{c}] b={bi},h={h},i={si},j={sj}: CUDA={cv:.6f} PT={pv:.6f}")
                all_ok = False
    if all_ok: print("  All match ✓")

    # Softmax
    num_sm_rows = H * B * T
    flat = C_masked.view(-1)
    sm_flat = torch.zeros(num_sm_rows * T)
    for r in range(num_sm_rows):
        row = flat[r*T:(r+1)*T].clone()
        sm = F.softmax(row, dim=0)
        sm_flat[r*T:(r+1)*T] = sm

    pt_attn = F.softmax(pt_scores, dim=-1)

    print("\n=== SOFTMAX COMPARISON ===")
    all_ok = True
    for r in range(M):
        bi = r // T; si = r % T
        for c in range(total_cols):
            h = c // T; sj = c % T
            cv = sm_flat[r * total_cols + c].item()
            pv = pt_attn[bi, h, si, sj].item()
            ok = abs(cv - pv) < 1e-3
            if not ok:
                print(f"  ✗ sm[{r},{c}] b={bi},h={h},i={si},j={sj}: CUDA={cv:.6f} PT={pv:.6f}")
                all_ok = False
    if all_ok: print("  All match ✓")

    # A*V
    A_2d = sm_flat.view(B*T, H*T)
    sA2, sB2, sK2 = T, hd, T
    M2, N2, K2 = B*T, D, H*T

    AV = torch.zeros(M2, N2)
    for r in range(M2):
        for c in range(N2):
            Bi = r // sA2; Bj = c // sB2
            v = 0.0
            for k in range(sK2):
                a = A_2d.view(-1)[r * K2 + Bj * sK2 + k].item()
                br = Bi * sK2 + k
                bc = c
                b = V[br, bc].item() if br < B*T else 0.0
                v += a * b
            AV[r, c] = v

    pt_ctx = (pt_attn @ V_h).transpose(1, 2).contiguous().view(B, T, D)

    print("\n=== A*V COMPARISON ===")
    all_ok = True
    mismatches = 0
    for i in range(AV.numel()):
        cv = AV.view(-1)[i].item()
        pv = pt_ctx.view(-1)[i].item()
        ok = abs(cv - pv) < 1e-3
        if not ok:
            if mismatches < 10:
                print(f"  ✗ AV[{i}] (b={i//(T*D)},s={(i%(T*D))//D},d={i%D}): CUDA={cv:.6f} PT={pv:.6f} diff={abs(cv-pv):.6f}")
            mismatches += 1
            all_ok = False
    if all_ok:
        print("  All match ✓")
    else:
        print(f"  Total A*V mismatches: {mismatches}/{AV.numel()}")

    # Final output
    cuda_out = (AV.view(B, T, D) @ W_o).view(-1)
    pt_out = (pt_ctx @ W_o).view(-1)

    print("\n=== FINAL OUTPUT COMPARISON ===")
    all_ok = True
    mismatches = 0
    for i in range(cuda_out.numel()):
        cv = cuda_out[i].item()
        pv = pt_out[i].item()
        ok = abs(cv - pv) < 1e-3
        if not ok:
            if mismatches < 10:
                print(f"  ✗ out[{i}]: CUDA={cv:.6f} PT={pv:.6f} diff={abs(cv-pv):.6f}")
            mismatches += 1
            all_ok = False
    if all_ok:
        print("  All match ✓")
    else:
        print(f"  Total output mismatches: {mismatches}/{cuda_out.numel()}")
