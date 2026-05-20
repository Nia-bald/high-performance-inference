#include "kernels.cuh"
#include <cuda_fp16.h>
#include <cstdio>

// ============================================================
// FP16-Weight GEMV: y[N] = x[K] * W_fp16[K, N] (+ bias)
// ============================================================
//
// Strategy: FP16 storage, FP32 compute.
// GTX 1050 Ti (SM 6.1) has no FP16 compute advantage, but
// FP16 weight storage HALVES the memory bandwidth required.
//
// GEMV is purely memory-bound (~0.25 arithmetic intensity).
// Halving weight reads ≈ 2× throughput improvement.
//
// KEY DESIGN:
//   - Weights stored as __half, loaded via half2 (32-bit loads)
//   - Each half2 load reads 2 weights in one 32-bit transaction
//   - Dequantized to float immediately for FP32 accumulation
//   - Activations (x), biases, and outputs remain FP32
//   - float4 equivalent: 4 × half2 = 8 weights per 128-bit load
//
// ============================================================

namespace kernels {

// ============================================================
// Kernel 1: half2 vectorized, direct write (large N)
// ============================================================
// Each thread processes 8 output columns (via 4 × half2 loads).
// This is the FP16 equivalent of the FP32 float4 kernel.
// For N such that N/8/256 ≥ 24 blocks.

#define GEMV_FP16_THREADS 256
#define GEMV_FP16_COLS_PER_THREAD 8   // 4 × half2 = 8 half values

__global__ void __launch_bounds__(GEMV_FP16_THREADS)
gemv_fp16w_vec(
    const float* __restrict__ x,       // [K] — FP32 activations
    const __half* __restrict__ W,      // [K, N] — FP16 weights
    float* __restrict__ y,             // [N] — FP32 output
    const float* __restrict__ bias,    // [N] or nullptr — FP32
    int N, int K
) {
    // Each thread handles 8 consecutive output columns
    const int col_base = (blockIdx.x * GEMV_FP16_THREADS + threadIdx.x) * GEMV_FP16_COLS_PER_THREAD;
    if (col_base >= N) return;

    // Accumulate in FP32
    float sum[GEMV_FP16_COLS_PER_THREAD] = {0.0f};

    // Cast weight pointer for half2 loads (4 half2 = 8 halfs = 16 bytes per thread per K)
    const half2* W2 = reinterpret_cast<const half2*>(W);
    const int N2 = N >> 1;  // number of half2 elements per row

    // Half2 index base for this thread
    const int h2_base = col_base >> 1;  // col_base / 2

    // 4× K-unroll for ILP
    int k = 0;
    for (; k + 3 < K; k += 4) {
        float x0 = x[k], x1 = x[k+1], x2 = x[k+2], x3 = x[k+3];

        // Load 4 half2 per K-step = 8 weights
        half2 w0_a = W2[(k)   * N2 + h2_base];
        half2 w0_b = W2[(k)   * N2 + h2_base + 1];
        half2 w0_c = W2[(k)   * N2 + h2_base + 2];
        half2 w0_d = W2[(k)   * N2 + h2_base + 3];

        half2 w1_a = W2[(k+1) * N2 + h2_base];
        half2 w1_b = W2[(k+1) * N2 + h2_base + 1];
        half2 w1_c = W2[(k+1) * N2 + h2_base + 2];
        half2 w1_d = W2[(k+1) * N2 + h2_base + 3];

        half2 w2_a = W2[(k+2) * N2 + h2_base];
        half2 w2_b = W2[(k+2) * N2 + h2_base + 1];
        half2 w2_c = W2[(k+2) * N2 + h2_base + 2];
        half2 w2_d = W2[(k+2) * N2 + h2_base + 3];

        half2 w3_a = W2[(k+3) * N2 + h2_base];
        half2 w3_b = W2[(k+3) * N2 + h2_base + 1];
        half2 w3_c = W2[(k+3) * N2 + h2_base + 2];
        half2 w3_d = W2[(k+3) * N2 + h2_base + 3];

        // Dequantize and accumulate
        sum[0] += x0 * __half2float(w0_a.x) + x1 * __half2float(w1_a.x) + x2 * __half2float(w2_a.x) + x3 * __half2float(w3_a.x);
        sum[1] += x0 * __half2float(w0_a.y) + x1 * __half2float(w1_a.y) + x2 * __half2float(w2_a.y) + x3 * __half2float(w3_a.y);
        sum[2] += x0 * __half2float(w0_b.x) + x1 * __half2float(w1_b.x) + x2 * __half2float(w2_b.x) + x3 * __half2float(w3_b.x);
        sum[3] += x0 * __half2float(w0_b.y) + x1 * __half2float(w1_b.y) + x2 * __half2float(w2_b.y) + x3 * __half2float(w3_b.y);
        sum[4] += x0 * __half2float(w0_c.x) + x1 * __half2float(w1_c.x) + x2 * __half2float(w2_c.x) + x3 * __half2float(w3_c.x);
        sum[5] += x0 * __half2float(w0_c.y) + x1 * __half2float(w1_c.y) + x2 * __half2float(w2_c.y) + x3 * __half2float(w3_c.y);
        sum[6] += x0 * __half2float(w0_d.x) + x1 * __half2float(w1_d.x) + x2 * __half2float(w2_d.x) + x3 * __half2float(w3_d.x);
        sum[7] += x0 * __half2float(w0_d.y) + x1 * __half2float(w1_d.y) + x2 * __half2float(w2_d.y) + x3 * __half2float(w3_d.y);
    }
    // Remainder K
    for (; k < K; ++k) {
        float xk = x[k];
        half2 wa = W2[k * N2 + h2_base];
        half2 wb = W2[k * N2 + h2_base + 1];
        half2 wc = W2[k * N2 + h2_base + 2];
        half2 wd = W2[k * N2 + h2_base + 3];
        sum[0] += xk * __half2float(wa.x);
        sum[1] += xk * __half2float(wa.y);
        sum[2] += xk * __half2float(wb.x);
        sum[3] += xk * __half2float(wb.y);
        sum[4] += xk * __half2float(wc.x);
        sum[5] += xk * __half2float(wc.y);
        sum[6] += xk * __half2float(wd.x);
        sum[7] += xk * __half2float(wd.y);
    }

    // Add bias and store
    if (bias != nullptr) {
        #pragma unroll
        for (int i = 0; i < GEMV_FP16_COLS_PER_THREAD; ++i) {
            if (col_base + i < N)
                y[col_base + i] = sum[i] + bias[col_base + i];
        }
    } else {
        #pragma unroll
        for (int i = 0; i < GEMV_FP16_COLS_PER_THREAD; ++i) {
            if (col_base + i < N)
                y[col_base + i] = sum[i];
        }
    }
}


// ============================================================
// Kernel 2: K-split half2 with atomicAdd (medium N)
// ============================================================
__global__ void __launch_bounds__(GEMV_FP16_THREADS)
gemv_fp16w_ksplit_vec(
    const float* __restrict__ x,
    const __half* __restrict__ W,
    float* __restrict__ y,
    const float* __restrict__ bias,
    int N, int K,
    int k_splits
) {
    const int col_base = (blockIdx.x * GEMV_FP16_THREADS + threadIdx.x) * GEMV_FP16_COLS_PER_THREAD;
    if (col_base >= N) return;

    const int split_id = blockIdx.y;
    const int stripe = (K + k_splits - 1) / k_splits;
    const int k_start = split_id * stripe;
    const int k_end = min(k_start + stripe, K);

    float sum[GEMV_FP16_COLS_PER_THREAD] = {0.0f};

    const half2* W2 = reinterpret_cast<const half2*>(W);
    const int N2 = N >> 1;
    const int h2_base = col_base >> 1;

    int k = k_start;
    for (; k + 3 < k_end; k += 4) {
        float x0 = x[k], x1 = x[k+1], x2 = x[k+2], x3 = x[k+3];

        half2 w0_a = W2[(k)   * N2 + h2_base];
        half2 w0_b = W2[(k)   * N2 + h2_base + 1];
        half2 w0_c = W2[(k)   * N2 + h2_base + 2];
        half2 w0_d = W2[(k)   * N2 + h2_base + 3];

        half2 w1_a = W2[(k+1) * N2 + h2_base];
        half2 w1_b = W2[(k+1) * N2 + h2_base + 1];
        half2 w1_c = W2[(k+1) * N2 + h2_base + 2];
        half2 w1_d = W2[(k+1) * N2 + h2_base + 3];

        half2 w2_a = W2[(k+2) * N2 + h2_base];
        half2 w2_b = W2[(k+2) * N2 + h2_base + 1];
        half2 w2_c = W2[(k+2) * N2 + h2_base + 2];
        half2 w2_d = W2[(k+2) * N2 + h2_base + 3];

        half2 w3_a = W2[(k+3) * N2 + h2_base];
        half2 w3_b = W2[(k+3) * N2 + h2_base + 1];
        half2 w3_c = W2[(k+3) * N2 + h2_base + 2];
        half2 w3_d = W2[(k+3) * N2 + h2_base + 3];

        sum[0] += x0*__half2float(w0_a.x) + x1*__half2float(w1_a.x) + x2*__half2float(w2_a.x) + x3*__half2float(w3_a.x);
        sum[1] += x0*__half2float(w0_a.y) + x1*__half2float(w1_a.y) + x2*__half2float(w2_a.y) + x3*__half2float(w3_a.y);
        sum[2] += x0*__half2float(w0_b.x) + x1*__half2float(w1_b.x) + x2*__half2float(w2_b.x) + x3*__half2float(w3_b.x);
        sum[3] += x0*__half2float(w0_b.y) + x1*__half2float(w1_b.y) + x2*__half2float(w2_b.y) + x3*__half2float(w3_b.y);
        sum[4] += x0*__half2float(w0_c.x) + x1*__half2float(w1_c.x) + x2*__half2float(w2_c.x) + x3*__half2float(w3_c.x);
        sum[5] += x0*__half2float(w0_c.y) + x1*__half2float(w1_c.y) + x2*__half2float(w2_c.y) + x3*__half2float(w3_c.y);
        sum[6] += x0*__half2float(w0_d.x) + x1*__half2float(w1_d.x) + x2*__half2float(w2_d.x) + x3*__half2float(w3_d.x);
        sum[7] += x0*__half2float(w0_d.y) + x1*__half2float(w1_d.y) + x2*__half2float(w2_d.y) + x3*__half2float(w3_d.y);
    }
    for (; k < k_end; ++k) {
        float xk = x[k];
        half2 wa = W2[k * N2 + h2_base];
        half2 wb = W2[k * N2 + h2_base + 1];
        half2 wc = W2[k * N2 + h2_base + 2];
        half2 wd = W2[k * N2 + h2_base + 3];
        sum[0] += xk * __half2float(wa.x);
        sum[1] += xk * __half2float(wa.y);
        sum[2] += xk * __half2float(wb.x);
        sum[3] += xk * __half2float(wb.y);
        sum[4] += xk * __half2float(wc.x);
        sum[5] += xk * __half2float(wc.y);
        sum[6] += xk * __half2float(wd.x);
        sum[7] += xk * __half2float(wd.y);
    }

    // atomicAdd partial sums
    #pragma unroll
    for (int i = 0; i < GEMV_FP16_COLS_PER_THREAD; ++i) {
        if (col_base + i < N)
            atomicAdd(&y[col_base + i], sum[i]);
    }

    // Bias added by last split only
    if (bias != nullptr && split_id == k_splits - 1) {
        #pragma unroll
        for (int i = 0; i < GEMV_FP16_COLS_PER_THREAD; ++i) {
            if (col_base + i < N)
                atomicAdd(&y[col_base + i], bias[col_base + i]);
        }
    }
}


// ============================================================
// Kernel 3: Scalar FP16w direct (N not div by 8, large N)
// ============================================================
#define GEMV_FP16_SCALAR_THREADS 256

__global__ void __launch_bounds__(GEMV_FP16_SCALAR_THREADS)
gemv_fp16w_scalar(
    const float* __restrict__ x,
    const __half* __restrict__ W,
    float* __restrict__ y,
    const float* __restrict__ bias,
    int N, int K
) {
    const int col = blockIdx.x * GEMV_FP16_SCALAR_THREADS + threadIdx.x;
    if (col >= N) return;

    float sum = 0.0f;
    int k = 0;
    for (; k + 3 < K; k += 4) {
        sum += x[k]   * __half2float(W[(k)   * N + col])
             + x[k+1] * __half2float(W[(k+1) * N + col])
             + x[k+2] * __half2float(W[(k+2) * N + col])
             + x[k+3] * __half2float(W[(k+3) * N + col]);
    }
    for (; k < K; ++k) {
        sum += x[k] * __half2float(W[k * N + col]);
    }

    if (bias != nullptr) sum += bias[col];
    y[col] = sum;
}

// ============================================================
// Kernel 4: Scalar FP16w K-split with atomicAdd
// ============================================================
__global__ void __launch_bounds__(GEMV_FP16_SCALAR_THREADS)
gemv_fp16w_ksplit_scalar(
    const float* __restrict__ x,
    const __half* __restrict__ W,
    float* __restrict__ y,
    const float* __restrict__ bias,
    int N, int K,
    int k_splits
) {
    const int col = blockIdx.x * GEMV_FP16_SCALAR_THREADS + threadIdx.x;
    if (col >= N) return;

    const int split_id = blockIdx.y;
    const int stripe = (K + k_splits - 1) / k_splits;
    const int k_start = split_id * stripe;
    const int k_end = min(k_start + stripe, K);

    float sum = 0.0f;
    int k = k_start;
    for (; k + 3 < k_end; k += 4) {
        sum += x[k]   * __half2float(W[(k)   * N + col])
             + x[k+1] * __half2float(W[(k+1) * N + col])
             + x[k+2] * __half2float(W[(k+2) * N + col])
             + x[k+3] * __half2float(W[(k+3) * N + col]);
    }
    for (; k < k_end; ++k) {
        sum += x[k] * __half2float(W[k * N + col]);
    }

    atomicAdd(&y[col], sum);
    if (bias != nullptr && split_id == k_splits - 1) {
        atomicAdd(&y[col], bias[col]);
    }
}


// ============================================================
// Launch wrapper — shape-adaptive kernel selection
// ============================================================
void launch_gemv_fp16w(
    const float* x,
    const __half* W,
    float* y,
    int N, int K,
    const float* bias,
    cudaStream_t stream
) {
    const int TARGET_BLOCKS = 24;

    if (N % GEMV_FP16_COLS_PER_THREAD == 0) {
        // Vectorized path: 8 columns per thread via half2
        int n_groups = N / GEMV_FP16_COLS_PER_THREAD;
        int n_blocks = (n_groups + GEMV_FP16_THREADS - 1) / GEMV_FP16_THREADS;

        if (n_blocks >= TARGET_BLOCKS) {
            gemv_fp16w_vec<<<n_blocks, GEMV_FP16_THREADS, 0, stream>>>(
                x, W, y, bias, N, K);
        } else {
            int k_splits = (TARGET_BLOCKS + n_blocks - 1) / n_blocks;
            int max_splits = max(1, K / 32);
            k_splits = min(k_splits, max_splits);

            cudaMemsetAsync(y, 0, N * sizeof(float), stream);
            dim3 grid(n_blocks, k_splits);
            gemv_fp16w_ksplit_vec<<<grid, GEMV_FP16_THREADS, 0, stream>>>(
                x, W, y, bias, N, K, k_splits);
        }
    } else {
        // Scalar fallback: 1 column per thread (for N not divisible by 8)
        int n_blocks = (N + GEMV_FP16_SCALAR_THREADS - 1) / GEMV_FP16_SCALAR_THREADS;

        if (n_blocks >= TARGET_BLOCKS) {
            gemv_fp16w_scalar<<<n_blocks, GEMV_FP16_SCALAR_THREADS, 0, stream>>>(
                x, W, y, bias, N, K);
        } else {
            int k_splits = (TARGET_BLOCKS + n_blocks - 1) / n_blocks;
            int max_splits = max(1, K / 32);
            k_splits = min(k_splits, max_splits);

            cudaMemsetAsync(y, 0, N * sizeof(float), stream);
            dim3 grid(n_blocks, k_splits);
            gemv_fp16w_ksplit_scalar<<<grid, GEMV_FP16_SCALAR_THREADS, 0, stream>>>(
                x, W, y, bias, N, K, k_splits);
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in GEMV_FP16W: %s\n", cudaGetErrorString(err));
    }
}

} // namespace kernels

