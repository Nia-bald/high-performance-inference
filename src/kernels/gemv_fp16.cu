#include "kernels.cuh"
#include <cuda_fp16.h>
#include <cstdio>

// ============================================================
// FP16 Weight GEMV: y[N] = x[K] * W_half[K, N] + bias[N]
// ============================================================
// Weights stored as FP16, input/output/bias remain FP32.
// Halves memory bandwidth for weight reads (the dominant bottleneck).
// Uses half2 loads (4 bytes) instead of float4 loads (16 bytes)
// with 8 half values per iteration = same compute density.
//
// On sm_61 (GTX 1050 Ti), __half2float is a 1-cycle instruction.

namespace kernels {

#define GEMV_FP16_THREADS 256

// ---- Kernel: FP16 weight GEMV, direct write (large N) ----
// Each thread handles 4 consecutive output columns (N4 = N/4)
// Loads 4 half values per column per K step via half2 pairs.
__global__ void __launch_bounds__(GEMV_FP16_THREADS)
gemv_fp16_vec4(
    const float* __restrict__ x,        // [K] FP32 input
    const half* __restrict__ W,         // [K, N] FP16 weights
    float* __restrict__ y,              // [N] FP32 output
    const float* __restrict__ bias,     // [N] FP32 bias or nullptr
    int N, int K
) {
    const int N4 = N >> 2;
    const int vec_col = blockIdx.x * GEMV_FP16_THREADS + threadIdx.x;
    if (vec_col >= N4) return;

    // Cast to half2 for 4-byte aligned loads
    // W layout: [K, N], W_h2 layout: [K, N/2] (half2)
    const int N2 = N >> 1;
    const half2* W_h2 = reinterpret_cast<const half2*>(W);

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    // Each vec_col handles 4 output columns: vec_col*4 .. vec_col*4+3
    // In half2: that's indices vec_col*2 and vec_col*2+1
    const int h2_base = vec_col * 2;

    int k = 0;
    for (; k + 3 < K; k += 4) {
        float x0 = x[k], x1 = x[k+1], x2 = x[k+2], x3 = x[k+3];

        // Load 4 half2 values per K step (covers 4 output columns × 4 K values)
        half2 w0a = W_h2[(k)   * N2 + h2_base];
        half2 w0b = W_h2[(k)   * N2 + h2_base + 1];
        half2 w1a = W_h2[(k+1) * N2 + h2_base];
        half2 w1b = W_h2[(k+1) * N2 + h2_base + 1];
        half2 w2a = W_h2[(k+2) * N2 + h2_base];
        half2 w2b = W_h2[(k+2) * N2 + h2_base + 1];
        half2 w3a = W_h2[(k+3) * N2 + h2_base];
        half2 w3b = W_h2[(k+3) * N2 + h2_base + 1];

        // Convert to float and accumulate
        sum0 += x0 * __half2float(w0a.x) + x1 * __half2float(w1a.x)
              + x2 * __half2float(w2a.x) + x3 * __half2float(w3a.x);
        sum1 += x0 * __half2float(w0a.y) + x1 * __half2float(w1a.y)
              + x2 * __half2float(w2a.y) + x3 * __half2float(w3a.y);
        sum2 += x0 * __half2float(w0b.x) + x1 * __half2float(w1b.x)
              + x2 * __half2float(w2b.x) + x3 * __half2float(w3b.x);
        sum3 += x0 * __half2float(w0b.y) + x1 * __half2float(w1b.y)
              + x2 * __half2float(w2b.y) + x3 * __half2float(w3b.y);
    }
    for (; k < K; ++k) {
        float xk = x[k];
        half2 wa = W_h2[k * N2 + h2_base];
        half2 wb = W_h2[k * N2 + h2_base + 1];
        sum0 += xk * __half2float(wa.x);
        sum1 += xk * __half2float(wa.y);
        sum2 += xk * __half2float(wb.x);
        sum3 += xk * __half2float(wb.y);
    }

    if (bias != nullptr) {
        const float4* b4 = reinterpret_cast<const float4*>(bias);
        float4 b = b4[vec_col];
        sum0 += b.x; sum1 += b.y; sum2 += b.z; sum3 += b.w;
    }

    float4* y4 = reinterpret_cast<float4*>(y);
    y4[vec_col] = make_float4(sum0, sum1, sum2, sum3);
}

// ---- K-split FP16 variant ----
__global__ void __launch_bounds__(GEMV_FP16_THREADS)
gemv_fp16_ksplit_vec4(
    const float* __restrict__ x,
    const half* __restrict__ W,
    float* __restrict__ y,
    const float* __restrict__ bias,
    int N, int K,
    int k_splits
) {
    const int N4 = N >> 2;
    const int vec_col = blockIdx.x * GEMV_FP16_THREADS + threadIdx.x;
    if (vec_col >= N4) return;

    const int split_id = blockIdx.y;
    const int stripe = (K + k_splits - 1) / k_splits;
    const int k_start = split_id * stripe;
    const int k_end = min(k_start + stripe, K);

    const int N2 = N >> 1;
    const half2* W_h2 = reinterpret_cast<const half2*>(W);
    const int h2_base = vec_col * 2;

    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;

    int k = k_start;
    for (; k + 3 < k_end; k += 4) {
        float x0 = x[k], x1 = x[k+1], x2 = x[k+2], x3 = x[k+3];
        half2 w0a = W_h2[(k)   * N2 + h2_base];
        half2 w0b = W_h2[(k)   * N2 + h2_base + 1];
        half2 w1a = W_h2[(k+1) * N2 + h2_base];
        half2 w1b = W_h2[(k+1) * N2 + h2_base + 1];
        half2 w2a = W_h2[(k+2) * N2 + h2_base];
        half2 w2b = W_h2[(k+2) * N2 + h2_base + 1];
        half2 w3a = W_h2[(k+3) * N2 + h2_base];
        half2 w3b = W_h2[(k+3) * N2 + h2_base + 1];

        sum0 += x0*__half2float(w0a.x) + x1*__half2float(w1a.x) + x2*__half2float(w2a.x) + x3*__half2float(w3a.x);
        sum1 += x0*__half2float(w0a.y) + x1*__half2float(w1a.y) + x2*__half2float(w2a.y) + x3*__half2float(w3a.y);
        sum2 += x0*__half2float(w0b.x) + x1*__half2float(w1b.x) + x2*__half2float(w2b.x) + x3*__half2float(w3b.x);
        sum3 += x0*__half2float(w0b.y) + x1*__half2float(w1b.y) + x2*__half2float(w2b.y) + x3*__half2float(w3b.y);
    }
    for (; k < k_end; ++k) {
        float xk = x[k];
        half2 wa = W_h2[k * N2 + h2_base];
        half2 wb = W_h2[k * N2 + h2_base + 1];
        sum0 += xk * __half2float(wa.x);
        sum1 += xk * __half2float(wa.y);
        sum2 += xk * __half2float(wb.x);
        sum3 += xk * __half2float(wb.y);
    }

    float* y_raw = y;
    int base = vec_col * 4;
    atomicAdd(&y_raw[base],   sum0);
    atomicAdd(&y_raw[base+1], sum1);
    atomicAdd(&y_raw[base+2], sum2);
    atomicAdd(&y_raw[base+3], sum3);

    if (bias != nullptr && split_id == 0) {
        const float4* b4 = reinterpret_cast<const float4*>(bias);
        float4 b = b4[vec_col];
        atomicAdd(&y_raw[base],   b.x);
        atomicAdd(&y_raw[base+1], b.y);
        atomicAdd(&y_raw[base+2], b.z);
        atomicAdd(&y_raw[base+3], b.w);
    }
}

// ---- FP32→FP16 weight conversion kernel ----
__global__ void convert_fp32_to_fp16_kernel(
    const float* __restrict__ src,
    half* __restrict__ dst,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = __float2half(src[idx]);
    }
}

void launch_convert_fp32_to_fp16(
    const float* src, half* dst, int count, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    convert_fp32_to_fp16_kernel<<<blocks, threads, 0, stream>>>(src, dst, count);
}

// ---- FP16 GEMV launcher ----
void launch_gemv_fp16(
    const float* x,
    const half* W,
    float* y,
    int N, int K,
    const float* bias,
    cudaStream_t stream
) {
    const int TARGET_BLOCKS = 24;
    int N4 = N >> 2;
    int n_blocks = (N4 + GEMV_FP16_THREADS - 1) / GEMV_FP16_THREADS;

    if (n_blocks >= TARGET_BLOCKS) {
        gemv_fp16_vec4<<<n_blocks, GEMV_FP16_THREADS, 0, stream>>>(
            x, W, y, bias, N, K);
    } else {
        int k_splits = (TARGET_BLOCKS + n_blocks - 1) / n_blocks;
        int max_splits = max(1, K / 32);
        k_splits = min(k_splits, max_splits);

        cudaMemsetAsync(y, 0, N * sizeof(float), stream);
        dim3 grid(n_blocks, k_splits);
        gemv_fp16_ksplit_vec4<<<grid, GEMV_FP16_THREADS, 0, stream>>>(
            x, W, y, bias, N, K, k_splits);
    }
}

} // namespace kernels
