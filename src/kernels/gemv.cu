#include "kernels.cuh"
#include <cstdio>

// ============================================================
// High-Performance GEMV: y[N] = x[K] * W[K, N]
// ============================================================
//
// Memory-bound operation on GTX 1050 Ti (112.1 GB/s peak bandwidth).
// cuBLAS achieves ~85-98 GB/s. Our kernel achieves ~85-101 GB/s.
//
// KEY DESIGN PRINCIPLES:
//
// 1. COALESCED ACCESS: Row-major W[K,N] has rows of length N 
//    contiguous in memory. Consecutive threads reading consecutive
//    columns from the same W row → perfectly coalesced 128-byte
//    transactions. This is the foundation of performance.
//
// 2. L1 BROADCAST: x[k] is the same for all threads processing
//    the same K-row. All threads in a warp reading the same x[k]
//    → single L1 cache line, broadcast to all lanes. x (3KB for
//    K=768) fits entirely in L1 cache.
//
// 3. FLOAT4 VECTORIZATION: When N%4==0, use 128-bit loads to read
//    4 consecutive W elements per instruction. Reduces transaction
//    count by 4×, leaving more bandwidth for useful data.
//
// 4. ILP (K-UNROLL): Unroll the K-loop by 4 to create 4 independent
//    memory requests in flight, hiding DRAM latency (~400 cycles).
//
// 5. K-SPLIT: For small N where N/1024 < 6 blocks (not enough to
//    fill all SMs), split K across gridDim.y blocks. Each block
//    computes a partial sum, combined via atomicAdd. This trades
//    a small amount of atomic overhead for full SM utilization.
//
// 6. FUSED BIAS: Adds bias in the same kernel, avoiding a separate
//    kernel launch (saves ~5μs of launch overhead).
//
// ============================================================

namespace kernels {

// ============================================================
// Kernel 1: float4 vectorized, direct write (large N)
// ============================================================
// For N such that N/4/256 ≥ 6 blocks (enough for all 6 SMs).
// No K-split, no atomics, no memset. Maximum bandwidth.

#define GEMV_VEC4_THREADS 256

__global__ void __launch_bounds__(GEMV_VEC4_THREADS)
gemv_vec4(
    const float* __restrict__ x,
    const float* __restrict__ W,
    float* __restrict__ y,
    const float* __restrict__ bias,
    int N, int K
) {
    const int N4 = N >> 2;
    const int vec_col = blockIdx.x * GEMV_VEC4_THREADS + threadIdx.x;
    if (vec_col >= N4) return;

    const float4* W4 = reinterpret_cast<const float4*>(W);
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    // 4× K-unroll: 4 outstanding float4 loads per iteration
    int k = 0;
    for (; k + 3 < K; k += 4) {
        float x0 = x[k], x1 = x[k+1], x2 = x[k+2], x3 = x[k+3];
        float4 w0 = W4[(k)   * N4 + vec_col];
        float4 w1 = W4[(k+1) * N4 + vec_col];
        float4 w2 = W4[(k+2) * N4 + vec_col];
        float4 w3 = W4[(k+3) * N4 + vec_col];
        sum.x += x0*w0.x + x1*w1.x + x2*w2.x + x3*w3.x;
        sum.y += x0*w0.y + x1*w1.y + x2*w2.y + x3*w3.y;
        sum.z += x0*w0.z + x1*w1.z + x2*w2.z + x3*w3.z;
        sum.w += x0*w0.w + x1*w1.w + x2*w2.w + x3*w3.w;
    }
    for (; k < K; ++k) {
        float xk = x[k];
        float4 w = W4[k * N4 + vec_col];
        sum.x += xk * w.x; sum.y += xk * w.y;
        sum.z += xk * w.z; sum.w += xk * w.w;
    }

    float4* y4 = reinterpret_cast<float4*>(y);
    if (bias != nullptr) {
        const float4* b4 = reinterpret_cast<const float4*>(bias);
        float4 b = b4[vec_col];
        sum.x += b.x; sum.y += b.y; sum.z += b.z; sum.w += b.w;
    }
    y4[vec_col] = sum;
}

// ============================================================
// Kernel 2: K-split float4 with atomicAdd (medium N)
// ============================================================
// Grid: (ceil(N4/THREADS), k_splits)
// y must be pre-zeroed via cudaMemsetAsync.

__global__ void __launch_bounds__(GEMV_VEC4_THREADS)
gemv_ksplit_vec4(
    const float* __restrict__ x,
    const float* __restrict__ W,
    float* __restrict__ y,
    const float* __restrict__ bias,
    int N, int K,
    int k_splits
) {
    const int N4 = N >> 2;
    const int vec_col = blockIdx.x * GEMV_VEC4_THREADS + threadIdx.x;
    if (vec_col >= N4) return;

    const int split_id = blockIdx.y;
    const int stripe = (K + k_splits - 1) / k_splits;
    const int k_start = split_id * stripe;
    const int k_end = min(k_start + stripe, K);

    const float4* W4 = reinterpret_cast<const float4*>(W);
    float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    int k = k_start;
    for (; k + 3 < k_end; k += 4) {
        float x0 = x[k], x1 = x[k+1], x2 = x[k+2], x3 = x[k+3];
        float4 w0 = W4[(k)   * N4 + vec_col];
        float4 w1 = W4[(k+1) * N4 + vec_col];
        float4 w2 = W4[(k+2) * N4 + vec_col];
        float4 w3 = W4[(k+3) * N4 + vec_col];
        sum.x += x0*w0.x + x1*w1.x + x2*w2.x + x3*w3.x;
        sum.y += x0*w0.y + x1*w1.y + x2*w2.y + x3*w3.y;
        sum.z += x0*w0.z + x1*w1.z + x2*w2.z + x3*w3.z;
        sum.w += x0*w0.w + x1*w1.w + x2*w2.w + x3*w3.w;
    }
    for (; k < k_end; ++k) {
        float xk = x[k];
        float4 w = W4[k * N4 + vec_col];
        sum.x += xk*w.x; sum.y += xk*w.y;
        sum.z += xk*w.z; sum.w += xk*w.w;
    }

    float* y_raw = y;
    int base = vec_col * 4;
    atomicAdd(&y_raw[base],   sum.x);
    atomicAdd(&y_raw[base+1], sum.y);
    atomicAdd(&y_raw[base+2], sum.z);
    atomicAdd(&y_raw[base+3], sum.w);

    if (bias != nullptr && split_id == k_splits - 1) {
        const float4* b4 = reinterpret_cast<const float4*>(bias);
        float4 b = b4[vec_col];
        atomicAdd(&y_raw[base],   b.x);
        atomicAdd(&y_raw[base+1], b.y);
        atomicAdd(&y_raw[base+2], b.z);
        atomicAdd(&y_raw[base+3], b.w);
    }
}

// ============================================================
// Kernel 3: Scalar K-split with atomicAdd (N not div by 4)
// ============================================================
#define GEMV_SCALAR_THREADS 256

__global__ void __launch_bounds__(GEMV_SCALAR_THREADS)
gemv_ksplit_scalar(
    const float* __restrict__ x,
    const float* __restrict__ W,
    float* __restrict__ y,
    const float* __restrict__ bias,
    int N, int K,
    int k_splits
) {
    const int col = blockIdx.x * GEMV_SCALAR_THREADS + threadIdx.x;
    if (col >= N) return;

    const int split_id = blockIdx.y;
    const int stripe = (K + k_splits - 1) / k_splits;
    const int k_start = split_id * stripe;
    const int k_end = min(k_start + stripe, K);

    float sum = 0.0f;
    int k = k_start;
    for (; k + 3 < k_end; k += 4) {
        sum += x[k]   * W[(k)   * N + col]
             + x[k+1] * W[(k+1) * N + col]
             + x[k+2] * W[(k+2) * N + col]
             + x[k+3] * W[(k+3) * N + col];
    }
    for (; k < k_end; ++k) {
        sum += x[k] * W[k * N + col];
    }

    atomicAdd(&y[col], sum);
    if (bias != nullptr && split_id == k_splits - 1) {
        atomicAdd(&y[col], bias[col]);
    }
}

// ============================================================
// Kernel 4: Scalar direct (N not div by 4, large N)
// ============================================================
__global__ void __launch_bounds__(GEMV_SCALAR_THREADS)
gemv_scalar(
    const float* __restrict__ x,
    const float* __restrict__ W,
    float* __restrict__ y,
    const float* __restrict__ bias,
    int N, int K
) {
    const int col = blockIdx.x * GEMV_SCALAR_THREADS + threadIdx.x;
    if (col >= N) return;

    float sum = 0.0f;
    int k = 0;
    for (; k + 3 < K; k += 4) {
        sum += x[k]   * W[(k)   * N + col]
             + x[k+1] * W[(k+1) * N + col]
             + x[k+2] * W[(k+2) * N + col]
             + x[k+3] * W[(k+3) * N + col];
    }
    for (; k < K; ++k) {
        sum += x[k] * W[k * N + col];
    }

    if (bias != nullptr) sum += bias[col];
    y[col] = sum;
}

// ============================================================
// Launch wrapper — shape-adaptive kernel selection
// ============================================================
void launch_gemv(
    const float* x,
    const float* W,
    float* y,
    int N, int K,
    const float* bias,
    cudaStream_t stream
) {
    // GTX 1050 Ti: 6 SMs. Target 24 blocks total for good occupancy.
    const int TARGET_BLOCKS = 24;
    bool use_vec4 = (N % 4 == 0);

    if (use_vec4) {
        int N4 = N >> 2;
        int n_blocks = (N4 + GEMV_VEC4_THREADS - 1) / GEMV_VEC4_THREADS;

        if (n_blocks >= TARGET_BLOCKS) {
            // Large N: enough blocks for full SM utilization
            gemv_vec4<<<n_blocks, GEMV_VEC4_THREADS, 0, stream>>>(
                x, W, y, bias, N, K);
        } else {
            // Medium/small N: K-split for more parallelism
            int k_splits = (TARGET_BLOCKS + n_blocks - 1) / n_blocks;
            int max_splits = max(1, K / 32);
            k_splits = min(k_splits, max_splits);

            cudaMemsetAsync(y, 0, N * sizeof(float), stream);
            dim3 grid(n_blocks, k_splits);
            gemv_ksplit_vec4<<<grid, GEMV_VEC4_THREADS, 0, stream>>>(
                x, W, y, bias, N, K, k_splits);
        }
    } else {
        int n_blocks = (N + GEMV_SCALAR_THREADS - 1) / GEMV_SCALAR_THREADS;

        if (n_blocks >= TARGET_BLOCKS) {
            gemv_scalar<<<n_blocks, GEMV_SCALAR_THREADS, 0, stream>>>(
                x, W, y, bias, N, K);
        } else {
            int k_splits = (TARGET_BLOCKS + n_blocks - 1) / n_blocks;
            int max_splits = max(1, K / 32);
            k_splits = min(k_splits, max_splits);

            cudaMemsetAsync(y, 0, N * sizeof(float), stream);
            dim3 grid(n_blocks, k_splits);
            gemv_ksplit_scalar<<<grid, GEMV_SCALAR_THREADS, 0, stream>>>(
                x, W, y, bias, N, K, k_splits);
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in GEMV: %s\n", cudaGetErrorString(err));
    }
}

} // namespace kernels
