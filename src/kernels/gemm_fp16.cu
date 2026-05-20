#include "kernels.cuh"
#include <cuda_fp16.h>
#include <cstdio>

// ============================================================
// Register-Tiled SGEMM with FP16 Weights
// ============================================================
// Same algorithm as gemm.cu but reads weights from __half storage,
// converting to FP32 on-load in shared memory. This halves the
// global memory bandwidth for the B matrix (weights).
//
// C[M,N] = A[M,K] (FP32 activations) × B[K,N] (FP16 weights)
//
// Parameters: BM=64, BN=64, BK=8, TM=8, TN=8 (same as FP32 version)
// ============================================================

#define FP16_BM 64
#define FP16_BN 64
#define FP16_BK 8
#define FP16_TM 8
#define FP16_TN 8

namespace kernels {

__global__ void gemm_register_tiled_fp16w(
    const float* __restrict__ A,      // [M, K] — FP32 activations
    const __half* __restrict__ B,     // [K, N] — FP16 weights
    float* __restrict__ C,            // [M, N] — FP32 output
    int M, int N, int K
) {
    const int threadRow = threadIdx.x / (FP16_BN / FP16_TN);
    const int threadCol = threadIdx.x % (FP16_BN / FP16_TN);

    __shared__ float smA[FP16_BM][FP16_BK];
    __shared__ float smB[FP16_BK][FP16_BN];

    float accum[FP16_TM][FP16_TN] = {};
    float rA[FP16_TM], rB[FP16_TN];

    const int bRow = blockIdx.y * FP16_BM;
    const int bCol = blockIdx.x * FP16_BN;

    const int ldA_col = threadIdx.x % FP16_BK;
    const int ldA_rowBase = threadIdx.x / FP16_BK;
    constexpr int ldA_rowStride = 64 / FP16_BK;  // 8

    const int ldB_col = threadIdx.x;  // 0..63

    for (int tile = 0; tile < K; tile += FP16_BK) {
        // Load A tile (FP32 → FP32, same as before)
        #pragma unroll
        for (int r = 0; r < FP16_BM; r += ldA_rowStride) {
            int row = r + ldA_rowBase;
            int gRow = bRow + row;
            int gCol = tile + ldA_col;
            smA[row][ldA_col] = (gRow < M && gCol < K)
                ? A[gRow * K + gCol] : 0.0f;
        }

        // Load B tile (FP16 → FP32 conversion on load)
        #pragma unroll
        for (int r = 0; r < FP16_BK; ++r) {
            int gRow = tile + r;
            int gCol = bCol + ldB_col;
            if (gRow < K && gCol < N) {
                smB[r][ldB_col] = __half2float(B[gRow * N + gCol]);
            } else {
                smB[r][ldB_col] = 0.0f;
            }
        }

        __syncthreads();

        // Register-tiled compute (identical to FP32 version)
        #pragma unroll
        for (int k = 0; k < FP16_BK; ++k) {
            #pragma unroll
            for (int i = 0; i < FP16_TM; ++i)
                rA[i] = smA[threadRow * FP16_TM + i][k];
            #pragma unroll
            for (int j = 0; j < FP16_TN; ++j)
                rB[j] = smB[k][threadCol * FP16_TN + j];
            #pragma unroll
            for (int i = 0; i < FP16_TM; ++i)
                #pragma unroll
                for (int j = 0; j < FP16_TN; ++j)
                    accum[i][j] += rA[i] * rB[j];
        }

        __syncthreads();
    }

    // Store result
    #pragma unroll
    for (int i = 0; i < FP16_TM; ++i) {
        int gRow = bRow + threadRow * FP16_TM + i;
        if (gRow < M) {
            #pragma unroll
            for (int j = 0; j < FP16_TN; ++j) {
                int gCol = bCol + threadCol * FP16_TN + j;
                if (gCol < N)
                    C[gRow * N + gCol] = accum[i][j];
            }
        }
    }
}

void launch_gemm_tiled_fp16w(
    const float* A,       // [M, K] — FP32
    const __half* B,      // [K, N] — FP16
    float* C,             // [M, N] — FP32
    int M, int N, int K,
    cudaStream_t stream
) {
    // M=1 fast path: dispatch to optimized FP16 GEMV
    if (M == 1) {
        launch_gemv_fp16w(A, B, C, N, K, nullptr, stream);
        return;
    }

    // General GEMM path (M > 1)
    constexpr int THREADS = (FP16_BM / FP16_TM) * (FP16_BN / FP16_TN);  // 64
    dim3 block(THREADS);
    dim3 grid(
        (N + FP16_BN - 1) / FP16_BN,
        (M + FP16_BM - 1) / FP16_BM
    );

    gemm_register_tiled_fp16w<<<grid, block, 0, stream>>>(A, B, C, M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in GEMM_FP16W: %s\n", cudaGetErrorString(err));
    }
}

} // namespace kernels
