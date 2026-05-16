#include "kernels.cuh"
#include "memory.h"
#include <cublas_v2.h>

// ============================================================
// Register-Tiled SGEMM
// ============================================================
// Block tile:  BM × BN elements of C, computed by (BM/TM)×(BN/TN) threads.
// Thread tile: each thread accumulates a TM × TN sub-tile in registers.
// K-dimension is walked in steps of BK.
//
// Parameters chosen for GTX 1050 Ti (SM 6.1, 48 KB smem, 65536 regs/SM):
//   BM=64, BN=64, BK=8, TM=8, TN=8  →  64 threads/block, 4 KB smem
// ============================================================

#define BM 64 //total output elements of C in M direction per block
#define BN 64 //total output elements of C in N direction per block
#define BK 8  //total share mem load of A nad B in K direction per block
#define TM 8  // same thing but per thread
#define TN 8  // same ..

namespace kernels {

__global__ void gemm_register_tiled(
    const float* __restrict__ A,  // [M, K]
    const float* __restrict__ B,  // [K, N]
    float* __restrict__ C,        // [M, N]
    int M, int N, int K
) {
    // ---- Thread identity within the block ----
    // Block has (BM/TM) × (BN/TN) = 8×8 = 64 threads, indexed 1-D.
    const int threadRow = threadIdx.x / (BN / TN);  // 0..7  (row of thread tiles)
    const int threadCol = threadIdx.x % (BN / TN);  // 0..7  (col of thread tiles)

    // ---- Shared-memory tiles ----
    __shared__ float smA[BM][BK];   // 64 × 8
    __shared__ float smB[BK][BN];   // 8 × 64

    // ---- Register accumulators (TM × TN = 64 floats) ----
    float accum[TM][TN] = {};
    float rA[TM], rB[TN];

    // ---- Global offsets for this block ----
    const int bRow = blockIdx.y * BM;
    const int bCol = blockIdx.x * BN;

    // ---- Precompute load indices for coalesced access ----
    // A tile (BM × BK): consecutive threads load consecutive columns.
    //   64 threads, BK=8 cols ⇒ 8 threads per row, 8 rows per iteration, 8 iters.
    const int ldA_col = threadIdx.x % BK;              // 0..7 within a row
    const int ldA_rowBase = threadIdx.x / BK;           // which row-group (0..7)
    constexpr int ldA_rowStride = 64 / BK;              // 8 rows per pass

    // B tile (BK × BN): consecutive threads load consecutive columns.
    //   64 threads, BN=64 cols ⇒ 1 thread per column, 1 row per iteration, 8 iters.
    const int ldB_col = threadIdx.x;                    // 0..63
    // ldB_row iterated 0..BK-1

    // ---- Main loop over K in steps of BK ----
    for (int tile = 0; tile < K; tile += BK) {

        // -- Load A tile (BM × BK) into smA, coalesced --
        #pragma unroll
        for (int r = 0; r < BM; r += ldA_rowStride) {
            int row = r + ldA_rowBase;
            int gRow = bRow + row;
            int gCol = tile + ldA_col;
            smA[row][ldA_col] = (gRow < M && gCol < K)
                ? A[gRow * K + gCol] : 0.0f;
        }

        // -- Load B tile (BK × BN) into smB, coalesced --
        #pragma unroll
        for (int r = 0; r < BK; ++r) {
            int gRow = tile + r;
            int gCol = bCol + ldB_col;
            smB[r][ldB_col] = (gRow < K && gCol < N)
                ? B[gRow * N + gCol] : 0.0f;
        }

        __syncthreads();

        // -- Register-tiled compute: outer products over BK --
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // Load A column fragment into registers
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                rA[i] = smA[threadRow * TM + i][k];

            // Load B row fragment into registers
            #pragma unroll
            for (int j = 0; j < TN; ++j)
                rB[j] = smB[k][threadCol * TN + j];

            // Rank-1 update
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    accum[i][j] += rA[i] * rB[j];
        }

        __syncthreads();
    }

    // ---- Store TM × TN result tile to global memory ----
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int gRow = bRow + threadRow * TM + i;
        if (gRow < M) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int gCol = bCol + threadCol * TN + j;
                if (gCol < N)
                    C[gRow * N + gCol] = accum[i][j];
            }
        }
    }
}

// Lazily-initialized cuBLAS handle
static cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = nullptr;
    if (!handle) {
        cublasCreate(&handle);
    }
    return handle;
}

void launch_gemm_tiled(
    const float* A,  // [M, K]
    const float* B,  // [K, N]
    float* C,        // [M, N]
    int M, int N, int K,
    cudaStream_t stream
) {
    // ---- M=1 fast path: dispatch to optimized GEMV kernel ----
    // When M==1, the operation is a row-vector × matrix multiply:
    //   C[1, N] = A[1, K] × B[K, N]
    // This is purely memory-bound (arithmetic intensity ≈ 0.25).
    // The dedicated GEMV kernel uses coalesced float4 loads and
    // K-split parallelism, achieving 85-101 GB/s on GTX 1050 Ti.
    //
    // The register-tiled GEMM kernel is optimized for compute-bound
    // workloads (large M) and has significant overhead for M=1:
    //   - Shared memory staging with __syncthreads (unnecessary for M=1)
    //   - 64 threads per block but only 1 row of useful work
    //   - TM=8 register tile wastes 7/8 of its computation on padding
    //
    // Bias is NOT passed here — all call sites add bias separately
    // via launch_bias_add() or launch_bias_gelu().
    if (M == 1) {
        launch_gemv(A, B, C, N, K, nullptr, stream);
        return;
    }

    // ---- General GEMM path (M > 1) ----
    constexpr int THREADS = (BM / TM) * (BN / TN);  // 64
    dim3 block(THREADS);
    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );

    gemm_register_tiled<<<grid, block, 0, stream>>>(A, B, C, M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in GEMM: %s\n", cudaGetErrorString(err));
    }

}

} // namespace kernels