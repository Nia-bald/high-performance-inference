#include "kernels.cuh"
#include "memory.h"
#include <stdio.h>

#define TILE_SIZE 16

// only works properly when stride length is multiple of tilesize

namespace kernels {

    __global__ void perform_batched_tiled_gmm(
        const float* A,  // [M, K]
        const float* B,  // [K, N]
        float* C, // [M, N]
        int M, int N, int K,
        int stride_A, int stride_B, int stride_K
    ){
        size_t row = blockIdx.y * blockDim.y + threadIdx.y;
        size_t col = blockIdx.x * blockDim.x + threadIdx.x;


        __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
        __shared__ float shared_B[TILE_SIZE][TILE_SIZE];

        int B_i = row/stride_A;
        int B_j = col/stride_B;

        float total = 0.0f;
        for (size_t tile{}; tile <  (stride_K + TILE_SIZE-1)/TILE_SIZE; ++tile){

            if (row < M && tile*TILE_SIZE + threadIdx.x < stride_K){ 
                shared_A[threadIdx.y][threadIdx.x] = A[row*K + B_j*stride_K + tile*TILE_SIZE + threadIdx.x];
            }
            else{
                shared_A[threadIdx.y][threadIdx.x] = 0.0f;
            }


            if (stride_B*(B_i - B_j) + col < N && tile*TILE_SIZE + threadIdx.y < stride_K){ 
                shared_B[threadIdx.y][threadIdx.x] = B[(B_j*stride_K + tile*TILE_SIZE + threadIdx.y)*N + stride_B*(B_i - B_j) + col];
            }
            else{
                shared_B[threadIdx.y][threadIdx.x] = 0.0f;
            }

            __syncthreads();


            for (size_t k{}; k < TILE_SIZE; ++k){
                // if (row < M && col < N){
                // printf("%d, %d, %f, %f, %d, %d, %d\n", (int)row, (int)col, (float)shared_A[threadIdx.y][k], (float)shared_B[k][threadIdx.x], B_i, B_j, B_j*stride_K + tile*TILE_SIZE + threadIdx.x);}
                total += shared_A[threadIdx.y][k]*shared_B[k][threadIdx.x];
            }

            __syncthreads();
        }

        if (row < M && col < N){
            C[row*N + col] = total;
        }

    }


    void launch_batched_gemm(
        const float* A,  // [M, K]
        const float* B,  // [K, N]
        float* C, // [M, N]
        int M, int N, int K,
        int stride_A, int stride_B, int stride_K,
        cudaStream_t stream
    ){
        dim3 blockDim(TILE_SIZE, TILE_SIZE);

        // grid size needs to be floor  floor N/T and floor M/T because the number of threads should match the number elements  in MXN
        dim3 gridDim(
            (N + TILE_SIZE - 1)/TILE_SIZE, // # of cols
            (M + TILE_SIZE - 1)/TILE_SIZE // # of rows
        );
        // printf("row, col, a, b, Bi, Bj, custom_val\n");
        perform_batched_tiled_gmm<<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K, stride_A, stride_B, stride_K);

        cudaError_t err = cudaGetLastError();

        if (err != cudaSuccess){
            printf("CUDA Error in GEMM: %s\n", cudaGetErrorString(err));
        }
    }


}