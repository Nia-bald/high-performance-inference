#include "kernels.cuh"
#include "memory.h"


#define TILE_SIZE 16

namespace kernels {

    __global__ void perform_tiled_gmm(
            const float* A,  // [M, K]
            const float* B,  // [K, N]
            float* C, // [M, N]
            int M, int N, int K
        ){
            size_t row = blockIdx.y * blockDim.y + threadIdx.y;
            size_t col = blockIdx.x * blockDim.x + threadIdx.x;


            __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
            __shared__ float shared_B[TILE_SIZE][TILE_SIZE];


            float total = 0.0f;
            for (size_t tile{}; tile <  (K + TILE_SIZE-1)/TILE_SIZE; ++tile){

                if (row < M && tile*TILE_SIZE + threadIdx.x < K){ 
                    shared_A[threadIdx.y][threadIdx.x] = A[row*K + tile*TILE_SIZE + threadIdx.x];
                }
                else{
                    shared_A[threadIdx.y][threadIdx.x] = 0.0f;
                }


                if (col < N && tile*TILE_SIZE + threadIdx.y < K){ 
                    shared_B[threadIdx.y][threadIdx.x] = B[(tile*TILE_SIZE + threadIdx.y)*N + col];
                }
                else{
                    shared_B[threadIdx.y][threadIdx.x] = 0.0f;
                }

                __syncthreads();

                #pragma unroll
                for (size_t k{}; k < TILE_SIZE; ++k){
                    total += shared_A[threadIdx.y][k]*shared_B[k][threadIdx.x];
                }

                __syncthreads();
            }

            if (row < M && col < N){
                C[row*N + col] = total;
            }

        }

    void launch_gemm_tiled(
        const float* A,  // [M, K]
        const float* B,  // [K, N]
        float* C, // [M, N]
        int M, int N, int K,
        cudaStream_t stream
    ){
        dim3 blockDim(TILE_SIZE, TILE_SIZE);

        // grid size needs to be floor  floor N/T and floor M/T because the number of threads should match the number elements  in MXN
        dim3 gridDim(
            (N + TILE_SIZE - 1)/TILE_SIZE, // # of cols
            (M + TILE_SIZE - 1)/TILE_SIZE // # of rows
        );

        perform_tiled_gmm<<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K);

        cudaError_t err = cudaGetLastError();

        if (err != cudaSuccess){
            printf("CUDA Error in GEMM: %s\n", cudaGetErrorString(err));
        }
    }


}