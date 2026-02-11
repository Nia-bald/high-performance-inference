#include "kernels.cuh"
#include "memory.h"
#include <stdio.h>

#define TILE_SIZE 16

// only works properly when stride length is multiple of tilesize

namespace kernels {

    __global__  void perform_addition(
        const float* A,  
        const float* B,  
        float* C,  
        const int length
    )
    {
        size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < length){
            C[index] = A[index] + B[index];
        }
    }


    void launch_addition(
        const float* A,  
        const float* B,  
        float* C,  
        const int length,
        cudaStream_t stream
    ){
        dim3 blockDim(TILE_SIZE);

        dim3 gridDim(
            (length + TILE_SIZE - 1)/TILE_SIZE
        );

        perform_addition<<<gridDim, blockDim, 0, stream>>>(A, B, C, length);

        cudaError_t err = cudaGetLastError();

        if (err != cudaSuccess){
            printf("CUDA Error in addition: %s\n", cudaGetErrorString(err));
        }
    }


}