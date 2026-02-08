#include "kernels.cuh"
#include <cstdio>

#define NEG_INF __int_as_float(0xff800000)
#define TILESIZE 32

namespace kernels {


    __global__ void batch_upper_triangulate(float* data, int rows, int cols, int stride_row, int stride_col){

        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y; 

        if (row < rows && col < cols){
            int LHS = row + (col/stride_col) * stride_col;
            int RHS = col + (row/stride_row) * stride_row;
            bool condition = LHS < RHS;
            if (condition){
                data[row*cols + col] = NEG_INF;
            }
        }
        
    }
    void launch_batch_upper_triangulate(
        float* data,
        int rows,
        int cols,
        int stride_row,
        int stride_col,
        cudaStream_t stream
    ){
        dim3 gridDim((cols + TILESIZE - 1)/TILESIZE , (rows + TILESIZE - 1)/TILESIZE);

        dim3 blockDim(32, 32);

        batch_upper_triangulate<<<gridDim, blockDim, 0, stream>>>(data, rows, cols, stride_row, stride_col);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess){
            printf("CUDA Error Transpose: %s\n", cudaGetErrorString(err));
        }
    }

}