#include "kernels.cuh"
#include <cstdio>

#define TILESIZE 32

namespace kernels {


    __global__ void transpose_kernel(float* odata, const float* idata, int rows, int cols){

        // the idea here is we will use shared memory to load the data
        // each thread would read data such that the shared memory is fully utilized
        // the + 1 is added to avoid writing or reading from the same bank
        __shared__ float tile[TILESIZE][TILESIZE + 1];


        // let's say our blockdimension is something like blockDim.x -> col and blockDim.y -> row
        // if each thread in the block needs to fully populate the TILSIZE x TILESIZE tile
        // then each thread will have to read TILESIZE * TILESIZE / blockDim.x * blockDim.y
        // if we just look at one axis the x dimension tiles will have to read TILESIZE / blockDim.x and y will read TILESIZE / blockDim.y

        // keeping in mind threads next to each other find it easier from addresses next to each other I am writing the following algo
        // keeping in mind threads next to each other find it easier to read from address next to each other it does not make sense to have a block which has its row dimension
        // not equal to tile length, reading row wise in strided manner each thread will have to read from completely different address which are not next to each other at the same time
        int i = threadIdx.y;
        int j = threadIdx.x;
        int B_x = blockIdx.x;
        int B_y = blockIdx.y;
        
        for (int stride{}; stride < TILESIZE; stride += blockDim.y){

            if (B_y*TILESIZE + i + stride < rows && B_x*TILESIZE + j < cols){
                tile[ threadIdx.y + stride][ threadIdx.x] = idata[(B_y*TILESIZE + i + stride)*cols + B_x*TILESIZE + j];
            }
        }

        __syncthreads();

        for (int stride{}; stride < TILESIZE; stride += blockDim.y){
            if (B_x*TILESIZE + i + stride < cols && B_y*TILESIZE + j < rows){
                odata[(B_x*TILESIZE + i + stride)*rows + B_y*TILESIZE + j ] = tile[ threadIdx.x ][ threadIdx.y + stride];
            }
        }
    }
    void launch_transpose(
        const float* input, float* output,
        int rows, int cols, // rows and cols of the input
        cudaStream_t stream
    ){
        dim3 gridDim((cols + TILESIZE - 1)/TILESIZE , (rows + TILESIZE - 1)/TILESIZE);

        dim3 blockDim(32, 8);

        transpose_kernel<<<gridDim, blockDim, 0, stream>>>(output, input, rows, cols);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess){
            printf("CUDA Error Transpose: %s\n", cudaGetErrorString(err));
        }
    }

}