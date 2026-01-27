#include "kernels.cuh"
#include "cstdio"

namespace kernels {

    __inline__ __device__ float warp_reduce_max(float val){

        for (int offset{16}; offset > 0; offset /=2){
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        }
        return val;
    }

    __inline__ __device__ float warp_reduce_sum(float val) {
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        return val;
    }
    
    __global__ void softmax_kernel(float*input, int cols){
        // 2 steps,
        // get the max
        // get the exponent sum
        int offset = blockIdx.x * cols;

        __shared__ float s_max;


        float local_max = -INFINITY;

        for (int i = threadIdx.x; i < cols; i += blockDim.x){
            local_max = fmaxf(local_max, input[offset + i]);
        }
        

        local_max = warp_reduce_max(local_max);

        // __syncthread();

        // we are not doing atomicmax cause GTX 1050 ti does not support it

        // the idea is we want to get the max value from all the threads
        // in our block, all threads combine at this stage spans the maximum of entire vector
        // we have chosen our block size to be 256, and 256 threads = 32 warps
        // the idea of the below variable is the following
        // each index i will have the maximum of all the warp i of the current block
        __shared__ float max_warp[32];

        if (threadIdx.x % 32 == 0){
            max_warp[threadIdx.x/32] = local_max;
        }

        __syncthreads();

        if (threadIdx.x == 0){
            float block_max = -INFINITY;

            for (int i{0}; i < blockDim.x / 32; i += 1){ // blockDim.x / 32 not 32 because not all values in max_warp would get populated
                block_max = fmaxf(block_max, max_warp[i]);
            }
            s_max = block_max;
        }

        __syncthreads();

        __shared__ float s_sum;

        if (threadIdx.x == 0){
            s_sum = 0;
        }
        float local_sum = 0;

        for (int i = threadIdx.x; i < cols; i+= blockDim.x){
            float exponent = expf(input[offset + i] - s_max);
            input[offset + i] = exponent;
            local_sum += exponent;
        }

        local_sum = warp_reduce_sum(local_sum);

        if (threadIdx.x%32 == 0){
            atomicAdd(&s_sum, local_sum);
        }

        __syncthreads();


        float inv_sum = 1.0f/s_sum;

        for (int i = threadIdx.x; i < cols; i += blockDim.x){
            input[offset + i] = input[offset + i]*inv_sum;
        }
    }


    void launch_softmax(
        float* input,
        int rows, int cols,
        cudaStream_t stream
    ){
        dim3 gridDim(rows);
        dim3 blockDim(256);

        softmax_kernel<<<gridDim, blockDim, 0, stream>>>(input, cols);

        cudaError_t err = cudaGetLastError();

        if (err != cudaSuccess){
            printf("CUDA ERROR Softmax: %s\n", cudaGetErrorString(err));
        }

    }

}