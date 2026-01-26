#include "kernels.cuh"
#include <cstdio>
#define EPSILON 1e-5f

namespace kernels {

    __inline__ __device__ float warp_reduce_sum(float val){

        for (int offset = 16; offset > 0; offset /= 2){
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        return val;
    }

    __global__ void layer_norm_kernel(
        const float* input, // [Batch size, seq_len, hidden_dim] (would be heap allocated as it could be huge)
        float* output, // [Batch size, seq_len, hidden_dim] (would be heap allocated as it could be huge)
        const float* gamma, // [hidden dim] (would be heap allocated as it could be huge) not returning a output as we would want to write straight to memory which has already been allocated 
        const float* beta, // [hidden dim] (would be heap allocated as it could be huge) not returning a output as we would want to write straight to memory which has already been allocated 
        int hidden_dim // can be stack allocated as these are just dimensions (not big)
    )
    {

        int offset = blockIdx.x * hidden_dim;

        __shared__ float s_mean;
        __shared__ float s_var;

        if (threadIdx.x == 0){
            s_mean = 0.0f;
            s_var = 0.0f;
        }

        __syncthreads();


        float local_sum = 0.0f;
        float local_sum_squared = 0.0f;

        for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x){
            local_sum += input[offset + i];
            local_sum_squared += input[offset + i]*input[offset + i];

        }

        // __syncthreads(); as the 32 thread are on the same they all finish executing at the same time
        // so commenting above code out as warp_reduce_sum only performs interaction between threads in the same warp

        local_sum = warp_reduce_sum(local_sum);
        local_sum_squared = warp_reduce_sum(local_sum_squared);

        if (threadIdx.x%32 == 0){
            atomicAdd(&s_mean, local_sum);
            atomicAdd(&s_var, local_sum_squared);
        }

        __syncthreads();


        float mean = s_mean / hidden_dim;
        float variance = (s_var / hidden_dim) - (mean*mean);
        float inv_std = rsqrtf(variance + EPSILON);

        for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x){
            float val = input[offset + i];
            output[offset + i] = (val - mean)*inv_std*gamma[i] + beta[i];
        }

    }

    void launch_layer_norm(
        const float* input, // [Batch size, seq_len, hidden_dim] (would be heap allocated as it could be huge)
        float* output, // [Batch size, seq_len, hidden_dim] (would be heap allocated as it could be huge)
        const float* gamma, // [hidden dim] (would be heap allocated as it could be huge) not returning a output as we would want to write straight to memory which has already been allocated 
        const float* beta, // [hidden dim] (would be heap allocated as it could be huge) not returning a output as we would want to write straight to memory which has already been allocated 
        int batch_size, int seq_len, int hidden_dim,
        cudaStream_t stream //which cuda stream to perform this operation in
    ){
        dim3 gridDim(batch_size*seq_len);
        dim3 blockDim(256);

        layer_norm_kernel<<<gridDim, blockDim, 0, stream>>>(
            input,
            output,
            gamma,
            beta,
            hidden_dim
        );

        cudaError_t err = cudaGetLastError();
        
        if (err != cudaSuccess){
            printf("CUDA Error LayerNorm: %s\n", cudaGetErrorString(err));
        }
    }
}