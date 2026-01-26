#include "kernels.cuh"
#include <cstdio>

namespace kernels {

    __global__ void embedding_lookup_kernel (
        const int* token_ids, // [Batch size, seq_len] 
        const float* table, // [Vocab size, Hidden dim] 
        float* output, // [Batch size, seq_len, hidden dim]  not returning a output as we would want to write straight to memory which has already been allocated 
        int vocab_size, int hidden_dim)
        {

        int token_idx = blockIdx.x;

        int token_id = token_ids[token_idx];

        if (token_id < 0 || token_id >= vocab_size){
            return;
        }

        int output_offset = token_idx * hidden_dim;

        int table_offset = token_id * hidden_dim;

        for (size_t i{threadIdx.x}; i < hidden_dim; i += blockDim.x){
            output[output_offset + i] = table[table_offset + i];
        }
    }

    void launch_embedding_lookup(
        const int* token_ids, // [Batch size, seq_len] (would be heap allocated as it could be huge)
        const float* table, // [Vocab size, Hidden dim] (would be heap allocated as it could be huge)
        float* output, // [Batch size, seq_len, hidden dim] (would be heap allocated as it could be huge) not returning a output as we would want to write straight to memory which has already been allocated 
        int batch_size, int seq_len, int hidden_dim, // can be stack allocated as these are just dimensions (not big)
        cudaStream_t stream//which cuda stream to perform this operation in
    ){
        int total_tokens = batch_size * seq_len;

        dim3 gridDim(total_tokens);

        dim3 blockDim(256);

        int vocab_size = 10000;

        embedding_lookup_kernel<<<gridDim, blockDim, 0, stream>>>(
            token_ids,
            table,
            output,
            vocab_size,
            hidden_dim
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess){
            printf("CUDA Error Embedding: %s\n", cudaGetErrorString(err));
        }

        
    }

}


