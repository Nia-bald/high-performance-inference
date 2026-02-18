#include "kernels.cuh"
#include <cstdio>

namespace kernels {

    // Grid: [Batch * Seq] blocks (One block per token)
    // Block: [d_model] threads
    __global__ void embedding_lookup_kernel(
        const int* __restrict__ token_ids,      // [Batch, Seq]
        const float* __restrict__ token_table,  // [Vocab, d_model]
        const float* __restrict__ pos_table,    // [MaxSeq, d_model]
        float* __restrict__ output,             // [Batch, Seq, d_model]
        int d_model,
        int current_seq_len)                    // The ACTUAL length of this batch
    {
        int global_token_idx = blockIdx.x; 
        
        // CRITICAL FIX:
        // Identify position within the sequence (0..seq_len-1)
        // If we used max_seq_len here, Batch 1's start index would result in the wrong position.
        // Example: Batch 0 ends at index 9. Index 10 is start of Batch 1.
        // 10 % 10 = 0. Correct!
        int seq_position = global_token_idx % current_seq_len; 

        int token_id = token_ids[global_token_idx];

        // Each thread handles a subset of the embedding vector dimensions
        for (int i = threadIdx.x; i < d_model; i += blockDim.x) {
            
            // A. Look up Token Vector (Shape: [Vocab, d_model])
            float token_val = token_table[token_id * d_model + i];
            
            // B. Look up Positional Vector (Shape: [MaxSeq, d_model])
            // We use the calculated position (0..seq_len-1)
            float pos_val = pos_table[seq_position * d_model + i];

            // C. Sum and Store (Shape: [Batch, Seq, d_model])
            output[global_token_idx * d_model + i] = token_val + pos_val;
        }
    }

    void launch_embedding_lookup(
        const int* token_ids, 
        const float* token_table, 
        const float* pos_table, 
        float* output, 
        int batch_size, 
        int current_seq_len, // Passed from forward()
        int d_model, 
        cudaStream_t stream)
    {
        int total_tokens = batch_size * current_seq_len;
        
        // Cap threads at 1024 or d_model
        int threads = (d_model > 1024) ? 1024 : d_model;
        
        // Launch one block per token
        embedding_lookup_kernel<<<total_tokens, threads, 0, stream>>>(
            token_ids, 
            token_table, 
            pos_table, 
            output, 
            d_model, 
            current_seq_len 
        );
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error in Embedding Lookup: %s\n", cudaGetErrorString(err));
        }
    }
}