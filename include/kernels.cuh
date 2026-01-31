#pragma once
#include <cuda_runtime.h>
#include <cstddef>

namespace kernels {

    void launch_embedding_lookup(
        const int* token_ids, // [Batch size, seq_len] (would be heap allocated as it could be huge)
        const float* table, // [Vocab size, Hidden dim] (would be heap allocated as it could be huge)
        float* output, // [Batch size, seq_len, hidden dim] (would be heap allocated as it could be huge) not returning a output as we would want to write straight to memory which has already been allocated 
        int batch_size, int seq_len, int hidden_dim, // can be stack allocated as these are just dimensions (not big)
        cudaStream_t stream = 0//which cuda stream to perform this operation in
    );

    void launch_layer_norm(
        const float* input, // [Batch size, seq_len, hidden_dim] (would be heap allocated as it could be huge)
        float* output, // [Batch size, seq_len, hidden_dim] (would be heap allocated as it could be huge)
        const float* gamma, // [Batch size, seq_len, hidden dim] (would be heap allocated as it could be huge) not returning a output as we would want to write straight to memory which has already been allocated 
        const float* beta, // [Batch size, seq_len, hidden dim] (would be heap allocated as it could be huge) not returning a output as we would want to write straight to memory which has already been allocated 
        int batch_size, int seq_len, int hidden_dim,
        cudaStream_t stream = 0//which cuda stream to perform this operation in
    );

    void launch_softmax(
        float* input,
        int rows, int cols,
        cudaStream_t stream = 0
    );

    void launch_transpose(
        const float* input, float* output,
        int rows, int cols, // rows and cols of the input
        cudaStream_t stream = 0
    );

    // simple matrix multiplication 
    // we want to perform C = (A X B) + C
    void launch_gemm_tiled(
        const float* A,  // [M, K]
        const float* B,  // [K, N]
        float* C, // [M, N]
        int M, int N, int K,
        cudaStream_t stream = 0
    );

}