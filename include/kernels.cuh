#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstddef>

namespace kernels {

    void launch_embedding_lookup(
        const int* token_ids, 
        const float* token_table, 
        const float* pos_table, 
        float* output, 
        int batch_size, 
        int current_seq_len, // Passed from forward()
        int d_model, 
        cudaStream_t stream = 0,
        int start_pos = 0);

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

    void launch_batched_gemm(
        const float* A,  // [M, K]
        const float* B,  // [K, N]
        float* C, // [M, N]
        int M, int N, int K,
        int stride_A, int stride_B, int stride_K,
        cudaStream_t stream = 0
    );
    void launch_batched_gemm_naive(
        const float* A,  // [M, K]
        const float* B,  // [K, N]
        float* C, // [M, N]
        int M, int N, int K,
        int stride_A, int stride_B, int stride_K,
        cudaStream_t stream = 0
    );

    void launch_batch_upper_triangulate(
        float* data,
        int rows,
        int cols,
        int stride_row,
        int stride_col,
        cudaStream_t stream = 0
    );
    void launch_batched_one_to_one_gemm(
        const float* A,  // [M, K]
        const float* B,  // [K, N]
        float* C, // [M, N]
        int M, int N, int K,
        int stride_A, int stride_B, int stride_K,
        cudaStream_t stream = 0
    );
    void launch_batched_one_to_one_gemm_naive(
        const float* A,  // [M, K]
        const float* B,  // [K, N]
        float* C, // [M, N]
        int M, int N, int K,
        int stride_A, int stride_B, int stride_K,
        cudaStream_t stream = 0
    );
    void launch_addition(
        const float* A,  
        const float* B,  
        float* C,  
        const int length,
        cudaStream_t stream = 0
    );

    void launch_bias_gelu(float* data,
        const float* bias,
        int rows,
        int cols,
        cudaStream_t stream = 0);

    // GELU activation only (no bias add)
    void launch_gelu(float* data, int length, cudaStream_t stream = 0);

    void launch_bias_add(float* data,
        const float* bias,
        int rows,
        int cols,
        cudaStream_t stream = 0);
    
    // Element-wise scaling: output = input * scale
    void launch_scale(
        float* data,
        float scale,
        int length,
        cudaStream_t stream = 0);
            // ... existing kernels ...
        
    // Greedy Sampling (Argmax)
    void launch_argmax(
        const float* logits, 
        int* output_ids, 
        int batch_size, 
        int seq_len, 
        int vocab_size, 
        int row_stride = 0,
        cudaStream_t stream = 0
    );

    // KV Cache: Append new K/V vectors into cache slots (coalesced write)
    // src:   [batch_size, seq_len, num_heads, head_dim]
    // cache: [batch_size, max_seq_len, num_heads, head_dim]
    // Inner dimensions match — naturally coalesced writes.
    void launch_cache_append(
        const float* src,
        float* cache,
        int pos,
        int seq_len,
        int batch_size,
        int num_heads,
        int max_seq_len,
        int head_dim,
        cudaStream_t stream = 0
    );

    // KV Cache: Gather valid entries from cache into flat buffer
    // cache:  [batch_size, max_seq_len, num_heads, head_dim]
    // output: [batch_size * pos, num_heads * head_dim]
    // NOTE: No longer needed during decode — attention reads directly from cache.
    void launch_cache_gather(
        const float* cache,
        float* output,
        int pos,
        int batch_size,
        int num_heads,
        int max_seq_len,
        int head_dim,
        cudaStream_t stream = 0
    );

    // KV Cache: Fused pitched-transpose — read directly from cache, write transposed
    // Replaces gather_K + transpose_K in the decode path.
    // cache:  [batch_size, max_seq_len, total_qk_dim]  (pitched, only first pos rows valid)
    // output: [total_qk_dim, batch_size * pos]          (transposed & packed, no gaps)
    void launch_cache_pitched_transpose(
        const float* cache,
        float* output,
        int pos,
        int max_seq_len,
        int total_qk_dim,
        int batch_size,
        cudaStream_t stream = 0
    );

    // GEMV: y = x * W (+ bias), optimized for M=1 decode path
    void launch_gemv(
        const float* x,     // [K]
        const float* W,     // [K, N]
        float* y,           // [N]
        int N, int K,
        const float* bias = nullptr,  // [N] or nullptr
        cudaStream_t stream = 0
    );

    // Split fused QKV [rows, 3*D] into contiguous Q, K, V [rows, D] each
    void launch_deinterleave_qkv(
        const float* qkv,   // [rows, 3*D]
        float* q,            // [rows, D]
        float* k,            // [rows, D]
        float* v,            // [rows, D]
        int rows, int D,
        cudaStream_t stream = 0
    );

    // Fused GEMV + Bias + GELU: y = GELU(x * W + bias), optimized for M=1
    void launch_gemv_bias_gelu(
        const float* x,     // [K]
        const float* W,     // [K, N]
        float* y,           // [N]
        int N, int K,
        const float* bias,  // [N]
        cudaStream_t stream = 0
    );

    // Fused GEMV + Bias + Residual: y = x * W + bias + residual, optimized for M=1
    void launch_gemv_bias_residual(
        const float* x,       // [K]
        const float* W,       // [K, N]
        float* y,             // [N]
        int N, int K,
        const float* bias,    // [N]
        const float* residual,// [N]
        cudaStream_t stream = 0
    );

    // Fused Decode Attention: Q×K^T + softmax + Attn×V in one kernel
    void launch_fused_decode_attention(
        const float* Q,         // [total_qk_dim] (pre-scaled)
        const float* K_T,       // [total_qk_dim, total_tokens]
        const float* V,         // [max_seq_len, total_qk_dim] cache base
        float* output,          // [total_qk_dim]
        int num_heads,
        int head_dim,
        int total_tokens,
        int total_qk_dim,
        int v_stride,           // stride between tokens in V cache
        float scale,
        cudaStream_t stream = 0
    );

    // FP16 weight GEMV: y[N] = x[K] * W_half[K,N] + bias[N]
    void launch_gemv_fp16(
        const float* x,     // [K] FP32
        const half* W,      // [K, N] FP16
        float* y,           // [N] FP32
        int N, int K,
        const float* bias = nullptr,
        cudaStream_t stream = 0
    );

    // Convert FP32 weights to FP16 on GPU
    void launch_convert_fp32_to_fp16(
        const float* src, half* dst, int count,
        cudaStream_t stream = 0
    );
}