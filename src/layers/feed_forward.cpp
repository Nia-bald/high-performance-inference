#include "transformer.h" // Assuming this header has the class definition and kernels.cuh
#include <cstdio>
#include <cuda_fp16.h>
#include <vector>

// --- Constructor ---
FeedForward::FeedForward(int d_model, int d_ff, GPUMemoryArena& weights_arena)
    : Layer("FeedForward"), d_model(d_model), d_ff(d_ff) 
{
    // FP16 weight storage: halves VRAM bandwidth for memory-bound projections
    // 1. Up Projection Weights
    d_W_up = weights_arena.allocate<__half>(d_model * d_ff);
    d_b_up = weights_arena.allocate<float>(d_ff);

    // 2. Down Projection Weights
    d_W_down = weights_arena.allocate<__half>(d_ff * d_model);
    d_b_down = weights_arena.allocate<float>(d_model);

    printf("[FeedForward] Initialized FP16 weights (Input: %d -> Hidden: %d -> Output: %d)\n", d_model, d_ff, d_model);
}

// --- Load Weights (FP32 -> FP16 conversion on host) ---
void FeedForward::load_weights(const float* h_W_up, const float* h_b_up, 
                               const float* h_W_down, const float* h_b_down) 
{
    // Convert up-projection weights FP32 -> FP16
    size_t up_count = d_model * d_ff;
    std::vector<__half> h_W_up_fp16(up_count);
    for (size_t i = 0; i < up_count; ++i)
        h_W_up_fp16[i] = __float2half(h_W_up[i]);
    cudaMemcpy(d_W_up, h_W_up_fp16.data(), up_count * sizeof(__half), cudaMemcpyHostToDevice);
    
    // Bias remains FP32
    cudaMemcpy(d_b_up, h_b_up, d_ff * sizeof(float), cudaMemcpyHostToDevice);
    
    // Convert down-projection weights FP32 -> FP16
    size_t down_count = d_ff * d_model;
    std::vector<__half> h_W_down_fp16(down_count);
    for (size_t i = 0; i < down_count; ++i)
        h_W_down_fp16[i] = __float2half(h_W_down[i]);
    cudaMemcpy(d_W_down, h_W_down_fp16.data(), down_count * sizeof(__half), cudaMemcpyHostToDevice);
    
    // Bias remains FP32
    cudaMemcpy(d_b_down, h_b_down, d_model * sizeof(float), cudaMemcpyHostToDevice);
}

// --- Forward Pass ---
void FeedForward::forward_impl(const float* d_input, float* d_output, int batch_size, int seq_len, GPUMemoryArena* inference_arena, cudaStream_t stream, IKVCache* /*kv_cache*/) 
{
    // Total number of tokens to process
    int total_rows = batch_size * seq_len;

    // --- Step 1: Up Projection (Expand) ---
    // Input [Rows, d_model] * W_up [d_model, d_ff] -> Hidden [Rows, d_ff]
    
    float* d_hidden = inference_arena->allocate<float>(total_rows * d_ff);

    kernels::launch_gemm_tiled_fp16w(
        d_input, 
        d_W_up, 
        d_hidden, 
        total_rows, // M
        d_ff,       // N
        d_model,    // K
        stream
    );

    // --- Step 2: Fused Bias + GELU ---
    // Hidden = GELU(Hidden + b_up)
    // Note: We modify d_hidden in-place
    kernels::launch_bias_gelu(d_hidden, d_b_up, total_rows, d_ff, stream);


    // --- Step 3: Down Projection (Contract) ---
    // Hidden [Rows, d_ff] * W_down [d_ff, d_model] -> Output [Rows, d_model]
    
    kernels::launch_gemm_tiled_fp16w(
        d_hidden, 
        d_W_down, 
        d_output, 
        total_rows, // M
        d_model,    // N
        d_ff,       // K
        stream
    );

    // Output = Output + b_down
    kernels::launch_bias_add(d_output, d_b_down, total_rows, d_model, stream);
}

size_t FeedForward::estimate_weight_memory(int d_model, int d_ff) {
    // FP16 weights + FP32 biases
    return (d_model * d_ff + d_ff * d_model) * sizeof(__half) + (d_ff + d_model) * sizeof(float);
}

size_t FeedForward::estimate_inference_scratch(int max_batch_size, int max_seq_len, int d_ff) {
    return (max_batch_size * max_seq_len * d_ff) * sizeof(float);
}