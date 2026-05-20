#include "transformer.h" // Assuming this header has the class definition and kernels.cuh
#include <cstdio>

// --- Constructor ---
FeedForward::FeedForward(int d_model, int d_ff, GPUMemoryArena& weights_arena)
    : Layer("FeedForward"), d_model(d_model), d_ff(d_ff) 
{
    // 1. Up Projection Weights
    d_W_up = weights_arena.allocate<float>(d_model * d_ff);
    d_b_up = weights_arena.allocate<float>(d_ff);

    // 2. Down Projection Weights
    d_W_down = weights_arena.allocate<float>(d_ff * d_model);
    d_b_down = weights_arena.allocate<float>(d_model);

    // 3. FP16 weight copies for decode path (halves memory bandwidth)
    cudaMalloc(&d_W_up_fp16, d_model * d_ff * sizeof(half));
    cudaMalloc(&d_W_down_fp16, d_ff * d_model * sizeof(half));

    printf("[FeedForward] Initialized (Input: %d -> Hidden: %d -> Output: %d)\n", d_model, d_ff, d_model);
}

// --- Load Weights ---
void FeedForward::load_weights(const float* h_W_up, const float* h_b_up, 
                               const float* h_W_down, const float* h_b_down) 
{
    // Copy FP32 weights to device
    cudaMemcpy(d_W_up, h_W_up, d_model * d_ff * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_up, h_b_up, d_ff * sizeof(float),           cudaMemcpyHostToDevice);
    cudaMemcpy(d_W_down, h_W_down, d_ff * d_model * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_down, h_b_down, d_model * sizeof(float),        cudaMemcpyHostToDevice);

    // Convert to FP16 for decode path
    kernels::launch_convert_fp32_to_fp16(d_W_up, d_W_up_fp16, d_model * d_ff);
    kernels::launch_convert_fp32_to_fp16(d_W_down, d_W_down_fp16, d_ff * d_model);
    cudaDeviceSynchronize();
}

// --- Forward Pass ---
void FeedForward::forward_impl(const float* d_input, float* d_output, int batch_size, int seq_len, GPUMemoryArena* inference_arena, cudaStream_t stream, IKVCache* /*kv_cache*/) 
{
    // Total number of tokens to process
    int total_rows = batch_size * seq_len;

    // --- Step 1: Up Projection (Expand) ---
    // Input [Rows, d_model] * W_up [d_model, d_ff] -> Hidden [Rows, d_ff]
    
    float* d_hidden = inference_arena->allocate<float>(total_rows * d_ff);

    kernels::launch_gemm_tiled(
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
    
    kernels::launch_gemm_tiled(
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

// --- Decode-Fused Forward: up=GEMV+bias+GELU, down=GEMV+bias+residual ---
void FeedForward::forward_decode_fused(const float* d_input, float* d_output, int batch_size,
                                        const float* d_residual, GPUMemoryArena* inference_arena,
                                        cudaStream_t stream)
{
    // Up Projection with FP16 weights: hidden = GELU(input × W_up_fp16 + b_up)
    // FP16 GEMV halves weight bandwidth (dominant bottleneck)
    float* d_hidden = inference_arena->allocate<float>(batch_size * d_ff);
    kernels::launch_gemv_fp16(d_input, d_W_up_fp16, d_hidden, d_ff, d_model, d_b_up, stream);
    kernels::launch_gelu(d_hidden, d_ff, stream);  // bias already added by FP16 GEMV

    // Down Projection with FP16 weights: output = hidden × W_down_fp16 + b_down + residual
    kernels::launch_gemv_fp16(d_hidden, d_W_down_fp16, d_output, d_model, d_ff, d_b_down, stream);
    kernels::launch_addition(d_output, d_residual, d_output, d_model, stream);
}

size_t FeedForward::estimate_weight_memory(int d_model, int d_ff) {
    return (d_model * d_ff + d_ff + d_ff * d_model + d_model) * sizeof(float);
}

size_t FeedForward::estimate_inference_scratch(int max_batch_size, int max_seq_len, int d_ff) {
    return (max_batch_size * max_seq_len * d_ff) * sizeof(float);
}