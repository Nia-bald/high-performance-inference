#include "transformer.h" // Assuming this header has the class definition and kernels.cuh
#include <cstdio>

// --- Constructor ---
FeedForward::FeedForward(int d_model, int d_ff, GPUMemoryArena& weights_arena)
    : d_model(d_model), d_ff(d_ff) 
{
    // 1. Up Projection Weights
    d_W_up = weights_arena.allocate<float>(d_model * d_ff);
    d_b_up = weights_arena.allocate<float>(d_ff);

    // 2. Down Projection Weights
    d_W_down = weights_arena.allocate<float>(d_ff * d_model);
    d_b_down = weights_arena.allocate<float>(d_model);

    printf("[FeedForward] Initialized (Input: %d -> Hidden: %d -> Output: %d)\n", d_model, d_ff, d_model);
}

// --- Load Weights ---
void FeedForward::load_weights(const float* h_W_up, const float* h_b_up, 
                               const float* h_W_down, const float* h_b_down) 
{
    // Simple Copy to Device
    cudaMemcpy(d_W_up, h_W_up, d_model * d_ff * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_up, h_b_up, d_ff * sizeof(float),           cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_W_down, h_W_down, d_ff * d_model * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_down, h_b_down, d_model * sizeof(float),        cudaMemcpyHostToDevice);
}

// --- Forward Pass ---
void FeedForward::forward(const float* d_input, float* d_output, GPUMemoryArena& inference_arena, 
                          int batch_size, int seq_len, cudaStream_t stream) 
{
    // Total number of tokens to process
    int total_rows = batch_size * seq_len;

    // --- Step 1: Up Projection (Expand) ---
    // Input [Rows, d_model] * W_up [d_model, d_ff] -> Hidden [Rows, d_ff]
    
    float* d_hidden = inference_arena.allocate<float>(total_rows * d_ff);

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

    // --- Step 4: Final Bias Add ---
    // Output = Output + b_down
    kernels::launch_bias_add(d_output, d_b_down, total_rows, d_model, stream);
}