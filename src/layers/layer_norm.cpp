#include "transformer.h"
#include <cstdio>

// Removed epsilon from constructor
LayerNorm::LayerNorm(int d_model, GPUMemoryArena& weights_arena)
    : d_model(d_model) 
{
    d_gamma = weights_arena.allocate<float>(d_model);
    d_beta  = weights_arena.allocate<float>(d_model);

    // No epsilon stored in class state anymore
    printf("[LayerNorm] Initialized (Dim: %d, Epsilon: Hardcoded)\n", d_model);
}

void LayerNorm::load_weights(const float* h_gamma, const float* h_beta) {
    cudaMemcpy(d_gamma, h_gamma, d_model * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta,  h_beta,  d_model * sizeof(float), cudaMemcpyHostToDevice);
}

void LayerNorm::forward(const float* d_input, float* d_output, 
                        int batch_size, int seq_len, cudaStream_t stream) {
    
    // Launch without passing epsilon
    kernels::launch_layer_norm(
        d_input, 
        d_output, 
        d_gamma, 
        d_beta, 
        batch_size, 
        seq_len, 
        d_model, 
        stream
    );
}