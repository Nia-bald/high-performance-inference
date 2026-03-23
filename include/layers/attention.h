#pragma once
#include "kernels.cuh"
#include <cuda_runtime.h>
#include "memory.h"

class SelfAttention {

public:
    SelfAttention(int d_model, int num_heads, GPUMemoryArena& weights_arena,
            int qk_dim = 0, int v_dim = 0);
    ~SelfAttention() = default;
    void forward(int batch_size, int seq_len, const float* d_input, float* d_output, GPUMemoryArena& inference_arena, cudaStream_t stream = 0);

    // Static Estimators
    static size_t estimate_weight_memory(int d_model, int num_heads, int qk_dim = 0, int v_dim = 0);
    static size_t estimate_inference_scratch(int max_batch_size, int max_seq_len, int d_model, int num_heads, int qk_dim = 0, int v_dim = 0);

    // helper function for testing
    // during end to end run weights will be loaded from .bin
    void load_weights(const float* h_W_q, const float* h_W_k, const float* h_W_v, const float* h_W_o,
                      const float* h_b_q, const float* h_b_k, const float* h_b_v, const float* h_b_o);

private:

    int d_model;
    int num_heads;
    int head_dim_qk;
    int head_dim_v;

    int total_qk_dim;
    int total_v_dim;

    // views into weights memory arena, will be same across users
    float *d_W_q, *d_W_k, *d_W_v, *d_W_o;
    float *d_b_q, *d_b_k, *d_b_v, *d_b_o;
    // float *d_Q, *d_K, *d_V; removed these cause these are owned by arena, idea is same attention object we can use to do simultaneous inferences 
    // float *d_K_transpose, *d_attention;
    // float *d_attention_heads_output;
};