#pragma once
#include "kernels.cuh"
#include <cuda_runtime.h>
#include "memory.h"

class SelfAttention {

public:
    SelfAttention(int seq_len, int d_model, int num_heads, GPUMemoryArena& weights_arena,
            int qk_dim = 0, int v_dim = 0);
    ~SelfAttention();

    void forward(int batch_size, const float* d_input, float* d_output, GPUMemoryArena& inference_arena, cudaStream_t stream = 0);

    // helper function for testing
    // during end to end run weights will be loaded from .bin
    void load_weights(const float* h_W_q, const float* h_W_k, const float* h_W_v, const float* h_W_o);

private:

    int batch_size;
    int seq_len;
    int d_model;
    int num_heads;
    int head_dim_qk;
    int head_dim_v;

    int total_qk_dim;
    int total_v_dim;

    // views into weights memory arena, will be same across users
    float *d_W_q, *d_W_k, *d_W_v, *d_W_o;
    // float *d_Q, *d_K, *d_V; removed these cause these are owned by arena, idea is same attention object we can use to do simultaneous inferences 
    // float *d_K_transpose, *d_attention;
    // float *d_attention_heads_output;
};