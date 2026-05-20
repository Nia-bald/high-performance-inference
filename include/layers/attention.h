#pragma once
#include "kernels.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "memory.h"

#include "layers/layer.h"

class SelfAttention : public Layer {

public:
    SelfAttention(int d_model, int num_heads, int layer_index, GPUMemoryArena& weights_arena,
            int qk_dim = 0, int v_dim = 0);
    ~SelfAttention() = default;
    void forward_impl(const float* d_input, float* d_output, int batch_size, int seq_len, GPUMemoryArena* inference_arena, cudaStream_t stream, IKVCache* kv_cache = nullptr) override;

    // Static Estimators
    static size_t estimate_weight_memory(int d_model, int num_heads, int qk_dim = 0, int v_dim = 0);
    static size_t estimate_inference_scratch(int max_batch_size, int max_seq_len, int d_model, int num_heads, int qk_dim = 0, int v_dim = 0);

    // helper function for testing
    // during end to end run weights will be loaded from .bin
    void load_weights(const float* h_W_q, const float* h_W_k, const float* h_W_v, const float* h_W_o,
                      const float* h_b_q, const float* h_b_k, const float* h_b_v, const float* h_b_o);

private:
    int layer_index_;
    int d_model;
    int num_heads;
    int head_dim_qk;
    int head_dim_v;

    int total_qk_dim;
    int total_v_dim;

    // FP16 weight storage (halves VRAM bandwidth for memory-bound GEMV)
    __half *d_W_qkv;  // [d_model, 3*total_qk_dim] — fused Q/K/V projection (FP16)
    __half *d_W_o;    // [total_v_dim, d_model]      — output projection (FP16)
    // Biases remain FP32 (tiny, no bandwidth benefit from FP16)
    float *d_b_qkv;  // [3*total_qk_dim]           — fused Q/K/V bias
    float *d_b_o;    // [d_model]                   — output bias
};