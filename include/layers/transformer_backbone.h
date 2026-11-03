#pragma once
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "memory.h"
#include "layers/layer.h"
#include "layers/attention.h"
#include "kernels.cuh"

// --- 1. LayerNorm ---
// Uses optimized Warp-Intrinsic kernel.
// Parameters: Gamma (Scale) and Beta (Shift)
class LayerNorm : public Layer {
public:
    LayerNorm(int d_model, GPUMemoryArena& weights_arena);
    ~LayerNorm() = default;

    // input: [Batch, Seq, d_model]
    void forward_impl(const float* d_input, float* d_output, int batch_size, int seq_len, GPUMemoryArena* inference_arena, cudaStream_t stream, IKVCache* kv_cache = nullptr) override;

    // Static Estimators
    static size_t estimate_weight_memory(int d_model);
    static size_t estimate_inference_scratch();

    // Helper to load weights from host
    void load_weights(const float* h_gamma, const float* h_beta);

private:
    int d_model;
    
    // Learnable Parameters
    float* d_gamma; // [d_model]
    float* d_beta;  // [d_model]
};

// --- 2. Feed Forward Network (FFN) ---
// Standard MLP: Input -> UpProj (Expansion) -> GELU -> DownProj (Contraction) -> Output
class FeedForward : public Layer {
public:
    // d_ff is usually 4 * d_model
    FeedForward(int d_model, int d_ff, GPUMemoryArena& weights_arena);
    ~FeedForward() = default;

    void forward_impl(const float* d_input, float* d_output, int batch_size, int seq_len, GPUMemoryArena* inference_arena, cudaStream_t stream, IKVCache* kv_cache = nullptr) override;

    // Static Estimators
    static size_t estimate_weight_memory(int d_model, int d_ff);
    static size_t estimate_inference_scratch(int max_batch_size, int max_seq_len, int d_ff);

    void load_weights(const float* h_W_up, const float* h_b_up, 
                      const float* h_W_down, const float* h_b_down);

private:
    int d_model;
    int d_ff; 
    
    // FP16 weight storage (halves VRAM bandwidth)
    __half* d_W_up;    // [d_model, d_ff] (FP16)
    float* d_b_up;     // [d_ff] (FP32 bias for GELU fusion)
    
    __half* d_W_down;  // [d_ff, d_model] (FP16)
    float* d_b_down;   // [d_model] (FP32 bias)
};

// --- 3. Transformer Block ---
// The Repeating Unit
// Structure (Pre-Norm):
//   1. Residual_1 = x + Attention(LayerNorm(x))
//   2. Output     = Residual_1 + FFN(LayerNorm(Residual_1))
class TransformerBlock : public Layer {
public:
    TransformerBlock(int d_model, int num_heads, int d_ff, 
                     int layer_index, GPUMemoryArena& weights_arena);
    ~TransformerBlock() = default;
    void forward_impl(const float* d_input, float* d_output, int batch_size, int seq_len, GPUMemoryArena* inference_arena, cudaStream_t stream, IKVCache* kv_cache = nullptr) override;

    // Static Estimators
    static size_t estimate_weight_memory(int d_model, int num_heads, int d_ff);
    static size_t estimate_inference_scratch(int max_batch_size, int max_seq_len, int d_model, int num_heads, int d_ff);

    // Helper to access sub-layers for weight loading
    LayerNorm& get_attn_norm() { return attention_norm; }
    SelfAttention& get_attention() { return attention; }
    LayerNorm& get_ffn_norm() { return ffn_norm; }
    FeedForward& get_ffn() { return feed_forward; }

private:
    int d_model;

    // Sub-Layers
    LayerNorm attention_norm;
    SelfAttention attention;
    
    LayerNorm ffn_norm;
    FeedForward feed_forward;
};

// --- 4. Transformer Backbone ---
// Pure, embedding-agnostic backbone transforming continuous float tensors [B, S, D] -> [B, S, D].
// Encapsulates layer stacking of TransformerBlock, final LayerNorm, and ping-pong memory buffer management.
class TransformerBackbone {
public:
    TransformerBackbone(int d_model, int num_heads, int num_layers, int d_ff, 
                        GPUMemoryArena& weights_arena);
    ~TransformerBackbone();

    TransformerBackbone(const TransformerBackbone&) = delete;
    TransformerBackbone& operator=(const TransformerBackbone&) = delete;

    // Core Forward Pass: transforms [Batch, Seq, d_model] -> [Batch, Seq, d_model]
    void forward(
        const float* d_input,
        float* d_output,
        int batch_size,
        int seq_len,
        GPUMemoryArena& arena,
        cudaStream_t stream,
        IKVCache* kv_cache = nullptr
    );

    // Static Estimators
    static size_t estimate_weight_memory(int d_model, int num_heads, int num_layers, int d_ff);
    static size_t estimate_inference_scratch(int max_batch_size, int max_seq_len, int d_model, int num_heads, int num_layers, int d_ff);

    // Accessors
    TransformerBlock* get_block(int i) { return layers[i]; }
    const TransformerBlock* get_block(int i) const { return layers[i]; }
    LayerNorm& get_final_norm() { return final_norm; }
    const LayerNorm& get_final_norm() const { return final_norm; }

    int get_d_model() const { return d_model; }
    int get_num_heads() const { return num_heads; }
    int get_num_layers() const { return num_layers; }
    int get_d_ff() const { return d_ff; }

private:
    int d_model;
    int num_heads;
    int num_layers;
    int d_ff;

    std::vector<TransformerBlock*> layers;
    LayerNorm final_norm;
};
