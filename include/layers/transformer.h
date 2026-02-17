#pragma once
#include <vector>
#include "memory.h"
#include "layers/attention.h"
#include "kernels.cuh"

// --- 1. LayerNorm ---
// Uses your optimized Warp-Intrinsic kernel.
// Parameters: Gamma (Scale) and Beta (Shift)
class LayerNorm {
public:
    LayerNorm(int d_model, float epsilon, GPUMemoryArena& weights_arena);
    ~LayerNorm() = default;

    // input: [Batch, Seq, d_model]
    void forward(const float* d_input, float* d_output, int batch_size, int seq_len, cudaStream_t stream);

    // Helper to load weights from host
    void load_weights(const float* h_gamma, const float* h_beta);

private:
    int d_model;
    float epsilon;
    
    // Learnable Parameters
    float* d_gamma; // [d_model]
    float* d_beta;  // [d_model]
};

// --- 2. Feed Forward Network (FFN) ---
// Standard MLP: Input -> UpProj (Expansion) -> GELU -> DownProj (Contraction) -> Output
class FeedForward {
public:
    // d_ff is usually 4 * d_model
    FeedForward(int d_model, int d_ff, GPUMemoryArena& weights_arena);
    ~FeedForward() = default;

    void forward(const float* d_input, float* d_output, GPUMemoryArena& inference_arena, 
                 int batch_size, int seq_len, cudaStream_t stream);

    void load_weights(const float* h_W_up, const float* h_b_up, 
                      const float* h_W_down, const float* h_b_down);

private:
    int d_model;
    int d_ff; 
    
    // Weights
    float* d_W_up;    // [d_model, d_ff]
    float* d_b_up;    // [d_ff] (Bias for first layer)
    
    float* d_W_down;  // [d_ff, d_model]
    float* d_b_down;  // [d_model] (Bias for second layer)
};

// --- 3. Transformer Block ---
// The Repeating Unit
// Structure (Pre-Norm):
//   1. Residual_1 = x + Attention(LayerNorm(x))
//   2. Output     = Residual_1 + FFN(LayerNorm(Residual_1))
class TransformerBlock {
public:
    TransformerBlock(int batch_size, int seq_len, int d_model, int num_heads, int d_ff, 
                     GPUMemoryArena& weights_arena);
    ~TransformerBlock() = default;

    void forward(const float* d_input, float* d_output, GPUMemoryArena& inference_arena, cudaStream_t stream);

    // Helper to access sub-layers for weight loading
    LayerNorm& get_attn_norm() { return attention_norm; }
    SelfAttention& get_attention() { return attention; }
    LayerNorm& get_ffn_norm() { return ffn_norm; }
    FeedForward& get_ffn() { return feed_forward; }

private:
    int d_model;
    int seq_len;
    int batch_size;

    // Sub-Layers
    LayerNorm attention_norm;
    SelfAttention attention;
    
    LayerNorm ffn_norm;
    FeedForward feed_forward;
};

// --- 4. The Transformer (Main Engine) ---
// Orchestrates the entire forward pass
class Transformer {
public:
    Transformer(int vocab_size, int max_seq_len, int d_model, int num_heads, int num_layers, int d_ff, 
                GPUMemoryArena& weights_arena);
    ~Transformer() = default;

    // Main Inference Function
    // Input: d_token_ids [Batch, Seq] (Integers)
    // Output: d_logits [Batch, Seq, Vocab] (Floats)
    void forward(const int* d_token_ids, float* d_logits, GPUMemoryArena& inference_arena, cudaStream_t stream);

    // Accessors for weight loading
    TransformerBlock* get_block(int i) { return layers[i]; }
    LayerNorm& get_final_norm() { return final_norm; }
    
    // Load embeddings/head
    void load_embeddings(const float* h_token_embed, const float* h_pos_embed);
    void load_head(const float* h_lm_head);

private:
    int d_model;
    int max_seq_len;
    int vocab_size;
    int num_layers;
    int batch_size; // Needed for internal sizing

    // Embeddings
    float* d_token_embedding_table; // [Vocab, d_model]
    float* d_pos_embedding_table;   // [MaxSeq, d_model]

    // Stack of Blocks
    std::vector<TransformerBlock*> layers; 
    
    // Final Layers
    LayerNorm final_norm;
    float* d_lm_head; // [d_model, Vocab]
};