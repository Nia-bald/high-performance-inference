#pragma once
#include <vector>
#include "layers/transformer_backbone.h"
#include "kernels.cuh"

// --- 4. The Transformer (Main Engine) ---
// Orchestrates the entire forward pass for autoregressive language models (e.g. GPT-2).
// Composes token and position embeddings, TransformerBackbone, and the vocabulary LM head.
class Transformer {
public:
    Transformer(int vocab_size, int max_seq_len, int d_model, int num_heads, int num_layers, int d_ff, 
                GPUMemoryArena& weights_arena);
    ~Transformer() = default;

    // Main Inference Function
    // Input: d_token_ids [Batch, Seq] (Integers)
    // Output: d_logits [Batch, Seq, Vocab] (Floats)
    void forward(const int* d_token_ids, float* d_logits, int current_batch_size, int current_seq_len, GPUMemoryArena& inference_arena, cudaStream_t stream, IKVCache* kv_cache = nullptr);

    // Static Estimators
    static size_t estimate_weight_memory(int vocab_size, int max_seq_len, int d_model, int num_heads, int num_layers, int d_ff);
    static size_t estimate_inference_scratch(int max_batch_size, int max_seq_len, int d_model, int num_heads, int num_layers, int d_ff);

    // Accessors for weight loading and sublayer inspection
    TransformerBlock* get_block(int i) { return backbone.get_block(i); }
    const TransformerBlock* get_block(int i) const { return backbone.get_block(i); }
    LayerNorm& get_final_norm() { return backbone.get_final_norm(); }
    const LayerNorm& get_final_norm() const { return backbone.get_final_norm(); }
    TransformerBackbone& get_backbone() { return backbone; }
    const TransformerBackbone& get_backbone() const { return backbone; }
    
    // Load embeddings/head
    void load_embeddings(const float* h_token_embed, const float* h_pos_embed);
    void load_head(const float* h_lm_head);

    // Getters
    int get_vocab_size() const { return vocab_size; }
    int get_padded_vocab_size() const { return padded_vocab_size; }
    int get_max_seq_len() const { return max_seq_len; }
    int get_d_model() const { return d_model; }
    int get_num_layers() const { return num_layers; }
    int get_num_heads() const { return num_heads; }

private:
    int d_model;
    int max_seq_len;
    int vocab_size;
    int padded_vocab_size;  // vocab_size rounded up to multiple of 8 for FP16 GEMV
    int num_layers;
    int num_heads;

    // Embeddings (FP32 — lookup, not matmul)
    float* d_token_embedding_table; // [Vocab, d_model]
    float* d_pos_embedding_table;   // [MaxSeq, d_model]

    // Composed backbone (repeating blocks + final LayerNorm)
    TransformerBackbone backbone;

    // LM Head
    __half* d_lm_head; // [d_model, Vocab] (FP16 weights)
};
