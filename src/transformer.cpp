#include "transformer.h"
#include "kv_cache/kv_cache.h"
#include <cstdio>
#include <cuda_fp16.h>
#include <vector>

// --- Constructor ---
Transformer::Transformer(int vocab_size, int max_seq_len, int d_model, int num_heads, int num_layers, int d_ff, 
                         GPUMemoryArena& weights_arena)
    : vocab_size(vocab_size), max_seq_len(max_seq_len), d_model(d_model), 
      num_heads(num_heads), num_layers(num_layers),
      backbone(d_model, num_heads, num_layers, d_ff, weights_arena)
{
    // Pad vocab for FP16 vectorized LM head GEMV (N must be divisible by 8)
    padded_vocab_size = (vocab_size + 7) & ~7;  // Round up to next multiple of 8
    printf("[Transformer] Vocab padding: %d -> %d (for FP16 half2 vectorization)\n", vocab_size, padded_vocab_size);

    // 1. Allocate Weights
    d_token_embedding_table = weights_arena.allocate<float>(vocab_size * d_model);
    printf("[Transformer] Allocated token embeddings: %.2f MB used, %.2f%% full\n", 
           weights_arena.get_used() / (1024.0 * 1024.0), weights_arena.get_usage_percent());
    
    d_pos_embedding_table   = weights_arena.allocate<float>(max_seq_len * d_model);
    printf("[Transformer] Allocated position embeddings: %.2f MB used, %.2f%% full\n", 
           weights_arena.get_used() / (1024.0 * 1024.0), weights_arena.get_usage_percent());
    
    d_lm_head               = weights_arena.allocate<__half>(d_model * padded_vocab_size);
    printf("[Transformer] Allocated LM head (FP16): %.2f MB used, %.2f%% full\n", 
           weights_arena.get_used() / (1024.0 * 1024.0), weights_arena.get_usage_percent());

    printf("[Transformer] Initialized: L=%d, H=%d, D=%d, Vocab=%d\n", num_layers, num_heads, d_model, vocab_size);
    printf("[Transformer] Final memory usage: %.2f MB / %.2f MB (%.2f%%)\n", 
           weights_arena.get_used() / (1024.0 * 1024.0), 
           weights_arena.get_total() / (1024.0 * 1024.0),
           weights_arena.get_usage_percent());
}

// --- Forward Pass ---
void Transformer::forward(const int* d_token_ids, float* d_logits, 
    int current_batch_size, int current_seq_len,
    GPUMemoryArena& inference_arena, cudaStream_t stream, IKVCache* kv_cache) 
{
    size_t state_size = static_cast<size_t>(current_batch_size) * current_seq_len * d_model;

    // 1. Allocate Buffer for Embeddings
    float* d_embeddings = inference_arena.allocate<float>(state_size);

    int start_pos = (kv_cache != nullptr) ? kv_cache->current_pos() : 0;

    // 2. Embeddings -> Write to d_embeddings
    kernels::launch_embedding_lookup(
        d_token_ids, d_token_embedding_table, d_pos_embedding_table, 
        d_embeddings, current_batch_size, current_seq_len, d_model, stream, start_pos
    );

    // 3. Backbone forward pass: [B, S, D] -> [B, S, D]
    backbone.forward(
        d_embeddings, d_embeddings, current_batch_size, current_seq_len, 
        inference_arena, stream, kv_cache
    );

    // 4. Head — FP16 weight GEMM with padded vocab (auto-dispatches to FP16 GEMV for M=1 decode)
    kernels::launch_gemm_tiled_fp16w(
        d_embeddings, d_lm_head, d_logits, 
        current_batch_size * current_seq_len, padded_vocab_size, d_model, stream
    );
}

// --- Helpers ---
void Transformer::load_embeddings(const float* h_token, const float* h_pos) {
    cudaMemcpy(d_token_embedding_table, h_token, vocab_size * d_model * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos_embedding_table,   h_pos,   max_seq_len * d_model * sizeof(float), cudaMemcpyHostToDevice);
}

void Transformer::load_head(const float* h_head) {
    // Convert FP32 -> FP16, zero-padding extra columns to padded_vocab_size
    size_t padded_count = d_model * padded_vocab_size;
    std::vector<__half> h_head_fp16(padded_count, __float2half(0.0f));  // zero-padded
    for (int k = 0; k < d_model; ++k) {
        for (int v = 0; v < vocab_size; ++v) {
            h_head_fp16[k * padded_vocab_size + v] = __float2half(h_head[k * vocab_size + v]);
        }
        // Columns [vocab_size, padded_vocab_size) remain zero
    }
    cudaMemcpy(d_lm_head, h_head_fp16.data(), padded_count * sizeof(__half), cudaMemcpyHostToDevice);
}

size_t Transformer::estimate_weight_memory(int vocab_size, int max_seq_len, int d_model, int num_heads, int num_layers, int d_ff) {
    size_t total = 0;
    total += vocab_size * d_model * sizeof(float);  // d_token_embedding_table (FP32)
    total += max_seq_len * d_model * sizeof(float); // d_pos_embedding_table (FP32)
    total += d_model * ((vocab_size + 7) & ~7) * sizeof(__half);  // d_lm_head (FP16, padded)
    total += TransformerBackbone::estimate_weight_memory(d_model, num_heads, num_layers, d_ff);
    return total;
}

size_t Transformer::estimate_inference_scratch(int max_batch_size, int max_seq_len, int d_model, int num_heads, int num_layers, int d_ff) {
    size_t state_size = static_cast<size_t>(max_batch_size) * max_seq_len * d_model;
    size_t total = 0;
    total += state_size * sizeof(float); // d_embeddings
    total += TransformerBackbone::estimate_inference_scratch(max_batch_size, max_seq_len, d_model, num_heads, num_layers, d_ff);
    return total;
}
