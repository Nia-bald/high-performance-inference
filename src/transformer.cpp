#include "transformer.h"
#include "kv_cache/kv_cache.h"
#include <cstdio>

// --- Constructor ---
Transformer::Transformer(int vocab_size, int max_seq_len, int d_model, int num_heads, int num_layers, int d_ff, 
                         GPUMemoryArena& weights_arena)
    : vocab_size(vocab_size), max_seq_len(max_seq_len), d_model(d_model), 
      num_heads(num_heads), num_layers(num_layers), final_norm(d_model, weights_arena)
{
    // 1. Allocate Weights
    d_token_embedding_table = weights_arena.allocate<float>(vocab_size * d_model);
    printf("[Transformer] Allocated token embeddings: %.2f MB used, %.2f%% full\n", 
           weights_arena.get_used() / (1024.0 * 1024.0), weights_arena.get_usage_percent());
    
    d_pos_embedding_table   = weights_arena.allocate<float>(max_seq_len * d_model);
    printf("[Transformer] Allocated position embeddings: %.2f MB used, %.2f%% full\n", 
           weights_arena.get_used() / (1024.0 * 1024.0), weights_arena.get_usage_percent());
    
    // Pad vocab_size to next multiple of 4 for float4-vectorized GEMV
    // 50257 → 50260 enables 4× bandwidth improvement on LM head projection
    vocab_size_padded = (vocab_size + 3) & ~3;
    d_lm_head               = weights_arena.allocate<float>(d_model * vocab_size_padded);
    // Zero the entire padded allocation so padded columns never win argmax
    cudaMemset(d_lm_head, 0, d_model * vocab_size_padded * sizeof(float));
    // FP16 copy for decode path
    cudaMalloc(&d_lm_head_fp16, d_model * vocab_size_padded * sizeof(half));
    cudaMemset(d_lm_head_fp16, 0, d_model * vocab_size_padded * sizeof(half));
    printf("[Transformer] Allocated LM head (padded %d→%d): %.2f MB used, %.2f%% full\n", 
           vocab_size, vocab_size_padded,
           weights_arena.get_used() / (1024.0 * 1024.0), weights_arena.get_usage_percent());

    // 2. Create Blocks
    for (int i = 0; i < num_layers; ++i) {
        layers.push_back(new TransformerBlock(d_model, num_heads, d_ff, i, weights_arena));
        if ((i + 1) % 3 == 0 || i == num_layers - 1) {
            printf("[Transformer] Created %d/%d blocks: %.2f MB used, %.2f%% full\n", 
                   i + 1, num_layers, weights_arena.get_used() / (1024.0 * 1024.0), 
                   weights_arena.get_usage_percent());
        }
    }

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
    size_t state_size = current_batch_size * current_seq_len * d_model;

    // 1. Allocate Two Buffers (Ping-Pong)
    float* d_buffer_1 = inference_arena.allocate<float>(state_size); // Initial State
    float* d_buffer_2 = inference_arena.allocate<float>(state_size); // Scratchpad

    int start_pos = (kv_cache != nullptr) ? kv_cache->current_pos() : 0;

    // 2. Embeddings -> Write to Buffer 1
    // d_buffer_1 now holds the initial embedding state
    kernels::launch_embedding_lookup(
    d_token_ids, d_token_embedding_table, d_pos_embedding_table, 
    d_buffer_1, current_batch_size, current_seq_len, d_model, stream, start_pos
    );

    // Pointers that we will swap
    float* d_in  = d_buffer_1;
    float* d_out = d_buffer_2;

    // 3. Layers Loop (Ping-Pong)
    for (int i = 0; i < num_layers; ++i) {

    // Run Layer: Read from d_in, Write to d_out — pass kv_cache through
    layers[i]->forward(d_in, d_out, current_batch_size, current_seq_len, &inference_arena, stream, kv_cache);

    // Optimization: Reset the arena to free specific "intra-layer" scratch memory 
    // (like Q, K, V projections) that isn't needed for the next layer.
    // *Note: This requires careful Arena management (stack-based reset).*

    // Swap Pointers
    // The output of this layer becomes the input of the next
    std::swap(d_in, d_out);
    }

    // 4. Final Norm
    // Note: After the loop, 'd_in' holds the valid result of the last layer
    // We can write the normalized output to 'd_out' (reusing Buffer 2)
    final_norm.forward(d_in, d_out, current_batch_size, current_seq_len, nullptr, stream);

    // 5. Head — FP16 GEMV for decode (M=1), tiled GEMM for prefill (M>1)
    int M = current_batch_size * current_seq_len;
    if (M == 1) {
        kernels::launch_gemv_fp16(d_out, d_lm_head_fp16, d_logits, vocab_size_padded, d_model, nullptr, stream);
    } else {
        kernels::launch_gemm_tiled(
            d_out, d_lm_head, d_logits, 
            M, vocab_size_padded, d_model, stream);
    }
}
// --- Helpers ---
void Transformer::load_embeddings(const float* h_token, const float* h_pos) {
    cudaMemcpy(d_token_embedding_table, h_token, vocab_size * d_model * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos_embedding_table,   h_pos,   max_seq_len * d_model * sizeof(float), cudaMemcpyHostToDevice);
}
void Transformer::load_head(const float* h_head) {
    // Load into padded buffer — original data fills [d_model, vocab_size],
    // padded columns [vocab_size..vocab_size_padded-1] stay zeroed.
    if (vocab_size_padded > vocab_size) {
        for (int row = 0; row < d_model; ++row) {
            cudaMemcpy(d_lm_head + row * vocab_size_padded,
                       h_head + row * vocab_size,
                       vocab_size * sizeof(float), cudaMemcpyHostToDevice);
        }
    } else {
        cudaMemcpy(d_lm_head, h_head, d_model * vocab_size * sizeof(float), cudaMemcpyHostToDevice);
    }

    // Convert to FP16 for decode path
    kernels::launch_convert_fp32_to_fp16(d_lm_head, d_lm_head_fp16, d_model * vocab_size_padded);
    cudaDeviceSynchronize();
}

size_t Transformer::estimate_weight_memory(int vocab_size, int max_seq_len, int d_model, int num_heads, int num_layers, int d_ff) {
    int vocab_size_padded = (vocab_size + 3) & ~3;
    size_t total = 0;
    total += vocab_size * d_model * sizeof(float);  // d_token_embedding_table
    total += max_seq_len * d_model * sizeof(float); // d_pos_embedding_table
    total += d_model * vocab_size_padded * sizeof(float);  // d_lm_head (padded)

    total += num_layers * TransformerBlock::estimate_weight_memory(d_model, num_heads, d_ff);
    total += LayerNorm::estimate_weight_memory(d_model); // final_norm
    return total;
}

size_t Transformer::estimate_inference_scratch(int max_batch_size, int max_seq_len, int d_model, int num_heads, int num_layers, int d_ff) {
    size_t state_size = max_batch_size * max_seq_len * d_model;
    size_t total = 0;
    total += 2 * state_size * sizeof(float); // d_buffer_1, d_buffer_2
    
    // Accumulate for all layers since we don't reset arena inside Transformer::forward
    total += num_layers * TransformerBlock::estimate_inference_scratch(max_batch_size, max_seq_len, d_model, num_heads, d_ff);
    return total;
}