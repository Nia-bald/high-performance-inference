#include "transformer.h"
#include <cstdio>

// --- Constructor ---
Transformer::Transformer(int vocab_size, int max_seq_len, int d_model, int num_heads, int num_layers, int d_ff, 
                         GPUMemoryArena& weights_arena)
    : vocab_size(vocab_size), max_seq_len(max_seq_len), d_model(d_model), 
      num_layers(num_layers), final_norm(d_model, weights_arena)
{
    // 1. Allocate Weights
    d_token_embedding_table = weights_arena.allocate<float>(vocab_size * d_model);
    printf("[Transformer] Allocated token embeddings: %.2f MB used, %.2f%% full\n", 
           weights_arena.get_user() / (1024.0 * 1024.0), weights_arena.get_usage_percent());
    
    d_pos_embedding_table   = weights_arena.allocate<float>(max_seq_len * d_model);
    printf("[Transformer] Allocated position embeddings: %.2f MB used, %.2f%% full\n", 
           weights_arena.get_user() / (1024.0 * 1024.0), weights_arena.get_usage_percent());
    
    d_lm_head               = weights_arena.allocate<float>(d_model * vocab_size);
    printf("[Transformer] Allocated LM head: %.2f MB used, %.2f%% full\n", 
           weights_arena.get_user() / (1024.0 * 1024.0), weights_arena.get_usage_percent());

    // 2. Create Blocks
    for (int i = 0; i < num_layers; ++i) {
        layers.push_back(new TransformerBlock(d_model, num_heads, d_ff, weights_arena));
        if ((i + 1) % 3 == 0 || i == num_layers - 1) {
            printf("[Transformer] Created %d/%d blocks: %.2f MB used, %.2f%% full\n", 
                   i + 1, num_layers, weights_arena.get_user() / (1024.0 * 1024.0), 
                   weights_arena.get_usage_percent());
        }
    }

    printf("[Transformer] Initialized: L=%d, H=%d, D=%d, Vocab=%d\n", num_layers, num_heads, d_model, vocab_size);
    printf("[Transformer] Final memory usage: %.2f MB / %.2f MB (%.2f%%)\n", 
           weights_arena.get_user() / (1024.0 * 1024.0), 
           weights_arena.get_total() / (1024.0 * 1024.0),
           weights_arena.get_usage_percent());
}

// --- Forward Pass ---
void Transformer::forward(const int* d_token_ids, float* d_logits, 
    int current_batch_size, int current_seq_len,
    GPUMemoryArena& inference_arena, cudaStream_t stream) 
{
    size_t state_size = current_batch_size * current_seq_len * d_model;

    // 1. Allocate Two Buffers (Ping-Pong)
    float* d_buffer_1 = inference_arena.allocate<float>(state_size); // Initial State
    float* d_buffer_2 = inference_arena.allocate<float>(state_size); // Scratchpad

    // 2. Embeddings -> Write to Buffer 1
    // d_buffer_1 now holds the initial embedding state
    kernels::launch_embedding_lookup(
    d_token_ids, d_token_embedding_table, d_pos_embedding_table, 
    d_buffer_1, current_batch_size, current_seq_len, d_model, stream
    );

    // Pointers that we will swap
    float* d_in  = d_buffer_1;
    float* d_out = d_buffer_2;

    // 3. Layers Loop (Ping-Pong)
    for (int i = 0; i < num_layers; ++i) {

    // Run Layer: Read from d_in, Write to d_out
    layers[i]->forward(current_batch_size, current_seq_len, d_in, d_out, inference_arena, stream);

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
    final_norm.forward(d_in, d_out, current_batch_size, current_seq_len, stream);

    // 5. Head
    kernels::launch_gemm_tiled(
    d_out, d_lm_head, d_logits, 
    current_batch_size * current_seq_len, vocab_size, d_model, stream
    );
}
// --- Helpers ---
void Transformer::load_embeddings(const float* h_token, const float* h_pos) {
    cudaMemcpy(d_token_embedding_table, h_token, vocab_size * d_model * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos_embedding_table,   h_pos,   max_seq_len * d_model * sizeof(float), cudaMemcpyHostToDevice);
}
void Transformer::load_head(const float* h_head) {
    cudaMemcpy(d_lm_head, h_head, d_model * vocab_size * sizeof(float), cudaMemcpyHostToDevice);
}