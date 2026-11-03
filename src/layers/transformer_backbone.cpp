#include "layers/transformer_backbone.h"
#include <cstdio>
#include <utility>

TransformerBackbone::TransformerBackbone(int d_model, int num_heads, int num_layers, int d_ff, 
                                         GPUMemoryArena& weights_arena)
    : d_model(d_model), num_heads(num_heads), num_layers(num_layers), d_ff(d_ff),
      final_norm(d_model, weights_arena)
{
    for (int i = 0; i < num_layers; ++i) {
        layers.push_back(new TransformerBlock(d_model, num_heads, d_ff, i, weights_arena));
        if ((i + 1) % 3 == 0 || i == num_layers - 1) {
            printf("[TransformerBackbone] Created %d/%d blocks: %.2f MB used, %.2f%% full\n", 
                   i + 1, num_layers, weights_arena.get_used() / (1024.0 * 1024.0), 
                   weights_arena.get_usage_percent());
        }
    }
    printf("[TransformerBackbone] Initialized: L=%d, H=%d, D=%d, FF=%d\n", num_layers, num_heads, d_model, d_ff);
}

TransformerBackbone::~TransformerBackbone() {
    for (auto* block : layers) {
        delete block;
    }
    layers.clear();
}

void TransformerBackbone::forward(
    const float* d_input,
    float* d_output,
    int batch_size,
    int seq_len,
    GPUMemoryArena& arena,
    cudaStream_t stream,
    IKVCache* kv_cache)
{
    if (num_layers == 0) {
        final_norm.forward(d_input, d_output, batch_size, seq_len, nullptr, stream);
        return;
    }

    size_t state_size = static_cast<size_t>(batch_size) * seq_len * d_model;

    // Allocate ping-pong buffers from inference arena
    float* d_buf1 = arena.allocate<float>(state_size);
    float* d_buf2 = (num_layers > 1) ? arena.allocate<float>(state_size) : nullptr;

    // Layer 0: Read from d_input, write to d_buf1
    layers[0]->forward(d_input, d_buf1, batch_size, seq_len, &arena, stream, kv_cache);

    float* d_in = d_buf1;
    float* d_out = d_buf2;

    // Remaining layers ping-pong between d_buf1 and d_buf2
    for (int i = 1; i < num_layers; ++i) {
        layers[i]->forward(d_in, d_out, batch_size, seq_len, &arena, stream, kv_cache);
        std::swap(d_in, d_out);
    }

    // d_in holds the output of the final TransformerBlock.
    // Apply final LayerNorm and write directly into d_output.
    final_norm.forward(d_in, d_output, batch_size, seq_len, nullptr, stream);
}

size_t TransformerBackbone::estimate_weight_memory(int d_model, int num_heads, int num_layers, int d_ff) {
    size_t total = 0;
    total += num_layers * TransformerBlock::estimate_weight_memory(d_model, num_heads, d_ff);
    total += LayerNorm::estimate_weight_memory(d_model);
    return total;
}

size_t TransformerBackbone::estimate_inference_scratch(int max_batch_size, int max_seq_len, int d_model, int num_heads, int num_layers, int d_ff) {
    size_t state_size = static_cast<size_t>(max_batch_size) * max_seq_len * d_model;
    size_t total = 0;
    int num_buffers = (num_layers >= 2) ? 2 : (num_layers == 1 ? 1 : 0);
    total += num_buffers * state_size * sizeof(float);
    total += num_layers * TransformerBlock::estimate_inference_scratch(max_batch_size, max_seq_len, d_model, num_heads, d_ff);
    return total;
}
