#include "transformer.h"
#include <cstdio>

TransformerBlock::TransformerBlock(int d_model, int num_heads, int d_ff, 
                                   int layer_index, GPUMemoryArena& weights_arena)
    : Layer("TransformerBlock"),
      d_model(d_model),
      attention_norm(d_model, weights_arena),
      attention(d_model, num_heads, layer_index, weights_arena),
      ffn_norm(d_model, weights_arena),
      feed_forward(d_model, d_ff, weights_arena)
{
    printf("[TransformerBlock] Initialized Block %d (D:%d)\n", layer_index, d_model);
}

void TransformerBlock::forward_impl(const float* d_input, float* d_output, int batch_size, int seq_len, GPUMemoryArena* inference_arena, cudaStream_t stream, IKVCache* kv_cache) 
{
    size_t tensor_size = batch_size * seq_len * d_model;

    // --- 1. Attention Path ---

    // A. Layer Norm 1
    float* d_norm1_out = inference_arena->allocate<float>(tensor_size);
    attention_norm.forward(d_input, d_norm1_out, batch_size, seq_len, nullptr, stream);

    // B. Attention — pass kv_cache (only layer that uses it)
    float* d_attn_out = inference_arena->allocate<float>(tensor_size);
    attention.forward(d_norm1_out, d_attn_out, batch_size, seq_len, inference_arena, stream, kv_cache);

    // C. Residual 1 (Input + Attn_Out)
    // Using YOUR kernel
    float* d_res1 = inference_arena->allocate<float>(tensor_size);
    kernels::launch_addition(
        d_input,      // A
        d_attn_out,   // B
        d_res1,       // C
        tensor_size,  // Length
        stream
    );


    // --- 2. FFN Path ---

    // A. Layer Norm 2
    float* d_norm2_out = inference_arena->allocate<float>(tensor_size);
    ffn_norm.forward(d_res1, d_norm2_out, batch_size, seq_len, nullptr, stream);

    // B. FFN
    float* d_ffn_out = inference_arena->allocate<float>(tensor_size);
    feed_forward.forward(d_norm2_out, d_ffn_out, batch_size, seq_len, inference_arena, stream);

    // C. Residual 2 (Res1 + FFN_Out) -> Output
    // Using YOUR kernel
    kernels::launch_addition(
        d_res1,       // A
        d_ffn_out,    // B
        d_output,     // C (Final Output)
        tensor_size,  // Length
        stream
    );
}

size_t TransformerBlock::estimate_weight_memory(int d_model, int num_heads, int d_ff) {
    return LayerNorm::estimate_weight_memory(d_model) +
           SelfAttention::estimate_weight_memory(d_model, num_heads) +
           LayerNorm::estimate_weight_memory(d_model) +
           FeedForward::estimate_weight_memory(d_model, d_ff);
}

size_t TransformerBlock::estimate_inference_scratch(int max_batch_size, int max_seq_len, int d_model, int num_heads, int d_ff) {
    size_t tensor_size = max_batch_size * max_seq_len * d_model;
    size_t block_scratch = 0;
    
    block_scratch += tensor_size * sizeof(float); // d_norm1_out
    block_scratch += tensor_size * sizeof(float); // d_attn_out
    block_scratch += SelfAttention::estimate_inference_scratch(max_batch_size, max_seq_len, d_model, num_heads);
    block_scratch += tensor_size * sizeof(float); // d_res1
    
    block_scratch += tensor_size * sizeof(float); // d_norm2_out
    block_scratch += tensor_size * sizeof(float); // d_ffn_out
    block_scratch += FeedForward::estimate_inference_scratch(max_batch_size, max_seq_len, d_ff);
    
    return block_scratch;
}