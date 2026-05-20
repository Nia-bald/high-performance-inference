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

    // ===================================================================
    // DECODE PATH: seq_len==1 && kv_cache available
    // Fuses residual additions into GEMV output writes to save 2 kernel
    // launches per block (24 launches saved per decode step total).
    // ===================================================================
    if (kv_cache != nullptr && seq_len == 1) {
        // --- 1. Attention Path ---
        float* d_norm1_out = inference_arena->allocate<float>(tensor_size);
        attention_norm.forward(d_input, d_norm1_out, batch_size, seq_len, nullptr, stream);

        // Attention output — fuse residual addition into output projection
        // Instead of: attn_out = O_proj(context) + bias; res1 = input + attn_out
        // We do:      res1 = O_proj(context) + bias + input  (single fused GEMV kernel)
        float* d_res1 = inference_arena->allocate<float>(tensor_size);
        attention.forward_decode_fused(d_norm1_out, d_res1, batch_size, d_input, inference_arena, stream, kv_cache);

        // --- 2. FFN Path ---
        float* d_norm2_out = inference_arena->allocate<float>(tensor_size);
        ffn_norm.forward(d_res1, d_norm2_out, batch_size, seq_len, nullptr, stream);

        // FFN output — fuse residual addition into down projection
        // Instead of: ffn_out = down_proj(hidden) + bias; output = res1 + ffn_out
        // We do:      output = down_proj(hidden) + bias + res1  (single fused GEMV kernel)
        feed_forward.forward_decode_fused(d_norm2_out, d_output, batch_size, d_res1, inference_arena, stream);

        return;
    }

    // ===================================================================
    // PREFILL PATH (unchanged)
    // ===================================================================

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