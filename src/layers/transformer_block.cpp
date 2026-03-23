#include "transformer.h"
#include <cstdio>

TransformerBlock::TransformerBlock(int d_model, int num_heads, int d_ff, 
                                   GPUMemoryArena& weights_arena)
    : d_model(d_model),
      attention_norm(d_model, weights_arena), // No epsilon needed now
      attention(d_model, num_heads, weights_arena),
      ffn_norm(d_model, weights_arena),
      feed_forward(d_model, d_ff, weights_arena)
{
    printf("[TransformerBlock] Initialized Block (D:%d)\n", d_model);
}

void TransformerBlock::forward(int current_batch_size, int current_seq_len, const float* d_input, float* d_output, 
                               GPUMemoryArena& inference_arena, cudaStream_t stream) 
{
    size_t tensor_size = current_batch_size * current_seq_len * d_model;

    // --- 1. Attention Path ---

    // A. Layer Norm 1
    float* d_norm1_out = inference_arena.allocate<float>(tensor_size);
    attention_norm.forward(d_input, d_norm1_out, current_batch_size, current_seq_len, stream);

    // B. Attention
    float* d_attn_out = inference_arena.allocate<float>(tensor_size);
    attention.forward(current_batch_size, current_seq_len, d_norm1_out, d_attn_out, inference_arena, stream);

    // C. Residual 1 (Input + Attn_Out)
    // Using YOUR kernel
    float* d_res1 = inference_arena.allocate<float>(tensor_size);
    kernels::launch_addition(
        d_input,      // A
        d_attn_out,   // B
        d_res1,       // C
        tensor_size,  // Length
        stream
    );


    // --- 2. FFN Path ---

    // A. Layer Norm 2
    float* d_norm2_out = inference_arena.allocate<float>(tensor_size);
    ffn_norm.forward(d_res1, d_norm2_out, current_batch_size, current_seq_len, stream);

    // B. FFN
    float* d_ffn_out = inference_arena.allocate<float>(tensor_size);
    feed_forward.forward(d_norm2_out, d_ffn_out, inference_arena, current_batch_size, current_seq_len, stream);

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