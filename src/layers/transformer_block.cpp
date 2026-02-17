#include "transformer.h"
#include <cstdio>

TransformerBlock::TransformerBlock(int batch_size, int seq_len, int d_model, int num_heads, int d_ff, 
                                   GPUMemoryArena& weights_arena)
    : batch_size(batch_size), seq_len(seq_len), d_model(d_model),
      attention_norm(d_model, weights_arena), // No epsilon needed now
      attention(batch_size, seq_len, d_model, num_heads, weights_arena),
      ffn_norm(d_model, weights_arena),
      feed_forward(d_model, d_ff, weights_arena)
{
    printf("[TransformerBlock] Initialized Block (B:%d, S:%d, D:%d)\n", batch_size, seq_len, d_model);
}

void TransformerBlock::forward(const float* d_input, float* d_output, 
                               GPUMemoryArena& inference_arena, cudaStream_t stream) 
{
    size_t tensor_size = batch_size * seq_len * d_model;

    // --- 1. Attention Path ---

    // A. Layer Norm 1
    float* d_norm1_out = inference_arena.allocate<float>(tensor_size);
    attention_norm.forward(d_input, d_norm1_out, batch_size, seq_len, stream);

    // B. Attention
    float* d_attn_out = inference_arena.allocate<float>(tensor_size);
    attention.forward(d_norm1_out, d_attn_out, inference_arena, stream);

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
    ffn_norm.forward(d_res1, d_norm2_out, batch_size, seq_len, stream);

    // B. FFN
    float* d_ffn_out = inference_arena.allocate<float>(tensor_size);
    feed_forward.forward(d_norm2_out, d_ffn_out, inference_arena, batch_size, seq_len, stream);

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