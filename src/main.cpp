#include "transformer.h"
#include "memory.h"
#include "kernels.cuh"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

// Helper defined in Step 2
void load_gpt2_weights(Transformer& gpt, const std::string& path, 
    int n_layers, int d_model, int vocab_size, int max_seq, int d_ff);
    
int main() {
    // --- 1. Hyperparameters (GPT-2 Small) ---
    int vocab_size = 50257;
    int max_seq_len = 1024;
    int d_model = 768;
    int num_heads = 12;
    int num_layers = 12;
    int d_ff = 768 * 4; // 3072

    // --- 2. Initialize Memory Arenas ---
    // Calculate memory requirements for GPT-2 Small
    // Token embeddings: vocab_size * d_model * 4 bytes
    size_t token_emb_mem = vocab_size * d_model * sizeof(float);
    // Position embeddings: max_seq_len * d_model * 4 bytes
    size_t pos_emb_mem = max_seq_len * d_model * sizeof(float);
    // LM head: d_model * vocab_size * 4 bytes
    size_t lm_head_mem = d_model * vocab_size * sizeof(float);
    // Final LayerNorm: 2 * d_model * 4 bytes
    size_t final_norm_mem = 2 * d_model * sizeof(float);
    
    // Per TransformerBlock:
    // 2 LayerNorms: 2 * 2 * d_model * 4 bytes
    size_t per_block_norm_mem = 2 * 2 * d_model * sizeof(float);
    // SelfAttention: 4 * (d_model * d_model * 4 bytes) for W_q, W_k, W_v, W_o
    size_t per_block_attn_mem = 4 * d_model * d_model * sizeof(float);
    // FeedForward: (d_model * d_ff + d_ff + d_ff * d_model + d_model) * 4 bytes
    size_t per_block_ffn_mem = (d_model * d_ff + d_ff + d_ff * d_model + d_model) * sizeof(float);
    
    size_t per_block_mem = per_block_norm_mem + per_block_attn_mem + per_block_ffn_mem;
    size_t total_blocks_mem = num_layers * per_block_mem;
    
    // Total estimated memory (with alignment overhead ~5%)
    size_t estimated_mem = token_emb_mem + pos_emb_mem + lm_head_mem + final_norm_mem + total_blocks_mem;
    size_t estimated_mem_with_overhead = estimated_mem * 105 / 100; // Add 5% for alignment overhead
    
    std::cout << "[Memory] Estimated requirements:" << std::endl;
    std::cout << "  Token embeddings: " << token_emb_mem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Position embeddings: " << pos_emb_mem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  LM head: " << lm_head_mem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Final LayerNorm: " << final_norm_mem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Per block: " << per_block_mem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Total blocks (" << num_layers << "): " << total_blocks_mem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Estimated total: " << estimated_mem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Estimated with overhead: " << estimated_mem_with_overhead / (1024 * 1024) << " MB" << std::endl;
    
    // Weights: Use estimated + safety margin
    size_t weight_mem = std::max(estimated_mem_with_overhead, (size_t)(700 * 1024 * 1024)); // At least 700MB
    GPUMemoryArena weight_arena(weight_mem);

    // Inference: Calculate memory requirements
    // Persistent buffers:
    size_t input_ids_mem = max_seq_len * sizeof(int);
    size_t logits_mem = max_seq_len * vocab_size * sizeof(float); // ~192 MB for max_seq_len=1024
    size_t next_token_mem = sizeof(int);
    size_t persistent_mem = input_ids_mem + logits_mem + next_token_mem;
    
    // Per forward pass (worst case with max_seq_len):
    // Transformer buffers: 2 ping-pong buffers
    size_t transformer_buffers_mem = 2 * max_seq_len * d_model * sizeof(float);
    
    // Per TransformerBlock (worst case):
    // - TransformerBlock intermediate buffers: 5 * max_seq_len * d_model * sizeof(float)
    size_t per_block_buffers_mem = 5 * max_seq_len * d_model * sizeof(float);
    
    // - SelfAttention: Q, K, K_transpose, V, attention matrix, A_mult_V
    size_t attn_qkv_mem = 4 * max_seq_len * d_model * sizeof(float); // Q, K, K_transpose, V
    size_t attn_matrix_mem = max_seq_len * max_seq_len * num_heads * sizeof(float); // Attention scores
    size_t attn_output_mem = max_seq_len * d_model * sizeof(float); // A_mult_V
    size_t per_block_attn_forward_mem = attn_qkv_mem + attn_matrix_mem + attn_output_mem;
    
    // - FeedForward: hidden state
    size_t per_block_ffn_forward_mem = max_seq_len * d_ff * sizeof(float);
    
    size_t per_block_forward_mem = per_block_buffers_mem + per_block_attn_forward_mem + per_block_ffn_forward_mem;
    size_t total_forward_mem = transformer_buffers_mem + (num_layers * per_block_forward_mem);
    
    // Total with safety margin (20% overhead for alignment)
    size_t estimated_inf_mem = persistent_mem + total_forward_mem;
    size_t inf_mem = estimated_inf_mem * 120 / 100; // Add 20% safety margin
    
    // Minimum 500MB to handle reasonable sequence lengths
    inf_mem = std::max(inf_mem, (size_t)(500 * 1024 * 1024));
    
    std::cout << "[Memory] Inference requirements:" << std::endl;
    std::cout << "  Persistent buffers: " << persistent_mem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Forward pass (max_seq_len=" << max_seq_len << "): " << total_forward_mem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Estimated total: " << estimated_inf_mem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Allocated: " << inf_mem / (1024 * 1024) << " MB" << std::endl;
    
    GPUMemoryArena inf_arena(inf_mem);

    std::cout << ">>> Initializing Engine..." << std::endl;
    Transformer gpt(vocab_size, max_seq_len, d_model, num_heads, num_layers, d_ff, weight_arena);

    // --- 3. Load Weights ---
    load_gpt2_weights(gpt, "/home/niare/Projects/transformer_inference_engine/gpt2_weights.bin", num_layers, d_model, vocab_size, max_seq_len, d_ff);

    // --- 4. Prepare Input ---
    // "Alan Turing was a" -> [36235, 39141, 373, 257] (Example IDs)
    std::vector<int> input_ids = {36235, 39141, 373, 257};
    int max_new_tokens = 20;

    // Allocate Input/Output on GPU (persistent buffers)
    // We need a growing buffer for input_ids
    int* d_input_ids = inf_arena.allocate<int>(max_seq_len);
    float* d_logits = inf_arena.allocate<float>(max_seq_len * vocab_size); // Reused
    int* d_next_token = inf_arena.allocate<int>(1);
    
    // Track where persistent buffers end so we can reset scratch memory between iterations
    size_t persistent_offset = inf_arena.get_user();
    std::cout << "[Memory] Persistent buffers allocated: " << persistent_offset / (1024 * 1024) << " MB" << std::endl;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::cout << "\n>>> Starting Inference: 'Alan Turing was a' ..." << std::endl;

    // --- 5. Autoregressive Loop ---
    for (int step = 0; step < max_new_tokens; ++step) {
        
        int current_seq_len = input_ids.size();
        
        // Reset arena to persistent buffer boundary (free scratch memory from previous iteration)
        // This allows us to reuse the same scratch memory for each forward pass
        inf_arena.reset_to(persistent_offset);
        
        // A. Copy current sequence to GPU
        cudaMemcpyAsync(d_input_ids, input_ids.data(), current_seq_len * sizeof(int), cudaMemcpyHostToDevice, stream);

        // B. Forward Pass
        // Note: For naive implementation, we re-process the whole sequence every time (KV-Cache is Step 4 optimization!)
        gpt.forward(d_input_ids, d_logits, 1, current_seq_len, inf_arena, stream);
        
        // Synchronize to ensure forward pass completes before sampling
        cudaStreamSynchronize(stream);

        // C. Sampling (Greedy Argmax)
        // We look at the logits of the LAST token
        float* last_logits = d_logits + (current_seq_len - 1) * vocab_size;
        
        // Debug: Check logits for first few iterations
        if (step < 3) {
            std::vector<float> debug_logits(10);
            cudaMemcpy(debug_logits.data(), last_logits, 10 * sizeof(float), cudaMemcpyDeviceToHost);
            std::cout << "\n[Debug] Step " << step << " - First 10 logits: ";
            for (float l : debug_logits) std::cout << l << " ";
            std::cout << std::endl;
        }
        
        kernels::launch_argmax(last_logits, d_next_token, 1, 1, vocab_size, stream);

        // D. Get Result back to Host
        int next_token_id;
        cudaMemcpyAsync(&next_token_id, d_next_token, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        // Debug: Check if token is valid
        if (next_token_id < 0 || next_token_id >= vocab_size) {
            std::cerr << "\n[WARNING] Invalid token ID: " << next_token_id << " (vocab_size=" << vocab_size << ")" << std::endl;
            // Try to read some logits for debugging
            std::vector<float> host_logits(vocab_size);
            cudaMemcpy(host_logits.data(), last_logits, vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
            float max_logit = *std::max_element(host_logits.begin(), host_logits.end());
            int max_idx = std::max_element(host_logits.begin(), host_logits.end()) - host_logits.begin();
            std::cerr << "Max logit value: " << max_logit << " at index: " << max_idx << std::endl;
            std::cerr << "First 10 logits: ";
            for (int i = 0; i < 10; i++) std::cerr << host_logits[i] << " ";
            std::cerr << std::endl;
            break; // Stop generation on error
        }

        // E. Print and Append
        std::cout << next_token_id << " " << std::flush;
        input_ids.push_back(next_token_id);

        // Reset Arena scratchpad (keeping weights intact)
        // Important: In a real arena, you'd reset the 'top' pointer to just after d_next_token
        // For this naive test, we assume the arena is big enough to just keep growing 
        // or we manually reset the offset if we implemented reset().
    }

    std::cout << "\n\n>>> Generation Complete." << std::endl;

    cudaStreamDestroy(stream);
    return 0;
}