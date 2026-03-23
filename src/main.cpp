#include "transformer.h"
#include "tokenizer.h"
#include "memory.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "pipeline/pipeline_engine.hpp"

void load_gpt2_weights(Transformer& gpt, const std::string& path, 
    int n_layers, int d_model, int vocab_size, int max_seq, int d_ff);
    
int main() {
    // --- 1. Hyperparameters (GPT-2 Small) ---
    int vocab_size = 50257;
    int max_seq_len = 1024;
    int d_model = 768;
    int num_heads = 12;
    int num_layers = 12;
    int d_ff = 768 * 4;

    size_t weight_mem = (size_t)(700 * 1024 * 1024);
    GPUMemoryArena weight_arena(weight_mem);

    size_t inf_mem = (size_t)(500 * 1024 * 1024);
    GPUMemoryArena inf_arena(inf_mem);

    std::cout << ">>> Initializing Engine..." << std::endl;
    Transformer gpt(vocab_size, max_seq_len, d_model, num_heads, num_layers, d_ff, weight_arena);

    // --- 2. Load Weights ---
    load_gpt2_weights(gpt, "/home/niare/Projects/transformer_inference_engine/gpt2_weights.bin", num_layers, d_model, vocab_size, max_seq_len, d_ff);

    // --- 3. Load Tokenizer ---
    GPT2Tokenizer tokenizer;
    if (!tokenizer.load("/home/niare/Projects/transformer_inference_engine/vocab.json",
                        "/home/niare/Projects/transformer_inference_engine/merges.txt")) {
        std::cerr << "Failed to load tokenizer files!" << std::endl;
        return 1;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // --- 4. Prepare Orchestrator Pipeline ---
    pipeline::PipelineEngine engine(gpt, tokenizer, inf_arena, stream);

    // --- 5. Prepare Input ---
    std::vector<int> input_ids = {36235, 39141, 373, 257}; // "Alan Turing was a"
    std::cout << "Prompt: " << tokenizer.decode(input_ids) << std::endl;

    pipeline::GenerationConfig config;
    config.max_new_tokens = 20;

    std::cout << "\n>>> Starting Inference: 'Alan Turing was a' ..." << std::endl;

    // --- 6. Run Engine ---
    auto result = engine.generate(input_ids, config);

    std::cout << result.decoded_text << std::flush;
    std::cout << "\n\n>>> Generation Complete." << std::endl;

    // --- 7. Print Metrics ---
    std::cout << "\n--- Performance Metrics ---" << std::endl;
    std::cout << "Prefill Time:  " << result.metrics.prefill_time_ms << " ms (" 
              << result.metrics.prefill_tokens_per_sec << " tok/s)" << std::endl;
    std::cout << "Decode Time:   " << result.metrics.decode_time_ms << " ms (" 
              << result.metrics.decode_tokens_per_sec << " tok/s) for " 
              << result.metrics.generated_tokens - 1 << " tokens" << std::endl;
    std::cout << "Total Time:    " << result.metrics.total_time_ms << " ms" << std::endl;

    cudaStreamDestroy(stream);
    return 0;
}