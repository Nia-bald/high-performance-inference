#include "transformer.h"
#include "tokenizer.h"
#include "memory.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "pipeline/pipeline_engine.hpp"
#include "batch_executor_orchestrator.hpp"

int main() {
    // --- 1. Hyperparameters (GPT-2 Small) ---
    ModelConfig config;
    config.vocab_size = 50257;
    config.max_seq_len = 1024;
    config.d_model = 768;
    config.num_heads = 12;
    config.num_layers = 12;
    config.d_ff = 768 * 4;

    std::cout << ">>> Initializing Engine..." << std::endl;
    BatchExecutorOrchestrator orchestrator(
        config, 
        "/home/niare/Projects/transformer_inference_engine/vocab.json", 
        "/home/niare/Projects/transformer_inference_engine/merges.txt",
        1 // max_batch_size
    );

    // --- 2. Load Weights ---
    orchestrator.load_weights("/home/niare/Projects/transformer_inference_engine/gpt2_weights.bin");

    // --- 3. Prepare Input ---
    std::vector<int> input_ids = {36235, 39141, 373, 257}; // "Alan Turing was a"
    std::cout << "Prompt: " << orchestrator.get_tokenizer().decode(input_ids) << std::endl;

    pipeline::GenerationConfig gen_config;
    gen_config.max_new_tokens = 20;

    std::cout << "\n>>> Starting Inference: 'Alan Turing was a' ..." << std::endl;

    // --- 4. Run Engine ---
    auto future_result = orchestrator.submit_single(input_ids, gen_config, StrategyType::STANDARD);
    auto result = future_result.get();

    std::cout << result.decoded_texts[0] << std::flush;
    std::cout << "\n\n>>> Generation Complete." << std::endl;

    // --- 5. Print Metrics ---
    std::cout << "\n--- Performance Metrics ---" << std::endl;
    std::cout << "Prefill Time:  " << result.metrics.prefill_time_ms << " ms (" 
              << result.metrics.prefill_tokens_per_sec << " tok/s)" << std::endl;
    std::cout << "Decode Time:   " << result.metrics.decode_time_ms << " ms (" 
              << result.metrics.decode_tokens_per_sec << " tok/s) for " 
              << result.metrics.generated_tokens - 1 << " tokens" << std::endl;
    std::cout << "Total Time:    " << result.metrics.total_time_ms << " ms" << std::endl;

    return 0;
}