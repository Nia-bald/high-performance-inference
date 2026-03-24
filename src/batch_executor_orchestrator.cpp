#include "batch_executor_orchestrator.hpp"
#include <iostream>

void load_gpt2_weights(Transformer& gpt, const std::string& path, 
    int n_layers, int d_model, int vocab_size, int max_seq, int d_ff);

BatchExecutorOrchestrator::BatchExecutorOrchestrator(const ModelConfig& config, const std::string& vocab_path, const std::string& merges_path, int max_batch_size) 
    : config(config), max_batch_size(max_batch_size) {
    
    // Calculate global memory requirement
    size_t weight_mem = Transformer::estimate_weight_memory(config.vocab_size, config.max_seq_len, config.d_model, config.num_heads, config.num_layers, config.d_ff);
    
    // Add some padding just in case
    weight_mem += 10 * 1024 * 1024; 
    
    weight_arena = std::make_unique<GPUMemoryArena>(weight_mem);
    
    model = std::make_unique<Transformer>(config.vocab_size, config.max_seq_len, config.d_model, config.num_heads, config.num_layers, config.d_ff, *weight_arena);
    
    if (!tokenizer.load(vocab_path, merges_path)) {
        std::cerr << "Failed to load tokenizer files!" << std::endl;
        throw std::runtime_error("Tokenizer load failed");
    }
    
    required_scratch_size = Transformer::estimate_inference_scratch(max_batch_size, config.max_seq_len, config.d_model, config.num_heads, config.num_layers, config.d_ff);
    
    // Provide an additional 50MB padding for scratch memory safely covering anything missed.
    required_scratch_size += 50 * 1024 * 1024;
    
    std::cout << "[Orchestrator] Initialized. Required scratch per executor: " << required_scratch_size / (1024.0 * 1024.0) << " MB" << std::endl;
}

void BatchExecutorOrchestrator::load_weights(const std::string& weights_path) {
    load_gpt2_weights(*model, weights_path, config.num_layers, config.d_model, config.vocab_size, config.max_seq_len, config.d_ff);
}

std::future<pipeline::GenerationResult> BatchExecutorOrchestrator::submit_batch(const std::vector<int>& prompt_ids, const pipeline::GenerationConfig& gen_config, StrategyType strategy) {
    
    return std::async(std::launch::async, [this, prompt_ids, gen_config, strategy]() {
        // Create an executor with exactly the memory it needs
        BatchExecutor executor(*model, tokenizer, strategy, required_scratch_size);
        
        // Execute the batch
        auto result = executor.execute(prompt_ids, gen_config);
        
        // Ensure stream is done before cleaning up
        executor.synchronize();
        
        return result;
    });
}
