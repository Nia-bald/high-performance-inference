#pragma once

#include "model_config.hpp"
#include "transformer.h"
#include "tokenizer.h"
#include "memory.h"
#include "batch_executor.hpp"
#include <memory>
#include <future>
#include <string>
#include <vector>

class BatchExecutorOrchestrator {
public:
    BatchExecutorOrchestrator(const ModelConfig& config, const std::string& vocab_path, const std::string& merges_path, int max_batch_size = 1);
    ~BatchExecutorOrchestrator() = default;

    // Load weights
    void load_weights(const std::string& weights_path);

    // Entry point: submit a batch of sequences
    std::future<pipeline::GenerationResult> submit_batch(const std::vector<std::vector<int>>& input_sequences, const pipeline::GenerationConfig& gen_config, StrategyType strategy = StrategyType::STANDARD);

    // Convenience: submit a single sequence (wraps in a batch of 1)
    std::future<pipeline::GenerationResult> submit_single(const std::vector<int>& prompt_ids, const pipeline::GenerationConfig& gen_config, StrategyType strategy = StrategyType::STANDARD);

    // Access to tokenizer
    GPT2Tokenizer& get_tokenizer() { return tokenizer; }

private:
    ModelConfig config;
    int max_batch_size;
    std::unique_ptr<GPUMemoryArena> weight_arena;
    std::unique_ptr<Transformer> model;
    GPT2Tokenizer tokenizer;
    
    size_t required_scratch_size;
};
