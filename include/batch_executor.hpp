#pragma once

#include "pipeline/pipeline_engine.hpp"
#include "memory.h"
#include <vector>
#include <memory>
#include <cuda_runtime.h>

enum class StrategyType {
    STANDARD,
    SPECULATIVE // Placeholder for future strategies
};

// BatchExecutor is an atomic context that manages the memory lifecycle 
// for a single batch and runs a specific execution strategy.
class BatchExecutor {
public:
    BatchExecutor(Transformer& model, GPT2Tokenizer& tokenizer, StrategyType strategy, size_t scratch_size);
    ~BatchExecutor();

    // Disable copy to maintain unique ownership of stream and memory
    BatchExecutor(const BatchExecutor&) = delete;
    BatchExecutor& operator=(const BatchExecutor&) = delete;

    // Moving is allowed
    BatchExecutor(BatchExecutor&&) noexcept;
    BatchExecutor& operator=(BatchExecutor&&) noexcept;

    // Execute the batch asynchronously on its own stream
    pipeline::GenerationResult execute(const std::vector<int>& input_ids, const pipeline::GenerationConfig& config);

    // Wait for stream to finish
    void synchronize();

private:
    cudaStream_t stream;
    std::unique_ptr<GPUMemoryArena> inference_arena;
    
    // The specific strategy
    StrategyType strategy_type;
    std::unique_ptr<pipeline::PipelineEngine> strategy;
};
