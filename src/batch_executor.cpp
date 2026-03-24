#include "batch_executor.hpp"
#include <stdexcept>
#include <utility>

BatchExecutor::BatchExecutor(Transformer& model, GPT2Tokenizer& tokenizer, StrategyType strategy_type, size_t scratch_size) 
    : strategy_type(strategy_type) {
    cudaStreamCreate(&stream);
    inference_arena = std::make_unique<GPUMemoryArena>(scratch_size);
    
    if (strategy_type == StrategyType::STANDARD) {
        strategy = std::make_unique<pipeline::PipelineEngine>(model, tokenizer, *inference_arena, stream);
    } else {
        throw std::invalid_argument("Unsupported strategy type");
    }
}

BatchExecutor::~BatchExecutor() {
    if (stream) {
        cudaStreamDestroy(stream);
    }
}

BatchExecutor::BatchExecutor(BatchExecutor&& other) noexcept 
    : stream(other.stream), inference_arena(std::move(other.inference_arena)), 
      strategy_type(other.strategy_type), strategy(std::move(other.strategy)) {
    other.stream = nullptr;
}

BatchExecutor& BatchExecutor::operator=(BatchExecutor&& other) noexcept {
    if (this != &other) {
        if (stream) {
            cudaStreamDestroy(stream);
        }
        stream = other.stream;
        inference_arena = std::move(other.inference_arena);
        strategy_type = other.strategy_type;
        strategy = std::move(other.strategy);
        other.stream = nullptr;
    }
    return *this;
}

pipeline::GenerationResult BatchExecutor::execute(const std::vector<int>& input_ids, const pipeline::GenerationConfig& config) {
    if (!strategy) {
        throw std::runtime_error("Strategy not initialized");
    }
    return strategy->generate(input_ids, config);
}

void BatchExecutor::synchronize() {
    if (stream) {
        cudaStreamSynchronize(stream);
    }
}
