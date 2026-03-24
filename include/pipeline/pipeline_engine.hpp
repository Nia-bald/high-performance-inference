#pragma once

#include "pipeline/execution_strategy.hpp"
#include "transformer.h"
#include "tokenizer.h"
#include "memory.h"
#include <vector>

namespace pipeline {

class PipelineEngine : public ExecutionStrategy {
public:
    PipelineEngine(Transformer& model, GPT2Tokenizer& tokenizer, GPUMemoryArena& inference_arena, cudaStream_t stream = 0);

protected:
    // Strategy hooks — pure execution logic, no timing, no metric math.
    void run_prefill(GenerationResult& result, const GenerationConfig& config) override;
    void run_decode(GenerationResult& result, const GenerationConfig& config) override;
    void finalize(GenerationResult& result) override;

    cudaStream_t get_stream() const override { return stream; }

private:
    Transformer& model;
    GPT2Tokenizer& tokenizer;
    GPUMemoryArena& inference_arena;
    cudaStream_t stream;

    // Persistent buffers
    int* d_input_ids;
    float* d_logits;
    int* d_next_token;
    size_t persistent_offset;
};

} // namespace pipeline
