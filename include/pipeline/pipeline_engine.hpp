#pragma once

#include "pipeline/execution_strategy.hpp"
#include "transformer.h"
#include "tokenizer.h"
#include "memory.h"
#include <vector>

namespace pipeline {

class PipelineEngine : public ExecutionStrategy {
public:
    PipelineEngine(Transformer& model, GPT2Tokenizer& tokenizer, GPUMemoryArena& inference_arena, 
                   int max_batch_size = 1, cudaStream_t stream = 0);

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
    int max_batch_size;

    // Persistent buffers (allocated for max_batch_size)
    int* d_input_ids;
    float* d_logits;
    int* d_next_tokens;  // one per sequence in batch
    size_t persistent_offset;

    // Helper: pad sequences to uniform length and pack into flat buffer
    // Returns the padded seq_len
    int pad_and_pack(const std::vector<std::vector<int>>& sequences, std::vector<int>& packed) const;
};

} // namespace pipeline
