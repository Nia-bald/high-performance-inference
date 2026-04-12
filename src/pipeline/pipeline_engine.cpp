#include "pipeline/pipeline_engine.hpp"
#include <iostream>
#include <algorithm>

namespace pipeline {

PipelineEngine::PipelineEngine(Transformer& model, GPT2Tokenizer& tokenizer, GPUMemoryArena& inference_arena, 
                               int max_batch_size, cudaStream_t stream)
    : model(model), tokenizer(tokenizer), inference_arena(inference_arena), 
      stream(stream), max_batch_size(max_batch_size) {
    
    int vocab_size = model.get_vocab_size();
    int max_seq_len = model.get_max_seq_len();

    // Allocate persistent buffers sized for max_batch_size
    d_input_ids = inference_arena.allocate<int>(max_batch_size * max_seq_len);
    d_logits = inference_arena.allocate<float>(max_batch_size * max_seq_len * vocab_size);
    d_next_tokens = inference_arena.allocate<int>(max_batch_size);

    persistent_offset = inference_arena.get_user();
}

int PipelineEngine::pad_and_pack(const std::vector<std::vector<int>>& sequences, std::vector<int>& packed) const {
    // Find max length across all sequences in the batch
    int max_len = 0;
    for (const auto& seq : sequences) {
        max_len = std::max(max_len, (int)seq.size());
    }

    // Pack into flat [batch_size, max_len] with 0-padding
    int batch_size = sequences.size();
    packed.resize(batch_size * max_len, 0);
    for (int b = 0; b < batch_size; ++b) {
        for (int t = 0; t < (int)sequences[b].size(); ++t) {
            packed[b * max_len + t] = sequences[b][t];
        }
    }
    return max_len;
}

void PipelineEngine::run_prefill(GenerationResult& result, const GenerationConfig& config) {
    int vocab_size = model.get_vocab_size();
    int batch_size = config.batch_size;

    inference_arena.reset_to(persistent_offset);

    // Pad and pack all sequences into a flat tensor
    std::vector<int> packed;
    int padded_seq_len = pad_and_pack(result.output_sequences, packed);

    cudaMemcpyAsync(d_input_ids, packed.data(), batch_size * padded_seq_len * sizeof(int), cudaMemcpyHostToDevice, stream);

    model.forward(d_input_ids, d_logits, batch_size, padded_seq_len, inference_arena, stream);

    // Argmax on last token's logits for each sequence in the batch
    // For each sequence b, the last valid logits are at position (seq_len_b - 1)
    // With padding, all sequences are padded_seq_len, so we use that for uniform argmax
    float* last_logits = d_logits + (padded_seq_len - 1) * vocab_size;
    int row_stride = padded_seq_len * vocab_size;
    kernels::launch_argmax(last_logits, d_next_tokens, batch_size, 1, vocab_size, row_stride, stream);

    // Copy all next tokens back
    std::vector<int> next_tokens(batch_size);
    cudaMemcpyAsync(next_tokens.data(), d_next_tokens, batch_size * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Append each next token to its respective sequence
    for (int b = 0; b < batch_size; ++b) {
        result.output_sequences[b].push_back(next_tokens[b]);
    }
    result.metrics.generated_tokens += batch_size;
}

void PipelineEngine::run_decode(GenerationResult& result, const GenerationConfig& config) {
    int vocab_size = model.get_vocab_size();
    int max_seq_len = model.get_max_seq_len();
    int max_new_tokens = config.max_new_tokens;
    int batch_size = config.batch_size;

    for (int step = 1; step < max_new_tokens; ++step) {
        // Re-use scratch memory for each step
        inference_arena.reset_to(persistent_offset);

        // Pad and pack current state of all sequences
        std::vector<int> packed;
        int padded_seq_len = pad_and_pack(result.output_sequences, packed);

        // Cap generation if any sequence would exceed max
        if (padded_seq_len >= max_seq_len) {
            std::cerr << "Warning: Generation length reached maximum sequence length!" << std::endl;
            break;
        }

        cudaMemcpyAsync(d_input_ids, packed.data(), batch_size * padded_seq_len * sizeof(int), cudaMemcpyHostToDevice, stream);

        model.forward(d_input_ids, d_logits, batch_size, padded_seq_len, inference_arena, stream);

        // Argmax on last position for each sequence
        float* last_logits = d_logits + (padded_seq_len - 1) * vocab_size;
        int row_stride = padded_seq_len * vocab_size;
        kernels::launch_argmax(last_logits, d_next_tokens, batch_size, 1, vocab_size, row_stride, stream);

        std::vector<int> next_tokens(batch_size);
        cudaMemcpyAsync(next_tokens.data(), d_next_tokens, batch_size * sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        for (int b = 0; b < batch_size; ++b) {
            result.output_sequences[b].push_back(next_tokens[b]);
        }
        result.metrics.generated_tokens += batch_size;
    }
}

void PipelineEngine::finalize(GenerationResult& result) {
    result.decoded_texts.resize(result.output_sequences.size());
    for (int b = 0; b < (int)result.output_sequences.size(); ++b) {
        // Find original prompt length — prompt_tokens is total across batch,
        // so for per-sequence decoding we approximate by checking what was added
        // For now, decode the full sequence — the caller knows the prompt boundary
        std::vector<int> new_tokens(
            result.output_sequences[b].begin() + (result.metrics.prompt_tokens / result.batch_size()), 
            result.output_sequences[b].end()
        );
        result.decoded_texts[b] = tokenizer.decode(new_tokens);
    }
}

} // namespace pipeline
