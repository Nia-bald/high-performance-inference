#include "pipeline/single_device_strategy.hpp"
#include "kv_cache/contiguous_kv_cache.h"
#include <iostream>
#include <algorithm>

namespace pipeline {

SingleDeviceStrategy::SingleDeviceStrategy(Transformer& model, GPT2Tokenizer& tokenizer, GPUMemoryArena& inference_arena, 
                               int max_batch_size, cudaStream_t stream)
    : model(model), tokenizer(tokenizer), inference_arena(inference_arena), 
      stream(stream), max_batch_size(max_batch_size) {
    
    int vocab_size = model.get_vocab_size();
    int padded_vocab = model.get_padded_vocab_size();
    int max_seq_len = model.get_max_seq_len();

    // Allocate persistent buffers sized for max_batch_size
    d_input_ids = inference_arena.allocate<int>(max_batch_size * max_seq_len);
    d_logits = inference_arena.allocate<float>(max_batch_size * max_seq_len * padded_vocab);
    d_next_tokens = inference_arena.allocate<int>(max_batch_size);

    // Allocate KV cache (persistent — lives for the entire generation session)
    kv_cache_ = KVCacheFactory::create_contiguous(
        model.get_num_layers(), model.get_num_heads(), max_seq_len,
        model.get_d_model() / model.get_num_heads(),  // head_dim
        max_batch_size, inference_arena);

    persistent_offset = inference_arena.get_used();
}

int SingleDeviceStrategy::pad_and_pack(const std::vector<std::vector<int>>& sequences, std::vector<int>& packed) const {
    // Find max length across all sequences in the batch
    int max_len = 0;
    for (const auto& seq : sequences) {
        max_len = std::max(max_len, (int)seq.size());
    }

    // Pack into flat [batch_size, max_len] with 0-padding on the LEFT
    int batch_size = sequences.size();
    packed.resize(batch_size * max_len, 0);
    for (int b = 0; b < batch_size; ++b) {
        int seq_len = sequences[b].size();
        int pad_len = max_len - seq_len;
        for (int t = 0; t < seq_len; ++t) {
            packed[b * max_len + pad_len + t] = sequences[b][t];
        }
    }
    return max_len;
}

void SingleDeviceStrategy::run_prefill(GenerationResult& result, const GenerationConfig& config) {
    int vocab_size = model.get_vocab_size();
    int batch_size = config.batch_size;

    inference_arena.reset_to(persistent_offset);

    // Reset cache for new generation
    kv_cache_->reset();

    // Pad and pack all sequences into a flat tensor
    std::vector<int> packed;
    int padded_seq_len = pad_and_pack(result.output_sequences, packed);

    cudaMemcpyAsync(d_input_ids, packed.data(), batch_size * padded_seq_len * sizeof(int), cudaMemcpyHostToDevice, stream);

    model.forward(d_input_ids, d_logits, batch_size, padded_seq_len, inference_arena, stream, kv_cache_.get());

    int padded_vocab = model.get_padded_vocab_size();

    // Argmax on last token's logits for each sequence in the batch
    // LM head outputs padded_vocab columns per token
    float* last_logits = d_logits + (padded_seq_len - 1) * padded_vocab;
    int row_stride = padded_seq_len * padded_vocab;
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

    // Mark cache position after prefill
    kv_cache_->set_pos(padded_seq_len);
}

void SingleDeviceStrategy::run_decode(GenerationResult& result, const GenerationConfig& config) {
    int vocab_size = model.get_vocab_size();
    int max_seq_len = model.get_max_seq_len();
    int max_new_tokens = config.max_new_tokens;
    int batch_size = config.batch_size;

    for (int step = 1; step < max_new_tokens; ++step) {
        // Re-use scratch memory for each step
        inference_arena.reset_to(persistent_offset);

        // Cap generation if cache would exceed max
        if (kv_cache_->current_pos() >= max_seq_len) {
            std::cerr << "Warning: Generation length reached maximum sequence length!" << std::endl;
            break;
        }

        // Pack only the LAST token from each sequence (seq_len = 1)
        std::vector<int> last_tokens(batch_size);
        for (int b = 0; b < batch_size; ++b) {
            last_tokens[b] = result.output_sequences[b].back();
        }

        cudaMemcpyAsync(d_input_ids, last_tokens.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice, stream);

        // Forward with seq_len=1 — attention uses cached K/V
        model.forward(d_input_ids, d_logits, batch_size, /*seq_len=*/1, inference_arena, stream, kv_cache_.get());

        // Argmax on the single token's logits (padded vocab width, but search only real vocab)
        int padded_vocab = model.get_padded_vocab_size();
        kernels::launch_argmax(d_logits, d_next_tokens, batch_size, 1, vocab_size, /*row_stride=*/0, stream);

        std::vector<int> next_tokens(batch_size);
        cudaMemcpyAsync(next_tokens.data(), d_next_tokens, batch_size * sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        for (int b = 0; b < batch_size; ++b) {
            result.output_sequences[b].push_back(next_tokens[b]);
        }
        result.metrics.generated_tokens += batch_size;

        // Advance cache position after each decode step
        kv_cache_->advance();
    }
}

void SingleDeviceStrategy::finalize(GenerationResult& result) {
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
