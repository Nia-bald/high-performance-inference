# Batch Configuration & Multi-Sequence Batching

**Date:** 2026-03-26  
**Scope:** Config-driven batch size + true multi-prompt batching through the full pipeline

---

## Problem

The batch size in `pipeline_engine.cpp` was hardcoded as `1`:

```cpp
model.forward(d_input_ids, d_logits, 1, current_seq_len, inference_arena, stream);
```

- No way to configure batch size from the benchmark harness or any client
- The pipeline only handled a single sequence — no support for running multiple prompts through the model simultaneously
- Benchmark couldn't measure throughput at different batch sizes

---

## Design Pattern: Layered Config Structs

```
ScenarioConfig              (benchmark-specific: batch_size, max_new_tokens)
        ↓ converted via
BenchmarkRunConfig::to_generation_config()
        ↓
GenerationConfig            (runtime params flowing through the execution pipeline)
        ↓ used by
PipelineEngine              (reads config.batch_size for model.forward)
```

**Why two config structs?** They'll diverge as the project grows:

| `GenerationConfig` (runtime) | `ScenarioConfig` (benchmark-only) |
|---|---|
| `batch_size`, `max_new_tokens` | `batch_size`, `max_new_tokens` |
| Future: `temperature`, `top_k`, `top_p` | Future: `warmup_runs`, `measure_runs`, `label` |

---

## Changes by File

### 1. `include/pipeline/metrics.hpp`

**`GenerationConfig`** — Added `batch_size` field:

```cpp
struct GenerationConfig {
    int batch_size = 1;      // NEW
    int max_new_tokens = 20;
};
```

**`GenerationResult`** — Changed from single-sequence to batched output:

```cpp
// BEFORE
struct GenerationResult {
    std::vector<int> output_sequence;
    std::string decoded_text;
    GenerationMetrics metrics;
};

// AFTER
struct GenerationResult {
    std::vector<std::vector<int>> output_sequences;  // one per batch item
    std::vector<std::string> decoded_texts;           // one per batch item
    GenerationMetrics metrics;                        // aggregate across batch

    int batch_size() const { return output_sequences.size(); }
};
```

---

### 2. `include/benchmark_config.hpp` (NEW FILE)

Benchmark-specific configuration:

```cpp
namespace benchmark {

struct ScenarioConfig {
    int batch_size = 1;
    int max_new_tokens = 50;
};

struct BenchmarkRunConfig {
    std::string weights_path;
    std::string vocab_path;
    std::string merges_path;
    std::string input_dir;

    std::vector<ScenarioConfig> scenarios = { ScenarioConfig{} };

    static pipeline::GenerationConfig to_generation_config(const ScenarioConfig& scenario);
};

} // namespace benchmark
```

---

### 3. `include/pipeline/execution_strategy.hpp`

**`generate()`** signature changed from single-sequence to batched:

```cpp
// BEFORE
GenerationResult generate(const std::vector<int>& input_ids, const GenerationConfig& config);

// AFTER
GenerationResult generate(const std::vector<std::vector<int>>& input_sequences, const GenerationConfig& config);
```

Metrics now aggregate `prompt_tokens` across the full batch:

```cpp
int total_prompt_tokens = 0;
for (const auto& seq : input_sequences) {
    total_prompt_tokens += seq.size();
}
result.metrics.prompt_tokens = total_prompt_tokens;
```

---

### 4. `include/pipeline/pipeline_engine.hpp`

Constructor now takes `max_batch_size` to pre-allocate GPU buffers:

```cpp
// BEFORE
PipelineEngine(Transformer& model, GPT2Tokenizer& tokenizer,
               GPUMemoryArena& inference_arena, cudaStream_t stream = 0);

// AFTER
PipelineEngine(Transformer& model, GPT2Tokenizer& tokenizer,
               GPUMemoryArena& inference_arena,
               int max_batch_size = 1, cudaStream_t stream = 0);
```

New private members:

```cpp
int max_batch_size;
int* d_next_tokens;  // was d_next_token (single int), now array of max_batch_size

// Helper: pad variable-length sequences and pack into flat [B × S] tensor
int pad_and_pack(const std::vector<std::vector<int>>& sequences, std::vector<int>& packed) const;
```

---

### 5. `src/pipeline/pipeline_engine.cpp` (Full Rewrite)

**Constructor** — allocates buffers for `max_batch_size`:

```cpp
d_input_ids   = inference_arena.allocate<int>(max_batch_size * max_seq_len);
d_logits      = inference_arena.allocate<float>(max_batch_size * max_seq_len * vocab_size);
d_next_tokens = inference_arena.allocate<int>(max_batch_size);
```

**`pad_and_pack()`** — new helper that:
1. Finds max length across all sequences in the batch
2. Creates a flat `[batch_size × max_len]` buffer with 0-padding for shorter sequences

**`run_prefill()`** — now operates on the full batch:
1. Calls `pad_and_pack()` to create a flat tensor from all sequences
2. Copies `[B × S]` tensor to GPU in one `cudaMemcpyAsync`
3. Single `model.forward()` with `batch_size = config.batch_size`
4. `launch_argmax()` extracts next token for each sequence in the batch
5. Copies `B` next tokens back and appends one to each sequence

**`run_decode()`** — same batched pattern per decode step:
1. Re-packs all sequences (they grow by 1 each step)
2. Single forward pass for the whole batch
3. Per-sequence argmax + append

**`finalize()`** — decodes each sequence independently via the tokenizer

---

### 6. `include/batch_executor.hpp`

Constructor accepts `max_batch_size`:

```cpp
// BEFORE
BatchExecutor(Transformer& model, GPT2Tokenizer& tokenizer,
              StrategyType strategy, size_t scratch_size);

// AFTER
BatchExecutor(Transformer& model, GPT2Tokenizer& tokenizer,
              StrategyType strategy, size_t scratch_size, int max_batch_size = 1);
```

`execute()` accepts batched input:

```cpp
// BEFORE
GenerationResult execute(const std::vector<int>& input_ids, const GenerationConfig& config);

// AFTER
GenerationResult execute(const std::vector<std::vector<int>>& input_sequences, const GenerationConfig& config);
```

---

### 7. `src/batch_executor.cpp`

Passes `max_batch_size` through to `PipelineEngine`:

```cpp
// BEFORE
strategy = std::make_unique<pipeline::PipelineEngine>(model, tokenizer, *inference_arena, stream);

// AFTER
strategy = std::make_unique<pipeline::PipelineEngine>(model, tokenizer, *inference_arena, max_batch_size, stream);
```

---

### 8. `include/batch_executor_orchestrator.hpp`

Two entry points:

```cpp
// Batched: submit multiple sequences
std::future<GenerationResult> submit_batch(
    const std::vector<std::vector<int>>& input_sequences,
    const GenerationConfig& gen_config, StrategyType strategy);

// Single: convenience wrapper (used by main.cpp)
std::future<GenerationResult> submit_single(
    const std::vector<int>& prompt_ids,
    const GenerationConfig& gen_config, StrategyType strategy);
```

---

### 9. `src/batch_executor_orchestrator.cpp`

`submit_single()` wraps the prompt in a batch of 1 and delegates to `submit_batch()`.

---

### 10. `src/main.cpp`

Minimal changes:

```cpp
// BEFORE
auto future_result = orchestrator.submit_batch(input_ids, gen_config, StrategyType::STANDARD);
std::cout << result.decoded_text;

// AFTER
auto future_result = orchestrator.submit_single(input_ids, gen_config, StrategyType::STANDARD);
std::cout << result.decoded_texts[0];
```

---

### 11. `tests/bench_performance.cu`

**Batch building** — collects all prompts from `dataset/input/`, cycles round-robin to fill `batch_size`:

```cpp
for (int i = 0; i < scenario.batch_size; ++i) {
    const auto& p = prompts[i % prompts.size()];
    batch_sequences.push_back(p.input_ids);
}
```

Example with 2 prompts, `batch_size=4`:
```
[0] prompt_A.txt (len=15)
[1] prompt_B.txt (len=23)
[2] prompt_A.txt (len=15)  ← cycled
[3] prompt_B.txt (len=23)  ← cycled
```

**Single `submit_batch()` call** — runs the entire batch through the model at once.

**Scenario sweep** — iterates over `BenchmarkRunConfig::scenarios`:

```cpp
run_config.scenarios = {
    { .batch_size = 1, .max_new_tokens = 50 },
    // { .batch_size = 4, .max_new_tokens = 50 },  // uncomment to sweep
};
```

---

## Data Flow Diagram

```
bench_performance.cu
  │
  ├── Collects prompts from dataset/input/*.txt
  ├── Cycles prompts to fill scenario.batch_size
  │
  └── orchestrator.submit_batch(vector<vector<int>>)
        │
        └── BatchExecutor(max_batch_size)
              │
              └── PipelineEngine(max_batch_size)
                    │
                    ├── pad_and_pack()     → flat [B × max_len] tensor
                    ├── cudaMemcpyAsync()  → one copy for whole batch
                    ├── model.forward()    → one GPU pass for B sequences
                    ├── launch_argmax()    → B next tokens
                    └── repeat for decode steps
```

---

## How to Add a New Batch Size Benchmark

Edit `tests/bench_performance.cu`, add to the scenarios vector:

```cpp
run_config.scenarios = {
    { .batch_size = 1, .max_new_tokens = 50 },
    { .batch_size = 2, .max_new_tokens = 50 },
    { .batch_size = 4, .max_new_tokens = 50 },
};
```

Each scenario runs independently with its own orchestrator and memory allocation.

---

## Known Limitations

1. **Padding waste**: Shorter sequences are 0-padded to the longest sequence's length. No attention masking is applied, so padded positions participate in attention computation. For equal-length prompts (or benchmarking), this is fine.

2. **No early stopping**: All sequences in the batch generate `max_new_tokens` tokens. If one sequence hits EOS, it still runs. This is acceptable for benchmarking.

3. **Per-sequence prompt length tracking**: `finalize()` approximates per-sequence prompt length as `total_prompt_tokens / batch_size`. This is exact when all sequences have the same length, and approximate otherwise.
