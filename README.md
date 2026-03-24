# Transformer Inference Engine

A from-scratch GPT-2 inference engine written in C++/CUDA, purpose-built to maximize token throughput on a **single NVIDIA GTX 1050 Ti** (4GB VRAM, 768 CUDA cores, Pascal SM 6.1).

**Goal:** Beat vLLM and standard inference libraries on next-token prediction latency and throughput for GPT-2 (124M) — on consumer-grade hardware that the big frameworks ignore.

## Why This Exists

Production inference frameworks (vLLM, TensorRT-LLM, etc.) are designed for data-center GPUs with 40-80GB of VRAM. They carry overhead — Python runtime, generic kernels, PagedAttention for multi-tenant serving — that makes no sense on a 4GB card.

This engine takes the opposite approach: every CUDA kernel, every memory allocation, and every architectural decision is hand-tuned for the constraints of a GTX 1050 Ti. No framework overhead. No Python in the hot path. Just C++ and raw CUDA.

## Current Performance

**GPU:** NVIDIA GeForce GTX 1050 Ti (4040 MB VRAM, 6 SMs, Compute Capability 6.1)
**Model:** GPT-2 Small (124M parameters, 12 layers, 768 dim, 12 heads)

| Metric | Current |
|---|---|
| **Prefill Throughput** | ~200-600 tok/s |
| **Decode Throughput** | ~27-31 tok/s |
| **TTFT (4 tokens)** | ~15 ms |
| **Weight Memory** | 622 MB / 632 MB (98.4%) |
| **Scratch Memory** | 1136 MB |

> These numbers are from the built-in benchmark suite running real text prompts. Decode throughput is currently without KV cache — every token recomputes the full sequence attention. This is the primary optimization target.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Client Code (main.cpp)                     │
│                  Submits prompts, reads results               │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│              BatchExecutorOrchestrator                        │
│  Owns: Transformer model, weights arena, tokenizer           │
│  Creates BatchExecutors per request with isolated memory     │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                    BatchExecutor                              │
│  Owns: CUDA stream, scratch memory arena                     │
│  Delegates to: ExecutionStrategy (via polymorphism)           │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│              ExecutionStrategy (Template Method)              │
│                                                              │
│  generate() {                                                │
│      prefill_time = time( run_prefill() )  ← defined once    │
│      decode_time  = time( run_decode()  )  ← defined once    │
│      compute metrics (tok/s, latency)      ← defined once    │
│  }                                                           │
│                                                              │
│  Concrete strategies only implement:                         │
│    run_prefill()  — process prompt, emit first token          │
│    run_decode()   — autoregressive token generation loop      │
│    finalize()     — detokenization                            │
└──────────────────────┬───────────────────────────────────────┘
                       │
              ┌────────┴──────────┐
              ▼                   ▼
     PipelineEngine        (Future: KVCacheEngine,
     (current impl)         FusedKernelEngine, ...)
```

### Key Design Decisions

- **Arena-based GPU memory management** — Pre-allocates bulk GPU memory and sub-allocates from it. No `cudaMalloc` in the hot path. Scratch memory is reset per-token via offset tracking.
- **Strategy pattern for execution** — The `ExecutionStrategy` base class enforces consistent metric definitions (what "prefill time" and "decode time" mean) across all strategies via the Template Method pattern. New strategies implement execution logic; the framework handles timing.
- **Stream-per-executor isolation** — Each `BatchExecutor` gets its own CUDA stream and memory arena, enabling concurrent execution without resource conflicts.
- **All custom CUDA kernels** — No cuBLAS, no cuDNN. Every kernel (GEMM, softmax, LayerNorm, attention, GELU, argmax) is hand-written and tuned for this GPU's constraints.

## Project Structure

```
├── include/
│   ├── pipeline/
│   │   ├── execution_strategy.hpp  # Abstract base — Template Method pattern
│   │   ├── pipeline_engine.hpp     # Current execution strategy
│   │   └── metrics.hpp             # GenerationMetrics, GenerationResult, GenerationConfig
│   ├── layers/
│   │   ├── transformer.h           # Transformer, TransformerBlock, LayerNorm, FeedForward
│   │   └── attention.h             # Multi-head self-attention
│   ├── batch_executor.hpp          # Per-batch execution context (stream + memory)
│   ├── batch_executor_orchestrator.hpp  # Top-level API — model + weight lifecycle
│   ├── kernels.cuh                 # All CUDA kernel declarations
│   ├── memory.h                    # GPUMemoryArena (bump allocator)
│   ├── tokenizer.h                 # BPE tokenizer (GPT-2 compatible)
│   └── model_config.hpp            # Hyperparameter struct
├── src/
│   ├── kernels/                    # CUDA kernel implementations
│   │   ├── gemm.cu                 #   Tiled GEMM
│   │   ├── batched_gemm.cu         #   Batched GEMM (attention scores)
│   │   ├── softmax.cu              #   Numerically stable softmax
│   │   ├── layernorm.cu            #   Warp-intrinsic LayerNorm
│   │   ├── embedding.cu            #   Token + positional embedding lookup
│   │   ├── activation.cu           #   Fused bias + GELU
│   │   ├── sampling.cu             #   Argmax (greedy decoding)
│   │   ├── transpose.cu            #   Matrix transpose
│   │   ├── addition.cu             #   Element-wise addition (residuals)
│   │   └── ...
│   ├── layers/                     # C++ layer implementations
│   │   ├── attention.cpp           #   Multi-head attention orchestration
│   │   ├── feed_forward.cpp        #   FFN (up-proj → GELU → down-proj)
│   │   ├── layer_norm.cpp          #   LayerNorm wrapper
│   │   └── transformer_block.cpp   #   Pre-norm transformer block
│   ├── pipeline/
│   │   └── pipeline_engine.cpp     # PipelineEngine strategy implementation
│   ├── transformer.cpp             # Full forward pass orchestration
│   ├── batch_executor.cpp          # BatchExecutor implementation
│   ├── batch_executor_orchestrator.cpp
│   ├── tokenizer.cpp               # BPE encode/decode
│   ├── memory.cpp                  # GPU arena allocator
│   └── main.cpp                    # CLI entry point
├── tests/
│   ├── bench_performance.cu        # Full benchmark suite (kernels + pipeline)
│   ├── test_transformer.cu         # End-to-end transformer correctness
│   ├── test_attention.cu           # Attention layer tests
│   ├── test_feed_forward.cu        # FFN tests
│   ├── test_gemm.cu                # GEMM correctness
│   └── ...                         # Per-kernel unit tests
├── tools/
│   ├── gpt2_exporter.py            # Export HuggingFace GPT-2 weights to binary
│   ├── hf_baseline.py              # HuggingFace reference for comparison
│   └── ...                         # Debug/validation scripts
├── docs/
│   └── performance_testing/        # Timestamped benchmark reports (CSV + summary)
├── dataset/
│   ├── input/                      # Benchmark prompt files
│   └── output/                     # Generated text per run
└── CMakeLists.txt
```

## Building

### Prerequisites

- CUDA Toolkit (tested with CUDA 12.x)
- CMake ≥ 3.18
- C++17 compiler
- NVIDIA GPU with Compute Capability ≥ 6.1

### Setup

```bash
# 1. Clone and enter
git clone <repo-url>
cd transformer_inference_engine

# 2. Export GPT-2 weights (requires Python + transformers)
python tools/gpt2_exporter.py

# 3. Build
mkdir -p build && cd build
cmake -DCMAKE_CUDA_ARCHITECTURES=61 ..
make -j$(nproc)
```

### Run

```bash
# Run inference
./gpt2_engine

# Run full benchmark suite
./bench_performance

# Run individual tests
./test_transformer
./test_attention
./test_gemm
# ... etc
```

### Example Output

```
>>> Initializing Engine...
>>> Starting Inference: 'Alan Turing was a' ...
 brilliant mathematician, and he was a great friend of mine.

--- Performance Metrics ---
Prefill Time:  18.29 ms (218.68 tok/s)
Decode Time:   392.78 ms (48.37 tok/s) for 19 tokens
Total Time:    411.07 ms
```

## Benchmark Suite

The benchmark (`tests/bench_performance.cu`) runs two tiers:

1. **Kernel-level** — Isolated timing of individual CUDA kernels (embedding lookup, GEMM, attention QK, softmax, LayerNorm) across different sequence lengths.
2. **Pipeline-level** — End-to-end generation from real text prompts with prefill/decode throughput measurement.

Reports are saved to `docs/performance_testing/run_<timestamp>/` as CSV + human-readable summary.

```bash
# Run benchmarks
cd build && ./bench_performance

# View latest report
cat docs/performance_testing/run_*/summary_*.txt
```

## Roadmap

The primary bottleneck is decode throughput (~30 tok/s). Key optimizations planned:

- [ ] **KV Cache** — Avoid recomputing attention over the full sequence at each decode step. This is the single biggest win available.
- [ ] **Kernel Fusion** — Fuse LayerNorm + QKV projection, bias + GELU, and other adjacent operations to reduce memory bandwidth pressure and kernel launch overhead.
- [ ] **Memory-Efficient Attention** — Reduce the O(S²) attention memory footprint to enable longer sequences within 4GB VRAM.
- [ ] **FP16 / INT8 Quantization** — Halve memory usage and leverage Pascal's FP16 throughput (with caveats — GTX 1050 Ti has limited FP16 support).
- [ ] **CUDA Graphs** — Capture the decode loop as a graph to eliminate per-step kernel launch overhead.

## License

This project is for educational and research purposes.
