# Architecture Design: Batch Executor and Orchestrator

## 1. Overview
The core objective of this design is to provide a clean, high-level API for the transformer inference engine while abstracting manual GPU memory management. Crucially, it implements a **Strategy Pattern** for batch execution, allowing different algorithms to process batches.

To achieve this, the system is divided into a three-tiered architecture:
1. **BatchExecutorOrchestrator:** A global manager that handles all global GPU memory allocations and coordinates concurrent batch spawning.
2. **BatchExecutor:** An atomic context that manages the memory lifecycle for a single batch and selects/drives the specific execution strategy.
3. **Execution Strategies (e.g., PipelineEngine):** The specific algorithmic implementations of how a batch is sequenced.

## 2. Core Components

### 2.1 Execution Strategies (e.g., PipelineEngine)
The `PipelineEngine` represents one specific strategy for generating tokens (e.g., standard autoregressive prefill + decode). In the future, other strategies (like Speculative Decoding, Contrastive Search, or Beam Search) can be implemented alongside it. It is purely an algorithmic component that runs on the memory provided to it.

### 2.2 BatchExecutor (The Context & Lifecycle Manager)
The `BatchExecutor` is strictly scoped to handle the **atomic processing of a single batch of input sequences**. 

**Responsibilities:**
* **Strategy Selection:** It chooses and wraps the specific execution strategy (like `PipelineEngine`) to generate the tokens.
* **Isolated Scratch Memory:** It allocates and exclusively owns the `inference_arena` (scratch space, KV cache buffer) required to safely execute its assigned strategy. This guarantees perfect isolation.
* **GPU Parallel Orchestration:** Synchronizes the CUDA stream for its specific batch execution and strategy.

### 2.3 BatchExecutorOrchestrator (The Global Manager)
The `BatchExecutorOrchestrator` acts as the engine CEO. It is the only class the caller directly configures.

**Responsibilities:**
* **Global Memory Allocation:** Internally computes the model's footprint and allocates the global `weight_arena`. 
* **Global Resource Ownership:** Owns the single `Transformer` model and tokenizer.
* **Executor Spawning & Routing:** When the caller submits a batch, the Orchestrator calculates the necessary scratch memory, allocates an isolated `inference_arena`, creates a new `BatchExecutor`, chooses its strategy, and dispatches the job.

## 3. API Usage Example (Vision)
With this architecture, client code leaks zero memory logic and gains massive flexibility in how batches are executed.

```cpp
ModelConfig config = {vocab_size, max_seq_len, d_model, num_heads, num_layers, d_ff};

// 1. Initialize orchestrator (allocates the 700MB model weights internally!)
BatchExecutorOrchestrator orchestrator(config, "vocab.json", "merges.txt");
orchestrator.load_weights("gpt2_weights.bin");

// 2. Submit batches for execution. 
// The Orchestrator spawns BatchExecutors and scratch arenas.
// We can dictate the execution strategy (e.g., STANDARD uses a PipelineEngine)
auto b1_future = orchestrator.submit_batch(prompt1_ids, Strategy::STANDARD);
auto b2_future = orchestrator.submit_batch(prompt2_ids, Strategy::SPECULATIVE);

// 3. Batches execute completely asynchronously on different CUDA streams without memory collisions.
```

## 4. Change Plan

Here are the step-by-step code modifications required:

### Phase 1: Smart Memory Estimators
* Modify `Transformer` down to the fundamental layers to add static memory estimators: `estimate_weight_memory()` and `estimate_inference_scratch(max_batch_size)`.

### Phase 2: Refine Execution Strategies
* Keep `PipelineEngine` intact as the fundamental algorithmic worker. Ensure it operates purely as an execution strategy that takes an `inference_arena` and standard config.

### Phase 3: Implement BatchExecutor
* Create `BatchExecutor` to encapsulate the chosen strategy (e.g., `PipelineEngine`), manage the lifecycle of the batch's `inference_arena`, and execute the sequence on its own CUDA stream.

### Phase 4: Implement BatchExecutorOrchestrator
* Create `BatchExecutorOrchestrator`.
* Implement the constructor to initialize global resources (weights array, tokenizer).
* Implement the `submit_batch` interface which dynamically provisions memory and instantiates `BatchExecutor`s.

### Phase 5: Client Code Cleanup
* Refactor `tests/bench_performance.cu`.
* Delete all manual `GPUMemoryArena` logic.
* Instantiate the new `BatchExecutorOrchestrator` and test multi-batch asynchronous processing.
