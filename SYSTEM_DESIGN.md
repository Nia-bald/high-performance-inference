# Transformer Inference Engine - System Design Document

## 1. Introduction

The **Transformer Inference Engine** is a custom, high-performance C++/CUDA implementation of a GPT-2 style autoregressive language model. It is designed from scratch to perform efficient inference, bypassing heavyweight frameworks like PyTorch or TensorFlow. By implementing core Transformer operators in highly optimized CUDA kernels and orchestrating them through a clean C++ object-oriented architecture, this engine maximizes GPU utilization and memory efficiency.

The system natively loads GPT-2 weights exported from Hugging Face's `transformers` library, tokenizes input text using a custom Byte-Pair Encoding (BPE) implementation, and generates text via a greedy autoregressive decoding loop.

## 2. High-Level Architecture

The architecture is divided into three primary layers:
1.  **Frontend / Orchestration (Host/CPU):** C++ classes that manage initialization, load weights, handle tokenization, orchestrate the inference loop, and manage GPU memory lifecycle.
2.  **Layer Abstractions (Host/CPU):** Object-oriented representations of Transformer modules (e.g., `SelfAttention`, `FeedForward`, `LayerNorm`, `TransformerBlock`). These classes encapsulate the logic to dispatch appropriate CUDA kernels.
3.  **Compute Kernels (Device/GPU):** Highly specialized CUDA kernels that execute the mathematical operations (e.g., General Matrix Multiplication (GEMM), Softmax, LayerNorm, activations) required for neural network inference.

### 2.1 Workflow Summary
1.  **Initialization:** The engine provisions a global GPU Memory Arena, estimating and allocating all required memory for weights and inference buffers up front.
2.  **Weight Loading:** Pre-trained GPT-2 weights (exported via `gpt2_exporter.py`) are loaded from disk directly into device memory.
3.  **Tokenization:** The `GPT2Tokenizer` converts text prompts into an array of integer token IDs using BPE.
4.  **Inference Loop:** The engine runs the text autoregressively. Tokens are embedded, processed through stacked `TransformerBlock`s, and projected to vocabulary logits. A greedy `argmax` kernel selects the next token.
5.  **Output:** Token IDs are decoded back to text on the fly.

---

## 3. Component Design

### 3.1 Memory Management (`memory.h`, `memory.cpp`)
Memory allocation on the GPU is often a bottleneck during inference. The engine implements a custom **Memory Arena (`GPUMemoryArena`)** to mitigate this.

*   **Design:** A single massive chunk of GPU memory is allocated at startup based on calculated requirements. 
*   **Linear Allocation:** The arena provides a bump-pointer allocator (`allocate<T>(size)`). No individual `cudaMalloc` or `cudaFree` calls occur during the inference payload.
*   **Alignment:** Allocations are automatically aligned to 256 boundaries to ensure optimal GPU memory coalescing and vectorized load/store operations.
*   **Scratchpad Reuse:** The `reset_to(offset)` functionality enables robust "ping-pong" buffer reuse. Temporary tensors (e.g., Q/K/V projections, attention matrices) generated during a single block's forward pass are discarded, and the same memory space is recycled for the next block or next token generation iteration.

### 3.2 Compute Kernels (`src/kernels/`, `include/kernels.cuh`)
The performance of the engine hinges on custom-authored CUDA kernels that implement standard operations efficiently.

*   **GEMM (`gemm.cu`):** Implements a highly optimized Tiled General Matrix Multiplication using 2D thread blocks and Shared Memory to maximize memory bandwidth utilization and floating-point throughput.
*   **Batched Operations (`batched_gemm.cu`, `batched_one_to_one_gemm.cu`):** Handles multi-head attention intrinsically. Instead of launching individual kernels per head or iterating sequentially, a single batched kernel computes the operations (e.g., $QK^T$ and $Attention \times V$) for all heads in parallel.
*   **Attention Specifics:** Includes custom kernels for Scaling, Masking (`batch_upper_triangulate`), and Softmax across sequences and heads.
*   **LayerNorm (`layernorm.cu`):** Calculates mean and variance across the embedding dimension efficiently, likely utilizing warp-level reductions.
*   **Embeddings (`embedding.cu`):** Performs fused lookup of Token Embeddings and Positional Embeddings in a single pass.
*   **Sampling (`sampling.cu`):** Implements `launch_argmax` to find the most probable next token across the vocabulary dimension efficiently on the device without requiring full transfer of logits to host.

### 3.3 Layer Abstractions (`src/layers/`, `include/layers/`)
The C++ layer encapsulates the complexity of CUDA kernel configurations (grid/block dimensions, shared memory setups).

*   **`SelfAttention`:** Manages pointers for $W_q, W_k, W_v, W_o$. Its `forward` method coordinates allocating intermediate tensors from the Memory Arena, launching Q, K, V projections recursively, transposing $K$, computing scaled dot-product attention, masking, applying softmax, and projecting the output via $W_o$.
*   **`FeedForward`:** Implements the $d_{model} \rightarrow d_{ff} \rightarrow d_{model}$ standard FFN module (typically with ratio 1:4). Fuses the bias addition with GELU activation using `launch_bias_gelu`.
*   **`LayerNorm`:** Encapsulates the scale ($\gamma$) and shift ($\beta$) learnable parameters.
*   **`TransformerBlock`:** Combines `LayerNorm`, `SelfAttention`, and `FeedForward` into a standard pre-norm residual block:
    *   $x' = x + Attention(LayerNorm(x))$
    *   $x'' = x' + FFN(LayerNorm(x'))$
*   **`Transformer`:** The top-level Model class. It initializes embedding tables, stacking $N$ layers (`TransformerBlock`), and the final layer norm / LM head. Handles the primary ping-pong buffer pattern to cycle data progressively through the deeper networking structures.

### 3.4 Tokenizer (`src/tokenizer.cpp`, `include/tokenizer.h`)
The tokenizer acts as the bridge between raw text strings and numeric ID arrays.

*   **GPT-2 Byte-Pair Encoding (BPE):** Simulates the exact behavior of Hugging Face's GPT-2 tokenizer.
*   **Byte Encoder:** Maps raw text bytes to a unique visual unicode string scheme ensuring all 256 possible bytes are encoded explicitly.
*   **JSON & Merge Loaders:** Unpacks `vocab.json` (into a `std::unordered_map`) and `merges.txt` (into pair merge definitions).
*   **Encoding:** Text is split into words, split into characters, and systematically merged using the established merge priority until tokens collapse into known vocab IDs.

### 3.5 Utilities (`src/utils/`, `tools/`)
*   **`loader.cpp`:** Stream-based unformatted binary reader that transfers exported Python contiguous flat buffer weights into the appropriately pre-allocated CUDA buffers.
*   **`gpt2_exporter.py`:** A Python utility that loads Hugging Face's generalized `GPT2LMHeadModel` and exports the parameters directly to binary files (`.bin`). This natively resolves tensor structuring differences (e.g., splitting a combined $QKV$ tensor into separated discrete $Q, K, V$ buffers suitable for specialized standalone engines).

---

## 4. Execution Data Flow

1.  **Host Prompt Construction:**
    *   Prompt string -> `GPT2Tokenizer::encode` -> Sequence of Token IDs (Int32).
2.  **Preparation (Host -> Device):**
    *   Token IDs are async-copied (`cudaMemcpyAsync`) into the persistent GPU Inference Arena.
3.  **Forward Pass (`Transformer::forward`):**
    *   **Embeddings:** Token IDs + Positions are merged through a fused lookup kernel into unified float-tensor representing the sequence representation (Buffer A).
    *   **Layer Traverse (Ping-Pong):** Sequence data goes through each `TransformerBlock`. Buffer A acts as input, operations populate Buffer B. Pointers are swapped iteratively layer-by-layer.
    *   **Finalizing:** The final state is LayerNormed. The result is channeled into the LM Head `launch_gemm_tiled` multiplying output representations by the vocabulary embeddings to produce `logits`.
4.  **Sampling & Output:**
    *   A kernel evaluates the last sequence position over the $N_{vocab}$ possibilities and outputs a definitive winner token utilizing Argmax.
    *   Token ID transfers (Device -> Host).
    *   Token ID -> `GPT2Tokenizer::decode` -> Output Text generation.
5.  **Autoregressive Shift:**
    *   The newly inferred token is appended to the host memory `std::vector<int> input_ids`.
    *   The GPU Arena resets inference scratchpad offset.
    *   The loop repeats until `max_new_tokens` is met or EOS token occurs.

---

## 5. Potential Bottlenecks & Future Optimizations

*   **Naive Recomputation (Lack of KV-Cache):** Currently, the Engine recomputes the entire prompt + generated text contexts via attention recursively. **Implementation of a Key-Value Cache** is a primary next step. This allows incremental caching of past token representations, changing scaling from $O(N^2)$ to $O(N)$ for long sequence generation.
*   **Kernel Fusion:** Additional fusing of small kernels (e.g., Fusing masking and softmax computation, fusing LayerNorm and Residual adds) will maximize arithmetic intensity against memory bandwidth.
*   **FlashAttention / Specialized Architecture:** Integrating FlashAttention semantics to chunk $Q, K, V$ logic tightly in shared memory natively circumvents instantiating the $N \times N$ attention matrix context in global VRAM.
*   **Quantization:** Reducing weights from `FP32` to `FP16` or `INT8` will substantially halve memory latency and increase throughput on modern architectures containing Tensor Cores.
