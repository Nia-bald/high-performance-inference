#include "attention.h"
#include "kv_cache/kv_cache.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

// Log first N elements of a device tensor to stdout (set to 0 to disable)
#define ATTENTION_LOG_SAMPLE 0

static void log_tensor_sample(const char* stage, const float* d_data, size_t total_elements, cudaStream_t stream) {
#if ATTENTION_LOG_SAMPLE > 0
    if (d_data == nullptr || total_elements == 0) return;
    cudaStreamSynchronize(stream);
    size_t n = (total_elements < ATTENTION_LOG_SAMPLE) ? total_elements : ATTENTION_LOG_SAMPLE;
    float h_sample[ATTENTION_LOG_SAMPLE];
    cudaMemcpy(h_sample, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);
    printf("[Attention] %s (first %zu of %zu): ", stage, n, total_elements);
    for (size_t i = 0; i < n; i++) printf("%.4f ", h_sample[i]);
    printf("\n");
#endif
}


SelfAttention::SelfAttention(int d_model, int num_heads, int layer_index, GPUMemoryArena& weights_arena, int qk_dim, int v_dim)
    : Layer("SelfAttention"), d_model(d_model), num_heads(num_heads), layer_index_(layer_index){


    if (qk_dim == 0){
        if (d_model % num_heads != 0) {
            throw std::invalid_argument("d_model must be divisible by num_heads");
        }    
        this->head_dim_qk = d_model/num_heads;
    }
    else{
        this->head_dim_qk = qk_dim;
    }
    
    if (v_dim == 0){
        if (d_model % num_heads != 0) {
            throw std::invalid_argument("d_model must be divisible by num_heads");
        }    
        this->head_dim_v = d_model/num_heads;
    }
    else{
        this->head_dim_v = v_dim;
    }

    this->total_qk_dim = this->head_dim_qk * num_heads;
    this->total_v_dim = this->head_dim_v * num_heads;

    printf("[SelfAttention] Initialized Generalized Attention:\n");
    printf("   >> Q/K Head Dim: %d (Total: %d)\n", head_dim_qk, total_qk_dim);
    printf("   >> V Head Dim:   %d (Total: %d)\n", head_dim_v, total_v_dim);

    // Fused QKV weight: [d_model, 3*total_qk_dim] — single GEMM for Q, K, V
    this->d_W_qkv = weights_arena.allocate<float>(d_model * 3 * this->total_qk_dim);
    this->d_b_qkv = weights_arena.allocate<float>(3 * this->total_qk_dim);
    this->d_W_o = weights_arena.allocate<float>(this->total_v_dim * d_model);
    this->d_b_o = weights_arena.allocate<float>(d_model);

}


void SelfAttention::forward_impl(const float* d_input, float* d_output, int batch_size, int seq_len, GPUMemoryArena* inference_arena, cudaStream_t stream, IKVCache* kv_cache){

    // ===================================================================
    // DECODE PATH: seq_len == 1 && kv_cache available
    // Projects only the new token, appends to cache, then gathers the
    // full K/V history from cache and runs attention against it.
    // ===================================================================
    if (kv_cache != nullptr && seq_len == 1) {
        int pos = kv_cache->current_pos();  // entries already in cache

        // --- 1. Fused QKV projection: single GEMM for Q, K, V ---
        //
        // input [1, d_model] × W_qkv [d_model, 3*D] = d_qkv [1, 3*D]
        //
        //  d_qkv memory layout (batch=1, single row):
        //  <───────── 3*D (2304 for GPT-2) ──────────>
        //  [q₀ q₁ ... q_D-1 | k₀ k₁ ... k_D-1 | v₀ v₁ ... v_D-1]
        //  ^                 ^                   ^
        //  d_q (offset 0)    d_k_new (offset D)  d_v_new (offset 2D)
        //
        //  For batch=1: Q, K, V are contiguous in memory → pointer slice (zero copy).
        //  For batch>1: each row has [Q|K|V] interleaved → need deinterleave kernel.
        //
        size_t proj_size = batch_size * this->total_qk_dim;
        int qkv_cols = 3 * this->total_qk_dim;

        float* d_qkv = inference_arena->allocate<float>(batch_size * qkv_cols);
        kernels::launch_gemm_tiled(d_input, this->d_W_qkv, d_qkv, batch_size, qkv_cols, this->d_model, stream);
        kernels::launch_bias_add(d_qkv, this->d_b_qkv, batch_size, qkv_cols, stream);

        // Slice Q, K, V from fused output
        float *d_q, *d_k_new, *d_v_new;
        if (batch_size == 1) {
            // Zero-cost pointer arithmetic — single row, so Q/K/V are contiguous chunks
            d_q     = d_qkv;
            d_k_new = d_qkv + this->total_qk_dim;
            d_v_new = d_qkv + 2 * this->total_qk_dim;
        } else {
            // Multi-row layout (each row has interleaved Q|K|V):
            //   row 0: [q₀...q_D | k₀...k_D | v₀...v_D]
            //   row 1: [q₀...q_D | k₀...k_D | v₀...v_D]
            //   ...
            // Downstream needs contiguous Q[batch, D], K[batch, D], V[batch, D]
            // so we deinterleave with a lightweight copy kernel.
            d_q     = inference_arena->allocate<float>(proj_size);
            d_k_new = inference_arena->allocate<float>(proj_size);
            d_v_new = inference_arena->allocate<float>(proj_size);
            kernels::launch_deinterleave_qkv(d_qkv, d_q, d_k_new, d_v_new, batch_size, this->total_qk_dim, stream);
        }

        // --- 2. Append new k, v to cache ---
        kv_cache->append_k(layer_index_, d_k_new, 1, stream);
        kv_cache->append_v(layer_index_, d_v_new, 1, stream);

        int total_tokens = pos + 1;  // valid entries after append

        // --- 3. Direct cache read: fused pitched-transpose for K ---
        // Reads directly from cache [batch, max_seq_len, total_qk_dim],
        // writes transposed & packed [total_qk_dim, batch * total_tokens].
        // Eliminates separate gather_K + transpose_K (saves 1 kernel + 1 buffer).
        size_t gathered_size = batch_size * total_tokens * this->total_qk_dim;

        float* d_K_T = inference_arena->allocate<float>(gathered_size);
        kernels::launch_cache_pitched_transpose(
            kv_cache->k_cache_base(layer_index_), d_K_T,
            total_tokens, kv_cache->get_max_seq_len(),
            this->total_qk_dim, batch_size, stream);

        // --- 4. Direct cache read for V ---
        // For batch_size=1: cache V is [max_seq_len, total_qk_dim], first total_tokens
        // rows are contiguous — use cache pointer directly (zero-copy, no kernel).
        // For batch_size>1: inter-batch padding exists, use coalesced gather.
        float* d_V_usable;
        if (batch_size == 1) {
            // Direct read: first total_tokens * total_qk_dim floats are exactly
            // what the GEMM needs — no gaps, no copy.
            d_V_usable = kv_cache->v_cache_base(layer_index_);
        } else {
            d_V_usable = inference_arena->allocate<float>(gathered_size);
            kernels::launch_cache_gather(
                kv_cache->v_cache_base(layer_index_), d_V_usable,
                total_tokens, batch_size, this->num_heads,
                kv_cache->get_max_seq_len(), this->head_dim_qk, stream);
        }

        // --- 5. Q × K^T: [batch, total_qk] × [total_qk, batch*total_tokens] ---
        // Per head: q_h[1, hd] × K_h^T[hd, total_tokens] = scores_h[1, total_tokens]
        size_t scores_size = batch_size * this->num_heads * total_tokens;
        float* d_scores = inference_arena->allocate<float>(scores_size);

        kernels::launch_batched_gemm_naive(
            d_q, d_K_T, d_scores,
            batch_size,                    // M (total rows of Q)
            batch_size * total_tokens,     // N (total rows of K, i.e. cols of K^T)
            this->total_qk_dim,            // K (total columns)
            1,                             // stride_A (rows per batch in Q = seq_len = 1)
            total_tokens,                  // stride_B (rows per batch in K = total_tokens)
            this->head_dim_qk,             // stride_K (columns per head)
            stream);

        // --- 6. Scale ---
        float scale_factor = 1.0f / sqrtf(static_cast<float>(this->head_dim_qk));
        kernels::launch_scale(d_scores, scale_factor, scores_size, stream);

        // (No causal mask needed — single query token can attend to all past tokens)
        // causal mask is removed because we only care about the last token, we possibly dont
        // even need to calculate the attention processed block for other we dont even need to compute them
        // because for subsequent layers we would get them from kv cache
        //yes what I mean was that we would have needed them to get the K and V matrix for the previous token, but since we cache them we dont need compute the modified vector for the previous token at all, 
        // --- 7. Softmax over total_tokens per head ---
        kernels::launch_softmax(d_scores, this->num_heads * batch_size, total_tokens, stream);

        // --- 8. Attn × V: scores[1, total_tokens] × V[total_tokens, hd] = ctx[1, hd] per head ---
        float* d_context = inference_arena->allocate<float>(proj_size);

        kernels::launch_batched_one_to_one_gemm_naive(
            d_scores, d_V_usable, d_context,
            batch_size,                               // M (total rows of scores)
            this->total_qk_dim,                       // N (total columns of V = total_qk_dim)
            this->num_heads * total_tokens,           // K (num batched sub-problems along K axis)
            1,                                        // stride_A (cols per head in scores, seq_len=1)
            this->head_dim_qk,                        // stride_B (cols per head in V)
            total_tokens,                             // stride_K (rows per batch in scores = total_tokens)
            stream);

        // --- 9. Output projection ---
        kernels::launch_gemm_tiled(d_context, d_W_o, d_output, batch_size, this->d_model, this->total_qk_dim, stream);
        kernels::launch_bias_add(d_output, this->d_b_o, batch_size, this->d_model, stream);

        return;  // decode done
    }

    // ===================================================================
    // PREFILL PATH (also used when kv_cache == nullptr for backward compat)
    // Full sequence attention with [seq, seq] score matrix
    // ===================================================================
    
    int total_rows = batch_size * seq_len;
    size_t qk_proj_size = total_rows * this->total_qk_dim;
    int qkv_cols = 3 * this->total_qk_dim;
    
    size_t attention_size = seq_len*seq_len*batch_size*this->num_heads;

    // --- Fused QKV projection: single GEMM ---
    //
    //  input [S, d_model] × W_qkv [d_model, 3*D] = d_QKV [S, 3*D]
    //
    //  d_QKV memory layout (S rows, each row has interleaved Q|K|V):
    //    row 0: [q₀ q₁ ... q_D-1 | k₀ k₁ ... k_D-1 | v₀ v₁ ... v_D-1]
    //    row 1: [q₀ q₁ ... q_D-1 | k₀ k₁ ... k_D-1 | v₀ v₁ ... v_D-1]
    //    ...    ←──── stride = 3*D ────→
    //
    //  Deinterleave copies into contiguous Q[S, D], K[S, D], V[S, D]:
    //    Q: [row0_q | row1_q | ...]   (stride = D, contiguous)
    //    K: [row0_k | row1_k | ...]   (stride = D, contiguous)
    //    V: [row0_v | row1_v | ...]   (stride = D, contiguous)
    //
    float* d_QKV = inference_arena->allocate<float>(total_rows * qkv_cols);
    kernels::launch_gemm_tiled(d_input, this->d_W_qkv, d_QKV, total_rows, qkv_cols, this->d_model, stream);
    kernels::launch_bias_add(d_QKV, this->d_b_qkv, total_rows, qkv_cols, stream);

    float* d_Q = inference_arena->allocate<float>(qk_proj_size);
    float* d_K = inference_arena->allocate<float>(qk_proj_size);
    float* d_V = inference_arena->allocate<float>(qk_proj_size);
    kernels::launch_deinterleave_qkv(d_QKV, d_Q, d_K, d_V, total_rows, this->total_qk_dim, stream);
    log_tensor_sample("Q projection", d_Q, qk_proj_size, stream);
    log_tensor_sample("K projection", d_K, qk_proj_size, stream);

    // Write K/V to cache during prefill if cache is available
    if (kv_cache != nullptr) {
        kv_cache->append_k(layer_index_, d_K, seq_len, stream);
    }

    float* d_K_transpose = inference_arena->allocate<float>(qk_proj_size);
    kernels::launch_transpose(d_K, d_K_transpose, total_rows, this->total_qk_dim, stream);
    log_tensor_sample("K transpose", d_K_transpose, qk_proj_size, stream);
    log_tensor_sample("V projection", d_V, qk_proj_size, stream);

    if (kv_cache != nullptr) {
        kv_cache->append_v(layer_index_, d_V, seq_len, stream);
    }

    float* d_attention = inference_arena->allocate<float>(attention_size);

    kernels::launch_batched_gemm_naive(  // transposed matrix batches matrix mult
        d_Q, 
        d_K_transpose, 
        d_attention, 
        batch_size*seq_len, 
        batch_size*seq_len, 
        this->total_qk_dim,
        seq_len,
        seq_len,
        this->head_dim_qk,
        stream);
    log_tensor_sample("Attention scores (Q*K^T)", d_attention, attention_size, stream);

    // Scale attention scores by 1/sqrt(head_dim) for numerical stability
    float scale_factor = 1.0f / sqrtf(static_cast<float>(this->head_dim_qk));
    kernels::launch_scale(
        d_attention,
        scale_factor,
        attention_size,
        stream);
    log_tensor_sample("Attention (scaled)", d_attention, attention_size, stream);

    kernels::launch_batch_upper_triangulate(
        d_attention, 
        seq_len*batch_size,
        seq_len*this->num_heads,
        seq_len,
        seq_len,
        stream);
    log_tensor_sample("Attention (masked)", d_attention, attention_size, stream);
    
    kernels::launch_softmax(
        d_attention,
        this->num_heads*batch_size*seq_len,
        seq_len,
        stream);
    log_tensor_sample("Attention (softmax)", d_attention, attention_size, stream);
    
    float* d_A_mult_V = inference_arena->allocate<float>(this->total_qk_dim * seq_len * batch_size);

    kernels::launch_batched_one_to_one_gemm_naive(
        d_attention, 
        d_V, 
        d_A_mult_V, 
        batch_size*seq_len, 
        this->total_qk_dim, 
        this->num_heads*seq_len,
        seq_len,
        this->head_dim_qk,
        seq_len,
        stream);
    
    size_t a_mult_v_size = this->total_qk_dim * seq_len * batch_size;
    log_tensor_sample("A*V (context)", d_A_mult_V, a_mult_v_size, stream);

    kernels::launch_gemm_tiled(
        d_A_mult_V,
        d_W_o,
        d_output,
        total_rows,
        this->d_model,
        this->total_qk_dim,
        stream
    );
    kernels::launch_bias_add(d_output, this->d_b_o, total_rows, this->d_model, stream);

    log_tensor_sample("Output (O projection)", d_output, batch_size * seq_len * this->d_model, stream);
}

// --- Add this to src/layers/attention.cpp ---

void SelfAttention::load_weights(const float* h_W_q, const float* h_W_k, 
    const float* h_W_v, const float* h_W_o, 
    const float* h_b_q, const float* h_b_k, 
    const float* h_b_v, const float* h_b_o) 
{
    int D = this->total_qk_dim;

    // Interleave W_q, W_k, W_v row-by-row into W_qkv [d_model, 3*D]
    //
    //  Separate (from exporter):       Fused (on device):
    //  W_q [d_model, D]                W_qkv [d_model, 3*D]
    //  W_k [d_model, D]                row i: [W_q[i] | W_k[i] | W_v[i]]
    //  W_v [d_model, D]
    //
    //  Why row interleave? GEMM reads W row-by-row along the K dimension.
    //  Columns map to output features. Placing Q/K/V columns adjacent means
    //  one GEMM produces [Q|K|V] output, which we can slice or deinterleave.
    //
    std::vector<float> h_W_qkv(d_model * 3 * D);
    for (int i = 0; i < d_model; ++i) {
        std::memcpy(&h_W_qkv[i * 3 * D],         &h_W_q[i * D], D * sizeof(float));
        std::memcpy(&h_W_qkv[i * 3 * D + D],     &h_W_k[i * D], D * sizeof(float));
        std::memcpy(&h_W_qkv[i * 3 * D + 2 * D], &h_W_v[i * D], D * sizeof(float));
    }
    cudaMemcpy(d_W_qkv, h_W_qkv.data(), d_model * 3 * D * sizeof(float), cudaMemcpyHostToDevice);

    // Concatenate biases: [b_q | b_k | b_v]
    std::vector<float> h_b_qkv(3 * D);
    std::memcpy(&h_b_qkv[0],     h_b_q, D * sizeof(float));
    std::memcpy(&h_b_qkv[D],     h_b_k, D * sizeof(float));
    std::memcpy(&h_b_qkv[2 * D], h_b_v, D * sizeof(float));
    cudaMemcpy(d_b_qkv, h_b_qkv.data(), 3 * D * sizeof(float), cudaMemcpyHostToDevice);

    // Output projection stays separate
    size_t matrix_size_o = this->total_qk_dim * d_model * sizeof(float);
    size_t bias_size_o = d_model * sizeof(float);
    cudaMemcpy(d_W_o, h_W_o, matrix_size_o, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_o, h_b_o, bias_size_o, cudaMemcpyHostToDevice);
}

size_t SelfAttention::estimate_weight_memory(int d_model, int num_heads, int qk_dim, int v_dim) {
    int head_dim_qk = (qk_dim == 0) ? d_model / num_heads : qk_dim;
    int head_dim_v = (v_dim == 0) ? d_model / num_heads : v_dim;
    int total_qk_dim = head_dim_qk * num_heads;
    int total_v_dim = head_dim_v * num_heads;

    size_t qk_w = d_model * total_qk_dim + total_qk_dim; // W_q, b_q
    size_t k_w  = d_model * total_qk_dim + total_qk_dim; // W_k, b_k
    size_t v_w  = d_model * total_v_dim + total_v_dim;   // W_v, b_v
    size_t o_w  = total_v_dim * d_model + d_model;       // W_o, b_o
    return (qk_w + k_w + v_w + o_w) * sizeof(float);
}

size_t SelfAttention::estimate_inference_scratch(int max_batch_size, int max_seq_len, int d_model, int num_heads, int qk_dim, int v_dim) {
    int head_dim_qk = (qk_dim == 0) ? d_model / num_heads : qk_dim;
    int total_qk_dim = head_dim_qk * num_heads;

    size_t qk_proj_size = max_batch_size * max_seq_len * total_qk_dim;
    size_t attention_size = max_seq_len * max_seq_len * max_batch_size * num_heads;

    // d_Q, d_K, d_K_transpose, d_V, d_A_mult_V are qk_proj_size
    return 5 * qk_proj_size * sizeof(float) + attention_size * sizeof(float);
}