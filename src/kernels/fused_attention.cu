#include "kernels.cuh"
#include <cfloat>
#include <cstdio>

// ============================================================
// Fused Decode Attention: Q·K + Softmax + Attn×V
// ============================================================
// 256 threads per head-block. Key optimizations:
//   - Q loaded to shared memory once (avoids 64 global reads per token)
//   - Pre-scaled Q (fused 1/sqrt(d_k) multiply into Q load)
//   - Phase 3 uses all 256 threads via strided reduction in smem
//   - K/V read directly from cache (no transpose needed)

namespace kernels {

#define FUSED_ATTN_THREADS 256

__global__ void __launch_bounds__(FUSED_ATTN_THREADS)
fused_decode_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    int num_heads,
    int head_dim,
    int total_tokens,
    int cache_stride,
    float scale
) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    const int head_offset = head * head_dim;

    // Shared memory layout:
    //   [0 .. head_dim-1]                    : s_q (pre-scaled Q)
    //   [head_dim .. head_dim+total_tokens-1]: scores
    //   [head_dim+total_tokens .. +256-1]    : Phase 3 reduction scratch
    extern __shared__ float smem[];
    float* s_q = smem;
    float* scores = smem + head_dim;

    // ---- Load Q to shared memory with pre-scaling ----
    if (tid < head_dim) {
        s_q[tid] = Q[head_offset + tid] * scale;
    }
    __syncthreads();

    // ---- Phase 1: scores[t] = s_q · K[t] ----
    for (int t = tid; t < total_tokens; t += FUSED_ATTN_THREADS) {
        float dot = 0.0f;
        const float* k_row = K + t * cache_stride + head_offset;
        int d = 0;
        for (; d + 3 < head_dim; d += 4) {
            dot += s_q[d]     * k_row[d]
                 + s_q[d + 1] * k_row[d + 1]
                 + s_q[d + 2] * k_row[d + 2]
                 + s_q[d + 3] * k_row[d + 3];
        }
        for (; d < head_dim; d++) {
            dot += s_q[d] * k_row[d];
        }
        scores[t] = dot;
    }
    __syncthreads();

    // ---- Phase 2: Softmax ----
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    __shared__ float s_reduce[8];

    // 2a. Max
    float thread_max = -FLT_MAX;
    for (int t = tid; t < total_tokens; t += FUSED_ATTN_THREADS) {
        thread_max = fmaxf(thread_max, scores[t]);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    if (lane_id == 0) s_reduce[warp_id] = thread_max;
    __syncthreads();
    if (tid == 0) {
        float m = -FLT_MAX;
        for (int w = 0; w < 8; w++) m = fmaxf(m, s_reduce[w]);
        s_reduce[0] = m;
    }
    __syncthreads();
    float max_val = s_reduce[0];

    // 2b. Exp + sum
    float thread_sum = 0.0f;
    for (int t = tid; t < total_tokens; t += FUSED_ATTN_THREADS) {
        float e = expf(scores[t] - max_val);
        scores[t] = e;
        thread_sum += e;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    if (lane_id == 0) s_reduce[warp_id] = thread_sum;
    __syncthreads();
    if (tid == 0) {
        float s = 0.0f;
        for (int w = 0; w < 8; w++) s += s_reduce[w];
        s_reduce[0] = s;
    }
    __syncthreads();
    float inv_sum = 1.0f / s_reduce[0];

    // 2c. Normalize
    for (int t = tid; t < total_tokens; t += FUSED_ATTN_THREADS) {
        scores[t] *= inv_sum;
    }
    __syncthreads();

    // ---- Phase 3: Context = scores × V ----
    // All 256 threads participate. With head_dim=64:
    //   dim_idx = tid % head_dim  (which output element)
    //   group   = tid / head_dim  (which sub-range of tokens)
    //   groups_per_dim = 256/64 = 4
    // Each group handles tokens [group, group+4, group+8, ...]
    const int groups_per_dim = FUSED_ATTN_THREADS / head_dim;  // 4
    const int dim_idx = tid % head_dim;
    const int group = tid / head_dim;

    float ctx = 0.0f;
    for (int t = group; t < total_tokens; t += groups_per_dim) {
        ctx += scores[t] * V[t * cache_stride + head_offset + dim_idx];
    }

    // Reduce across groups for same dim_idx using shared memory
    // Use scratch area after scores: smem[head_dim + total_tokens + ...]
    float* ctx_scratch = scores + total_tokens;  // [FUSED_ATTN_THREADS] floats
    ctx_scratch[tid] = ctx;
    __syncthreads();

    // Sequential reduction: group 0 accumulates results from groups 1..3
    if (group == 0 && dim_idx < head_dim) {
        float result = ctx_scratch[dim_idx];
        for (int g = 1; g < groups_per_dim; g++) {
            result += ctx_scratch[g * head_dim + dim_idx];
        }
        output[head_offset + dim_idx] = result;
    }
}

void launch_fused_decode_attention(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int num_heads,
    int head_dim,
    int total_tokens,
    int total_qk_dim,
    int v_stride,
    float scale,
    cudaStream_t stream
) {
    // Shared memory: head_dim (Q) + total_tokens (scores) + FUSED_ATTN_THREADS (ctx scratch)
    int smem_size = (head_dim + total_tokens + FUSED_ATTN_THREADS) * sizeof(float);
    fused_decode_attention_kernel<<<num_heads, FUSED_ATTN_THREADS, smem_size, stream>>>(
        Q, K, V, output, num_heads, head_dim, total_tokens, v_stride, scale);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in fused decode attention: %s\n", cudaGetErrorString(err));
    }
}

} // namespace kernels
