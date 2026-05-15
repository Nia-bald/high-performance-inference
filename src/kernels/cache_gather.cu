#include "kernels.cuh"

// -------------------------------------------------------------------
// Kernel: Gather valid K/V entries from cache into flat contiguous buffer
// -------------------------------------------------------------------
// Cache layout:  [batch, max_seq_len, num_heads, head_dim]
// Output layout: [batch * pos, num_heads * head_dim]
//
// With the optimized cache layout, each row at position t already contains
// [num_heads * head_dim] contiguous floats — this is a simple strided copy.
//
// For each position t in [0, pos-1], each batch b:
//   output[(b*pos + t) * total_qk + idx] = cache[b*S*H*D + t*H*D + idx]
//
// Grid: dim3(pos, batch_size), Threads: num_heads * head_dim
__global__ void kernel_cache_gather(
    const float* __restrict__ cache,   // [batch, max_seq_len, num_heads, head_dim]
    float* __restrict__ output,        // [batch * pos, num_heads * head_dim]
    int pos,
    int num_heads,
    int max_seq_len,
    int head_dim)
{
    int t = blockIdx.x;      // position index [0, pos)
    int b = blockIdx.y;      // batch index
    int idx = threadIdx.x;   // index into total_qk_dim = num_heads * head_dim

    int total_qk_dim = num_heads * head_dim;
    if (idx >= total_qk_dim) return;

    // Cache: [b, t, idx] — already contiguous per token!
    // No need to decompose idx into (h, d) and jump across head blocks.
    int cache_idx = b * max_seq_len * total_qk_dim
                  + t * total_qk_dim + idx;

    // Output: [b*pos + t, idx]
    int out_idx = (b * pos + t) * total_qk_dim + idx;

    output[out_idx] = cache[cache_idx];
}

// -------------------------------------------------------------------
// Kernel: Fused pitched-transpose — read directly from cache, write transposed
// -------------------------------------------------------------------
// Replaces gather_K + transpose_K in the decode path.
//
// Cache layout:  [batch, max_seq_len, total_qk_dim]  (only first pos rows valid)
// Output layout: [total_qk_dim, batch * pos]          (transposed & packed)
//
// For each batch b, token t in [0, pos-1], each element idx in [0, total_qk_dim-1]:
//   Read:  cache[b * max_seq_len * total_qk_dim + t * total_qk_dim + idx]  (coalesced)
//   Write: output[idx * (batch_size * pos) + b * pos + t]                   (strided)
//
// Grid: dim3(pos, batch_size), Threads: total_qk_dim
__global__ void kernel_cache_pitched_transpose(
    const float* __restrict__ cache,   // [batch, max_seq_len, total_qk_dim]
    float* __restrict__ output,        // [total_qk_dim, batch * pos]
    int pos,
    int max_seq_len,
    int total_qk_dim,
    int batch_size)
{
    int t = blockIdx.x;      // token position [0, pos)
    int b = blockIdx.y;      // batch index
    int idx = threadIdx.x;   // index into total_qk_dim

    if (idx >= total_qk_dim) return;

    // Read from cache: pitched layout — stride is max_seq_len * total_qk_dim between batches
    int cache_idx = b * max_seq_len * total_qk_dim
                  + t * total_qk_dim + idx;

    // Write transposed & packed: output[idx][b * pos + t]
    int total_cols = batch_size * pos;
    int out_idx = idx * total_cols + b * pos + t;

    output[out_idx] = cache[cache_idx];
}

namespace kernels {

void launch_cache_gather(
    const float* cache,
    float* output,
    int pos,
    int batch_size,
    int num_heads,
    int max_seq_len,
    int head_dim,
    cudaStream_t stream)
{
    int total_qk_dim = num_heads * head_dim;
    dim3 grid(pos, batch_size);
    // For models where total_qk_dim > 1024, we'd need to handle this differently,
    // but GPT-2 has total_qk_dim = 768 which fits in one block.
    kernel_cache_gather<<<grid, total_qk_dim, 0, stream>>>(
        cache, output, pos, num_heads, max_seq_len, head_dim);
}

void launch_cache_pitched_transpose(
    const float* cache,
    float* output,
    int pos,
    int max_seq_len,
    int total_qk_dim,
    int batch_size,
    cudaStream_t stream)
{
    dim3 grid(pos, batch_size);
    // Same thread-count constraint as gather: total_qk_dim must fit in one block.
    // GPT-2 has total_qk_dim = 768 which fits.
    kernel_cache_pitched_transpose<<<grid, total_qk_dim, 0, stream>>>(
        cache, output, pos, max_seq_len, total_qk_dim, batch_size);
}

} // namespace kernels
