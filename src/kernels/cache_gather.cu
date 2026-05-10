#include "kernels.cuh"

// -------------------------------------------------------------------
// Kernel: Gather valid K/V entries from cache into flat contiguous buffer
// -------------------------------------------------------------------
// Cache layout:  [batch, num_heads, max_seq_len, head_dim]
// Output layout: [batch * pos, num_heads * head_dim]  (same as GEMM projection output)
//
// For each position t in [0, pos-1], each batch b:
//   output[(b*pos + t) * total_qk + h*hd + d] = cache[b*H*S*D + h*S*D + t*D + d]
//
// Grid: dim3(pos, batch_size), Threads: num_heads * head_dim
__global__ void kernel_cache_gather(
    const float* __restrict__ cache,   // [batch, num_heads, max_seq_len, head_dim]
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

    int h = idx / head_dim;
    int d = idx % head_dim;

    // Cache: [b, h, t, d]
    int cache_idx = b * num_heads * max_seq_len * head_dim
                  + h * max_seq_len * head_dim
                  + t * head_dim + d;

    // Output: [b*pos + t, h*head_dim + d]
    int out_idx = (b * pos + t) * total_qk_dim + idx;

    output[out_idx] = cache[cache_idx];
}

// output
// <----------------------l1----------------------->
// <-----------b1----------><---------b2----------->
// <----t1-----><-----t2---><----t1----><----t2---->
// <-h1-><-h2--><-h1-><-h2-><-h1-><-h2-><-h1-><-h2->

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

} // namespace kernels
