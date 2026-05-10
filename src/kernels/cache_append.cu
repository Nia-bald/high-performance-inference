#include "kernels.cuh"

// -------------------------------------------------------------------
// Kernel: Scatter-append [batch, num_heads, head_dim] into cache slots
// -------------------------------------------------------------------
// Each head's new K/V vector (head_dim floats) must be copied to a
// different offset in the cache, because heads are stored in separate
// contiguous blocks of [max_seq_len, head_dim].
//
// Grid: dim3(num_heads, batch_size * seq_len), Threads: head_dim
__global__ void kernel_cache_append(
    const float* __restrict__ src,   // [batch_size, seq_len, num_heads, head_dim]
    float* __restrict__ cache,       // [batch_size, num_heads, max_seq_len, head_dim]
    int pos,
    int seq_len,
    int batch_size,
    int num_heads,
    int max_seq_len,
    int head_dim)
{
    int head  = blockIdx.x;
    int batch_seq_idx = blockIdx.y;
    int batch = batch_seq_idx / seq_len;
    int seq_t = batch_seq_idx % seq_len;
    int d     = threadIdx.x;

    if (d < head_dim) {
        // src layout is [batch, seq_len, num_heads, head_dim] -> this matches total_qk_dim
        int src_idx = batch * seq_len * num_heads * head_dim
                    + seq_t * num_heads * head_dim
                    + head * head_dim + d;
                    
        // cache layout is [batch, num_heads, max_seq_len, head_dim]
        int cache_idx = batch * num_heads * max_seq_len * head_dim
                      + head * max_seq_len * head_dim
                      + (pos + seq_t) * head_dim + d;
                      
        cache[cache_idx] = src[src_idx];
    }
}

namespace kernels {

void launch_cache_append(
    const float* src,
    float* cache,
    int pos,
    int seq_len,
    int batch_size,
    int num_heads,
    int max_seq_len,
    int head_dim,
    cudaStream_t stream)
{
    dim3 grid(num_heads, batch_size * seq_len);
    kernel_cache_append<<<grid, head_dim, 0, stream>>>(
        src, cache, pos, seq_len, batch_size, num_heads, max_seq_len, head_dim);
}

} // namespace kernels
