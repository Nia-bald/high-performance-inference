#include "kernels.cuh"

// -------------------------------------------------------------------
// Kernel: Append [batch, seq_len, num_heads, head_dim] into cache slots
// -------------------------------------------------------------------
// Source and cache now share the same inner layout [num_heads, head_dim],
// so the write is naturally coalesced — adjacent threads write adjacent
// memory addresses within a token's row.
//
// Cache layout: [batch_size, max_seq_len, num_heads, head_dim]
// Src layout:   [batch_size, seq_len,     num_heads, head_dim]
//
// Grid: dim3(num_heads, batch_size * seq_len), Threads: head_dim
__global__ void kernel_cache_append(
    const float* __restrict__ src,   // [batch_size, seq_len, num_heads, head_dim]
    float* __restrict__ cache,       // [batch_size, max_seq_len, num_heads, head_dim]
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
        // src layout is [batch, seq_len, num_heads, head_dim]
        int src_idx = batch * seq_len * num_heads * head_dim
                    + seq_t * num_heads * head_dim
                    + head * head_dim + d;
                    
        // cache layout is [batch, max_seq_len, num_heads, head_dim]
        // Inner dimensions match src — coalesced write!
        int cache_idx = batch * max_seq_len * num_heads * head_dim
                      + (pos + seq_t) * num_heads * head_dim
                      + head * head_dim + d;
                      
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
