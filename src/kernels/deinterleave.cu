#include "kernels.cuh"
#include <cstdio>

namespace kernels {

// Splits fused QKV [rows, 3*D] into three contiguous [rows, D] buffers.
// Reads are coalesced (adjacent threads read adjacent columns from qkv).
// Writes are coalesced (adjacent threads write adjacent columns to q/k/v).
__global__ void deinterleave_qkv_kernel(
    const float* __restrict__ qkv,  // [rows, 3*D]
    float* __restrict__ q,          // [rows, D]
    float* __restrict__ k,          // [rows, D]
    float* __restrict__ v,          // [rows, D]
    int rows, int D)
{
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows || col >= D) return;

    int src_base = row * 3 * D + col;
    int dst_idx  = row * D + col;

    q[dst_idx] = qkv[src_base];
    k[dst_idx] = qkv[src_base + D];
    v[dst_idx] = qkv[src_base + 2 * D];
}

void launch_deinterleave_qkv(
    const float* qkv,
    float* q, float* k, float* v,
    int rows, int D,
    cudaStream_t stream)
{
    constexpr int THREADS = 256;
    dim3 block(THREADS);
    dim3 grid((D + THREADS - 1) / THREADS, rows);

    deinterleave_qkv_kernel<<<grid, block, 0, stream>>>(qkv, q, k, v, rows, D);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in deinterleave_qkv: %s\n", cudaGetErrorString(err));
    }
}

} // namespace kernels
