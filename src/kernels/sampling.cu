#include "kernels.cuh"
#include <cstdio>
#include <cfloat>

#define FINAL_MASK 0xffffffff

namespace kernels {

    // --- Helper: Warp Reduction for (Value, Index) ---
    // Finds the max value and its corresponding index within a warp
    __inline__ __device__ void warp_reduce_max(float& val, int& idx) {
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_val = __shfl_down_sync(FINAL_MASK, val, offset);
            int other_idx   = __shfl_down_sync(FINAL_MASK, idx, offset);

            if (other_val > val) {
                val = other_val;
                idx = other_idx;
            }
        }
    }

    // --- Main Kernel ---
    // Grid: [Batch * Seq] blocks
    // Block: 256 threads (standard for reductions)
    __global__ void argmax_kernel(
        const float* __restrict__ logits, // [Batch*Seq, Vocab]
        int* __restrict__ output_ids,     // [Batch*Seq]
        int vocab_size) 
    {
        // 1. Identify which row (token position) this block is handling
        int row_idx = blockIdx.x;
        
        // Pointer to the start of this row's logits
        const float* row_logits = logits + row_idx * vocab_size;

        // 2. Thread-Local Max
        // Initialize with lowest possible float
        float max_val = -FLT_MAX;
        int max_idx = -1;

        // Stride over the vocabulary
        for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
            float val = row_logits[i];
            if (val > max_val) {
                max_val = val;
                max_idx = i;
            }
        }

        // 3. Block Reduction
        // A. Warp Reduce (Find max within each warp)
        warp_reduce_max(max_val, max_idx);

        // B. Store Warp Leaders in Shared Memory
        static __shared__ float s_max_vals[32]; // Max 32 warps per block (1024 threads)
        static __shared__ int s_max_idxs[32];
        
        int lane = threadIdx.x % 32;
        int warp_id = threadIdx.x / 32;

        if (lane == 0) {
            s_max_vals[warp_id] = max_val;
            s_max_idxs[warp_id] = max_idx;
        }
        __syncthreads();

        // C. Final Reduce by Warp 0
        // Only the first warp needs to run to reduce the shared memory values
        if (warp_id == 0) {
            // Reload the warp leaders
            // Ensure we handle cases where blockDim < 1024 (fewer than 32 warps)
            int num_warps = (blockDim.x + 31) / 32;
            
            if (lane < num_warps) {
                max_val = s_max_vals[lane];
                max_idx = s_max_idxs[lane];
            } else {
                max_val = -FLT_MAX;
                max_idx = -1;
            }

            // Final Warp Reduce
            warp_reduce_max(max_val, max_idx);

            // 4. Write Result
            if (lane == 0) {
                output_ids[row_idx] = max_idx;
            }
        }
    }

    void launch_argmax(
        const float* logits, 
        int* output_ids, 
        int batch_size, 
        int seq_len, 
        int vocab_size, 
        cudaStream_t stream) 
    {
        int total_rows = batch_size * seq_len;
        int threads = 256; // Robust size for reductions
        
        argmax_kernel<<<total_rows, threads, 0, stream>>>(
            logits, 
            output_ids, 
            vocab_size
        );
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error in Argmax: %s\n", cudaGetErrorString(err));
        }
    }
}