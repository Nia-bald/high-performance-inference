# GEMV Kernel Optimization Notes

This document contains ongoing notes, insights, and potential future improvements for the high-performance GEMV kernels as we discuss and explore them.

## 1. Kernel Strategies and Routing
The high-performance GEMV (`y[N] = x[K] * W[K, N]`) implementation relies on four distinct kernel strategies depending on the dimensions $N$ and $K$. 

### The 4 Kernels:
1. **`gemv_vec4` (Vectorized, Direct Write):** Uses `float4` to process 4 columns simultaneously. Used when $N$ is large enough to saturate the SMs and $N$ is a multiple of 4. Maximum memory bandwidth utilization.
2. **`gemv_ksplit_vec4` (Vectorized, K-Split with Atomics):** Used for small $N$ (multiple of 4) where standard blocks leave the GPU underutilized. Splits work along the $K$ dimension and uses `atomicAdd` to accumulate results, trading minor atomic overhead for high SM occupancy.
3. **`gemv_scalar` (Scalar, Direct Write):** Fallback for large $N$ that is not a multiple of 4. 
4. **`gemv_ksplit_scalar` (Scalar, K-Split with Atomics):** Fallback for small $N$ that is not a multiple of 4.

### The Dispatcher (`launch_gemv`):
- Checks if $N$ is a multiple of 4 to route to `float4` vectorized loads.
- Checks if the number of blocks `n_blocks` exceeds `TARGET_BLOCKS` (24 for GTX 1050 Ti). If not, it applies a K-split strategy to extract more parallelism.

## 2. Register Arithmetic and L1 Cache Broadcasting
- Computations inside the kernel (like `const int N4 = N >> 2;`) use ultra-fast arithmetic and are stored directly in registers. They cost 1 cycle and are heavily optimized by the compiler, meaning passing them as arguments from the CPU via constant memory provides no benefit and clutters the API.
- When loading `x[k]`, all 32 threads in a warp access the exact same memory address simultaneously. The GPU performs a highly efficient broadcast from the L1 cache, distributing the value to all threads in 1 cycle. The entire `x` vector easily fits in the L1 cache (e.g. $K=768$ is ~3KB).

## 3. The "Cleanup" Epilogue Loop
The main computation loop operates in unrolled chunks of 4 (`k += 4`). 
```cpp
for (; k < K; ++k) {
    float xk = x[k];
    // ...
}
```
This cleanup loop (epilogue) completely eliminates the need for the CPU dispatcher to check if $K$ is a multiple of 4. The main loop safely processes the largest chunk of `K` in groups of 4, and the cleanup loop gracefully processes any remaining 1-3 elements. 

## 4. Future Experiment: Vectorized `float4` Load for `x`
Currently, `x` is loaded as 4 separate `float` reads:
```cpp
float x0 = x[k], x1 = x[k+1], x2 = x[k+2], x3 = x[k+3];
```
While the compiler often automatically vectorizes this to a single `LDG.E.128` instruction because the pointer is `__restrict__` and accesses are sequential, we could explicitly test loading it as a `float4`:
```cpp
float4 x_vec = reinterpret_cast<const float4*>(x)[k/4];
```

**Pros:** 
- Guarantees a single 128-bit memory load instruction at the source code level.

**Cons/Gotchas:**
- Requires the base pointer of `x` to be strictly 16-byte aligned. While usually true for `cudaMalloc`, passing sub-tensors, slices, or offset pointers in the future could cause misaligned memory access crashes.
- Since the overall latency is bound by reading the massively larger `W` matrix from DRAM, speeding up the `x` load from L1 cache may result in zero measurable net performance gain.

**Actionable Test:** 
- Implement an explicitly cast `float4` version of `x` in `gemv_vec4`.
- Validate that `x` is properly aligned.
- Benchmark against the existing kernel using `bench_gemv` to determine if explicit vectorization yields any microsecond latency improvements over the compiler's default optimizations.

## 5. Future Experiment: Passing `N4` as a Kernel Argument
Currently, the kernel calculates `N4` inside each thread:
```cpp
const int N4 = N >> 2;
```
While a bitwise shift (`>> 2`) is extremely fast (1 ALU cycle) and the compiler optimizes its computation across the block, it could theoretically be computed once on the CPU and passed as a kernel parameter via constant memory.

**Actionable Test:**
- Modify the `gemv_vec4` kernel signature to accept `int N4` as an argument.
- Update `launch_gemv` to compute `N4 = N >> 2;` on the host and pass it in.
- Use Nsight Compute (`ncu`) to inspect the PTX/SASS assembly and measure if eliminating this single arithmetic instruction yields any reduction in register pressure or instruction count, keeping in mind the tradeoff of adding an extra constant memory load.

## 6. Future Experiment: Tiling `x` into Shared Memory
Because `W` is incredibly large, warps running ahead of others will flood the L1 cache with `W` elements. This can lead to **cache thrashing**, where the lagging warps find that `x` has been evicted from the L1 cache and must fetch it from the L2 cache instead.

To perfectly protect `x` from eviction, we could implement shared memory tiling.

**Actionable Test:**
- Allocate shared memory: `__shared__ float sx[TILE_SIZE];`
- Have threads cooperatively load a chunk of `x` into `sx`.
- Add a `__syncthreads()` barrier to ensure the entire tile is loaded before computation.
- Compute the partial dot product using `sx`, then repeat for the next tile.
- **Benchmark Analysis:** The main test here is to see if the cost of `__syncthreads()` (forcing fast warps to stop and wait) is more expensive than the cost of occasionally fetching `x` from the L2 cache when warps desynchronize. In memory-bound kernels like GEMV, avoiding synchronization is usually the winning strategy, but explicit testing provides definitive proof!
