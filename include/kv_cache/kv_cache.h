#pragma once
#include <memory>
#include <cuda_runtime.h>

class GPUMemoryArena;

// IKVCache: Abstract interface for Key-Value Cache strategies.
// 
// SelfAttention interacts ONLY with this interface — it never sees
// ContiguousKVCache, GPUMemoryArena, or any allocation details.
// This decouples the compute kernels from the physical memory layout.
//
// Setting a method = 0 makes it "pure virtual": subclasses MUST implement it,
// and you cannot instantiate IKVCache directly.
class IKVCache {
public:
    virtual ~IKVCache() = default;

    // --- Data Writing (Layout Agnostic) ---
    // Appends new K/V vectors (shape: [batch_size, seq_len, total_qk_dim]) to the cache
    virtual void append_k(int layer, const float* d_k_new, int seq_len, cudaStream_t stream) = 0;
    virtual void append_v(int layer, const float* d_v_new, int seq_len, cudaStream_t stream) = 0;

    // --- Data Reading (Layout Agnostic) ---
    // Returns pointer to a specific (batch, head)'s K/V history for a layer.
    // The returned pointer addresses [max_seq_len, head_dim] contiguous floats,
    // but only entries [0..current_pos-1] are valid.
    virtual float* k_head(int layer, int batch, int head) = 0;
    virtual float* v_head(int layer, int batch, int head) = 0;

    // Returns the raw base pointer for a layer's K/V cache.
    // Layout: [batch_size, max_seq_len, num_heads, head_dim]
    // Used by attention kernels that read directly from the cache.
    virtual float* k_cache_base(int layer) = 0;
    virtual float* v_cache_base(int layer) = 0;

    // --- Position Tracking ---
    // Position = number of tokens whose K,V have been written into the cache.
    virtual int  current_pos() const = 0;
    virtual void set_pos(int pos) = 0;     // Set after prefill (e.g., set_pos(prompt_len))
    virtual void advance() = 0;            // current_pos++ after each decode step

    // --- Metadata ---
    virtual int get_num_layers() const = 0;
    virtual int get_num_heads() const = 0;
    virtual int get_head_dim() const = 0;
    virtual int get_max_seq_len() const = 0;
    virtual int get_batch_size() const = 0;

    // --- Lifecycle ---
    virtual void reset() = 0;              // Clear cache for a new generation

    // --- Memory Estimation ---
    static size_t estimate_memory(int num_layers, int num_heads, 
                                   int max_seq_len, int head_dim, int batch_size);
};

// Factory for creating KV cache instances.
// SingleDeviceStrategy uses this — it's the only place that knows about concrete strategies.
class KVCacheFactory {
public:
    static std::unique_ptr<IKVCache> create_contiguous(
        int num_layers, int num_heads, int max_seq_len, int head_dim,
        int batch_size, GPUMemoryArena& arena);
};
