#pragma once
#include "kv_cache/kv_cache.h"
#include "memory.h"
#include <vector>

// ContiguousKVCache: Pre-allocates one big contiguous slab per layer.
//
// Internal layout per layer: [num_heads, max_seq_len, head_dim]
//   - Each head owns a contiguous block of max_seq_len * head_dim floats
//   - Positions within a head are contiguous (good for appending & attention reads)
//
// This is the simplest strategy — no paging, no fragmentation, just pointer math.
class ContiguousKVCache : public IKVCache {
public:
    ContiguousKVCache(int num_layers, int num_heads, int max_seq_len, 
                      int head_dim, int batch_size, GPUMemoryArena& arena);

    // --- IKVCache interface ---
    void append_k(int layer, const float* d_k_new, cudaStream_t stream) override;
    void append_v(int layer, const float* d_v_new, cudaStream_t stream) override;

    float* k_head(int layer, int batch, int head) override;
    float* v_head(int layer, int batch, int head) override;

    float* k_cache_base(int layer) override;
    float* v_cache_base(int layer) override;

    int  current_pos() const override;
    void set_pos(int pos) override;
    void advance() override;

    int get_num_layers() const override;
    int get_num_heads() const override;
    int get_head_dim() const override;
    int get_max_seq_len() const override;

    void reset() override;

private:
    int num_layers_;
    int num_heads_;
    int max_seq_len_;
    int head_dim_;
    int batch_size_;
    int current_pos_;

    // One pointer per layer — each points to [num_heads, max_seq_len, head_dim] floats
    std::vector<float*> k_caches_;
    std::vector<float*> v_caches_;
};
