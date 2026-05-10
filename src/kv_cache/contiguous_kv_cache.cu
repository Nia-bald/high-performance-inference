#include "kv_cache/contiguous_kv_cache.h"
#include "kernels.cuh"
#include <cstdio>
#include <cuda_runtime.h>

// current memory layout explained SHOULD BE OPTIMIZED FOR EASY ACCESS
// <----------------------layer1-------------------------->
// <--------batch 1------------><----------batch2---------> 
// <--head1-----><--head2------><--head1-----><--head2------>
// [-max_seq*hd-][--max_seq*hd-][-max_seq*hd-][-max_seq*hd-]
// zoom in on single head
// <------------------------head-------------------------->
// <----tok1-------><-----tok2-------><--------tok3------->
// [--head_dim-----][--head_dim------][------head_dim-----]

// -------------------------------------------------------------------
// Constructor
// -------------------------------------------------------------------
ContiguousKVCache::ContiguousKVCache(int num_layers, int num_heads, int max_seq_len,
                                     int head_dim, int batch_size, GPUMemoryArena& arena)
    : num_layers_(num_layers), num_heads_(num_heads), max_seq_len_(max_seq_len),
      head_dim_(head_dim), batch_size_(batch_size), current_pos_(0)
{
    size_t per_layer = (size_t)batch_size * num_heads * max_seq_len * head_dim;

    for (int i = 0; i < num_layers; ++i) {
        k_caches_.push_back(arena.allocate<float>(per_layer));
        v_caches_.push_back(arena.allocate<float>(per_layer));
    }

    size_t total_bytes = 2 * num_layers * per_layer * sizeof(float);
    printf("[KVCache] Allocated ContiguousKVCache: %zu MB for %d layers\n",
           total_bytes / (1024 * 1024), num_layers);
}

// -------------------------------------------------------------------
// append_k / append_v — delegate to kernels::launch_cache_append
// -------------------------------------------------------------------
void ContiguousKVCache::append_k(int layer, const float* d_k_new, int seq_len, cudaStream_t stream) {
    kernels::launch_cache_append(
        d_k_new, k_caches_[layer], current_pos_,
        seq_len, batch_size_, num_heads_, max_seq_len_, head_dim_, stream);
}

void ContiguousKVCache::append_v(int layer, const float* d_v_new, int seq_len, cudaStream_t stream) {
    kernels::launch_cache_append(
        d_v_new, v_caches_[layer], current_pos_,
        seq_len, batch_size_, num_heads_, max_seq_len_, head_dim_, stream);
}

// -------------------------------------------------------------------
// k_head / v_head — return pointer to a specific head's history
// -------------------------------------------------------------------
// For head h, the data starts at: base + h * max_seq_len * head_dim
// This gives the caller [max_seq_len, head_dim] contiguous floats.
float* ContiguousKVCache::k_head(int layer, int batch, int head) {
    // Layout: [batch_size, num_heads, max_seq_len, head_dim]
    return k_caches_[layer] + batch * num_heads_ * max_seq_len_ * head_dim_
                            + head * max_seq_len_ * head_dim_;
}

float* ContiguousKVCache::v_head(int layer, int batch, int head) {
    return v_caches_[layer] + batch * num_heads_ * max_seq_len_ * head_dim_
                            + head * max_seq_len_ * head_dim_;
}

float* ContiguousKVCache::k_cache_base(int layer) { return k_caches_[layer]; }
float* ContiguousKVCache::v_cache_base(int layer) { return v_caches_[layer]; }
// -------------------------------------------------------------------
// Position tracking
// -------------------------------------------------------------------
int  ContiguousKVCache::current_pos() const { return current_pos_; }
void ContiguousKVCache::set_pos(int pos)    { current_pos_ = pos; }
void ContiguousKVCache::advance()           { current_pos_++; }

// -------------------------------------------------------------------
// Metadata
// -------------------------------------------------------------------
int ContiguousKVCache::get_num_layers() const  { return num_layers_; }
int ContiguousKVCache::get_num_heads() const   { return num_heads_; }
int ContiguousKVCache::get_head_dim() const    { return head_dim_; }
int ContiguousKVCache::get_max_seq_len() const { return max_seq_len_; }

// -------------------------------------------------------------------
// Lifecycle
// -------------------------------------------------------------------
void ContiguousKVCache::reset() {
    current_pos_ = 0;
    // No need to zero the memory — current_pos guards valid entries
}

// -------------------------------------------------------------------
// Factory + Memory estimation (defined here, declared in IKVCache)
// -------------------------------------------------------------------
size_t IKVCache::estimate_memory(int num_layers, int num_heads,
                                  int max_seq_len, int head_dim, int batch_size) {
    // 2 caches (K + V) × layers × heads × seq × dim × sizeof(float)
    return 2 * (size_t)batch_size * num_layers * num_heads * max_seq_len * head_dim * sizeof(float);
}

std::unique_ptr<IKVCache> KVCacheFactory::create_contiguous(
    int num_layers, int num_heads, int max_seq_len, int head_dim,
    int batch_size, GPUMemoryArena& arena)
{
    return std::make_unique<ContiguousKVCache>(
        num_layers, num_heads, max_seq_len, head_dim, batch_size, arena);
}
