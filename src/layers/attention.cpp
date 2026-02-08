#include "attention.h"


SelfAttention::SelfAttention(int batch_size, int seq_len, int d_model, int num_heads, GPUMemoryArena& weights_arena, int qk_dim, int v_dim)
    : batch_size(batch_size), seq_len(seq_len), d_model(d_model), num_heads(num_heads){


    if (qk_dim == 0){
        if (d_model % num_heads != 0) {
            throw std::invalid_argument("d_model must be divisible by num_heads");
        }    
        this->head_dim_qk = d_model/num_heads;
    }
    else{
        this->head_dim_qk = qk_dim;
    }
    
    if (v_dim == 0){
        if (d_model % num_heads != 0) {
            throw std::invalid_argument("d_model must be divisible by num_heads");
        }    
        this->head_dim_v = d_model/num_heads;
    }
    else{
        this->head_dim_v = v_dim;
    }

    this->total_qk_dim = this->head_dim_qk * num_heads;
    this->total_v_dim = this->head_dim_v * num_heads;

    printf("[SelfAttention] Initialized Generalized Attention:\n");
    printf("   >> Q/K Head Dim: %d (Total: %d)\n", head_dim_qk, total_qk_dim);
    printf("   >> V Head Dim:   %d (Total: %d)\n", head_dim_v, total_v_dim);

    this->d_W_q = weights_arena.allocate<float>(d_model*this->total_qk_dim);
    this->d_W_k = weights_arena.allocate<float>(d_model*this->total_qk_dim);
    this->d_W_v = weights_arena.allocate<float>(d_model*this->total_v_dim);
    this->d_W_o = weights_arena.allocate<float>(this->total_v_dim*d_model);

}


void SelfAttention::forward(const float* d_input, float* d_output,
         GPUMemoryArena& inference_arena, cudaStream_t stream){
    
    size_t qk_proj_size = this->batch_size*this->seq_len*this->total_qk_dim;
    
    size_t attention_size = this->seq_len*this->seq_len*this->batch_size*this->num_heads;


    float* d_Q = inference_arena.allocate<float>(qk_proj_size);
    kernels::launch_gemm_tiled(d_input, this->W_q, d_Q, this->batch_size*this->seq_len, this->total_qk_dim, this->d_model, stream);
    // cudaDeviceSynchronize();

    float* d_K = inference_arena.allocate<float>(qk_proj_size);
    kernels::launch_gemm_tiled(d_input, this->W_K, d_K, this->batch_size*this->seq_len, this->total_qk_dim, this->d_model, stream);

    float* d_K_transpose = inference_arena.allocate<float>(qk_proj_size);
    kernels::launch_transpose(d_K, d_K_transpose, this->batch_size*this->seq_len, this->total_qk_dim, stream);

    float* d_attention = inference_arena.allocate<float>(attention_size);

    kernels::launch_batched_gemm(
        d_Q, 
        d_K_transpose, 
        d_attention, 
        this->batch_size*this->seq_len, 
        this->batch_size*this->seq_len, 
        this->total_qk_dim,
        this->seq_len,
        this->seq_len,
        this->head_dim_qk
        stream);

    kernels::launch_batch_upper_triangulate(
        d_attention, 
        this->seq_len*this->batch_size,
        this->total_qk_dim,
        this->seq_len,
        this->seq_len
        stream);
    
    kernels::launch_softmax(
        d_attention,
        this->num_heads*this->batch_size*this->seq_len,
        this->seq_len,
        stream);


}