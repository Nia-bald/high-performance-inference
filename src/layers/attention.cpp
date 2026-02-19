#include "attention.h"
#include <cmath>


SelfAttention::SelfAttention(int d_model, int num_heads, GPUMemoryArena& weights_arena, int qk_dim, int v_dim)
    : d_model(d_model), num_heads(num_heads){


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


void SelfAttention::forward(int batch_size, int seq_len, const float* d_input, float* d_output,
         GPUMemoryArena& inference_arena, cudaStream_t stream){
    
    size_t qk_proj_size = batch_size*seq_len*this->total_qk_dim;
    
    size_t attention_size = seq_len*seq_len*batch_size*this->num_heads;


    float* d_Q = inference_arena.allocate<float>(qk_proj_size);
    kernels::launch_gemm_tiled(d_input, this->d_W_q, d_Q, batch_size*seq_len, this->total_qk_dim, this->d_model, stream);
    // cudaDeviceSynchronize();

    float* d_K = inference_arena.allocate<float>(qk_proj_size);
    kernels::launch_gemm_tiled(d_input, this->d_W_k, d_K, batch_size*seq_len, this->total_qk_dim, this->d_model, stream);

    float* d_K_transpose = inference_arena.allocate<float>(qk_proj_size);
    kernels::launch_transpose(d_K, d_K_transpose, batch_size*seq_len, this->total_qk_dim, stream);

    float* d_V = inference_arena.allocate<float>(qk_proj_size);
    kernels::launch_gemm_tiled(d_input, this->d_W_v, d_V, batch_size*seq_len, this->total_qk_dim, this->d_model, stream);

    float* d_attention = inference_arena.allocate<float>(attention_size);

    kernels::launch_batched_gemm(  // transposed matrix batches matrix mult
        d_Q, 
        d_K_transpose, 
        d_attention, 
        batch_size*seq_len, 
        batch_size*seq_len, 
        this->total_qk_dim,
        seq_len,
        seq_len,
        this->head_dim_qk,
        stream);

    kernels::launch_batch_upper_triangulate(
        d_attention, 
        seq_len*batch_size,
        this->total_qk_dim,
        seq_len,
        seq_len,
        stream);
    
    // Scale attention scores by 1/sqrt(head_dim) for numerical stability
    // This prevents the Q*K^T scores from becoming too large before softmax
    float scale_factor = 1.0f / sqrtf(static_cast<float>(this->head_dim_qk));
    kernels::launch_scale(
        d_attention,
        scale_factor,
        attention_size,
        stream);
    
    kernels::launch_softmax(
        d_attention,
        this->num_heads*batch_size*seq_len,
        seq_len,
        stream);
    
    float* d_A_mult_V = inference_arena.allocate<float>(this->total_qk_dim * seq_len * batch_size);

    kernels::launch_batched_one_to_one_gemm(
        d_attention, 
        d_V, 
        d_A_mult_V, 
        batch_size*seq_len, 
        this->total_qk_dim, 
        batch_size*seq_len,
        seq_len,
        this->head_dim_qk,
        seq_len,
        stream);

    kernels::launch_gemm_tiled(
        d_A_mult_V,
        d_W_o,
        d_output,
        batch_size*seq_len,
        this->total_qk_dim,
        this->total_qk_dim,
        stream
    );
}

// --- Add this to src/layers/attention.cpp ---

void SelfAttention::load_weights(const float* h_W_q, const float* h_W_k, 
    const float* h_W_v, const float* h_W_o) 
{
// Each weight matrix in standard attention is [d_model, d_model]
size_t matrix_size = d_model * d_model * sizeof(float);

cudaMemcpy(d_W_q, h_W_q, matrix_size, cudaMemcpyHostToDevice);
cudaMemcpy(d_W_k, h_W_k, matrix_size, cudaMemcpyHostToDevice);
cudaMemcpy(d_W_v, h_W_v, matrix_size, cudaMemcpyHostToDevice);
cudaMemcpy(d_W_o, h_W_o, matrix_size, cudaMemcpyHostToDevice);
}