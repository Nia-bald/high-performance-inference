#include "transformer.h"
#include <fstream>
#include <vector>
#include <iostream>

// Helper to read N floats from file into host memory
void read_floats(std::ifstream& f, std::vector<float>& buffer, size_t size) {
    buffer.resize(size);
    f.read(reinterpret_cast<char*>(buffer.data()), size * sizeof(float));
}

void load_gpt2_weights(Transformer& gpt, const std::string& path, 
                       int n_layers, int d_model, int vocab_size, int max_seq, int d_ff) 
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "Error: Could not open " << path << std::endl;
        exit(1);
    }

    std::vector<float> buf;

    // 1. Embeddings
    read_floats(f, buf, vocab_size * d_model);
    std::vector<float> token_emb = buf; // Copy needed? optimize later
    
    read_floats(f, buf, max_seq * d_model);
    std::vector<float> pos_emb = buf;

    gpt.load_embeddings(token_emb.data(), pos_emb.data());

    // 2. Layers
    for (int i = 0; i < n_layers; ++i) {
        TransformerBlock* block = gpt.get_block(i); // You need to add this accessor to Transformer class!
        
        // A. Attn Norm
        read_floats(f, buf, d_model); auto gamma1 = buf;
        read_floats(f, buf, d_model); auto beta1 = buf;
        block->get_attn_norm().load_weights(gamma1.data(), beta1.data());

        // B. Attn Weights
        // Q
        read_floats(f, buf, d_model * d_model); auto wq = buf;
        // K
        read_floats(f, buf, d_model * d_model); auto wk = buf;
        // V
        read_floats(f, buf, d_model * d_model); auto wv = buf;
        // O
        read_floats(f, buf, d_model * d_model); auto wo = buf;
        
        block->get_attention().load_weights(wq.data(), wk.data(), wv.data(), wo.data());

        // C. FFN Norm
        read_floats(f, buf, d_model); auto gamma2 = buf;
        read_floats(f, buf, d_model); auto beta2 = buf;
        block->get_ffn_norm().load_weights(gamma2.data(), beta2.data());

        // D. FFN Weights
        // Up [d_model, d_ff]
        read_floats(f, buf, d_model * d_ff); auto w_up = buf;
        read_floats(f, buf, d_ff);           auto b_up = buf;
        
        // Down [d_ff, d_model]
        read_floats(f, buf, d_ff * d_model); auto w_down = buf;
        read_floats(f, buf, d_model);        auto b_down = buf;

        block->get_ffn().load_weights(w_up.data(), b_up.data(), w_down.data(), b_down.data());
    }

    // 3. Final Norm
    read_floats(f, buf, d_model); auto f_gamma = buf;
    read_floats(f, buf, d_model); auto f_beta = buf;
    gpt.get_final_norm().load_weights(f_gamma.data(), f_beta.data());

    // 4. LM Head
    read_floats(f, buf, d_model * vocab_size);
    gpt.load_head(buf.data());

    f.close();
    std::cout << ">>> Weights Loaded Successfully!" << std::endl;
}