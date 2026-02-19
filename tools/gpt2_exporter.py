import torch
from transformers import GPT2LMHeadModel
import struct
import os

def export_weights():
    print("Loading Hugging Face GPT-2...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    # Output file
    f = open("gpt2_weights.bin", "wb")

    def write_tensor(tensor):
        # Flatten and write as float32
        data = tensor.detach().cpu().numpy().astype("float32").flatten()
        f.write(data.tobytes())
        print(f"  -> Wrote {len(data)} floats")

    print("--- Exporting Embeddings ---")
    # 1. Token Embeddings [Vocab, Dim]
    write_tensor(model.transformer.wte.weight)
    # 2. Positional Embeddings [MaxSeq, Dim]
    write_tensor(model.transformer.wpe.weight)

    print("--- Exporting Layers ---")
    for i, block in enumerate(model.transformer.h):
        print(f"Layer {i}")
        
        # Order MUST match your C++ Constructor calls!
        
        # A. Attention Norm (Gamma, Beta)
        write_tensor(block.ln_1.weight) # Gamma
        write_tensor(block.ln_1.bias)   # Beta

        # B. Attention Weights (Q, K, V, O)
        # HF stores Q,K,V in a single "c_attn" matrix [Dim, 3*Dim]. We need to split them?
        # Actually, your implementation expects separate pointers, but we can load them as chunks.
        # However, HF's Conv1D weights are transposed [Dim, 3*Dim]. 
        # Your code expects [Dim, Dim] for each.
        
        # Let's split the Conv1D weight carefully.
        # HF c_attn weight is [Dim, 3*Dim].
        qkv_w = block.attn.c_attn.weight # [768, 2304]
        q_w, k_w, v_w = torch.split(qkv_w, 768, dim=1)
        
        write_tensor(q_w)
        write_tensor(k_w)
        write_tensor(v_w)
        
        # Attention Output Projection
        write_tensor(block.attn.c_proj.weight) 

        # C. FFN Norm (Gamma, Beta)
        write_tensor(block.ln_2.weight)
        write_tensor(block.ln_2.bias)

        # D. FFN Weights (Up, Bias, Down, Bias)
        # HF: c_fc (Up), c_proj (Down)
        write_tensor(block.mlp.c_fc.weight) # Up [Dim, 4*Dim]
        write_tensor(block.mlp.c_fc.bias)   # Up Bias
        
        write_tensor(block.mlp.c_proj.weight) # Down [4*Dim, Dim]
        write_tensor(block.mlp.c_proj.bias)   # Down Bias

    print("--- Exporting Final Norm ---")
    write_tensor(model.transformer.ln_f.weight)
    write_tensor(model.transformer.ln_f.bias)

    print("--- Exporting LM Head ---")
    # GPT-2 ties weights, but we write it explicitly for simplicity
    write_tensor(model.lm_head.weight)

    f.close()
    print("Done! Saved to gpt2_weights.bin")

if __name__ == "__main__":
    export_weights()