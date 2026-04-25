#!/usr/bin/env python3
import os
import sys
import argparse
import time
import csv
import torch
from transformers import GPT2Tokenizer
from transformers.generation.streamers import BaseStreamer

try:
    from optimum.onnxruntime import ORTModelForCausalLM
except ImportError:
    print("Error: optimum or onnxruntime-gpu not installed.")
    sys.exit(1)

class TimingStreamer(BaseStreamer):
    def __init__(self):
        self.first_token_time = None
        
    def put(self, value):
        if self.first_token_time is None:
            torch.cuda.synchronize()
            self.first_token_time = time.perf_counter()
            
    def end(self):
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--session-id", type=str, required=True)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "GPU required for ONNX Runtime."
    
    print("Loading ONNX Runtime GPT-2...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Export to ONNX if not cached
    model = ORTModelForCausalLM.from_pretrained("gpt2", export=True, provider="CUDAExecutionProvider")

    prompts = []
    if os.path.exists(args.dataset_dir):
        for f in os.listdir(args.dataset_dir):
            if f.endswith(".txt"):
                with open(os.path.join(args.dataset_dir, f), 'r') as fp:
                    prompts.append(fp.read())

    batch_prompts = []
    for i in range(args.batch_size):
        batch_prompts.append(prompts[i % len(prompts)])

    inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True).to("cuda")
    seq_len = inputs['input_ids'].shape[1]

    print("ONNX Warmup...")
    model.generate(**inputs, max_new_tokens=2, do_sample=False)

    print("ONNX Benchmarking...")
    streamer = TimingStreamer()
    
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, min_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id, streamer=streamer)
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    prefill_time = streamer.first_token_time - start_time
    total_time = end_time - start_time
    decode_time = total_time - prefill_time
    
    total_decode_tokens = args.batch_size * args.max_new_tokens
    prefill_tok_sec = (args.batch_size * seq_len) / prefill_time if prefill_time > 0 else 0
    decode_tok_sec = total_decode_tokens / decode_time if decode_time > 0 else 0

    csv_path = os.path.join(args.output_dir, f"pipeline_benchmark_onnx_{args.session_id}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["batch_size", "seq_len", "max_new_tokens", "prefill_ms", "decode_ms", "total_ms", "prefill_tok_sec", "decode_tok_sec", "prefill_ms_gpu", "decode_ms_gpu", "total_ms_gpu", "prefill_tok_sec_gpu", "decode_tok_sec_gpu"])
        writer.writerow([args.batch_size, seq_len, args.max_new_tokens, prefill_time*1000, decode_time*1000, total_time*1000, prefill_tok_sec, decode_tok_sec, "", "", "", "", ""])

    print(f"ONNX baseline complete. CSV saved to {csv_path}")

if __name__ == "__main__":
    main()
