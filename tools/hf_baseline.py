#!/usr/bin/env python3
import os
import sys
import argparse
import time
import csv
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers.generation.streamers import BaseStreamer

class TimingStreamer(BaseStreamer):
    def __init__(self):
        self.first_token_time = None
        self.prompt_skipped = False
        
    def put(self, value):
        if not self.prompt_skipped:
            self.prompt_skipped = True
            return
            
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

    assert torch.cuda.is_available(), "GPU is strictly required for benchmarking!"
    device = "cuda"
    print(f"Loading HF GPT-2 on {device}...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()

    # Load prompts
    prompts = []
    filenames = []
    if os.path.exists(args.dataset_dir):
        for f in os.listdir(args.dataset_dir):
            if f.endswith(".txt"):
                with open(os.path.join(args.dataset_dir, f), 'r') as fp:
                    prompts.append(fp.read())
                    filenames.append(f)
                    
    if not prompts:
        print("Error: No prompts found.")
        sys.exit(1)

    # Batching logic
    batch_prompts = []
    batch_filenames = []
    for i in range(args.batch_size):
        idx = i % len(prompts)
        batch_prompts.append(prompts[idx])
        batch_filenames.append(filenames[idx])

    inputs = tokenizer(batch_prompts, return_tensors='pt', padding=True).to(device)
    seq_len = inputs['input_ids'].shape[1]

    # Warmup
    print("HF Warmup...")
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=2, do_sample=False, pad_token_id=tokenizer.eos_token_id)

    # Track
    print("HF Benchmarking...")
    streamer = TimingStreamer()
    
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer
        )
        
    decoded_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    for b in range(args.batch_size):
        suffix = f"_b{b}" if args.batch_size > 1 else ""
        out_path = os.path.join(args.output_dir, f"{batch_filenames[b]}{suffix}_hf.out.txt")
        with open(out_path, 'w') as f:
            f.write(f"Prompt:\n{batch_prompts[b]}\n\nGenerated:\n{decoded_texts[b]}\n")
        
    torch.cuda.synchronize()
    end_time = time.perf_counter()

    prefill_time = streamer.first_token_time - start_time
    total_time = end_time - start_time
    decode_time = total_time - prefill_time

    total_decode_tokens = args.batch_size * args.max_new_tokens
    prefill_tok_sec = (args.batch_size * seq_len) / prefill_time if prefill_time > 0 else 0
    decode_tok_sec = total_decode_tokens / decode_time if decode_time > 0 else 0

    prefill_ms = prefill_time * 1000
    decode_ms = decode_time * 1000
    total_ms = total_time * 1000

    csv_path = os.path.join(args.output_dir, f"pipeline_benchmark_hf_{args.session_id}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["batch_size", "seq_len", "max_new_tokens", "prefill_ms", "decode_ms", "total_ms", "prefill_tok_sec", "decode_tok_sec", "prefill_ms_gpu", "decode_ms_gpu", "total_ms_gpu", "prefill_tok_sec_gpu", "decode_tok_sec_gpu"])
        writer.writerow([args.batch_size, seq_len, args.max_new_tokens, prefill_ms, decode_ms, total_ms, prefill_tok_sec, decode_tok_sec, "", "", "", "", ""])

    print(f"HF baseline complete. CSV saved to {csv_path}")

if __name__ == "__main__":
    main()
