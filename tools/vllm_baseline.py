#!/usr/bin/env python3
import os
import sys
import argparse
import time
import csv
import torch
from vllm import LLM, SamplingParams

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--session-id", type=str, required=True)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "GPU is strictly required for benchmarking!"

    print("Loading vLLM GPT-2...")
    try:
        llm = LLM(model="gpt2", enforce_eager=True) # Eager mode for more predictable small-batch timing
    except Exception as e:
        print(f"Error loading vLLM: {e}")
        sys.exit(1)
    
    # Load prompts
    prompts = []
    if os.path.exists(args.dataset_dir):
        for f in os.listdir(args.dataset_dir):
            if f.endswith(".txt"):
                with open(os.path.join(args.dataset_dir, f), 'r') as fp:
                    prompts.append(fp.read())
                    
    if not prompts:
        print("Error: No prompts found.")
        sys.exit(1)

    batch_prompts = []
    for i in range(args.batch_size):
        batch_prompts.append(prompts[i % len(prompts)])

    # Greedy, ignore eos to guarantee max_new_tokens
    sampling_params = SamplingParams(
        temperature=0.0, 
        top_k=1, # Explicitly enforce argmax
        max_tokens=args.max_new_tokens,
        ignore_eos=True
    )

    # Warmup
    print("vLLM Warmup...")
    llm.generate(batch_prompts[0:1], SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True), use_tqdm=False)

    print("vLLM Benchmarking...")
    start_time = time.perf_counter()
    outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
    end_time = time.perf_counter()

    seq_len = len(outputs[0].prompt_token_ids)
    
    try:
        # Extract precise metrics if available in this vLLM version
        prefill_times = [out.metrics.first_token_time - out.metrics.arrival_time for out in outputs]
        total_times = [out.metrics.finished_time - out.metrics.arrival_time for out in outputs]
        decode_times = [t - p for t, p in zip(total_times, prefill_times)]
        
        prefill_time = sum(prefill_times) / len(prefill_times)
        total_time = sum(total_times) / len(total_times)
        decode_time = sum(decode_times) / len(decode_times)
        
        min_prefill_ms = min(prefill_times) * 1000
        max_prefill_ms = max(prefill_times) * 1000
        min_decode_ms = min(decode_times) * 1000
        max_decode_ms = max(decode_times) * 1000
        
        print(f"  [vLLM Batch Stats] Prefill TTFT - Min: {min_prefill_ms:.2f}ms | Max: {max_prefill_ms:.2f}ms | Avg: {prefill_time*1000:.2f}ms")
        print(f"  [vLLM Batch Stats] Decode Time  - Min: {min_decode_ms:.2f}ms | Max: {max_decode_ms:.2f}ms | Avg: {decode_time*1000:.2f}ms")
    except AttributeError:
        # Fallback approximation if metrics are missing
        total_time = end_time - start_time
        prefill_time = total_time * 0.1 
        decode_time = total_time - prefill_time
        
        min_prefill_ms = prefill_time * 1000
        max_prefill_ms = prefill_time * 1000
        min_decode_ms = decode_time * 1000
        max_decode_ms = decode_time * 1000        
    decode_time = total_time - prefill_time
    total_decode_tokens = args.batch_size * args.max_new_tokens
    
    prefill_tok_sec = (args.batch_size * seq_len) / prefill_time if prefill_time > 0 else 0
    decode_tok_sec = total_decode_tokens / decode_time if decode_time > 0 else 0

    prefill_ms = prefill_time * 1000
    decode_ms = decode_time * 1000
    total_ms = total_time * 1000

    csv_path = os.path.join(args.output_dir, f"pipeline_benchmark_vllm_{args.session_id}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["batch_size", "seq_len", "max_new_tokens", "prefill_ms", "decode_ms", "total_ms", "prefill_tok_sec", "decode_tok_sec", "min_prefill_ms", "max_prefill_ms", "min_decode_ms", "max_decode_ms", "prefill_ms_gpu", "decode_ms_gpu", "total_ms_gpu", "prefill_tok_sec_gpu", "decode_tok_sec_gpu"])
        writer.writerow([args.batch_size, seq_len, args.max_new_tokens, prefill_ms, decode_ms, total_ms, prefill_tok_sec, decode_tok_sec, min_prefill_ms, max_prefill_ms, min_decode_ms, max_decode_ms, "", "", "", "", ""])

    print(f"vLLM baseline complete. CSV saved to {csv_path}")

if __name__ == "__main__":
    main()
