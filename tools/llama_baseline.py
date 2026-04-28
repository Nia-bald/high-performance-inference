#!/usr/bin/env python3
import os
import sys
import argparse
import time
import csv

try:
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama
except ImportError:
    print("Error: llama-cpp-python or huggingface_hub not installed.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--session-id", type=str, required=True)
    args = parser.parse_args()

    # Download GGUF (Using Q8_0 as a standard quantized comparison, since F16 is rare for gpt2 on Hub)
    # TheBloke/gpt2-GGUF has a Q8_0 model that works very well.
    print("Downloading/Loading GPT-2 GGUF...")
    try:
        model_path = hf_hub_download(repo_id="QuantFactory/gpt2-GGUF", filename="gpt2.Q8_0.gguf")
    except Exception as e:
        print("Error downloading GGUF: ", e)
        sys.exit(1)

    print("Loading llama.cpp GPT-2...")
    llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=2048, verbose=False)

    prompts = []
    filenames = []
    if os.path.exists(args.dataset_dir):
        for f in os.listdir(args.dataset_dir):
            if f.endswith(".txt"):
                with open(os.path.join(args.dataset_dir, f), 'r') as fp:
                    prompts.append(fp.read())
                    filenames.append(f)

    batch_prompts = []
    batch_filenames = []
    for i in range(args.batch_size):
        idx = i % len(prompts)
        batch_prompts.append(prompts[idx])
        batch_filenames.append(filenames[idx])

    print("llama.cpp Warmup...")
    llm.create_completion("Warmup", max_tokens=2, temperature=0.0)

    print("llama.cpp Benchmarking...")
    
    # 1. Measure Prefill Time
    start_prefill = time.perf_counter()
    for p in batch_prompts:
        llm.create_completion(p, max_tokens=1, temperature=0.0)
    end_prefill = time.perf_counter()
    prefill_time = end_prefill - start_prefill

    # 2. Measure Total Time
    start_total = time.perf_counter()
    outputs = []
    for p in batch_prompts:
        outputs.append(llm.create_completion(p, max_tokens=args.max_new_tokens, temperature=0.0))
    end_total = time.perf_counter()
    total_time = end_total - start_total

    for b, out in enumerate(outputs):
        generated = out['choices'][0]['text']
        suffix = f"_b{b}" if args.batch_size > 1 else ""
        out_path = os.path.join(args.output_dir, f"{batch_filenames[b]}{suffix}_llama.out.txt")
        with open(out_path, 'w') as f:
            f.write(f"Prompt:\n{batch_prompts[b]}\n\nGenerated:\n{generated}\n")

    decode_time = total_time - prefill_time
    seq_len = sum(o['usage']['prompt_tokens'] for o in outputs) // args.batch_size

    total_decode_tokens = args.batch_size * (args.max_new_tokens - 1)
    prefill_tok_sec = (args.batch_size * seq_len) / prefill_time if prefill_time > 0 and seq_len > 0 else 0
    decode_tok_sec = total_decode_tokens / decode_time if decode_time > 0 else 0

    csv_path = os.path.join(args.output_dir, f"pipeline_benchmark_llama_{args.session_id}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["batch_size", "seq_len", "max_new_tokens", "prefill_ms", "decode_ms", "total_ms", "prefill_tok_sec", "decode_tok_sec", "prefill_ms_gpu", "decode_ms_gpu", "total_ms_gpu", "prefill_tok_sec_gpu", "decode_tok_sec_gpu"])
        writer.writerow([args.batch_size, seq_len, args.max_new_tokens, prefill_time*1000, decode_time*1000, total_time*1000, prefill_tok_sec, decode_tok_sec, "", "", "", "", ""])

    print(f"llama.cpp baseline complete. CSV saved to {csv_path}")

if __name__ == "__main__":
    main()
