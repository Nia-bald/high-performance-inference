import os
import sys
import time
import csv
import argparse
import subprocess

try:
    import torch
    import transformers
    import ctranslate2
except ImportError:
    print("Error: ctranslate2 or transformers not installed.")
    sys.exit(1) 

def main():
    parser = argparse.ArgumentParser(description="CTranslate2 Baseline Pipeline Benchmark")
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="./docs/performance_testing")
    parser.add_argument("--session-id", type=str, default="test")
    args = parser.parse_args()

    # Model paths
    hf_model_id = "gpt2"
    model_dir = "gpt2-ct2"
    
    if not os.path.exists(model_dir):
        print(f"Converting HF {hf_model_id} to CTranslate2 format...")
        subprocess.run(["ct2-transformers-converter", "--model", hf_model_id, 
                        "--output_dir", model_dir, "--quantization", "int8_float16", 
                        "--copy_files", "tokenizer.json", "vocab.json", "merges.txt"], check=True)

    print("Loading CTranslate2 GPT-2...")
    tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
    generator = ctranslate2.Generator(model_dir, device="cuda")

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

    tokens_batch = [tokenizer.tokenize(p) for p in batch_prompts]
    seq_len = len(tokens_batch[0])

    print("CTranslate2 Warmup...")
    generator.generate_batch(tokens_batch, max_length=2, sampling_temperature=0.0)

    print("CTranslate2 Benchmarking...")
    
    # 1. Measure Prefill (Time to First Token) by generating exactly 1 token
    start_prefill = time.perf_counter()
    generator.generate_batch(tokens_batch, max_length=1, min_length=1, sampling_temperature=0.0)
    end_prefill = time.perf_counter()
    prefill_time = end_prefill - start_prefill

    # 2. Measure Total Pipeline Time (Prefill + Decode)
    start_total = time.perf_counter()
    results = generator.generate_batch(tokens_batch, max_length=args.max_new_tokens, min_length=args.max_new_tokens, sampling_temperature=0.0, include_prompt_in_result=False)
    end_total = time.perf_counter()
    total_time = end_total - start_total

    decoded_texts = [tokenizer.decode(res.sequences_ids[0]) for res in results]
    for b in range(args.batch_size):
        suffix = f"_b{b}" if args.batch_size > 1 else ""
        out_path = os.path.join(args.output_dir, f"{batch_filenames[b]}{suffix}_ct2.out.txt")
        with open(out_path, 'w') as f:
            f.write(f"Prompt:\n{batch_prompts[b]}\n\nGenerated:\n{decoded_texts[b]}\n")

    # 3. Calculate Decode
    decode_time = total_time - prefill_time
    
    # Total run generated args.max_new_tokens. The first token was prefill.
    # Therefore, decode generated (max_new_tokens - 1) tokens.
    total_decode_tokens = args.batch_size * (args.max_new_tokens - 1)
    
    prefill_tok_sec = (args.batch_size * seq_len) / prefill_time if prefill_time > 0 else 0
    decode_tok_sec = total_decode_tokens / decode_time if decode_time > 0 else 0

    csv_path = os.path.join(args.output_dir, f"pipeline_benchmark_ct2_{args.session_id}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["batch_size", "seq_len", "max_new_tokens", "prefill_ms", "decode_ms", "total_ms", "prefill_tok_sec", "decode_tok_sec", "prefill_ms_gpu", "decode_ms_gpu", "total_ms_gpu", "prefill_tok_sec_gpu", "decode_tok_sec_gpu"])
        writer.writerow([args.batch_size, seq_len, args.max_new_tokens, prefill_time*1000, decode_time*1000, total_time*1000, prefill_tok_sec, decode_tok_sec, "", "", "", "", ""])

    print(f"CTranslate2 baseline complete. CSV saved to {csv_path}")

if __name__ == "__main__":
    main()
