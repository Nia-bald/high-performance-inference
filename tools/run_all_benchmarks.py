#!/usr/bin/env python3
import os
import sys
import json
import argparse
import datetime
import subprocess

def update_runs_json(runs_json_path, session_id, engine_name):
    runs = []
    if os.path.exists(runs_json_path):
        try:
            with open(runs_json_path, 'r') as f:
                runs = json.load(f)
        except Exception as e:
            print(f"Warning: Could not read {runs_json_path}, creating a new one. Error: {e}")

    # Parse session_id to format ISO timestamp and label
    try:
        dt = datetime.datetime.strptime(session_id, "%Y%m%d_%H%M%S")
        label_time = dt.strftime("%Y-%m-%d %H:%M:%S")
        iso_timestamp = dt.isoformat() + "Z"
    except ValueError:
        label_time = session_id
        iso_timestamp = session_id

    run_name = f"run_{session_id}"
    label = f"Run {label_time} ({engine_name})"

    new_run = {
        "engine": engine_name,
        "timestamp": iso_timestamp,
        "name": run_name,
        "label": label,
        "pipeline_csv": f"{run_name}/pipeline_benchmark_{session_id}.csv",
        "kernel_csv": f"{run_name}/kernel_benchmark_{session_id}.csv" if engine_name == "Custom CPP" else None
    }

    runs.append(new_run)

    with open(runs_json_path, 'w') as f:
        json.dump(runs, f, indent=4)
        
    print(f"  Dashboard updated: {runs_json_path}\n")

def main():
    parser = argparse.ArgumentParser(description="Multi-Engine Inference Benchmark Orchestrator")
    parser.add_argument("--dataset-dir", type=str, default="./dataset/input", help="Path to input prompts")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for benchmark")
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Max new tokens to generate")
    args = parser.parse_args()

    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"================================================================")
    print(f"  Starting Multi-Engine Benchmark Session: {session_id}")
    print(f"================================================================")

    output_dir = f"./docs/performance_testing/run_{session_id}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Output directory created: {output_dir}")
    print(f"  ALL generated text outputs and CSV reports will be saved directly into this directory!\n")

    # 1. Run Custom C++ Engine
    cpp_executable = "./build/bench_performance"
    if not os.path.exists(cpp_executable):
        print(f"Error: C++ executable not found at {cpp_executable}. Please build the project before running benchmarks.")
        sys.exit(1)

    cpp_command = [
        cpp_executable,
        "--dataset-dir", args.dataset_dir,
        "--batch-size", str(args.batch_size),
        "--max-new-tokens", str(args.max_new_tokens),
        "--output-dir", output_dir,
        "--session-id", session_id
    ]

    runs_json_path = "./docs/performance_testing/runs.json"

    print(f">>> Running Custom C++ Engine...")
    try:
        subprocess.run(cpp_command, check=True)
        print(">>> C++ Engine Benchmark Complete.\n")
        update_runs_json(runs_json_path, session_id, "Custom CPP")
    except subprocess.CalledProcessError as e:
        print(f"Error: C++ Engine failed with exit code {e.returncode}")

    # 2. Run HuggingFace Baseline
    python_executable = sys.executable
    hf_script = "./tools/hf_baseline.py"
    if os.path.exists(hf_script):
        hf_command = [
            python_executable, hf_script,
            "--dataset-dir", args.dataset_dir,
            "--batch-size", str(args.batch_size),
            "--max-new-tokens", str(args.max_new_tokens),
            "--output-dir", output_dir,
            "--session-id", session_id
        ]
        print(f">>> Running HuggingFace Baseline...")
        try:
            subprocess.run(hf_command, check=True)
            print(">>> HF Benchmark Complete.\n")
            update_runs_json(runs_json_path, session_id, "HuggingFace")
        except subprocess.CalledProcessError as e:
            print(f"Error: HF Baseline failed with exit code {e.returncode}")
    else:
        print(f"Warning: {hf_script} not found. Skipping HuggingFace baseline.")

    # 3. Run vLLM Baseline (Commented out due to sm_61 incompatibility)
    # vllm_script = "./tools/vllm_baseline.py"
    # if os.path.exists(vllm_script):
    #     vllm_command = [
    #         python_executable, vllm_script,
    #         "--dataset-dir", args.dataset_dir,
    #         "--batch-size", str(args.batch_size),
    #         "--max-new-tokens", str(args.max_new_tokens),
    #         "--output-dir", output_dir,
    #         "--session-id", session_id
    #     ]
    #     print(f">>> Running vLLM Baseline...")
    #     try:
    #         subprocess.run(vllm_command, check=True)
    #         print(">>> vLLM Benchmark Complete.\n")
    #         update_runs_json(runs_json_path, session_id, "vLLM")
    #     except subprocess.CalledProcessError as e:
    #         print(f"Error: vLLM Baseline failed with exit code {e.returncode}")

    # 4. Run CTranslate2
    ct2_script = "./tools/ct2_baseline.py"
    if os.path.exists(ct2_script):
        ct2_command = [
            python_executable, ct2_script,
            "--dataset-dir", args.dataset_dir,
            "--batch-size", str(args.batch_size),
            "--max-new-tokens", str(args.max_new_tokens),
            "--output-dir", output_dir,
            "--session-id", session_id
        ]
        print(f">>> Running CTranslate2 Baseline...")
        try:
            subprocess.run(ct2_command, check=True)
            print(">>> CTranslate2 Benchmark Complete.\n")
            update_runs_json(runs_json_path, session_id, "CTranslate2")
        except subprocess.CalledProcessError as e:
            print(f"Error: CTranslate2 Baseline failed with exit code {e.returncode}")

    # 5. Run ONNX Runtime
    # onnx_script = "./tools/onnx_baseline.py"
    # if os.path.exists(onnx_script):
    #     onnx_command = [
    #         python_executable, onnx_script,
    #         "--dataset-dir", args.dataset_dir,
    #         "--batch-size", str(args.batch_size),
    #         "--max-new-tokens", str(args.max_new_tokens),
    #         "--output-dir", output_dir,
    #         "--session-id", session_id
    #     ]
    #     print(f">>> Running ONNX Runtime Baseline...")
    #     try:
    #         subprocess.run(onnx_command, check=True)
    #         print(">>> ONNX Benchmark Complete.\n")
    #         update_runs_json(runs_json_path, session_id, "ONNX Runtime")
    #     except subprocess.CalledProcessError as e:
    #         print(f"Error: ONNX Baseline failed with exit code {e.returncode}")

    # 6. Run llama.cpp
    llama_script = "./tools/llama_baseline.py"
    if os.path.exists(llama_script):
        llama_command = [
            python_executable, llama_script,
            "--dataset-dir", args.dataset_dir,
            "--batch-size", str(args.batch_size),
            "--max-new-tokens", str(args.max_new_tokens),
            "--output-dir", output_dir,
            "--session-id", session_id
        ]
        print(f">>> Running llama.cpp Baseline...")
        try:
            subprocess.run(llama_command, check=True)
            print(">>> llama.cpp Benchmark Complete.\n")
            update_runs_json(runs_json_path, session_id, "llama.cpp")
        except subprocess.CalledProcessError as e:
            print(f"Error: llama.cpp Baseline failed with exit code {e.returncode}")

if __name__ == "__main__":
    main()
