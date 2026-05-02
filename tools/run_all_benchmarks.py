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

    suffix_map = {
        "Custom CPP": "",
        "HuggingFace": "_hf",
        "CTranslate2": "_ct2",
        "llama.cpp": "_llama",
        "vLLM": "_vllm",
        "ONNX Runtime": "_onnx"
    }
    suffix = suffix_map.get(engine_name, "")

    new_run = {
        "engine": engine_name,
        "timestamp": iso_timestamp,
        "name": run_name,
        "label": label,
        "pipeline_csv": f"{run_name}/pipeline_benchmark{suffix}_{session_id}.csv",
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
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens to generate")
    args = parser.parse_args()

    session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"================================================================")
    print(f"  Starting Multi-Engine Benchmark Session: {session_id}")
    print(f"================================================================")

    output_dir = f"./docs/performance_testing/run_{session_id}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Output directory created: {output_dir}")
    print(f"  ALL generated text outputs and CSV reports will be saved directly into this directory!\n")

    runs_json_path = "./docs/performance_testing/runs.json"
    python_executable = sys.executable

    engines = [
        {
            "name": "Custom CPP",
            "type": "binary",
            "path": "./build/bench_performance",
            "enabled": True,
            "required": True
        },
        {
            "name": "HuggingFace",
            "type": "python",
            "path": "./tools/hf_baseline.py",
            "enabled": True,
            "required": False
        },
        # {
        #     "name": "vLLM",
        #     "type": "python",
        #     "path": "./tools/vllm_baseline.py",
        #     "enabled": False, # Commented out due to sm_61 incompatibility
        #     "required": False
        # },
        {
            "name": "CTranslate2",
            "type": "python",
            "path": "./tools/ct2_baseline.py",
            "enabled": True,
            "required": False
        },
        {
            "name": "ONNX Runtime",
            "type": "python",
            "path": "./tools/onnx_baseline.py",
            "enabled": False,
            "required": False
        },
        {
            "name": "llama.cpp",
            "type": "python",
            "path": "./tools/llama_baseline.py",
            "enabled": True,
            "required": False
        }
    ]

    for engine in engines:
        if not engine["enabled"]:
            continue
            
        executable_path = engine["path"]
        engine_name = engine["name"]
        
        if not os.path.exists(executable_path):
            if engine.get("required"):
                print(f"Error: {executable_path} not found for {engine_name}. Please build the project or install prerequisites.")
                sys.exit(1)
            else:
                print(f"Warning: {executable_path} not found. Skipping {engine_name} baseline.")
                continue

        command = []
        if engine["type"] == "python":
            command.append(python_executable)
        command.append(executable_path)
        
        command.extend([
            "--dataset-dir", args.dataset_dir,
            "--batch-size", str(args.batch_size),
            "--max-new-tokens", str(args.max_new_tokens),
            "--output-dir", output_dir,
            "--session-id", session_id
        ])

        print(f">>> Running {engine_name}...")
        try:
            subprocess.run(command, check=True)
            print(f">>> {engine_name} Benchmark Complete.\n")
            update_runs_json(runs_json_path, session_id, engine_name)
        except subprocess.CalledProcessError as e:
            print(f"Error: {engine_name} failed with exit code {e.returncode}")

if __name__ == "__main__":
    main()
