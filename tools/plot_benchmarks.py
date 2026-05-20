import argparse
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser(description="Generate benchmark comparison charts")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Target a specific run by name (e.g. run_20260521_012525). "
                             "If not set, uses the latest run per engine.")
    parser.add_argument("--output", type=str, default="docs/benchmark_comparison.png",
                        help="Output path for the chart image")
    parser.add_argument("--subtitle", type=str, default=None,
                        help="Optional subtitle shown below the main title")
    args = parser.parse_args()

    runs_json_path = "docs/performance_testing/runs.json"
    with open(runs_json_path, 'r') as f:
        runs = json.load(f)

    if args.run_name:
        # Pick runs matching the specified run name
        selected_runs = {}
        for r in runs:
            if r.get("name") == args.run_name:
                selected_runs[r.get("engine", "Custom CPP")] = r
    else:
        # Get latest run for each engine
        selected_runs = {}
        for r in runs:
            engine = r.get("engine", "Custom CPP")
            if engine not in selected_runs:
                selected_runs[engine] = r
            else:
                if r["timestamp"] > selected_runs[engine]["timestamp"]:
                    selected_runs[engine] = r

    data = []
    max_new_tokens = None
    for engine, r in selected_runs.items():
        csv_path = os.path.join("docs/performance_testing", r["pipeline_csv"])
        if not os.path.exists(csv_path):
            continue
            
        df = pd.read_csv(csv_path)
        # Assuming just one row per engine right now since we unified the dataset
        row = df.iloc[0]
        if max_new_tokens is None:
            max_new_tokens = int(row["max_new_tokens"])
        
        data.append({
            "Engine": engine,
            "Prefill Tokens/s": row["prefill_tok_sec"],
            "Decode Tokens/s": row["decode_tok_sec"],
            "Prefill Latency (ms)": row["prefill_ms"],
            "Decode Latency (ms)": row["decode_ms"],
            "Total Latency (ms)": row["total_ms"]
        })
        
    if not data:
        print("No data found!")
        return
        
    df_plot = pd.DataFrame(data)
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    title = 'Transformer Inference Engines Performance on GTX 1050 Ti'
    subtitle = args.subtitle or (f"{max_new_tokens} Output Tokens" if max_new_tokens else "")
    if subtitle:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.99)
        fig.text(0.5, 0.955, subtitle, ha='center', fontsize=12, style='italic', color='gray')
    else:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    metrics = [
        ("Prefill Tokens/s", axes[0, 0], "Tokens / sec (Higher is better)", sns.color_palette("Blues_d")),
        ("Decode Tokens/s", axes[0, 1], "Tokens / sec (Higher is better)", sns.color_palette("Greens_d")),
        ("Prefill Latency (ms)", axes[1, 0], "Latency in ms (Lower is better)", sns.color_palette("Oranges_d")),
        ("Decode Latency (ms)", axes[1, 1], "Latency in ms (Lower is better)", sns.color_palette("Reds_d"))
    ]
    
    for metric, ax, ylabel, palette in metrics:
        df_sorted = df_plot.sort_values(metric, ascending=(not "Latency" in metric))
        sns.barplot(x="Engine", y=metric, data=df_sorted, ax=ax, hue="Engine", palette=palette, legend=False)
        ax.set_title(metric, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=15)
        
        # Add value labels
        for p in ax.patches:
            val = p.get_height()
            ax.annotate(f"{val:.1f}", (p.get_x() + p.get_width() / 2., val),
                        ha='center', va='bottom', fontsize=10, xytext=(0, 5),
                        textcoords='offset points')
                        
    plt.tight_layout(pad=2.0)
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {args.output}")

if __name__ == "__main__":
    main()
