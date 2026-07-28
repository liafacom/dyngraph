#!/usr/bin/env python3
"""Command-line entry point for the DynGraphBERT experiments."""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import pandas as pd

DATASET_NAMES = ("ohsumed", "r8", "agnews", "snippets", "dblp")
DEFAULT_SEEDS = (11, 35, 8, 3, 23)

def parse_args():
    parser = argparse.ArgumentParser(description="Run the semi-supervised DynGraphBERT experiment.")
    parser.add_argument("--dataset", choices=DATASET_NAMES, nargs="+", default=list(DATASET_NAMES))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--setup", choices=("A", "B", "C"), default="C")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases (disabled by default).")
    parser.add_argument("--wandb-project")
    parser.add_argument("--model", dest="dir_model", help="Local Hugging Face model directory.")
    return parser.parse_args()

def ensure_directories():
    for path in ("artifacts/results", "artifacts/models", "artifacts/embeddings", "artifacts/logs", "artifacts/imgs"):
        Path(path).mkdir(parents=True, exist_ok=True)

def dataset_registry():
    from utils.utils import get_agnew, get_dblp, get_ohsumed, get_r8, get_snippets
    return {"ohsumed": get_ohsumed, "r8": get_r8, "agnews": get_agnew, "snippets": get_snippets, "dblp": get_dblp}

def main():
    args = parse_args()
    os.environ.setdefault("WANDB_MODE", "online" if args.wandb else "disabled")
    ensure_directories()
    from models.imodel_simple_gcn_semisup import run_experiment2
    registry = dataset_registry()
    all_results = []
    for dataset_name in args.dataset:
        Path("artifacts/logs", dataset_name).mkdir(parents=True, exist_ok=True)
        Path("artifacts/imgs", dataset_name).mkdir(parents=True, exist_ok=True)
        settings = {"setup": args.setup, "pre_train": False}
        if args.wandb_project:
            settings["proj"] = args.wandb_project
        if args.dir_model:
            settings["dir_model"] = args.dir_model
        for seed in args.seeds:
            result = run_experiment2(seed, registry[dataset_name], settings.copy())
            result["requested_dataset"] = dataset_name
            all_results.append(dict(result))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_base = Path("artifacts/results") / f"dyngraphbert_{timestamp}"
    frame = pd.DataFrame(all_results)
    frame.to_pickle(output_base.with_suffix(".pkl"))
    csv_frame = frame.drop(columns=["prediction", "predict_val"], errors="ignore")
    csv_frame.to_csv(output_base.with_suffix(".csv"), index=False)
    with output_base.with_suffix(".json").open("w") as handle:
        json.dump(csv_frame.to_dict(orient="records"), handle, indent=2, default=str)
    print(f"Results saved to {output_base}.[csv|json|pkl]")

if __name__ == "__main__":
    main()
