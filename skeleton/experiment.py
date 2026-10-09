"""Example runner for a shared split, forest search, and final evaluation.

Adapt this flow to your experimental design. Results stay in memory; choose how
to save them and record the settings needed to reproduce your study.
"""

from __future__ import annotations

import argparse
import json
import csv
from datetime import datetime
import numpy as np

from pathlib import Path
from time import perf_counter
from typing import Any

from data_loading import DATASETS, load_and_split, prepare_data, prepare_final_data
from hyperband import optimise_hyperband
from random_forest import final_test_evaluation, make_evaluator
from random_search import optimise_random_search
from smbo import optimise_smbo
from tabular_foundation import run_foundation_model

# Update this if the provided largest dataset is replaced.
FOUNDATION_DATASET = "covertype"

# n_trials is the example evaluation budget for each of Random Search and SMBO.
# Choose budgets and a Hyperband schedule that support your justified comparison.
PROFILES: dict[str, dict[str, Any]] = {
    "smoke": {
        "max_samples": 2_500,
        "min_trees": 3,
        "max_trees": 27,
        "n_trials": 9,
    },
    "course": {
        "max_samples": None,
        "min_trees": 3,
        "max_trees": 81,
        "n_trials": 16,
    },
    "full": {
        "max_samples": None,
        "min_trees": 3,
        "max_trees": 243,
        "n_trials": 24,
    },
}

# tabpfn_sk_KSiX8TO6DED_f8l3lGWEgORCow0fER2XioKkVNIIyhM

def parse_args() -> argparse.Namespace:
    """Parse the reproducible experiment command-line options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="covertype", choices=[*DATASETS, "all"])
    parser.add_argument("--profile", default="course", choices=PROFILES)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["default", "random", "smbo", "hyperband", "foundation"],
        choices=["default", "random", "smbo", "hyperband", "foundation"],
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    return parser.parse_args()


def run_dataset(name: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Return example in-memory results; their structure is yours to adapt."""

    if args.methods == ["foundation"] and name != FOUNDATION_DATASET:
        if args.dataset == "all":
            return []
        raise ValueError(f"Foundation-only runs require --dataset {FOUNDATION_DATASET}")

    profile = PROFILES[args.profile]
    splits = load_and_split(name, args.cache_dir, profile["max_samples"], args.split_seed)
    results = []
    forest_methods = {"default", "random", "smbo", "hyperband"}.intersection(args.methods)
    if forest_methods:
        X_train, X_valid = prepare_data(splits)
        evaluator = make_evaluator(
            X_train, splits.y_train, X_valid, splits.y_valid
        )
        final_arrays = prepare_final_data(splits)
        min_trees = int(profile["min_trees"])
        max_trees = int(profile["max_trees"])

        for method in ("default", "random", "smbo", "hyperband"):
            if method not in forest_methods:
                continue
            start = perf_counter()
            if method == "default":
                config = {}  # Library defaults, with the common tree count.
                history = [evaluator(config, max_trees, args.seed)]
            elif method == "random":
                config, history = optimise_random_search(
                    evaluator, profile["n_trials"], max_trees, args.seed
                )
            elif method == "smbo":
                config, history = optimise_smbo(
                    evaluator, profile["n_trials"], max_trees, args.seed
                )
            else:
                config, history = optimise_hyperband(
                    evaluator, min_trees, max_trees, args.seed
                )
            search_seconds = perf_counter() - start
            final_result = final_test_evaluation(
                config, max_trees, args.seed, *final_arrays
            )
            print(f"{name} / {method} / seed {args.seed}: {final_result}", flush=True)
            results.append({
                "dataset": name,
                "method": method,
                "seed": args.seed,
                "configuration": config,
                "history": history,
                "search_seconds": search_seconds,
                "final_result": final_result,
            })

    if "foundation" in args.methods and name == FOUNDATION_DATASET:
        result = run_foundation_model(splits, seed=args.seed)
        print(f"{name} / foundation / seed {args.seed}: {result}", flush=True)
        results.append({
            "dataset": name,
            "method": "foundation",
            "seed": args.seed,
            "result": result,
        })
    return results


def main() -> None:
    """Run the selected examples; add result saving before the main study."""

    args = parse_args()
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for name in names:
        print(f"Running {name} with seed {args.seed}", flush=True)
        results = run_dataset(name, args)
        save_results(results, args, name)

        # TODO: save results in a format of your choice, along with the settings
        # needed to reproduce the run. Retain enough information for your plots
        # and tables. This example only prints final results; it saves no files.

def json_default(value):
    # Converts common python and numpy objects into compatible values.
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)

    raise TypeError(f"Cannot serialize {type(value).__name__}")

def save_results(results, args, dataset_name):
    # Save full experiment results and a flat CSV summary
    output_dir = Path("experiment_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (
        f"{dataset_name}_{args.profile}_seed{args.seed}_{timestamp}"
    )

    # Record the settings needed to understand and reproduce the run.
    record = {
        "dataset": dataset_name,
        "profile": args.profile,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "methods": args.methods,
        "profile_settings": PROFILES[args.profile],
        "results": results,
    }

    # Keep large prediction arrays out of the JSON file.
    for result in record["results"]:
        if result["method"] == "foundation":
            foundation = result["result"]

            predictions = foundation.pop("predictions", None)
            y_test = foundation.pop("y_test", None)

            if predictions is not None and y_test is not None:
                predictions_path = output_dir / f"{run_name}_predictions.npz"
                np.savez_compressed(
                    predictions_path,
                    predictions=predictions,
                    y_test=y_test,
                )
                foundation["predictions_file"] = str(predictions_path)

    # Save the complete record.
    json_path = output_dir / f"{run_name}.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(record, file, indent=2, default=json_default)

    # Create one flat row per method for easier comparison.
    summary_rows = []
    for result in results:
        row = {
            "dataset": result["dataset"],
            "method": result["method"],
            "seed": result["seed"],
        }

        if result["method"] == "foundation":
            details = result["result"]
            row.update({
                "search_seconds": None,
                "fit_seconds": details.get("fit_seconds"),
                "prediction_seconds": details.get("prediction_seconds"),
                "total_seconds": details.get("total_seconds"),
                "context_size": details.get("context_size"),
            })
            metrics = details.get("metrics", {})
        else:
            row["search_seconds"] = result.get("search_seconds")
            row["fit_seconds"] = None
            row["prediction_seconds"] = None
            row["total_seconds"] = None
            row["context_size"] = None
            metrics = result["final_result"].get("metrics", {})

        for metric in ("accuracy", "precision", "recall", "f1"):
            row[metric] = metrics.get(metric)

        row["configuration"] = json.dumps(
            result.get("configuration", {}),
            default=json_default,
        )
        summary_rows.append(row)

    csv_path = output_dir / f"{run_name}_summary.csv"
    if summary_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file, fieldnames=list(summary_rows[0].keys())
            )
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"Saved full results to {json_path}")
    print(f"Saved summary to {csv_path}")


if __name__ == "__main__":
    main()
