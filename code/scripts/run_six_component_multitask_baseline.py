#!/usr/bin/env python
"""Train/evaluate a direct six-component multitask ALIGNN baseline.

This script is intended as a baseline against the routed/subtask tensor
workflow. It trains one ALIGNN model with six graph-level outputs:

    xx, yy, zz, xy, xz, yz

The target transforms are the existing ``six_transformed`` transforms from
``prepare_polarizability_alignn.py``:

    log1p(xx), log1p(yy), log1p(zz / 0.033699), asinh(xy), asinh(xz), asinh(yz)
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from cross_validate_polarizability_tensor import (
    aggregate_predictions,
    cv_splits_for_submodel,
    write_split_artifacts,
)
from evaluate_polarizability_tensor import (
    COMPONENTS,
    build_summary,
    load_split,
    tensor_from_components,
)
from prepare_polarizability_alignn import load_labels
from train_polarizability_tensor import (
    deep_update,
    format_elapsed,
    has_prediction_file,
    load_workflow_config,
    prepare_submodel,
    refresh_raw_tail_loss_config,
    refresh_submodel,
    train_submodel,
)


SIX_COMPONENT_SPEC = [
    ("xx", "log1p"),
    ("yy", "log1p"),
    ("zz", "scaled_log1p_zz"),
    ("xy", "asinh"),
    ("xz", "asinh"),
    ("yz", "asinh"),
]
SIX_COMPONENT_COLUMNS = [column for column, _ in SIX_COMPONENT_SPEC]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a single multitask ALIGNN baseline for the six independent "
            "polarizability tensor components."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/polarizability_tensor_config.json",
        help=(
            "Existing tensor workflow config used for labels, structures, "
            "descriptors, and default model/training hyperparameters."
        ),
    )
    parser.add_argument("--labels", default=None)
    parser.add_argument("--structures", default=None)
    parser.add_argument("--descriptors", default=None)
    parser.add_argument(
        "--task-dir",
        default=None,
        help=(
            "Output run directory. The default name includes the configured "
            "stratification mode when needed to preserve earlier baselines."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Prepared data directory. The default name includes the configured "
            "stratification mode when needed to preserve earlier datasets."
        ),
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--base-submodel",
        choices=["inplane", "outplane", "outplane_aux"],
        default="inplane",
        help=(
            "Which existing submodel block supplies default hyperparameters "
            "when the config has no six_component block."
        ),
    )
    parser.add_argument(
        "--baseline-key",
        default="six_component",
        help=(
            "Optional config block name whose training/model entries override "
            "the inherited defaults."
        ),
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Run out-of-fold cross validation instead of one train/val/test split.",
    )
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Retrain even when prediction files already exist.",
    )
    parser.add_argument(
        "--keep-existing-splits",
        action="store_true",
        help="Reuse existing ids_train_val_test.json files in CV mode.",
    )
    return parser.parse_args()


def apply_path_overrides(config: dict, args: argparse.Namespace) -> dict:
    config = deep_update({}, config)
    for key in ["labels", "structures", "descriptors"]:
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    return config


def build_baseline_config(config: dict, args: argparse.Namespace) -> dict:
    """Return one split_cfg dict compatible with prepare/train helpers."""
    base_submodel = config.get(args.base_submodel, {})
    explicit = config.get(args.baseline_key, {})

    training = deep_update(config.get("training", {}), base_submodel.get("training", {}))
    training = deep_update(training, explicit.get("training", {}))
    if training.get("stratify_mode") != "fixed_physics_cluster_alpha_bins":
        training.pop("stratify_alpha_bins", None)

    for attr, key in [
        ("epochs", "epochs"),
        ("batch_size", "batch_size"),
        ("learning_rate", "learning_rate"),
    ]:
        value = getattr(args, attr)
        if value is not None:
            training[key] = value

    training["target_columns"] = list(SIX_COMPONENT_COLUMNS)
    training["prediction_columns"] = list(SIX_COMPONENT_COLUMNS)
    if "component_loss_weights" not in explicit.get("training", {}):
        training["component_loss_weights"] = {
            column: 1.0 for column in SIX_COMPONENT_COLUMNS
        }
    else:
        training["component_loss_weights"] = {
            column: float(training["component_loss_weights"].get(column, 1.0))
            for column in SIX_COMPONENT_COLUMNS
        }

    model = deep_update(config.get("model", {}), base_submodel.get("model", {}))
    model = deep_update(model, explicit.get("model", {}))
    model.pop("output_features", None)

    return {
        "training": training,
        "model": model,
        "data_name": explicit.get("data_name", "six_component_multitask"),
        "run_name": explicit.get("run_name", "six_component_multitask"),
        "target": "six_transformed",
    }


def write_metrics_csv(summary: dict[str, object], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in summary.items():
            writer.writerow({"metric": key, "value": value})


def evaluate_six_component_model(run_dir: Path, out_dir: Path) -> None:
    data = load_split(run_dir, SIX_COMPONENT_SPEC)
    ids = sorted(data)
    if not ids:
        raise ValueError(f"No predictions found in {run_dir}.")

    records = []
    true_components = []
    pred_components = []
    true_tensors = []
    pred_tensors = []

    for uid in ids:
        target = data[uid]["target"]
        prediction = data[uid]["prediction"]
        target_tensor = tensor_from_components(target)
        prediction_tensor = tensor_from_components(prediction)

        true_components.append([target[column] for column in COMPONENTS])
        pred_components.append([prediction[column] for column in COMPONENTS])
        true_tensors.append(target_tensor)
        pred_tensors.append(prediction_tensor)

        record = {"id": uid}
        for component in COMPONENTS:
            record[f"target_{component}"] = target[component]
            record[f"prediction_{component}"] = prediction[component]
        record["target_trace"] = float(np.trace(target_tensor))
        record["prediction_trace"] = float(np.trace(prediction_tensor))
        record["target_frobenius_norm"] = float(np.linalg.norm(target_tensor))
        record["prediction_frobenius_norm"] = float(np.linalg.norm(prediction_tensor))
        records.append(record)

    summary = build_summary(
        np.asarray(true_tensors, dtype=float),
        np.asarray(pred_tensors, dtype=float),
        np.asarray(true_components, dtype=float),
        np.asarray(pred_components, dtype=float),
    )
    summary.update(
        {
            "baseline": "six_component_multitask",
            "target": "six_transformed",
            "target_columns": list(SIX_COMPONENT_COLUMNS),
            "target_transforms": [
                {"column": column, "transform": transform}
                for column, transform in SIX_COMPONENT_SPEC
            ],
        }
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_metrics_csv(summary, out_dir / "metrics.csv")

    with (out_dir / "reconstructed_predictions.csv").open("w", newline="") as f:
        fieldnames = list(records[0])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(json.dumps(summary, indent=2), flush=True)


def run_single_split(
    root: Path,
    config: dict,
    split_cfg: dict,
    args: argparse.Namespace,
) -> None:
    split_suffix = (
        "_fixed_physics_cluster"
        if split_cfg["training"].get("stratify_mode") == "fixed_physics_cluster"
        else ""
    )
    task_dir = Path(
        args.task_dir or f"runs/six_component_multitask_baseline{split_suffix}"
    )
    data_dir = Path(
        args.data_dir or f"data/six_component_multitask_baseline{split_suffix}"
    )
    run_dir = task_dir / split_cfg["run_name"]

    if args.skip_prepare:
        refresh_submodel(root, config, split_cfg, data_dir, run_dir)
    else:
        prepare_submodel(root, config, split_cfg, data_dir, run_dir)

    if not args.skip_train:
        if args.overwrite or not has_prediction_file(run_dir):
            train_submodel(root, data_dir, run_dir)
        else:
            print(f"Skipping training; prediction file exists: {run_dir}", flush=True)

    if not has_prediction_file(run_dir):
        raise FileNotFoundError(f"Missing prediction file: {run_dir}")

    if not args.skip_eval:
        evaluate_six_component_model(run_dir, task_dir / "evaluation")

    print("\nSix-component baseline complete.", flush=True)
    print(f"Run directory: {run_dir}", flush=True)
    print(f"Evaluation directory: {task_dir / 'evaluation'}", flush=True)


def run_cross_validation(
    root: Path,
    config: dict,
    split_cfg: dict,
    args: argparse.Namespace,
) -> None:
    cv_config = config.get("cross_validation", {})
    n_folds = int(args.folds if args.folds is not None else cv_config.get("folds", 5))
    split_suffix = (
        "_fixed_physics_cluster"
        if split_cfg["training"].get("stratify_mode") == "fixed_physics_cluster"
        else ""
    )
    task_dir = Path(
        args.task_dir
        or f"runs/six_component_multitask_baseline{split_suffix}_cv{n_folds}"
    )
    data_root = Path(
        args.data_dir
        or f"data/six_component_multitask_baseline{split_suffix}_cv{n_folds}"
    )
    task_dir.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    labels = load_labels(Path(config["labels"]))
    splits, labels_by_uid = cv_splits_for_submodel(
        labels, split_cfg["training"], n_folds
    )

    fold_rows: list[dict[str, object]] = []
    fold_run_dirs: list[Path] = []
    for fold_idx, split in enumerate(splits):
        fold_name = f"fold_{fold_idx + 1:02d}"
        data_dir = data_root / fold_name
        run_dir = task_dir / "folds" / fold_name

        if args.skip_prepare:
            refresh_submodel(root, config, split_cfg, data_dir, run_dir)
        else:
            prepare_submodel(root, config, split_cfg, data_dir, run_dir)

        if not args.keep_existing_splits:
            write_split_artifacts(
                data_dir=data_dir,
                run_dir=run_dir,
                split=split,
                labels_by_uid=labels_by_uid,
            )
            refresh_raw_tail_loss_config(
                Path(config["labels"]), data_dir, run_dir, split_cfg["training"]
            )

        if not args.skip_train:
            if args.overwrite or not has_prediction_file(run_dir):
                train_submodel(root, data_dir, run_dir)
            else:
                print(
                    f"Skipping {fold_name}; prediction file exists: {run_dir}",
                    flush=True,
                )

        if not has_prediction_file(run_dir):
            raise FileNotFoundError(f"Missing prediction file: {run_dir}")

        fold_run_dirs.append(run_dir)
        prediction_count = len(load_split(run_dir, SIX_COMPONENT_SPEC))
        fold_rows.append(
            {
                "fold": fold_idx + 1,
                "n_train": len(split["id_train"]),
                "n_best_model_test": len(split["id_best_model_test"]),
                "n_prediction_val": len(split["id_prediction_val"]),
                "n_oof_prediction": len(split["id_test"]),
                "n_predictions": prediction_count,
                "run_dir": str(run_dir),
            }
        )

    aggregate_dir = task_dir / "oof_predictions" / split_cfg["run_name"]
    aggregate_stats = aggregate_predictions(fold_run_dirs, aggregate_dir)

    with (task_dir / "cv_fold_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fold_rows[0]))
        writer.writeheader()
        writer.writerows(fold_rows)

    (task_dir / "cv_aggregate_summary.json").write_text(
        json.dumps(
            {
                "folds": n_folds,
                "baseline": "six_component_multitask",
                "cluster_counts": dict(Counter(labels_by_uid.values())),
                "aggregate_predictions": aggregate_stats,
            },
            indent=2,
        )
        + "\n"
    )

    if not args.skip_eval:
        evaluate_six_component_model(aggregate_dir, task_dir / "evaluation")

    print("\nSix-component CV baseline complete.", flush=True)
    print(f"CV task directory: {task_dir}", flush=True)
    print(f"OOF evaluation directory: {task_dir / 'evaluation'}", flush=True)


def main() -> None:
    args = parse_args()
    config = apply_path_overrides(load_workflow_config(Path(args.config)), args)
    split_cfg = build_baseline_config(config, args)
    cv_enabled = args.cv or args.folds is not None
    if not cv_enabled:
        cv_enabled = bool(config.get("cross_validation", {}).get("enabled", False))

    root = Path.cwd()
    if cv_enabled:
        run_cross_validation(root, config, split_cfg, args)
    else:
        run_single_split(root, config, split_cfg, args)


if __name__ == "__main__":
    start_time = time.perf_counter()
    try:
        main()
    finally:
        elapsed = time.perf_counter() - start_time
        print(
            f"\nTotal six-component baseline elapsed time: {format_elapsed(elapsed)}",
            flush=True,
        )
