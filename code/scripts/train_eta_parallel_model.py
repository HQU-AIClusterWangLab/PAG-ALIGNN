#!/usr/bin/env python
"""Train a standalone ALIGNN model for in-plane anisotropy eta."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from train_polarizability_tensor import (
    deep_update,
    format_elapsed,
    has_prediction_file,
    run,
    train_submodel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and train the standalone eta_parallel ALIGNN model."
    )
    parser.add_argument(
        "--config",
        default="configs/eta_parallel_config.json",
        help="Standalone eta model config.",
    )
    parser.add_argument("--labels", default=None)
    parser.add_argument("--structures", default=None)
    parser.add_argument("--descriptors", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--task-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Use existing prepared eta data and refresh its config.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Prepare data/config only; do not train.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Train even when prediction files already exist.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip metrics generation after training.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    config = deep_update({}, config)
    for attr in ["labels", "structures", "descriptors", "data_dir", "task_dir"]:
        value = getattr(args, attr)
        if value is not None:
            config[attr] = value
    training_overrides = {}
    if args.epochs is not None:
        training_overrides["epochs"] = args.epochs
    if args.batch_size is not None:
        training_overrides["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        training_overrides["learning_rate"] = args.learning_rate
    if training_overrides:
        config.setdefault("training", {}).update(training_overrides)
    return config


def format_eta_bins(value: object) -> str:
    if value is None:
        value = [0.01, 0.05, 0.1, 0.2, 0.5, 0.8]
    if isinstance(value, str):
        return value
    return ",".join(str(item) for item in value)


def eta_training_target(training: dict) -> str:
    return str(training.get("target", "eta_parallel"))


def configured_target_columns(training: dict, alignn_config: dict) -> list[str]:
    columns = training.get("target_columns") or alignn_config.get("target_columns")
    if not columns:
        return ["eta_parallel"]
    return [str(column) for column in columns]


def configured_target_transforms(
    training: dict, alignn_config: dict, target_columns: list[str]
) -> list[dict[str, str]]:
    transforms = training.get("target_transforms") or alignn_config.get(
        "target_transforms"
    )
    if transforms:
        return [
            {
                "column": str(item["column"]),
                "transform": str(item.get("transform", "identity")),
            }
            for item in transforms
        ]
    return [{"column": column, "transform": "identity"} for column in target_columns]


def configured_component_loss_weights(
    training: dict, target_columns: list[str]
) -> dict[str, float]:
    configured = training.get("component_loss_weights") or {}
    return {
        column: float(configured.get(column, 1.0))
        for column in target_columns
    }


def prepare_eta_data(root: Path, config: dict, data_dir: Path) -> None:
    training = config["training"]
    target = eta_training_target(training)
    cmd = [
        sys.executable,
        "scripts/prepare_polarizability_alignn.py",
        "--labels",
        config["labels"],
        "--structures",
        config["structures"],
        "--descriptors",
        config["descriptors"],
        "--target",
        target,
        "--out-dir",
        str(data_dir),
        "--epochs",
        str(training.get("epochs", 300)),
        "--batch-size",
        str(training.get("batch_size", 16)),
        "--learning-rate",
        str(training.get("learning_rate", 0.001)),
    ]
    if target == "columns":
        columns = training.get("target_columns")
        if not columns:
            raise ValueError(
                "training.target_columns is required when training.target is columns."
            )
        cmd.extend(["--columns", ",".join(str(column) for column in columns)])
    if training.get("tail_weight_enabled", False):
        cmd.extend(
            [
                "--tail-weight-enabled",
                "--tail-weight-score",
                str(training.get("tail_weight_score", "alpha_parallel")),
                "--tail-weight-quantile",
                str(training.get("tail_weight_quantile", 0.75)),
                "--tail-weight-lambda",
                str(training.get("tail_weight_lambda", 1.65)),
                "--tail-weight-power",
                str(training.get("tail_weight_power", 0.3)),
            ]
        )
        if training.get("xy_tail_extra_weight_enabled", False):
            cmd.extend(
                [
                    "--xy-tail-extra-weight-enabled",
                    "--xy-tail-extra-eta-threshold",
                    str(training.get("xy_tail_extra_eta_threshold", 0.8)),
                    "--xy-tail-extra-multiplier",
                    str(training.get("xy_tail_extra_multiplier", 1.15)),
                    "--xy-tail-extra-weight-cap",
                    str(training.get("xy_tail_extra_weight_cap", 2.0)),
                ]
            )
    cmd.extend(
        [
            "--eta-pairwise-loss-target",
            str(training.get("eta_pairwise_loss_target", "eta_parallel")),
            "--eta-pairwise-loss-lambda",
            str(training.get("eta_pairwise_loss_lambda", 0.02)),
            "--eta-pairwise-loss-min-delta",
            str(training.get("eta_pairwise_loss_min_delta", 0.5)),
            "--eta-pairwise-loss-max-pairs",
            str(training.get("eta_pairwise_loss_max_pairs", 512)),
            "--eta-pairwise-loss-pairs-per-batch",
            str(training.get("eta_pairwise_loss_pairs_per_batch", 2)),
            "--eta-pairwise-loss-buffer-max-pairs",
            str(training.get("eta_pairwise_loss_buffer_max_pairs", 2048)),
            "--eta-embedding-contrastive-loss-target",
            str(
                training.get(
                    "eta_embedding_contrastive_loss_target",
                    "eta_parallel",
                )
            ),
            "--eta-embedding-contrastive-loss-lambda",
            str(training.get("eta_embedding_contrastive_loss_lambda", 0.01)),
            "--eta-embedding-contrastive-loss-min-delta",
            str(training.get("eta_embedding_contrastive_loss_min_delta", 0.5)),
            "--eta-embedding-contrastive-loss-margin",
            str(training.get("eta_embedding_contrastive_loss_margin", 1.0)),
            "--eta-embedding-contrastive-loss-max-pairs",
            str(training.get("eta_embedding_contrastive_loss_max_pairs", 512)),
            "--eta-embedding-contrastive-pull-lambda",
            str(training.get("eta_embedding_contrastive_pull_lambda", 0.0)),
            "--eta-embedding-contrastive-pull-max-delta",
            str(training.get("eta_embedding_contrastive_pull_max_delta", 0.05)),
            "--eta-embedding-contrastive-pull-max-pairs",
            str(training.get("eta_embedding_contrastive_pull_max_pairs", 512)),
        ]
    )
    if training.get("eta_pairwise_loss_enabled", False):
        cmd.append("--eta-pairwise-loss-enabled")
    if training.get("eta_pairwise_loss_use_abs_delta", False):
        cmd.append("--eta-pairwise-loss-use-abs-delta")
    if not training.get("eta_pairwise_loss_sampler_enabled", True):
        cmd.append("--no-eta-pairwise-loss-sampler-enabled")
    if training.get("eta_pairwise_loss_batch_enabled", False):
        cmd.append("--eta-pairwise-loss-batch-enabled")
    if training.get("eta_embedding_contrastive_loss_enabled", False):
        cmd.append("--eta-embedding-contrastive-loss-enabled")
    if not training.get("eta_embedding_contrastive_loss_normalize", True):
        cmd.append("--no-eta-embedding-contrastive-loss-normalize")
    if training.get("eta_embedding_contrastive_pull_enabled", False):
        cmd.append("--eta-embedding-contrastive-pull-enabled")
    if training.get("stratify_split", True):
        cmd.extend(
            [
                "--stratify-split",
                "--stratify-mode",
                str(training.get("stratify_mode", "inplane_eta_cluster")),
                "--stratify-trace-floor",
                str(training.get("stratify_trace_floor", 1e-3)),
                "--stratify-xy-coupling-threshold",
                str(training.get("stratify_xy_coupling_threshold", 0.30)),
                "--stratify-diag-anisotropy-threshold",
                str(training.get("stratify_diag_anisotropy_threshold", 0.30)),
                "--stratify-min-count",
                str(training.get("stratify_min_count", 2)),
                "--stratify-eta-bins",
                format_eta_bins(training.get("stratify_eta_bins")),
                "--train-ratio",
                str(training.get("train_ratio", 0.8)),
                "--val-ratio",
                str(training.get("val_ratio", 0.1)),
                "--test-ratio",
                str(training.get("test_ratio", 0.1)),
                "--split-seed",
                str(training.get("random_seed", 123)),
            ]
        )
    run(cmd, root)


def refresh_eta_config(data_dir: Path, run_dir: Path, config: dict) -> None:
    config_path = data_dir / "config.json"
    with config_path.open() as f:
        alignn_config = json.load(f)

    training = config["training"]
    for key, value in training.items():
        if key.startswith("stratify_") or key == "target":
            continue
        alignn_config[key] = value

    target_columns = configured_target_columns(training, alignn_config)
    target_transforms = configured_target_transforms(
        training, alignn_config, target_columns
    )
    prediction_columns = [
        str(column)
        for column in training.get(
            "prediction_columns",
            alignn_config.get("prediction_columns", target_columns),
        )
    ]
    alignn_config["target"] = "target"
    alignn_config["target_columns"] = target_columns
    alignn_config["target_transforms"] = target_transforms
    alignn_config["prediction_columns"] = prediction_columns
    alignn_config["component_loss_weights"] = configured_component_loss_weights(
        training, target_columns
    )
    alignn_config["output_dir"] = str(run_dir)
    alignn_config["filename"] = "eta_parallel"
    alignn_config["model"] = deep_update(
        alignn_config.get("model", {}), config.get("model", {})
    )
    alignn_config["model"]["output_features"] = len(target_columns)
    if alignn_config.get("atom_features") == "cgcnn":
        alignn_config["model"].setdefault("atom_input_features", 92)
    elif alignn_config.get("atom_features") == "basic":
        alignn_config["model"].setdefault("atom_input_features", 11)
    elif alignn_config.get("atom_features") == "atomic_number":
        alignn_config["model"].setdefault("atom_input_features", 1)

    with config_path.open("w") as f:
        json.dump(alignn_config, f, indent=2)
        f.write("\n")

    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w") as f:
        json.dump(alignn_config, f, indent=2)
        f.write("\n")


def evaluate_eta_model(root: Path, run_dir: Path) -> None:
    run(
        [
            sys.executable,
            "scripts/evaluate_eta_parallel_model.py",
            "--run-dir",
            str(run_dir),
            "--out-dir",
            str(run_dir / "evaluation"),
        ],
        root,
    )


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()

    root = Path.cwd()
    config = apply_cli_overrides(load_config(Path(args.config)), args)
    data_dir = Path(config["data_dir"])
    run_dir = Path(config["task_dir"])

    if not args.skip_prepare or not (data_dir / "config.json").exists():
        prepare_eta_data(root, config, data_dir)
    refresh_eta_config(data_dir, run_dir, config)

    if not args.skip_train:
        if args.overwrite or not has_prediction_file(run_dir):
            train_submodel(root, data_dir, run_dir)
        else:
            print(f"Skipping eta_parallel; prediction file exists: {run_dir}")

    if not args.skip_eval and has_prediction_file(run_dir):
        evaluate_eta_model(root, run_dir)

    print("\nStandalone eta_parallel workflow complete.", flush=True)
    print(f"Prepared data directory: {data_dir}", flush=True)
    print(f"Run directory: {run_dir}", flush=True)
    if not args.skip_eval:
        print(f"Evaluation directory: {run_dir / 'evaluation'}", flush=True)


if __name__ == "__main__":
    start = time.perf_counter()
    parsed_args = parse_args()
    try:
        main(parsed_args)
    finally:
        print(
            f"\nTotal eta_parallel workflow elapsed time: "
            f"{format_elapsed(time.perf_counter() - start)}",
            flush=True,
        )
