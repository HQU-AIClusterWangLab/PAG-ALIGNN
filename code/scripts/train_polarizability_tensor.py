#!/usr/bin/env python
"""Train split ALIGNN submodels and evaluate the reconstructed tensor.

This is a Windows-friendly orchestration script for the polarizability tensor
task. It treats the in-plane and out-of-plane ALIGNN models as two submodels of
one tensor prediction workflow:

1. prepare transformed in-plane labels: log1p(xx), log1p(yy), asinh(xy)
2. prepare transformed out-of-plane labels: log1p(zz/s), asinh(xz), asinh(yz)
3. train the in-plane submodel
4. train the out-of-plane submodel
5. reconstruct the full symmetric 3x3 tensor and evaluate metrics
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

from prepare_polarizability_alignn import (
    compute_tail_weights,
    load_labels,
    split_ids,
    stratification_labels,
)


def format_elapsed(seconds: float) -> str:
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    millis = seconds - total_seconds
    if hours:
        return f"{hours}h {minutes:02d}m {secs + millis:04.1f}s ({seconds:.1f} s)"
    if minutes:
        return f"{minutes}m {secs + millis:04.1f}s ({seconds:.1f} s)"
    return f"{seconds:.1f} s"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full split-submodel ALIGNN tensor workflow."
    )
    parser.add_argument(
        "--config",
        default="configs/polarizability_tensor_config.json",
        help="Unified workflow config for both tensor submodels.",
    )
    parser.add_argument("--labels", default=None)
    parser.add_argument("--structures", default=None)
    parser.add_argument("--descriptors", default=None)
    parser.add_argument("--task-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Use existing prepared in-plane/out-plane data directories.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and only run tensor reconstruction/evaluation.",
    )
    parser.add_argument(
        "--skip-inplane-train",
        action="store_true",
        help="Skip only the in-plane submodel training step.",
    )
    parser.add_argument(
        "--skip-outplane-train",
        action="store_true",
        help="Skip only the out-of-plane submodel training step.",
    )
    parser.add_argument(
        "--skip-outplane-aux-train",
        action="store_true",
        help="Skip only the auxiliary out-of-plane submodel training step.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip tensor reconstruction/evaluation after training.",
    )
    parser.add_argument(
        "--eval-scope",
        choices=["full", "inplane", "outplane", "outplane_aux", "outplane_both"],
        default="full",
        help="Evaluate the full tensor or only one submodel.",
    )
    parser.add_argument(
        "--outplane-aux-dir",
        default=None,
        help=(
            "Optional auxiliary out-plane run directory used only for full "
            "tensor evaluation. Its xz/yz predictions override the main "
            "out-plane model xz/yz predictions."
        ),
    )
    return parser.parse_args()


def deep_update(base: dict, updates: dict) -> dict:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_workflow_config(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    config = deep_update({}, config)
    for attr in ["labels", "structures", "descriptors", "task_dir", "data_dir"]:
        value = getattr(args, attr)
        if value is not None:
            config[attr] = value
    cli_training = {}
    if args.epochs is not None:
        cli_training["epochs"] = args.epochs
    if args.batch_size is not None:
        cli_training["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        cli_training["learning_rate"] = args.learning_rate
    if cli_training:
        for key in ["inplane", "outplane", "outplane_aux"]:
            config.setdefault(key, {}).setdefault("training", {})
            config[key]["training"].update(cli_training)
    return config


def submodel_config(config: dict, key: str) -> dict:
    sub = config.get(key, {})
    default_targets = {
        "inplane": "inplane_transformed",
        "outplane": "outplane_transformed",
        "outplane_aux": "xz_yz_transformed",
    }
    default_target = default_targets.get(key, "outplane_transformed")
    return {
        "training": deep_update(config.get("training", {}), sub.get("training", {})),
        "model": deep_update(config.get("model", {}), sub.get("model", {})),
        "data_name": sub.get("data_name", f"polarizability_{key}"),
        "run_name": sub.get("run_name", key),
        "target": sub.get("target", default_target),
    }


def rewrite_alignn_config(
    config_path: Path,
    run_dir: Path,
    split_config: dict,
) -> None:
    with config_path.open() as f:
        alignn_config = json.load(f)

    training = split_config["training"]
    model = split_config["model"]
    for key, value in training.items():
        if key.startswith("stratify_"):
            continue
        alignn_config[key] = value
    if "tail_weight_enabled" in training:
        alignn_config["use_sample_weights"] = bool(
            training["tail_weight_enabled"]
        )
        alignn_config["sample_weight_csv"] = (
            str((config_path.parent / "sample_weights.csv").resolve())
            if training["tail_weight_enabled"]
            else None
        )
        alignn_config["sample_weight_id_key"] = "uid"
        alignn_config["sample_weight_value_key"] = "weight"
    alignn_config["output_dir"] = str(run_dir)
    alignn_config["filename"] = "polarizability"
    alignn_config["model"] = deep_update(alignn_config.get("model", {}), model)
    if alignn_config.get("atom_features") == "cgcnn":
        alignn_config["model"].setdefault("atom_input_features", 92)
    elif alignn_config.get("atom_features") == "basic":
        alignn_config["model"].setdefault("atom_input_features", 11)
    elif alignn_config.get("atom_features") == "atomic_number":
        alignn_config["model"].setdefault("atom_input_features", 1)

    with config_path.open("w") as f:
        json.dump(alignn_config, f, indent=2)
        f.write("\n")


def run(cmd: list[str], cwd: Path) -> None:
    print("\n>>> " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def has_prediction_file(run_dir: Path) -> bool:
    return (
        (run_dir / "multi_out_predictions.json").exists()
        or (run_dir / "Test_results.json").exists()
    )


def sync_split_files(data_dir: Path, run_dir: Path) -> None:
    split_file = data_dir / "ids_train_val_test.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_split_file = run_dir / "ids_train_val_test.json"
    run_stratify_file = run_dir / "stratify_labels.csv"
    if not split_file.exists():
        for stale_file in [run_split_file, run_stratify_file]:
            if stale_file.exists():
                stale_file.unlink()
        return
    shutil.copyfile(split_file, run_split_file)
    stratify_file = data_dir / "stratify_labels.csv"
    if stratify_file.exists():
        shutil.copyfile(stratify_file, run_stratify_file)
    elif run_stratify_file.exists():
        run_stratify_file.unlink()


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile of an empty list.")
    if not 0 <= q <= 1:
        raise ValueError("Percentile q must be in [0, 1].")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def compute_raw_tail_loss_parameters(
    labels_path: Path,
    split_path: Path,
    training_config: dict,
) -> dict[str, float | str | bool | None]:
    enabled = bool(training_config.get("raw_tail_loss_enabled", False))
    params: dict[str, float | str | bool | None] = {
        "raw_tail_loss_enabled": enabled,
    }
    if not enabled:
        return params

    labels = load_labels(labels_path)
    rows_by_uid = {row["uid"]: row for row in labels}
    if split_path.exists():
        with split_path.open() as f:
            split = json.load(f)
        train_ids = [uid for uid in split.get("id_train", []) if uid in rows_by_uid]
    else:
        train_ids = [row["uid"] for row in labels]
    if not train_ids:
        raise ValueError(f"No train ids available for raw tail loss: {split_path}")

    alpha = [
        0.5 * (float(rows_by_uid[uid]["xx"]) + float(rows_by_uid[uid]["yy"]))
        for uid in train_ids
    ]
    quantile = float(training_config.get("raw_tail_loss_quantile", 0.8))
    threshold = percentile(alpha, quantile)
    configured_scale = training_config.get("raw_tail_loss_scale")
    if configured_scale is not None and float(configured_scale) > 0:
        scale = float(configured_scale)
        scale_mode = "configured"
    else:
        scale_mode = str(training_config.get("raw_tail_loss_scale_mode", "mean_alpha"))
        if scale_mode == "p90_minus_p10":
            scale = percentile(alpha, 0.9) - percentile(alpha, 0.1)
        elif scale_mode == "mean_alpha":
            scale = sum(alpha) / len(alpha)
        else:
            raise ValueError(f"Unknown raw_tail_loss_scale_mode: {scale_mode}")
    scale = max(float(scale), 1e-12)
    params.update(
        {
            "raw_tail_loss_lambda": float(
                training_config.get("raw_tail_loss_lambda", 0.05)
            ),
            "raw_tail_loss_quantile": quantile,
            "raw_tail_loss_scale_mode": scale_mode,
            "raw_tail_loss_alpha_weight": float(
                training_config.get("raw_tail_loss_alpha_weight", 1.0)
            ),
            "raw_tail_loss_under_factor": float(
                training_config.get("raw_tail_loss_under_factor", 1.5)
            ),
            "raw_tail_loss_threshold": threshold,
            "raw_tail_loss_scale": scale,
        }
    )
    return params


def refresh_raw_tail_loss_config(
    labels_path: Path,
    data_dir: Path,
    run_dir: Path,
    training_config: dict,
) -> None:
    config_path = data_dir / "config.json"
    if not config_path.exists():
        return
    params = compute_raw_tail_loss_parameters(
        labels_path, data_dir / "ids_train_val_test.json", training_config
    )
    for path in [config_path, run_dir / "config.json"]:
        if path.exists():
            with path.open() as f:
                config = json.load(f)
            config.update(params)
            with path.open("w") as f:
                json.dump(config, f, indent=2)
                f.write("\n")


def refresh_config_driven_artifacts(
    labels_path: Path,
    data_dir: Path,
    training_config: dict,
) -> None:
    """Refresh files whose contents are controlled by the unified config."""
    data_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(labels_path)

    sample_weight_path = data_dir / "sample_weights.csv"
    if training_config.get("tail_weight_enabled", False):
        tail_weights = compute_tail_weights(
            labels=labels,
            score_name=str(training_config.get("tail_weight_score", "alpha_parallel")),
            quantile=float(training_config.get("tail_weight_quantile", 0.75)),
            weight_lambda=float(training_config.get("tail_weight_lambda", 1.65)),
            power=float(training_config.get("tail_weight_power", 0.3)),
        )
        with sample_weight_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["uid", "score", "rank", "ramp", "weight"]
            )
            writer.writeheader()
            for uid, values in sorted(tail_weights.items()):
                writer.writerow({"uid": uid, **values})
    elif sample_weight_path.exists():
        sample_weight_path.unlink()

    split_path = data_dir / "ids_train_val_test.json"
    stratify_path = data_dir / "stratify_labels.csv"
    if training_config.get("stratify_split", False):
        stratify_args = argparse.Namespace(
            stratify_mode=str(
                training_config.get("stratify_mode", "fixed_physics_cluster")
            ),
            stratify_trace_floor=float(
                training_config.get("stratify_trace_floor", 1e-3)
            ),
            stratify_xy_coupling_threshold=float(
                training_config.get("stratify_xy_coupling_threshold", 0.30)
            ),
            stratify_diag_anisotropy_threshold=float(
                training_config.get("stratify_diag_anisotropy_threshold", 0.30)
            ),
            stratify_min_count=int(training_config.get("stratify_min_count", 2)),
            stratify_alpha_bins=training_config.get("stratify_alpha_bins"),
        )
        labels_by_uid = stratification_labels(labels, stratify_args)
        split = split_ids(
            rows=labels,
            labels_by_uid=labels_by_uid,
            train_ratio=float(training_config.get("train_ratio", 0.8)),
            val_ratio=float(training_config.get("val_ratio", 0.1)),
            test_ratio=float(training_config.get("test_ratio", 0.1)),
            seed=int(training_config.get("random_seed", 123)),
        )
        stratify_path.write_text(
            "uid,physics_cluster\n"
            + "\n".join(f"{uid},{labels_by_uid[uid]}" for uid in sorted(labels_by_uid))
            + "\n"
        )
        split_path.write_text(json.dumps(split, indent=2) + "\n")
    else:
        for stale_file in [split_path, stratify_path]:
            if stale_file.exists():
                stale_file.unlink()


def tail_weight_args(training_config: dict) -> list[str]:
    if not training_config.get("tail_weight_enabled", False):
        return []
    return [
        "--tail-weight-enabled",
        "--tail-weight-score",
        str(training_config.get("tail_weight_score", "alpha_parallel")),
        "--tail-weight-quantile",
        str(training_config.get("tail_weight_quantile", 0.75)),
        "--tail-weight-lambda",
        str(training_config.get("tail_weight_lambda", 1.65)),
        "--tail-weight-power",
        str(training_config.get("tail_weight_power", 0.3)),
    ]


def format_bin_arg(value: object, default: list[float]) -> str:
    if value is None:
        value = default
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)


def stratify_args(training_config: dict) -> list[str]:
    if not training_config.get("stratify_split", False):
        return []
    return [
        "--stratify-split",
        "--stratify-mode",
        str(training_config.get("stratify_mode", "fixed_physics_cluster")),
        "--stratify-trace-floor",
        str(training_config.get("stratify_trace_floor", 1e-3)),
        "--stratify-xy-coupling-threshold",
        str(training_config.get("stratify_xy_coupling_threshold", 0.30)),
        "--stratify-diag-anisotropy-threshold",
        str(training_config.get("stratify_diag_anisotropy_threshold", 0.30)),
        "--stratify-min-count",
        str(training_config.get("stratify_min_count", 2)),
        "--stratify-alpha-bins",
        format_bin_arg(
            training_config.get("stratify_alpha_bins"),
            [0.50, 0.75, 0.85, 0.95],
        ),
        "--train-ratio",
        str(training_config.get("train_ratio", 0.8)),
        "--val-ratio",
        str(training_config.get("val_ratio", 0.1)),
        "--test-ratio",
        str(training_config.get("test_ratio", 0.1)),
        "--split-seed",
        str(training_config.get("random_seed", 123)),
    ]


def prepare_submodel(
    root: Path,
    config: dict,
    split_cfg: dict,
    data_dir: Path,
    run_dir: Path,
) -> None:
    run(
        [
            sys.executable,
            "scripts/prepare_polarizability_alignn.py",
            "--labels",
            config["labels"],
            "--structures",
            config["structures"],
            "--descriptors",
            config["descriptors"],
            "--target",
            str(split_cfg["target"]),
            "--out-dir",
            str(data_dir),
            "--epochs",
            str(split_cfg["training"].get("epochs", 300)),
            "--batch-size",
            str(split_cfg["training"].get("batch_size", 32)),
            "--learning-rate",
            str(split_cfg["training"].get("learning_rate", 0.001)),
        ]
        + tail_weight_args(split_cfg["training"])
        + stratify_args(split_cfg["training"]),
        root,
    )
    rewrite_alignn_config(data_dir / "config.json", run_dir, split_cfg)
    sync_split_files(data_dir, run_dir)
    refresh_raw_tail_loss_config(
        Path(config["labels"]), data_dir, run_dir, split_cfg["training"]
    )


def refresh_submodel(
    root: Path,
    config: dict,
    split_cfg: dict,
    data_dir: Path,
    run_dir: Path,
) -> None:
    if not (data_dir / "config.json").exists():
        prepare_submodel(root, config, split_cfg, data_dir, run_dir)
        return
    summary_path = data_dir / "summary.json"
    if summary_path.exists():
        with summary_path.open() as f:
            summary = json.load(f)
        if summary.get("target") != split_cfg["target"]:
            prepare_submodel(root, config, split_cfg, data_dir, run_dir)
            return
        expected_columns = split_cfg["training"].get("target_columns")
        if expected_columns and summary.get("target_columns") != expected_columns:
            prepare_submodel(root, config, split_cfg, data_dir, run_dir)
            return
    refresh_config_driven_artifacts(
        Path(config["labels"]), data_dir, split_cfg["training"]
    )
    rewrite_alignn_config(data_dir / "config.json", run_dir, split_cfg)
    sync_split_files(data_dir, run_dir)
    refresh_raw_tail_loss_config(
        Path(config["labels"]), data_dir, run_dir, split_cfg["training"]
    )


def train_submodel(root: Path, data_dir: Path, run_dir: Path) -> None:
    run(
        [
            sys.executable,
            "-m",
            "alignn.train_alignn",
            "--root_dir",
            str(data_dir),
            "--config_name",
            str(data_dir / "config.json"),
            "--file_format",
            "poscar",
            "--output_dir",
            str(run_dir),
        ],
        root,
    )


def evaluate_split(
    root: Path,
    run_dir: Path,
    split_name: str,
    out_dir: Path,
) -> None:
    run(
        [
            sys.executable,
            "scripts/evaluate_polarizability_split.py",
            "--run-dir",
            str(run_dir),
            "--split",
            split_name,
            "--out-dir",
            str(out_dir),
        ],
        root,
    )


def run_cross_validation_workflow(root: Path, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "scripts/cross_validate_polarizability_tensor.py",
        "--config",
        args.config,
        "--suppress-timer",
    ]
    for attr, flag in [
        ("labels", "--labels"),
        ("structures", "--structures"),
        ("descriptors", "--descriptors"),
        ("task_dir", "--task-dir"),
        ("data_dir", "--data-dir"),
        ("epochs", "--epochs"),
        ("batch_size", "--batch-size"),
        ("learning_rate", "--learning-rate"),
    ]:
        value = getattr(args, attr)
        if value is not None:
            cmd.extend([flag, str(value)])
    if args.skip_prepare:
        cmd.append("--skip-prepare")
    if args.skip_train:
        cmd.append("--skip-train")
    if args.skip_eval:
        cmd.append("--skip-eval")
    run(cmd, root)


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
    config = apply_cli_overrides(load_workflow_config(Path(args.config)), args)
    if config.get("cross_validation", {}).get("enabled", False):
        run_cross_validation_workflow(Path.cwd(), args)
        return

    in_cfg = submodel_config(config, "inplane")
    out_cfg = submodel_config(config, "outplane")
    aux_cfg = (
        submodel_config(config, "outplane_aux")
        if "outplane_aux" in config
        else None
    )

    root = Path.cwd()
    data_root = Path(config["data_dir"])
    task_dir = Path(config["task_dir"])

    inplane_data = data_root / in_cfg["data_name"]
    outplane_data = data_root / out_cfg["data_name"]
    aux_data = data_root / aux_cfg["data_name"] if aux_cfg is not None else None
    inplane_run = task_dir / in_cfg["run_name"]
    outplane_run = task_dir / out_cfg["run_name"]
    aux_run = task_dir / aux_cfg["run_name"] if aux_cfg is not None else None
    eval_dir = task_dir / "evaluation"

    if not args.skip_prepare:
        prepare_submodel(root, config, in_cfg, inplane_data, inplane_run)
        prepare_submodel(root, config, out_cfg, outplane_data, outplane_run)
        if aux_cfg is not None and aux_data is not None and aux_run is not None:
            prepare_submodel(root, config, aux_cfg, aux_data, aux_run)
    else:
        refresh_submodel(root, config, in_cfg, inplane_data, inplane_run)
        refresh_submodel(root, config, out_cfg, outplane_data, outplane_run)
        if aux_cfg is not None and aux_data is not None and aux_run is not None:
            refresh_submodel(root, config, aux_cfg, aux_data, aux_run)

    if not args.skip_train and not args.skip_inplane_train:
        train_submodel(root, inplane_data, inplane_run)
    if not args.skip_train and not args.skip_outplane_train:
        train_submodel(root, outplane_data, outplane_run)
    if (
        aux_cfg is not None
        and aux_data is not None
        and aux_run is not None
        and not args.skip_train
        and not args.skip_outplane_aux_train
    ):
        train_submodel(root, aux_data, aux_run)

    if not args.skip_eval and args.eval_scope == "full":
        missing_runs = [
            name
            for name, run_dir in [
                ("inplane", inplane_run),
                ("outplane", outplane_run),
                ("outplane_aux", aux_run),
            ]
            if run_dir is not None and not has_prediction_file(run_dir)
        ]
        if missing_runs:
            raise FileNotFoundError(
                "Full tensor evaluation requires prediction files for all "
                f"submodels. Missing: {', '.join(missing_runs)}. "
                "Train the missing submodel first, use --eval-scope "
                "outplane_both for out-plane-only evaluation, or add --skip-eval."
            )
        eval_cmd = [
            sys.executable,
            "scripts/evaluate_polarizability_tensor.py",
            "--inplane-dir",
            str(inplane_run),
            "--outplane-dir",
            str(outplane_run),
            "--out-dir",
            str(eval_dir),
        ]
        outplane_aux_dir = args.outplane_aux_dir
        if outplane_aux_dir is None and aux_run is not None:
            outplane_aux_dir = str(aux_run)
        if outplane_aux_dir is not None:
            eval_cmd.extend(["--outplane-aux-dir", outplane_aux_dir])
        run(eval_cmd, root)
    elif not args.skip_eval and args.eval_scope == "outplane_both":
        evaluate_split(
            root,
            outplane_run,
            "outplane",
            task_dir / "outplane_evaluation",
        )
        if aux_run is None:
            raise ValueError("eval-scope outplane_both requires outplane_aux config.")
        evaluate_split(
            root,
            aux_run,
            "outplane",
            task_dir / "outplane_aux_evaluation",
        )
    elif not args.skip_eval:
        if args.eval_scope == "inplane":
            split_run = inplane_run
            split_name = "inplane"
        elif args.eval_scope == "outplane":
            split_run = outplane_run
            split_name = "outplane"
        else:
            if aux_run is None:
                raise ValueError("eval-scope outplane_aux requires outplane_aux config.")
            split_run = aux_run
            split_name = "outplane"
        evaluate_split(
            root,
            split_run,
            split_name,
            task_dir / f"{args.eval_scope}_evaluation",
        )

    print("\nWorkflow complete.", flush=True)
    print(f"In-plane submodel: {inplane_run}", flush=True)
    print(f"Out-of-plane submodel: {outplane_run}", flush=True)
    if aux_run is not None:
        print(f"Auxiliary out-of-plane submodel: {aux_run}", flush=True)
    print(f"Tensor evaluation: {eval_dir}", flush=True)


if __name__ == "__main__":
    parsed_args = parse_args()
    workflow_start_time = time.perf_counter()
    try:
        main(parsed_args)
    finally:
        elapsed = time.perf_counter() - workflow_start_time
        print(f"\nTotal workflow elapsed time: {format_elapsed(elapsed)}", flush=True)
