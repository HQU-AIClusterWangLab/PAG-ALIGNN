#!/usr/bin/env python
"""Run a strict five-fold ablation of the site_node_01 eta model.

The eight experiments differ only in:
1. task formulation (pure eta-only single task versus three-output multitask),
2. eta tail sample weighting, and
3. high-eta underestimation penalty.

All other settings are frozen to the recorded site_node_01 configuration.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

from cross_validate_eta_parallel import kfold_splits, load_source_cv_splits
from prepare_polarizability_alignn import load_labels


ROOT = Path(__file__).resolve().parents[1]
TARGET_COLUMNS = ["eta_parallel", "a_diag", "a_xy"]

# These values are taken from runs/eta_parallel_multitask_1801_cv5_site_node_01.
# They are deliberately applied even if configs/eta_parallel_config.json changes later.
SITE_NODE_01_TRAINING = {
    "target": "eta_parallel_multitask",
    "target_columns": TARGET_COLUMNS,
    "target_transforms": [
        {"column": "eta_parallel", "transform": "identity"},
        {"column": "a_diag", "transform": "identity"},
        {"column": "a_xy", "transform": "identity"},
    ],
    "prediction_columns": TARGET_COLUMNS,
    "tail_weight_score": "eta_parallel",
    "tail_weight_quantile": 0.86,
    "tail_weight_lambda": 0.9,
    "tail_weight_power": 0.5,
    "xy_tail_extra_weight_enabled": False,
    "eta_underestimate_penalty_target": "eta_parallel",
    "eta_underestimate_penalty_threshold": 0.8,
    "eta_underestimate_penalty_lambda": 0.1,
    "eta_underestimate_penalty_power": 1.0,
    "eta_underestimate_penalty_scale": 0.2,
    "eta_pairwise_loss_enabled": False,
    "eta_pairwise_loss_lambda": 0.0,
    "eta_embedding_contrastive_loss_enabled": False,
    "stratify_split": True,
    "stratify_mode": "inplane_eta_cluster_bins",
    "stratify_eta_bins": [0.01, 0.05, 0.1, 0.2, 0.5, 0.65, 0.8, 0.9],
    "stratify_trace_floor": 0.001,
    "stratify_xy_coupling_threshold": 0.3,
    "stratify_diag_anisotropy_threshold": 0.3,
    "stratify_min_count": 10,
}

SITE_NODE_01_MODEL = {
    "name": "alignn_atomwise",
    "atom_input_features": 92,
    "alignn_layers": 4,
    "gcn_layers": 2,
    "embedding_features": 64,
    "hidden_features": 64,
    "output_features": 3,
    "calculate_gradient": False,
    "graphwise_weight": 1.0,
    "gradwise_weight": 0.0,
    "stresswise_weight": 0.0,
    "atomwise_weight": 0.0,
    "link": "identity",
    "energy_mult_natoms": False,
    "extra_features": 101,
    "use_extra_feature_gated_fusion": False,
    "extra_feature_fusion_gamma_init": 0.05,
    "extra_feature_fusion_gamma_max": 0.5,
    "use_site_symmetry_node_features": True,
    "site_symmetry_node_features": 80,
    "site_symmetry_symprec": 0.001,
    "site_symmetry_angle_tolerance": 5.0,
    "site_symmetry_gamma_init": 0.03,
    "site_symmetry_gamma_max": 0.2,
    "use_inplane_direction_attention": True,
    "inplane_direction_attention_mode": "direction",
    "inplane_direction_attention_gamma_init": 0.05,
    "inplane_direction_attention_gamma_max": 0.3,
    "use_inplane_angle_direction_gate": False,
    "use_readout_attention": False,
    "use_component_readout_attention": False,
}

MULTITASK_WEIGHTS = {"eta_parallel": 1.0, "a_diag": 0.48, "a_xy": 0.12}
SINGLE_TASK_WEIGHTS = {"eta_parallel": 1.0}

ABLATION_CASES = [
    ("00_single_no_tail_no_underpen", "single", False, False),
    ("01_single_tail_no_underpen", "single", True, False),
    ("02_single_no_tail_underpen", "single", False, True),
    ("03_single_tail_underpen", "single", True, True),
    ("04_multitask_no_tail_no_underpen", "multitask", False, False),
    ("05_multitask_tail_no_underpen", "multitask", True, False),
    ("06_multitask_no_tail_underpen", "multitask", False, True),
    ("07_multitask_tail_underpen_site_node01", "multitask", True, True),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run strict site_node_01 eta ablations with a shared deterministic "
            "five-fold split."
        )
    )
    parser.add_argument("--config", default="configs/eta_parallel_config.json")
    parser.add_argument(
        "--out-root",
        default="runs/eta_parallel_site_node01_ablation_cv5",
        help="Root directory for the eight CV runs and final summaries.",
    )
    parser.add_argument(
        "--data-root",
        default="data/eta_parallel_site_node01_ablation_cv5",
        help="Root directory for prepared data of the eight CV runs.",
    )
    parser.add_argument(
        "--generated-config-dir",
        default="configs/generated/eta_parallel_site_node01_ablation_cv5",
    )
    parser.add_argument("--labels", default=None)
    parser.add_argument("--structures", default=None)
    parser.add_argument("--descriptors", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--cases",
        default=None,
        help=(
            "Comma-separated ablation indices or names, e.g. 00,03,07. "
            "Defaults to all eight cases."
        ),
    )
    parser.add_argument(
        "--folds-to-run",
        default=None,
        help="Optional comma-separated 1-based fold list, useful after a failure.",
    )
    parser.add_argument(
        "--source-cv-data-root",
        default=None,
        help=(
            "Optional source CV data root whose fold split files are reused for "
            "every ablation case, e.g. data/eta_parallel_dataset_cleaning_ablation_cv5/raw_2001."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Retrain folds with existing prediction files.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not pass --keep-existing-splits to the CV workflow.",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop at the first failed fold instead of moving to the next case.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip per-case OOF evaluation and summary metrics.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write configs and manifests, but do not start training.",
    )
    return parser.parse_args()


def root_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def command_path(path: Path) -> str:
    """Prefer repo-relative paths, while still allowing an explicit absolute root."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def apply_path_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    for key in ("labels", "structures", "descriptors"):
        value = getattr(args, key)
        if value:
            config[key] = value


def baseline_config(base_config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Freeze every non-ablated setting to the recorded site_node_01 state."""
    config = copy.deepcopy(base_config)
    apply_path_overrides(config, args)
    training = config.setdefault("training", {})
    model = config.setdefault("model", {})
    training.update(copy.deepcopy(SITE_NODE_01_TRAINING))
    model.update(copy.deepcopy(SITE_NODE_01_MODEL))
    if args.epochs is not None:
        training["epochs"] = args.epochs
    if args.batch_size is not None:
        training["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        training["learning_rate"] = args.learning_rate
    validate_baseline(config)
    return config


def validate_baseline(config: dict[str, Any]) -> None:
    training = config["training"]
    model = config["model"]
    if training["target_columns"] != TARGET_COLUMNS:
        raise ValueError("site_node_01 ablation requires the three fixed target columns.")
    if not model.get("use_site_symmetry_node_features", False):
        raise ValueError("site_node_01 ablation requires site-symmetry node features.")
    if model.get("output_features") != 3:
        raise ValueError("site_node_01 ablation requires a fixed three-output head.")
    if training.get("eta_embedding_contrastive_loss_enabled", False):
        raise ValueError(
            "site_node_01 uses batch contrastive loss disabled; change the base "
            "experiment before using a different contrastive setting."
        )


def case_config(
    base: dict[str, Any],
    *,
    name: str,
    task_mode: str,
    tail_weight_enabled: bool,
    underpen_enabled: bool,
    task_dir: Path,
    data_dir: Path,
) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config["task_dir"] = str(task_dir)
    config["data_dir"] = str(data_dir)
    training = config["training"]
    model = config["model"]
    if task_mode == "single":
        # A true single-task baseline has one target and one prediction head.
        # It is intentionally not a three-head model with zeroed auxiliary loss.
        training["target"] = "eta_parallel"
        training["target_columns"] = ["eta_parallel"]
        training["target_transforms"] = [
            {"column": "eta_parallel", "transform": "identity"}
        ]
        training["prediction_columns"] = ["eta_parallel"]
        training["component_loss_weights"] = copy.deepcopy(SINGLE_TASK_WEIGHTS)
        model["output_features"] = 1
    else:
        training["target"] = "eta_parallel_multitask"
        training["target_columns"] = copy.deepcopy(TARGET_COLUMNS)
        training["target_transforms"] = copy.deepcopy(
            SITE_NODE_01_TRAINING["target_transforms"]
        )
        training["prediction_columns"] = copy.deepcopy(TARGET_COLUMNS)
        training["component_loss_weights"] = copy.deepcopy(MULTITASK_WEIGHTS)
        model["output_features"] = 3
    training["tail_weight_enabled"] = tail_weight_enabled
    training["eta_underestimate_penalty_enabled"] = underpen_enabled
    config["ablation"] = {
        "name": name,
        "task_mode": task_mode,
        "pure_single_task": task_mode == "single",
        "tail_weight_enabled": tail_weight_enabled,
        "eta_underestimate_penalty_enabled": underpen_enabled,
        "fixed_reference": "site_node_01",
    }
    return config


def select_cases(value: str | None) -> list[tuple[str, str, bool, bool]]:
    if not value:
        return ABLATION_CASES
    requested = {item.strip() for item in value.split(",") if item.strip()}
    chosen = []
    for index, case in enumerate(ABLATION_CASES):
        name = case[0]
        if f"{index:02d}" in requested or name in requested:
            chosen.append(case)
    if not chosen:
        known = ", ".join(f"{i:02d}" for i in range(len(ABLATION_CASES)))
        raise ValueError(f"No ablation cases matched --cases. Valid indices: {known}")
    return chosen


def build_split_manifest(
    config: dict[str, Any], source_cv_data_root: str | None = None
) -> dict[str, Any]:
    labels_path = root_path(str(config["labels"]))
    rows = load_labels(labels_path)
    splits, labels_by_uid = kfold_splits(rows, config["training"], 5)
    source_path = None
    if source_cv_data_root:
        source_path = root_path(source_cv_data_root)
        splits = load_source_cv_splits(
            source_data_root=source_path,
            valid_ids={str(row["uid"]) for row in rows},
            n_folds=5,
        )
    return {
        "reference": "site_node_01",
        "labels": str(labels_path),
        "source_cv_data_root": "" if source_path is None else str(source_path),
        "random_seed": int(config["training"].get("random_seed", 123)),
        "stratify": {
            key: config["training"].get(key)
            for key in (
                "stratify_split",
                "stratify_mode",
                "stratify_eta_bins",
                "stratify_trace_floor",
                "stratify_xy_coupling_threshold",
                "stratify_diag_anisotropy_threshold",
                "stratify_min_count",
                "val_ratio",
            )
        },
        "cluster_labels_by_uid": labels_by_uid,
        "folds": splits,
    }


def validate_case_splits(data_dir: Path, manifest: dict[str, Any]) -> None:
    for fold_index, expected in enumerate(manifest["folds"], start=1):
        path = data_dir / f"fold_{fold_index:02d}" / "ids_train_val_test.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing split artifact: {path}")
        actual = read_json(path)
        for key, expected_ids in expected.items():
            if set(actual.get(key, [])) != set(expected_ids):
                raise ValueError(
                    f"Split mismatch in {path} for {key}; strict ablation is invalid."
                )


def cv_command(args: argparse.Namespace, config_path: Path, task_dir: Path, data_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "scripts/cross_validate_eta_parallel.py",
        "--config",
        command_path(config_path),
        "--folds",
        "5",
        "--task-dir",
        command_path(task_dir),
        "--data-dir",
        command_path(data_dir),
    ]
    if args.folds_to_run:
        command.extend(["--folds-to-run", args.folds_to_run])
    if args.source_cv_data_root:
        command.extend(
            ["--source-cv-data-root", command_path(root_path(args.source_cv_data_root))]
        )
    if args.overwrite:
        command.append("--overwrite")
    if not args.no_resume:
        command.append("--keep-existing-splits")
    if not args.stop_on_failure:
        command.append("--continue-on-failure")
    if args.skip_eval:
        command.append("--skip-eval")
    return command


def prediction_metrics(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / "evaluation" / "metrics.json"
    predictions_path = run_dir / "evaluation" / "predictions.csv"
    if not metrics_path.exists() or not predictions_path.exists():
        return {"status": "incomplete"}
    metrics = read_json(metrics_path)
    with predictions_path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    targets = [float(row["target_eta_parallel"]) for row in rows]
    predictions = [float(row["prediction_eta_parallel"]) for row in rows]
    errors = [prediction - target for target, prediction in zip(targets, predictions)]

    def subset_metrics(threshold: float) -> dict[str, Any]:
        values = [
            error for target, error in zip(targets, errors) if target >= threshold
        ]
        if not values:
            return {"n": 0, "mae": math.nan, "rmse": math.nan}
        return {
            "n": len(values),
            "mae": sum(abs(value) for value in values) / len(values),
            "rmse": math.sqrt(sum(value * value for value in values) / len(values)),
        }

    high_05 = subset_metrics(0.5)
    high_08 = subset_metrics(0.8)
    return {
        "status": "complete",
        "n_predictions": len(rows),
        "eta_mae": metrics.get("eta_mae"),
        "eta_rmse": metrics.get("eta_rmse"),
        "eta_r2": metrics.get("eta_r2"),
        "eta_mean_error": metrics.get("eta_mean_error"),
        "eta_raw_mae": metrics.get("raw_eta_mae"),
        "n_clipped_low": metrics.get("n_clipped_low"),
        "high_eta_ge_05_n": high_05["n"],
        "high_eta_ge_05_mae": high_05["mae"],
        "high_eta_ge_05_rmse": high_05["rmse"],
        "high_eta_ge_08_n": high_08["n"],
        "high_eta_ge_08_mae": high_08["mae"],
        "high_eta_ge_08_rmse": high_08["rmse"],
        "abs_ge_0p3": sum(abs(error) >= 0.3 for error in errors),
        "abs_ge_0p5": sum(abs(error) >= 0.5 for error in errors),
        "severe_under": sum(error <= -0.3 for error in errors),
        "severe_over": sum(error >= 0.3 for error in errors),
        "high_eta_severe_under": sum(
            target >= 0.8 and error <= -0.3 for target, error in zip(targets, errors)
        ),
        "low_eta_severe_over": sum(
            target <= 0.1 and error >= 0.3 for target, error in zip(targets, errors)
        ),
    }


def case_summary_row(
    *,
    index: int,
    name: str,
    task_mode: str,
    tail_weight_enabled: bool,
    underpen_enabled: bool,
    run_dir: Path,
    data_dir: Path,
) -> dict[str, Any]:
    metrics = prediction_metrics(run_dir)
    return {
        "index": f"{index:02d}",
        "experiment": name,
        "task_mode": task_mode,
        "tail_weight_enabled": tail_weight_enabled,
        "eta_underestimate_penalty_enabled": underpen_enabled,
        "component_loss_weight_eta_parallel": 1.0,
        "component_loss_weight_a_diag": (
            MULTITASK_WEIGHTS["a_diag"] if task_mode == "multitask" else 0.0
        ),
        "component_loss_weight_a_xy": (
            MULTITASK_WEIGHTS["a_xy"] if task_mode == "multitask" else 0.0
        ),
        "model_output_features": 3 if task_mode == "multitask" else 1,
        "site_symmetry_node_features": True,
        "embedding_contrastive_loss": False,
        "run_dir": str(run_dir),
        "data_dir": str(data_dir),
        **metrics,
    }


def add_reference_deltas(rows: list[dict[str, Any]]) -> None:
    reference = next((row for row in rows if row["index"] == "07"), None)
    if not reference or reference.get("status") != "complete":
        return
    for row in rows:
        for metric in ("eta_mae", "eta_rmse", "high_eta_ge_08_mae"):
            value = row.get(metric)
            baseline = reference.get(metric)
            if isinstance(value, (int, float)) and isinstance(baseline, (int, float)):
                row[f"delta_{metric}_vs_07"] = value - baseline


def write_markdown_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "index",
        "task_mode",
        "tail_weight_enabled",
        "eta_underestimate_penalty_enabled",
        "status",
        "eta_mae",
        "eta_rmse",
        "eta_r2",
        "high_eta_ge_08_mae",
        "severe_under",
        "severe_over",
    ]
    lines = [
        "# site_node_01 eta ablation summary",
        "",
        "All runs share the same deterministic five-fold split and fixed "
        "site_node_01 backbone settings. Single-task cases are pure eta-only "
        "models with one output head; multitask cases use eta_parallel, a_diag, "
        "and a_xy with loss weights 1.0, 0.48, and 0.12.",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_config = read_json(root_path(args.config))
    frozen_base = baseline_config(base_config, args)
    out_root = root_path(args.out_root)
    data_root = root_path(args.data_root)
    config_root = root_path(args.generated_config_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    config_root.mkdir(parents=True, exist_ok=True)

    manifest = build_split_manifest(frozen_base, args.source_cv_data_root)
    write_json(out_root / "fold_manifest.json", manifest)
    write_json(out_root / "site_node01_frozen_base_config.json", frozen_base)

    selected = select_cases(args.cases)
    all_rows: list[dict[str, Any]] = []
    for index, (name, task_mode, tail_enabled, underpen_enabled) in enumerate(
        ABLATION_CASES
    ):
        run_dir = out_root / name
        data_dir = data_root / name
        config = case_config(
            frozen_base,
            name=name,
            task_mode=task_mode,
            tail_weight_enabled=tail_enabled,
            underpen_enabled=underpen_enabled,
            task_dir=run_dir,
            data_dir=data_dir,
        )
        config_path = config_root / f"{index:02d}_{name}.json"
        write_json(config_path, config)

        if (name, task_mode, tail_enabled, underpen_enabled) not in selected:
            all_rows.append(
                case_summary_row(
                    index=index,
                    name=name,
                    task_mode=task_mode,
                    tail_weight_enabled=tail_enabled,
                    underpen_enabled=underpen_enabled,
                    run_dir=run_dir,
                    data_dir=data_dir,
                )
            )
            continue

        command = cv_command(args, config_path, run_dir, data_dir)
        print(f"\n=== Ablation {index:02d}/07: {name} ===", flush=True)
        print(" ".join(command), flush=True)
        if not args.dry_run:
            result = subprocess.run(command, cwd=ROOT, check=False)
            if result.returncode != 0:
                message = f"Ablation {name} failed with exit code {result.returncode}."
                if args.stop_on_failure:
                    raise RuntimeError(message)
                print(message + " Continuing to the next case.", flush=True)
            elif all(
                (data_dir / f"fold_{fold:02d}" / "ids_train_val_test.json").exists()
                for fold in range(1, 6)
            ):
                validate_case_splits(data_dir, manifest)
                print("Shared five-fold split verified.", flush=True)

        all_rows.append(
            case_summary_row(
                index=index,
                name=name,
                task_mode=task_mode,
                tail_weight_enabled=tail_enabled,
                underpen_enabled=underpen_enabled,
                run_dir=run_dir,
                data_dir=data_dir,
            )
        )

    add_reference_deltas(all_rows)
    summary_csv = out_root / "ablation_summary.csv"
    summary_markdown = out_root / "ablation_summary.md"
    write_csv(summary_csv, all_rows)
    write_markdown_summary(summary_markdown, all_rows)
    print(f"\nSummary CSV: {summary_csv}", flush=True)
    print(f"Summary Markdown: {summary_markdown}", flush=True)
    print(f"Fold manifest: {out_root / 'fold_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
