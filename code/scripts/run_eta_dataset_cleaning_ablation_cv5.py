#!/usr/bin/env python
"""Compare eta multi-task training across raw and cleaned label datasets.

The model and training recipe are frozen to the recorded site_node_01 setup.
Only the label file (and consequently its UID population) changes between
experiments.  Each completed run is evaluated twice:

* native OOF metrics on its own dataset;
* matched metrics on the UID intersection of all completed runs.

The matched table prevents an apparent improvement caused solely by removing
hard samples from a cleaned dataset.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prepare_polarizability_alignn import load_labels
from run_eta_site_node01_ablation_cv5 import (
    MULTITASK_WEIGHTS,
    SITE_NODE_01_MODEL,
    SITE_NODE_01_TRAINING,
    TARGET_COLUMNS,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REUSE_1801 = "runs/eta_parallel_multitask_1801_cv5_site_node_01"


@dataclass(frozen=True)
class DatasetSpec:
    """One label-set condition in the dataset-cleaning ablation."""

    index: int
    name: str
    display_name: str
    label_path: str
    expected_samples: int
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a site_node_01-controlled eta dataset-cleaning ablation with "
            "five-fold CV. Only label datasets differ between conditions."
        )
    )
    parser.add_argument("--config", default="configs/eta_parallel_config.json")
    parser.add_argument(
        "--out-root",
        default="runs/eta_parallel_dataset_cleaning_ablation_cv5",
        help="Root directory for ablation summaries and newly trained runs.",
    )
    parser.add_argument(
        "--data-root",
        default="data/eta_parallel_dataset_cleaning_ablation_cv5",
        help="Root directory for prepared fold data.",
    )
    parser.add_argument(
        "--generated-config-dir",
        default="configs/generated/eta_parallel_dataset_cleaning_ablation_cv5",
    )
    parser.add_argument("--structures", default=None)
    parser.add_argument("--descriptors", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--datasets",
        default=None,
        help=(
            "Comma-separated conditions to run, by index or name. "
            "Examples: 00,02 or raw_2001,cleaned_1801. Defaults to all "
            "four primary datasets."
        ),
    )
    parser.add_argument(
        "--include-1798",
        action="store_true",
        help="Include the secondary manual three-sample removal condition.",
    )
    parser.add_argument(
        "--labels-2001",
        default="static_lattice_polarizability_tensors.csv",
    )
    parser.add_argument(
        "--labels-1563",
        default="inplane_anisotropy_cleaned_labels_exclude_C0_C1_C6E0.csv",
    )
    parser.add_argument(
        "--labels-1801",
        default="inplane_anisotropy_cleaned_labels_exclude_C0_C1_C6M1.csv",
    )
    parser.add_argument(
        "--labels-1642",
        default="inplane_anisotropy_cleaned_labels_exclude_C0_C1_C6M1_C6M2E0E1.csv",
    )
    parser.add_argument(
        "--labels-1798",
        default="inplane_anisotropy_cleaned_labels_exclude_C0_C1_C6M1_1798.csv",
    )
    parser.add_argument(
        "--reuse-1801-run",
        default=DEFAULT_REUSE_1801,
        help=(
            "Existing completed site_node_01 1801 CV run to reuse. Use "
            "--retrain-1801 to train it again under this ablation root."
        ),
    )
    parser.add_argument(
        "--retrain-1801",
        action="store_true",
        help=(
            "Train the 1801 condition inside --out-root instead of reusing "
            "--reuse-1801-run. Prefer this flag on Windows/PyDev."
        ),
    )
    parser.add_argument(
        "--folds-to-run",
        default=None,
        help="Optional comma-separated 1-based fold list for resume/debugging.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Retrain folds that already have final prediction files.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not reuse existing fold split files in this ablation root.",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop at the first failed dataset condition.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation while training; summary rows remain incomplete.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write manifests/configs and print commands without training.",
    )
    return parser.parse_args()


def root_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def command_path(path: Path) -> str:
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
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def dataset_specs(args: argparse.Namespace) -> list[DatasetSpec]:
    specs = [
        DatasetSpec(
            0,
            "raw_2001",
            "raw 2001",
            args.labels_2001,
            2001,
            "Uncleaned full label dataset.",
        ),
        DatasetSpec(
            1,
            "cleaned_1563",
            "cleaned 1563",
            args.labels_1563,
            1563,
            "Stronger cleaning: exclude C0, C1, and C6E0.",
        ),
        DatasetSpec(
            2,
            "cleaned_1801",
            "cleaned 1801",
            args.labels_1801,
            1801,
            "Candidate final dataset: exclude C0, C1, and C6M1.",
        ),
        DatasetSpec(
            3,
            "cleaned_1642",
            "cleaned 1642",
            args.labels_1642,
            1642,
            "More aggressive cleaning: 1801 exclusions plus C6M2E0E1.",
        ),
    ]
    if args.include_1798:
        specs.append(
            DatasetSpec(
                4,
                "cleaned_1798",
                "cleaned 1798",
                args.labels_1798,
                1798,
                "Secondary sensitivity set: 1801 plus three manual removals.",
            )
        )
    return specs


def select_specs(
    specs: list[DatasetSpec], value: str | None
) -> list[DatasetSpec]:
    if not value:
        return specs
    requested = {item.strip() for item in value.split(",") if item.strip()}
    selected = [
        spec
        for spec in specs
        if f"{spec.index:02d}" in requested or spec.name in requested
    ]
    if not selected:
        known = ", ".join(f"{spec.index:02d}:{spec.name}" for spec in specs)
        raise ValueError(f"No dataset conditions matched --datasets. Valid: {known}")
    return selected


def label_uids(spec: DatasetSpec) -> set[str]:
    path = root_path(spec.label_path)
    if not path.exists():
        raise FileNotFoundError(f"Labels file is missing for {spec.name}: {path}")
    rows = load_labels(path)
    ids = [str(row["uid"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate uid values found in {path}")
    if len(ids) != spec.expected_samples:
        raise ValueError(
            f"{spec.name} expected {spec.expected_samples} samples, found {len(ids)}: {path}"
        )
    return set(ids)


def freeze_site_node01_config(
    base_config: dict[str, Any],
    args: argparse.Namespace,
    spec: DatasetSpec,
    task_dir: Path,
    data_dir: Path,
) -> dict[str, Any]:
    """Freeze all non-dataset settings to the recorded site_node_01 state."""
    config = copy.deepcopy(base_config)
    config["labels"] = spec.label_path
    config["task_dir"] = str(task_dir)
    config["data_dir"] = str(data_dir)
    if args.structures:
        config["structures"] = args.structures
    if args.descriptors:
        config["descriptors"] = args.descriptors

    training = config.setdefault("training", {})
    model = config.setdefault("model", {})
    training.update(copy.deepcopy(SITE_NODE_01_TRAINING))
    model.update(copy.deepcopy(SITE_NODE_01_MODEL))

    # The original site_node_01 condition is the fully enabled multi-task case.
    training["target"] = "eta_parallel_multitask"
    training["target_columns"] = copy.deepcopy(TARGET_COLUMNS)
    training["prediction_columns"] = copy.deepcopy(TARGET_COLUMNS)
    training["component_loss_weights"] = copy.deepcopy(MULTITASK_WEIGHTS)
    training["tail_weight_enabled"] = True
    training["eta_underestimate_penalty_enabled"] = True
    training["eta_pairwise_loss_enabled"] = False
    training["eta_embedding_contrastive_loss_enabled"] = False
    model["output_features"] = 3

    if args.epochs is not None:
        training["epochs"] = args.epochs
    if args.batch_size is not None:
        training["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        training["learning_rate"] = args.learning_rate

    config["ablation"] = {
        "type": "eta_dataset_cleaning",
        "condition": spec.name,
        "display_name": spec.display_name,
        "expected_samples": spec.expected_samples,
        "fixed_reference": "site_node_01",
        "only_changed_input": "labels",
    }
    return config


def validate_frozen_config(config: dict[str, Any]) -> None:
    training = config["training"]
    model = config["model"]
    if training.get("target_columns") != TARGET_COLUMNS:
        raise ValueError("Dataset ablation requires the fixed three-output target.")
    if training.get("component_loss_weights") != MULTITASK_WEIGHTS:
        raise ValueError("Dataset ablation requires fixed site_node_01 loss weights.")
    if not training.get("tail_weight_enabled"):
        raise ValueError("Dataset ablation requires site_node_01 tail weighting enabled.")
    if not training.get("eta_underestimate_penalty_enabled"):
        raise ValueError("Dataset ablation requires the underestimation penalty enabled.")
    if training.get("eta_embedding_contrastive_loss_enabled"):
        raise ValueError("site_node_01 reference requires contrastive loss disabled.")
    if not model.get("use_site_symmetry_node_features") or model.get("output_features") != 3:
        raise ValueError("Dataset ablation requires the fixed site_node_01 model.")


def cv_command(
    args: argparse.Namespace, config_path: Path, task_dir: Path, data_dir: Path
) -> list[str]:
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
    if args.overwrite:
        command.append("--overwrite")
    if not args.no_resume:
        command.append("--keep-existing-splits")
    if not args.stop_on_failure:
        command.append("--continue-on-failure")
    if args.skip_eval:
        command.append("--skip-eval")
    return command


def prediction_rows(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "evaluation" / "predictions.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def error_metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {"status": "incomplete", "n_predictions": 0}
    targets = [float(row["target_eta_parallel"]) for row in rows]
    predictions = [float(row["prediction_eta_parallel"]) for row in rows]
    errors = [prediction - target for target, prediction in zip(targets, predictions)]
    n_rows = len(rows)
    mse = sum(error * error for error in errors) / n_rows
    mean_target = sum(targets) / n_rows
    target_var = sum((target - mean_target) ** 2 for target in targets)
    r2 = math.nan if target_var == 0.0 else 1.0 - sum(error * error for error in errors) / target_var

    def subset(threshold: float) -> tuple[int, float, float]:
        selected = [error for target, error in zip(targets, errors) if target >= threshold]
        if not selected:
            return 0, math.nan, math.nan
        return (
            len(selected),
            sum(abs(error) for error in selected) / len(selected),
            math.sqrt(sum(error * error for error in selected) / len(selected)),
        )

    high_05_n, high_05_mae, high_05_rmse = subset(0.5)
    high_08_n, high_08_mae, high_08_rmse = subset(0.8)
    return {
        "status": "complete",
        "n_predictions": n_rows,
        "eta_mae": sum(abs(error) for error in errors) / n_rows,
        "eta_rmse": math.sqrt(mse),
        "eta_r2": r2,
        "eta_mean_error": sum(errors) / n_rows,
        "high_eta_ge_05_n": high_05_n,
        "high_eta_ge_05_mae": high_05_mae,
        "high_eta_ge_05_rmse": high_05_rmse,
        "high_eta_ge_08_n": high_08_n,
        "high_eta_ge_08_mae": high_08_mae,
        "high_eta_ge_08_rmse": high_08_rmse,
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
        "n_clipped_low": sum(int(row.get("clipped_low", "0")) for row in rows),
        "n_clipped_high": sum(int(row.get("clipped_high", "0")) for row in rows),
    }


def native_summary_row(
    spec: DatasetSpec,
    run_dir: Path,
    data_dir: Path | None,
    uid_set: set[str],
    raw_ids: set[str],
    reused: bool,
) -> dict[str, Any]:
    return {
        "index": f"{spec.index:02d}",
        "condition": spec.name,
        "display_name": spec.display_name,
        "labels": spec.label_path,
        "expected_samples": spec.expected_samples,
        "label_unique_uids": len(uid_set),
        "removed_vs_raw_2001": len(raw_ids - uid_set),
        "extra_vs_raw_2001": len(uid_set - raw_ids),
        "note": spec.note,
        "reused_existing_run": reused,
        "run_dir": str(run_dir),
        "data_dir": "" if data_dir is None else str(data_dir),
        **error_metrics(prediction_rows(run_dir)),
    }


def matched_common_rows(
    completed: list[tuple[DatasetSpec, Path]]
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    predictions: dict[str, dict[str, dict[str, str]]] = {}
    for spec, run_dir in completed:
        rows = prediction_rows(run_dir)
        if not rows:
            continue
        by_id = {str(row["id"]): row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError(f"Duplicate OOF prediction ids found in {run_dir}")
        predictions[spec.name] = by_id
    if not predictions:
        return [], []

    common_ids = set.intersection(*(set(rows) for rows in predictions.values()))
    common_manifest = [{"uid": uid} for uid in sorted(common_ids)]
    rows_out: list[dict[str, Any]] = []
    for spec, run_dir in completed:
        by_id = predictions.get(spec.name)
        if by_id is None:
            continue
        selected = [by_id[uid] for uid in sorted(common_ids)]
        rows_out.append(
            {
                "index": f"{spec.index:02d}",
                "condition": spec.name,
                "display_name": spec.display_name,
                "run_dir": str(run_dir),
                "common_uid_count": len(common_ids),
                **error_metrics(selected),
            }
        )
    return rows_out, common_manifest


def add_reference_deltas(rows: list[dict[str, Any]], reference: str = "cleaned_1801") -> None:
    baseline = next(
        (row for row in rows if row.get("condition") == reference and row.get("status") == "complete"),
        None,
    )
    if baseline is None:
        return
    for row in rows:
        for metric in ("eta_mae", "eta_rmse", "eta_r2", "high_eta_ge_08_mae"):
            value = row.get(metric)
            reference_value = baseline.get(metric)
            if isinstance(value, (float, int)) and isinstance(reference_value, (float, int)):
                row[f"delta_{metric}_vs_1801"] = value - reference_value


def format_value(value: Any) -> str:
    return f"{value:.6f}" if isinstance(value, float) and math.isfinite(value) else str(value)


def write_markdown_summary(
    path: Path,
    native_rows: list[dict[str, Any]],
    matched_rows: list[dict[str, Any]],
) -> None:
    columns = [
        "condition",
        "expected_samples",
        "eta_mae",
        "eta_rmse",
        "eta_r2",
        "high_eta_ge_08_mae",
        "severe_under",
        "severe_over",
        "status",
    ]
    lines = [
        "# eta dataset-cleaning ablation summary",
        "",
        "All conditions use the fixed site_node_01 multi-task configuration. "
        "Only the label dataset changes.",
        "",
        "## Native OOF metrics",
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in native_rows:
        lines.append("| " + " | ".join(format_value(row.get(column, "")) for column in columns) + " |")
    if matched_rows:
        lines.extend(
            [
                "",
                "## Matched common-UID OOF metrics",
                "",
                "This section evaluates every completed condition on the same UID "
                "intersection. Each prediction remains OOF for its own dataset.",
                "",
                "| " + " | ".join(columns) + " |",
                "|" + "|".join(["---"] * len(columns)) + "|",
            ]
        )
        for row in matched_rows:
            lines.append("| " + " | ".join(format_value(row.get(column, "")) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    specs = dataset_specs(args)
    selected_specs = select_specs(specs, args.datasets)
    base_config = read_json(root_path(args.config))
    out_root = root_path(args.out_root)
    data_root = root_path(args.data_root)
    config_root = root_path(args.generated_config_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    config_root.mkdir(parents=True, exist_ok=True)

    uid_sets = {spec.name: label_uids(spec) for spec in specs}
    raw_ids = uid_sets["raw_2001"]
    inventory = [
        {
            "index": f"{spec.index:02d}",
            "condition": spec.name,
            "display_name": spec.display_name,
            "labels": spec.label_path,
            "expected_samples": spec.expected_samples,
            "label_unique_uids": len(uid_sets[spec.name]),
            "removed_vs_raw_2001": len(raw_ids - uid_sets[spec.name]),
            "extra_vs_raw_2001": len(uid_sets[spec.name] - raw_ids),
            "note": spec.note,
        }
        for spec in specs
    ]
    write_csv(out_root / "dataset_inventory.csv", inventory)
    write_json(
        out_root / "experiment_manifest.json",
        {
            "reference": "site_node_01",
            "fixed_changes": "All model/training settings are frozen; only labels differ.",
            "selected_conditions": [spec.name for spec in selected_specs],
            "all_conditions": [spec.name for spec in specs],
            "reuse_1801_run": args.reuse_1801_run,
            "folds": 5,
        },
    )

    completed_runs: list[tuple[DatasetSpec, Path]] = []
    native_rows: list[dict[str, Any]] = []
    for spec in specs:
        run_dir = out_root / spec.name
        data_dir = data_root / spec.name
        reuse_run = (
            spec.name == "cleaned_1801"
            and not args.retrain_1801
            and bool(args.reuse_1801_run)
            and root_path(args.reuse_1801_run).exists()
        )
        effective_run_dir = root_path(args.reuse_1801_run) if reuse_run else run_dir
        config = freeze_site_node01_config(
            base_config, args, spec, run_dir, data_dir
        )
        validate_frozen_config(config)
        config_path = config_root / f"{spec.index:02d}_{spec.name}.json"
        write_json(config_path, config)

        if spec in selected_specs and not reuse_run:
            command = cv_command(args, config_path, run_dir, data_dir)
            print(f"\n=== Dataset {spec.index:02d}: {spec.display_name} ===", flush=True)
            print(" ".join(command), flush=True)
            if not args.dry_run:
                result = subprocess.run(command, cwd=ROOT, check=False)
                if result.returncode != 0:
                    message = f"Dataset condition {spec.name} failed with exit code {result.returncode}."
                    if args.stop_on_failure:
                        raise RuntimeError(message)
                    print(message + " Continuing to the next condition.", flush=True)
        elif spec in selected_specs and reuse_run:
            print(
                f"\n=== Dataset {spec.index:02d}: {spec.display_name} (reused) ===\n"
                f"Using completed reference run: {effective_run_dir}",
                flush=True,
            )

        row = native_summary_row(
            spec,
            effective_run_dir,
            None if reuse_run else data_dir,
            uid_sets[spec.name],
            raw_ids,
            reuse_run,
        )
        native_rows.append(row)
        if row.get("status") == "complete":
            completed_runs.append((spec, effective_run_dir))

    add_reference_deltas(native_rows)
    matched_rows, common_manifest = matched_common_rows(completed_runs)
    add_reference_deltas(matched_rows)
    write_csv(out_root / "native_oof_summary.csv", native_rows)
    write_csv(out_root / "matched_common_uid_oof_summary.csv", matched_rows)
    write_csv(out_root / "matched_common_uids.csv", common_manifest)
    write_markdown_summary(out_root / "dataset_cleaning_ablation_summary.md", native_rows, matched_rows)

    print(f"\nDataset inventory: {out_root / 'dataset_inventory.csv'}", flush=True)
    print(f"Native OOF summary: {out_root / 'native_oof_summary.csv'}", flush=True)
    print(f"Matched OOF summary: {out_root / 'matched_common_uid_oof_summary.csv'}", flush=True)
    print(f"Markdown summary: {out_root / 'dataset_cleaning_ablation_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
