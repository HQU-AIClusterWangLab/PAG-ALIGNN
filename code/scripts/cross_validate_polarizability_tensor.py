#!/usr/bin/env python
"""Run out-of-fold cross validation for the split polarizability tensor models."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

from prepare_polarizability_alignn import (
    load_labels,
    stratification_labels,
)
from train_polarizability_tensor import (
    apply_cli_overrides,
    has_prediction_file,
    load_workflow_config,
    prepare_submodel,
    refresh_raw_tail_loss_config,
    refresh_submodel,
    run,
    submodel_config,
    sync_split_files,
    train_submodel,
)


SUBMODEL_KEYS = ["inplane", "outplane", "outplane_aux"]


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
        description=(
            "Train five-fold out-of-fold split tensor models and evaluate the "
            "reconstructed full tensor on the aggregated OOF predictions."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/polarizability_tensor_config.json",
        help="Unified workflow config.",
    )
    parser.add_argument("--labels", default=None)
    parser.add_argument("--structures", default=None)
    parser.add_argument("--descriptors", default=None)
    parser.add_argument(
        "--task-dir",
        default=None,
        help="Output run directory. Defaults to <config task_dir>_cv5.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Prepared data directory. Defaults to <config data_dir>/polarizability_tensor_cv5.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Use existing fold data/config files and refresh config-controlled files.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip fold training and only aggregate/evaluate existing fold predictions.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Retrain a fold even when its prediction file already exists.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip final OOF tensor evaluation.",
    )
    parser.add_argument(
        "--keep-existing-splits",
        action="store_true",
        help=(
            "Do not regenerate ids_train_val_test.json for existing fold data "
            "directories. Use this only to resume an already prepared CV run."
        ),
    )
    parser.add_argument(
        "--suppress-timer",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def split_namespace(training_config: dict) -> argparse.Namespace:
    return argparse.Namespace(
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


def stratified_test_folds(
    rows: list[dict[str, str]],
    labels_by_uid: dict[str, str],
    n_folds: int,
    seed: int,
) -> list[list[str]]:
    rng = random.Random(seed)
    groups: dict[str, list[str]] = {}
    for row in rows:
        groups.setdefault(labels_by_uid[row["uid"]], []).append(row["uid"])

    folds = [[] for _ in range(n_folds)]
    for label in sorted(groups):
        ids = list(groups[label])
        rng.shuffle(ids)
        for idx, uid in enumerate(ids):
            folds[idx % n_folds].append(uid)

    for fold in folds:
        rng.shuffle(fold)
    return folds


def split_heldout_for_selection_and_prediction(
    heldout_ids: list[str],
    labels_by_uid: dict[str, str],
    seed: int,
) -> tuple[list[str], list[str]]:
    """Split a 20% CV holdout into 10% model-selection and 10% prediction IDs."""
    rng = random.Random(seed)
    groups: dict[str, list[str]] = {}
    for uid in heldout_ids:
        groups.setdefault(labels_by_uid[uid], []).append(uid)

    selection_ids: list[str] = []
    prediction_ids: list[str] = []
    for label in sorted(groups):
        ids = list(groups[label])
        rng.shuffle(ids)
        n_selection = len(ids) // 2
        if len(ids) >= 2 and n_selection == 0:
            n_selection = 1
        selection_ids.extend(ids[:n_selection])
        prediction_ids.extend(ids[n_selection:])
    rng.shuffle(selection_ids)
    rng.shuffle(prediction_ids)
    return selection_ids, prediction_ids


def cv_splits_for_submodel(
    rows: list[dict[str, str]],
    training_config: dict,
    n_folds: int,
) -> tuple[list[dict[str, list[str]]], dict[str, str]]:
    if n_folds < 2:
        raise ValueError("--folds must be at least 2.")

    if training_config.get("stratify_split", False):
        labels_by_uid = stratification_labels(rows, split_namespace(training_config))
    else:
        labels_by_uid = {row["uid"]: "all" for row in rows}

    seed = int(training_config.get("random_seed", 123))
    heldout_folds = stratified_test_folds(rows, labels_by_uid, n_folds, seed)
    all_ids = [row["uid"] for row in rows]
    splits = []
    for fold_idx, heldout in enumerate(heldout_folds):
        heldout_set = set(heldout)
        best_model_test, prediction_val = split_heldout_for_selection_and_prediction(
            heldout_ids=heldout,
            labels_by_uid=labels_by_uid,
            seed=seed + 1009 * (fold_idx + 1),
        )
        train = [uid for uid in all_ids if uid not in heldout_set]
        # ALIGNN uses id_val to select the best checkpoint and id_test for
        # final prediction export. Keep both 10% holdout halves in id_test so
        # the aggregated OOF predictions cover the whole dataset after 5 folds.
        splits.append(
            {
                "id_train": train,
                "id_val": best_model_test,
                "id_test": list(heldout),
                "id_best_model_test": best_model_test,
                "id_prediction_val": prediction_val,
            }
        )
    return splits, labels_by_uid


def write_split_artifacts(
    data_dir: Path,
    run_dir: Path,
    split: dict[str, list[str]],
    labels_by_uid: dict[str, str],
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "ids_train_val_test.json").write_text(
        json.dumps(split, indent=2) + "\n"
    )
    (data_dir / "stratify_labels.csv").write_text(
        "uid,physics_cluster\n"
        + "\n".join(f"{uid},{labels_by_uid[uid]}" for uid in sorted(labels_by_uid))
        + "\n"
    )
    sync_split_files(data_dir, run_dir)


def prediction_ids(run_dir: Path) -> set[str]:
    path = run_dir / "multi_out_predictions.json"
    if not path.exists():
        path = run_dir / "Test_results.json"
    with path.open() as f:
        rows = json.load(f)
    ids = set()
    for row in rows:
        uid = row.get("id", row.get("jid"))
        if isinstance(uid, list):
            ids.update(str(item) for item in uid)
        else:
            ids.add(str(uid))
    return ids


def aggregate_predictions(
    fold_run_dirs: list[Path],
    aggregate_dir: Path,
) -> dict[str, int]:
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    seen: set[str] = set()
    duplicates: list[str] = []
    source_counts = {}
    for run_dir in fold_run_dirs:
        pred_file = run_dir / "multi_out_predictions.json"
        if not pred_file.exists():
            pred_file = run_dir / "Test_results.json"
        if not pred_file.exists():
            raise FileNotFoundError(f"No prediction file found in {run_dir}")
        with pred_file.open() as f:
            fold_rows = json.load(f)
        rows.extend(fold_rows)
        ids = prediction_ids(run_dir)
        source_counts[run_dir.name] = len(ids)
        overlap = seen.intersection(ids)
        duplicates.extend(sorted(overlap))
        seen.update(ids)

    if duplicates:
        examples = ", ".join(duplicates[:10])
        raise ValueError(f"Duplicate OOF prediction ids found: {examples}")

    shutil.copyfile(fold_run_dirs[0] / "config.json", aggregate_dir / "config.json")
    (aggregate_dir / "multi_out_predictions.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )
    return {"n_rows": len(rows), "n_unique_ids": len(seen), **source_counts}


def evaluate_cv_tensor(
    root: Path,
    aggregate_dirs: dict[str, Path],
    out_dir: Path,
) -> None:
    cmd = [
        sys.executable,
        "scripts/evaluate_polarizability_tensor.py",
        "--inplane-dir",
        str(aggregate_dirs["inplane"]),
        "--outplane-dir",
        str(aggregate_dirs["outplane"]),
        "--out-dir",
        str(out_dir),
        "--outplane-aux-dir",
        str(aggregate_dirs["outplane_aux"]),
    ]
    run(cmd, root)


def write_cv_summary(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()
    config = apply_cli_overrides(load_workflow_config(Path(args.config)), args)
    cv_config = config.get("cross_validation", {})
    n_folds = int(args.folds if args.folds is not None else cv_config.get("folds", 5))
    skip_prepare = args.skip_prepare or bool(cv_config.get("skip_prepare", False))
    skip_train = args.skip_train or bool(cv_config.get("skip_train", False))
    skip_eval = args.skip_eval or bool(cv_config.get("skip_eval", False))
    overwrite = args.overwrite or bool(cv_config.get("overwrite", False))
    keep_existing_splits = args.keep_existing_splits or bool(
        cv_config.get("keep_existing_splits", False)
    )

    if args.task_dir is None:
        config["task_dir"] = str(
            cv_config.get(
                "task_dir",
                Path(config["task_dir"]).with_name(Path(config["task_dir"]).name + "_cv5"),
            )
        )
    if args.data_dir is None:
        config["data_dir"] = str(
            cv_config.get(
                "data_dir",
                Path(config["data_dir"]) / "polarizability_tensor_cv5",
            )
        )

    root = Path.cwd()
    labels = load_labels(Path(config["labels"]))
    task_dir = Path(config["task_dir"])
    data_root = Path(config["data_dir"])
    task_dir.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    submodels = {
        key: submodel_config(config, key)
        for key in SUBMODEL_KEYS
        if key in config or key != "outplane_aux"
    }
    if "outplane_aux" not in submodels:
        raise ValueError("Cross-validation requires the outplane_aux submodel.")

    split_by_key = {}
    labels_by_key = {}
    for key, split_cfg in submodels.items():
        splits, labels_by_uid = cv_splits_for_submodel(
            labels, split_cfg["training"], n_folds
        )
        split_by_key[key] = splits
        labels_by_key[key] = labels_by_uid

    fold_rows: list[dict[str, object]] = []
    aggregate_dirs: dict[str, Path] = {}
    aggregate_stats = {}

    for key, split_cfg in submodels.items():
        fold_run_dirs = []
        for fold_idx, split in enumerate(split_by_key[key]):
            fold_name = f"fold_{fold_idx + 1:02d}"
            data_dir = data_root / split_cfg["data_name"] / fold_name
            run_dir = task_dir / split_cfg["run_name"] / fold_name
            if skip_prepare:
                refresh_submodel(root, config, split_cfg, data_dir, run_dir)
            else:
                prepare_submodel(root, config, split_cfg, data_dir, run_dir)
            if not keep_existing_splits:
                write_split_artifacts(
                    data_dir=data_dir,
                    run_dir=run_dir,
                    split=split,
                    labels_by_uid=labels_by_key[key],
                )
                refresh_raw_tail_loss_config(
                    Path(config["labels"]), data_dir, run_dir, split_cfg["training"]
                )

            if not skip_train:
                if overwrite or not has_prediction_file(run_dir):
                    train_submodel(root, data_dir, run_dir)
                else:
                    print(
                        f"Skipping {key} {fold_name}; prediction file exists.",
                        flush=True,
                    )

            if not has_prediction_file(run_dir):
                raise FileNotFoundError(
                    f"Missing prediction file for {key} {fold_name}: {run_dir}"
                )
            ids = prediction_ids(run_dir)
            fold_run_dirs.append(run_dir)
            fold_rows.append(
                {
                    "submodel": key,
                    "fold": fold_idx + 1,
                    "n_train": len(split["id_train"]),
                    "n_best_model_test": len(split["id_best_model_test"]),
                    "n_prediction_val": len(split["id_prediction_val"]),
                    "n_oof_prediction": len(split["id_test"]),
                    "n_predictions": len(ids),
                    "run_dir": str(run_dir),
                }
            )

        aggregate_dir = task_dir / "oof_predictions" / split_cfg["run_name"]
        aggregate_dirs[key] = aggregate_dir
        aggregate_stats[key] = aggregate_predictions(fold_run_dirs, aggregate_dir)

    write_cv_summary(task_dir / "cv_fold_summary.csv", fold_rows)
    (task_dir / "cv_aggregate_summary.json").write_text(
        json.dumps(
            {
                "folds": n_folds,
                "submodels": list(submodels),
                "cluster_counts": {
                    key: dict(Counter(labels_by_key[key].values()))
                    for key in submodels
                },
                "aggregate_predictions": aggregate_stats,
            },
            indent=2,
        )
        + "\n"
    )

    if not skip_eval:
        evaluate_cv_tensor(root, aggregate_dirs, task_dir / "evaluation")

    print("\nCross-validation workflow complete.", flush=True)
    print(f"CV task directory: {task_dir}", flush=True)
    print(f"OOF tensor evaluation: {task_dir / 'evaluation'}", flush=True)


if __name__ == "__main__":
    workflow_start_time = time.perf_counter()
    parsed_args = parse_args()
    try:
        main(parsed_args)
    finally:
        if not parsed_args.suppress_timer:
            elapsed = time.perf_counter() - workflow_start_time
            print(
                f"\nTotal cross-validation elapsed time: {format_elapsed(elapsed)}",
                flush=True,
            )
