#!/usr/bin/env python
"""Run K-fold cross validation for the standalone eta_parallel model."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

from cross_validate_polarizability_tensor import (
    aggregate_predictions,
    prediction_ids,
    stratified_test_folds,
    write_split_artifacts,
)
from prepare_polarizability_alignn import load_labels, stratification_labels
from train_eta_parallel_model import (
    apply_cli_overrides,
    load_config,
    prepare_eta_data,
    refresh_eta_config,
)
from train_polarizability_tensor import (
    format_elapsed,
    has_prediction_file,
    run,
    train_submodel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train standalone eta_parallel ALIGNN models with selectable "
            "5-fold or 10-fold cross validation."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/eta_parallel_config.json",
        help="Standalone eta model config.",
    )
    parser.add_argument("--labels", default=None)
    parser.add_argument("--structures", default=None)
    parser.add_argument("--descriptors", default=None)
    parser.add_argument(
        "--task-dir",
        default=None,
        help="Output run directory. Defaults to <config task_dir>_cv<folds>.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Prepared data directory. Defaults to <config data_dir>_cv<folds>.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--folds",
        type=int,
        choices=[5, 10],
        default=None,
        help="Number of folds to run. Choose 5 or 10.",
    )
    parser.add_argument(
        "--folds-to-run",
        default=None,
        help=(
            "Optional comma-separated 1-based fold numbers to train/evaluate. "
            "Useful for resuming, e.g. --folds-to-run 3,4."
        ),
    )
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
        help="Skip per-fold and aggregated OOF metrics generation.",
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
        "--source-cv-data-root",
        default=None,
        help=(
            "Optional prepared CV data root whose fold_XX/ids_train_val_test.json "
            "files define the exact train/validation/test splits to reuse."
        ),
    )
    parser.add_argument(
        "--no-clip",
        action="store_true",
        help="Disable eta prediction clipping to [0, 1] during evaluation.",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help=(
            "Record a failed fold and continue later folds. Aggregation is "
            "skipped until every fold has predictions."
        ),
    )
    parser.add_argument(
        "--suppress-timer",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def parse_fold_selection(value: str | None, n_folds: int) -> set[int] | None:
    if not value:
        return None
    selected: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        fold = int(part)
        if fold < 1 or fold > n_folds:
            raise ValueError(f"Selected fold {fold} is outside 1..{n_folds}.")
        selected.add(fold)
    if not selected:
        raise ValueError("--folds-to-run did not contain any fold numbers.")
    return selected


def split_namespace(training_config: dict) -> argparse.Namespace:
    return argparse.Namespace(
        stratify_mode=str(
            training_config.get("stratify_mode", "inplane_eta_cluster")
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
        stratify_eta_bins=training_config.get(
            "stratify_eta_bins", [0.01, 0.05, 0.1, 0.2, 0.5, 0.8]
        ),
    )


def split_validation_from_pool(
    *,
    candidate_ids: list[str],
    labels_by_uid: dict[str, str],
    total_label_counts: Counter[str],
    val_ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Take a stratified validation subset from the non-test pool."""
    rng = random.Random(seed)
    groups: dict[str, list[str]] = {}
    for uid in candidate_ids:
        groups.setdefault(labels_by_uid[uid], []).append(uid)

    id_val: list[str] = []
    id_train: list[str] = []
    for label in sorted(groups):
        ids = list(groups[label])
        rng.shuffle(ids)
        target_val = int(round(total_label_counts[label] * val_ratio))
        if val_ratio > 0 and len(ids) >= 2 and target_val == 0:
            target_val = 1
        target_val = min(max(target_val, 0), max(0, len(ids) - 1))
        id_val.extend(ids[:target_val])
        id_train.extend(ids[target_val:])

    rng.shuffle(id_train)
    rng.shuffle(id_val)
    return id_train, id_val


def kfold_splits(
    rows: list[dict[str, str]],
    training_config: dict,
    n_folds: int,
) -> tuple[list[dict[str, list[str]]], dict[str, str]]:
    if training_config.get("stratify_split", False):
        labels_by_uid = stratification_labels(rows, split_namespace(training_config))
    else:
        labels_by_uid = {row["uid"]: "all" for row in rows}

    seed = int(training_config.get("random_seed", 123))
    val_ratio = float(training_config.get("val_ratio", 0.1))
    test_folds = stratified_test_folds(rows, labels_by_uid, n_folds, seed)
    all_ids = [row["uid"] for row in rows]
    total_label_counts = Counter(labels_by_uid[uid] for uid in all_ids)

    splits: list[dict[str, list[str]]] = []
    for fold_idx, id_test in enumerate(test_folds):
        test_set = set(id_test)
        train_val_pool = [uid for uid in all_ids if uid not in test_set]
        id_train, id_val = split_validation_from_pool(
            candidate_ids=train_val_pool,
            labels_by_uid=labels_by_uid,
            total_label_counts=total_label_counts,
            val_ratio=val_ratio,
            seed=seed + 1009 * (fold_idx + 1),
        )
        splits.append(
            {
                "id_train": id_train,
                "id_val": id_val,
                "id_test": list(id_test),
                "id_best_model_test": id_val,
                "id_prediction_val": list(id_test),
            }
        )
    return splits, labels_by_uid


def load_source_cv_splits(
    source_data_root: Path,
    valid_ids: set[str],
    n_folds: int,
) -> list[dict[str, list[str]]]:
    """Load and validate fixed folds from a completed/prepared eta CV run."""
    splits: list[dict[str, list[str]]] = []
    seen_test_ids: set[str] = set()
    required = ("id_train", "id_val", "id_test")
    for fold_number in range(1, n_folds + 1):
        path = source_data_root / f"fold_{fold_number:02d}" / "ids_train_val_test.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing source CV split file: {path}")
        with path.open() as file:
            source = json.load(file)
        missing = [key for key in required if key not in source]
        if missing:
            raise ValueError(f"Source split {path} is missing keys: {missing}")
        split = {
            "id_train": [str(uid) for uid in source["id_train"]],
            "id_val": [str(uid) for uid in source["id_val"]],
            "id_test": [str(uid) for uid in source["id_test"]],
            "id_best_model_test": [
                str(uid)
                for uid in source.get("id_best_model_test", source["id_val"])
            ],
            "id_prediction_val": [
                str(uid)
                for uid in source.get("id_prediction_val", source["id_test"])
            ],
        }
        train_ids, val_ids, test_ids = (
            set(split["id_train"]),
            set(split["id_val"]),
            set(split["id_test"]),
        )
        if (
            len(train_ids) != len(split["id_train"])
            or len(val_ids) != len(split["id_val"])
            or len(test_ids) != len(split["id_test"])
        ):
            raise ValueError(f"Source split {path} contains duplicate IDs.")
        if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
            raise ValueError(f"Source split {path} has overlapping train/val/test IDs.")
        if train_ids | val_ids | test_ids != valid_ids:
            raise ValueError(
                f"Source split {path} does not cover the current labels UID set exactly."
            )
        if seen_test_ids & test_ids:
            raise ValueError(f"Source test folds overlap at {path}.")
        seen_test_ids.update(test_ids)
        splits.append(split)
    if seen_test_ids != valid_ids:
        raise ValueError("Source test folds do not cover the current labels UID set exactly.")
    return splits


def default_cv_path(path_text: str, suffix: str) -> str:
    path = Path(path_text)
    return str(path.with_name(path.name + suffix))


def existing_split_matches(
    data_dir: Path,
    expected_split: dict[str, list[str]],
) -> bool:
    path = data_dir / "ids_train_val_test.json"
    if not path.exists():
        return False
    with path.open() as f:
        existing = json.load(f)
    for key, expected_ids in expected_split.items():
        if set(existing.get(key, [])) != set(expected_ids):
            return False
    return True


def evaluate_eta_predictions(
    *,
    root: Path,
    run_dir: Path,
    out_dir: Path,
    no_clip: bool,
) -> None:
    cmd = [
        sys.executable,
        "scripts/evaluate_eta_parallel_model.py",
        "--run-dir",
        str(run_dir),
        "--out-dir",
        str(out_dir),
    ]
    if no_clip:
        cmd.append("--no-clip")
    run(cmd, root)


def load_metrics(path: Path) -> dict[str, object]:
    with (path / "metrics.json").open() as f:
        return json.load(f)


def write_metric_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "fold",
        "n_train",
        "n_val",
        "n_test",
        "n_predictions",
        "eta_mae",
        "eta_rmse",
        "eta_r2",
        "eta_medae",
        "eta_mean_error",
        "n_clipped_low",
        "n_clipped_high",
        "eval_dir",
        "run_dir",
    ]
    for key in reversed(preferred):
        if key in fieldnames:
            fieldnames.insert(0, fieldnames.pop(fieldnames.index(key)))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_rows_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fold_summary_row(
    *,
    fold_number: int,
    split: dict[str, list[str]],
    run_dir: Path,
    data_dir: Path,
) -> dict[str, object]:
    return {
        "fold": fold_number,
        "n_train": len(split["id_train"]),
        "n_val": len(split["id_val"]),
        "n_test": len(split["id_test"]),
        "n_predictions": len(prediction_ids(run_dir)),
        "run_dir": str(run_dir),
        "data_dir": str(data_dir),
    }


def maybe_evaluate_fold(
    *,
    root: Path,
    task_dir: Path,
    fold_name: str,
    fold_row: dict[str, object],
    run_dir: Path,
    no_clip: bool,
) -> dict[str, object]:
    eval_dir = task_dir / "fold_evaluations" / fold_name
    evaluate_eta_predictions(
        root=root,
        run_dir=run_dir,
        out_dir=eval_dir,
        no_clip=no_clip,
    )
    metric_row = dict(fold_row)
    metric_row["eval_dir"] = str(eval_dir)
    metric_row.update(load_metrics(eval_dir))
    return metric_row


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()

    config = apply_cli_overrides(load_config(Path(args.config)), args)
    cv_config = config.get("cross_validation", {})
    n_folds = int(args.folds if args.folds is not None else cv_config.get("folds", 5))
    if n_folds not in {5, 10}:
        raise ValueError("--folds must be either 5 or 10.")

    if args.task_dir is None:
        config["task_dir"] = str(
            cv_config.get("task_dir", default_cv_path(config["task_dir"], f"_cv{n_folds}"))
        )
    if args.data_dir is None:
        config["data_dir"] = str(
            cv_config.get("data_dir", default_cv_path(config["data_dir"], f"_cv{n_folds}"))
        )

    skip_prepare = args.skip_prepare or bool(cv_config.get("skip_prepare", False))
    skip_train = args.skip_train or bool(cv_config.get("skip_train", False))
    skip_eval = args.skip_eval or bool(cv_config.get("skip_eval", False))
    overwrite = args.overwrite or bool(cv_config.get("overwrite", False))
    keep_existing_splits = args.keep_existing_splits or bool(
        cv_config.get("keep_existing_splits", False)
    )
    folds_to_run = parse_fold_selection(args.folds_to_run, n_folds)

    root = Path.cwd()
    labels = load_labels(Path(config["labels"]))
    training = config["training"]
    splits, labels_by_uid = kfold_splits(labels, training, n_folds)
    if args.source_cv_data_root:
        source_data_root = Path(args.source_cv_data_root)
        splits = load_source_cv_splits(
            source_data_root=source_data_root,
            valid_ids={str(row["uid"]) for row in labels},
            n_folds=n_folds,
        )
        print(f"Using fixed source CV splits: {source_data_root}", flush=True)

    task_dir = Path(config["task_dir"])
    data_root = Path(config["data_dir"])
    task_dir.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    fold_rows: list[dict[str, object]] = []
    fold_run_dirs: list[Path] = []
    fold_metric_rows: list[dict[str, object]] = []
    failed_rows: list[dict[str, object]] = []

    for fold_idx, split in enumerate(splits):
        fold_number = fold_idx + 1
        fold_name = f"fold_{fold_number:02d}"
        data_dir = data_root / fold_name
        run_dir = task_dir / fold_name

        if folds_to_run is not None and fold_number not in folds_to_run:
            if has_prediction_file(run_dir):
                fold_row = fold_summary_row(
                    fold_number=fold_number,
                    split=split,
                    run_dir=run_dir,
                    data_dir=data_dir,
                )
                fold_rows.append(fold_row)
                fold_run_dirs.append(run_dir)
                if not skip_eval:
                    fold_metric_rows.append(
                        maybe_evaluate_fold(
                            root=root,
                            task_dir=task_dir,
                            fold_name=fold_name,
                            fold_row=fold_row,
                            run_dir=run_dir,
                            no_clip=args.no_clip,
                        )
                    )
            continue

        split_written = False
        prepared_now = False
        if (skip_prepare or keep_existing_splits) and (data_dir / "config.json").exists():
            refresh_eta_config(data_dir, run_dir, config)
            if keep_existing_splits and not existing_split_matches(data_dir, split):
                print(
                    f"Existing split for eta_parallel {fold_name} does not "
                    "match the requested CV split; rewriting split artifacts.",
                    flush=True,
                )
                write_split_artifacts(
                    data_dir=data_dir,
                    run_dir=run_dir,
                    split=split,
                    labels_by_uid=labels_by_uid,
                )
                split_written = True
        else:
            prepare_eta_data(root, config, data_dir)
            refresh_eta_config(data_dir, run_dir, config)
            prepared_now = True

        if (not keep_existing_splits or prepared_now) and not split_written:
            write_split_artifacts(
                data_dir=data_dir,
                run_dir=run_dir,
                split=split,
                labels_by_uid=labels_by_uid,
            )

        if not skip_train:
            if overwrite or not has_prediction_file(run_dir):
                try:
                    train_submodel(root, data_dir, run_dir)
                except Exception as exc:
                    failed_row: dict[str, object] = {
                        "fold": fold_number,
                        "run_dir": str(run_dir),
                        "data_dir": str(data_dir),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                    returncode = getattr(exc, "returncode", None)
                    if returncode is not None:
                        failed_row["returncode"] = returncode
                    cmd = getattr(exc, "cmd", None)
                    if cmd is not None:
                        failed_row["cmd"] = " ".join(str(part) for part in cmd)
                    failed_rows.append(failed_row)
                    if not args.continue_on_failure:
                        raise
                    print(
                        f"Fold {fold_name} failed; continuing because "
                        "--continue-on-failure was set.",
                        flush=True,
                    )
                    continue
            else:
                print(f"Skipping eta_parallel {fold_name}; prediction file exists.")

        if not has_prediction_file(run_dir):
            raise FileNotFoundError(
                f"Missing prediction file for eta_parallel {fold_name}: {run_dir}"
            )

        ids = prediction_ids(run_dir)
        fold_run_dirs.append(run_dir)
        fold_row = fold_summary_row(
            fold_number=fold_number,
            split=split,
            run_dir=run_dir,
            data_dir=data_dir,
        )
        fold_rows.append(fold_row)

        if not skip_eval:
            fold_metric_rows.append(
                maybe_evaluate_fold(
                    root=root,
                    task_dir=task_dir,
                    fold_name=fold_name,
                    fold_row=fold_row,
                    run_dir=run_dir,
                    no_clip=args.no_clip,
                )
            )

    write_metric_rows_csv(task_dir / f"cv{n_folds}_fold_summary.csv", fold_rows)
    if fold_metric_rows:
        write_metric_rows_csv(
            task_dir / f"cv{n_folds}_fold_metrics_summary.csv",
            fold_metric_rows,
        )
        (task_dir / f"cv{n_folds}_fold_metrics_summary.json").write_text(
            json.dumps(fold_metric_rows, indent=2) + "\n"
        )
    if failed_rows:
        write_rows_csv(task_dir / f"cv{n_folds}_failed_folds.csv", failed_rows)
        (task_dir / f"cv{n_folds}_failed_folds.json").write_text(
            json.dumps(failed_rows, indent=2) + "\n"
        )

    if len(fold_run_dirs) != n_folds:
        missing = [
            f"fold_{idx + 1:02d}"
            for idx in range(n_folds)
            if not has_prediction_file(task_dir / f"fold_{idx + 1:02d}")
        ]
        missing_summary = {
            "folds": n_folds,
            "completed_prediction_folds": len(fold_run_dirs),
            "missing_folds": missing,
            "failed_folds": failed_rows,
        }
        (task_dir / f"cv{n_folds}_incomplete_summary.json").write_text(
            json.dumps(missing_summary, indent=2) + "\n"
        )
        if args.continue_on_failure:
            print(
                "CV run is incomplete; skipping OOF aggregation because "
                "prediction files are missing for: " + ", ".join(missing),
                flush=True,
            )
            return
        raise FileNotFoundError(
            "Cannot aggregate CV predictions because some folds are missing: "
            + ", ".join(missing)
        )

    aggregate_dir = task_dir / "oof_predictions"
    aggregate_stats = aggregate_predictions(fold_run_dirs, aggregate_dir)

    summary = {
        "folds": n_folds,
        "split": {
            "test_fold_fraction": 1.0 / n_folds,
            "validation_fraction": float(training.get("val_ratio", 0.1)),
            "validation_source": "non_test_pool",
        },
        "target": "eta_parallel",
        "cluster_counts": dict(Counter(labels_by_uid.values())),
        "aggregate_predictions": aggregate_stats,
    }
    (task_dir / f"cv{n_folds}_aggregate_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    if not skip_eval:
        evaluate_eta_predictions(
            root=root,
            run_dir=aggregate_dir,
            out_dir=task_dir / "evaluation",
            no_clip=args.no_clip,
        )

    print(f"\nEta_parallel {n_folds}-fold CV workflow complete.", flush=True)
    print(f"CV task directory: {task_dir}", flush=True)
    print(f"OOF prediction directory: {aggregate_dir}", flush=True)
    if not skip_eval:
        print(f"OOF evaluation directory: {task_dir / 'evaluation'}", flush=True)


if __name__ == "__main__":
    workflow_start_time = time.perf_counter()
    parsed_args = parse_args()
    try:
        main(parsed_args)
    finally:
        if not parsed_args.suppress_timer:
            elapsed = time.perf_counter() - workflow_start_time
            print(
                f"\nTotal eta_parallel CV elapsed time: {format_elapsed(elapsed)}",
                flush=True,
            )
