#!/usr/bin/env python
"""Train a standalone binary ALIGNN classifier for in-plane polarizability.

The positive class is defined as:
    0.5 * (xx + yy) > threshold

This script is intentionally independent from the split tensor workflow. It
prepares a folder-style ALIGNN dataset, writes a config.json, and optionally
starts training through alignn.train_alignn.train_for_folder.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_LABELS = "static_lattice_polarizability_tensors.csv"
DEFAULT_STRUCTURES = "structures_static_lattice_2001.json"
DEFAULT_DESCRIPTORS = "aux_physical_priors.csv"
DEFAULT_DATA_DIR = "data/inplane_avg_gt_16p1_classifier"
DEFAULT_RUN_DIR = "runs/inplane_avg_gt_16p1_classifier"
TENSOR_COLUMNS = ["xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare and train a standalone binary classifier for whether "
            "(xx + yy) / 2 is greater than a threshold."
        )
    )
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--structures", default=DEFAULT_STRUCTURES)
    parser.add_argument("--descriptors", default=DEFAULT_DESCRIPTORS)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--threshold", type=float, default=16.1)
    parser.add_argument(
        "--min-negative-retention",
        type=float,
        default=0.9,
        help=(
            "Minimum validation specificity/negative-class retention used "
            "by max_recall_at_min_specificity threshold selection."
        ),
    )
    parser.add_argument(
        "--min-positive-recall",
        type=float,
        default=0.9,
        help=(
            "Minimum validation positive-class recall used by "
            "max_specificity_at_min_recall threshold selection."
        ),
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=None,
        help=(
            "Optional fixed probability threshold for class 1. Leave unset "
            "to select it from the validation set."
        ),
    )
    parser.add_argument(
        "--threshold-strategy",
        choices=[
            "max_specificity_at_min_recall",
            "max_recall_at_min_specificity",
        ],
        default="max_specificity_at_min_recall",
        help=(
            "Validation-set rule used when --decision-threshold is unset."
        ),
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help=(
            "Number of stratified cross-validation folds. The default 5 "
            "trains fold_01 ... fold_05 and aggregates the final metrics."
        ),
    )
    parser.add_argument(
        "--single-split",
        action="store_true",
        help="Use the legacy single train/val/test split instead of CV.",
    )
    parser.add_argument(
        "--full-train",
        action="store_true",
        help=(
            "Train a final deployment model with every usable sample in the "
            "training set. The validation/test ids are also populated with "
            "all usable samples only to keep the ALIGNN training and "
            "prediction machinery available; metrics are in-sample."
        ),
    )
    parser.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="balanced",
        help=(
            "balanced writes per-sample weights so positive examples receive "
            "n_negative / n_positive relative weight."
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only write the ALIGNN dataset/config files; do not train.",
    )
    return parser.parse_args()


def load_labels(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in labels file: {path}")
    missing = {"uid", *TENSOR_COLUMNS} - set(rows[0])
    if missing:
        raise ValueError(f"Labels file is missing columns: {sorted(missing)}")
    return rows


def load_structures(path: Path) -> dict[str, dict]:
    with path.open() as f:
        rows = json.load(f)
    return {str(row["uid"]): row["structure"] for row in rows}


def load_descriptor_ids(path: Path) -> set[str]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "uid" not in reader.fieldnames:
            raise ValueError("Descriptor CSV must include a uid header column.")
        return {str(row["uid"]).strip() for row in reader}


def inplane_average(row: dict[str, str]) -> float:
    return 0.5 * (float(row["xx"]) + float(row["yy"]))


def class_label(row: dict[str, str], threshold: float) -> int:
    return int(inplane_average(row) > threshold)


def split_ids_by_class(
    rows: list[dict[str, str]],
    threshold: float,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    if train_ratio + val_ratio + test_ratio > 1.0 + 1e-12:
        raise ValueError("train_ratio + val_ratio + test_ratio must be <= 1.")

    rng = random.Random(seed)
    groups = {0: [], 1: []}
    for row in rows:
        groups[class_label(row, threshold)].append(row["uid"])

    split = {"id_train": [], "id_val": [], "id_test": []}
    for ids in groups.values():
        ids = list(ids)
        rng.shuffle(ids)
        n = len(ids)
        n_test = int(test_ratio * n)
        n_val = int(val_ratio * n)
        if n >= 3:
            if test_ratio > 0 and n_test == 0:
                n_test = 1
            if val_ratio > 0 and n_val == 0:
                n_val = 1
            if n_test + n_val >= n:
                n_val = max(0, n - n_test - 1)
        n_train = n - n_val - n_test
        split["id_train"].extend(ids[:n_train])
        split["id_val"].extend(ids[n_train : n_train + n_val])
        split["id_test"].extend(ids[n_train + n_val :])

    for ids in split.values():
        rng.shuffle(ids)
    return split


def stratified_fold_ids(
    rows: list[dict[str, str]],
    threshold: float,
    folds: int,
    seed: int,
) -> list[list[str]]:
    if folds < 2:
        raise ValueError("--cv-folds must be at least 2 for CV mode.")
    rng = random.Random(seed)
    fold_ids = [[] for _ in range(folds)]
    groups = {0: [], 1: []}
    for row in rows:
        groups[class_label(row, threshold)].append(row["uid"])
    for ids in groups.values():
        ids = list(ids)
        rng.shuffle(ids)
        for idx, uid in enumerate(ids):
            fold_ids[idx % folds].append(uid)
    for ids in fold_ids:
        rng.shuffle(ids)
    return fold_ids


def cv_split_for_fold(
    rows: list[dict[str, str]],
    fold_ids: list[list[str]],
    fold_index: int,
    threshold: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    row_by_uid = {row["uid"]: row for row in rows}
    all_ids = set(row_by_uid)
    test_ids = set(fold_ids[fold_index])
    remaining_ids = sorted(all_ids - test_ids)

    rng = random.Random(seed + fold_index)
    val_ids = []
    for label in [0, 1]:
        label_total = sum(
            class_label(row, threshold) == label for row in rows
        )
        candidates = [
            uid
            for uid in remaining_ids
            if class_label(row_by_uid[uid], threshold) == label
        ]
        rng.shuffle(candidates)
        n_val = int(round(val_ratio * label_total))
        if val_ratio > 0 and label_total > 0 and n_val == 0:
            n_val = 1
        n_val = min(n_val, max(0, len(candidates) - 1))
        val_ids.extend(candidates[:n_val])

    val_set = set(val_ids)
    train_ids = sorted(set(remaining_ids) - val_set)
    test_list = sorted(test_ids)
    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    rng.shuffle(test_list)
    return {
        "id_train": train_ids,
        "id_val": val_ids,
        "id_test": test_list,
    }


def write_poscar(path: Path, uid: str, structure: dict) -> None:
    lattice = structure["lattice"]["matrix"]
    sites = structure["sites"]

    element_order: list[str] = []
    grouped: dict[str, list[dict]] = {}
    for site in sites:
        element = site["species"][0]["element"]
        if element not in grouped:
            element_order.append(element)
            grouped[element] = []
        grouped[element].append(site)

    lines = [uid, "1.0"]
    lines.extend("  " + "  ".join(f"{x:.16g}" for x in row) for row in lattice)
    lines.append("  " + "  ".join(element_order))
    lines.append("  " + "  ".join(str(len(grouped[el])) for el in element_order))
    lines.append("Direct")
    for element in element_order:
        for site in grouped[element]:
            lines.append("  " + "  ".join(f"{x:.16g}" for x in site["abc"]))
    path.write_text("\n".join(lines) + "\n")


def write_sample_weights(
    path: Path,
    rows: list[dict[str, str]],
    threshold: float,
    mode: str,
) -> dict[str, float]:
    labels = [class_label(row, threshold) for row in rows]
    counts = Counter(labels)
    weights = {}
    if mode == "balanced" and counts[1] > 0:
        positive_weight = counts[0] / counts[1]
    else:
        positive_weight = 1.0
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["uid", "class", "weight"])
        writer.writeheader()
        for row in rows:
            label = class_label(row, threshold)
            weight = positive_weight if label == 1 else 1.0
            weights[row["uid"]] = weight
            writer.writerow(
                {"uid": row["uid"], "class": label, "weight": weight}
            )
    return weights


def build_config(
    args: argparse.Namespace,
    sample_weight_path: Path,
    run_dir: Path | None = None,
) -> dict:
    use_sample_weights = args.class_weight != "none"
    output_dir = Path(args.run_dir) if run_dir is None else run_dir
    return {
        "dataset": "user_data",
        "target": "target",
        "atom_features": "cgcnn",
        "neighbor_strategy": "k-nearest",
        "id_tag": "jid",
        "dtype": "float32",
        "random_seed": args.split_seed,
        "classification_threshold": 0.5,
        "classification_decision_threshold": args.decision_threshold,
        "classification_threshold_strategy": args.threshold_strategy,
        "classification_min_specificity": args.min_negative_retention,
        "classification_min_recall": args.min_positive_recall,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "weight_decay": args.weight_decay,
        "learning_rate": args.learning_rate,
        "criterion": "mse",
        "optimizer": "adamw",
        "scheduler": "onecycle",
        "n_early_stopping": 10,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "write_checkpoint": True,
        "write_predictions": True,
        "store_outputs": True,
        "progress": True,
        "compute_line_graph": True,
        "cutoff": 8.0,
        "max_neighbors": 12,
        "keep_data_order": False,
        "use_lmdb": False,
        "output_dir": str(output_dir),
        "filename": "inplane_avg_classifier",
        "use_sample_weights": use_sample_weights,
        "sample_weight_csv": str(sample_weight_path.resolve())
        if use_sample_weights
        else None,
        "sample_weight_id_key": "uid",
        "sample_weight_value_key": "weight",
        "target_columns": ["inplane_avg_gt_threshold"],
        "prediction_columns": ["inplane_avg_gt_threshold"],
        "use_extra_features_csv": True,
        "extra_features_csv": str(Path(args.descriptors).resolve()),
        "extra_features_id_key": "uid",
        "model": {
            "name": "alignn_atomwise",
            "atom_input_features": 92,
            "alignn_layers": 4,
            "gcn_layers": 2,
            "embedding_features": 64,
            "hidden_features": 64,
            "output_features": 1,
            "calculate_gradient": False,
            "graphwise_weight": 1.0,
            "gradwise_weight": 0.0,
            "stresswise_weight": 0.0,
            "atomwise_weight": 0.0,
            "energy_mult_natoms": False,
            "extra_features": 0,
            "use_extra_feature_gated_fusion":  True,
            "use_inplane_direction_attention": False,
            "use_readout_attention": False,
            "use_component_readout_attention": False,
            "use_xy_readout_attention": False,
        },
    }


def load_usable_data(
    args: argparse.Namespace,
) -> tuple[list[dict[str, str]], dict[str, dict]]:
    labels_path = Path(args.labels)
    structures_path = Path(args.structures)
    descriptors_path = Path(args.descriptors)

    labels = load_labels(labels_path)
    structures = load_structures(structures_path)
    descriptor_ids = load_descriptor_ids(descriptors_path)

    usable_rows = []
    missing_structures = []
    missing_descriptors = []
    for row in labels:
        uid = row["uid"]
        if uid not in structures:
            missing_structures.append(uid)
            continue
        if uid not in descriptor_ids:
            missing_descriptors.append(uid)
            continue
        usable_rows.append(row)

    if missing_structures:
        raise ValueError(
            "Missing structures for "
            + f"{len(missing_structures)} samples. "
            + "Examples: "
            + ", ".join(missing_structures[:10])
        )
    if missing_descriptors:
        raise ValueError(
            "Missing descriptors for "
            + f"{len(missing_descriptors)} samples. "
            + "Examples: "
            + ", ".join(missing_descriptors[:10])
        )
    return usable_rows, structures


def write_prepared_dataset(
    args: argparse.Namespace,
    usable_rows: list[dict[str, str]],
    structures: dict[str, dict],
    data_dir: Path,
    run_dir: Path,
    split: dict[str, list[str]],
    summary_extra: dict | None = None,
) -> dict:
    data_dir.mkdir(parents=True, exist_ok=True)
    id_prop_path = data_dir / "id_prop.csv"
    with id_prop_path.open("w", newline="") as f:
        writer = csv.writer(f)
        for row in usable_rows:
            uid = row["uid"]
            write_poscar(data_dir / uid, uid, structures[uid])
            writer.writerow([uid, class_label(row, args.threshold)])

    labels_by_uid = {
        row["uid"]: class_label(row, args.threshold) for row in usable_rows
    }
    (data_dir / "ids_train_val_test.json").write_text(
        json.dumps(split, indent=2) + "\n"
    )
    (data_dir / "stratify_labels.csv").write_text(
        "uid,class\n"
        + "\n".join(
            f"{uid},{label}" for uid, label in sorted(labels_by_uid.items())
        )
        + "\n"
    )

    sample_weight_path = data_dir / "sample_weights.csv"
    weights = write_sample_weights(
        path=sample_weight_path,
        rows=usable_rows,
        threshold=args.threshold,
        mode=args.class_weight,
    )

    config = build_config(args, sample_weight_path, run_dir=run_dir)
    (data_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    class_counts = Counter(labels_by_uid.values())
    alpha_values = [inplane_average(row) for row in usable_rows]
    positive_weight = next(
        (
            weights[row["uid"]]
            for row in usable_rows
            if class_label(row, args.threshold) == 1
        ),
        1.0,
    )
    summary = {
        "samples": len(usable_rows),
        "labels": str(Path(args.labels)),
        "structures": str(Path(args.structures)),
        "descriptors": str(Path(args.descriptors)),
        "threshold": args.threshold,
        "positive_definition": "(xx + yy) / 2 > threshold",
        "class_counts": {
            "negative": class_counts.get(0, 0),
            "positive": class_counts.get(1, 0),
        },
        "alpha_inplane": {
            "min": min(alpha_values),
            "max": max(alpha_values),
        },
        "split_counts": {key: len(value) for key, value in split.items()},
        "class_weight": {
            "mode": args.class_weight,
            "negative_weight": 1.0,
            "positive_weight": positive_weight,
            "sample_weight_csv": str(sample_weight_path),
        },
        "decision_threshold": {
            "fixed": args.decision_threshold,
            "strategy": args.threshold_strategy,
            "min_negative_retention": args.min_negative_retention,
            "min_positive_recall": args.min_positive_recall,
        },
        "id_prop_csv": str(id_prop_path),
        "config": str(data_dir / "config.json"),
        "run_dir": str(run_dir),
    }
    if summary_extra:
        summary.update(summary_extra)
    (data_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def prepare_dataset(args: argparse.Namespace) -> dict:
    usable_rows, structures = load_usable_data(args)
    split = split_ids_by_class(
        rows=usable_rows,
        threshold=args.threshold,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.split_seed,
    )
    return write_prepared_dataset(
        args=args,
        usable_rows=usable_rows,
        structures=structures,
        data_dir=Path(args.data_dir),
        run_dir=Path(args.run_dir),
        split=split,
        summary_extra={"mode": "single_split"},
    )


def prepare_full_training_dataset(args: argparse.Namespace) -> dict:
    usable_rows, structures = load_usable_data(args)
    ids = [row["uid"] for row in usable_rows]
    split = {
        "id_train": list(ids),
        "id_val": list(ids),
        "id_test": list(ids),
    }
    return write_prepared_dataset(
        args=args,
        usable_rows=usable_rows,
        structures=structures,
        data_dir=Path(args.data_dir),
        run_dir=Path(args.run_dir),
        split=split,
        summary_extra={
            "mode": "full_train",
            "full_train_note": (
                "All usable samples are included in id_train. id_val and "
                "id_test also contain all usable samples so the existing "
                "ALIGNN training/prediction code can run; reported metrics "
                "from this run are in-sample and should not be used as an "
                "unbiased estimate."
            ),
        },
    )


def prepare_cross_validation(args: argparse.Namespace) -> dict:
    usable_rows, structures = load_usable_data(args)
    fold_ids = stratified_fold_ids(
        rows=usable_rows,
        threshold=args.threshold,
        folds=args.cv_folds,
        seed=args.split_seed,
    )
    fold_summaries = []
    for fold_index in range(args.cv_folds):
        fold_name = f"fold_{fold_index + 1:02d}"
        split = cv_split_for_fold(
            rows=usable_rows,
            fold_ids=fold_ids,
            fold_index=fold_index,
            threshold=args.threshold,
            val_ratio=args.val_ratio,
            seed=args.split_seed,
        )
        fold_summary = write_prepared_dataset(
            args=args,
            usable_rows=usable_rows,
            structures=structures,
            data_dir=Path(args.data_dir) / fold_name,
            run_dir=Path(args.run_dir) / fold_name,
            split=split,
            summary_extra={
                "mode": "cross_validation",
                "fold": fold_index + 1,
                "cv_folds": args.cv_folds,
                "fold_name": fold_name,
                "cv_note": (
                    "id_test is the held-out fold; id_val is stratified "
                    "from the non-test samples using --val-ratio."
                ),
            },
        )
        fold_summaries.append(fold_summary)

    summary = {
        "mode": "cross_validation",
        "cv_folds": args.cv_folds,
        "samples": len(usable_rows),
        "labels": str(Path(args.labels)),
        "structures": str(Path(args.structures)),
        "descriptors": str(Path(args.descriptors)),
        "threshold": args.threshold,
        "positive_definition": "(xx + yy) / 2 > threshold",
        "decision_threshold": {
            "fixed": args.decision_threshold,
            "strategy": args.threshold_strategy,
            "min_negative_retention": args.min_negative_retention,
            "min_positive_recall": args.min_positive_recall,
        },
        "class_counts": dict(
            Counter(class_label(row, args.threshold) for row in usable_rows)
        ),
        "data_dir": str(Path(args.data_dir)),
        "run_dir": str(Path(args.run_dir)),
        "folds": fold_summaries,
    }
    base_data_dir = Path(args.data_dir)
    base_data_dir.mkdir(parents=True, exist_ok=True)
    (base_data_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (base_data_dir / "summary_cv.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def train_prepared_dataset(
    args: argparse.Namespace, data_dir: Path, run_dir: Path
) -> None:
    from alignn.train_alignn import train_for_folder

    train_for_folder(
        rank=0,
        world_size=0,
        root_dir=str(data_dir),
        config_name=str(data_dir / "config.json"),
        classification_threshold=None,
        batch_size=None,
        epochs=None,
        id_key="jid",
        target_key="target",
        file_format="poscar",
        output_dir=str(run_dir),
        extra_features_csv=args.descriptors,
        extra_features_id_key="uid",
    )


def classification_metrics(
    targets: list[int],
    probabilities: list[float],
    predictions: list[int],
    threshold: float | str = 0.5,
) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    cm = confusion_matrix(targets, predictions, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    negative_total = tn + fp
    specificity = None
    false_positive_rate = None
    if negative_total > 0:
        specificity = float(tn / negative_total)
        false_positive_rate = float(fp / negative_total)
    metrics = {
        "n_samples": int(len(targets)),
        "threshold": threshold,
        "decision_threshold": threshold,
        "positive_label": 1,
        "negative_label": 0,
        "class_counts": {
            "negative": int(sum(target == 0 for target in targets)),
            "positive": int(sum(target == 1 for target in targets)),
        },
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(
            precision_score(targets, predictions, zero_division=0)
        ),
        "recall": float(recall_score(targets, predictions, zero_division=0)),
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "specificity": specificity,
        "negative_retention": specificity,
        "false_positive_rate": false_positive_rate,
        "roc_auc": None,
        "average_precision": None,
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": cm.astype(int).tolist(),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(targets, probabilities))
    except ValueError:
        pass
    try:
        metrics["average_precision"] = float(
            average_precision_score(targets, probabilities)
        )
    except ValueError:
        pass
    return metrics


def summarize_fold_metrics(fold_metrics: list[dict]) -> dict:
    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "negative_retention",
        "false_positive_rate",
        "roc_auc",
        "average_precision",
        "decision_threshold",
    ]
    summary = {}
    for metric in metric_names:
        values = [
            float(item[metric])
            for item in fold_metrics
            if item.get(metric) is not None
        ]
        if not values:
            summary[metric] = {"mean": None, "std": None}
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        summary[metric] = {
            "mean": mean,
            "std": variance**0.5,
        }
    return summary


def aggregate_cross_validation_results(
    args: argparse.Namespace, cv_summary: dict
) -> dict:
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    combined_path = run_dir / "cv_prediction_results_test_set.csv"
    targets: list[int] = []
    probabilities: list[float] = []
    predictions: list[int] = []
    fold_metrics = []

    with combined_path.open("w", newline="") as out_file:
        writer = csv.writer(out_file)
        writer.writerow(
            [
                "fold",
                "id",
                "target",
                "probability",
                "prediction",
                "decision_threshold",
            ]
        )
        for fold_summary in cv_summary["folds"]:
            fold = int(fold_summary["fold"])
            fold_run_dir = Path(fold_summary["run_dir"])
            metrics_path = fold_run_dir / "classification_metrics.json"
            with metrics_path.open() as f:
                metrics = json.load(f)
            fold_threshold = metrics.get(
                "decision_threshold", metrics.get("threshold", "")
            )
            pred_path = fold_run_dir / "prediction_results_test_set.csv"
            with pred_path.open(newline="") as pred_file:
                reader = csv.DictReader(pred_file)
                for row in reader:
                    target = int(row["target"])
                    probability = float(row["probability"])
                    prediction = int(row["prediction"])
                    decision_threshold = (
                        row.get("decision_threshold") or fold_threshold
                    )
                    writer.writerow(
                        [
                            fold,
                            row["id"],
                            target,
                            probability,
                            prediction,
                            decision_threshold,
                        ]
                    )
                    targets.append(target)
                    probabilities.append(probability)
                    predictions.append(prediction)

            metrics["fold"] = fold
            metrics["run_dir"] = str(fold_run_dir)
            fold_metrics.append(metrics)

    pooled = classification_metrics(
        targets,
        probabilities,
        predictions,
        threshold="per_fold_selected",
    )
    cv_metrics = {
        "mode": "cross_validation",
        "cv_folds": int(args.cv_folds),
        "threshold": float(args.threshold),
        "positive_definition": "(xx + yy) / 2 > threshold",
        "decision_threshold": {
            "fixed": args.decision_threshold,
            "strategy": args.threshold_strategy,
            "min_negative_retention": args.min_negative_retention,
            "min_positive_recall": args.min_positive_recall,
            "pooled": "per_fold_selected",
        },
        "pooled_test_metrics": pooled,
        "fold_metric_summary": summarize_fold_metrics(fold_metrics),
        "fold_metrics": fold_metrics,
        "combined_prediction_file": str(combined_path),
    }

    for filename in [
        "classification_metrics.json",
        "cv_classification_metrics.json",
    ]:
        (run_dir / filename).write_text(json.dumps(cv_metrics, indent=2) + "\n")

    with (run_dir / "classification_metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "pooled", "fold_mean", "fold_std"])
        fold_summary = cv_metrics["fold_metric_summary"]
        for metric in [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "specificity",
            "negative_retention",
            "false_positive_rate",
            "roc_auc",
            "average_precision",
            "decision_threshold",
        ]:
            writer.writerow(
                [
                    metric,
                    pooled[metric],
                    fold_summary[metric]["mean"],
                    fold_summary[metric]["std"],
                ]
            )
        cm = pooled["confusion_matrix"]
        for key in ["tn", "fp", "fn", "tp"]:
            writer.writerow([key, cm[key], "", ""])
    (run_dir / "cv_classification_metrics.csv").write_text(
        (run_dir / "classification_metrics.csv").read_text()
    )
    return cv_metrics


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.min_negative_retention <= 1.0:
        raise ValueError("--min-negative-retention must be between 0 and 1.")
    if not 0.0 <= args.min_positive_recall <= 1.0:
        raise ValueError("--min-positive-recall must be between 0 and 1.")
    if args.decision_threshold is not None and not (
        0.0 <= args.decision_threshold <= 1.0
    ):
        raise ValueError("--decision-threshold must be between 0 and 1.")
    if args.full_train and args.single_split:
        raise ValueError("--full-train and --single-split are mutually exclusive.")
    if args.full_train:
        summary = prepare_full_training_dataset(args)
        print(json.dumps(summary, indent=2))
        if args.prepare_only:
            return
        train_prepared_dataset(
            args=args,
            data_dir=Path(args.data_dir),
            run_dir=Path(args.run_dir),
        )
        return
    use_cv = not args.single_split and args.cv_folds > 1
    if use_cv:
        summary = prepare_cross_validation(args)
        print(json.dumps(summary, indent=2))
        if args.prepare_only:
            return
        for fold_summary in summary["folds"]:
            print(
                "\n>>> Training "
                + f"fold {fold_summary['fold']}/{args.cv_folds}",
                flush=True,
            )
            train_prepared_dataset(
                args=args,
                data_dir=Path(fold_summary["config"]).parent,
                run_dir=Path(fold_summary["run_dir"]),
            )
        cv_metrics = aggregate_cross_validation_results(args, summary)
        print(json.dumps(cv_metrics, indent=2))
    else:
        summary = prepare_dataset(args)
        print(json.dumps(summary, indent=2))
        if args.prepare_only:
            return
        train_prepared_dataset(
            args=args,
            data_dir=Path(args.data_dir),
            run_dir=Path(args.run_dir),
        )


if __name__ == "__main__":
    main()
