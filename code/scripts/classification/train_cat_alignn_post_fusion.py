#!/usr/bin/env python
"""Post-fuse CatBoost gate OOF probabilities with ALIGNN OOF probabilities.

The evaluation target is recall-first:
    keep positive recall >= --min-positive-recall, then maximize specificity.

This script trains only a lightweight meta model on already generated OOF
probabilities. It does not retrain CatBoost or ALIGNN.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_CAT_PRED = (
    "cat/high_alpha_inplane_gate/reports/oof_gate_predictions.csv"
)
DEFAULT_ALIGNN_PRED = (
    "runs/inplane_avg_gt_16p1_classifier_02/cv_prediction_results_test_set.csv"
)
DEFAULT_OUTPUT_DIR = "runs/cat_alignn_post_fusion"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a lightweight post-fusion classifier from CatBoost and "
            "ALIGNN out-of-fold probabilities."
        )
    )
    parser.add_argument("--cat-pred", default=DEFAULT_CAT_PRED)
    parser.add_argument(
        "--alignn-pred",
        action="append",
        default=None,
        help=(
            "ALIGNN cv_prediction_results_test_set.csv. Can be passed more "
            "than once. Defaults to the latest classifier_02 result."
        ),
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-positive-recall", type=float, default=0.90)
    parser.add_argument("--max-fn", type=int, default=17)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--feature-mode",
        choices=["probabilities", "probabilities_logits"],
        default="probabilities_logits",
    )
    return parser.parse_args()


def _clip_probability(value: float) -> float:
    return min(max(float(value), 1e-7), 1.0 - 1e-7)


def _logit(value: float) -> float:
    value = _clip_probability(value)
    return math.log(value / (1.0 - value))


def read_cat_predictions(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "sample_id",
            "true_high_alpha",
            "p_gate_a_high_alpha",
            "p_gate_b_severe_high_alpha",
            "pred_gate_a_high_alpha",
            "pred_high_alpha_gate",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Cat prediction file missing columns: {missing}")
        for row in reader:
            rows[row["sample_id"]] = {
                "target": int(row["true_high_alpha"]),
                "p_cat_gate_a": float(row["p_gate_a_high_alpha"]),
                "p_cat_gate_b": float(row["p_gate_b_severe_high_alpha"]),
                "pred_cat_gate_a": int(row["pred_gate_a_high_alpha"]),
                "pred_cat_combined": int(row["pred_high_alpha_gate"]),
            }
    return rows


def read_alignn_predictions(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "target", "probability", "prediction"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"ALIGNN prediction file missing columns: {missing}")
        for row in reader:
            rows[row["id"]] = {
                "target": int(row["target"]),
                "probability": float(row["probability"]),
                "prediction": int(row["prediction"]),
            }
    return rows


def confusion_counts(targets: list[int], predictions: list[int]) -> dict:
    tn = fp = fn = tp = 0
    for target, prediction in zip(targets, predictions):
        if target == 0 and prediction == 0:
            tn += 1
        elif target == 0 and prediction == 1:
            fp += 1
        elif target == 1 and prediction == 0:
            fn += 1
        elif target == 1 and prediction == 1:
            tp += 1
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def metrics_from_predictions(
    targets: list[int], predictions: list[int], label
) -> dict:
    counts = confusion_counts(targets, predictions)
    return metrics_from_counts(
        label=label,
        n_samples=len(targets),
        tn=counts["tn"],
        fp=counts["fp"],
        fn=counts["fn"],
        tp=counts["tp"],
    )


def metrics_from_counts(label, n_samples: int, tn: int, fp: int, fn: int, tp: int):
    negative = tn + fp
    positive = tp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positive if positive else 0.0
    specificity = tn / negative if negative else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "label": label,
        "n_samples": n_samples,
        "negative": negative,
        "positive": positive,
        "accuracy": (tn + tp) / n_samples if n_samples else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
        "negative_retention": specificity,
        "false_positive_rate": fp / negative if negative else 0.0,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def metrics_at_threshold(
    targets: list[int], probabilities: list[float], threshold: float
) -> dict:
    predictions = [int(probability >= threshold) for probability in probabilities]
    metrics = metrics_from_predictions(targets, predictions, threshold)
    metrics["threshold"] = threshold
    return metrics


def threshold_curve(targets: list[int], probabilities: list[float]) -> list[dict]:
    """Return metrics at all useful probability cutoffs in O(n log n)."""
    pairs = sorted(
        ((float(probability), int(target)) for target, probability in zip(targets, probabilities)),
        key=lambda item: item[0],
        reverse=True,
    )
    n_samples = len(targets)
    total_positive = sum(int(target) for target in targets)
    total_negative = n_samples - total_positive
    tp = 0
    fp = 0
    fn = total_positive
    tn = total_negative
    curve = []
    index = 0
    while index < len(pairs):
        threshold = pairs[index][0]
        while index < len(pairs) and pairs[index][0] == threshold:
            target = pairs[index][1]
            if target == 1:
                tp += 1
                fn -= 1
            else:
                fp += 1
                tn -= 1
            index += 1
        metrics = metrics_from_counts(
            label=threshold,
            n_samples=n_samples,
            tn=tn,
            fp=fp,
            fn=fn,
            tp=tp,
        )
        metrics["threshold"] = threshold
        curve.append(metrics)
    if not curve or curve[-1]["threshold"] > 0.0:
        metrics = metrics_from_counts(
            label=0.0,
            n_samples=n_samples,
            tn=0,
            fp=total_negative,
            fn=0,
            tp=total_positive,
        )
        metrics["threshold"] = 0.0
        curve.append(metrics)
    return curve


def best_at_min_recall(
    targets: list[int], probabilities: list[float], min_recall: float
) -> dict:
    candidates = threshold_curve(targets, probabilities)
    candidates = [item for item in candidates if item["recall"] >= min_recall]
    if not candidates:
        raise ValueError("No threshold can satisfy the minimum recall.")
    return max(
        candidates,
        key=lambda item: (
            item["specificity"],
            item["precision"],
            item["f1"],
            item["threshold"],
        ),
    )


def best_at_max_fn(
    targets: list[int], probabilities: list[float], max_fn: int
) -> dict:
    candidates = threshold_curve(targets, probabilities)
    candidates = [item for item in candidates if item["fn"] <= max_fn]
    if not candidates:
        raise ValueError("No threshold can satisfy the max FN constraint.")
    return max(
        candidates,
        key=lambda item: (
            item["specificity"],
            item["precision"],
            item["f1"],
            item["threshold"],
        ),
    )


def build_aligned_table(
    cat_rows: dict[str, dict],
    alignn_rows_by_name: dict[str, dict[str, dict]],
    feature_mode: str,
) -> tuple[list[dict], list[str]]:
    common_ids = set(cat_rows)
    for rows in alignn_rows_by_name.values():
        common_ids &= set(rows)
    aligned = []
    feature_names = ["p_cat_gate_a", "p_cat_gate_b"]
    if feature_mode == "probabilities_logits":
        feature_names += ["logit_cat_gate_a", "logit_cat_gate_b"]
    for name in alignn_rows_by_name:
        feature_names.append(f"p_{name}")
        if feature_mode == "probabilities_logits":
            feature_names.append(f"logit_{name}")

    mismatched_labels = []
    for uid in sorted(common_ids):
        target = cat_rows[uid]["target"]
        item = {
            "sample_id": uid,
            "target": target,
            "p_cat_gate_a": cat_rows[uid]["p_cat_gate_a"],
            "p_cat_gate_b": cat_rows[uid]["p_cat_gate_b"],
            "pred_cat_gate_a": cat_rows[uid]["pred_cat_gate_a"],
            "pred_cat_combined": cat_rows[uid]["pred_cat_combined"],
        }
        item["logit_cat_gate_a"] = _logit(item["p_cat_gate_a"])
        item["logit_cat_gate_b"] = _logit(item["p_cat_gate_b"])
        for name, rows in alignn_rows_by_name.items():
            if rows[uid]["target"] != target:
                mismatched_labels.append(uid)
            probability = rows[uid]["probability"]
            item[f"p_{name}"] = probability
            item[f"logit_{name}"] = _logit(probability)
            item[f"pred_{name}"] = rows[uid]["prediction"]
        aligned.append(item)
    if mismatched_labels:
        raise ValueError(
            "Mismatched labels for aligned samples: "
            + ", ".join(mismatched_labels[:20])
        )
    return aligned, feature_names


def train_stacking_oof(
    rows: list[dict],
    feature_names: list[str],
    n_splits: int,
    seed: int,
) -> tuple[list[float], object]:
    x = np.array([[row[name] for name in feature_names] for row in rows])
    y = np.array([row["target"] for row in rows])
    oof = np.zeros(len(rows), dtype=float)
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    for train_idx, val_idx in splitter.split(x, y):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                C=0.5,
                solver="liblinear",
                max_iter=1000,
                random_state=seed,
            ),
        )
        model.fit(x[train_idx], y[train_idx])
        oof[val_idx] = model.predict_proba(x[val_idx])[:, 1]

    final_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            C=0.5,
            solver="liblinear",
            max_iter=1000,
            random_state=seed,
        ),
    )
    final_model.fit(x, y)
    return oof.tolist(), final_model


def weighted_average_search(
    targets: list[int],
    cat_probability: list[float],
    alignn_probability: list[float],
    min_recall: float,
    max_fn: int,
) -> list[dict]:
    results = []
    for step in range(0, 101):
        cat_weight = step / 100.0
        probabilities = [
            cat_weight * p_cat + (1.0 - cat_weight) * p_alignn
            for p_cat, p_alignn in zip(cat_probability, alignn_probability)
        ]
        recall_metrics = best_at_min_recall(
            targets, probabilities, min_recall
        )
        fn_metrics = best_at_max_fn(targets, probabilities, max_fn)
        results.append(
            {
                "cat_weight": cat_weight,
                "alignn_weight": 1.0 - cat_weight,
                "recall90_threshold": recall_metrics["threshold"],
                "recall90_specificity": recall_metrics["specificity"],
                "recall90_recall": recall_metrics["recall"],
                "recall90_precision": recall_metrics["precision"],
                "recall90_f1": recall_metrics["f1"],
                "recall90_tn": recall_metrics["tn"],
                "recall90_fp": recall_metrics["fp"],
                "recall90_fn": recall_metrics["fn"],
                "recall90_tp": recall_metrics["tp"],
                "fn_threshold": fn_metrics["threshold"],
                "fn_specificity": fn_metrics["specificity"],
                "fn_recall": fn_metrics["recall"],
                "fn_precision": fn_metrics["precision"],
                "fn_f1": fn_metrics["f1"],
                "fn_tn": fn_metrics["tn"],
                "fn_fp": fn_metrics["fp"],
                "fn_fn": fn_metrics["fn"],
                "fn_tp": fn_metrics["tp"],
            }
        )
    return results


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.min_positive_recall <= 1.0:
        raise ValueError("--min-positive-recall must be between 0 and 1.")
    if args.max_fn < 0:
        raise ValueError("--max-fn must be >= 0.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    alignn_paths = args.alignn_pred or [DEFAULT_ALIGNN_PRED]
    cat_rows = read_cat_predictions(Path(args.cat_pred))
    alignn_rows_by_name = {}
    for index, path_text in enumerate(alignn_paths, start=1):
        path = Path(path_text)
        name = path.parent.name
        if name in alignn_rows_by_name:
            name = f"{name}_{index}"
        alignn_rows_by_name[name] = read_alignn_predictions(path)

    rows, feature_names = build_aligned_table(
        cat_rows, alignn_rows_by_name, args.feature_mode
    )
    targets = [row["target"] for row in rows]
    fusion_oof, final_model = train_stacking_oof(
        rows, feature_names, args.n_splits, args.seed
    )
    for row, probability in zip(rows, fusion_oof):
        row["p_fusion_logistic"] = probability

    cat_gate_a_prob = [row["p_cat_gate_a"] for row in rows]
    cat_gate_b_prob = [row["p_cat_gate_b"] for row in rows]
    first_alignn_name = next(iter(alignn_rows_by_name))
    first_alignn_prob = [row[f"p_{first_alignn_name}"] for row in rows]

    baselines = {
        "cat_gate_a_selected": metrics_from_predictions(
            targets, [row["pred_cat_gate_a"] for row in rows], "selected"
        ),
        "cat_combined_selected": metrics_from_predictions(
            targets, [row["pred_cat_combined"] for row in rows], "selected"
        ),
        f"{first_alignn_name}_selected": metrics_from_predictions(
            targets, [row[f"pred_{first_alignn_name}"] for row in rows], "selected"
        ),
        "cat_gate_a_recall_constraint": best_at_min_recall(
            targets, cat_gate_a_prob, args.min_positive_recall
        ),
        "cat_gate_a_fn_constraint": best_at_max_fn(
            targets, cat_gate_a_prob, args.max_fn
        ),
        "cat_gate_b_recall_constraint": best_at_min_recall(
            targets, cat_gate_b_prob, args.min_positive_recall
        ),
        "fusion_logistic_recall_constraint": best_at_min_recall(
            targets, fusion_oof, args.min_positive_recall
        ),
        "fusion_logistic_fn_constraint": best_at_max_fn(
            targets, fusion_oof, args.max_fn
        ),
    }
    for name in alignn_rows_by_name:
        probabilities = [row[f"p_{name}"] for row in rows]
        baselines[f"{name}_recall_constraint"] = best_at_min_recall(
            targets, probabilities, args.min_positive_recall
        )
        baselines[f"{name}_fn_constraint"] = best_at_max_fn(
            targets, probabilities, args.max_fn
        )

    weighted_results = weighted_average_search(
        targets,
        cat_gate_a_prob,
        first_alignn_prob,
        args.min_positive_recall,
        args.max_fn,
    )
    best_weight_recall = max(
        weighted_results,
        key=lambda item: (
            item["recall90_specificity"],
            item["recall90_precision"],
            item["recall90_f1"],
            item["cat_weight"],
        ),
    )
    best_weight_fn = max(
        weighted_results,
        key=lambda item: (
            item["fn_specificity"],
            item["fn_precision"],
            item["fn_f1"],
            item["cat_weight"],
        ),
    )

    write_rows(output_dir / "fusion_oof_predictions.csv", rows)
    write_rows(output_dir / "weighted_average_search.csv", weighted_results)

    threshold_scan = [
        metrics_at_threshold(targets, fusion_oof, round(step / 1000.0, 3))
        for step in range(0, 1001)
    ]
    write_rows(output_dir / "fusion_threshold_scan.csv", threshold_scan)

    with (output_dir / "fusion_logistic_model.pkl").open("wb") as f:
        pickle.dump(
            {
                "model": final_model,
                "feature_names": feature_names,
                "alignn_sources": alignn_paths,
                "cat_source": args.cat_pred,
            },
            f,
        )

    summary = {
        "task": "cat_alignn_post_fusion",
        "n_samples": len(rows),
        "class_counts": {
            "negative": int(sum(target == 0 for target in targets)),
            "positive": int(sum(target == 1 for target in targets)),
        },
        "cat_prediction_file": args.cat_pred,
        "alignn_prediction_files": alignn_paths,
        "feature_names": feature_names,
        "feature_mode": args.feature_mode,
        "meta_model": "StandardScaler + LogisticRegression(class_weight=balanced)",
        "stacking_cv": {
            "n_splits": args.n_splits,
            "seed": args.seed,
        },
        "constraints": {
            "min_positive_recall": args.min_positive_recall,
            "max_fn": args.max_fn,
        },
        "probability_metrics": {
            "cat_gate_a": {
                "roc_auc": float(roc_auc_score(targets, cat_gate_a_prob)),
                "average_precision": float(
                    average_precision_score(targets, cat_gate_a_prob)
                ),
            },
            first_alignn_name: {
                "roc_auc": float(roc_auc_score(targets, first_alignn_prob)),
                "average_precision": float(
                    average_precision_score(targets, first_alignn_prob)
                ),
            },
            "fusion_logistic": {
                "roc_auc": float(roc_auc_score(targets, fusion_oof)),
                "average_precision": float(
                    average_precision_score(targets, fusion_oof)
                ),
            },
        },
        "operating_points": baselines,
        "best_weighted_average": {
            "recall_constraint": best_weight_recall,
            "fn_constraint": best_weight_fn,
        },
        "outputs": {
            "fusion_oof_predictions": str(
                output_dir / "fusion_oof_predictions.csv"
            ),
            "fusion_threshold_scan": str(
                output_dir / "fusion_threshold_scan.csv"
            ),
            "weighted_average_search": str(
                output_dir / "weighted_average_search.csv"
            ),
            "fusion_logistic_model": str(
                output_dir / "fusion_logistic_model.pkl"
            ),
        },
    }
    (output_dir / "fusion_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    report_rows = []
    for name, metrics in baselines.items():
        row = {"name": name}
        row.update(metrics)
        report_rows.append(row)
    write_rows(output_dir / "fusion_operating_points.csv", report_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
