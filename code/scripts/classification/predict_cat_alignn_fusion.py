#!/usr/bin/env python
"""Apply the fixed CatBoost Gate A + ALIGNN weighted fusion classifier."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_MANIFEST = (
    "runs/cat_alignn_weighted_fusion_classifier/fusion_classifier_manifest.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse CatBoost Gate A and ALIGNN probabilities."
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--cat-prob",
        default=None,
        help=(
            "CSV with sample ids and CatBoost Gate A probabilities. If this "
            "is omitted, --cat-descriptors is used with the model in manifest."
        ),
    )
    parser.add_argument(
        "--cat-descriptors",
        default=None,
        help="Descriptor CSV for CatBoost Gate A model prediction.",
    )
    parser.add_argument(
        "--alignn-pred",
        required=True,
        help="ALIGNN prediction CSV with id and probability columns.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--uid-column", default="uid")
    parser.add_argument("--cat-prob-column", default="p_gate_a_high_alpha")
    parser.add_argument("--alignn-id-column", default="id")
    parser.add_argument("--alignn-prob-column", default="probability")
    return parser.parse_args()


def read_probability_csv(
    path: Path,
    id_candidates: list[str],
    probability_candidates: list[str],
) -> dict[str, float]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        id_col = next((col for col in id_candidates if col in fieldnames), None)
        prob_col = next(
            (col for col in probability_candidates if col in fieldnames), None
        )
        if id_col is None:
            raise ValueError(
                f"No id column found in {path}. Tried: {id_candidates}"
            )
        if prob_col is None:
            raise ValueError(
                f"No probability column found in {path}. "
                + f"Tried: {probability_candidates}"
            )
        return {str(row[id_col]).strip(): float(row[prob_col]) for row in reader}


def predict_catboost_probabilities(
    descriptor_path: Path,
    manifest: dict,
    uid_column: str,
) -> dict[str, float]:
    try:
        import pandas as pd
        from catboost import CatBoostClassifier, Pool
    except ImportError as exp:
        raise ImportError(
            "Predicting CatBoost probabilities from descriptors requires "
            "pandas and catboost. Use --cat-prob if probabilities are already "
            "available."
        ) from exp

    cat_cfg = manifest["catboost_gate_a"]
    feature_columns = cat_cfg["feature_columns"]
    categorical_features = cat_cfg.get("categorical_features", [])
    data = pd.read_csv(descriptor_path)
    if uid_column not in data.columns:
        for fallback in ["sample_id", "id", "jid"]:
            if fallback in data.columns:
                uid_column = fallback
                break
    if uid_column not in data.columns:
        raise ValueError(f"Descriptor CSV is missing uid column: {uid_column}")
    missing = [col for col in feature_columns if col not in data.columns]
    if missing:
        raise ValueError(
            "Descriptor CSV is missing CatBoost feature columns: "
            + ", ".join(missing[:20])
        )
    model = CatBoostClassifier()
    model.load_model(cat_cfg["model_file"])
    pool = Pool(
        data[feature_columns],
        cat_features=[col for col in categorical_features if col in feature_columns]
        or None,
    )
    probabilities = model.predict_proba(pool)[:, 1]
    return {
        str(uid).strip(): float(prob)
        for uid, prob in zip(data[uid_column].tolist(), probabilities)
    }


def read_optional_targets(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        id_col = next(
            (col for col in ["sample_id", "uid", "id", "jid"] if col in fieldnames),
            None,
        )
        target_col = next(
            (
                col
                for col in [
                    "target",
                    "true_high_alpha",
                    "label",
                    "class",
                ]
                if col in fieldnames
            ),
            None,
        )
        if id_col is None or target_col is None:
            return {}
        return {str(row[id_col]).strip(): int(row[target_col]) for row in reader}


def metrics(rows: list[dict]) -> dict | None:
    labeled = [row for row in rows if row.get("target") not in (None, "")]
    if not labeled:
        return None
    tn = fp = fn = tp = 0
    for row in labeled:
        target = int(row["target"])
        pred = int(row["prediction"])
        if target == 0 and pred == 0:
            tn += 1
        elif target == 0 and pred == 1:
            fp += 1
        elif target == 1 and pred == 0:
            fn += 1
        elif target == 1 and pred == 1:
            tp += 1
    negative = tn + fp
    positive = tp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positive if positive else 0.0
    specificity = tn / negative if negative else 0.0
    return {
        "n_labeled": len(labeled),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "negative_retention": specificity,
    }


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    formula = manifest["fusion_formula"]
    cat_weight = float(formula["cat_weight"])
    alignn_weight = float(formula["alignn_weight"])
    threshold = float(formula["decision_threshold"])

    if args.cat_prob is not None:
        cat_prob = read_probability_csv(
            Path(args.cat_prob),
            id_candidates=["sample_id", "uid", "id", "jid"],
            probability_candidates=[
                args.cat_prob_column,
                "p_cat_gate_a",
                "p_high_alpha",
                "probability",
            ],
        )
    elif args.cat_descriptors is not None:
        cat_prob = predict_catboost_probabilities(
            Path(args.cat_descriptors), manifest, args.uid_column
        )
    else:
        raise ValueError("Provide either --cat-prob or --cat-descriptors.")

    alignn_pred_path = Path(args.alignn_pred)
    alignn_prob = read_probability_csv(
        alignn_pred_path,
        id_candidates=[args.alignn_id_column, "sample_id", "uid", "jid"],
        probability_candidates=[args.alignn_prob_column, "p_alignn"],
    )
    targets = read_optional_targets(alignn_pred_path)

    common_ids = sorted(set(cat_prob) & set(alignn_prob))
    if not common_ids:
        raise ValueError("No overlapping sample ids between CatBoost and ALIGNN.")
    missing_cat = sorted(set(alignn_prob) - set(cat_prob))
    missing_alignn = sorted(set(cat_prob) - set(alignn_prob))
    rows = []
    for uid in common_ids:
        p_cat = cat_prob[uid]
        p_alignn = alignn_prob[uid]
        p_final = cat_weight * p_cat + alignn_weight * p_alignn
        row = {
            "sample_id": uid,
            "p_cat_gate_a": p_cat,
            "p_alignn": p_alignn,
            "p_final": p_final,
            "decision_threshold": threshold,
            "prediction": int(p_final >= threshold),
        }
        if uid in targets:
            row["target"] = targets[uid]
        rows.append(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "p_cat_gate_a",
        "p_alignn",
        "p_final",
        "decision_threshold",
        "prediction",
    ]
    if any("target" in row for row in rows):
        fieldnames.append("target")
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "manifest": str(manifest_path),
        "output": str(output_path),
        "n_predicted": len(rows),
        "missing_cat_probability_count": len(missing_cat),
        "missing_alignn_probability_count": len(missing_alignn),
        "fusion_formula": formula,
        "metrics": metrics(rows),
    }
    report_path = output_path.with_suffix(".metrics.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
