#!/usr/bin/env python
"""Create a deployment manifest for the fixed CatBoost + ALIGNN fusion gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_CAT_MODEL = (
    "cat/high_alpha_inplane_gate/models/gate_a_high_alpha_classifier.cbm"
)
DEFAULT_CAT_MANIFEST = "cat/high_alpha_inplane_gate/high_alpha_gate_manifest.json"
DEFAULT_ALIGNN_RUN_DIR = "runs/inplane_avg_gt_16p1_classifier_02_full"
DEFAULT_OUTPUT_DIR = "runs/cat_alignn_weighted_fusion_classifier"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a manifest for the weighted fusion classifier."
    )
    parser.add_argument("--cat-model", default=DEFAULT_CAT_MODEL)
    parser.add_argument("--cat-manifest", default=DEFAULT_CAT_MANIFEST)
    parser.add_argument("--alignn-run-dir", default=DEFAULT_ALIGNN_RUN_DIR)
    parser.add_argument("--cat-weight", type=float, default=0.96)
    parser.add_argument("--alignn-weight", type=float, default=0.04)
    parser.add_argument("--decision-threshold", type=float, default=0.06761944)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--allow-missing-alignn",
        action="store_true",
        help=(
            "Allow creating the manifest before the full-data ALIGNN run "
            "exists. Useful when preparing the package before training."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cat_weight < 0 or args.alignn_weight < 0:
        raise ValueError("Fusion weights must be non-negative.")
    total_weight = args.cat_weight + args.alignn_weight
    if total_weight <= 0:
        raise ValueError("At least one fusion weight must be positive.")
    cat_weight = args.cat_weight / total_weight
    alignn_weight = args.alignn_weight / total_weight
    if not 0 <= args.decision_threshold <= 1:
        raise ValueError("--decision-threshold must be between 0 and 1.")

    cat_model = Path(args.cat_model).resolve()
    cat_manifest = Path(args.cat_manifest).resolve()
    alignn_run_dir = Path(args.alignn_run_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not cat_model.exists():
        raise FileNotFoundError(f"CatBoost Gate A model not found: {cat_model}")
    if not cat_manifest.exists():
        raise FileNotFoundError(f"CatBoost manifest not found: {cat_manifest}")
    alignn_files = {
        "run_dir": str(alignn_run_dir),
        "config": str(alignn_run_dir / "config.json"),
        "best_model": str(alignn_run_dir / "best_model.pt"),
    }
    missing_alignn = [
        path for path in alignn_files.values() if not Path(path).exists()
    ]
    if missing_alignn and not args.allow_missing_alignn:
        raise FileNotFoundError(
            "ALIGNN full-data run is incomplete or missing: "
            + ", ".join(missing_alignn)
        )

    cat_payload = json.loads(cat_manifest.read_text())
    manifest = {
        "model_type": "weighted_catboost_gate_a_alignn_classifier",
        "label_definition": {
            "positive_class": "alpha_inplane = (xx + yy) / 2 > 16.1",
            "negative_class": "alpha_inplane <= 16.1",
        },
        "fusion_formula": {
            "p_final": (
                f"{cat_weight:.12g} * p_cat_gate_a + "
                f"{alignn_weight:.12g} * p_alignn"
            ),
            "cat_weight": cat_weight,
            "alignn_weight": alignn_weight,
            "decision_threshold": float(args.decision_threshold),
            "positive_if": f"p_final >= {args.decision_threshold:.12g}",
        },
        "selected_from_oof": {
            "source": "runs/cat_alignn_post_fusion_02/weighted_average_search.csv",
            "constraint": "FN <= 17",
            "tn": 1587,
            "fp": 215,
            "fn": 17,
            "tp": 179,
            "positive_recall": 0.9132653061224489,
            "negative_retention": 0.8806881243063263,
            "precision": 0.4543147208121827,
            "f1": 0.6067796610169491,
        },
        "catboost_gate_a": {
            "model_file": str(cat_model),
            "manifest_file": str(cat_manifest),
            "feature_columns": cat_payload.get("feature_columns", []),
            "categorical_features": cat_payload.get(
                "categorical_features", []
            ),
            "probability_name": "p_cat_gate_a",
            "probability_expression": "gate_a_model.predict_proba(X)[:, 1]",
        },
        "alignn_classifier": {
            **alignn_files,
            "probability_name": "p_alignn",
            "recommended_training_command": (
                "C:\\Users\\admin\\.conda\\envs\\my_alignn\\python.exe "
                "scripts\\classification\\train_inplane_avg_classifier.py --full-train "
                "--data-dir data\\inplane_avg_gt_16p1_classifier_02_full "
                "--run-dir runs\\inplane_avg_gt_16p1_classifier_02_full "
                "--min-positive-recall 0.90"
            ),
            "note": (
                "Train this final ALIGNN model on all usable samples before "
                "using it for production fusion. Its own in-sample metrics "
                "are not an unbiased validation estimate."
            ),
        },
        "prediction": {
            "script": "scripts/classification/predict_cat_alignn_fusion_end_to_end.py",
            "probability_fusion_script": "scripts/classification/predict_cat_alignn_fusion.py",
            "required_inputs": [
                "CatBoost descriptor CSV",
                "Structures JSON or POSCAR directory",
                "ALIGNN auxiliary descriptor CSV",
            ],
            "output_columns": [
                "sample_id",
                "p_cat_gate_a",
                "p_alignn",
                "p_final",
                "prediction",
            ],
            "example_command": (
                "C:\\Users\\admin\\.conda\\envs\\my_alignn\\python.exe "
                "scripts\\classification\\predict_cat_alignn_fusion_end_to_end.py "
                "--manifest runs\\cat_alignn_weighted_fusion_classifier\\fusion_classifier_manifest.json "
                "--cat-descriptors path\\to\\tree_descriptors.csv "
                "--structures-json path\\to\\structures.json "
                "--alignn-aux-csv path\\to\\aux_physical_priors.csv "
                "--output path\\to\\fusion_predictions.csv"
            ),
        },
    }
    manifest_path = output_dir / "fusion_classifier_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote fusion classifier manifest: {manifest_path}")
    if missing_alignn:
        print("ALIGNN files not found yet:")
        for path in missing_alignn:
            print(f"  {path}")


if __name__ == "__main__":
    main()
