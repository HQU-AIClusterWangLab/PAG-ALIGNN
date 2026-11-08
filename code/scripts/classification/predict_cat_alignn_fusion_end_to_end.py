#!/usr/bin/env python
"""End-to-end prediction for the CatBoost Gate A + ALIGNN fusion classifier.

Inputs for new samples:
  1) CatBoost descriptor CSV used by the tree gate.
  2) Structures, either a structures JSON with ``uid`` and ``structure`` or a
     POSCAR directory with files named by uid.
  3) ALIGNN auxiliary descriptor CSV with uid and the 101 physical priors.

The script computes:
    p_cat_gate_a -> p_alignn -> p_final -> class prediction
using the fixed fusion rule stored in the manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from alignn.config import TrainingConfig
from alignn.data import get_train_val_loaders
from alignn.models.alignn import ALIGNN
from alignn.models.alignn_atomwise import ALIGNNAtomWise
from alignn.models.ealignn_atomwise import eALIGNNAtomWise
from alignn.train import (
    _classification_positive_probability,
    _model_input_from_batch,
)
from jarvis.core.atoms import Atoms

from predict_cat_alignn_fusion import (
    predict_catboost_probabilities,
)


DEFAULT_MANIFEST = (
    "runs/cat_alignn_weighted_fusion_classifier/fusion_classifier_manifest.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run CatBoost Gate A, ALIGNN classifier, and weighted fusion in "
            "one command."
        )
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--cat-descriptors", required=True)
    parser.add_argument("--alignn-aux-csv", required=True)
    parser.add_argument("--structures-json", default=None)
    parser.add_argument("--poscar-dir", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--work-dir",
        default="runs/cat_alignn_weighted_fusion_classifier/predict_work",
        help="Temporary working directory for ALIGNN graph construction.",
    )
    parser.add_argument("--uid-column", default="uid")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Keep the temporary ALIGNN prediction work directory.",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Fusion manifest not found: {path}")
    return json.loads(path.read_text())


def load_aux_features(path: Path, feature_names: list[str]) -> dict[str, list[float]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        uid_col = "uid" if "uid" in fieldnames else None
        if uid_col is None:
            for fallback in ["sample_id", "id", "jid"]:
                if fallback in fieldnames:
                    uid_col = fallback
                    break
        if uid_col is None:
            raise ValueError("ALIGNN aux CSV must contain uid/sample_id/id/jid.")
        missing = [name for name in feature_names if name not in fieldnames]
        if missing:
            raise ValueError(
                "ALIGNN aux CSV missing required feature columns: "
                + ", ".join(missing[:20])
            )
        return {
            str(row[uid_col]).strip(): [float(row[name]) for name in feature_names]
            for row in reader
        }


def load_structures(
    structures_json: str | None,
    poscar_dir: str | None,
    uid_column: str,
) -> dict[str, dict]:
    if structures_json is None and poscar_dir is None:
        raise ValueError("Provide --structures-json or --poscar-dir.")
    structures: dict[str, dict] = {}
    if structures_json is not None:
        with Path(structures_json).open() as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            iterable = payload.values()
        else:
            iterable = payload
        for row in iterable:
            uid = str(row.get(uid_column, row.get("uid", row.get("sample_id"))))
            if uid in ("None", ""):
                raise ValueError("Each structure JSON row must include a uid.")
            structure = row.get("structure", row)
            structures[uid] = structure
    if poscar_dir is not None:
        for path in Path(poscar_dir).iterdir():
            if path.is_file():
                structures[path.name] = Atoms.from_poscar(path).to_dict()
    if not structures:
        raise ValueError("No structures loaded.")
    return structures


def load_alignn_config(manifest: dict) -> tuple[TrainingConfig, Path, Path]:
    alignn_cfg = manifest["alignn_classifier"]
    config_path = Path(alignn_cfg["config"])
    model_path = Path(alignn_cfg["best_model"])
    if not config_path.exists():
        raise FileNotFoundError(
            "ALIGNN config not found. Train the full-data ALIGNN model first: "
            + str(config_path)
        )
    if not model_path.exists():
        raise FileNotFoundError(
            "ALIGNN best_model.pt not found. Train the full-data ALIGNN model first: "
            + str(model_path)
        )
    config = TrainingConfig(**json.loads(config_path.read_text()))
    config.model.classification = True
    return config, config_path, model_path


def load_alignn_feature_names(run_dir: Path, aux_csv: Path) -> list[str]:
    feature_name_path = run_dir / "extra_feature_names.json"
    if feature_name_path.exists():
        return json.loads(feature_name_path.read_text())
    with aux_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
    uid_cols = {"uid", "sample_id", "id", "jid"}
    return [name for name in fieldnames if name not in uid_cols]


def build_alignn_dataset(
    structures: dict[str, dict],
    aux_features: dict[str, list[float]],
) -> list[dict]:
    missing_aux = sorted(set(structures) - set(aux_features))
    if missing_aux:
        raise ValueError(
            "Missing ALIGNN aux features for structures: "
            + ", ".join(missing_aux[:20])
        )
    dataset = []
    for uid in sorted(structures):
        dataset.append(
            {
                "jid": uid,
                "target": 0.0,
                "atoms": structures[uid],
                "extra_features": aux_features[uid],
            }
        )
    return dataset


def instantiate_alignn_model(config: TrainingConfig, model_path: Path, device):
    model_cls = {
        "alignn_atomwise": ALIGNNAtomWise,
        "ealignn_atomwise": eALIGNNAtomWise,
        "alignn": ALIGNN,
    }[config.model.name]
    model = model_cls(config.model)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_alignn_probabilities(
    structures: dict[str, dict],
    aux_csv: Path,
    manifest: dict,
    work_dir: Path,
    device_text: str | None,
) -> dict[str, float]:
    config, _, model_path = load_alignn_config(manifest)
    run_dir = Path(manifest["alignn_classifier"]["run_dir"])
    feature_names = load_alignn_feature_names(run_dir, aux_csv)
    aux_features = load_aux_features(aux_csv, feature_names)
    dataset = build_alignn_dataset(structures, aux_features)

    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    split = {
        "id_train": [row["jid"] for row in dataset],
        "id_val": [row["jid"] for row in dataset],
        "id_test": [row["jid"] for row in dataset],
    }
    (work_dir / "ids_train_val_test.json").write_text(
        json.dumps(split, indent=2) + "\n"
    )

    device = (
        torch.device(device_text)
        if device_text is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    line_graph = config.compute_line_graph > 0
    _, _, test_loader, _ = get_train_val_loaders(
        dataset=config.dataset,
        dataset_array=dataset,
        target=config.target,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        batch_size=config.batch_size,
        atom_features=config.atom_features,
        neighbor_strategy=config.neighbor_strategy,
        standardize=config.atom_features != "cgcnn",
        line_graph=line_graph,
        id_tag=config.id_tag,
        pin_memory=config.pin_memory,
        workers=config.num_workers,
        save_dataloader=False,
        use_canonize=config.use_canonize,
        filename=str(work_dir / "predict"),
        cutoff=config.cutoff,
        max_neighbors=config.max_neighbors,
        output_features=config.model.output_features,
        classification_threshold=config.classification_threshold,
        target_multiplication_factor=config.target_multiplication_factor,
        standard_scalar_and_pca=False,
        keep_data_order=True,
        output_dir=str(work_dir),
        use_lmdb=False,
        dtype=config.dtype,
    )
    model = instantiate_alignn_model(config, model_path, device)
    probabilities: dict[str, float] = {}
    ids = list(test_loader.dataset.ids)
    offset = 0
    with torch.no_grad():
        for batch in test_loader:
            out = model(_model_input_from_batch(batch, config, device))["out"]
            prob = _classification_positive_probability(out).detach().cpu()
            target = batch[-1].view(-1)
            batch_ids = ids[offset : offset + target.shape[0]]
            offset += target.shape[0]
            for uid, value in zip(batch_ids, prob.tolist()):
                probabilities[str(uid)] = float(value)
    return probabilities


def write_fusion_output(
    output_path: Path,
    cat_prob: dict[str, float],
    alignn_prob: dict[str, float],
    manifest: dict,
) -> dict:
    formula = manifest["fusion_formula"]
    cat_weight = float(formula["cat_weight"])
    alignn_weight = float(formula["alignn_weight"])
    threshold = float(formula["decision_threshold"])
    common_ids = sorted(set(cat_prob) & set(alignn_prob))
    if not common_ids:
        raise ValueError("No overlapping ids between CatBoost and ALIGNN probabilities.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "p_cat_gate_a",
                "p_alignn",
                "p_final",
                "decision_threshold",
                "prediction",
            ],
        )
        writer.writeheader()
        for uid in common_ids:
            p_final = cat_weight * cat_prob[uid] + alignn_weight * alignn_prob[uid]
            writer.writerow(
                {
                    "sample_id": uid,
                    "p_cat_gate_a": cat_prob[uid],
                    "p_alignn": alignn_prob[uid],
                    "p_final": p_final,
                    "decision_threshold": threshold,
                    "prediction": int(p_final >= threshold),
                }
            )
    return {
        "n_predicted": len(common_ids),
        "missing_cat_probability_count": len(set(alignn_prob) - set(cat_prob)),
        "missing_alignn_probability_count": len(set(cat_prob) - set(alignn_prob)),
        "output": str(output_path),
        "fusion_formula": formula,
    }


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)
    structures = load_structures(
        args.structures_json,
        args.poscar_dir,
        args.uid_column,
    )
    cat_prob = predict_catboost_probabilities(
        Path(args.cat_descriptors),
        manifest,
        args.uid_column,
    )
    alignn_prob = predict_alignn_probabilities(
        structures=structures,
        aux_csv=Path(args.alignn_aux_csv),
        manifest=manifest,
        work_dir=Path(args.work_dir),
        device_text=args.device,
    )
    report = write_fusion_output(
        Path(args.output),
        cat_prob=cat_prob,
        alignn_prob=alignn_prob,
        manifest=manifest,
    )
    report["manifest"] = str(manifest_path)
    report_path = Path(args.output).with_suffix(".metrics.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if not args.keep_work_dir:
        shutil.rmtree(args.work_dir, ignore_errors=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
