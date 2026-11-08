#!/usr/bin/env python
"""Run the routed polarizability prediction workflow.

The workflow is:

1. Predict the high-amplitude in-plane gate with the bundled CatBoost + ALIGNN
   weighted fusion classifier.
2. Predict the eta multi-task head for every sample with the eta CV ensemble.
3. Predict in-plane xx/yy/xy for every sample with the in-plane CV ensemble.
4. Predict out-plane zz and xz/yz for every sample with the out-plane ensembles.
5. Write compact routed outputs: tensor-route samples get a complete tensor,
   while eta-route samples get eta_parallel plus the out-plane components.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CLASSIFICATION_DIR = ROOT / "scripts" / "classification"
if str(CLASSIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSIFICATION_DIR))

from alignn.config import TrainingConfig
from alignn.data import get_train_val_loaders
from alignn.models.alignn import ALIGNN
from alignn.models.alignn_atomwise import ALIGNNAtomWise
from alignn.models.ealignn_atomwise import eALIGNNAtomWise
from alignn.train import _model_input_from_batch
from jarvis.core.atoms import Atoms
from predict_cat_alignn_fusion import predict_catboost_probabilities
from predict_cat_alignn_fusion_end_to_end import (
    predict_alignn_probabilities,
)


DEFAULT_BUNDLE_DIR = "models"
DEFAULT_OUTPUT = "outputs/routed_polarizability_predictions/predictions.csv"
EPS = 1e-12
ZZ_LOG1P_SCALE = 0.033699
ATOMIC_SYMBOLS = [
    "",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]


MODEL_GROUPS = {
    "eta": {
        "relative_fold_dir": "eta/folds",
        "components": ["eta_parallel", "a_diag", "a_xy"],
    },
    "inplane": {
        "relative_fold_dir": "inplane/folds",
        "components": ["xx", "yy", "xy"],
    },
    "outplane_zz": {
        "relative_fold_dir": "outplane_zz/folds",
        "components": ["zz"],
    },
    "outplane_aux": {
        "relative_fold_dir": "outplane_aux/folds",
        "components": ["xz", "yz"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict polarizability with the high-alpha routed ensemble."
    )
    parser.add_argument(
        "--model-profile",
        default=None,
        help=(
            "Optional JSON model profile. It can select the bundle, eta model, "
            "and in-plane model together. Explicit model path arguments take "
            "priority over profile values."
        ),
    )
    parser.add_argument(
        "--bundle-dir",
        default=None,
        help=(
            "Model bundle directory. Defaults to the bundle configured by "
            "--model-profile, or models."
        ),
    )
    parser.add_argument("--cat-descriptors", required=True)
    parser.add_argument("--alignn-aux-csv", required=True)
    parser.add_argument("--structures-json", default=None)
    parser.add_argument("--poscar-dir", default=None)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--work-dir",
        default="outputs/routed_polarizability_predictions/work",
        help="Temporary graph construction directory.",
    )
    parser.add_argument("--uid-column", default="uid")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Keep temporary graph construction files.",
    )
    parser.add_argument(
        "--no-clip-eta",
        action="store_true",
        help="Do not clip eta_parallel predictions to [0, 1].",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional debug limit; by default all input structures are predicted.",
    )
    parser.add_argument(
        "--write-all-model-outputs",
        action="store_true",
        help=(
            "Include raw eta/in-plane/out-plane ensemble outputs for every "
            "sample in the main CSV."
        ),
    )
    parser.add_argument(
        "--eta-branch-full-tensor-output",
        default=None,
        help=(
            "Optional CSV path for eta-routed samples with complete tensor "
            "predictions from the tensor submodels. If omitted and "
            "--write-all-model-outputs is set, writes next to --output using "
            "the suffix _eta_branch_full_tensor.csv."
        ),
    )
    parser.add_argument(
        "--inplane-fold-dir",
        default=None,
        help=(
            "Optional in-plane model directory override. Accepts either a "
            "model root containing folds/ or the folds directory itself. "
            "If omitted, uses bundle_dir/inplane/folds."
        ),
    )
    parser.add_argument(
        "--inplane-model-label",
        default=None,
        help="Optional label recorded in the summary for the selected in-plane model.",
    )
    parser.add_argument(
        "--eta-fold-dir",
        default=None,
        help=(
            "Optional eta model directory override. Accepts either a model "
            "root containing folds/ or a directory containing fold_01..fold_05. "
            "If omitted, uses bundle_dir/eta/folds."
        ),
    )
    parser.add_argument(
        "--eta-model-label",
        default=None,
        help="Optional label recorded in the summary for the selected eta model.",
    )
    return parser.parse_args()


def resolve_existing(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def load_model_profile(
    profile_arg: str | None,
) -> tuple[dict[str, Any], Path | None]:
    """Load an optional model-profile JSON and resolve its relative paths later."""
    if profile_arg is None:
        return {}, None
    profile_path = resolve_existing(Path(profile_arg), "Model profile")
    profile = load_json(profile_path)
    if not isinstance(profile, dict):
        raise ValueError(f"Model profile must be a JSON object: {profile_path}")
    return profile, profile_path


def profile_path_value(
    profile: dict[str, Any], profile_path: Path | None, key: str
) -> Path | None:
    """Return a profile path, resolving relative entries from its JSON file."""
    value = profile.get(key)
    if value is None:
        return None
    path = Path(str(value))
    if not path.is_absolute() and profile_path is not None:
        path = profile_path.parent / path
    return path


def resolve_group_fold_root(
    group_name: str,
    bundle_dir: Path,
    fold_dir_override: Path | None = None,
) -> Path:
    group = MODEL_GROUPS[group_name]
    if fold_dir_override is None:
        path = bundle_dir / group["relative_fold_dir"]
        label = f"{group_name} fold directory"
    else:
        path = fold_dir_override
        label = f"{group_name} override fold directory"
    path = resolve_existing(path, label)
    if (path / "folds").is_dir():
        path = resolve_existing(path / "folds", f"{label}/folds")
    return path


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_aux_features(path: Path, feature_names: list[str]) -> dict[str, list[float]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        uid_col = next(
            (name for name in ["uid", "sample_id", "id", "jid"] if name in fieldnames),
            None,
        )
        if uid_col is None:
            raise ValueError("Aux CSV must contain uid/sample_id/id/jid.")
        missing = [name for name in feature_names if name not in fieldnames]
        if missing:
            raise ValueError(
                "Aux CSV missing required ALIGNN feature columns: "
                + ", ".join(missing[:20])
            )
        return {
            str(row[uid_col]).strip(): [float(row[name]) for name in feature_names]
            for row in reader
        }


def _site_element(site: dict[str, Any]) -> str:
    species = site.get("species")
    if isinstance(species, list) and species:
        first = species[0]
        if isinstance(first, dict) and "element" in first:
            return str(first["element"])
    if isinstance(species, dict) and species:
        return str(next(iter(species)))
    if "label" in site:
        return str(site["label"])
    raise ValueError(f"Cannot infer element from site: {site}")


def structure_to_atoms_dict(structure: dict[str, Any]) -> dict[str, Any]:
    if len(structure) == 1:
        only_value = next(iter(structure.values()))
        if isinstance(only_value, dict):
            structure = only_value
    if {"lattice_mat", "coords", "elements"}.issubset(structure):
        return structure
    if {"cell", "positions", "numbers"}.issubset(structure):
        elements = []
        for number in structure["numbers"]:
            z = int(number)
            if z <= 0 or z >= len(ATOMIC_SYMBOLS):
                raise ValueError(f"Unsupported atomic number: {number}")
            elements.append(ATOMIC_SYMBOLS[z])
        return Atoms(
            lattice_mat=structure["cell"],
            coords=structure["positions"],
            elements=elements,
            cartesian=True,
        ).to_dict()
    if "lattice" in structure and "sites" in structure:
        lattice = structure["lattice"]["matrix"]
        elements = []
        coords = []
        for site in structure["sites"]:
            elements.append(_site_element(site))
            if "xyz" in site:
                coords.append(site["xyz"])
            elif "abc" in site:
                frac = np.asarray(site["abc"], dtype=float)
                coords.append((frac @ np.asarray(lattice, dtype=float)).tolist())
            else:
                raise ValueError(f"Cannot infer coordinates from site: {site}")
        return Atoms(
            lattice_mat=lattice,
            coords=coords,
            elements=elements,
            cartesian=True,
        ).to_dict()
    raise ValueError("Unsupported structure dictionary format.")


def load_prediction_structures(
    structures_json: str | None,
    poscar_dir: str | None,
    uid_column: str,
) -> dict[str, dict[str, Any]]:
    structures: dict[str, dict[str, Any]] = {}
    if structures_json is not None:
        payload = load_json(Path(structures_json).resolve())
        iterable = payload.values() if isinstance(payload, dict) else payload
        for row in iterable:
            uid = str(row.get(uid_column, row.get("uid", row.get("sample_id", ""))))
            if not uid:
                raise ValueError("Each structure JSON row must include a uid.")
            structure = row.get("structure", row)
            structures[uid] = structure_to_atoms_dict(structure)
    if poscar_dir is not None:
        for path in Path(poscar_dir).resolve().iterdir():
            if not path.is_file():
                continue
            try:
                structures[path.name] = Atoms.from_poscar(path).to_dict()
            except Exception:
                continue
    if not structures:
        raise ValueError("No structures loaded from --structures-json/--poscar-dir.")
    return structures


def load_fold_feature_names(fold_dir: Path, aux_csv: Path) -> list[str]:
    feature_path = fold_dir / "extra_feature_names.json"
    if feature_path.exists():
        return [str(item) for item in load_json(feature_path)]
    with aux_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
    uid_cols = {"uid", "sample_id", "id", "jid"}
    return [name for name in fieldnames if name not in uid_cols]


def model_class(name: str):
    mapping = {
        "alignn_atomwise": ALIGNNAtomWise,
        "ealignn_atomwise": eALIGNNAtomWise,
        "alignn": ALIGNN,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported model name: {name}")
    return mapping[name]


def instantiate_model(config: TrainingConfig, model_path: Path, device: torch.device):
    model = model_class(config.model.name)(config.model)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def target_spec(config_dict: dict[str, Any]) -> list[tuple[str, str]]:
    target_columns = [str(item) for item in config_dict.get("target_columns") or []]
    if not target_columns:
        width = int(config_dict.get("model", {}).get("output_features", 1))
        target_columns = [f"output_{idx}" for idx in range(width)]
    transforms = config_dict.get("target_transforms") or []
    transform_by_column = {
        str(item["column"]): str(item.get("transform", "identity"))
        for item in transforms
    }
    return [
        (column, transform_by_column.get(column, "identity"))
        for column in target_columns
    ]


def prediction_columns(config_dict: dict[str, Any]) -> list[str]:
    columns = config_dict.get("prediction_columns") or config_dict.get("target_columns")
    if not columns:
        width = int(config_dict.get("model", {}).get("output_features", 1))
        return [f"output_{idx}" for idx in range(width)]
    return [str(item) for item in columns]


def inverse_transform(value: float, transform: str) -> float:
    if transform == "identity":
        return value
    if transform == "log1p":
        return math.expm1(value)
    if transform == "scaled_log1p_zz":
        return ZZ_LOG1P_SCALE * math.expm1(value)
    if transform == "log1p_floor":
        return math.expm1(value)
    if transform == "asinh":
        return math.sinh(value)
    if transform == "inplane_xy_ratio":
        return value
    raise ValueError(f"Unknown target transform: {transform}")


def decode_output(vector: np.ndarray, spec: list[tuple[str, str]]) -> dict[str, float]:
    decoded: dict[str, float] = {}
    deferred: list[tuple[str, float]] = []
    for idx, (column, transform) in enumerate(spec):
        value = float(vector[idx])
        if transform == "inplane_xy_ratio":
            deferred.append((column, inverse_transform(value, transform)))
        else:
            decoded[column] = inverse_transform(value, transform)
    for column, ratio in deferred:
        if column == "xy":
            decoded[column] = 0.5 * ratio * (decoded["xx"] + decoded["yy"])
        else:
            decoded[column] = ratio
    return decoded


def build_regression_dataset(
    sample_ids: list[str],
    structures: dict[str, dict],
    aux_features: dict[str, list[float]],
    width: int,
) -> list[dict[str, Any]]:
    missing_structures = [uid for uid in sample_ids if uid not in structures]
    missing_aux = [uid for uid in sample_ids if uid not in aux_features]
    if missing_structures:
        raise ValueError(
            "Missing structures for ids: " + ", ".join(missing_structures[:20])
        )
    if missing_aux:
        raise ValueError(
            "Missing ALIGNN aux features for ids: " + ", ".join(missing_aux[:20])
        )
    return [
        {
            "jid": uid,
            "target": [0.0] * width,
            "atoms": structures[uid],
            "extra_features": aux_features[uid],
        }
        for uid in sample_ids
    ]


def write_predict_split(work_dir: Path, sample_ids: list[str]) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    first_id = sample_ids[0]
    split = {
        "id_train": [first_id],
        "id_val": [first_id],
        "id_test": sample_ids,
    }
    write_json(work_dir / "ids_train_val_test.json", split)


def predict_regression_fold(
    fold_dir: Path,
    structures: dict[str, dict],
    sample_ids: list[str],
    aux_csv: Path,
    work_dir: Path,
    device: torch.device,
    clip_eta: bool,
) -> dict[str, dict[str, float]]:
    config_path = resolve_existing(fold_dir / "config.json", "Fold config")
    model_path = resolve_existing(fold_dir / "best_model.pt", "Fold model")
    config_dict = load_json(config_path)
    spec = target_spec(config_dict)
    columns = prediction_columns(config_dict)
    unknown = sorted(set(columns) - {column for column, _ in spec})
    if unknown:
        raise ValueError(f"{fold_dir} prediction_columns not in target spec: {unknown}")

    config_dict["output_dir"] = str(work_dir)
    config_dict["use_lmdb"] = False
    config_dict["num_workers"] = 0
    config = TrainingConfig(**config_dict)

    feature_names = load_fold_feature_names(fold_dir, aux_csv)
    aux_features = load_aux_features(aux_csv, feature_names)
    dataset = build_regression_dataset(
        sample_ids=sample_ids,
        structures=structures,
        aux_features=aux_features,
        width=len(spec),
    )
    if work_dir.exists():
        shutil.rmtree(work_dir)
    write_predict_split(work_dir, sample_ids)

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
        line_graph=config.compute_line_graph > 0,
        id_tag=config.id_tag,
        pin_memory=config.pin_memory,
        workers=config.num_workers,
        save_dataloader=False,
        use_canonize=config.use_canonize,
        filename=str(work_dir / "predict"),
        cutoff=config.cutoff,
        cutoff_extra=config.cutoff_extra,
        max_neighbors=config.max_neighbors,
        output_features=config.model.output_features,
        classification_threshold=config.classification_threshold,
        target_multiplication_factor=config.target_multiplication_factor,
        standard_scalar_and_pca=False,
        keep_data_order=True,
        output_dir=str(work_dir),
        use_lmdb=False,
        dtype=config.dtype,
        use_site_symmetry_node_features=getattr(
            config.model, "use_site_symmetry_node_features", False
        ),
        site_symmetry_symprec=getattr(config.model, "site_symmetry_symprec", 1e-3),
        site_symmetry_angle_tolerance=getattr(
            config.model, "site_symmetry_angle_tolerance", 5.0
        ),
        site_symmetry_node_features_dim=getattr(
            config.model, "site_symmetry_node_features", 80
        ),
    )
    model = instantiate_model(config, model_path, device)
    ids = list(test_loader.dataset.ids)
    offset = 0
    predictions: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for batch in test_loader:
            output = model(_model_input_from_batch(batch, config, device))["out"]
            output_arr = np.asarray(output.detach().cpu(), dtype=float)
            if output_arr.ndim == 1:
                output_arr = output_arr.reshape(1, -1)
            target = batch[-1]
            batch_size = int(target.shape[0]) if target.ndim > 0 else 1
            batch_ids = ids[offset : offset + batch_size]
            offset += batch_size
            for uid, vector in zip(batch_ids, output_arr):
                decoded = decode_output(vector, spec)
                row = {column: float(decoded[column]) for column in columns}
                if clip_eta and "eta_parallel" in row:
                    row["eta_parallel"] = float(np.clip(row["eta_parallel"], 0.0, 1.0))
                predictions[str(uid)] = row
    return predictions


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr))


def predict_ensemble_group(
    group_name: str,
    bundle_dir: Path,
    structures: dict[str, dict],
    sample_ids: list[str],
    aux_csv: Path,
    work_dir: Path,
    device: torch.device,
    clip_eta: bool,
    fold_dir_override: Path | None = None,
) -> dict[str, dict[str, Any]]:
    group = MODEL_GROUPS[group_name]
    fold_root = resolve_group_fold_root(group_name, bundle_dir, fold_dir_override)
    # Training runs may also include folders such as fold_evaluations. Only a
    # directory with a deployed checkpoint is a prediction ensemble member.
    fold_dirs = sorted(
        path
        for path in fold_root.iterdir()
        if path.is_dir()
        and path.name.startswith("fold_")
        and (path / "best_model.pt").is_file()
    )
    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories found in {fold_root}")

    by_uid: dict[str, dict[str, list[float]]] = {
        uid: {component: [] for component in group["components"]}
        for uid in sample_ids
    }
    for fold_dir in fold_dirs:
        fold_work_dir = work_dir / group_name / fold_dir.name
        fold_predictions = predict_regression_fold(
            fold_dir=fold_dir,
            structures=structures,
            sample_ids=sample_ids,
            aux_csv=aux_csv,
            work_dir=fold_work_dir,
            device=device,
            clip_eta=clip_eta,
        )
        missing = [uid for uid in sample_ids if uid not in fold_predictions]
        if missing:
            raise ValueError(
                f"{group_name} {fold_dir.name} missing predictions for: "
                + ", ".join(missing[:20])
            )
        for uid in sample_ids:
            for component in group["components"]:
                by_uid[uid][component].append(float(fold_predictions[uid][component]))

    aggregated: dict[str, dict[str, Any]] = {}
    for uid, component_values in by_uid.items():
        row: dict[str, Any] = {"_values": component_values, "_n_folds": len(fold_dirs)}
        for component, values in component_values.items():
            mean, std = mean_std(values)
            row[component] = mean
            row[f"{component}_std"] = std
        aggregated[uid] = row
    return aggregated


def inplane_anisotropy(xx: float, yy: float, xy: float) -> dict[str, float]:
    denom = xx + yy + EPS
    a_diag = abs(xx - yy) / denom
    a_xy = 2.0 * abs(xy) / denom
    eta = math.sqrt((xx - yy) ** 2 + 4.0 * xy**2) / denom
    return {
        "eta_parallel": eta,
        "a_diag": a_diag,
        "a_xy": a_xy,
    }


def aggregate_inplane_anisotropy(inplane_row: dict[str, Any]) -> dict[str, float]:
    values = inplane_row["_values"]
    fold_eta = []
    fold_a_diag = []
    fold_a_xy = []
    for xx, yy, xy in zip(values["xx"], values["yy"], values["xy"]):
        metrics = inplane_anisotropy(float(xx), float(yy), float(xy))
        fold_eta.append(metrics["eta_parallel"])
        fold_a_diag.append(metrics["a_diag"])
        fold_a_xy.append(metrics["a_xy"])
    eta_mean, eta_std = mean_std(fold_eta)
    a_diag_mean, a_diag_std = mean_std(fold_a_diag)
    a_xy_mean, a_xy_std = mean_std(fold_a_xy)
    return {
        "eta_parallel": eta_mean,
        "eta_parallel_std": eta_std,
        "a_diag": a_diag_mean,
        "a_diag_std": a_diag_std,
        "a_xy": a_xy_mean,
        "a_xy_std": a_xy_std,
    }


def gate_predictions(
    bundle_dir: Path,
    structures: dict[str, dict],
    sample_ids: list[str],
    cat_descriptors: Path,
    aux_csv: Path,
    work_dir: Path,
    uid_column: str,
    device_text: str | None,
) -> dict[str, dict[str, float | int]]:
    manifest_path = resolve_existing(
        bundle_dir / "classification" / "weighted_fusion" / "fusion_classifier_manifest.json",
        "Fusion manifest",
    )
    manifest = load_json(manifest_path)
    formula = manifest["fusion_formula"]
    cat_weight = float(formula["cat_weight"])
    alignn_weight = float(formula["alignn_weight"])
    threshold = float(formula["decision_threshold"])

    cat_prob = predict_catboost_probabilities(cat_descriptors, manifest, uid_column)
    alignn_prob = predict_alignn_probabilities(
        structures=structures,
        aux_csv=aux_csv,
        manifest=manifest,
        work_dir=work_dir / "classification_alignn_gate",
        device_text=device_text,
    )

    missing_cat = [uid for uid in sample_ids if uid not in cat_prob]
    missing_alignn = [uid for uid in sample_ids if uid not in alignn_prob]
    if missing_cat:
        raise ValueError(
            "CatBoost gate missing probabilities for: " + ", ".join(missing_cat[:20])
        )
    if missing_alignn:
        raise ValueError(
            "ALIGNN gate missing probabilities for: " + ", ".join(missing_alignn[:20])
        )

    rows: dict[str, dict[str, float | int]] = {}
    for uid in sample_ids:
        p_cat = float(cat_prob[uid])
        p_alignn = float(alignn_prob[uid])
        p_final = cat_weight * p_cat + alignn_weight * p_alignn
        rows[uid] = {
            "gate_p_cat": p_cat,
            "gate_p_alignn": p_alignn,
            "gate_p_final": p_final,
            "gate_threshold": threshold,
            "gate_high_alpha": int(p_final >= threshold),
        }
    return rows


def flatten_model_outputs(
    prefix: str,
    row: dict[str, Any],
    components: list[str],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for component in components:
        out[f"{prefix}_{component}"] = float(row[component])
        out[f"{prefix}_{component}_std"] = float(row[f"{component}_std"])
    return out


def build_output_rows(
    sample_ids: list[str],
    gate: dict[str, dict[str, float | int]],
    eta: dict[str, dict[str, Any]],
    inplane: dict[str, dict[str, Any]],
    outplane_zz: dict[str, dict[str, Any]],
    outplane_aux: dict[str, dict[str, Any]],
    include_model_outputs: bool = False,
) -> list[dict[str, Any]]:
    rows = []
    for uid in sample_ids:
        high_alpha = int(gate[uid]["gate_high_alpha"])
        route = "eta_model" if high_alpha else "tensor_model"
        inplane_eta = aggregate_inplane_anisotropy(inplane[uid])

        row: dict[str, Any] = {
            "sample_id": uid,
            **gate[uid],
            "prediction_branch": route,
            "final_xx": None,
            "final_yy": None,
            "final_zz": outplane_zz[uid]["zz"],
            "final_xy": None,
            "final_xz": outplane_aux[uid]["xz"],
            "final_yz": outplane_aux[uid]["yz"],
            "final_eta_parallel": None,
        }
        if high_alpha:
            row["final_eta_parallel"] = eta[uid]["eta_parallel"]
        else:
            row["final_xx"] = inplane[uid]["xx"]
            row["final_yy"] = inplane[uid]["yy"]
            row["final_xy"] = inplane[uid]["xy"]

        if include_model_outputs:
            row.update(
                {
                    "eta_model_eta_parallel": eta[uid]["eta_parallel"],
                    "eta_model_eta_parallel_std": eta[uid]["eta_parallel_std"],
                    "eta_model_a_diag": eta[uid].get("a_diag"),
                    "eta_model_a_diag_std": eta[uid].get("a_diag_std"),
                    "eta_model_a_xy": eta[uid].get("a_xy"),
                    "eta_model_a_xy_std": eta[uid].get("a_xy_std"),
                    "inplane_model_xx": inplane[uid]["xx"],
                    "inplane_model_xx_std": inplane[uid]["xx_std"],
                    "inplane_model_yy": inplane[uid]["yy"],
                    "inplane_model_yy_std": inplane[uid]["yy_std"],
                    "inplane_model_xy": inplane[uid]["xy"],
                    "inplane_model_xy_std": inplane[uid]["xy_std"],
                    "inplane_model_eta_parallel_reconstructed": inplane_eta[
                        "eta_parallel"
                    ],
                    "inplane_model_eta_parallel_reconstructed_std": inplane_eta[
                        "eta_parallel_std"
                    ],
                    "inplane_model_a_diag_reconstructed": inplane_eta["a_diag"],
                    "inplane_model_a_diag_reconstructed_std": inplane_eta[
                        "a_diag_std"
                    ],
                    "inplane_model_a_xy_reconstructed": inplane_eta["a_xy"],
                    "inplane_model_a_xy_reconstructed_std": inplane_eta["a_xy_std"],
                    "outplane_zz_model_zz": outplane_zz[uid]["zz"],
                    "outplane_zz_model_zz_std": outplane_zz[uid]["zz_std"],
                    "outplane_aux_model_xz": outplane_aux[uid]["xz"],
                    "outplane_aux_model_xz_std": outplane_aux[uid]["xz_std"],
                    "outplane_aux_model_yz": outplane_aux[uid]["yz"],
                    "outplane_aux_model_yz_std": outplane_aux[uid]["yz_std"],
                }
            )
        rows.append(row)
    return rows


def build_eta_branch_full_tensor_rows(
    sample_ids: list[str],
    gate: dict[str, dict[str, float | int]],
    eta: dict[str, dict[str, Any]],
    inplane: dict[str, dict[str, Any]],
    outplane_zz: dict[str, dict[str, Any]],
    outplane_aux: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for uid in sample_ids:
        if int(gate[uid]["gate_high_alpha"]) != 1:
            continue
        inplane_eta = aggregate_inplane_anisotropy(inplane[uid])
        rows.append(
            {
                "sample_id": uid,
                **gate[uid],
                "prediction_branch": "eta_model",
                "tensor_xx": inplane[uid]["xx"],
                "tensor_xx_std": inplane[uid]["xx_std"],
                "tensor_yy": inplane[uid]["yy"],
                "tensor_yy_std": inplane[uid]["yy_std"],
                "tensor_zz": outplane_zz[uid]["zz"],
                "tensor_zz_std": outplane_zz[uid]["zz_std"],
                "tensor_xy": inplane[uid]["xy"],
                "tensor_xy_std": inplane[uid]["xy_std"],
                "tensor_xz": outplane_aux[uid]["xz"],
                "tensor_xz_std": outplane_aux[uid]["xz_std"],
                "tensor_yz": outplane_aux[uid]["yz"],
                "tensor_yz_std": outplane_aux[uid]["yz_std"],
                "tensor_eta_parallel_reconstructed": inplane_eta["eta_parallel"],
                "tensor_eta_parallel_reconstructed_std": inplane_eta[
                    "eta_parallel_std"
                ],
                "eta_model_eta_parallel": eta[uid]["eta_parallel"],
                "eta_model_eta_parallel_std": eta[uid]["eta_parallel_std"],
                "eta_model_a_diag": eta[uid].get("a_diag"),
                "eta_model_a_diag_std": eta[uid].get("a_diag_std"),
                "eta_model_a_xy": eta[uid].get("a_xy"),
                "eta_model_a_xy_std": eta[uid].get("a_xy_std"),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    rows: list[dict[str, Any]],
    output_path: Path,
    bundle_dir: Path,
    model_profile_name: str | None,
    model_profile_path: Path | None,
    eta_fold_dir: Path,
    eta_model_label: str,
    inplane_fold_dir: Path,
    inplane_model_label: str,
) -> dict[str, Any]:
    n = len(rows)
    n_high = sum(int(row["gate_high_alpha"]) for row in rows)
    n_low = n - n_high
    return {
        "output": str(output_path),
        "bundle_dir": str(bundle_dir),
        "model_profile_name": model_profile_name,
        "model_profile_path": str(model_profile_path)
        if model_profile_path is not None
        else None,
        "eta_model_label": eta_model_label,
        "eta_fold_dir": str(eta_fold_dir),
        "inplane_model_label": inplane_model_label,
        "inplane_fold_dir": str(inplane_fold_dir),
        "n_samples": n,
        "n_routed_eta_model": n_high,
        "n_routed_tensor_model": n_low,
        "high_alpha_fraction": n_high / n if n else None,
    }


def main() -> None:
    args = parse_args()
    profile, profile_path = load_model_profile(args.model_profile)
    profile_bundle_dir = profile_path_value(profile, profile_path, "bundle_dir")
    bundle_dir_arg = Path(args.bundle_dir) if args.bundle_dir is not None else profile_bundle_dir
    if bundle_dir_arg is None:
        bundle_dir_arg = Path(DEFAULT_BUNDLE_DIR)
    bundle_dir = resolve_existing(bundle_dir_arg, "Bundle directory")
    cat_descriptors = resolve_existing(Path(args.cat_descriptors), "Cat descriptors")
    aux_csv = resolve_existing(Path(args.alignn_aux_csv), "ALIGNN aux CSV")
    if args.structures_json is None and args.poscar_dir is None:
        raise ValueError("Provide --structures-json or --poscar-dir.")
    structures = load_prediction_structures(
        args.structures_json,
        args.poscar_dir,
        args.uid_column,
    )
    sample_ids = sorted(structures)
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive when provided.")
        sample_ids = sample_ids[: args.max_samples]
        structures = {uid: structures[uid] for uid in sample_ids}
    if not sample_ids:
        raise ValueError("No structures found for prediction.")
    output_path = Path(args.output).resolve()
    work_dir = Path(args.work_dir).resolve()
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    device = (
        torch.device(args.device)
        if args.device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    clip_eta = not args.no_clip_eta
    profile_eta_fold_dir = profile_path_value(profile, profile_path, "eta_fold_dir")
    eta_fold_dir_arg = (
        Path(args.eta_fold_dir)
        if args.eta_fold_dir is not None
        else profile_eta_fold_dir
    )
    eta_fold_dir = resolve_group_fold_root("eta", bundle_dir, eta_fold_dir_arg)
    eta_model_label = args.eta_model_label or profile.get("eta_model_label")
    if eta_model_label is None:
        eta_model_label = (
            "bundle_active_eta_parallel_best"
            if eta_fold_dir_arg is None
            else eta_fold_dir.parent.name
            if eta_fold_dir.name == "folds"
            else eta_fold_dir.name
        )

    profile_inplane_fold_dir = profile_path_value(
        profile, profile_path, "inplane_fold_dir"
    )
    inplane_fold_dir_arg = (
        Path(args.inplane_fold_dir)
        if args.inplane_fold_dir is not None
        else profile_inplane_fold_dir
    )
    inplane_fold_dir = resolve_group_fold_root(
        "inplane",
        bundle_dir,
        inplane_fold_dir_arg,
    )
    inplane_model_label = args.inplane_model_label or profile.get("inplane_model_label")
    if inplane_model_label is None:
        if inplane_fold_dir_arg is None:
            inplane_model_label = "bundle_active_inplane_best_cv5"
        else:
            inplane_model_label = (
                inplane_fold_dir.parent.name
                if inplane_fold_dir.name == "folds"
                else inplane_fold_dir.name
            )

    gate = gate_predictions(
        bundle_dir=bundle_dir,
        structures=structures,
        sample_ids=sample_ids,
        cat_descriptors=cat_descriptors,
        aux_csv=aux_csv,
        work_dir=work_dir,
        uid_column=args.uid_column,
        device_text=args.device,
    )
    eta = predict_ensemble_group(
        "eta",
        bundle_dir,
        structures,
        sample_ids,
        aux_csv,
        work_dir,
        device,
        clip_eta,
        fold_dir_override=eta_fold_dir,
    )
    inplane = predict_ensemble_group(
        "inplane",
        bundle_dir,
        structures,
        sample_ids,
        aux_csv,
        work_dir,
        device,
        clip_eta,
        fold_dir_override=inplane_fold_dir,
    )
    outplane_zz = predict_ensemble_group(
        "outplane_zz",
        bundle_dir,
        structures,
        sample_ids,
        aux_csv,
        work_dir,
        device,
        clip_eta,
    )
    outplane_aux = predict_ensemble_group(
        "outplane_aux",
        bundle_dir,
        structures,
        sample_ids,
        aux_csv,
        work_dir,
        device,
        clip_eta,
    )
    rows = build_output_rows(
        sample_ids=sample_ids,
        gate=gate,
        eta=eta,
        inplane=inplane,
        outplane_zz=outplane_zz,
        outplane_aux=outplane_aux,
        include_model_outputs=args.write_all_model_outputs,
    )
    write_csv(output_path, rows)
    eta_tensor_output = None
    if args.eta_branch_full_tensor_output is not None:
        eta_tensor_output = Path(args.eta_branch_full_tensor_output).resolve()
    elif args.write_all_model_outputs:
        eta_tensor_output = output_path.with_name(
            output_path.stem + "_eta_branch_full_tensor.csv"
        )
    if eta_tensor_output is not None:
        eta_tensor_rows = build_eta_branch_full_tensor_rows(
            sample_ids=sample_ids,
            gate=gate,
            eta=eta,
            inplane=inplane,
            outplane_zz=outplane_zz,
            outplane_aux=outplane_aux,
        )
        if eta_tensor_rows:
            write_csv(eta_tensor_output, eta_tensor_rows)
        else:
            print("No eta-routed samples found; skipped eta branch tensor CSV.")
    summary = summarize(
        rows,
        output_path,
        bundle_dir,
        profile.get("model_name"),
        profile_path,
        eta_fold_dir,
        eta_model_label,
        inplane_fold_dir,
        inplane_model_label,
    )
    if eta_tensor_output is not None:
        summary["eta_branch_full_tensor_output"] = str(eta_tensor_output)
    write_json(output_path.with_suffix(".summary.json"), summary)
    if not args.keep_work_dir:
        shutil.rmtree(work_dir, ignore_errors=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
