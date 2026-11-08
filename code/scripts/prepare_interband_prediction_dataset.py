#!/usr/bin/env python
"""Prepare interband-only samples as an ALIGNN prediction dataset.

The current eta-parallel ALIGNN configuration expects one structure file per
sample plus a CSV of 101 auxiliary physical descriptors. This script builds
that same shape from folders containing ``data.json`` and ``structure.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple


DEFAULT_SOURCE_DIR = "interband_only"
DEFAULT_TRAIN_DESCRIPTOR_CSV = "aux_physical_priors.csv"
DEFAULT_CONFIG_TEMPLATE = "data/eta_parallel_multitask/config.json"
DEFAULT_OUT_DIR = "data/interband_only_eta_parallel_predict"

TARGET_PLACEHOLDER = ("0.0", "0.0", "0.0")
EPS = 1e-12


# Atomic masses and Pauling electronegativities for the elements present in the
# known 2D-material datasets. Unknown values are reported in summary.json.
ELEMENTS: Dict[str, Tuple[int, float, Optional[float]]] = {
    "H": (1, 1.008, 2.20),
    "He": (2, 4.002602, None),
    "Li": (3, 6.94, 0.98),
    "Be": (4, 9.0121831, 1.57),
    "B": (5, 10.81, 2.04),
    "C": (6, 12.011, 2.55),
    "N": (7, 14.007, 3.04),
    "O": (8, 15.999, 3.44),
    "F": (9, 18.998403163, 3.98),
    "Ne": (10, 20.1797, None),
    "Na": (11, 22.98976928, 0.93),
    "Mg": (12, 24.305, 1.31),
    "Al": (13, 26.9815385, 1.61),
    "Si": (14, 28.085, 1.90),
    "P": (15, 30.973761998, 2.19),
    "S": (16, 32.06, 2.58),
    "Cl": (17, 35.45, 3.16),
    "Ar": (18, 39.948, None),
    "K": (19, 39.0983, 0.82),
    "Ca": (20, 40.078, 1.00),
    "Sc": (21, 44.955908, 1.36),
    "Ti": (22, 47.867, 1.54),
    "V": (23, 50.9415, 1.63),
    "Cr": (24, 51.9961, 1.66),
    "Mn": (25, 54.938044, 1.55),
    "Fe": (26, 55.845, 1.83),
    "Co": (27, 58.933194, 1.88),
    "Ni": (28, 58.6934, 1.91),
    "Cu": (29, 63.546, 1.90),
    "Zn": (30, 65.38, 1.65),
    "Ga": (31, 69.723, 1.81),
    "Ge": (32, 72.63, 2.01),
    "As": (33, 74.921595, 2.18),
    "Se": (34, 78.971, 2.55),
    "Br": (35, 79.904, 2.96),
    "Kr": (36, 83.798, None),
    "Rb": (37, 85.4678, 0.82),
    "Sr": (38, 87.62, 0.95),
    "Y": (39, 88.90584, 1.22),
    "Zr": (40, 91.224, 1.33),
    "Nb": (41, 92.90637, 1.60),
    "Mo": (42, 95.95, 2.16),
    "Tc": (43, 98.0, 1.90),
    "Ru": (44, 101.07, 2.20),
    "Rh": (45, 102.9055, 2.28),
    "Pd": (46, 106.42, 2.20),
    "Ag": (47, 107.8682, 1.93),
    "Cd": (48, 112.414, 1.69),
    "In": (49, 114.818, 1.78),
    "Sn": (50, 118.71, 1.96),
    "Sb": (51, 121.76, 2.05),
    "Te": (52, 127.6, 2.10),
    "I": (53, 126.90447, 2.66),
    "Xe": (54, 131.293, 2.60),
    "Cs": (55, 132.90545196, 0.79),
    "Ba": (56, 137.327, 0.89),
    "La": (57, 138.90547, 1.10),
    "Ce": (58, 140.116, 1.12),
    "Pr": (59, 140.90766, 1.13),
    "Nd": (60, 144.242, 1.14),
    "Pm": (61, 145.0, 1.13),
    "Sm": (62, 150.36, 1.17),
    "Eu": (63, 151.964, None),
    "Gd": (64, 157.25, 1.20),
    "Tb": (65, 158.92535, None),
    "Dy": (66, 162.5, 1.22),
    "Ho": (67, 164.93033, 1.23),
    "Er": (68, 167.259, 1.24),
    "Tm": (69, 168.93422, 1.25),
    "Yb": (70, 173.045, None),
    "Lu": (71, 174.9668, 1.27),
    "Hf": (72, 178.49, 1.30),
    "Ta": (73, 180.94788, 1.50),
    "W": (74, 183.84, 2.36),
    "Re": (75, 186.207, 1.90),
    "Os": (76, 190.23, 2.20),
    "Ir": (77, 192.217, 2.20),
    "Pt": (78, 195.084, 2.28),
    "Au": (79, 196.966569, 2.54),
    "Hg": (80, 200.592, 2.00),
    "Tl": (81, 204.38, 1.62),
    "Pb": (82, 207.2, 2.33),
    "Bi": (83, 208.98040, 2.02),
}
NUMBER_TO_SYMBOL = {meta[0]: symbol for symbol, meta in ELEMENTS.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--train-descriptor-csv", default=DEFAULT_TRAIN_DESCRIPTOR_CSV)
    parser.add_argument("--config-template", default=DEFAULT_CONFIG_TEMPLATE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_descriptor_columns(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    if not header or header[0] != "uid":
        raise ValueError(f"{path} must start with a uid column.")
    return header[1:]


def safe_uid(uid: str) -> str:
    text = str(uid).strip()
    text = re.sub(r"[\\/:*?\"<>|]+", "_", text)
    if not text:
        raise ValueError("Empty uid cannot be used as a filename.")
    return text


def as_bool_number(value: Any) -> float:
    if isinstance(value, str):
        return 1.0 if value.strip().lower() in {"1", "true", "yes", "y"} else 0.0
    return 1.0 if bool(value) else 0.0


def numeric(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "0"
    if not math.isfinite(number):
        return "0"
    return f"{number:.12g}"


def normalize_pointgroup(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace(" ", "")
    text = text.replace("/", "_over_").replace("-", "neg")
    text = re.sub(r"_+", "_", text)
    if text and text[0].isdigit():
        text = f"n_{text}"
    return text


def normalize_layergroup(value: Any) -> str:
    text = str(value).strip().lower()
    text = text.replace(" ", "")
    text = text.replace("/", "_over_").replace("-", "neg")
    text = re.sub(r"_+", "_", text)
    return text


def site_species(site: Mapping[str, Any]) -> Optional[str]:
    species = site.get("species")
    if isinstance(species, list) and species:
        first = species[0]
        if isinstance(first, Mapping):
            element = first.get("element")
            if element:
                return str(element)
    label = site.get("label")
    if label:
        return str(label)
    return None


def unwrap_ase_structure(structure: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    if "cell" in structure and "numbers" in structure and "positions" in structure:
        return structure
    if len(structure) == 1:
        only_value = next(iter(structure.values()))
        if (
            isinstance(only_value, Mapping)
            and "cell" in only_value
            and "numbers" in only_value
            and "positions" in only_value
        ):
            return only_value
    return None


def composition_from_structure(structure: Mapping[str, Any]) -> Counter:
    ase_structure = unwrap_ase_structure(structure)
    if ase_structure is not None:
        counts: Counter = Counter()
        for number in ase_structure.get("numbers", []):
            symbol = NUMBER_TO_SYMBOL.get(int(number), f"Z{int(number)}")
            counts[symbol] += 1.0
        return counts

    counts: Counter = Counter()
    for site in structure.get("sites", []):
        if not isinstance(site, Mapping):
            continue
        element = site_species(site)
        if not element:
            continue
        occupancy = 1.0
        species = site.get("species")
        if isinstance(species, list) and species and isinstance(species[0], Mapping):
            occupancy = numeric(species[0].get("occu"), 1.0) or 1.0
        counts[element] += occupancy
    return counts


def get_coords(site: Mapping[str, Any]) -> List[float]:
    coords = site.get("abc")
    if not isinstance(coords, list):
        coords = site.get("xyz")
    if not isinstance(coords, list) or len(coords) != 3:
        return [0.0, 0.0, 0.0]
    return [float(coords[0]), float(coords[1]), float(coords[2])]


def write_poscar(path: Path, structure: Mapping[str, Any], comment: str) -> None:
    ase_structure = unwrap_ase_structure(structure)
    if ase_structure is not None:
        matrix = ase_structure.get("cell")
        if not isinstance(matrix, list) or len(matrix) != 3:
            raise ValueError(f"Structure for {comment} has no 3x3 cell matrix.")
        numbers = ase_structure.get("numbers", [])
        positions = ase_structure.get("positions", [])
        if len(numbers) != len(positions):
            raise ValueError(f"Structure for {comment} has mismatched numbers/positions.")

        species_order: List[str] = []
        grouped_positions: Dict[str, List[List[float]]] = defaultdict(list)
        for number, position in zip(numbers, positions):
            element = NUMBER_TO_SYMBOL.get(int(number), f"Z{int(number)}")
            if element not in grouped_positions:
                species_order.append(element)
            grouped_positions[element].append([float(x) for x in position])

        lines = [comment, "1.0"]
        for vec in matrix:
            lines.append(" ".join(f"{float(x):.16f}" for x in vec))
        lines.append(" ".join(species_order))
        lines.append(" ".join(str(len(grouped_positions[el])) for el in species_order))
        lines.append("Cartesian")
        for element in species_order:
            for position in grouped_positions[element]:
                lines.append(" ".join(f"{x:.16f}" for x in position))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lattice = structure.get("lattice", {})
    matrix = lattice.get("matrix")
    if not isinstance(matrix, list) or len(matrix) != 3:
        raise ValueError(f"Structure for {comment} has no 3x3 lattice matrix.")

    sites = [site for site in structure.get("sites", []) if isinstance(site, Mapping)]
    species_order: List[str] = []
    species_counts: Counter = Counter()
    grouped_sites: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for site in sites:
        element = site_species(site)
        if element is None:
            continue
        if element not in species_counts:
            species_order.append(element)
        species_counts[element] += 1
        grouped_sites[element].append(site)

    lines = [comment, "1.0"]
    for vec in matrix:
        lines.append(" ".join(f"{float(x):.16f}" for x in vec))
    lines.append(" ".join(species_order))
    lines.append(" ".join(str(int(species_counts[el])) for el in species_order))
    lines.append("Direct")
    for element in species_order:
        for site in grouped_sites[element]:
            lines.append(" ".join(f"{x:.16f}" for x in get_coords(site)))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def z_element_pairs(structure: Mapping[str, Any]) -> List[Tuple[float, str]]:
    ase_structure = unwrap_ase_structure(structure)
    if ase_structure is not None:
        pairs: List[Tuple[float, str]] = []
        for number, position in zip(ase_structure.get("numbers", []), ase_structure.get("positions", [])):
            if not isinstance(position, list) or len(position) != 3:
                continue
            pairs.append((float(position[2]), NUMBER_TO_SYMBOL.get(int(number), f"Z{int(number)}")))
        return pairs

    pairs = []
    for site in structure.get("sites", []):
        if not isinstance(site, Mapping):
            continue
        element = site_species(site)
        if element is None:
            continue
        coords = site.get("xyz")
        if not isinstance(coords, list):
            coords = site.get("abc")
        if not isinstance(coords, list) or len(coords) != 3:
            continue
        pairs.append((float(coords[2]), element))
    return pairs


def infer_janus_like(structure: Mapping[str, Any], surface_tolerance: float = 0.2) -> Tuple[float, float]:
    pairs = z_element_pairs(structure)
    if len(pairs) <= 2:
        return 0.0, 1.0
    z_values = [z for z, _ in pairs]
    z_min = min(z_values)
    z_max = max(z_values)
    if z_max - z_min <= EPS:
        return 0.0, 0.0
    bottom = {element for z, element in pairs if z - z_min <= surface_tolerance}
    top = {element for z, element in pairs if z_max - z <= surface_tolerance}
    return (1.0 if bottom != top else 0.0), 0.0


def weighted_mean_var(values: Iterable[Tuple[float, float]]) -> Tuple[float, float]:
    items = list(values)
    weight_sum = sum(weight for _, weight in items)
    if weight_sum <= 0:
        return 0.0, 0.0
    mean = sum(value * weight for value, weight in items) / weight_sum
    var = sum(weight * (value - mean) ** 2 for value, weight in items) / weight_sum
    return mean, var


def descriptor_row(
    uid: str,
    data: Mapping[str, Any],
    structure: Mapping[str, Any],
    feature_columns: List[str],
    pointgroup_columns: set,
    layergroup_columns: set,
    report: MutableMapping[str, Any],
) -> Dict[str, float]:
    row = {column: 0.0 for column in feature_columns}
    counts = composition_from_structure(structure)
    total_atoms = sum(counts.values())
    if total_atoms <= 0:
        report["bad_composition"].append(uid)
        return row

    missing_mass = sorted(el for el in counts if el not in ELEMENTS)
    if missing_mass:
        report["missing_atomic_data"][uid] = missing_mass
    known_elements = {el: ELEMENTS[el] for el in counts if el in ELEMENTS}

    mass_items = [(meta[1], counts[el]) for el, meta in known_elements.items()]
    mass_mean, mass_var = weighted_mean_var(mass_items)
    row["mass_avg_log1p"] = math.log1p(max(mass_mean, 0.0))
    row["mass_var_log1p"] = math.log1p(max(mass_var, 0.0))
    row["light_element_fraction"] = (
        sum(counts[el] for el, meta in known_elements.items() if meta[0] <= 18) / total_atoms
    )
    row["heavy_element_fraction"] = (
        sum(counts[el] for el, meta in known_elements.items() if meta[0] >= 37) / total_atoms
    )

    en_items = [
        (meta[2], counts[el])
        for el, meta in known_elements.items()
        if meta[2] is not None
    ]
    en_mean, en_var = weighted_mean_var((float(value), weight) for value, weight in en_items)
    row["en_atom_mean"] = en_mean
    row["en_atom_std"] = math.sqrt(max(en_var, 0.0))

    pair_diffs: List[Tuple[float, float]] = []
    elements = sorted(known_elements)
    missing_en = any(known_elements[el][2] is None for el in elements)
    for i, left in enumerate(elements):
        left_en = known_elements[left][2]
        if left_en is None:
            continue
        for right in elements[i + 1 :]:
            right_en = known_elements[right][2]
            if right_en is None:
                continue
            pair_diffs.append((abs(float(left_en) - float(right_en)), 1.0))
    if pair_diffs and not missing_en:
        diff_mean, diff_var = weighted_mean_var(pair_diffs)
        row["en_diff_mean_filled"] = diff_mean
        row["en_diff_max_filled"] = max(value for value, _ in pair_diffs)
        row["en_diff_var_filled"] = diff_var
        row["en_diff_missing_flag"] = 0.0
        row["is_ionic_like_clean"] = 1.0 if row["en_diff_max_filled"] >= 1.7 else 0.0
    else:
        row["en_diff_missing_flag"] = 1.0
        row["is_ionic_like_clean"] = 0.0

    gap = numeric(data.get("gap"), 0.0) or 0.0
    row["gap_pbe_log1p"] = math.log1p(max(gap, 0.0))
    row["gap_pbe_near_zero_flag"] = 1.0 if gap <= 0.05 else 0.0

    alpha_x = numeric(data.get("alphax_el"))
    alpha_y = numeric(data.get("alphay_el"))
    alpha_z = numeric(data.get("alphaz_el"))
    if alpha_x is None or alpha_y is None or alpha_z is None:
        report["missing_alpha_el"].append(uid)
    else:
        alpha_values = [alpha_x, alpha_y, alpha_z]
        alpha_mean = sum(alpha_values) / 3.0
        alpha_xy = (alpha_x + alpha_y) / 2.0
        alpha_anis = math.sqrt(
            (
                (alpha_x - alpha_y) ** 2
                + (alpha_y - alpha_z) ** 2
                + (alpha_z - alpha_x) ** 2
            )
            / 2.0
        )
        row["alpha_el_mean_log1p"] = math.log1p(max(alpha_mean, 0.0))
        row["alpha_el_inplane_mean_log1p"] = math.log1p(max(alpha_xy, 0.0))
        row["alpha_el_out_of_plane_log1p"] = math.log1p(max(alpha_z, 0.0))
        row["alpha_el_anisotropy_log1p"] = math.log1p(max(alpha_anis, 0.0))
        row["alpha_el_anisotropy_ratio"] = alpha_anis / (alpha_mean + EPS)
        row["alpha_el_ratio_z_xy"] = alpha_z / (alpha_xy + EPS)
        row["alpha_el_xy_anisotropy_ratio"] = abs(alpha_x - alpha_y) / (alpha_xy + EPS)

    row["has_inversion_symmetry"] = as_bool_number(data.get("has_inversion_symmetry", False))
    if "is_janus_like" in data:
        row["is_janus_like_clean"] = as_bool_number(data.get("is_janus_like"))
        row["is_janus_like_missing_flag"] = 0.0
    else:
        (
            row["is_janus_like_clean"],
            row["is_janus_like_missing_flag"],
        ) = infer_janus_like(structure)

    pointgroup = data.get("pointgroup")
    if pointgroup is not None:
        column = f"pointgroup__{normalize_pointgroup(pointgroup)}"
        if column in pointgroup_columns:
            row[column] = 1.0
        else:
            report["unknown_pointgroups"][str(pointgroup)] += 1

    layergroup = data.get("layergroup")
    if layergroup is not None:
        column = f"layergroup__{normalize_layergroup(layergroup)}"
        if column in layergroup_columns:
            row[column] = 1.0
        else:
            report["unknown_layergroups"][str(layergroup)] += 1

    return row


def sample_paths(source_dir: Path, limit: Optional[int]) -> Iterable[Path]:
    paths = sorted(source_dir.rglob("data.json"))
    if limit is not None:
        paths = paths[:limit]
    return paths


def make_report() -> Dict[str, Any]:
    return {
        "descriptor_origin": {
            "mass_avg_log1p": "computed from structure composition and atomic masses",
            "mass_var_log1p": "computed from structure composition and atomic masses",
            "light_element_fraction": "computed from structure composition, Z <= 18",
            "heavy_element_fraction": "computed from structure composition, Z >= 37",
            "en_atom_mean": "computed from structure composition and Pauling electronegativity",
            "en_atom_std": "computed from structure composition and Pauling electronegativity",
            "en_diff_mean_filled": "computed from unweighted unique element-pair electronegativity differences",
            "en_diff_max_filled": "computed from element-pair electronegativity differences",
            "en_diff_var_filled": "computed from unweighted unique element-pair electronegativity differences",
            "en_diff_missing_flag": "computed; 1 when no valid element-pair electronegativity difference exists",
            "is_ionic_like_clean": "computed; 1 when max electronegativity difference is at least 1.7",
            "gap_pbe_log1p": "from data.json gap",
            "gap_pbe_near_zero_flag": "from data.json gap",
            "alpha_el_*": "from data.json alphax_el, alphay_el, alphaz_el",
            "has_inversion_symmetry": "from data.json has_inversion_symmetry",
            "is_janus_like_clean": "from data.json is_janus_like when present, otherwise inferred from top/bottom surface element sets",
            "is_janus_like_missing_flag": "computed; 1 for two-atom structures where top/bottom surface chemistry is underdetermined",
            "pointgroup__*": "one-hot from data.json pointgroup, using training descriptor columns",
            "layergroup__*": "one-hot from data.json layergroup, using training descriptor columns",
        },
        "bad_composition": [],
        "missing_alpha_el": [],
        "missing_atomic_data": {},
        "unknown_pointgroups": defaultdict(int),
        "unknown_layergroups": defaultdict(int),
    }


def prepare_dataset(
    source_dir: Path,
    descriptor_csv: Path,
    config_template: Path,
    out_dir: Path,
    limit: Optional[int],
    overwrite: bool,
) -> Dict[str, Any]:
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{out_dir} already exists and is not empty. Use --overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = read_descriptor_columns(descriptor_csv)
    pointgroup_columns = {column for column in feature_columns if column.startswith("pointgroup__")}
    layergroup_columns = {column for column in feature_columns if column.startswith("layergroup__")}
    report = make_report()

    descriptor_path = out_dir / "aux_physical_priors.csv"
    id_prop_path = out_dir / "id_prop.csv"
    source_index_path = out_dir / "source_index.csv"
    structures_path = out_dir / "structures_interband_only.json"

    rows_written = 0
    skipped: List[Dict[str, str]] = []
    seen_uids: set = set()
    ids: List[str] = []
    structures_index: List[Dict[str, Any]] = []

    with descriptor_path.open("w", encoding="utf-8", newline="") as desc_f, id_prop_path.open(
        "w", encoding="utf-8", newline=""
    ) as id_f, source_index_path.open("w", encoding="utf-8", newline="") as index_f:
        desc_writer = csv.writer(desc_f)
        id_writer = csv.writer(id_f)
        index_writer = csv.writer(index_f)
        desc_writer.writerow(["uid"] + feature_columns)
        index_writer.writerow(["uid", "olduid", "sample_dir", "data_json", "structure_json"])

        for data_path in sample_paths(source_dir, limit):
            structure_path = data_path.parent / "structure.json"
            if not structure_path.exists():
                skipped.append({"path": str(data_path), "reason": "missing structure.json"})
                continue

            data = load_json(data_path)
            structure = load_json(structure_path)
            uid = safe_uid(str(data.get("uid") or data_path.parent.name))
            if uid in seen_uids:
                skipped.append({"path": str(data_path), "reason": f"duplicate uid {uid}"})
                continue
            seen_uids.add(uid)

            row = descriptor_row(
                uid,
                data,
                structure,
                feature_columns,
                pointgroup_columns,
                layergroup_columns,
                report,
            )
            write_poscar(out_dir / uid, structure, uid)
            desc_writer.writerow([uid] + [fmt(row[column]) for column in feature_columns])
            id_writer.writerow([uid] + list(TARGET_PLACEHOLDER))
            index_writer.writerow(
                [
                    uid,
                    data.get("olduid", ""),
                    str(data_path.parent.resolve()),
                    str(data_path.resolve()),
                    str(structure_path.resolve()),
                ]
            )
            ids.append(uid)
            rows_written += 1
            structures_index.append(
                {
                    "uid": uid,
                    "source_data_json": str(data_path.resolve()),
                    "source_structure_json": str(structure_path.resolve()),
                    "structure": structure,
                }
            )

    write_json(structures_path, structures_index)
    write_json(out_dir / "ids_predict.json", ids)
    write_json(out_dir / "ids_train_val_test.json", {"id_train": ids, "id_val": ids, "id_test": ids})

    if config_template.exists():
        config = load_json(config_template)
    else:
        config = {}
    config["use_extra_features_csv"] = True
    config["extra_features_csv"] = str(descriptor_path.resolve())
    config["extra_features_id_key"] = "uid"
    config["use_sample_weights"] = False
    config["sample_weight_csv"] = None
    config["output_dir"] = str((Path("runs") / "interband_only_eta_parallel_predict").resolve())
    config.setdefault("model", {})
    config["model"]["extra_features"] = len(feature_columns)
    write_json(out_dir / "config.json", config)

    report["unknown_pointgroups"] = dict(report["unknown_pointgroups"])
    report["unknown_layergroups"] = dict(report["unknown_layergroups"])
    report["feature_count"] = len(feature_columns)
    write_json(out_dir / "descriptor_source_report.json", report)

    summary = {
        "source_dir": str(source_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "samples_written": rows_written,
        "samples_skipped": len(skipped),
        "skipped": skipped[:50],
        "feature_count": len(feature_columns),
        "descriptor_csv": str(descriptor_path.resolve()),
        "id_prop_csv": str(id_prop_path.resolve()),
        "structures_json": str(structures_path.resolve()),
        "config_json": str((out_dir / "config.json").resolve()),
        "split_policy": "all sample ids are listed in train, val, and test for inference-only loading",
        "unknown_pointgroups": report["unknown_pointgroups"],
        "unknown_layergroups": report["unknown_layergroups"],
        "missing_atomic_data_count": len(report["missing_atomic_data"]),
        "missing_alpha_el_count": len(report["missing_alpha_el"]),
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = prepare_dataset(
        source_dir=Path(args.source_dir),
        descriptor_csv=Path(args.train_descriptor_csv),
        config_template=Path(args.config_template),
        out_dir=Path(args.out_dir),
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
