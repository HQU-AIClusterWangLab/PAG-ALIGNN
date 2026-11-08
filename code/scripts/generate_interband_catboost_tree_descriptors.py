#!/usr/bin/env python
"""Generate CatBoost tree-model descriptors for interband-only prediction samples.

The high-alpha CatBoost gate was trained on
``tree_descriptors_v2_model_ready.before_low_cost_descriptors.csv``.  This
script rebuilds the same descriptor schema for the 3351 interband-only samples
using each sample's ``data.json`` and ``structure.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_interband_prediction_dataset import (  # noqa: E402
    ELEMENTS,
    NUMBER_TO_SYMBOL,
    load_json,
    numeric,
    unwrap_ase_structure,
)


DEFAULT_TEMPLATE = (
    "cat/high_alpha_inplane_gate/"
    "tree_descriptors_v2_model_ready.before_low_cost_descriptors.csv"
)
DEFAULT_SOURCE_INDEX = "data/interband_only_eta_parallel_predict/source_index.csv"
DEFAULT_OUTPUT = (
    "data/interband_only_eta_parallel_predict/"
    "tree_descriptors_v2_model_ready.before_low_cost_descriptors.csv"
)

EPS = 1e-12
DIPOLE_THRESHOLD = 1e-6
BOND_CUTOFF_FACTOR = 1.25

GROUPS = {
    "H": 1,
    "Li": 1,
    "Na": 1,
    "K": 1,
    "Rb": 1,
    "Cs": 1,
    "Be": 2,
    "Mg": 2,
    "Ca": 2,
    "Sr": 2,
    "Ba": 2,
    "Sc": 3,
    "Y": 3,
    "La": 3,
    "Ti": 4,
    "Zr": 4,
    "Hf": 4,
    "V": 5,
    "Nb": 5,
    "Ta": 5,
    "Cr": 6,
    "Mo": 6,
    "W": 6,
    "Mn": 7,
    "Tc": 7,
    "Re": 7,
    "Fe": 8,
    "Ru": 8,
    "Os": 8,
    "Co": 9,
    "Rh": 9,
    "Ir": 9,
    "Ni": 10,
    "Pd": 10,
    "Pt": 10,
    "Cu": 11,
    "Ag": 11,
    "Au": 11,
    "Zn": 12,
    "Cd": 12,
    "Hg": 12,
    "B": 13,
    "Al": 13,
    "Ga": 13,
    "In": 13,
    "Tl": 13,
    "C": 14,
    "Si": 14,
    "Ge": 14,
    "Sn": 14,
    "Pb": 14,
    "N": 15,
    "P": 15,
    "As": 15,
    "Sb": 15,
    "Bi": 15,
    "O": 16,
    "S": 16,
    "Se": 16,
    "Te": 16,
    "F": 17,
    "Cl": 17,
    "Br": 17,
    "I": 17,
    "He": 18,
    "Ne": 18,
    "Ar": 18,
    "Kr": 18,
    "Xe": 18,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--source-index", default=DEFAULT_SOURCE_INDEX)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--report",
        default=None,
        help="Optional report JSON path. Defaults to output path with .report.json.",
    )
    return parser.parse_args()


def read_template_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
    if not header or header[0] != "sample_id":
        raise ValueError(f"{path} must start with sample_id.")
    return header


def read_source_index(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"uid", "data_json", "structure_json"}
    if not rows:
        raise ValueError(f"No rows found in source index: {path}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return rows


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    if number.is_integer():
        return str(int(number))
    return repr(number)


def vector_sub(a: list[float], b: list[float]) -> list[float]:
    return [a[i] - b[i] for i in range(3)]


def vector_add(a: list[float], b: list[float]) -> list[float]:
    return [a[i] + b[i] for i in range(3)]


def vector_scale(a: list[float], scale: float) -> list[float]:
    return [scale * a[i] for i in range(3)]


def dot(a: list[float], b: list[float]) -> float:
    return sum(a[i] * b[i] for i in range(3))


def norm(a: list[float]) -> float:
    return math.sqrt(max(dot(a, a), 0.0))


def cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def mean(values: Iterable[float]) -> float | None:
    items = list(values)
    if not items:
        return None
    return sum(items) / len(items)


def population_std(values: Iterable[float]) -> float | None:
    items = list(values)
    if not items:
        return None
    mu = sum(items) / len(items)
    return math.sqrt(sum((value - mu) ** 2 for value in items) / len(items))


def population_var(values: Iterable[float]) -> float | None:
    items = list(values)
    if not items:
        return None
    mu = sum(items) / len(items)
    return sum((value - mu) ** 2 for value in items) / len(items)


def weighted_mean_var(values: Iterable[tuple[float, float]]) -> tuple[float | None, float | None]:
    items = list(values)
    weight_sum = sum(weight for _, weight in items)
    if weight_sum <= 0:
        return None, None
    mu = sum(value * weight for value, weight in items) / weight_sum
    var = sum(weight * (value - mu) ** 2 for value, weight in items) / weight_sum
    return mu, var


def ase_payload(structure: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = unwrap_ase_structure(structure)
    if payload is None:
        raise ValueError("Only ASE-style canonical structure JSON is supported.")
    return payload


def positions_cell_numbers(structure: Mapping[str, Any]) -> tuple[list[list[float]], list[list[float]], list[int]]:
    payload = ase_payload(structure)
    cell = [[float(x) for x in vec] for vec in payload["cell"]]
    positions = [[float(x) for x in pos] for pos in payload["positions"]]
    numbers = [int(number) for number in payload["numbers"]]
    if len(positions) != len(numbers):
        raise ValueError("Structure has mismatched numbers and positions.")
    return positions, cell, numbers


def minimum_image_delta(
    pi: list[float],
    pj: list[float],
    cell: list[list[float]],
    include_zero_translation: bool,
) -> tuple[float, list[float]]:
    base = vector_sub(pj, pi)
    best_dist = math.inf
    best_delta = base
    for ia in (-1, 0, 1):
        for ib in (-1, 0, 1):
            if not include_zero_translation and ia == 0 and ib == 0:
                continue
            delta = vector_add(base, vector_add(vector_scale(cell[0], ia), vector_scale(cell[1], ib)))
            dist = norm(delta)
            if dist < best_dist:
                best_dist = dist
                best_delta = delta
    return best_dist, best_delta


def nearest_neighbor_distances(positions: list[list[float]], cell: list[list[float]]) -> list[float]:
    distances: list[float] = []
    for i, pi in enumerate(positions):
        best = math.inf
        for j, pj in enumerate(positions):
            include_zero = i != j
            dist, _ = minimum_image_delta(pi, pj, cell, include_zero_translation=include_zero)
            if dist < best:
                best = dist
        if math.isfinite(best):
            distances.append(best)
    return distances


def unique_pair_distances(
    positions: list[list[float]],
    cell: list[list[float]],
) -> list[tuple[int, int, float, list[float]]]:
    pairs: list[tuple[int, int, float, list[float]]] = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist, delta = minimum_image_delta(
                positions[i],
                positions[j],
                cell,
                include_zero_translation=True,
            )
            pairs.append((i, j, dist, delta))
    return pairs


def bond_graph(
    positions: list[list[float]],
    cell: list[list[float]],
) -> tuple[list[tuple[int, int, float, list[float]]], dict[int, list[list[float]]]]:
    pairs = unique_pair_distances(positions, cell)
    if not pairs:
        return [], defaultdict(list)
    min_pair = min(dist for _, _, dist, _ in pairs)
    cutoff = min_pair * BOND_CUTOFF_FACTOR
    bonds = [(i, j, dist, delta) for i, j, dist, delta in pairs if dist <= cutoff + 1e-9]
    neighbors: dict[int, list[list[float]]] = defaultdict(list)
    for i, j, _, delta in bonds:
        neighbors[i].append(delta)
        neighbors[j].append(vector_scale(delta, -1.0))
    return bonds, neighbors


def bond_angles(neighbors: Mapping[int, list[list[float]]]) -> list[float]:
    values: list[float] = []
    for vectors in neighbors.values():
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                ni = norm(vectors[i])
                nj = norm(vectors[j])
                if ni <= EPS or nj <= EPS:
                    continue
                cosine = max(-1.0, min(1.0, dot(vectors[i], vectors[j]) / (ni * nj)))
                values.append(math.degrees(math.acos(cosine)))
    return values


def valence_proxy(symbol: str) -> float | None:
    group = GROUPS.get(symbol)
    if group is None:
        return None
    if group <= 12:
        return float(group)
    return float(group - 10)


def composition(numbers: list[int]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for number in numbers:
        counts[NUMBER_TO_SYMBOL.get(number, f"Z{number}")] += 1
    return counts


def infer_janus_like(
    positions: list[list[float]],
    numbers: list[int],
    threshold: float = 0.2,
) -> float | None:
    if len(positions) <= 2:
        return None
    z_values = [pos[2] for pos in positions]
    z_min = min(z_values)
    z_max = max(z_values)
    if z_max - z_min <= EPS:
        return 0.0
    top = {
        NUMBER_TO_SYMBOL.get(number, f"Z{number}")
        for number, pos in zip(numbers, positions)
        if z_max - pos[2] <= threshold
    }
    bottom = {
        NUMBER_TO_SYMBOL.get(number, f"Z{number}")
        for number, pos in zip(numbers, positions)
        if pos[2] - z_min <= threshold
    }
    return 1.0 if top != bottom else 0.0


def geometry_descriptors(
    positions: list[list[float]],
    cell: list[list[float]],
) -> dict[str, Any]:
    a_len = norm(cell[0])
    b_len = norm(cell[1])
    gamma = math.degrees(
        math.acos(max(-1.0, min(1.0, dot(cell[0], cell[1]) / ((a_len * b_len) + EPS))))
    )
    area_2d = norm(cross(cell[0], cell[1]))
    z_values = [pos[2] for pos in positions]
    nn = nearest_neighbor_distances(positions, cell)
    bonds, neighbors = bond_graph(positions, cell)
    bond_lengths = [dist for _, _, dist, _ in bonds]
    coord = [len(neighbors.get(i, [])) for i in range(len(positions))]
    angles = bond_angles(neighbors)

    return {
        "a_len": a_len,
        "b_len": b_len,
        "gamma_angle": gamma,
        "area_2d": area_2d,
        "area_per_atom": area_2d / max(len(positions), 1),
        "buckling": population_std(z_values),
        "nn_dist_mean": mean(nn),
        "nn_dist_std": population_std(nn),
        "nn_dist_min": min(nn) if nn else None,
        "nn_dist_max": max(nn) if nn else None,
        "bond_count": len(bonds),
        "bond_len_mean": mean(bond_lengths),
        "bond_len_std": population_std(bond_lengths),
        "bond_len_min": min(bond_lengths) if bond_lengths else None,
        "bond_len_max": max(bond_lengths) if bond_lengths else None,
        "coord_mean": mean(coord),
        "coord_std": population_std(coord),
        "coord_min": min(coord) if coord else None,
        "coord_max": max(coord) if coord else None,
        "bond_angle_mean": mean(angles),
        "bond_angle_std": population_std(angles),
        "bond_angle_min": min(angles) if angles else None,
        "bond_angle_max": max(angles) if angles else None,
    }


def composition_descriptors(numbers: list[int]) -> tuple[dict[str, Any], list[str]]:
    counts = composition(numbers)
    natoms = sum(counts.values())
    known = {symbol: ELEMENTS[symbol] for symbol in counts if symbol in ELEMENTS}
    missing = sorted(symbol for symbol in counts if symbol not in ELEMENTS)

    masses = [(meta[1], counts[symbol]) for symbol, meta in known.items()]
    mass_mean, mass_var = weighted_mean_var(masses)
    mass_values = [meta[1] for meta in known.values()]

    atomic_numbers = [(float(meta[0]), counts[symbol]) for symbol, meta in known.items()]
    atomic_mean, atomic_var = weighted_mean_var(atomic_numbers)
    atomic_values = [float(meta[0]) for meta in known.values()]

    en_values = [
        (float(meta[2]), counts[symbol])
        for symbol, meta in known.items()
        if meta[2] is not None
    ]
    en_mean, en_var = weighted_mean_var(en_values)

    elements_with_en = sorted(symbol for symbol, meta in known.items() if meta[2] is not None)
    en_diffs: list[float] = []
    for i, left in enumerate(elements_with_en):
        left_en = float(known[left][2])
        for right in elements_with_en[i + 1 :]:
            right_en = float(known[right][2])
            en_diffs.append(abs(left_en - right_en))

    groups = [(float(GROUPS[symbol]), counts[symbol]) for symbol in known if symbol in GROUPS]
    group_mean, group_var = weighted_mean_var(groups)

    valences = [
        (float(valence_proxy(symbol)), counts[symbol])
        for symbol in known
        if valence_proxy(symbol) is not None
    ]
    valence_mean, valence_var = weighted_mean_var(valences)
    valence_values = [float(valence_proxy(symbol)) for symbol in known if valence_proxy(symbol) is not None]

    descriptors = {
        "natoms": natoms,
        "nspecies": len(counts),
        "mass_avg": mass_mean,
        "mass_min": min(mass_values) if mass_values else None,
        "mass_max": max(mass_values) if mass_values else None,
        "mass_var": mass_var,
        "atomic_number_mean": atomic_mean,
        "atomic_number_min": min(atomic_values) if atomic_values else None,
        "atomic_number_max": max(atomic_values) if atomic_values else None,
        "atomic_number_var": atomic_var,
        "light_element_fraction": (
            sum(counts[symbol] for symbol, meta in known.items() if meta[0] <= 18) / natoms
        ),
        "heavy_element_fraction": (
            sum(counts[symbol] for symbol, meta in known.items() if meta[0] >= 37) / natoms
        ),
        "en_atom_mean": en_mean,
        "en_atom_std": math.sqrt(en_var) if en_var is not None else None,
        "en_diff_mean": mean(en_diffs),
        "en_diff_max": max(en_diffs) if en_diffs else None,
        "en_diff_var": population_var(en_diffs),
        "is_ionic_like": (
            1.0 if en_diffs and max(en_diffs) >= 1.7 else 0.0 if en_diffs else None
        ),
        "group_mean": group_mean,
        "group_std": math.sqrt(group_var) if group_var is not None else None,
        "valence_proxy_mean": valence_mean,
        "valence_proxy_max": max(valence_values) if valence_values else None,
        "valence_proxy_var": valence_var,
    }
    return descriptors, missing


def sample_descriptor(uid: str, data: Mapping[str, Any], structure: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    positions, cell, numbers = positions_cell_numbers(structure)
    row: dict[str, Any] = {"sample_id": uid}
    row.update(composition_descriptors(numbers)[0])
    row.update(geometry_descriptors(positions, cell))
    row["thickness"] = numeric(data.get("thickness"), max(pos[2] for pos in positions) - min(pos[2] for pos in positions))
    row["gap_pbe"] = numeric(data.get("gap"))
    row["alphax_el"] = numeric(data.get("alphax_el"))
    row["alphay_el"] = numeric(data.get("alphay_el"))
    row["alphaz_el"] = numeric(data.get("alphaz_el"))
    alpha_values = [row["alphax_el"], row["alphay_el"], row["alphaz_el"]]
    if all(value is not None for value in alpha_values):
        row["alpha_el_mean_diag"] = sum(float(value) for value in alpha_values) / 3.0
    else:
        row["alpha_el_mean_diag"] = None
    row["layergroup"] = str(data.get("layergroup", ""))
    row["pointgroup"] = str(data.get("pointgroup", ""))
    row["has_inversion_symmetry"] = (
        1.0 if bool(data.get("has_inversion_symmetry", False)) else 0.0
    )
    if "is_janus_like" in data:
        row["is_janus_like"] = 1.0 if bool(data["is_janus_like"]) else 0.0
    else:
        row["is_janus_like"] = infer_janus_like(positions, numbers)
    dipz = numeric(data.get("dipz"))
    row["has_out_of_plane_dipole"] = (
        1.0 if dipz is not None and abs(dipz) > DIPOLE_THRESHOLD else 0.0
        if dipz is not None
        else None
    )
    _, missing = composition_descriptors(numbers)
    return row, missing


def to_ordered_row(row: Mapping[str, Any], header: list[str]) -> dict[str, str]:
    return {column: fmt(row.get(column)) for column in header}


def main() -> None:
    args = parse_args()
    template = Path(args.template)
    source_index = Path(args.source_index)
    output = Path(args.output)
    report_path = Path(args.report) if args.report else output.with_suffix(".report.json")

    header = read_template_header(template)
    samples = read_source_index(source_index)
    output.parent.mkdir(parents=True, exist_ok=True)

    missing_atomic: dict[str, list[str]] = {}
    errors: dict[str, str] = {}
    rows: list[dict[str, str]] = []
    for item in samples:
        uid = str(item["uid"])
        try:
            data = load_json(Path(item["data_json"]))
            structure = load_json(Path(item["structure_json"]))
            descriptor, missing = sample_descriptor(uid, data, structure)
            if missing:
                missing_atomic[uid] = missing
            rows.append(to_ordered_row(descriptor, header))
        except Exception as exc:  # keep report useful for batch diagnosis
            errors[uid] = str(exc)

    if errors:
        raise RuntimeError(
            f"Failed to generate {len(errors)} descriptor rows. "
            f"First failure: {next(iter(errors.items()))}"
        )

    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "template": str(template.resolve()),
        "source_index": str(source_index.resolve()),
        "output": str(output.resolve()),
        "rows_written": len(rows),
        "columns_written": len(header),
        "feature_columns": header[1:],
        "categorical_columns": ["layergroup", "pointgroup"],
        "dipole_threshold": DIPOLE_THRESHOLD,
        "bond_cutoff_factor": BOND_CUTOFF_FACTOR,
        "missing_atomic_data_count": len(missing_atomic),
        "missing_atomic_data": missing_atomic,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
