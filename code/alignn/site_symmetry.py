"""Node-level Wyckoff and site-symmetry descriptors."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np
from jarvis.core.specie import Specie


WYCKOFF_LETTERS = tuple("abcdefghijklmnopqrstuvwxyz")
SITE_SYMMETRY_GROUPS = (
    "1",
    "-1",
    "2",
    "m",
    "2/m",
    "222",
    "mm2",
    "mmm",
    "4",
    "-4",
    "4/m",
    "422",
    "4mm",
    "-42m",
    "-4m2",
    "4/mmm",
    "3",
    "-3",
    "32",
    "3m",
    "-3m",
    "6",
    "-6",
    "6/m",
    "622",
    "6mm",
    "-6m2",
    "-62m",
    "6/mmm",
    "23",
    "m-3",
    "432",
    "-43m",
    "m-3m",
)
SITE_SYMMETRY_NODE_FEATURES = (
    1
    + 1
    + len(WYCKOFF_LETTERS)
    + len(SITE_SYMMETRY_GROUPS)
    + 1
    + 9
    + 3
    + 3
    + 2
)


def _atomic_numbers(elements: Sequence[str]) -> list[int]:
    """Return atomic numbers for a sequence of element symbols."""
    numbers = []
    for element in elements:
        numbers.append(int(Specie(element).Z))
    return numbers


def _compact_site_symbol(symbol: object) -> str:
    """Normalize common spglib site-symmetry spelling variants."""
    text = str(symbol).strip().replace(" ", "")
    if not text:
        return "1"
    text = text.replace("\u2212", "-")
    compact = text.replace(".", "")
    if not compact:
        return "1"
    if compact in SITE_SYMMETRY_GROUPS:
        return compact
    chars = list(compact)
    if chars.count("m") == 2 and "2" in chars and len(chars) == 3:
        return "mm2"
    if chars.count("m") == 3 and len(chars) == 3:
        return "mmm"
    if chars.count("2") == 3 and len(chars) == 3:
        return "222"
    if chars.count("m") == 1 and len(chars) == 1:
        return "m"
    for rot in ("2", "3", "4", "6"):
        if chars.count(rot) == 1 and len(chars) == 1:
            return rot
    return compact


def _axis_flags(symbol: object) -> tuple[list[float], list[float]]:
    """Extract weak orientation flags from raw 3-position site symbols."""
    text = str(symbol).strip().replace(" ", "")
    if len(text) != 3:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    mirror = [1.0 if char == "m" else 0.0 for char in text]
    rotation = [1.0 if char in {"2", "3", "4", "6"} else 0.0 for char in text]
    return mirror, rotation


def _safe_dataset_value(dataset, key: str, default=None):
    value = getattr(dataset, key, None)
    if value is not None:
        return value
    try:
        return dataset[key]
    except Exception:
        return default


def site_symmetry_node_features(
    atoms,
    symprec: float = 1e-3,
    angle_tolerance: float = 5.0,
    feature_dim: int = SITE_SYMMETRY_NODE_FEATURES,
) -> np.ndarray:
    """Return fixed-size Wyckoff/site-symmetry features for each atom.

    The descriptor is intentionally separate from the base atom feature vector.
    If symmetry detection fails, a zero matrix is returned so callers can keep
    batching stable while the model learns to ignore missing information.
    """
    if feature_dim != SITE_SYMMETRY_NODE_FEATURES:
        raise ValueError(
            "site_symmetry_node_features expects feature_dim="
            f"{SITE_SYMMETRY_NODE_FEATURES}, got {feature_dim}."
        )
    n_atoms = int(atoms.num_atoms)
    features = np.zeros((n_atoms, SITE_SYMMETRY_NODE_FEATURES), dtype=float)

    try:
        import spglib

        cell = (
            np.asarray(atoms.lattice_mat, dtype=float),
            np.asarray(atoms.frac_coords, dtype=float),
            _atomic_numbers(atoms.elements),
        )
        dataset = spglib.get_symmetry_dataset(
            cell,
            symprec=float(symprec),
            angle_tolerance=float(angle_tolerance),
        )
    except Exception:
        return features
    if dataset is None:
        return features

    wyckoffs = list(_safe_dataset_value(dataset, "wyckoffs", []))
    site_symbols = list(
        _safe_dataset_value(dataset, "site_symmetry_symbols", [])
    )
    equivalent_atoms = list(_safe_dataset_value(dataset, "equivalent_atoms", []))
    if (
        len(wyckoffs) != n_atoms
        or len(site_symbols) != n_atoms
        or len(equivalent_atoms) != n_atoms
    ):
        return features

    wyckoff_to_index = {letter: idx for idx, letter in enumerate(WYCKOFF_LETTERS)}
    site_to_index = {
        symbol: idx for idx, symbol in enumerate(SITE_SYMMETRY_GROUPS)
    }
    equiv_counts = Counter(int(idx) for idx in equivalent_atoms)
    denom = max(n_atoms - 1, 1)

    for atom_idx in range(n_atoms):
        offset = 0
        features[atom_idx, offset] = 1.0
        offset += 1

        letter = str(wyckoffs[atom_idx]).strip().lower()
        letter_idx = wyckoff_to_index.get(letter)
        if letter_idx is None:
            features[atom_idx, offset] = 1.0
        offset += 1
        if letter_idx is not None:
            features[atom_idx, offset + letter_idx] = 1.0
        offset += len(WYCKOFF_LETTERS)

        compact_symbol = _compact_site_symbol(site_symbols[atom_idx])
        site_idx = site_to_index.get(compact_symbol)
        if site_idx is not None:
            features[atom_idx, offset + site_idx] = 1.0
        offset += len(SITE_SYMMETRY_GROUPS)
        if site_idx is None:
            features[atom_idx, offset] = 1.0
        offset += 1

        mirror_count = compact_symbol.count("m")
        char_flags = [
            min(mirror_count / 3.0, 1.0),
            1.0 if "2" in compact_symbol else 0.0,
            1.0 if "3" in compact_symbol else 0.0,
            1.0 if "4" in compact_symbol else 0.0,
            1.0 if "6" in compact_symbol else 0.0,
            1.0 if "-" in compact_symbol else 0.0,
            1.0 if compact_symbol in {"1", "-1"} else 0.0,
            1.0 if "/" in compact_symbol else 0.0,
            min(len(compact_symbol) / 6.0, 1.0),
        ]
        features[atom_idx, offset : offset + len(char_flags)] = char_flags
        offset += len(char_flags)

        mirror_axis, rotation_axis = _axis_flags(site_symbols[atom_idx])
        features[atom_idx, offset : offset + 3] = mirror_axis
        offset += 3
        features[atom_idx, offset : offset + 3] = rotation_axis
        offset += 3

        equiv_id = int(equivalent_atoms[atom_idx])
        features[atom_idx, offset] = equiv_counts.get(equiv_id, 1) / max(
            n_atoms, 1
        )
        offset += 1
        features[atom_idx, offset] = equiv_id / denom

    return features
