#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full-scan canonical-axis preprocessing for C2DB-style samples.

Fixes the earlier file-loss issue:
- every file inside each sample directory is copied to the output tree;
- known coordinate-dependent files are transformed;
- other files are copied unchanged;
- generic JSON files are scanned for recognizable vectors/tensors.

Coordinate rule:
- identify the vacuum direction and map it to canonical z;
- use only a signed axis permutation, not a continuous rotation;
- transform full tensors by v'=P v, A'=P A P^T, C'=P P P P C.

python canonicalize_c2db_axes_fullscan.py ^
  --root "E:/Yjlong/log/C2DB/C2DB/materials" ^
  --batch ^
  --output-root "E:/Yjlong/log/C2DB/canonicalized_dataset_fullscan"

"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

AXES = ("x", "y", "z")
IDX_TO_AXIS = {0: "x", 1: "y", 2: "z"}

EVEN_SINGLE_AXIS_FORCE_REGEXES = [
    re.compile(r"^alpha[xyz](_.*)?$"),
    re.compile(r"^plasmafrequency_[xyz]$"),
    re.compile(r"^eps[xyz](_.*)?$"),
    re.compile(r"^epsilon[xyz](_.*)?$"),
    re.compile(r"^emass[xyz](_.*)?$"),
    re.compile(r"^mass[xyz](_.*)?$"),
    re.compile(r"^mobility[xyz](_.*)?$"),
]
ODD_SINGLE_AXIS_FORCE_REGEXES = [
    re.compile(r"^dip[xyz]$"),
    re.compile(r"^P_spontaneous_[xyz]$"),
    re.compile(r"^polarization_[xyz]$"),
    re.compile(r"^moment_[xyz]$"),
]
PAIR_AXIS_FORCE_REGEXES = [
    re.compile(r"^dE_[xyz][xyz]$"),
    re.compile(r"^alpha[xyz][xyz](_.*)?$"),
    re.compile(r"^eps[xyz][xyz](_.*)?$"),
    re.compile(r"^epsilon[xyz][xyz](_.*)?$"),
]
AXIS_LABEL_FIELDS = {"spin_axis", "axis", "direction"}
EVEN_HINT_KEYWORDS = (
    "alpha", "polarizability", "plasmafrequency", "dielectric", "eps", "epsilon",
    "emass", "effective_mass", "mass", "mobility", "stiffness", "elastic",
    "conductivity", "susceptibility", "tensor", "matrix",
)
ODD_HINT_KEYWORDS = ("dip", "dipole", "polarization", "moment", "borncharge", "born_charge")
RANK1_HINTS = ("vector", "dip", "dipole", "polarization", "moment", "force", "velocity", "spin")
RANK2_TENSOR_HINTS = (
    "tensor", "matrix", "alpha", "polarizability", "epsilon", "dielectric",
    "emass", "effective_mass", "mass_tensor", "born", "charge_tensor",
)
STIFFNESS_HINTS = ("stiffness", "elastic", "elasticity", "cij", "c_ij", "c_vv", "elastic_tensor", "elastic_constants")
DO_NOT_TRANSFORM_ARRAY_HINTS = ("positions", "scaled_positions", "frac", "fractional", "kpoint", "kpoints", "qpoint", "qpoints", "path")

SINGLE_PATTERNS = [
    re.compile(r"^(.*?_)([xyz])$"),
    re.compile(r"^(.*?)([xyz])(_.*)$"),
    re.compile(r"^(.*?)([xyz])$"),
]
PAIR_PATTERNS = [
    re.compile(r"^(.*?_)([xyz])([xyz])$"),
    re.compile(r"^(.*?)([xyz])([xyz])(_.*)$"),
]

VOIGT6_PAIRS = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
VOIGT3_2D_PAIRS = [(0, 0), (1, 1), (0, 1)]


@dataclass
class TransformInfo:
    old_z_index: int
    new_axes_old_idx: List[int]
    new_axes_signs: List[int]
    old_to_new_axis_name: Dict[str, str]
    high_symmetry_inplane: bool
    inplane_lengths_old: List[float]
    action: str
    centered_slab: bool
    z_reassigned: bool
    inplane_reordered: bool
    sign_flipped: bool
    actual_changed: bool
    high_symmetry_kept: bool


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def path_is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def frac_from_cart(cart: np.ndarray, cell: np.ndarray) -> np.ndarray:
    return np.asarray(cart, dtype=float) @ np.linalg.inv(np.asarray(cell, dtype=float))


def cart_from_frac(frac: np.ndarray, cell: np.ndarray) -> np.ndarray:
    return np.asarray(frac, dtype=float) @ np.asarray(cell, dtype=float)


def build_signed_permutation_matrix(new_axes_old_idx: Sequence[int], new_axes_signs: Sequence[int]) -> np.ndarray:
    P = np.zeros((3, 3), dtype=float)
    for new_i, old_i in enumerate(new_axes_old_idx):
        P[new_i, old_i] = float(new_axes_signs[new_i])
    return P


def transform_rank2_tensor(A: np.ndarray, P: np.ndarray) -> np.ndarray:
    return P @ A @ P.T


def transform_rank2_stack(A: np.ndarray, P: np.ndarray) -> np.ndarray:
    return np.einsum("ia,...ab,jb->...ij", P, A, P)


def transform_rank4_tensor(C: np.ndarray, P: np.ndarray) -> np.ndarray:
    return np.einsum("ia,jb,kc,ld,abcd->ijkl", P, P, P, P, C)


def transform_rank4_stack(C: np.ndarray, P: np.ndarray) -> np.ndarray:
    return np.einsum("ia,jb,kc,ld,...abcd->...ijkl", P, P, P, P, C)


def voigt_to_rank4(M: np.ndarray, pairs: List[Tuple[int, int]], dim: int = 3) -> np.ndarray:
    C = np.zeros((dim, dim, dim, dim), dtype=float)
    for I, (i, j) in enumerate(pairs):
        for J, (k, l) in enumerate(pairs):
            val = M[I, J]
            for a, b in {(i, j), (j, i)}:
                for c, d in {(k, l), (l, k)}:
                    C[a, b, c, d] = val
    return C


def rank4_to_voigt(C: np.ndarray, pairs: List[Tuple[int, int]]) -> np.ndarray:
    M = np.zeros((len(pairs), len(pairs)), dtype=float)
    for I, (i, j) in enumerate(pairs):
        for J, (k, l) in enumerate(pairs):
            M[I, J] = C[i, j, k, l]
    return M


def transform_voigt6(M: np.ndarray, P: np.ndarray) -> np.ndarray:
    return rank4_to_voigt(transform_rank4_tensor(voigt_to_rank4(M, VOIGT6_PAIRS, 3), P), VOIGT6_PAIRS)


def transform_voigt3_2d(M: np.ndarray, P: np.ndarray) -> np.ndarray:
    return rank4_to_voigt(transform_rank4_tensor(voigt_to_rank4(M, VOIGT3_2D_PAIRS, 3), P), VOIGT3_2D_PAIRS)


def infer_z_axis(cell: np.ndarray, pbc: Optional[Sequence[bool]]) -> int:
    lengths = np.linalg.norm(cell, axis=1)
    if pbc is not None and len(pbc) == 3:
        non_periodic = [i for i, periodic in enumerate(pbc) if not periodic]
        if len(non_periodic) == 1:
            return non_periodic[0]
    return int(np.argmax(lengths))


def is_high_symmetry_inplane(data: dict, inplane_lengths: Sequence[float], length_rel_tol: float = 1e-3) -> bool:
    l1, l2 = float(inplane_lengths[0]), float(inplane_lengths[1])
    if max(l1, l2) > 0 and abs(l1 - l2) / max(l1, l2) <= length_rel_tol:
        return True
    tokens = []
    for k in ("pointgroup", "layergroup", "bravais_type", "bravais_search", "international"):
        if data.get(k) is not None:
            tokens.append(str(data[k]).lower())
    joined = " | ".join(tokens)
    markers = ["hexagonal", "trigonal", "square", "tetragonal", "6", "4", "-3m", "3m", "6mm", "p-3m1"]
    return any(m in joined for m in markers)


def choose_canonical_transform(data: dict, structure: dict, center_slab: bool = True, length_rel_tol: float = 1e-3) -> Tuple[TransformInfo, np.ndarray, np.ndarray, np.ndarray]:
    s = structure["1"]
    cell_old = np.asarray(s["cell"], dtype=float)
    pos_old = np.asarray(s["positions"], dtype=float)
    pbc_old = s.get("pbc", None)

    old_z = infer_z_axis(cell_old, pbc_old)
    inplane_old = [i for i in range(3) if i != old_z]
    inplane_lengths = [float(np.linalg.norm(cell_old[i])) for i in inplane_old]
    high_sym = is_high_symmetry_inplane(data, inplane_lengths, length_rel_tol)

    if high_sym or inplane_lengths[0] >= inplane_lengths[1]:
        new_axes_old_idx = [inplane_old[0], inplane_old[1], old_z]
        action = "keep_inplane_order"
    else:
        new_axes_old_idx = [inplane_old[1], inplane_old[0], old_z]
        action = "swap_xy_by_inplane_length"

    new_axes_signs = [1, 1, 1]
    new_cell = cell_old[new_axes_old_idx, :].copy()
    if float(np.linalg.det(new_cell)) < 0:
        new_axes_signs[1] = -1
        new_cell[1, :] *= -1.0
        action += " + flip_new_y_for_right_handedness"

    frac_old = frac_from_cart(pos_old, cell_old)
    frac_new = frac_old[:, new_axes_old_idx].copy()
    for j, sign in enumerate(new_axes_signs):
        if sign < 0:
            frac_new[:, j] = (-frac_new[:, j]) % 1.0

    centered = False
    if center_slab:
        zf = frac_new[:, 2]
        shift = 0.5 - 0.5 * (float(np.min(zf)) + float(np.max(zf)))
        frac_new[:, 2] = (frac_new[:, 2] + shift) % 1.0
        centered = True

    pos_new = cart_from_frac(frac_new, new_cell)
    old_to_new = {IDX_TO_AXIS[old_i]: new_name for new_name, old_i in zip(AXES, new_axes_old_idx)}

    z_reassigned = old_z != 2
    inplane_reordered = new_axes_old_idx[:2] != [0, 1]
    sign_flipped = new_axes_signs != [1, 1, 1]
    actual_changed = z_reassigned or inplane_reordered or sign_flipped
    high_symmetry_kept = high_sym and action.startswith("keep_inplane_order")

    info = TransformInfo(
        old_z_index=int(old_z),
        new_axes_old_idx=[int(x) for x in new_axes_old_idx],
        new_axes_signs=[int(x) for x in new_axes_signs],
        old_to_new_axis_name=old_to_new,
        high_symmetry_inplane=bool(high_sym),
        inplane_lengths_old=[float(x) for x in inplane_lengths],
        action=action,
        centered_slab=centered,
        z_reassigned=bool(z_reassigned),
        inplane_reordered=bool(inplane_reordered),
        sign_flipped=bool(sign_flipped),
        actual_changed=bool(actual_changed),
        high_symmetry_kept=bool(high_symmetry_kept),
    )
    P = build_signed_permutation_matrix(new_axes_old_idx, new_axes_signs)
    return info, new_cell, pos_new, P


def _force_single_kind(key: str) -> Optional[str]:
    for rx in ODD_SINGLE_AXIS_FORCE_REGEXES:
        if rx.match(key):
            return "odd"
    for rx in EVEN_SINGLE_AXIS_FORCE_REGEXES:
        if rx.match(key):
            return "even"
    return None


def _force_pair(key: str) -> bool:
    return any(rx.match(key) for rx in PAIR_AXIS_FORCE_REGEXES)


def discover_single_axis_groups(keys: Iterable[str]) -> Dict[str, dict]:
    groups: Dict[str, dict] = {}
    for key in keys:
        force_kind = _force_single_kind(key)
        for rx in SINGLE_PATTERNS:
            m = rx.match(key)
            if not m:
                continue
            if len(m.groups()) == 2:
                prefix, axis = m.groups(); suffix = ""
            else:
                prefix, axis, suffix = m.groups()
            template = f"{prefix}{{}}{suffix}"
            entry = groups.setdefault(template, {"keys": {}, "kind_hint": force_kind})
            entry["keys"][axis] = key
            if force_kind is not None:
                entry["kind_hint"] = force_kind
    keep = {}
    for template, entry in groups.items():
        kind_hint = entry["kind_hint"]
        if kind_hint is None:
            t = template.lower()
            if any(k in t for k in EVEN_HINT_KEYWORDS):
                kind_hint = "even"
            elif any(k in t for k in ODD_HINT_KEYWORDS):
                kind_hint = "odd"
        if len(entry["keys"]) >= 2 or kind_hint is not None:
            keep[template] = {"keys": entry["keys"], "kind": kind_hint or "even"}
    return keep


def discover_pair_axis_groups(keys: Iterable[str]) -> Dict[str, dict]:
    groups: Dict[str, dict] = {}
    for key in keys:
        forced = _force_pair(key)
        for rx in PAIR_PATTERNS:
            m = rx.match(key)
            if not m:
                continue
            if len(m.groups()) == 3:
                prefix, a1, a2 = m.groups(); suffix = ""
            else:
                prefix, a1, a2, suffix = m.groups()
            template = f"{prefix}{{}}{{}}{suffix}"
            entry = groups.setdefault(template, {"pairs": {}, "forced": False})
            entry["pairs"][(a1, a2)] = key
            entry["forced"] = entry["forced"] or forced
    return {t: e for t, e in groups.items() if len(e["pairs"]) >= 2 or e["forced"]}


def transform_axis_named_dict_fields(data: dict, info: TransformInfo) -> Tuple[dict, int]:
    out = copy.deepcopy(data)
    old_to_new = info.old_to_new_axis_name
    sign_by_new = {AXES[i]: info.new_axes_signs[i] for i in range(3)}
    changed = 0

    for template, meta in discover_single_axis_groups(list(data.keys())).items():
        values = {old_axis: data[key] for old_axis, key in meta["keys"].items() if key in data}
        for key in meta["keys"].values():
            if key in out:
                out.pop(key); changed += 1
        for old_axis, value in values.items():
            new_axis = old_to_new.get(old_axis, old_axis)
            new_key = template.format(new_axis)
            new_value = value
            if meta["kind"] == "odd" and isinstance(value, (int, float)):
                new_value = sign_by_new[new_axis] * value
            out[new_key] = new_value
            changed += int(new_key != meta["keys"].get(old_axis))

    for template, meta in discover_pair_axis_groups(list(data.keys())).items():
        values = {(a1, a2): data[key] for (a1, a2), key in meta["pairs"].items() if key in data}
        for key in meta["pairs"].values():
            if key in out:
                out.pop(key); changed += 1
        for (old_a1, old_a2), value in values.items():
            out[template.format(old_to_new.get(old_a1, old_a1), old_to_new.get(old_a2, old_a2))] = value
            changed += 1

    for key in list(out.keys()):
        if key in AXIS_LABEL_FIELDS and isinstance(out[key], str) and out[key] in old_to_new:
            old_val = out[key]
            out[key] = old_to_new[old_val]
            changed += int(out[key] != old_val)
    return out, changed


def transform_data_json(data: dict, info: TransformInfo) -> Tuple[dict, int]:
    out, n = transform_axis_named_dict_fields(data, info)
    out["_canonical_axis_transform"] = asdict(info)
    return out, n + 1


def transform_structure_json(structure: dict, new_cell: np.ndarray, new_positions: np.ndarray, info: TransformInfo) -> dict:
    out = copy.deepcopy(structure)
    s = copy.deepcopy(structure["1"])
    s["cell"] = np.asarray(new_cell, dtype=float).tolist()
    s["positions"] = np.asarray(new_positions, dtype=float).tolist()
    if "pbc" in s and isinstance(s["pbc"], list) and len(s["pbc"]) == 3:
        s["pbc"] = [s["pbc"][i] for i in info.new_axes_old_idx]
    s["_canonical_axis_transform"] = asdict(info)
    out["1"] = s
    return out


def transform_ir_json(ir_data: dict, P: np.ndarray, info: TransformInfo) -> Tuple[dict, int]:
    out = copy.deepcopy(ir_data)
    changed = 0
    for key in ("alpha_re_wvv", "alpha_im_wvv"):
        if key in out:
            arr = np.asarray(out[key], dtype=float)
            if arr.ndim != 3 or arr.shape[1:] != (3, 3):
                raise ValueError(f"{key} has unexpected shape {arr.shape}; expected (n_w,3,3).")
            out[key] = transform_rank2_stack(arr, P).tolist(); changed += 1
    for key, value in list(out.items()):
        if key in ("alpha_re_wvv", "alpha_im_wvv"):
            continue
        arr = np.asarray(value)
        if arr.shape == (3, 3) and np.issubdtype(arr.dtype, np.number):
            out[key] = transform_rank2_tensor(arr.astype(float), P).tolist(); changed += 1
    out["_canonical_axis_transform"] = asdict(info)
    return out, changed + 1


def key_has_any(key_path: str, hints: Sequence[str]) -> bool:
    k = key_path.lower()
    return any(h in k for h in hints)


def is_numeric_array_like(x: Any) -> bool:
    try:
        arr = np.asarray(x)
    except Exception:
        return False
    return arr.size > 0 and np.issubdtype(arr.dtype, np.number)


def transform_array_by_context(value: Any, key_path: str, P: np.ndarray) -> Tuple[Any, int, List[str]]:
    notes: List[str] = []
    if not is_numeric_array_like(value) or key_has_any(key_path, DO_NOT_TRANSFORM_ARRAY_HINTS):
        return value, 0, notes
    arr = np.asarray(value, dtype=float)
    shape = arr.shape
    lower = key_path.lower()
    is_stiffness = key_has_any(lower, STIFFNESS_HINTS)
    is_rank2 = key_has_any(lower, RANK2_TENSOR_HINTS)
    is_rank1 = key_has_any(lower, RANK1_HINTS)

    if is_stiffness and shape == (3, 3, 3, 3):
        notes.append(f"{key_path}: transformed rank-4 tensor")
        return transform_rank4_tensor(arr, P).tolist(), 1, notes
    if is_stiffness and len(shape) >= 5 and shape[-4:] == (3, 3, 3, 3):
        notes.append(f"{key_path}: transformed stacked rank-4 tensor {shape}")
        return transform_rank4_stack(arr, P).tolist(), 1, notes
    if is_stiffness and shape == (6, 6):
        notes.append(f"{key_path}: transformed 6x6 Voigt stiffness")
        return transform_voigt6(arr, P).tolist(), 1, notes
    if is_stiffness and len(shape) >= 3 and shape[-2:] == (6, 6):
        out = np.empty_like(arr, dtype=float)
        for i, M in enumerate(arr.reshape((-1, 6, 6))):
            out.reshape((-1, 6, 6))[i] = transform_voigt6(M, P)
        notes.append(f"{key_path}: transformed stacked 6x6 Voigt stiffness {shape}")
        return out.tolist(), 1, notes
    if is_stiffness and shape == (3, 3):
        notes.append(f"{key_path}: transformed 3x3 2D-Voigt stiffness")
        return transform_voigt3_2d(arr, P).tolist(), 1, notes
    if is_rank2 and shape == (3, 3):
        notes.append(f"{key_path}: transformed rank-2 tensor")
        return transform_rank2_tensor(arr, P).tolist(), 1, notes
    if is_rank2 and len(shape) >= 3 and shape[-2:] == (3, 3):
        notes.append(f"{key_path}: transformed stacked rank-2 tensor {shape}")
        return transform_rank2_stack(arr, P).tolist(), 1, notes
    if is_rank1 and shape == (3,):
        notes.append(f"{key_path}: transformed rank-1 vector")
        return (P @ arr).tolist(), 1, notes
    return value, 0, notes


def transform_axis_dict_xyz(value: dict, key_path: str, P: np.ndarray, info: TransformInfo) -> Tuple[dict, int, List[str]]:
    notes: List[str] = []
    if not {"x", "y", "z"}.issubset(set(value.keys())):
        return value, 0, notes
    vals = [value["x"], value["y"], value["z"]]
    if not all(isinstance(v, (int, float)) for v in vals):
        return value, 0, notes
    out = copy.deepcopy(value)
    lower = key_path.lower()
    if key_has_any(lower, ODD_HINT_KEYWORDS) or key_has_any(lower, RANK1_HINTS):
        new_vec = P @ np.array(vals, dtype=float)
        for i, ax in enumerate(AXES):
            out[ax] = float(new_vec[i])
        notes.append(f"{key_path}: transformed xyz dict as vector")
        return out, 1, notes
    if key_has_any(lower, EVEN_HINT_KEYWORDS):
        new_vals = {}
        for old_axis in AXES:
            new_vals[info.old_to_new_axis_name.get(old_axis, old_axis)] = value[old_axis]
        for ax in AXES:
            out[ax] = new_vals[ax]
        notes.append(f"{key_path}: remapped xyz dict as even diagonal components")
        return out, 1, notes
    return value, 0, notes


def transform_generic_json(obj: Any, P: np.ndarray, info: TransformInfo, key_path: str = "") -> Tuple[Any, int, List[str]]:
    if isinstance(obj, dict):
        out = {}
        total = 0
        notes: List[str] = []
        for k, v in obj.items():
            child_path = f"{key_path}.{k}" if key_path else str(k)
            vn, n, ns = transform_generic_json(v, P, info, child_path)
            out[k] = vn; total += n; notes.extend(ns)
        out, n2, ns2 = transform_axis_dict_xyz(out, key_path, P, info)
        total += n2; notes.extend(ns2)
        out, n3 = transform_axis_named_dict_fields(out, info)
        if n3:
            total += n3; notes.append(f"{key_path or '<root>'}: remapped axis-named fields")
        return out, total, notes
    if isinstance(obj, list):
        arr_new, n_arr, arr_notes = transform_array_by_context(obj, key_path, P)
        if n_arr:
            return arr_new, n_arr, arr_notes
        out_list = []
        total = 0
        notes: List[str] = []
        for i, item in enumerate(obj):
            item_new, n, ns = transform_generic_json(item, P, info, f"{key_path}[{i}]")
            out_list.append(item_new); total += n; notes.extend(ns)
        return out_list, total, notes
    return obj, 0, []


def process_json_file(path: Path, out_path: Path, structure: dict, info: TransformInfo, new_cell: np.ndarray, new_positions: np.ndarray, P: np.ndarray) -> Dict[str, Any]:
    rec: Dict[str, Any] = {"file": path.name, "input_path": str(path), "output_path": str(out_path), "mode": None, "changed": False, "notes": []}
    if path.name == "data.json":
        out_obj, _ = transform_data_json(read_json(path), info)
        write_json(out_obj, out_path)
        rec.update(mode="dedicated_data_json", changed=True, notes=["transformed data.json axis-dependent fields"])
        return rec
    if path.name == "structure.json":
        write_json(transform_structure_json(structure, new_cell, new_positions, info), out_path)
        rec.update(mode="dedicated_structure_json", changed=True, notes=["transformed cell/positions/pbc"])
        return rec
    if path.name == "ir-polarizability.json":
        out_obj, _ = transform_ir_json(read_json(path), P, info)
        write_json(out_obj, out_path)
        rec.update(mode="dedicated_ir_json", changed=True, notes=["transformed IR polarizability tensors"])
        return rec

    obj = read_json(path)
    out_obj, n, notes = transform_generic_json(obj, P, info, key_path=path.name)
    if n:
        if isinstance(out_obj, dict):
            out_obj["_canonical_axis_transform"] = asdict(info)
            out_obj["_canonical_axis_generic_notes"] = notes[:200]
        write_json(out_obj, out_path)
        rec.update(mode="generic_json_transformed", changed=True, notes=notes[:200])
    else:
        safe_copy(path, out_path)
        rec.update(mode="copied_json_no_axis_transform", changed=False, notes=["no recognized axis-dependent content; copied unchanged"])
    return rec


def process_sample_dir(sample_dir: Path, outdir: Path, center_slab: bool = True, length_rel_tol: float = 1e-3, include_hidden: bool = False) -> Dict[str, Any]:
    data_path = sample_dir / "data.json"
    structure_path = sample_dir / "structure.json"
    if not data_path.exists() or not structure_path.exists():
        raise FileNotFoundError(f"Sample dir must contain data.json and structure.json: {sample_dir}")
    data = read_json(data_path)
    structure = read_json(structure_path)
    info, new_cell, new_positions, P = choose_canonical_transform(data, structure, center_slab, length_rel_tol)

    outdir.mkdir(parents=True, exist_ok=True)
    entries = sorted(p for p in sample_dir.iterdir() if p.is_file())
    if not include_hidden:
        entries = [p for p in entries if not p.name.startswith(".")]

    file_records = []
    for path in entries:
        out_path = outdir / path.name
        try:
            if path.suffix.lower() == ".json":
                rec = process_json_file(path, out_path, structure, info, new_cell, new_positions, P)
            else:
                safe_copy(path, out_path)
                rec = {"file": path.name, "input_path": str(path), "output_path": str(out_path), "mode": "copied_non_json", "changed": False, "notes": ["non-JSON file copied unchanged"]}
        except Exception as e:
            write_json({"input_path": str(path), "error": repr(e)}, outdir / f"{path.name}.canonicalization_error.json")
            rec = {"file": path.name, "input_path": str(path), "output_path": str(out_path), "mode": "error", "changed": False, "notes": [repr(e)]}
        file_records.append(rec)

    transformed = [r["file"] for r in file_records if r.get("changed")]
    copied = [r["file"] for r in file_records if not r.get("changed") and r.get("mode") != "error"]
    errors = [r["file"] for r in file_records if r.get("mode") == "error"]
    metadata = {
        "sample_dir": str(sample_dir),
        "output_dir": str(outdir),
        "transform": asdict(info),
        "P": P.tolist(),
        "n_files_seen": len(entries),
        "n_files_transformed": len(transformed),
        "n_files_copied": len(copied),
        "n_files_error": len(errors),
        "transformed_files": transformed,
        "copied_files": copied,
        "error_files": errors,
        "files": file_records,
    }
    write_json(metadata, outdir / "canonical_transform_metadata.json")
    return metadata


def find_sample_dirs(root: Path, output_root: Optional[Path] = None) -> List[Path]:
    out_resolved = output_root.resolve() if output_root is not None else None
    sample_dirs = []
    for p in sorted(root.rglob("*")):
        if not p.is_dir():
            continue
        if out_resolved is not None and path_is_relative_to(p, out_resolved):
            continue
        if (p / "data.json").exists() and (p / "structure.json").exists():
            sample_dirs.append(p)
    return sample_dirs


def write_global_manifest(records: List[Dict[str, Any]], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(records, output_root / "canonicalization_manifest.json")
    with open(output_root / "canonicalization_manifest.csv", "w", encoding="utf-8", newline="") as f:
        fields = ["sample_dir", "output_dir", "actual_changed", "z_reassigned", "inplane_reordered", "sign_flipped", "high_symmetry_inplane", "high_symmetry_kept", "n_files_seen", "n_files_transformed", "n_files_copied", "n_files_error", "transformed_files", "copied_files", "error_files"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            t = r.get("transform", {})
            writer.writerow({
                "sample_dir": r.get("sample_dir"),
                "output_dir": r.get("output_dir"),
                "actual_changed": t.get("actual_changed"),
                "z_reassigned": t.get("z_reassigned"),
                "inplane_reordered": t.get("inplane_reordered"),
                "sign_flipped": t.get("sign_flipped"),
                "high_symmetry_inplane": t.get("high_symmetry_inplane"),
                "high_symmetry_kept": t.get("high_symmetry_kept"),
                "n_files_seen": r.get("n_files_seen"),
                "n_files_transformed": r.get("n_files_transformed"),
                "n_files_copied": r.get("n_files_copied"),
                "n_files_error": r.get("n_files_error"),
                "transformed_files": ";".join(r.get("transformed_files", [])),
                "copied_files": ";".join(r.get("copied_files", [])),
                "error_files": ";".join(r.get("error_files", [])),
            })


def process_tree(root: Path, output_root: Path, center_slab: bool = True, length_rel_tol: float = 1e-3, include_hidden: bool = False) -> None:
    sample_dirs = find_sample_dirs(root, output_root)
    records = []
    processed = 0
    skipped = 0
    for sample_dir in sample_dirs:
        outdir = output_root / sample_dir.relative_to(root)
        try:
            records.append(process_sample_dir(sample_dir, outdir, center_slab, length_rel_tol, include_hidden))
            processed += 1
        except Exception as e:
            skipped += 1
            outdir.mkdir(parents=True, exist_ok=True)
            write_json({"sample_dir": str(sample_dir), "error": repr(e)}, outdir / "canonicalization_error.json")
            records.append({"sample_dir": str(sample_dir), "output_dir": str(outdir), "transform": {}, "n_files_seen": 0, "n_files_transformed": 0, "n_files_copied": 0, "n_files_error": 1, "transformed_files": [], "copied_files": [], "error_files": ["<sample>"], "error": repr(e)})
    write_global_manifest(records, output_root)
    print(f"sample_dirs_found={len(sample_dirs)}")
    print(f"processed={processed}, skipped={skipped}")
    print(f"samples_with_actual_axis_change={sum(bool(r.get('transform', {}).get('actual_changed')) for r in records)}")
    print(f"samples_high_symmetry_kept={sum(bool(r.get('transform', {}).get('high_symmetry_kept')) for r in records)}")
    print(f"files_transformed={sum(int(r.get('n_files_transformed', 0)) for r in records)}, files_copied={sum(int(r.get('n_files_copied', 0)) for r in records)}")
    print(f"manifest={output_root / 'canonicalization_manifest.csv'}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Full-scan canonical axis preprocessing for C2DB-style samples.")
    p.add_argument("--data", type=Path, help="Path to data.json for single-sample mode")
    p.add_argument("--structure", type=Path, help="Path to structure.json for single-sample mode")
    p.add_argument("--outdir", type=Path, default=Path("canonicalized_sample"))
    p.add_argument("--root", type=Path, help="Root directory for batch mode")
    p.add_argument("--output-root", type=Path, default=Path("canonicalized_tree"))
    p.add_argument("--batch", action="store_true")
    p.add_argument("--no-center-slab", action="store_true")
    p.add_argument("--length-rel-tol", type=float, default=1e-3)
    p.add_argument("--include-hidden", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    center_slab = not args.no_center_slab
    if args.batch:
        if args.root is None:
            raise SystemExit("--batch requires --root")
        process_tree(args.root, args.output_root, center_slab, args.length_rel_tol, args.include_hidden)
        return
    if args.data is None or args.structure is None:
        raise SystemExit("single-sample mode requires --data and --structure")
    process_sample_dir(args.data.parent, args.outdir, center_slab, args.length_rel_tol, args.include_hidden)
    print(f"Wrote canonicalized files to: {args.outdir}")


if __name__ == "__main__":
    main()
