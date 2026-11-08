#!/usr/bin/env python
"""Prepare local polarizability data for ALIGNN training.

The training entry point in alignn/train_alignn.py expects a folder with:
  - id_prop.csv: rows of "structure_file,target_1,target_2,..."
  - one POSCAR-like structure file per row
  - config.json

This script converts the local pymatgen-style structures.json and the label
CSV into that layout without requiring pymatgen or jarvis at preparation time.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path


DEFAULT_LABELS = "polarizability_cleaned_alpha_inplane_le_p95.csv"
DEFAULT_STRUCTURES = "structures.json"
DEFAULT_DESCRIPTORS = "aux_physical_priors.csv"
DEFAULT_OUT_DIR = "data/polarizability_alignn"
TENSOR_COLUMNS = ["xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"]
ALPHA_TENSOR_COLUMN_MAP = {
    f"alpha_{column}": column for column in TENSOR_COLUMNS
}
INPLANE_CLEANED_COLUMN_MAP = {
    "alpha_xx": "xx",
    "alpha_yy": "yy",
    "alpha_xy_sym": "xy",
}
ZZ_LOG1P_SCALE = 0.033699
INPLANE_TRACE_FLOOR = 1e-12
DEFAULT_INPLANE_ETA_BINS = [0.01, 0.05, 0.1, 0.2, 0.5, 0.8]
DEFAULT_INPLANE_ALPHA_BIN_QUANTILES = [0.50, 0.75, 0.85, 0.95]
TARGET_SPECS = {
    "eta_parallel": [("eta_parallel", "identity")],
    "eta_parallel_multitask": [
        ("eta_parallel", "identity"),
        ("a_diag", "identity"),
        ("a_xy", "identity"),
    ],
    "inplane_transformed": [
        ("xx", "log1p"),
        ("yy", "log1p"),
        ("xy", "asinh"),
        ("eta_parallel", "asinh"),
        ("inplane_trace_raw", "log1p_floor"),
    ],
    "outplane_transformed": [
        ("zz", "scaled_log1p_zz"),
        ("xz", "asinh"),
        ("yz", "asinh"),
    ],
    "zz_transformed": [("zz", "scaled_log1p_zz")],
    "xz_yz_transformed": [("xz", "asinh"), ("yz", "asinh")],
    "six_transformed": [
        ("xx", "log1p"),
        ("yy", "log1p"),
        ("zz", "scaled_log1p_zz"),
        ("xy", "asinh"),
        ("xz", "asinh"),
        ("yz", "asinh"),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare polarizability labels and structures for ALIGNN."
    )
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--structures", default=DEFAULT_STRUCTURES)
    parser.add_argument("--descriptors", default=DEFAULT_DESCRIPTORS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--target",
        choices=[
            "full_tensor",
            "alpha_inplane",
            "eta_parallel",
            "eta_parallel_multitask",
            "columns",
            "inplane_transformed",
            "outplane_transformed",
            "zz_transformed",
            "xz_yz_transformed",
            "six_transformed",
        ],
        default="full_tensor",
        help=(
            "full_tensor writes xx,xy,xz,yx,yy,yz,zx,zy,zz; "
            "alpha_inplane writes (xx+yy)/2; columns uses --columns; "
            "eta_parallel writes sqrt((xx-yy)^2+4xy^2)/(xx+yy); "
            "inplane_transformed writes log1p(xx),log1p(yy),asinh(xy),"
            "asinh(eta_parallel),log1p(max(xx+yy,floor)); "
            "outplane_transformed writes log1p(zz/s),asinh(xz),asinh(yz); "
            "zz_transformed writes log1p(zz/s); "
            "xz_yz_transformed writes asinh(xz),asinh(yz)."
        ),
    )
    parser.add_argument(
        "--columns",
        default="",
        help="Comma-separated label columns used when --target columns.",
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--tail-weight-enabled", action="store_true")
    parser.add_argument(
        "--tail-weight-score",
        choices=["alpha_parallel", "zz", "trace", "eta_parallel"],
        default="alpha_parallel",
    )
    parser.add_argument("--tail-weight-quantile", type=float, default=0.75)
    parser.add_argument("--tail-weight-lambda", type=float, default=1.65)
    parser.add_argument("--tail-weight-power", type=float, default=0.3)
    parser.add_argument(
        "--xy-tail-extra-weight-enabled",
        action="store_true",
        help=(
            "Apply a small extra sample-weight multiplier only to "
            "cluster_b high-eta in-plane anisotropy samples."
        ),
    )
    parser.add_argument("--xy-tail-extra-eta-threshold", type=float, default=0.8)
    parser.add_argument("--xy-tail-extra-multiplier", type=float, default=1.15)
    parser.add_argument("--xy-tail-extra-weight-cap", type=float, default=2.0)
    parser.add_argument("--eta-pairwise-loss-enabled", action="store_true")
    parser.add_argument("--eta-pairwise-loss-target", default="eta_parallel")
    parser.add_argument("--eta-pairwise-loss-lambda", type=float, default=0.02)
    parser.add_argument("--eta-pairwise-loss-min-delta", type=float, default=0.5)
    parser.add_argument("--eta-pairwise-loss-max-pairs", type=int, default=512)
    parser.add_argument("--eta-pairwise-loss-use-abs-delta", action="store_true")
    parser.add_argument(
        "--eta-pairwise-loss-sampler-enabled",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--eta-pairwise-loss-batch-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--eta-pairwise-loss-pairs-per-batch", type=int, default=2)
    parser.add_argument("--eta-pairwise-loss-buffer-max-pairs", type=int, default=2048)
    parser.add_argument("--eta-embedding-contrastive-loss-enabled", action="store_true")
    parser.add_argument(
        "--eta-embedding-contrastive-loss-target",
        default="eta_parallel",
    )
    parser.add_argument(
        "--eta-embedding-contrastive-loss-lambda",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--eta-embedding-contrastive-loss-min-delta",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--eta-embedding-contrastive-loss-margin",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--eta-embedding-contrastive-loss-max-pairs",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--eta-embedding-contrastive-loss-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--eta-embedding-contrastive-pull-enabled", action="store_true")
    parser.add_argument(
        "--eta-embedding-contrastive-pull-lambda",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--eta-embedding-contrastive-pull-max-delta",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--eta-embedding-contrastive-pull-max-pairs",
        type=int,
        default=512,
    )
    parser.add_argument("--stratify-split", action="store_true")
    parser.add_argument(
        "--stratify-mode",
        choices=[
            "fixed_physics_cluster",
            "fixed_physics_cluster_alpha_bins",
            "inplane_eta_cluster",
            "inplane_eta_cluster_bins",
            "inplane_eta_cluster_bins_xy_tail",
            "zz_value_bins",
        ],
        default="fixed_physics_cluster",
    )
    parser.add_argument(
        "--stratify-eta-bins",
        default=",".join(str(value) for value in DEFAULT_INPLANE_ETA_BINS),
        help=(
            "Comma-separated eta_parallel cut points used only by "
            "inplane_eta_cluster_bins modes."
        ),
    )
    parser.add_argument(
        "--stratify-alpha-bins",
        default=",".join(
            str(value) for value in DEFAULT_INPLANE_ALPHA_BIN_QUANTILES
        ),
        help=(
            "Comma-separated alpha_parallel rank quantile cut points used by "
            "fixed_physics_cluster_alpha_bins."
        ),
    )
    parser.add_argument("--stratify-trace-floor", type=float, default=1e-3)
    parser.add_argument("--stratify-xy-coupling-threshold", type=float, default=0.30)
    parser.add_argument("--stratify-diag-anisotropy-threshold", type=float, default=0.30)
    parser.add_argument("--stratify-min-count", type=int, default=2)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=123)
    return parser.parse_args()


def load_labels(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in labels file: {path}")
    if not set(TENSOR_COLUMNS).issubset(rows[0]) and set(
        ALPHA_TENSOR_COLUMN_MAP
    ).issubset(rows[0]):
        rows = [
            {
                **row,
                **{
                    target: row[source]
                    for source, target in ALPHA_TENSOR_COLUMN_MAP.items()
                },
            }
            for row in rows
        ]
    if not set(TENSOR_COLUMNS).issubset(rows[0]) and set(
        INPLANE_CLEANED_COLUMN_MAP
    ).issubset(rows[0]):
        rows = [
            {
                **row,
                **{
                    "xx": row["alpha_xx"],
                    "xy": row["alpha_xy_sym"],
                    "xz": row.get("alpha_xz", "0.0"),
                    "yx": row.get("alpha_yx", row["alpha_xy_sym"]),
                    "yy": row["alpha_yy"],
                    "yz": row.get("alpha_yz", "0.0"),
                    "zx": row.get("alpha_zx", "0.0"),
                    "zy": row.get("alpha_zy", "0.0"),
                    "zz": row.get("alpha_zz", "0.0"),
                },
            }
            for row in rows
        ]
    missing = {"uid", *TENSOR_COLUMNS} - set(rows[0])
    if missing:
        raise ValueError(f"Labels file is missing columns: {sorted(missing)}")
    return rows


def load_structures(path: Path) -> dict[str, dict]:
    with path.open() as f:
        rows = json.load(f)
    return {str(row["uid"]): row["structure"] for row in rows}


def target_value(row: dict[str, str], column: str) -> float:
    if column == "eta_parallel":
        xx = float(row["xx"])
        yy = float(row["yy"])
        xy = float(row["xy"])
        eps = 1e-12
        return math.sqrt((xx - yy) ** 2 + 4.0 * xy**2) / (xx + yy + eps)
    if column == "a_diag":
        xx = float(row["xx"])
        yy = float(row["yy"])
        eps = 1e-12
        return abs(xx - yy) / (xx + yy + eps)
    if column == "a_xy":
        xx = float(row["xx"])
        yy = float(row["yy"])
        xy = float(row["xy"])
        eps = 1e-12
        return 2.0 * abs(xy) / (xx + yy + eps)
    if column == "inplane_trace_raw":
        return float(row["xx"]) + float(row["yy"])
    return float(row[column])


def selected_targets(row: dict[str, str], target: str, columns: list[str]) -> list[float]:
    if target in TARGET_SPECS:
        values = []
        for column, transform in TARGET_SPECS[target]:
            value = target_value(row, column)
            if transform == "log1p":
                if value <= -1.0:
                    raise ValueError(
                        f"Cannot apply log1p to {column}={value} for {row['uid']}"
                    )
                values.append(math.log1p(value))
            elif transform == "identity":
                values.append(value)
            elif transform == "log1p_floor":
                values.append(math.log1p(max(value, INPLANE_TRACE_FLOOR)))
            elif transform == "scaled_log1p_zz":
                scaled = value / ZZ_LOG1P_SCALE
                if scaled <= -1.0:
                    raise ValueError(
                        f"Cannot apply log1p to {column}/s={scaled} for {row['uid']}"
                    )
                values.append(math.log1p(scaled))
            elif transform == "asinh":
                values.append(math.asinh(value))
            else:
                raise ValueError(f"Unknown transform: {transform}")
        return values
    if target == "full_tensor":
        return [float(row[col]) for col in TENSOR_COLUMNS]
    if target == "alpha_inplane":
        return [(float(row["xx"]) + float(row["yy"])) / 2.0]
    return [float(row[col]) for col in columns]


def selected_target_columns(target: str, columns: list[str]) -> list[str]:
    if target in TARGET_SPECS:
        return [column for column, _ in TARGET_SPECS[target]]
    if target == "full_tensor":
        return TENSOR_COLUMNS
    if target == "alpha_inplane":
        return ["alpha_inplane"]
    return columns


def compute_tail_weights(
    labels: list[dict[str, str]],
    score_name: str,
    quantile: float,
    weight_lambda: float,
    power: float,
) -> dict[str, dict[str, float]]:
    if not 0 <= quantile < 1:
        raise ValueError("--tail-weight-quantile must be >= 0 and < 1.")
    if power <= 0:
        raise ValueError("--tail-weight-power must be > 0.")

    scored = []
    for row in labels:
        if score_name == "alpha_parallel":
            score = 0.5 * (float(row["xx"]) + float(row["yy"]))
        elif score_name == "zz":
            score = float(row["zz"])
        elif score_name == "trace":
            score = float(row["xx"]) + float(row["yy"]) + float(row["zz"])
        elif score_name == "eta_parallel":
            score = target_value(row, "eta_parallel")
        else:
            raise ValueError(f"Unknown tail weight score: {score_name}")
        scored.append((row["uid"], score))

    sorted_scores = sorted(scored, key=lambda item: item[1])
    n = len(sorted_scores)
    weights = {}
    for idx, (uid, score) in enumerate(sorted_scores):
        rank = 1.0 if n == 1 else idx / (n - 1)
        ramp = max(0.0, (rank - quantile) / (1.0 - quantile))
        weight = 1.0 + weight_lambda * ramp**power
        weights[uid] = {
            "score": score,
            "rank": rank,
            "ramp": ramp,
            "weight": weight,
        }
    return weights


def fixed_physics_cluster(row: dict[str, str], args: argparse.Namespace) -> str:
    xx = float(row["xx"])
    yy = float(row["yy"])
    zz = float(row["zz"])
    xy = float(row["xy"])
    eps = 1e-12
    trace = xx + yy + zz
    a_diag = abs(xx - yy) / (xx + yy + eps)
    a_xy = 2.0 * abs(xy) / (xx + yy + eps)

    if trace < args.stratify_trace_floor:
        return "cluster_d_near_zero_pathological"
    if a_xy > args.stratify_xy_coupling_threshold:
        return "cluster_b_xy_coupled_anisotropic"
    if a_diag > args.stratify_diag_anisotropy_threshold:
        return "cluster_c_diag_anisotropic"
    return "cluster_a_inplane_isotropic"


def inplane_eta_cluster(row: dict[str, str], args: argparse.Namespace) -> str:
    xx = float(row["xx"])
    yy = float(row["yy"])
    zz = float(row["zz"])
    xy = float(row["xy"])
    eps = 1e-12
    trace = xx + yy + zz
    denom = xx + yy + eps
    a_xy = 2.0 * abs(xy) / denom
    a_diag = abs(xx - yy) / denom

    if trace >= args.stratify_trace_floor and a_xy > args.stratify_xy_coupling_threshold:
        return "cluster_b_xy_coupled_anisotropic"
    if trace >= args.stratify_trace_floor and a_diag > args.stratify_diag_anisotropy_threshold:
        return "cluster_c_diag_anisotropic"
    return "cluster_a_inplane_isotropic"


def apply_xy_tail_extra_weights(
    *,
    labels: list[dict[str, str]],
    weights: dict[str, dict[str, float]],
    args: argparse.Namespace,
) -> dict[str, float | int | bool | None]:
    """Boost only xy-coupled high-eta samples while capping final weight."""
    enabled = bool(getattr(args, "xy_tail_extra_weight_enabled", False))
    threshold = float(getattr(args, "xy_tail_extra_eta_threshold", 0.8))
    multiplier = float(getattr(args, "xy_tail_extra_multiplier", 1.15))
    cap = getattr(args, "xy_tail_extra_weight_cap", 2.0)
    cap_value = None if cap is None else float(cap)

    stats: dict[str, float | int | bool | None] = {
        "enabled": enabled,
        "eta_threshold": threshold,
        "multiplier": multiplier,
        "weight_cap": cap_value,
        "applied_count": 0,
        "max_base_weight": None,
        "max_uncapped_weight": None,
        "max_final_weight": None,
    }
    if not enabled:
        return stats
    if not weights:
        raise ValueError(
            "--xy-tail-extra-weight-enabled requires --tail-weight-enabled."
        )
    if not 0 <= threshold <= 1:
        raise ValueError("--xy-tail-extra-eta-threshold must be in [0, 1].")
    if multiplier <= 0:
        raise ValueError("--xy-tail-extra-multiplier must be > 0.")
    if cap_value is not None and cap_value <= 0:
        raise ValueError("--xy-tail-extra-weight-cap must be > 0.")

    base_weights: list[float] = []
    uncapped_weights: list[float] = []
    final_weights: list[float] = []
    for row in labels:
        uid = row["uid"]
        if uid not in weights:
            continue
        eta = target_value(row, "eta_parallel")
        cluster = inplane_eta_cluster(row, args)
        if cluster != "cluster_b_xy_coupled_anisotropic" or eta < threshold:
            continue
        values = weights[uid]
        base_weight = float(values["weight"])
        uncapped_weight = base_weight * multiplier
        final_weight = (
            min(uncapped_weight, cap_value)
            if cap_value is not None
            else uncapped_weight
        )
        values["base_weight"] = base_weight
        values["xy_tail_extra_applied"] = 1.0
        values["xy_tail_extra_multiplier"] = multiplier
        values["xy_tail_extra_uncapped_weight"] = uncapped_weight
        values["weight"] = final_weight
        base_weights.append(base_weight)
        uncapped_weights.append(uncapped_weight)
        final_weights.append(final_weight)

    stats["applied_count"] = len(final_weights)
    if final_weights:
        stats["max_base_weight"] = max(base_weights)
        stats["max_uncapped_weight"] = max(uncapped_weights)
        stats["max_final_weight"] = max(final_weights)
    return stats


def parse_unit_bins(
    value: object,
    default: list[float],
    name: str,
) -> list[float]:
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple)):
        bins = [float(item) for item in value]
    else:
        bins = [
            float(part.strip())
            for part in str(value).split(",")
            if part.strip()
        ]
    if not bins:
        return list(default)
    if any(not 0 < item < 1 for item in bins):
        raise ValueError(f"{name} bins must be inside (0, 1).")
    if any(left >= right for left, right in zip(bins, bins[1:])):
        raise ValueError(f"{name} bins must be strictly increasing.")
    return bins


def parse_float_bins(value: object) -> list[float]:
    return parse_unit_bins(value, DEFAULT_INPLANE_ETA_BINS, "stratify eta")


def parse_alpha_bins(value: object) -> list[float]:
    return parse_unit_bins(
        value,
        DEFAULT_INPLANE_ALPHA_BIN_QUANTILES,
        "stratify alpha",
    )


def number_tag(value: float) -> str:
    text = f"{value:g}".replace("-", "neg").replace(".", "p")
    return text


def unit_bin_label(prefix: str, value: float, bins: list[float]) -> str:
    edges = [0.0, *bins, 1.0]
    upper_label = "1p0"
    for idx, upper in enumerate(bins):
        lower = edges[idx]
        if value < upper:
            return (
                f"{prefix}_bin_{idx}_"
                f"{number_tag(lower)}_to_{number_tag(upper)}"
            )
    return f"{prefix}_bin_{len(bins)}_{number_tag(bins[-1])}_to_{upper_label}"


def eta_bin_label(value: float, bins: list[float]) -> str:
    return unit_bin_label("eta", value, bins)


def inplane_eta_cluster_bin(row: dict[str, str], args: argparse.Namespace) -> str:
    cluster = inplane_eta_cluster(row, args)
    bins = parse_float_bins(getattr(args, "stratify_eta_bins", None))
    eta = target_value(row, "eta_parallel")
    return f"{cluster}__{eta_bin_label(eta, bins)}"


def inplane_eta_cluster_bin_xy_tail(
    row: dict[str, str], args: argparse.Namespace
) -> str:
    """Merge sparse high-eta xy-coupled bins into one stable stratification bin."""
    cluster = inplane_eta_cluster(row, args)
    eta = target_value(row, "eta_parallel")
    if cluster == "cluster_b_xy_coupled_anisotropic" and eta >= 0.8:
        return f"{cluster}__eta_bin_7_0p8_to_1p0"
    return inplane_eta_cluster_bin(row, args)


def alpha_parallel_bin_labels(
    labels: list[dict[str, str]], args: argparse.Namespace
) -> dict[str, str]:
    bins = parse_alpha_bins(getattr(args, "stratify_alpha_bins", None))
    scored = sorted(
        (
            (
                row["uid"],
                0.5 * (float(row["xx"]) + float(row["yy"])),
            )
            for row in labels
        ),
        key=lambda item: (item[1], item[0]),
    )
    n = len(scored)
    labels_by_uid: dict[str, str] = {}
    for idx, (uid, _) in enumerate(scored):
        rank = 0.0 if n == 1 else idx / (n - 1)
        labels_by_uid[uid] = unit_bin_label("alpha_parallel", rank, bins)
    return labels_by_uid


def zz_value_bin(row: dict[str, str]) -> str:
    zz = float(row["zz"])
    if zz <= 0.01:
        return "zz_bin_0_le_0p01"
    if zz <= 0.05:
        return "zz_bin_1_0p01_to_0p05"
    if zz <= 0.1:
        return "zz_bin_2_0p05_to_0p1"
    if zz <= 0.2:
        return "zz_bin_3_0p1_to_0p2"
    return "zz_bin_4_gt_0p2"


def stratification_labels(
    labels: list[dict[str, str]], args: argparse.Namespace
) -> dict[str, str]:
    mode = getattr(args, "stratify_mode", "fixed_physics_cluster")
    fallback_by_uid: dict[str, str] | None = None
    if mode == "fixed_physics_cluster":
        raw = {row["uid"]: fixed_physics_cluster(row, args) for row in labels}
        fallback_label = "cluster_a_inplane_isotropic"
    elif mode == "fixed_physics_cluster_alpha_bins":
        physics = {
            row["uid"]: fixed_physics_cluster(row, args)
            for row in labels
        }
        alpha_bins = alpha_parallel_bin_labels(labels, args)
        raw = {
            uid: f"{physics[uid]}__{alpha_bins[uid]}"
            for uid in physics
        }
        fallback_by_uid = physics
        fallback_label = "cluster_a_inplane_isotropic"
    elif mode == "inplane_eta_cluster":
        raw = {row["uid"]: inplane_eta_cluster(row, args) for row in labels}
        fallback_label = "cluster_a_inplane_isotropic"
    elif mode == "inplane_eta_cluster_bins":
        raw = {row["uid"]: inplane_eta_cluster_bin(row, args) for row in labels}
        fallback_by_uid = {row["uid"]: inplane_eta_cluster(row, args) for row in labels}
        fallback_label = "cluster_a_inplane_isotropic"
    elif mode == "inplane_eta_cluster_bins_xy_tail":
        raw = {
            row["uid"]: inplane_eta_cluster_bin_xy_tail(row, args)
            for row in labels
        }
        fallback_by_uid = {row["uid"]: inplane_eta_cluster(row, args) for row in labels}
        fallback_label = "cluster_a_inplane_isotropic"
    elif mode == "zz_value_bins":
        raw = {row["uid"]: zz_value_bin(row) for row in labels}
        fallback_label = "zz_bin_4_gt_0p2"
    else:
        raise ValueError(f"Unknown stratify mode: {mode}")
    counts = Counter(raw.values())
    mapped = {}
    for uid, label in raw.items():
        if counts[label] < args.stratify_min_count:
            mapped[uid] = (
                fallback_by_uid[uid] if fallback_by_uid is not None else fallback_label
            )
        else:
            mapped[uid] = label
    return mapped


def split_ids(
    rows: list[dict[str, str]],
    labels_by_uid: dict[str, str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    if train_ratio + val_ratio + test_ratio > 1.0 + 1e-12:
        raise ValueError("train_ratio + val_ratio + test_ratio must be <= 1.")

    rng = random.Random(seed)
    groups: dict[str, list[str]] = {}
    for row in rows:
        groups.setdefault(labels_by_uid[row["uid"]], []).append(row["uid"])

    id_train: list[str] = []
    id_val: list[str] = []
    id_test: list[str] = []
    for group_ids in groups.values():
        ids = list(group_ids)
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
        id_train.extend(ids[:n_train])
        id_val.extend(ids[n_train : n_train + n_val])
        id_test.extend(ids[n_train + n_val :])

    rng.shuffle(id_train)
    rng.shuffle(id_val)
    rng.shuffle(id_test)
    return {"id_train": id_train, "id_val": id_val, "id_test": id_test}


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


def build_config(
    out_dir: Path,
    descriptors: Path,
    output_features: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    tail_weight_enabled: bool,
    tail_weight_score: str,
    sample_weight_csv: Path,
    tail_weight_quantile: float,
    tail_weight_lambda: float,
    tail_weight_power: float,
    xy_tail_extra_weight_enabled: bool,
    xy_tail_extra_eta_threshold: float,
    xy_tail_extra_multiplier: float,
    xy_tail_extra_weight_cap: float,
    eta_pairwise_loss_enabled: bool,
    eta_pairwise_loss_target: str,
    eta_pairwise_loss_lambda: float,
    eta_pairwise_loss_min_delta: float,
    eta_pairwise_loss_max_pairs: int,
    eta_pairwise_loss_use_abs_delta: bool,
    eta_pairwise_loss_sampler_enabled: bool,
    eta_pairwise_loss_batch_enabled: bool,
    eta_pairwise_loss_pairs_per_batch: int,
    eta_pairwise_loss_buffer_max_pairs: int,
    eta_embedding_contrastive_loss_enabled: bool,
    eta_embedding_contrastive_loss_target: str,
    eta_embedding_contrastive_loss_lambda: float,
    eta_embedding_contrastive_loss_min_delta: float,
    eta_embedding_contrastive_loss_margin: float,
    eta_embedding_contrastive_loss_max_pairs: int,
    eta_embedding_contrastive_loss_normalize: bool,
    eta_embedding_contrastive_pull_enabled: bool,
    eta_embedding_contrastive_pull_lambda: float,
    eta_embedding_contrastive_pull_max_delta: float,
    eta_embedding_contrastive_pull_max_pairs: int,
    target_columns: list[str],
    target_transforms: list[dict[str, str]],
) -> dict:
    run_dir = Path("runs") / out_dir.name
    return {
        "dataset": "user_data",
        "target": "target",
        "atom_features": "cgcnn",
        "neighbor_strategy": "k-nearest",
        "id_tag": "jid",
        "random_seed": 123,
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "epochs": epochs,
        "batch_size": batch_size,
        "weight_decay": 1e-5,
        "learning_rate": learning_rate,
        "criterion": "mse",
        "optimizer": "adamw",
        "scheduler": "onecycle",
        "n_early_stopping": 50,
        "num_workers": 0,
        "pin_memory": False,
        "write_checkpoint": True,
        "write_predictions": True,
        "store_outputs": True,
        "progress": True,
        "compute_line_graph": True,
        "cutoff": 8.0,
        "max_neighbors": 12,
        "keep_data_order": False,
        "use_lmdb": False,
        "output_dir": str(run_dir),
        "filename": "polarizability",
        "use_sample_weights": tail_weight_enabled,
        "sample_weight_csv": str(sample_weight_csv.resolve()) if tail_weight_enabled else None,
        "sample_weight_id_key": "uid",
        "sample_weight_value_key": "weight",
        "tail_weight_enabled": tail_weight_enabled,
        "tail_weight_score": tail_weight_score,
        "tail_weight_quantile": tail_weight_quantile,
        "tail_weight_lambda": tail_weight_lambda,
        "tail_weight_power": tail_weight_power,
        "xy_tail_extra_weight_enabled": xy_tail_extra_weight_enabled,
        "xy_tail_extra_eta_threshold": xy_tail_extra_eta_threshold,
        "xy_tail_extra_multiplier": xy_tail_extra_multiplier,
        "xy_tail_extra_weight_cap": xy_tail_extra_weight_cap,
        "eta_pairwise_loss_enabled": eta_pairwise_loss_enabled,
        "eta_pairwise_loss_target": eta_pairwise_loss_target,
        "eta_pairwise_loss_lambda": eta_pairwise_loss_lambda,
        "eta_pairwise_loss_min_delta": eta_pairwise_loss_min_delta,
        "eta_pairwise_loss_max_pairs": eta_pairwise_loss_max_pairs,
        "eta_pairwise_loss_use_abs_delta": eta_pairwise_loss_use_abs_delta,
        "eta_pairwise_loss_sampler_enabled": eta_pairwise_loss_sampler_enabled,
        "eta_pairwise_loss_batch_enabled": eta_pairwise_loss_batch_enabled,
        "eta_pairwise_loss_pairs_per_batch": eta_pairwise_loss_pairs_per_batch,
        "eta_pairwise_loss_buffer_max_pairs": eta_pairwise_loss_buffer_max_pairs,
        "eta_embedding_contrastive_loss_enabled": eta_embedding_contrastive_loss_enabled,
        "eta_embedding_contrastive_loss_target": eta_embedding_contrastive_loss_target,
        "eta_embedding_contrastive_loss_lambda": eta_embedding_contrastive_loss_lambda,
        "eta_embedding_contrastive_loss_min_delta": eta_embedding_contrastive_loss_min_delta,
        "eta_embedding_contrastive_loss_margin": eta_embedding_contrastive_loss_margin,
        "eta_embedding_contrastive_loss_max_pairs": eta_embedding_contrastive_loss_max_pairs,
        "eta_embedding_contrastive_loss_normalize": eta_embedding_contrastive_loss_normalize,
        "eta_embedding_contrastive_pull_enabled": eta_embedding_contrastive_pull_enabled,
        "eta_embedding_contrastive_pull_lambda": eta_embedding_contrastive_pull_lambda,
        "eta_embedding_contrastive_pull_max_delta": eta_embedding_contrastive_pull_max_delta,
        "eta_embedding_contrastive_pull_max_pairs": eta_embedding_contrastive_pull_max_pairs,
        "target_columns": target_columns,
        "target_transforms": target_transforms,
        "prediction_columns": target_columns,
        "component_loss_weights": {
            **{column: 1.0 for column in target_columns},
            **(
                {"inplane_trace_recon": 1.0}
                if {"xx", "yy", "inplane_trace_raw"}.issubset(target_columns)
                else {}
            ),
            **(
                {"inplane_eta_recon": 1.0}
                if {"xx", "yy", "eta_parallel"}.issubset(target_columns)
                else {}
            ),
        },
        "use_extra_features_csv": True,
        "extra_features_csv": str(descriptors.resolve()),
        "extra_features_id_key": "uid",
        "model": {
            "name": "alignn_atomwise",
            "atom_input_features": 92,
            "alignn_layers": 2,
            "gcn_layers": 2,
            "embedding_features": 64,
            "hidden_features": 64,
            "output_features": output_features,
            "calculate_gradient": False,
            "graphwise_weight": 1.0,
            "gradwise_weight": 0.0,
            "stresswise_weight": 0.0,
            "atomwise_weight": 0.0,
            "energy_mult_natoms": False,
            "extra_features": 0,
            "use_site_symmetry_node_features": False,
            "site_symmetry_node_features": 80,
            "site_symmetry_symprec": 0.001,
            "site_symmetry_angle_tolerance": 5.0,
            "site_symmetry_gamma_init": 0.03,
            "site_symmetry_gamma_max": 0.2,
        },
    }


def main() -> None:
    args = parse_args()
    if args.xy_tail_extra_weight_enabled and not args.tail_weight_enabled:
        raise ValueError(
            "--xy-tail-extra-weight-enabled requires --tail-weight-enabled."
        )
    labels_path = Path(args.labels)
    structures_path = Path(args.structures)
    descriptors_path = Path(args.descriptors)
    out_dir = Path(args.out_dir)

    labels = load_labels(labels_path)
    structures = load_structures(structures_path)

    columns = []
    if args.target == "columns":
        columns = [col.strip() for col in args.columns.split(",") if col.strip()]
        if not columns:
            raise ValueError("--columns is required when --target columns.")
        missing_cols = set(columns) - set(labels[0])
        if missing_cols:
            raise ValueError(f"Unknown label columns: {sorted(missing_cols)}")

    out_dir.mkdir(parents=True, exist_ok=True)

    labels_by_uid = {}
    split = None
    if args.stratify_split:
        labels_by_uid = stratification_labels(labels, args)
        split = split_ids(
            rows=labels,
            labels_by_uid=labels_by_uid,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.split_seed,
        )
        (out_dir / "stratify_labels.csv").write_text(
            "uid,physics_cluster\n"
            + "\n".join(f"{uid},{labels_by_uid[uid]}" for uid in sorted(labels_by_uid))
            + "\n"
        )
        (out_dir / "ids_train_val_test.json").write_text(
            json.dumps(split, indent=2) + "\n"
        )
    else:
        for stale_split_file in [
            out_dir / "stratify_labels.csv",
            out_dir / "ids_train_val_test.json",
        ]:
            if stale_split_file.exists():
                stale_split_file.unlink()

    tail_weights = {}
    xy_tail_extra_stats = {
        "enabled": args.xy_tail_extra_weight_enabled,
        "eta_threshold": args.xy_tail_extra_eta_threshold,
        "multiplier": args.xy_tail_extra_multiplier,
        "weight_cap": args.xy_tail_extra_weight_cap,
        "applied_count": 0,
        "max_base_weight": None,
        "max_uncapped_weight": None,
        "max_final_weight": None,
    }
    sample_weight_path = out_dir / "sample_weights.csv"
    if args.tail_weight_enabled:
        tail_weights = compute_tail_weights(
            labels=labels,
            score_name=args.tail_weight_score,
            quantile=args.tail_weight_quantile,
            weight_lambda=args.tail_weight_lambda,
            power=args.tail_weight_power,
        )
        xy_tail_extra_stats = apply_xy_tail_extra_weights(
            labels=labels,
            weights=tail_weights,
            args=args,
        )
        fieldnames = ["uid", "score", "rank", "ramp", "weight"]
        if args.xy_tail_extra_weight_enabled:
            fieldnames = [
                "uid",
                "score",
                "rank",
                "ramp",
                "base_weight",
                "xy_tail_extra_applied",
                "xy_tail_extra_multiplier",
                "xy_tail_extra_uncapped_weight",
                "weight",
            ]
        with sample_weight_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for uid, values in sorted(tail_weights.items()):
                writer.writerow({"uid": uid, **values})
    elif sample_weight_path.exists():
        sample_weight_path.unlink()

    rows_written = 0
    missing_structures: list[str] = []
    id_prop_path = out_dir / "id_prop.csv"
    with id_prop_path.open("w", newline="") as f:
        writer = csv.writer(f)
        for row in labels:
            uid = row["uid"]
            structure = structures.get(uid)
            if structure is None:
                missing_structures.append(uid)
                continue
            targets = selected_targets(row, args.target, columns)
            write_poscar(out_dir / uid, uid, structure)
            writer.writerow([uid, *targets])
            rows_written += 1

    if missing_structures:
        examples = ", ".join(missing_structures[:10])
        raise ValueError(
            f"Missing {len(missing_structures)} structures. Examples: {examples}"
        )

    target_columns = selected_target_columns(args.target, columns)
    target_transforms = (
        [{"column": col, "transform": transform} for col, transform in TARGET_SPECS[args.target]]
        if args.target in TARGET_SPECS
        else [{"column": col, "transform": "identity"} for col in target_columns]
    )
    output_features = len(selected_targets(labels[0], args.target, columns))
    config = build_config(
        out_dir=out_dir,
        descriptors=descriptors_path,
        output_features=output_features,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        tail_weight_enabled=args.tail_weight_enabled,
        tail_weight_score=args.tail_weight_score,
        sample_weight_csv=sample_weight_path,
        tail_weight_quantile=args.tail_weight_quantile,
        tail_weight_lambda=args.tail_weight_lambda,
        tail_weight_power=args.tail_weight_power,
        xy_tail_extra_weight_enabled=args.xy_tail_extra_weight_enabled,
        xy_tail_extra_eta_threshold=args.xy_tail_extra_eta_threshold,
        xy_tail_extra_multiplier=args.xy_tail_extra_multiplier,
        xy_tail_extra_weight_cap=args.xy_tail_extra_weight_cap,
        eta_pairwise_loss_enabled=args.eta_pairwise_loss_enabled,
        eta_pairwise_loss_target=args.eta_pairwise_loss_target,
        eta_pairwise_loss_lambda=args.eta_pairwise_loss_lambda,
        eta_pairwise_loss_min_delta=args.eta_pairwise_loss_min_delta,
        eta_pairwise_loss_max_pairs=args.eta_pairwise_loss_max_pairs,
        eta_pairwise_loss_use_abs_delta=args.eta_pairwise_loss_use_abs_delta,
        eta_pairwise_loss_sampler_enabled=args.eta_pairwise_loss_sampler_enabled,
        eta_pairwise_loss_batch_enabled=args.eta_pairwise_loss_batch_enabled,
        eta_pairwise_loss_pairs_per_batch=args.eta_pairwise_loss_pairs_per_batch,
        eta_pairwise_loss_buffer_max_pairs=args.eta_pairwise_loss_buffer_max_pairs,
        eta_embedding_contrastive_loss_enabled=(
            args.eta_embedding_contrastive_loss_enabled
        ),
        eta_embedding_contrastive_loss_target=(
            args.eta_embedding_contrastive_loss_target
        ),
        eta_embedding_contrastive_loss_lambda=(
            args.eta_embedding_contrastive_loss_lambda
        ),
        eta_embedding_contrastive_loss_min_delta=(
            args.eta_embedding_contrastive_loss_min_delta
        ),
        eta_embedding_contrastive_loss_margin=(
            args.eta_embedding_contrastive_loss_margin
        ),
        eta_embedding_contrastive_loss_max_pairs=(
            args.eta_embedding_contrastive_loss_max_pairs
        ),
        eta_embedding_contrastive_loss_normalize=(
            args.eta_embedding_contrastive_loss_normalize
        ),
        eta_embedding_contrastive_pull_enabled=(
            args.eta_embedding_contrastive_pull_enabled
        ),
        eta_embedding_contrastive_pull_lambda=(
            args.eta_embedding_contrastive_pull_lambda
        ),
        eta_embedding_contrastive_pull_max_delta=(
            args.eta_embedding_contrastive_pull_max_delta
        ),
        eta_embedding_contrastive_pull_max_pairs=(
            args.eta_embedding_contrastive_pull_max_pairs
        ),
        target_columns=target_columns,
        target_transforms=target_transforms,
    )
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    summary = {
        "samples": rows_written,
        "target": args.target,
        "target_columns": target_columns,
        "target_transforms": (
            target_transforms
            if args.target in TARGET_SPECS
            else []
        ),
        "output_features": output_features,
        "id_prop_csv": str(id_prop_path),
        "config": str(out_dir / "config.json"),
        "descriptors": str(descriptors_path),
        "tail_weight": {
            "enabled": args.tail_weight_enabled,
            "score": args.tail_weight_score,
            "quantile": args.tail_weight_quantile,
            "lambda": args.tail_weight_lambda,
            "power": args.tail_weight_power,
            "sample_weight_csv": str(sample_weight_path)
            if args.tail_weight_enabled
            else None,
            "min_weight": min([v["weight"] for v in tail_weights.values()])
            if tail_weights
            else None,
            "max_weight": max([v["weight"] for v in tail_weights.values()])
            if tail_weights
            else None,
            "xy_tail_extra": xy_tail_extra_stats,
        },
        "same_formula_losses": {
            "pairwise_output": {
                "enabled": args.eta_pairwise_loss_enabled,
                "target": args.eta_pairwise_loss_target,
                "lambda": args.eta_pairwise_loss_lambda,
                "min_delta": args.eta_pairwise_loss_min_delta,
                "max_pairs": args.eta_pairwise_loss_max_pairs,
                "use_abs_delta": args.eta_pairwise_loss_use_abs_delta,
                "sampler_enabled": args.eta_pairwise_loss_sampler_enabled,
                "batch_enabled": args.eta_pairwise_loss_batch_enabled,
                "pairs_per_batch": args.eta_pairwise_loss_pairs_per_batch,
                "buffer_max_pairs": args.eta_pairwise_loss_buffer_max_pairs,
            },
            "embedding_contrastive": {
                "enabled": args.eta_embedding_contrastive_loss_enabled,
                "target": args.eta_embedding_contrastive_loss_target,
                "lambda": args.eta_embedding_contrastive_loss_lambda,
                "min_delta": args.eta_embedding_contrastive_loss_min_delta,
                "margin": args.eta_embedding_contrastive_loss_margin,
                "max_pairs": args.eta_embedding_contrastive_loss_max_pairs,
                "normalize": args.eta_embedding_contrastive_loss_normalize,
                "mode": "batch",
                "pull_enabled": args.eta_embedding_contrastive_pull_enabled,
                "pull_lambda": args.eta_embedding_contrastive_pull_lambda,
                "pull_max_delta": args.eta_embedding_contrastive_pull_max_delta,
                "pull_max_pairs": args.eta_embedding_contrastive_pull_max_pairs,
            },
        },
        "stratify": {
            "enabled": args.stratify_split,
            "mode": args.stratify_mode if args.stratify_split else None,
            "trace_floor": args.stratify_trace_floor if args.stratify_split else None,
            "xy_coupling_threshold": args.stratify_xy_coupling_threshold
            if args.stratify_split
            else None,
            "diag_anisotropy_threshold": args.stratify_diag_anisotropy_threshold
            if args.stratify_split
            else None,
            "min_count": args.stratify_min_count if args.stratify_split else None,
            "eta_bins": (
                parse_float_bins(args.stratify_eta_bins)
                if args.stratify_split
                and args.stratify_mode
                in {"inplane_eta_cluster_bins", "inplane_eta_cluster_bins_xy_tail"}
                else None
            ),
            "alpha_bins": (
                parse_alpha_bins(args.stratify_alpha_bins)
                if args.stratify_split
                and args.stratify_mode == "fixed_physics_cluster_alpha_bins"
                else None
            ),
            "cluster_counts": dict(Counter(labels_by_uid.values()))
            if args.stratify_split
            else None,
            "split_counts": {k: len(v) for k, v in split.items()}
            if split is not None
            else None,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
