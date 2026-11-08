#!/usr/bin/env python
"""Evaluate one split polarizability submodel."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


SPECS = {
    "inplane": [("xx", "log1p"), ("yy", "log1p"), ("xy", "asinh")],
    "outplane": [("zz", "scaled_log1p_zz"), ("xz", "asinh"), ("yz", "asinh")],
}
COMPONENT_TRANSFORMS = {
    "xx": "log1p",
    "yy": "log1p",
    "zz": "scaled_log1p_zz",
    "xy": "asinh",
    "xz": "asinh",
    "yz": "asinh",
    "eta_parallel": "asinh",
    "inplane_trace_raw": "log1p_floor",
}
EPS = 1e-12
ZZ_LOG1P_SCALE = 0.033699


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one tensor submodel.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", choices=["inplane", "outplane"], required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


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
    raise ValueError(f"Unknown transform: {transform}")


def prediction_file(run_dir: Path) -> Path:
    multi_out = run_dir / "multi_out_predictions.json"
    if multi_out.exists():
        return multi_out
    test_results = run_dir / "Test_results.json"
    if test_results.exists():
        return test_results
    raise FileNotFoundError(
        f"Expected {multi_out} or {test_results}, but neither exists."
    )


def split_specs(
    run_dir: Path, split: str
) -> tuple[list[tuple[str, str]], list[str]]:
    config_path = run_dir / "config.json"
    if config_path.exists():
        with config_path.open() as f:
            config = json.load(f)
        target_columns = config.get("target_columns")
        if target_columns:
            prediction_columns = config.get("prediction_columns") or target_columns
            unknown = set(prediction_columns) - set(target_columns)
            if unknown:
                raise ValueError(
                    "prediction_columns must be a subset of target_columns. "
                    f"Unknown columns: {sorted(unknown)}"
                )
            target_transforms = config.get("target_transforms")
            if target_transforms:
                return [
                    (item["column"], item["transform"])
                    for item in target_transforms
                ], list(prediction_columns)
            return [
                (column, COMPONENT_TRANSFORMS[column])
                for column in target_columns
            ], list(prediction_columns)
    spec = SPECS[split]
    return spec, [column for column, _ in spec]


def decode_components(
    vector: np.ndarray, spec: list[tuple[str, str]]
) -> dict[str, float]:
    decoded: dict[str, float] = {}
    deferred_ratios: list[tuple[str, float]] = []
    for idx, (component, transform) in enumerate(spec):
        value = float(vector[idx])
        if transform == "inplane_xy_ratio":
            deferred_ratios.append((component, inverse_transform(value, transform)))
        else:
            decoded[component] = inverse_transform(value, transform)
    for component, ratio in deferred_ratios:
        if component == "xy":
            decoded[component] = 0.5 * ratio * (decoded["xx"] + decoded["yy"])
        else:
            decoded[component] = inverse_transform(ratio, "identity")
    return decoded


def as_2d(values: object, expected_width: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1 and arr.size == expected_width:
        return arr.reshape(1, expected_width)
    if arr.ndim == 1 and expected_width == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 1 and arr.size % expected_width == 0:
        return arr.reshape(-1, expected_width)
    if arr.ndim == 2 and arr.shape[1] == expected_width:
        return arr
    raise ValueError(f"Expected width {expected_width}, got shape {arr.shape}.")


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_pred - y_true) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float("nan") if ss_tot <= EPS else float(1.0 - ss_res / ss_tot)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    err = y_pred - y_true
    rel = np.abs(err) / np.maximum(np.abs(y_true), EPS)
    return {
        "mse": float(np.mean(err**2)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "medae": float(np.median(np.abs(err))),
        "r2": r2_score(y_true, y_pred),
        "relative_mae": float(np.mean(rel)),
        "relative_rmse": float(np.sqrt(np.mean(rel**2))),
        "median_relative_error": float(np.median(rel)),
    }


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    spec, prediction_columns = split_specs(run_dir, args.split)
    prediction_column_set = set(prediction_columns)
    width = len(spec)

    with prediction_file(run_dir).open() as f:
        rows = json.load(f)

    records = []
    y_true = []
    y_pred = []
    for row in rows:
        uid = row["id"]
        target_key = "target" if "target" in row else "target_out"
        pred_key = "predictions" if "predictions" in row else "pred_out"
        target = as_2d(row[target_key], width)[0]
        pred = as_2d(row[pred_key], width)[0]
        rec = {"id": uid}
        true_vals_by_component = decode_components(target, spec)
        pred_vals_by_component = decode_components(pred, spec)
        for component, _ in spec:
            true_val = true_vals_by_component[component]
            pred_val = pred_vals_by_component[component]
            if component in prediction_column_set:
                rec[f"target_{component}"] = true_val
                rec[f"prediction_{component}"] = pred_val
        true_vals = [true_vals_by_component[column] for column in prediction_columns]
        pred_vals = [pred_vals_by_component[column] for column in prediction_columns]
        records.append(rec)
        y_true.append(true_vals)
        y_pred.append(pred_vals)

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    summary = {
        "n_samples": int(y_true_arr.shape[0]),
        "split": args.split,
        "target_columns": [column for column, _ in spec],
        "prediction_columns": prediction_columns,
    }
    overall = metrics(y_true_arr, y_pred_arr)
    for key, value in overall.items():
        summary[f"component_{key}"] = value
    for idx, component in enumerate(prediction_columns):
        comp_metrics = metrics(y_true_arr[:, idx], y_pred_arr[:, idx])
        for key, value in comp_metrics.items():
            summary[f"{component}_{key}"] = value

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (out_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in summary.items():
            writer.writerow({"metric": key, "value": value})
    with (out_dir / "predictions.csv").open("w", newline="") as f:
        fieldnames = list(records[0].keys()) if records else ["id"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
