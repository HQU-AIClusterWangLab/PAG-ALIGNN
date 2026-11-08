#!/usr/bin/env python
"""Evaluate split polarizability ALIGNN models as reconstructed tensors."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


INPLANE_SPEC = [("xx", "log1p"), ("yy", "log1p"), ("xy", "asinh")]
OUTPLANE_SPEC = [("zz", "scaled_log1p_zz"), ("xz", "asinh"), ("yz", "asinh")]
OUTPLANE_AUX_SPEC = [("xz", "asinh"), ("yz", "asinh")]
COMPONENTS = ["xx", "yy", "zz", "xy", "xz", "yz"]
DIAGONAL_COMPONENTS = ["xx", "yy", "zz"]
EPS = 1e-12
ZZ_LOG1P_SCALE = 0.033699
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate in-plane and out-of-plane ALIGNN outputs."
    )
    parser.add_argument("--inplane-dir", required=True)
    parser.add_argument("--outplane-dir", required=True)
    parser.add_argument(
        "--outplane-aux-dir",
        default=None,
        help=(
            "Optional auxiliary out-plane model directory. When provided, "
            "xz/yz predictions are taken from this model and zz is taken "
            "from --outplane-dir."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="runs/polarizability_tensor_eval",
        help="Directory for metrics and reconstructed predictions.",
    )
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
        "Prediction file not found. Expected either "
        f"{multi_out} or {test_results}."
    )


def as_2d(values: object, expected_width: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        if arr.size == expected_width:
            return arr.reshape(1, expected_width)
        if arr.size % expected_width == 0:
            return arr.reshape(-1, expected_width)
    if arr.ndim == 2 and arr.shape[1] == expected_width:
        return arr
    raise ValueError(
        f"Expected width {expected_width}, got shape {arr.shape} from predictions."
    )


def split_spec(run_dir: Path, fallback_spec: list[tuple[str, str]]) -> list[tuple[str, str]]:
    config_path = run_dir / "config.json"
    if config_path.exists():
        with config_path.open() as f:
            config = json.load(f)
        target_columns = config.get("target_columns")
        target_transforms = config.get("target_transforms")
        if target_transforms:
            return [
                (item["column"], item["transform"])
                for item in target_transforms
            ]
        if target_columns:
            return [
                (column, COMPONENT_TRANSFORMS[column])
                for column in target_columns
            ]
    return fallback_spec


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


def load_split(
    run_dir: Path, spec: list[tuple[str, str]]
) -> dict[str, dict[str, dict[str, float]]]:
    spec = split_spec(run_dir, spec)
    with prediction_file(run_dir).open() as f:
        rows = json.load(f)

    data: dict[str, dict[str, dict[str, float]]] = {}
    width = len(spec)
    for row in rows:
        ids = row["id"]
        if isinstance(ids, str):
            ids = [ids]
        target_key = "target" if "target" in row else "target_out"
        pred_key = "predictions" if "predictions" in row else "pred_out"
        targets = as_2d(row[target_key], width)
        preds = as_2d(row[pred_key], width)
        if len(ids) != len(targets):
            raise ValueError(
                "Prediction rows do not contain one id per sample. "
                "Retrain with the patched multi-output exporter."
            )
        for uid, target_vec, pred_vec in zip(ids, targets, preds):
            target_components = decode_components(target_vec, spec)
            pred_components = decode_components(pred_vec, spec)
            data[str(uid)] = {
                "target": target_components,
                "prediction": pred_components,
            }
    return data


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    ss_res = float(np.sum((y_pred - y_true) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float("nan") if ss_tot <= EPS else float(1.0 - ss_res / ss_tot)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    err = y_pred - y_true
    return {
        "mse": float(np.mean(err**2)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "medae": float(np.median(np.abs(err))),
        "r2": r2_score(y_true, y_pred),
    }


def relative_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    rel = np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), EPS)
    return {
        "relative_mae": float(np.mean(rel)),
        "relative_rmse": float(np.sqrt(np.mean(rel**2))),
        "median_relative_error": float(np.median(rel)),
    }


def tensor_from_components(row: dict[str, float]) -> np.ndarray:
    return np.array(
        [
            [row["xx"], row["xy"], row["xz"]],
            [row["xy"], row["yy"], row["yz"]],
            [row["xz"], row["yz"], row["zz"]],
        ],
        dtype=float,
    )


def frobenius_norms(tensors: np.ndarray) -> np.ndarray:
    return np.linalg.norm(tensors.reshape(tensors.shape[0], -1), axis=1)


def add_prefixed_metrics(
    summary: dict[str, float],
    prefix: str,
    true_tensors: np.ndarray,
    pred_tensors: np.ndarray,
    true_components: np.ndarray,
    pred_components: np.ndarray,
    true_trace: np.ndarray,
    pred_trace: np.ndarray,
) -> None:
    tensor_diff = pred_tensors - true_tensors
    frob_abs = frobenius_norms(tensor_diff)
    true_frob = frobenius_norms(true_tensors)
    rel_frob = frob_abs / np.maximum(true_frob, EPS)

    summary[f"{prefix}frobenius_absolute_error"] = float(np.mean(frob_abs))
    summary[f"{prefix}relative_frobenius_error"] = float(np.mean(rel_frob))
    summary[f"{prefix}trace_mae"] = float(np.mean(np.abs(pred_trace - true_trace)))
    summary[f"{prefix}trace_relative_mae"] = float(
        np.mean(np.abs(pred_trace - true_trace) / np.maximum(np.abs(true_trace), EPS))
    )

    diag_true = true_components[:, :3]
    diag_pred = pred_components[:, :3]
    diag_rel = np.abs(diag_pred - diag_true) / np.maximum(np.abs(diag_true), EPS)
    summary[f"{prefix}diagonal_component_relative_mae"] = float(np.mean(diag_rel))


def build_summary(
    true_tensors: np.ndarray,
    pred_tensors: np.ndarray,
    true_components: np.ndarray,
    pred_components: np.ndarray,
) -> dict[str, float]:
    summary: dict[str, float] = {"n_samples": int(true_tensors.shape[0])}

    component = regression_metrics(true_components, pred_components)
    summary["component_mse"] = component["mse"]
    summary["component_rmse"] = component["rmse"]
    summary["component_mae"] = component["mae"]
    summary["component_medae"] = component["medae"]

    component_r2 = [
        r2_score(true_components[:, i], pred_components[:, i])
        for i in range(true_components.shape[1])
    ]
    summary["r2_uniform_average"] = float(np.nanmean(component_r2))

    tensor_diff = pred_tensors - true_tensors
    frob_sq = np.sum(tensor_diff**2, axis=(1, 2))
    frob_abs = np.sqrt(frob_sq)
    true_frob = frobenius_norms(true_tensors)
    summary["frobenius_mse"] = float(np.mean(frob_sq))
    summary["frobenius_rmse"] = float(np.sqrt(np.mean(frob_sq)))
    summary["frobenius_absolute_error"] = float(np.mean(frob_abs))
    summary["relative_frobenius_error"] = float(
        np.mean(frob_abs / np.maximum(true_frob, EPS))
    )

    pred_eigvals = np.linalg.eigvalsh(pred_tensors)
    min_pred_eigvals = np.min(pred_eigvals, axis=1)
    summary["positive_definite_rate"] = float(np.mean(min_pred_eigvals > 0))
    summary["min_predicted_eigenvalue"] = float(np.min(min_pred_eigvals))
    summary["max_symmetry_error"] = float(
        np.max(np.abs(pred_tensors - np.transpose(pred_tensors, (0, 2, 1))))
    )

    true_trace = np.trace(true_tensors, axis1=1, axis2=2)
    pred_trace = np.trace(pred_tensors, axis1=1, axis2=2)
    trace = regression_metrics(true_trace, pred_trace)
    trace_rel = relative_metrics(true_trace, pred_trace)
    summary["trace_mae"] = trace["mae"]
    summary["trace_rmse"] = trace["rmse"]
    summary["trace_medae"] = trace["medae"]
    summary["trace_relative_mae"] = trace_rel["relative_mae"]
    summary["trace_relative_rmse"] = trace_rel["relative_rmse"]
    summary["trace_median_relative_error"] = trace_rel["median_relative_error"]
    summary["trace_r2"] = trace["r2"]
    summary["trace_mean_error"] = float(np.mean(pred_trace - true_trace))

    diag_true = true_components[:, :3]
    diag_pred = pred_components[:, :3]
    diag_rel = relative_metrics(diag_true, diag_pred)
    summary["diagonal_component_relative_mae"] = diag_rel["relative_mae"]
    summary["diagonal_component_relative_rmse"] = diag_rel["relative_rmse"]
    summary["diagonal_component_median_relative_error"] = diag_rel[
        "median_relative_error"
    ]

    for idx, component_name in enumerate(COMPONENTS):
        m = regression_metrics(true_components[:, idx], pred_components[:, idx])
        summary[f"{component_name}_mae"] = m["mae"]
        summary[f"{component_name}_rmse"] = m["rmse"]
        summary[f"{component_name}_r2"] = m["r2"]
        summary[f"{component_name}_medae"] = m["medae"]
        if component_name in DIAGONAL_COMPONENTS:
            rel = relative_metrics(true_components[:, idx], pred_components[:, idx])
            summary[f"{component_name}_relative_mae"] = rel["relative_mae"]
            summary[f"{component_name}_relative_rmse"] = rel["relative_rmse"]
            summary[f"{component_name}_median_relative_error"] = rel[
                "median_relative_error"
            ]

    for fraction, label in [(0.20, "trace_top_20"), (0.10, "trace_top_10")]:
        n_top = max(1, int(math.ceil(len(true_trace) * fraction)))
        top_idx = np.argsort(true_trace)[-n_top:]
        add_prefixed_metrics(
            summary,
            f"{label}_",
            true_tensors[top_idx],
            pred_tensors[top_idx],
            true_components[top_idx],
            pred_components[top_idx],
            true_trace[top_idx],
            pred_trace[top_idx],
        )

    return summary


def main() -> None:
    args = parse_args()
    inplane = load_split(Path(args.inplane_dir), INPLANE_SPEC)
    outplane = load_split(Path(args.outplane_dir), OUTPLANE_SPEC)
    outplane_aux = (
        load_split(Path(args.outplane_aux_dir), OUTPLANE_AUX_SPEC)
        if args.outplane_aux_dir is not None
        else None
    )
    id_sets = [set(inplane), set(outplane)]
    if outplane_aux is not None:
        id_sets.append(set(outplane_aux))
    common_ids = sorted(set.intersection(*id_sets))
    if not common_ids:
        raise ValueError("No overlapping ids found between prediction files.")

    records = []
    true_components = []
    pred_components = []
    true_tensors = []
    pred_tensors = []

    for uid in common_ids:
        target = {**inplane[uid]["target"], **outplane[uid]["target"]}
        prediction = {**inplane[uid]["prediction"], **outplane[uid]["prediction"]}
        if outplane_aux is not None:
            target.update(outplane_aux[uid]["target"])
            prediction.update(
                {
                    "xz": outplane_aux[uid]["prediction"]["xz"],
                    "yz": outplane_aux[uid]["prediction"]["yz"],
                }
            )
        target_tensor = tensor_from_components(target)
        prediction_tensor = tensor_from_components(prediction)

        true_components.append([target[c] for c in COMPONENTS])
        pred_components.append([prediction[c] for c in COMPONENTS])
        true_tensors.append(target_tensor)
        pred_tensors.append(prediction_tensor)

        record = {"id": uid}
        for component in COMPONENTS:
            record[f"target_{component}"] = target[component]
            record[f"prediction_{component}"] = prediction[component]
        record["target_trace"] = float(np.trace(target_tensor))
        record["prediction_trace"] = float(np.trace(prediction_tensor))
        record["target_frobenius_norm"] = float(np.linalg.norm(target_tensor))
        record["prediction_frobenius_norm"] = float(np.linalg.norm(prediction_tensor))
        records.append(record)

    true_components_arr = np.asarray(true_components, dtype=float)
    pred_components_arr = np.asarray(pred_components, dtype=float)
    true_tensors_arr = np.asarray(true_tensors, dtype=float)
    pred_tensors_arr = np.asarray(pred_tensors, dtype=float)
    summary = build_summary(
        true_tensors_arr,
        pred_tensors_arr,
        true_components_arr,
        pred_components_arr,
    )
    summary["outplane_prediction_source"] = (
        "zz_from_main_xz_yz_from_aux"
        if outplane_aux is not None
        else "zz_xz_yz_from_main"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")

    with (out_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in summary.items():
            writer.writerow({"metric": key, "value": value})

    with (out_dir / "reconstructed_predictions.csv").open("w", newline="") as f:
        fieldnames = ["id"]
        for component in COMPONENTS:
            fieldnames.extend([f"target_{component}", f"prediction_{component}"])
        fieldnames.extend(
            [
                "target_trace",
                "prediction_trace",
                "target_frobenius_norm",
                "prediction_frobenius_norm",
            ]
        )
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
