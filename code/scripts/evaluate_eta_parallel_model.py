#!/usr/bin/env python
"""Evaluate the standalone eta_parallel model predictions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate standalone eta_parallel predictions."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Defaults to <run-dir>/evaluation.",
    )
    parser.add_argument(
        "--no-clip",
        action="store_true",
        help="Disable the default eta_parallel prediction clipping to [0, 1].",
    )
    return parser.parse_args()


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_pred - y_true) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float("nan") if ss_tot <= EPS else float(1.0 - ss_res / ss_tot)


def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, *, prefix: str = "eta"
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    err = y_pred - y_true
    rel = np.abs(err) / np.maximum(np.abs(y_true), EPS)
    return {
        f"{prefix}_mse": float(np.mean(err**2)),
        f"{prefix}_rmse": float(np.sqrt(np.mean(err**2))),
        f"{prefix}_mae": float(np.mean(np.abs(err))),
        f"{prefix}_medae": float(np.median(np.abs(err))),
        f"{prefix}_r2": r2_score(y_true, y_pred),
        f"{prefix}_mean_error": float(np.mean(err)),
        f"{prefix}_relative_mae": float(np.mean(rel)),
        f"{prefix}_relative_rmse": float(np.sqrt(np.mean(rel**2))),
        f"{prefix}_median_relative_error": float(np.median(rel)),
    }


def prediction_path(run_dir: Path) -> Path:
    csv_path = run_dir / "prediction_results_test_set.csv"
    if csv_path.exists():
        return csv_path
    json_path = run_dir / "Test_results.json"
    if json_path.exists():
        return json_path
    multi_out_path = run_dir / "multi_out_predictions.json"
    if multi_out_path.exists():
        return multi_out_path
    raise FileNotFoundError(
        "Expected prediction_results_test_set.csv, Test_results.json, "
        f"or multi_out_predictions.json in {run_dir}."
    )


def prediction_columns(run_dir: Path) -> list[str]:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return ["eta_parallel"]
    with config_path.open() as f:
        config = json.load(f)
    columns = config.get("prediction_columns") or config.get("target_columns")
    if not columns:
        return ["eta_parallel"]
    return [str(column) for column in columns]


def as_matrix(value: object, width: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        if width != 1:
            raise ValueError(f"Expected width {width}, got scalar prediction.")
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        if arr.size == width:
            return arr.reshape(1, width)
        if arr.size % width == 0:
            return arr.reshape(-1, width)
    if arr.ndim == 2 and arr.shape[1] == width:
        return arr
    raise ValueError(f"Expected prediction width {width}, got shape {arr.shape}.")


def read_predictions(
    path: Path, columns: list[str]
) -> list[dict[str, float | str]]:
    width = len(columns)
    if path.suffix.lower() == ".csv":
        if width != 1:
            raise ValueError(
                "CSV prediction_results_test_set.csv supports only one output. "
                "Use multi_out_predictions.json for multitask eta models."
            )
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        column = columns[0]
        return [
            {
                "id": row.get("id", ""),
                f"target_{column}": float(row["target"]),
                f"prediction_{column}": float(row["prediction"]),
            }
            for row in rows
        ]

    with path.open() as f:
        rows = json.load(f)
    records = []
    for row in rows:
        pred_key = "predictions" if "predictions" in row else "pred_out"
        target_key = "target" if "target" in row else "target_out"
        targets = as_matrix(row[target_key], width)
        preds = as_matrix(row[pred_key], width)
        if targets.shape != preds.shape:
            raise ValueError(
                f"Target/prediction shape mismatch: {targets.shape} != {preds.shape}"
            )
        ids = row.get("id", row.get("jid", ""))
        if isinstance(ids, list):
            if len(ids) != targets.shape[0]:
                raise ValueError(
                    f"Prediction row has {len(ids)} ids but {targets.shape[0]} rows."
                )
            id_values = [str(item) for item in ids]
        else:
            if targets.shape[0] != 1:
                raise ValueError(
                    "Prediction row has multiple target rows but a scalar id."
                )
            id_values = [str(ids)]
        for uid, target_vec, pred_vec in zip(id_values, targets, preds):
            record: dict[str, float | str] = {"id": uid}
            for idx, column in enumerate(columns):
                record[f"target_{column}"] = float(target_vec[idx])
                record[f"prediction_{column}"] = float(pred_vec[idx])
            records.append(record)
    return records


def clip_eta_predictions(
    records: list[dict[str, float | str]],
    *,
    lower: float = 0.0,
    upper: float = 1.0,
) -> list[dict[str, float | str]]:
    clipped = []
    for record in records:
        raw = float(record["prediction_eta_parallel"])
        pred = float(np.clip(raw, lower, upper))
        clipped.append(
            {
                **record,
                "raw_prediction_eta_parallel": raw,
                "prediction_eta_parallel": pred,
                "clipped_low": int(raw < lower),
                "clipped_high": int(raw > upper),
            }
        )
    return clipped


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "evaluation"

    columns = prediction_columns(run_dir)
    eta_column = "eta_parallel" if "eta_parallel" in columns else columns[0]
    path = prediction_path(run_dir)
    records = read_predictions(path, columns)
    if not records:
        raise ValueError(f"No prediction rows found in {path}.")

    y_true = np.asarray([r[f"target_{eta_column}"] for r in records], dtype=float)
    raw_pred = np.asarray(
        [r[f"prediction_{eta_column}"] for r in records], dtype=float
    )
    clip_predictions = not args.no_clip
    if clip_predictions and eta_column == "eta_parallel":
        records = clip_eta_predictions(records)
    else:
        records = [
            {
                **record,
                f"raw_prediction_{eta_column}": float(
                    record[f"prediction_{eta_column}"]
                ),
                "clipped_low": 0,
                "clipped_high": 0,
            }
            for record in records
        ]
    eta_clipping_enabled = clip_predictions and eta_column == "eta_parallel"
    clip_summary = {
        "clip_predictions": clip_predictions,
        "prediction_clip_min": 0.0 if eta_clipping_enabled else None,
        "prediction_clip_max": 1.0 if eta_clipping_enabled else None,
        "n_clipped_low": int(np.sum(raw_pred < 0.0))
        if eta_clipping_enabled
        else 0,
        "n_clipped_high": int(np.sum(raw_pred > 1.0))
        if eta_clipping_enabled
        else 0,
        "n_clipped_total": int(
            np.sum((raw_pred < 0.0) | (raw_pred > 1.0))
        )
        if eta_clipping_enabled
        else 0,
    }
    y_pred = np.asarray([r[f"prediction_{eta_column}"] for r in records], dtype=float)
    summary = {
        "n_test": int(len(records)),
        "run_dir": str(run_dir),
        "prediction_file": str(path),
        "target": eta_column,
        "target_columns": columns,
        **clip_summary,
        **regression_metrics(y_true, y_pred, prefix="eta"),
    }
    for column in columns:
        if column == eta_column:
            continue
        column_true = np.asarray([r[f"target_{column}"] for r in records], dtype=float)
        column_pred = np.asarray(
            [r[f"prediction_{column}"] for r in records], dtype=float
        )
        summary.update(
            regression_metrics(column_true, column_pred, prefix=column)
        )
    if eta_clipping_enabled:
        raw_metrics = regression_metrics(y_true, raw_pred, prefix="eta")
        summary.update({f"raw_{key}": value for key, value in raw_metrics.items()})

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (out_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in summary.items():
            writer.writerow({"metric": key, "value": value})
    with (out_dir / "predictions.csv").open("w", newline="") as f:
        fieldnames = ["id"]
        for column in columns:
            fieldnames.extend(
                [
                    f"target_{column}",
                    f"raw_prediction_{column}",
                    f"prediction_{column}",
                    f"error_{column}",
                    f"absolute_error_{column}",
                ]
            )
        fieldnames.extend(["error", "absolute_error", "clipped_low", "clipped_high"])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = dict(record)
            for column in columns:
                row.setdefault(
                    f"raw_prediction_{column}",
                    float(row[f"prediction_{column}"]),
                )
                err = float(row[f"prediction_{column}"]) - float(
                    row[f"target_{column}"]
                )
                row[f"error_{column}"] = err
                row[f"absolute_error_{column}"] = abs(err)
            eta_err = row[f"error_{eta_column}"]
            row["error"] = eta_err
            row["absolute_error"] = abs(float(eta_err))
            writer.writerow(row)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
