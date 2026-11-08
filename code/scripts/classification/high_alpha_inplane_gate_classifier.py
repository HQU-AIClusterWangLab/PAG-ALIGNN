"""Train a high-alpha-inplane gate classifier with CatBoost.

This script builds a binary safety gate for the body tensor regression model:

    alpha_inplane = (alpha_xx + alpha_yy) / 2
    Gate A: high_alpha = 1 if alpha_inplane > 16.1 else 0
    Gate B: severe_high_alpha = 1 if alpha_inplane > 50.0 else 0

The main objective is high recall for high-alpha samples. The final routing
thresholds are selected from out-of-fold predictions, not fixed at 0.5. During
inference, a sample is routed away from body regression if Gate A OR Gate B fires.

Recommended first run:
    python high_alpha_inplane_gate_classifier.py \
      --alpha-threshold 16.1 \
      --target-recall 0.90 \
      --class-weight-mode auto \
      --gray-low 15.0 \
      --gray-high 17.0 \
      --gray-weight 0.3 \
      --positive-severity-weight-mode piecewise \
      --severe-alpha-threshold 50.0 \
      --severe-target-recall 0.95
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

if TYPE_CHECKING:
    from catboost import CatBoostClassifier, Pool


DEFAULT_LABEL_PATH = Path(
    r"E:\Yjlong\Catboost\data\static_lattice_polarizability_tensors.csv"
)
DEFAULT_DESCRIPTOR_PATH = Path(
    r"E:\Yjlong\Catboost\data\describtors\tree_descriptors_v2_model_ready.before_low_cost_descriptors.csv"
)
DEFAULT_OUTPUT_DIR = Path(r"E:\Yjlong\Catboost\outputs\high_alpha_inplane_gate")
DEFAULT_BAD_UIDS = ("2P-1", "2As-1", "2Sb-1")
SEVERE_FN_ALPHA_THRESHOLDS = (20.0, 30.0, 50.0, 100.0, 1000.0)
PIECEWISE_SEVERITY_BINS = (
    (16.1, 30.0, 1.0),
    (30.0, 50.0, 2.0),
    (50.0, 100.0, 3.0),
    (100.0, 1000.0, 5.0),
    (1000.0, np.inf, 8.0),
)

# =============================================================================
# Tunable defaults
# =============================================================================
# Edit this block directly for normal experiments. Command-line arguments still
# override these values when provided.
TUNABLE_CONFIG: Dict[str, Any] = {
    # Gate label and threshold selection.
    "alpha_threshold": 16.1,
    "target_recall": 0.90,
    "severe_alpha_threshold": 50.0,
    "severe_target_recall": 0.95,
    "combined_body_retention_min": 0.795,
    "combined_body_retention_max": 0.805,
    "n_splits": 5,
    "random_seed": 42,

    # CatBoost model capacity and regularization.
    "iterations": 5000,
    "early_stopping_rounds": 300,
    "learning_rate": 0.03,
    "depth": 5,
    "l2_leaf_reg": 30.0,
    "random_strength": 5.0,
    "bagging_temperature": 0.5,
    "border_count": 128,

    # Runtime.
    "verbose": 100,
    "task_type": "GPU",
    "devices": None,
    "fallback_to_cpu": False,

    # Class imbalance handling.
    # Use one of: "auto", "manual", "none".
    "class_weight_mode": "auto",
    "positive_class_weight": 8.0,
    "severe_class_weight_mode": "manual",
    "severe_positive_class_weight": 12.0,

    # Samples close to the label boundary are noisy, so down-weight them.
    "gray_low": 15.0,
    "gray_high": 17.0,
    "gray_weight": 0.3,

    # Inside the positive class, make very large alpha samples more important.
    # Use one of: "piecewise", "smooth", "none".
    "positive_severity_weight_mode": "piecewise",
    "positive_severity_smooth_scale": 1.5,
    "positive_severity_max_weight": 8.0,
}

# Columns that are labels or direct tensor targets and must never be used as features.
LABEL_COLUMNS = {
    "alpha_xx",
    "alpha_xy",
    "alpha_xz",
    "alpha_yx",
    "alpha_yy",
    "alpha_yz",
    "alpha_zx",
    "alpha_zy",
    "alpha_zz",
    "alpha_inplane",
    "trace",
    "trace_inplane",
    "trace_full",
    "frobenius_norm",
    "tensor_frobenius_norm",
    "d",
    "b",
    "eta_parallel",
    "anisotropy_inplane",
    "high_alpha",
    "high_amp",
    "label",
    "target",
}

# Extra substring checks for likely leakage columns in descriptor tables.
# Intentionally avoid broad words such as "eta" or "polarizability" to reduce false positives.
LEAKAGE_SUBSTRINGS = (
    "alpha_xx",
    "alpha_yy",
    "alpha_xy",
    "alpha_xz",
    "alpha_yx",
    "alpha_yz",
    "alpha_zx",
    "alpha_zy",
    "alpha_zz",
    "alpha_inplane",
    "trace_inplane",
    "trace_full",
    "frobenius",
    "eta_parallel",
    "anisotropy_inplane",
    "offdiag_frobenius_ratio",
    "high_alpha",
    "high_amp",
    "target",
    "label",
)

DIAGNOSTIC_COLUMNS = [
    "alpha_xx",
    "alpha_yy",
    "alpha_xy",
    "alpha_zz",
    "alpha_inplane",
    "d",
    "b",
    "eta_parallel",
    "high_alpha",
]


# The CatBoost imports are delayed so --dry-run can work in environments
# where catboost is not installed.
def require_catboost() -> None:
    global CatBoostClassifier, CatBoostError, Pool
    try:
        from catboost import CatBoostClassifier, CatBoostError, Pool
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "CatBoost is required for training this gate classifier. "
            "Install it in the Python environment you use for this project, "
            "or run the script from the environment where your existing CatBoost "
            "training scripts work."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a high-alpha-inplane safety gate classifier."
    )
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABEL_PATH)
    parser.add_argument("--descriptors", type=Path, default=DEFAULT_DESCRIPTOR_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--alpha-threshold", type=float, default=TUNABLE_CONFIG["alpha_threshold"])
    parser.add_argument("--target-recall", type=float, default=TUNABLE_CONFIG["target_recall"])
    parser.add_argument("--severe-alpha-threshold", type=float, default=TUNABLE_CONFIG["severe_alpha_threshold"])
    parser.add_argument("--severe-target-recall", type=float, default=TUNABLE_CONFIG["severe_target_recall"])
    parser.add_argument("--combined-body-retention-min", type=float, default=TUNABLE_CONFIG["combined_body_retention_min"])
    parser.add_argument("--combined-body-retention-max", type=float, default=TUNABLE_CONFIG["combined_body_retention_max"])
    parser.add_argument("--n-splits", type=int, default=TUNABLE_CONFIG["n_splits"])
    parser.add_argument("--random-seed", type=int, default=TUNABLE_CONFIG["random_seed"])

    # More conservative defaults for ~2000 samples.
    parser.add_argument("--iterations", type=int, default=TUNABLE_CONFIG["iterations"])
    parser.add_argument("--early-stopping-rounds", type=int, default=TUNABLE_CONFIG["early_stopping_rounds"])
    parser.add_argument("--learning-rate", type=float, default=TUNABLE_CONFIG["learning_rate"])
    parser.add_argument("--depth", type=int, default=TUNABLE_CONFIG["depth"])
    parser.add_argument("--l2-leaf-reg", type=float, default=TUNABLE_CONFIG["l2_leaf_reg"])
    parser.add_argument("--random-strength", type=float, default=TUNABLE_CONFIG["random_strength"])
    parser.add_argument("--bagging-temperature", type=float, default=TUNABLE_CONFIG["bagging_temperature"])
    parser.add_argument("--border-count", type=int, default=TUNABLE_CONFIG["border_count"])

    parser.add_argument("--verbose", type=int, default=TUNABLE_CONFIG["verbose"])
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default=TUNABLE_CONFIG["task_type"])
    parser.add_argument("--devices", type=str, default=TUNABLE_CONFIG["devices"])
    parser.add_argument(
        "--fallback-to-cpu",
        action=argparse.BooleanOptionalAction,
        default=TUNABLE_CONFIG["fallback_to_cpu"],
        help="Retry on CPU if GPU training fails.",
    )
    parser.add_argument(
        "--class-weight-mode",
        choices=("auto", "manual", "none"),
        default=TUNABLE_CONFIG["class_weight_mode"],
        help=(
            "auto uses auto_class_weights=Balanced; manual uses class_weights; "
            "none disables class-level weighting."
        ),
    )
    parser.add_argument(
        "--positive-class-weight",
        type=float,
        default=TUNABLE_CONFIG["positive_class_weight"],
        help="Positive class weight used only when --class-weight-mode manual.",
    )
    parser.add_argument(
        "--severe-class-weight-mode",
        choices=("auto", "manual", "none"),
        default=TUNABLE_CONFIG["severe_class_weight_mode"],
        help="Class weighting mode for Gate B, the severe high-alpha classifier.",
    )
    parser.add_argument(
        "--severe-positive-class-weight",
        type=float,
        default=TUNABLE_CONFIG["severe_positive_class_weight"],
        help="Gate B positive class weight used only when --severe-class-weight-mode manual.",
    )
    parser.add_argument("--gray-low", type=float, default=TUNABLE_CONFIG["gray_low"])
    parser.add_argument("--gray-high", type=float, default=TUNABLE_CONFIG["gray_high"])
    parser.add_argument("--gray-weight", type=float, default=TUNABLE_CONFIG["gray_weight"])
    parser.add_argument(
        "--positive-severity-weight-mode",
        choices=("piecewise", "smooth", "none"),
        default=TUNABLE_CONFIG["positive_severity_weight_mode"],
        help=(
            "Additional weighting inside the positive class. piecewise uses "
            "1/2/3/5/8 for alpha_inplane bins; smooth uses a clipped log rule."
        ),
    )
    parser.add_argument(
        "--positive-severity-smooth-scale",
        type=float,
        default=TUNABLE_CONFIG["positive_severity_smooth_scale"],
        help="Scale for smooth positive severity weighting.",
    )
    parser.add_argument(
        "--positive-severity-max-weight",
        type=float,
        default=TUNABLE_CONFIG["positive_severity_max_weight"],
        help="Maximum positive severity weight for smooth mode.",
    )
    parser.add_argument(
        "--bad-uids",
        nargs="*",
        default=list(DEFAULT_BAD_UIDS),
        help=(
            "Sample ids to drop before training. Pass --bad-uids with no values "
            "to disable dropping."
        ),
    )
    parser.add_argument(
        "--allow-suspect-feature-names",
        action="store_true",
        help=(
            "Allow descriptor feature names that look like target leakage columns. "
            "By default, the script raises an error when such names are found."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only load data and print label/feature statistics.",
    )
    return parser.parse_args()


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(to_builtin(payload), file, indent=2, ensure_ascii=False)


def normalize_sample_id(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    out = frame.copy()
    if "sample_id" not in out.columns and "uid" in out.columns:
        out = out.rename(columns={"uid": "sample_id"})
    if "sample_id" not in out.columns:
        raise ValueError(f"{source_name} must contain sample_id or uid.")
    out["sample_id"] = out["sample_id"].astype(str)
    return out


def compute_label_diagnostics(labels: pd.DataFrame) -> pd.DataFrame:
    """Add alpha_inplane, d, b and eta_parallel to the label frame.

    These diagnostic columns are used only for reports and error analysis.
    They are explicitly excluded from model features.
    """
    out = labels.copy()
    out["alpha_xx"] = out["alpha_xx"].astype(float)
    out["alpha_yy"] = out["alpha_yy"].astype(float)
    out["alpha_inplane"] = (out["alpha_xx"] + out["alpha_yy"]) / 2.0

    if "alpha_xy" in out.columns:
        out["alpha_xy"] = out["alpha_xy"].astype(float)
    else:
        out["alpha_xy"] = np.nan

    if "alpha_zz" in out.columns:
        out["alpha_zz"] = out["alpha_zz"].astype(float)
    else:
        out["alpha_zz"] = np.nan

    den = out["alpha_xx"] + out["alpha_yy"]
    valid_den = np.isfinite(den) & (np.abs(den) > 0.0)
    out["d"] = np.nan
    out["b"] = np.nan
    out["eta_parallel"] = np.nan

    out.loc[valid_den, "d"] = (
        out.loc[valid_den, "alpha_xx"] - out.loc[valid_den, "alpha_yy"]
    ) / den.loc[valid_den]
    out.loc[valid_den, "b"] = (
        2.0 * out.loc[valid_den, "alpha_xy"]
    ) / den.loc[valid_den]
    out.loc[valid_den, "eta_parallel"] = np.sqrt(
        (out.loc[valid_den, "alpha_xx"] - out.loc[valid_den, "alpha_yy"]) ** 2
        + 4.0 * out.loc[valid_den, "alpha_xy"] ** 2
    ) / den.loc[valid_den]

    return out


def suspect_feature_columns(feature_cols: Sequence[str]) -> List[str]:
    suspects: List[str] = []
    for col in feature_cols:
        low = col.lower()
        if low in LABEL_COLUMNS:
            suspects.append(col)
            continue
        if any(pattern in low for pattern in LEAKAGE_SUBSTRINGS):
            suspects.append(col)
    return sorted(set(suspects), key=list(feature_cols).index)


def read_inputs(
    labels_path: Path,
    descriptors_path: Path,
    alpha_threshold: float,
    bad_uids: Sequence[str],
    allow_suspect_feature_names: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, List[str]]:
    labels_path = labels_path.expanduser().resolve()
    descriptors_path = descriptors_path.expanduser().resolve()
    if not labels_path.exists():
        raise FileNotFoundError(f"Label CSV not found: {labels_path}")
    if not descriptors_path.exists():
        raise FileNotFoundError(f"Descriptor CSV not found: {descriptors_path}")

    labels = normalize_sample_id(pd.read_csv(labels_path), "labels")
    descriptors = normalize_sample_id(pd.read_csv(descriptors_path), "descriptors")

    required = {"sample_id", "alpha_xx", "alpha_yy"}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"Label CSV is missing columns: {missing}")
    if "alpha_xy" not in labels.columns:
        print(
            "WARNING: Label CSV does not contain alpha_xy. "
            "eta_parallel and b will be NaN in reports."
        )

    labels = labels.drop_duplicates("sample_id")
    descriptors = descriptors.drop_duplicates("sample_id")

    labels = compute_label_diagnostics(labels)
    labels["high_alpha"] = (labels["alpha_inplane"] > alpha_threshold).astype(int)

    available_diag_cols = [
        col for col in DIAGNOSTIC_COLUMNS if col in labels.columns
    ]
    data = descriptors.merge(
        labels[["sample_id", *available_diag_cols]],
        on="sample_id",
        how="inner",
    )
    if bad_uids:
        bad_set = {str(uid) for uid in bad_uids}
        data = data.loc[~data["sample_id"].isin(bad_set)].copy()
    if data.empty:
        raise ValueError("No rows remained after merging descriptors and labels.")

    feature_cols = [
        col
        for col in descriptors.columns
        if col != "sample_id" and col not in LABEL_COLUMNS
    ]
    suspects = suspect_feature_columns(feature_cols)
    if suspects:
        message = (
            "Possible target leakage columns found in descriptors: "
            f"{suspects}. Remove them from descriptors, or rerun with "
            "--allow-suspect-feature-names after manual confirmation."
        )
        if not allow_suspect_feature_names:
            raise ValueError(message)
        print("WARNING: " + message)

    X = data[feature_cols].copy()
    y = data["high_alpha"].astype(int)
    alpha_inplane = data["alpha_inplane"].astype(float)
    sample_ids = data["sample_id"].astype(str)
    X = prepare_features(X)
    return data, X, y, alpha_inplane, sample_ids, suspects


def prepare_features(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    for col in out.columns:
        if pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].astype(int)
    return out


def categorical_feature_names(X: pd.DataFrame) -> List[str]:
    names: List[str] = []
    for col in X.columns:
        if col in {"layergroup", "pointgroup"}:
            names.append(col)
        elif pd.api.types.is_object_dtype(X[col]) or pd.api.types.is_string_dtype(X[col]):
            names.append(col)
        elif isinstance(X[col].dtype, pd.CategoricalDtype):
            names.append(col)
    return sorted(set(names), key=list(X.columns).index)


def finalize_categorical_features(
    X: pd.DataFrame,
    cat_features: Sequence[str],
) -> pd.DataFrame:
    out = X.copy()
    for col in cat_features:
        out[col] = out[col].fillna("__MISSING__").astype(str)
    for col in out.columns:
        if col not in cat_features:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def gray_zone_weights_for_gate(
    alpha_inplane: pd.Series,
    gray_low: float,
    gray_high: float,
    gray_weight: float,
) -> np.ndarray:
    weights = np.ones(len(alpha_inplane), dtype=float)
    gray_mask = (alpha_inplane.to_numpy() > gray_low) & (
        alpha_inplane.to_numpy() < gray_high
    )
    weights[gray_mask] = gray_weight
    return weights


def positive_severity_weights_for_gate(
    alpha_inplane: pd.Series,
    alpha_threshold: float,
    mode: str,
    smooth_scale: float,
    max_weight: float,
) -> np.ndarray:
    values = alpha_inplane.to_numpy(dtype=float)
    weights = np.ones(len(values), dtype=float)
    positive_mask = values > alpha_threshold
    if mode == "none":
        return weights
    if mode == "piecewise":
        for low, high, weight in PIECEWISE_SEVERITY_BINS:
            bin_mask = positive_mask & (values > low) & (values <= high)
            weights[bin_mask] = weight
        weights[positive_mask & (values > 1000.0)] = 8.0
        return weights
    if mode == "smooth":
        if smooth_scale < 0.0:
            raise ValueError("--positive-severity-smooth-scale must be non-negative.")
        if max_weight < 1.0:
            raise ValueError("--positive-severity-max-weight must be >= 1.")
        weights[positive_mask] = np.clip(
            1.0 + smooth_scale * np.log1p(values[positive_mask] / alpha_threshold),
            1.0,
            max_weight,
        )
        return weights
    raise ValueError(f"Unknown positive severity weight mode: {mode}")


def sample_weights_for_gate(
    alpha_inplane: pd.Series,
    gray_low: float,
    gray_high: float,
    gray_weight: float,
    alpha_threshold: float,
    severity_mode: str,
    severity_smooth_scale: float,
    severity_max_weight: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray_weights = gray_zone_weights_for_gate(
        alpha_inplane,
        gray_low,
        gray_high,
        gray_weight,
    )
    severity_weights = positive_severity_weights_for_gate(
        alpha_inplane,
        alpha_threshold,
        severity_mode,
        severity_smooth_scale,
        severity_max_weight,
    )
    return gray_weights * severity_weights, gray_weights, severity_weights


def catboost_params(
    args: argparse.Namespace,
    random_seed: int,
    iterations: Optional[int] = None,
    class_weight_mode: Optional[str] = None,
    positive_class_weight: Optional[float] = None,
) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "loss_function": "Logloss",
        "eval_metric": "PRAUC",
        "iterations": int(iterations or args.iterations),
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "l2_leaf_reg": args.l2_leaf_reg,
        "random_strength": args.random_strength,
        "bagging_temperature": args.bagging_temperature,
        "border_count": args.border_count,
        "bootstrap_type": "Bayesian",
        "random_seed": random_seed,
        "task_type": args.task_type,
        "verbose": args.verbose,
        "allow_writing_files": False,
    }
    if args.devices is not None:
        params["devices"] = args.devices
    weight_mode = class_weight_mode or args.class_weight_mode
    positive_weight = (
        float(positive_class_weight)
        if positive_class_weight is not None
        else float(args.positive_class_weight)
    )
    if weight_mode == "auto":
        params["auto_class_weights"] = "Balanced"
    elif weight_mode == "manual":
        params["class_weights"] = [1.0, positive_weight]
    return params


def fit_model(
    model: CatBoostClassifier,
    train_pool: Pool,
    args: argparse.Namespace,
    eval_pool: Optional[Pool] = None,
) -> CatBoostClassifier:
    fit_kwargs: Dict[str, Any] = {}
    if eval_pool is not None:
        fit_kwargs["eval_set"] = eval_pool
        fit_kwargs["early_stopping_rounds"] = args.early_stopping_rounds
        fit_kwargs["use_best_model"] = True
    try:
        model.fit(train_pool, **fit_kwargs)
        return model
    except CatBoostError as exc:
        if args.task_type == "GPU" and args.fallback_to_cpu:
            print("WARNING: GPU training failed. Retrying on CPU.")
            print(f"Original CatBoostError: {exc}")
            params = model.get_params()
            params["task_type"] = "CPU"
            retry_model = CatBoostClassifier(**params)
            retry_model.fit(train_pool, **fit_kwargs)
            return retry_model
        raise


def positive_probabilities(
    model: CatBoostClassifier,
    X: pd.DataFrame,
    cat_features: Sequence[str],
) -> np.ndarray:
    pool = Pool(X, cat_features=list(cat_features) or None)
    raw = model.predict_proba(pool)
    class_to_pos = {int(cls): pos for pos, cls in enumerate(model.classes_)}
    if 1 not in class_to_pos:
        return np.zeros(len(X), dtype=float)
    return raw[:, class_to_pos[1]]


def safe_auc(metric_fn: Any, y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    return float(metric_fn(y_true, scores))


def safe_rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def format_optional_float(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "NA"
    numeric_value = float(value)
    if np.isnan(numeric_value):
        return "NA"
    return f"{numeric_value:.{digits}f}"


def metrics_from_predictions(
    y_true: np.ndarray,
    pred: np.ndarray,
    alpha_inplane: Optional[Sequence[float]] = None,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    pred = np.asarray(pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    positives = tp + fn
    negatives = tn + fp

    # Body regression receives predicted-body samples: true body (TN) plus leaked
    # high-alpha samples (FN). These are the gate-specific safety metrics.
    predicted_body_count = tn + fn
    true_body_count = tn + fp
    metrics: Dict[str, Any] = {
        "threshold": None if threshold is None else float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "recall": 0.0 if positives == 0 else float(tp / positives),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "specificity": 0.0 if negatives == 0 else float(tn / negatives),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "leakage_rate": 0.0 if positives == 0 else float(fn / positives),
        "high_alpha_leakage_rate": 0.0 if positives == 0 else float(fn / positives),
        "body_contamination_rate": safe_rate(int(fn), int(predicted_body_count)),
        "body_retention_rate": safe_rate(int(tn), int(true_body_count)),
        "predicted_body_count": int(predicted_body_count),
        "predicted_body_true_body_count": int(tn),
        "predicted_body_leaked_high_alpha_count": int(fn),
        "true_body_count": int(true_body_count),
        "true_body_retained_count": int(tn),
        "true_body_overflagged_count": int(fp),
        "flagged_rate": float(np.mean(pred)),
        "positive_rate_true": float(np.mean(y_true)),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    if alpha_inplane is not None:
        alpha_values = np.asarray(alpha_inplane, dtype=float)
        if alpha_values.shape[0] != y_true.shape[0]:
            raise ValueError("alpha_inplane must have the same length as y_true.")
        fn_mask = (y_true == 1) & (pred == 0)
        fn_alpha = alpha_values[fn_mask]
        for alpha_threshold in SEVERE_FN_ALPHA_THRESHOLDS:
            key_value = str(alpha_threshold).replace(".", "p")
            if key_value.endswith("p0"):
                key_value = key_value[:-2]
            metrics[f"fn_alpha_gt_{key_value}"] = int(np.sum(fn_alpha > alpha_threshold))
        metrics["fn_alpha_max"] = None if fn_alpha.size == 0 else float(np.max(fn_alpha))
        metrics["fn_alpha_mean"] = None if fn_alpha.size == 0 else float(np.mean(fn_alpha))
        metrics["fn_alpha_median"] = None if fn_alpha.size == 0 else float(np.median(fn_alpha))
    return metrics


def metrics_at_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    alpha_inplane: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(int)
    return metrics_from_predictions(
        y_true,
        pred,
        alpha_inplane=alpha_inplane,
        threshold=threshold,
    )


def build_threshold_table(
    y_true: np.ndarray,
    scores: np.ndarray,
    alpha_inplane: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    thresholds = np.unique(np.concatenate([[0.0, 0.5, 1.0], scores]))
    rows = [
        metrics_at_threshold(y_true, scores, float(thr), alpha_inplane)
        for thr in thresholds
    ]
    table = pd.DataFrame(rows)
    return table.sort_values("threshold").reset_index(drop=True)


def select_gate_threshold(
    threshold_table: pd.DataFrame,
    target_recall: float,
) -> Dict[str, Any]:
    candidates = threshold_table.loc[threshold_table["recall"] >= target_recall].copy()
    if candidates.empty:
        chosen = threshold_table.sort_values(
            ["recall", "precision", "threshold"],
            ascending=[False, False, False],
        ).iloc[0]
        result = chosen.to_dict()
        result["selection_note"] = (
            f"No threshold reached target_recall={target_recall:.4f}; "
            "selected the highest-recall threshold."
        )
        return result

    chosen = candidates.sort_values(
        ["precision", "recall", "threshold"],
        ascending=[False, False, False],
    ).iloc[0]
    result = chosen.to_dict()
    result["selection_note"] = (
        f"Selected highest precision among thresholds with recall >= {target_recall:.4f}."
    )
    return result


def median_best_iteration(models: Sequence[CatBoostClassifier], fallback: int) -> int:
    values: List[int] = []
    for model in models:
        best = model.get_best_iteration()
        if best is not None and best >= 0:
            values.append(int(best) + 1)
    if not values:
        return int(fallback)
    return max(1, int(np.median(values)))


def train_gate_oof(
    gate_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    sample_weight: np.ndarray,
    alpha_inplane: pd.Series,
    cat_features: Sequence[str],
    args: argparse.Namespace,
    target_recall: float,
    class_weight_mode: str,
    positive_class_weight: float,
    random_seed_offset: int,
) -> Dict[str, Any]:
    class_counts = np.bincount(y, minlength=2)
    usable_splits = min(args.n_splits, int(class_counts[class_counts > 0].min()))
    if usable_splits < 2:
        raise ValueError(
            f"Cannot run StratifiedKFold for {gate_name} with class counts {class_counts.tolist()}."
        )

    print(f"\n===== {gate_name}: OOF training =====")
    print(f"{gate_name} class counts [0, 1]: {class_counts.tolist()}")
    print(f"{gate_name} n_splits: {usable_splits}")

    oof_scores = np.zeros(len(y), dtype=float)
    fold_id = np.zeros(len(y), dtype=int)
    fold_models: List[CatBoostClassifier] = []
    fold_metrics: List[Dict[str, Any]] = []
    splitter = StratifiedKFold(
        n_splits=usable_splits,
        shuffle=True,
        random_state=args.random_seed + random_seed_offset,
    )

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
        print(f"\n===== {gate_name} Fold {fold}/{usable_splits} =====")
        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_train = y[train_idx]
        y_valid = y[valid_idx]
        weight_train = sample_weight[train_idx]
        weight_valid = sample_weight[valid_idx]

        train_pool = Pool(
            X_train,
            label=y_train,
            weight=weight_train,
            cat_features=list(cat_features) or None,
        )
        valid_pool = Pool(
            X_valid,
            label=y_valid,
            weight=weight_valid,
            cat_features=list(cat_features) or None,
        )
        model = CatBoostClassifier(
            **catboost_params(
                args,
                args.random_seed + random_seed_offset + fold,
                class_weight_mode=class_weight_mode,
                positive_class_weight=positive_class_weight,
            )
        )
        model = fit_model(model, train_pool, args, eval_pool=valid_pool)
        scores = positive_probabilities(model, X_valid, cat_features)
        oof_scores[valid_idx] = scores
        fold_id[valid_idx] = fold
        fold_models.append(model)

        fold_alpha = alpha_inplane.iloc[valid_idx].to_numpy(dtype=float)
        fold_table = build_threshold_table(y_valid, scores, fold_alpha)
        fold_selected = select_gate_threshold(fold_table, target_recall)
        fold_metrics.append(
            {
                "fold": fold,
                "train_size": int(len(train_idx)),
                "valid_size": int(len(valid_idx)),
                "positive_valid_count": int(y_valid.sum()),
                "best_iteration": model.get_best_iteration(),
                "selected_threshold_metrics": fold_selected,
                "pr_auc": safe_auc(average_precision_score, y_valid, scores),
                "roc_auc": safe_auc(roc_auc_score, y_valid, scores),
            }
        )
        print(
            f"{gate_name} fold selected: "
            f"threshold={fold_selected['threshold']:.6f}, "
            f"recall={fold_selected['recall']:.4f}, "
            f"precision={fold_selected['precision']:.4f}, "
            f"leakage={fold_selected['leakage_rate']:.4f}, "
            f"body_contamination={format_optional_float(fold_selected['body_contamination_rate'])}, "
            f"body_retention={format_optional_float(fold_selected['body_retention_rate'])}"
        )

    alpha_values = alpha_inplane.to_numpy(dtype=float)
    threshold_table = build_threshold_table(y, oof_scores, alpha_values)
    selected = select_gate_threshold(threshold_table, target_recall)
    threshold = float(selected["threshold"])
    metrics_selected = metrics_at_threshold(y, oof_scores, threshold, alpha_values)
    metrics_0p5 = metrics_at_threshold(y, oof_scores, 0.5, alpha_values)
    pred = (oof_scores >= threshold).astype(int)

    return {
        "gate_name": gate_name,
        "usable_splits": usable_splits,
        "oof_scores": oof_scores,
        "fold_id": fold_id,
        "fold_models": fold_models,
        "fold_metrics": fold_metrics,
        "threshold_table": threshold_table,
        "selected_gate_threshold": selected,
        "threshold": threshold,
        "metrics_at_selected_threshold": metrics_selected,
        "metrics_at_0p5_for_reference_only": metrics_0p5,
        "pred": pred,
        "pr_auc": safe_auc(average_precision_score, y, oof_scores),
        "roc_auc": safe_auc(roc_auc_score, y, oof_scores),
        "target_recall": target_recall,
        "class_weight_mode": class_weight_mode,
        "positive_class_weight": positive_class_weight,
        "class_counts": class_counts.tolist(),
    }


def conditional_gate_b_threshold_table(
    y_high: np.ndarray,
    y_severe: np.ndarray,
    gate_a_pred: np.ndarray,
    gate_b_scores: np.ndarray,
    alpha_inplane: Sequence[float],
) -> pd.DataFrame:
    y_high = np.asarray(y_high, dtype=int)
    y_severe = np.asarray(y_severe, dtype=int)
    gate_a_pred = np.asarray(gate_a_pred, dtype=int)
    gate_b_scores = np.asarray(gate_b_scores, dtype=float)
    alpha_values = np.asarray(alpha_inplane, dtype=float)
    gate_a_negative = gate_a_pred == 0
    thresholds = np.unique(np.concatenate([[0.0, 0.5, 1.0], gate_b_scores[gate_a_negative]]))
    rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        gate_b_conditional_pred = (gate_a_negative & (gate_b_scores >= threshold)).astype(int)
        combined_pred = ((gate_a_pred == 1) | (gate_b_conditional_pred == 1)).astype(int)
        high_metrics = metrics_from_predictions(
            y_high,
            combined_pred,
            alpha_inplane=alpha_values,
            threshold=float(threshold),
        )
        severe_metrics = metrics_from_predictions(
            y_severe,
            combined_pred,
            alpha_inplane=alpha_values,
            threshold=float(threshold),
        )
        gate_a_fn_mask = (y_high == 1) & (gate_a_pred == 0)
        rows.append(
            {
                "gate_b_conditional_threshold": float(threshold),
                "gate_b_conditional_positive_count": int(np.sum(gate_b_conditional_pred)),
                "gate_a_fn_rescued_by_gate_b": int(
                    np.sum(gate_a_fn_mask & (gate_b_conditional_pred == 1))
                ),
                "high_recall": high_metrics["recall"],
                "high_precision": high_metrics["precision"],
                "high_fn": high_metrics["fn"],
                "body_retention_rate": high_metrics["body_retention_rate"],
                "body_contamination_rate": high_metrics["body_contamination_rate"],
                "flagged_rate": high_metrics["flagged_rate"],
                "severe_recall": severe_metrics["recall"],
                "severe_precision": severe_metrics["precision"],
                "severe_fn": severe_metrics["fn"],
                "fn_alpha_gt_50": high_metrics.get("fn_alpha_gt_50"),
                "fn_alpha_gt_100": high_metrics.get("fn_alpha_gt_100"),
                "fn_alpha_gt_1000": high_metrics.get("fn_alpha_gt_1000"),
                "fn_alpha_max": high_metrics.get("fn_alpha_max"),
            }
        )
    return pd.DataFrame(rows).sort_values("gate_b_conditional_threshold").reset_index(drop=True)


def select_conditional_gate_b_threshold(
    conditional_table: pd.DataFrame,
    body_retention_min: float,
    body_retention_max: float,
) -> Dict[str, Any]:
    candidates = conditional_table.loc[
        (conditional_table["body_retention_rate"] >= body_retention_min)
        & (conditional_table["body_retention_rate"] <= body_retention_max)
    ].copy()
    if candidates.empty:
        constrained = conditional_table.copy()
        constrained["body_retention_distance"] = np.minimum(
            np.abs(constrained["body_retention_rate"] - body_retention_min),
            np.abs(constrained["body_retention_rate"] - body_retention_max),
        )
        chosen = constrained.sort_values(
            [
                "body_retention_distance",
                "fn_alpha_gt_1000",
                "fn_alpha_gt_100",
                "fn_alpha_gt_50",
                "severe_fn",
                "body_contamination_rate",
            ],
            ascending=[True, True, True, True, True, True],
        ).iloc[0]
        result = chosen.to_dict()
        result["selection_note"] = (
            "No Gate B conditional threshold satisfied the requested body-retention "
            f"range [{body_retention_min:.4f}, {body_retention_max:.4f}]; selected "
            "the closest retention point with severe FN tie-breakers."
        )
        return result

    chosen = candidates.sort_values(
        [
            "fn_alpha_gt_1000",
            "fn_alpha_gt_100",
            "fn_alpha_gt_50",
            "severe_fn",
            "body_contamination_rate",
            "body_retention_rate",
        ],
        ascending=[True, True, True, True, True, False],
    ).iloc[0]
    result = chosen.to_dict()
    result["selection_note"] = (
        "Selected Gate B threshold on Gate A-negative samples within body-retention "
        f"range [{body_retention_min:.4f}, {body_retention_max:.4f}], prioritizing "
        "severe FN reduction."
    )
    return result


def error_type_labels(y_true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.select(
        [
            (y_true == 1) & (pred == 1),
            (y_true == 1) & (pred == 0),
            (y_true == 0) & (pred == 1),
            (y_true == 0) & (pred == 0),
        ],
        ["TP", "FN", "FP", "TN"],
        default="UNKNOWN",
    )


def build_error_summary(oof_predictions: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["error_type"]
    agg_map: Dict[str, Tuple[str, str]] = {
        "n": ("sample_id", "count"),
        "alpha_inplane_mean": ("alpha_inplane", "mean"),
        "alpha_inplane_median": ("alpha_inplane", "median"),
        "alpha_inplane_max": ("alpha_inplane", "max"),
        "p_high_alpha_mean": ("p_high_alpha", "mean"),
        "p_high_alpha_median": ("p_high_alpha", "median"),
    }
    if "eta_parallel" in oof_predictions.columns:
        agg_map.update(
            {
                "eta_parallel_mean": ("eta_parallel", "mean"),
                "eta_parallel_median": ("eta_parallel", "median"),
                "eta_ge_0p5_count": ("eta_ge_0p5", "sum"),
                "eta_ge_0p8_count": ("eta_ge_0p8", "sum"),
            }
        )
    if "p_gate_b_severe_high_alpha" in oof_predictions.columns:
        agg_map.update(
            {
                "p_gate_b_severe_high_alpha_mean": ("p_gate_b_severe_high_alpha", "mean"),
                "p_gate_b_severe_high_alpha_median": ("p_gate_b_severe_high_alpha", "median"),
            }
        )
    summary = oof_predictions.groupby(group_cols, dropna=False).agg(**agg_map)
    return summary.reset_index()


def build_eta_route_summary(oof_predictions: pd.DataFrame) -> Optional[pd.DataFrame]:
    if "eta_parallel" not in oof_predictions.columns:
        return None
    bins = [-np.inf, 0.05, 0.2, 0.5, 0.8, np.inf]
    labels = ["eta<0.05", "0.05<=eta<0.2", "0.2<=eta<0.5", "0.5<=eta<0.8", "eta>=0.8"]
    tmp = oof_predictions.copy()
    tmp["eta_bin"] = pd.cut(tmp["eta_parallel"], bins=bins, labels=labels)
    summary = tmp.groupby(["eta_bin", "route"], dropna=False).agg(
        n=("sample_id", "count"),
        alpha_inplane_median=("alpha_inplane", "median"),
        alpha_inplane_max=("alpha_inplane", "max"),
        p_high_alpha_median=("p_high_alpha", "median"),
    )
    return summary.reset_index()


def print_label_summary(
    row_count: int,
    y: pd.Series,
    alpha_inplane: pd.Series,
    bad_uids: Sequence[str],
    alpha_threshold: float,
    severe_alpha_threshold: float,
    gray_low: float,
    gray_high: float,
    severity_mode: str,
    positive_severity_weight: Optional[np.ndarray] = None,
    sample_weight: Optional[np.ndarray] = None,
    data: Optional[pd.DataFrame] = None,
) -> None:
    positive_count = int(y.sum())
    print("===== High-alpha-inplane gate data summary =====")
    print(f"rows after merge/drop: {row_count}")
    print(f"dropped bad_uids: {list(bad_uids)}")
    print(f"alpha threshold: > {alpha_threshold}")
    print(f"high_alpha count: {positive_count}")
    severe_count = int((alpha_inplane > severe_alpha_threshold).sum())
    print(f"severe high-alpha threshold: > {severe_alpha_threshold}")
    print(f"severe high-alpha count: {severe_count}")
    print(f"severe high-alpha prevalence: {severe_count / max(len(y), 1):.4f}")
    print(f"body count: {int(len(y) - positive_count)}")
    print(f"high_alpha prevalence: {positive_count / max(len(y), 1):.4f}")
    print(
        "alpha_inplane quantiles: "
        + json.dumps(
            {
                "p50": float(alpha_inplane.quantile(0.50)),
                "p80": float(alpha_inplane.quantile(0.80)),
                "p90": float(alpha_inplane.quantile(0.90)),
                "p95": float(alpha_inplane.quantile(0.95)),
                "p99": float(alpha_inplane.quantile(0.99)),
                "max": float(alpha_inplane.max()),
            },
            ensure_ascii=False,
        )
    )
    gray_count = int(((alpha_inplane > gray_low) & (alpha_inplane < gray_high)).sum())
    print(f"gray-zone count ({gray_low}, {gray_high}): {gray_count}")
    severity_counts: Dict[str, int] = {}
    alpha_values = alpha_inplane.to_numpy(dtype=float)
    for low, high, weight in PIECEWISE_SEVERITY_BINS:
        if np.isinf(high):
            label = f"alpha>{low:g} weight={weight:g}"
            mask = alpha_values > low
        else:
            label = f"{low:g}<alpha<={high:g} weight={weight:g}"
            mask = (alpha_values > low) & (alpha_values <= high)
        severity_counts[label] = int(mask.sum())
    print(f"positive severity weighting mode: {severity_mode}")
    print("positive severity bin counts: " + json.dumps(severity_counts, ensure_ascii=False))
    if positive_severity_weight is not None and sample_weight is not None:
        print(
            "weight summary: "
            + json.dumps(
                {
                    "positive_severity_weight_min": float(np.min(positive_severity_weight)),
                    "positive_severity_weight_max": float(np.max(positive_severity_weight)),
                    "sample_weight_min": float(np.min(sample_weight)),
                    "sample_weight_max": float(np.max(sample_weight)),
                    "sample_weight_mean": float(np.mean(sample_weight)),
                },
                ensure_ascii=False,
            )
        )

    if data is not None and "eta_parallel" in data.columns:
        eta = data["eta_parallel"]
        eta_ge_05 = int((eta >= 0.5).sum())
        eta_ge_08 = int((eta >= 0.8).sum())
        high_alpha_eta_ge_05 = int(((data["high_alpha"] == 1) & (eta >= 0.5)).sum())
        print(
            "eta_parallel summary: "
            + json.dumps(
                {
                    "p50": None if eta.isna().all() else float(eta.quantile(0.50)),
                    "p75": None if eta.isna().all() else float(eta.quantile(0.75)),
                    "p95": None if eta.isna().all() else float(eta.quantile(0.95)),
                    "eta>=0.5_count": eta_ge_05,
                    "eta>=0.8_count": eta_ge_08,
                    "high_alpha_and_eta>=0.5_count": high_alpha_eta_ge_05,
                },
                ensure_ascii=False,
            )
        )


def main() -> None:
    args = parse_args()
    if not (0.0 < args.target_recall <= 1.0):
        raise ValueError("--target-recall must be in (0, 1].")
    if not (0.0 < args.severe_target_recall <= 1.0):
        raise ValueError("--severe-target-recall must be in (0, 1].")
    if not (0.0 <= args.combined_body_retention_min <= args.combined_body_retention_max <= 1.0):
        raise ValueError(
            "--combined-body-retention-min/max must satisfy 0 <= min <= max <= 1."
        )
    if args.severe_alpha_threshold <= args.alpha_threshold:
        raise ValueError("--severe-alpha-threshold must be higher than --alpha-threshold.")
    if not (0.0 <= args.gray_weight <= 1.0):
        raise ValueError("--gray-weight must be in [0, 1].")
    if args.gray_low >= args.gray_high:
        raise ValueError("--gray-low must be lower than --gray-high.")
    if args.positive_severity_smooth_scale < 0.0:
        raise ValueError("--positive-severity-smooth-scale must be non-negative.")
    if args.positive_severity_max_weight < 1.0:
        raise ValueError("--positive-severity-max-weight must be >= 1.")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    reports_dir = output_dir / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    data, X, y_series, alpha_inplane, sample_ids, suspect_cols = read_inputs(
        args.labels,
        args.descriptors,
        args.alpha_threshold,
        args.bad_uids,
        args.allow_suspect_feature_names,
    )
    y_severe = (alpha_inplane.to_numpy(dtype=float) > args.severe_alpha_threshold).astype(int)
    data["severe_high_alpha"] = y_severe
    cat_features = categorical_feature_names(X)
    X = finalize_categorical_features(X, cat_features)
    sample_weight, gray_zone_weight, positive_severity_weight = sample_weights_for_gate(
        alpha_inplane,
        args.gray_low,
        args.gray_high,
        args.gray_weight,
        args.alpha_threshold,
        args.positive_severity_weight_mode,
        args.positive_severity_smooth_scale,
        args.positive_severity_max_weight,
    )

    print_label_summary(
        len(data),
        y_series,
        alpha_inplane,
        args.bad_uids,
        args.alpha_threshold,
        args.severe_alpha_threshold,
        args.gray_low,
        args.gray_high,
        args.positive_severity_weight_mode,
        positive_severity_weight,
        sample_weight,
        data=data,
    )
    print(f"feature count: {X.shape[1]}")
    print(f"categorical features: {cat_features}")
    print(f"Gate A class_weight_mode: {args.class_weight_mode}")
    print(f"Gate B class_weight_mode: {args.severe_class_weight_mode}")
    if suspect_cols:
        print(f"suspect feature names allowed: {suspect_cols}")
    if args.dry_run:
        return

    require_catboost()

    y = y_series.to_numpy(dtype=int)
    severe_sample_weight = positive_severity_weights_for_gate(
        alpha_inplane,
        args.severe_alpha_threshold,
        args.positive_severity_weight_mode,
        args.positive_severity_smooth_scale,
        args.positive_severity_max_weight,
    )

    gate_a = train_gate_oof(
        "gate_a_high_alpha",
        X,
        y,
        sample_weight,
        alpha_inplane,
        cat_features,
        args,
        target_recall=args.target_recall,
        class_weight_mode=args.class_weight_mode,
        positive_class_weight=args.positive_class_weight,
        random_seed_offset=0,
    )
    gate_b = train_gate_oof(
        "gate_b_severe_high_alpha",
        X,
        y_severe,
        severe_sample_weight,
        alpha_inplane,
        cat_features,
        args,
        target_recall=args.severe_target_recall,
        class_weight_mode=args.severe_class_weight_mode,
        positive_class_weight=args.severe_positive_class_weight,
        random_seed_offset=5000,
    )

    alpha_values_all = alpha_inplane.to_numpy(dtype=float)
    oof_scores = gate_a["oof_scores"]
    oof_scores_severe = gate_b["oof_scores"]
    selected = gate_a["selected_gate_threshold"]
    selected_severe = gate_b["selected_gate_threshold"]
    gate_threshold = float(gate_a["threshold"])
    severe_gate_threshold_independent = float(gate_b["threshold"])
    selected_metrics = gate_a["metrics_at_selected_threshold"]
    selected_metrics_severe = gate_b["metrics_at_selected_threshold"]
    metrics_at_0p5 = gate_a["metrics_at_0p5_for_reference_only"]
    metrics_at_0p5_severe = gate_b["metrics_at_0p5_for_reference_only"]
    selected_pred_a = gate_a["pred"]
    selected_pred_b_independent = gate_b["pred"]
    conditional_gate_b_table = conditional_gate_b_threshold_table(
        y,
        y_severe,
        selected_pred_a,
        oof_scores_severe,
        alpha_values_all,
    )
    conditional_gate_b_selection = select_conditional_gate_b_threshold(
        conditional_gate_b_table,
        args.combined_body_retention_min,
        args.combined_body_retention_max,
    )
    severe_gate_threshold = float(conditional_gate_b_selection["gate_b_conditional_threshold"])
    gate_a_negative = selected_pred_a == 0
    selected_pred_b = (
        gate_a_negative & (oof_scores_severe >= severe_gate_threshold)
    ).astype(int)
    selected_pred = ((selected_pred_a == 1) | (selected_pred_b == 1)).astype(int)
    combined_metrics = metrics_from_predictions(
        y,
        selected_pred,
        alpha_inplane=alpha_values_all,
    )
    combined_severe_metrics = metrics_from_predictions(
        y_severe,
        selected_pred,
        alpha_inplane=alpha_values_all,
    )
    gate_a_fn_mask = (y == 1) & (selected_pred_a == 0)
    gate_a_fn_rescued_by_gate_b = int(np.sum(gate_a_fn_mask & (selected_pred_b == 1)))

    gate_a["threshold_table"].to_csv(reports_dir / "gate_a_threshold_table_oof.csv", index=False)
    gate_b["threshold_table"].to_csv(reports_dir / "gate_b_threshold_table_oof.csv", index=False)
    conditional_gate_b_table.to_csv(
        reports_dir / "gate_b_conditional_threshold_table_oof.csv",
        index=False,
    )
    # Backward-compatible alias for earlier single-gate experiments.
    gate_a["threshold_table"].to_csv(reports_dir / "threshold_table_oof.csv", index=False)
    operating_point_metrics = pd.DataFrame(
        [
            {"operating_point": "gate_a_selected_oof_threshold", **selected_metrics},
            {"operating_point": "gate_a_threshold_0p5_reference", **metrics_at_0p5},
            {"operating_point": "gate_b_independent_selected_oof_threshold", **selected_metrics_severe},
            {"operating_point": "gate_b_threshold_0p5_reference", **metrics_at_0p5_severe},
            {"operating_point": "gate_b_conditional_selected_threshold", **conditional_gate_b_selection},
            {"operating_point": "combined_or_route_for_high_alpha", **combined_metrics},
            {"operating_point": "combined_or_route_for_severe_high_alpha", **combined_severe_metrics},
        ]
    )
    operating_point_metrics.to_csv(
        reports_dir / "gate_operating_point_metrics.csv",
        index=False,
    )

    oof_payload: Dict[str, Any] = {
        "sample_id": sample_ids.to_numpy(),
        "gate_a_fold_id": gate_a["fold_id"],
        "gate_b_fold_id": gate_b["fold_id"],
        "alpha_inplane": alpha_inplane.to_numpy(dtype=float),
        "true_high_alpha": y,
        "true_severe_high_alpha": y_severe,
        "gray_zone_weight": gray_zone_weight,
        "positive_severity_weight": positive_severity_weight,
        "severe_sample_weight": severe_sample_weight,
        "sample_weight": sample_weight,
        "p_high_alpha": oof_scores,
        "p_gate_a_high_alpha": oof_scores,
        "p_gate_b_severe_high_alpha": oof_scores_severe,
        "gate_threshold": gate_threshold,
        "gate_a_threshold": gate_threshold,
        "gate_b_threshold": severe_gate_threshold,
        "gate_b_independent_threshold": severe_gate_threshold_independent,
        "gate_b_conditional_threshold": severe_gate_threshold,
        "pred_gate_a_high_alpha": selected_pred_a,
        "pred_gate_b_independent_severe_high_alpha": selected_pred_b_independent,
        "pred_gate_b_severe_high_alpha": selected_pred_b,
        "pred_gate_b_conditional_severe_high_alpha": selected_pred_b,
        "pred_high_alpha_gate": selected_pred,
        "route": np.where(selected_pred == 1, "high_alpha_candidate", "body_regression"),
        "gate_a_error_type": error_type_labels(y, selected_pred_a),
        "gate_b_independent_error_type": error_type_labels(y_severe, selected_pred_b_independent),
        "gate_b_error_type": error_type_labels(y_severe, selected_pred_b),
        "gate_b_conditional_error_type": error_type_labels(y_severe, selected_pred_b),
        "error_type": error_type_labels(y, selected_pred),
        "alpha_distance_to_threshold": alpha_inplane.to_numpy(dtype=float) - args.alpha_threshold,
        "alpha_distance_to_severe_threshold": alpha_inplane.to_numpy(dtype=float) - args.severe_alpha_threshold,
    }
    for col in ["alpha_xx", "alpha_yy", "alpha_xy", "alpha_zz", "d", "b", "eta_parallel"]:
        if col in data.columns:
            oof_payload[col] = data[col].to_numpy(dtype=float)
    if "eta_parallel" in data.columns:
        eta_values = data["eta_parallel"].to_numpy(dtype=float)
        oof_payload["eta_ge_0p2"] = (eta_values >= 0.2).astype(int)
        oof_payload["eta_ge_0p5"] = (eta_values >= 0.5).astype(int)
        oof_payload["eta_ge_0p8"] = (eta_values >= 0.8).astype(int)

    oof_predictions = pd.DataFrame(oof_payload)
    oof_predictions.to_csv(reports_dir / "oof_gate_predictions.csv", index=False)

    error_summary = build_error_summary(oof_predictions)
    error_summary.to_csv(reports_dir / "error_summary.csv", index=False)
    oof_predictions.loc[oof_predictions["error_type"] == "FN"].to_csv(
        reports_dir / "false_negative_high_alpha_leakage_cases.csv",
        index=False,
    )
    oof_predictions.loc[oof_predictions["error_type"] == "FP"].to_csv(
        reports_dir / "false_positive_overflagged_body_cases.csv",
        index=False,
    )
    eta_route_summary = build_eta_route_summary(oof_predictions)
    if eta_route_summary is not None:
        eta_route_summary.to_csv(reports_dir / "eta_route_summary.csv", index=False)

    final_iterations_a = median_best_iteration(gate_a["fold_models"], args.iterations)
    final_iterations_b = median_best_iteration(gate_b["fold_models"], args.iterations)
    print("\n===== Training final full-data Gate A model =====")
    print(f"Gate A final iterations from OOF median best iteration: {final_iterations_a}")
    final_pool_a = Pool(
        X,
        label=y,
        weight=sample_weight,
        cat_features=list(cat_features) or None,
    )
    final_model_a = CatBoostClassifier(
        **catboost_params(
            args,
            args.random_seed + 1000,
            iterations=final_iterations_a,
            class_weight_mode=args.class_weight_mode,
            positive_class_weight=args.positive_class_weight,
        )
    )
    final_model_a = fit_model(final_model_a, final_pool_a, args, eval_pool=None)
    model_path_a = models_dir / "gate_a_high_alpha_classifier.cbm"
    final_model_a.save_model(str(model_path_a))
    # Backward-compatible alias for earlier single-gate experiments.
    model_path = models_dir / "high_alpha_inplane_gate_classifier.cbm"
    final_model_a.save_model(str(model_path))

    print("\n===== Training final full-data Gate B model =====")
    print(f"Gate B final iterations from OOF median best iteration: {final_iterations_b}")
    final_pool_b = Pool(
        X,
        label=y_severe,
        weight=severe_sample_weight,
        cat_features=list(cat_features) or None,
    )
    final_model_b = CatBoostClassifier(
        **catboost_params(
            args,
            args.random_seed + 6000,
            iterations=final_iterations_b,
            class_weight_mode=args.severe_class_weight_mode,
            positive_class_weight=args.severe_positive_class_weight,
        )
    )
    final_model_b = fit_model(final_model_b, final_pool_b, args, eval_pool=None)
    model_path_b = models_dir / "gate_b_severe_high_alpha_classifier.cbm"
    final_model_b.save_model(str(model_path_b))

    feature_importance_a = pd.DataFrame(
        {
            "feature": list(X.columns),
            "importance": final_model_a.get_feature_importance(final_pool_a),
        }
    ).sort_values("importance", ascending=False)
    feature_importance_a.to_csv(reports_dir / "gate_a_feature_importance.csv", index=False)
    # Backward-compatible alias.
    feature_importance_a.to_csv(reports_dir / "feature_importance.csv", index=False)
    feature_importance_b = pd.DataFrame(
        {
            "feature": list(X.columns),
            "importance": final_model_b.get_feature_importance(final_pool_b),
        }
    ).sort_values("importance", ascending=False)
    feature_importance_b.to_csv(reports_dir / "gate_b_feature_importance.csv", index=False)

    metrics = {
        "task": "high_alpha_inplane_gate_classifier",
        "label_definition": {
            "alpha_inplane": "(alpha_xx + alpha_yy) / 2",
            "gate_a_positive_class": f"alpha_inplane > {args.alpha_threshold}",
            "gate_b_positive_class": f"alpha_inplane > {args.severe_alpha_threshold}",
            "body_class": f"alpha_inplane <= {args.alpha_threshold}",
        },
        "paths": {
            "labels": args.labels.expanduser().resolve(),
            "descriptors": args.descriptors.expanduser().resolve(),
            "output_dir": output_dir,
            "gate_a_model": model_path_a,
            "gate_a_model_compatibility_alias": model_path,
            "gate_b_model": model_path_b,
        },
        "data": {
            "row_count": len(y),
            "dropped_bad_uids": list(args.bad_uids),
            "gate_a_positive_count": int(y.sum()),
            "gate_b_positive_count": int(y_severe.sum()),
            "body_count": int(len(y) - y.sum()),
            "gate_a_positive_rate": float(np.mean(y)),
            "gate_b_positive_rate": float(np.mean(y_severe)),
            "gray_zone": {
                "low": args.gray_low,
                "high": args.gray_high,
                "weight": args.gray_weight,
                "count": int(
                    ((alpha_inplane > args.gray_low) & (alpha_inplane < args.gray_high)).sum()
                ),
            },
            "positive_severity_weighting": {
                "mode": args.positive_severity_weight_mode,
                "piecewise_bins": [
                    {
                        "alpha_gt": low,
                        "alpha_lte": None if np.isinf(high) else high,
                        "weight": weight,
                        "count": int(
                            (
                                (alpha_inplane.to_numpy(dtype=float) > low)
                                & (
                                    np.ones(len(alpha_inplane), dtype=bool)
                                    if np.isinf(high)
                                    else (alpha_inplane.to_numpy(dtype=float) <= high)
                                )
                            ).sum()
                        ),
                    }
                    for low, high, weight in PIECEWISE_SEVERITY_BINS
                ],
                "smooth_scale": args.positive_severity_smooth_scale,
                "max_weight": args.positive_severity_max_weight,
                "severity_weight_min": float(np.min(positive_severity_weight)),
                "severity_weight_max": float(np.max(positive_severity_weight)),
                "final_sample_weight_min": float(np.min(sample_weight)),
                "final_sample_weight_max": float(np.max(sample_weight)),
                "final_sample_weight_mean": float(np.mean(sample_weight)),
            },
        },
        "feature_safety": {
            "suspect_feature_columns": suspect_cols,
            "allow_suspect_feature_names": bool(args.allow_suspect_feature_names),
        },
        "training": {
            "gate_a_n_splits": gate_a["usable_splits"],
            "gate_b_n_splits": gate_b["usable_splits"],
            "gate_a_catboost_params": catboost_params(
                args,
                args.random_seed,
                class_weight_mode=args.class_weight_mode,
                positive_class_weight=args.positive_class_weight,
            ),
            "gate_b_catboost_params": catboost_params(
                args,
                args.random_seed,
                class_weight_mode=args.severe_class_weight_mode,
                positive_class_weight=args.severe_positive_class_weight,
            ),
            "gate_a_final_iterations": final_iterations_a,
            "gate_b_final_iterations": final_iterations_b,
            "categorical_features": cat_features,
            "feature_count": X.shape[1],
        },
        "oof_global": {
            "gate_a": {
                "pr_auc": gate_a["pr_auc"],
                "roc_auc": gate_a["roc_auc"],
                "selected_gate_threshold": selected,
                "metrics_at_selected_threshold": selected_metrics,
                "metrics_at_0p5_for_reference_only": metrics_at_0p5,
            },
            "gate_b": {
                "pr_auc": gate_b["pr_auc"],
                "roc_auc": gate_b["roc_auc"],
                "independent_selected_gate_threshold": selected_severe,
                "conditional_selected_gate_threshold": conditional_gate_b_selection,
                "metrics_at_selected_threshold": selected_metrics_severe,
                "metrics_at_0p5_for_reference_only": metrics_at_0p5_severe,
            },
            "combined_or": {
                "metrics_for_gate_a_high_alpha_label": combined_metrics,
                "metrics_for_gate_b_severe_label": combined_severe_metrics,
                "gate_a_fn_rescued_by_gate_b": gate_a_fn_rescued_by_gate_b,
            },
        },
        "folds": {
            "gate_a": gate_a["fold_metrics"],
            "gate_b": gate_b["fold_metrics"],
        },
        "inference_rule": {
            "p_gate_a_high_alpha": "gate_a_model.predict_proba(X)[:, 1]",
            "p_gate_b_severe_high_alpha": "gate_b_model.predict_proba(X)[:, 1]",
            "gate_a_positive_if": f"p_gate_a_high_alpha >= {gate_threshold:.12g}",
            "gate_b_positive_if": (
                "gate_a_negative AND "
                f"p_gate_b_severe_high_alpha >= {severe_gate_threshold:.12g}"
            ),
            "route_to_high_alpha_candidate_if": "gate_a_positive OR conditional_gate_b_positive",
            "route_to_body_regression_if": "NOT (gate_a_positive OR conditional_gate_b_positive)",
            "note": (
                "Do not use 0.5 as the default gate threshold. Gate B threshold is "
                "selected conditionally on Gate A-negative OOF samples."
            ),
        },
        "reports": {
            "oof_predictions": reports_dir / "oof_gate_predictions.csv",
            "threshold_table": reports_dir / "threshold_table_oof.csv",
            "gate_a_threshold_table": reports_dir / "gate_a_threshold_table_oof.csv",
            "gate_b_threshold_table": reports_dir / "gate_b_threshold_table_oof.csv",
            "gate_b_conditional_threshold_table": reports_dir / "gate_b_conditional_threshold_table_oof.csv",
            "operating_point_metrics": reports_dir / "gate_operating_point_metrics.csv",
            "error_summary": reports_dir / "error_summary.csv",
            "false_negative_cases": reports_dir / "false_negative_high_alpha_leakage_cases.csv",
            "false_positive_cases": reports_dir / "false_positive_overflagged_body_cases.csv",
            "eta_route_summary": reports_dir / "eta_route_summary.csv",
            "feature_importance": reports_dir / "feature_importance.csv",
            "gate_a_feature_importance": reports_dir / "gate_a_feature_importance.csv",
            "gate_b_feature_importance": reports_dir / "gate_b_feature_importance.csv",
        },
    }
    save_json(output_dir / "high_alpha_gate_metrics.json", metrics)
    save_json(
        output_dir / "high_alpha_gate_manifest.json",
        {
            "gate_a_model_file": model_path_a,
            "gate_a_model_compatibility_alias": model_path,
            "gate_b_model_file": model_path_b,
            "gate_a_selected_threshold": gate_threshold,
            "gate_b_independent_selected_threshold": severe_gate_threshold_independent,
            "gate_b_conditional_selected_threshold": severe_gate_threshold,
            "gate_b_conditional_selection": conditional_gate_b_selection,
            "gate_a_target_recall": args.target_recall,
            "gate_b_target_recall": args.severe_target_recall,
            "combined_body_retention_range": {
                "min": args.combined_body_retention_min,
                "max": args.combined_body_retention_max,
            },
            "feature_columns": list(X.columns),
            "categorical_features": cat_features,
            "label_definition": metrics["label_definition"],
            "bad_uids": list(args.bad_uids),
            "gray_zone_weighting": metrics["data"]["gray_zone"],
            "positive_severity_weighting": metrics["data"]["positive_severity_weighting"],
            "feature_safety": metrics["feature_safety"],
            "inference_rule": metrics["inference_rule"],
            "reports": metrics["reports"],
        },
    )

    print("\n===== Final OOF two-gate summary =====")
    print(f"Gate A PR-AUC: {metrics['oof_global']['gate_a']['pr_auc']}")
    print(f"Gate A ROC-AUC: {metrics['oof_global']['gate_a']['roc_auc']}")
    print(f"Gate A selected threshold: {gate_threshold:.6f}")
    print(f"recall: {selected_metrics['recall']:.4f}")
    print(f"precision: {selected_metrics['precision']:.4f}")
    print(f"leakage rate: {selected_metrics['leakage_rate']:.4f}")
    print(
        "body contamination: "
        f"{format_optional_float(selected_metrics['body_contamination_rate'])}"
    )
    print(
        "body retention: "
        f"{format_optional_float(selected_metrics['body_retention_rate'])}"
    )
    print(
        "severe FN counts: "
        f">20={selected_metrics['fn_alpha_gt_20']}, "
        f">30={selected_metrics['fn_alpha_gt_30']}, "
        f">50={selected_metrics['fn_alpha_gt_50']}, "
        f">100={selected_metrics['fn_alpha_gt_100']}, "
        f">1000={selected_metrics['fn_alpha_gt_1000']}"
    )
    print(f"FN alpha max: {selected_metrics['fn_alpha_max']}")
    print(f"Gate A flagged rate: {selected_metrics['flagged_rate']:.4f}")
    print(f"Gate B PR-AUC: {metrics['oof_global']['gate_b']['pr_auc']}")
    print(f"Gate B ROC-AUC: {metrics['oof_global']['gate_b']['roc_auc']}")
    print(f"Gate B independent selected threshold: {severe_gate_threshold_independent:.6f}")
    print(f"Gate B independent recall: {selected_metrics_severe['recall']:.4f}")
    print(f"Gate B independent precision: {selected_metrics_severe['precision']:.4f}")
    print(f"Gate B independent leakage rate: {selected_metrics_severe['leakage_rate']:.4f}")
    print(f"Gate B conditional selected threshold: {severe_gate_threshold:.6f}")
    print(
        "Gate B conditional selection: "
        f"body_retention={format_optional_float(conditional_gate_b_selection['body_retention_rate'])}, "
        f"body_contamination={format_optional_float(conditional_gate_b_selection['body_contamination_rate'])}, "
        f"severe_fn={int(conditional_gate_b_selection['severe_fn'])}, "
        f"fn_alpha_gt_1000={int(conditional_gate_b_selection['fn_alpha_gt_1000'])}"
    )
    print("\n===== Combined OR route summary =====")
    print(f"combined recall for high-alpha label: {combined_metrics['recall']:.4f}")
    print(f"combined precision for high-alpha label: {combined_metrics['precision']:.4f}")
    print(f"combined leakage rate: {combined_metrics['leakage_rate']:.4f}")
    print(
        "combined body contamination: "
        f"{format_optional_float(combined_metrics['body_contamination_rate'])}"
    )
    print(
        "combined body retention: "
        f"{format_optional_float(combined_metrics['body_retention_rate'])}"
    )
    print(f"Gate A FN rescued by Gate B: {gate_a_fn_rescued_by_gate_b}")
    print(f"Gate A model saved to: {model_path_a}")
    print(f"Gate B model saved to: {model_path_b}")
    print(f"metrics saved to: {output_dir / 'high_alpha_gate_metrics.json'}")
    print(f"OOF predictions saved to: {reports_dir / 'oof_gate_predictions.csv'}")
    print(f"error summary saved to: {reports_dir / 'error_summary.csv'}")


if __name__ == "__main__":
    main()
