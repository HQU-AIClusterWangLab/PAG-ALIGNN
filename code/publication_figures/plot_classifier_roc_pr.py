"""Plot ROC and precision-recall curves for the CatBoost/ALIGNN classifiers.

The default inputs are the saved five-fold out-of-fold (OOF) predictions.
The three models are evaluated on their common sample ids so that every curve
is directly comparable.

The output figure contains:
(a) ROC curves
(b) Precision-recall curves

The figure does not contain internal subplot titles. The subplot meanings
are described in the LaTeX caption.

Use --layout vertical for an 85 mm wide compact single-column figure with ROC above PR.
Use --output-dir to preserve existing horizontal exports.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from publication_panel_labels import add_panel_labels
from sklearn.metrics import (
    average_precision_score,
    auc,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


# Project root:
# E:\Yjlong\ALIGNN
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CAT_OOF = (
    PROJECT_ROOT
    / "cat"
    / "high_alpha_inplane_gate"
    / "reports"
    / "oof_gate_predictions.csv"
)

DEFAULT_ALIGNN_OOF = (
    PROJECT_ROOT
    / "runs"
    / "inplane_avg_gt_16p1_classifier_02"
    / "cv_prediction_results_test_set.csv"
)

DEFAULT_FUSION_OOF = (
    PROJECT_ROOT
    / "runs"
    / "cat_alignn_post_fusion_02"
    / "fusion_oof_predictions.csv"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "runs"
    / "classification_curves"
)

# ---------------------------------------------------------------------
# Global figure style
# ---------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 8.5,
        "axes.linewidth": 1.1,
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot ROC and PR curves from CatBoost/ALIGNN OOF predictions."
    )

    parser.add_argument(
        "--cat-oof",
        default=DEFAULT_CAT_OOF,
        help="CatBoost OOF prediction CSV.",
    )

    parser.add_argument(
        "--alignn-oof",
        default=DEFAULT_ALIGNN_OOF,
        help="ALIGNN OOF prediction CSV.",
    )

    parser.add_argument(
        "--fusion-oof",
        default=DEFAULT_FUSION_OOF,
        help="Fusion OOF prediction CSV.",
    )

    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory.",
    )

    parser.add_argument(
        "--cat-prob-column",
        default="p_gate_a_high_alpha",
        help="CatBoost probability column.",
    )

    parser.add_argument(
        "--alignn-prob-column",
        default="probability",
        help="ALIGNN probability column.",
    )

    parser.add_argument(
        "--fusion-prob-column",
        default="p_final",
        help="Fusion probability column.",
    )

    parser.add_argument(
        "--cat-weight",
        type=float,
        default=0.96,
        help="CatBoost weight in the fusion model.",
    )

    parser.add_argument(
        "--alignn-weight",
        type=float,
        default=0.04,
        help="ALIGNN weight in the fusion model.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Resolution of the PNG output.",
    )

    parser.add_argument("--layout", choices=("horizontal", "vertical"), default="horizontal",
                        help="Side-by-side double-column or stacked single-column panels.")
    parser.add_argument("--width-mm", type=float)
    parser.add_argument("--height-mm", type=float)

    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _first_column(
    rows: Iterable[dict[str, str]],
    candidates: list[str],
) -> str:
    rows = list(rows)

    if not rows:
        raise ValueError("CSV contains no data rows.")

    available = set(rows[0])

    for candidate in candidates:
        if candidate in available:
            return candidate

    raise ValueError(
        f"None of the columns {candidates!r} was found. "
        f"Available columns: {sorted(available)!r}"
    )


def _as_probability(
    value: str,
    path: Path,
    column: str,
) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid probability {value!r} in "
            f"{path} column {column!r}."
        ) from exc

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            f"Probability {probability} in {path} column {column!r} "
            "is outside [0, 1]."
        )

    return probability


def _load_model_rows(
    path: Path,
    probability_column: str,
    model_name: str,
) -> dict[str, dict[str, float | int]]:
    rows = _read_csv(path)

    if not rows:
        raise ValueError(f"CSV contains no rows: {path}")

    id_column = _first_column(
        rows,
        ["sample_id", "id", "jid", "uid"],
    )

    target_column = _first_column(
        rows,
        ["target", "label", "true_high_alpha"],
    )

    if probability_column not in rows[0]:
        raise ValueError(
            f"Column {probability_column!r} was not found in {path}. "
            f"Available columns: {sorted(rows[0])!r}"
        )

    loaded: dict[str, dict[str, float | int]] = {}

    for row in rows:
        sample_id = str(row[id_column]).strip()

        if not sample_id:
            continue

        target = int(float(row[target_column]))

        probability = _as_probability(
            row[probability_column],
            path,
            probability_column,
        )

        if target not in (0, 1):
            raise ValueError(
                f"Invalid binary target {target!r} in {path}."
            )

        if sample_id in loaded:
            raise ValueError(
                f"Duplicate sample id {sample_id!r} in {path}."
            )

        loaded[sample_id] = {
            "target": target,
            model_name: probability,
        }

    return loaded


def _merge_predictions(
    cat_rows: dict[str, dict[str, float | int]],
    alignn_rows: dict[str, dict[str, float | int]],
    fusion_rows: dict[str, dict[str, float | int]],
    cat_weight: float,
    alignn_weight: float,
) -> list[dict[str, float | int | str]]:
    common_ids = sorted(
        set(cat_rows)
        & set(alignn_rows)
        & set(fusion_rows)
    )

    if not common_ids:
        raise ValueError(
            "No common sample ids were found across the three OOF files."
        )

    merged: list[dict[str, float | int | str]] = []

    for sample_id in common_ids:
        target = int(cat_rows[sample_id]["target"])

        if target != int(alignn_rows[sample_id]["target"]):
            raise ValueError(
                "Target mismatch between CatBoost and ALIGNN "
                f"for {sample_id!r}."
            )

        if target != int(fusion_rows[sample_id]["target"]):
            raise ValueError(
                "Target mismatch between CatBoost and fusion "
                f"for {sample_id!r}."
            )

        p_cat = float(cat_rows[sample_id]["catboost"])
        p_alignn = float(alignn_rows[sample_id]["alignn"])

        p_fusion = (
            cat_weight * p_cat
            + alignn_weight * p_alignn
        )

        merged.append(
            {
                "sample_id": sample_id,
                "target": target,
                "p_catboost": p_cat,
                "p_alignn": p_alignn,
                "p_fusion": p_fusion,
            }
        )

    return merged


def _write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def _curve_rows(
    targets: np.ndarray,
    probabilities: np.ndarray,
    model_name: str,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
]:
    fpr, tpr, roc_thresholds = roc_curve(
        targets,
        probabilities,
    )

    precision, recall, pr_thresholds = precision_recall_curve(
        targets,
        probabilities,
    )

    roc_rows = [
        {
            "model": model_name,
            "fpr": float(x),
            "tpr": float(y),
            "threshold": float(threshold),
        }
        for x, y, threshold in zip(
            fpr,
            tpr,
            roc_thresholds,
        )
    ]

    pr_rows = [
        {
            "model": model_name,
            "recall": float(x),
            "precision": float(y),
            "threshold": (
                float(pr_thresholds[index])
                if index < len(pr_thresholds)
                else ""
            ),
        }
        for index, (x, y) in enumerate(
            zip(recall, precision)
        )
    ]

    return roc_rows, pr_rows


def _metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    fpr, tpr, _ = roc_curve(
        targets,
        probabilities,
    )

    precision, recall, _ = precision_recall_curve(
        targets,
        probabilities,
    )

    return {
        "roc_auc": float(
            roc_auc_score(targets, probabilities)
        ),
        "roc_curve_auc": float(
            auc(fpr, tpr)
        ),
        "average_precision": float(
            average_precision_score(
                targets,
                probabilities,
            )
        ),
        "pr_curve_auc": float(
            auc(recall, precision)
        ),
    }


def _plot(
    targets: np.ndarray,
    model_probabilities: dict[str, np.ndarray],
    metrics: dict[str, dict[str, float]],
    output_path: Path,
    dpi: int,
    layout: str = "horizontal",
    width_mm: float | None = None,
    height_mm: float | None = None,
) -> None:
    colors = {
        "CatBoost Gate A": "#d95f02",
        "ALIGNN": "#1b9e77",
        "Weighted Fusion": "#7570b3",
    }

    vertical = layout == "vertical"
    width_mm = width_mm or (85 if vertical else 177)
    height_mm = height_mm or (116 if vertical else 82)
    label_size, tick_size, legend_size = (9, 8, 7.5) if vertical else (11, 10, 8.5)
    # Size the canvas and fonts for the final physical insertion width.
    fig, axes = plt.subplots(
        2 if vertical else 1,
        1 if vertical else 2,
        figsize=(width_mm / 25.4, height_mm / 25.4),
        constrained_layout=not vertical,
    )

    roc_axis, pr_axis = axes

    for name, probabilities in model_probabilities.items():
        color = colors[name]

        display_name = (
            "CatBoost"
            if name == "CatBoost Gate A"
            else name
        )

        fpr, tpr, _ = roc_curve(
            targets,
            probabilities,
        )

        precision, recall, _ = precision_recall_curve(
            targets,
            probabilities,
        )

        roc_axis.plot(
            fpr,
            tpr,
            color=color,
            linewidth=1.4 if vertical else 2.2,
            label=(
                f"{display_name} "
                f"(AUC={metrics[name]['roc_auc']:.4f})"
            ),
        )

        pr_axis.plot(
            recall,
            precision,
            color=color,
            linewidth=1.4 if vertical else 2.2,
            label=(
                f"{display_name} "
                f"(AP={metrics[name]['average_precision']:.4f})"
            ),
        )

    positive_rate = float(np.mean(targets))

    roc_axis.plot(
        [0, 1],
        [0, 1],
        color="#888888",
        linewidth=1.2,
        linestyle="--",
    )

    pr_axis.axhline(
        positive_rate,
        color="#888888",
        linewidth=1.2,
        linestyle="--",
        label=(
            f"No-skill "
            f"(prevalence={positive_rate:.3f})"
        ),
    )

    # -------------------------------------------------------------
    # Panel (a): ROC curve
    # No internal subplot title is used.
    # -------------------------------------------------------------

    roc_axis.set_xlabel(
        "False positive rate",
        fontsize=label_size,
        labelpad=5,
    )

    roc_axis.set_ylabel(
        "True positive rate",
        fontsize=label_size,
        labelpad=5,
    )

    roc_axis.set_xlim(0, 1)
    roc_axis.set_ylim(0, 1.02)

    roc_axis.tick_params(
        axis="both",
        which="major",
        labelsize=tick_size,
    )

    roc_axis.grid(False)

    roc_axis.set_axisbelow(True)

    roc_axis.legend(
        loc="lower right",
        bbox_to_anchor=(0.99, 0.01),
        fontsize=legend_size - 0.5 if vertical else legend_size,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1.0,
        borderpad=0.4,
        labelspacing=0.45,
        handlelength=1.5 if vertical else 2.4,
    )

    roc_axis.text(
        -0.10,
        1.02,
        "(a)",
        transform=roc_axis.transAxes,
        fontsize=label_size,
        fontfamily="Times New Roman",
        va="bottom",
        ha="right",
        clip_on=False,
    )

    # -------------------------------------------------------------
    # Panel (b): Precision-recall curve
    # No internal subplot title is used.
    # -------------------------------------------------------------

    pr_axis.set_xlabel(
        "Recall",
        fontsize=label_size,
        labelpad=5,
    )

    pr_axis.set_ylabel(
        "Precision",
        fontsize=label_size,
        labelpad=5,
    )

    pr_axis.set_xlim(0, 1)
    pr_axis.set_ylim(0, 1.02)

    pr_axis.tick_params(
        axis="both",
        which="major",
        labelsize=tick_size,
    )

    pr_axis.grid(False)

    pr_axis.set_axisbelow(True)

    pr_axis.legend(
        loc="upper left",
        bbox_to_anchor=(0.04, 0.55 if vertical else 0.47),
        fontsize=legend_size,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=1.0,
        borderpad=0.4,
        labelspacing=0.45,
        handlelength=1.5 if vertical else 2.4,
    )

    pr_axis.text(
        -0.10,
        1.02,
        "(b)",
        transform=pr_axis.transAxes,
        fontsize=label_size,
        fontfamily="Times New Roman",
        va="bottom",
        ha="right",
        clip_on=False,
    )

    if vertical:
        # Lay out in millimetres; remove whitespace without rescaling fonts.
        for axis in axes:
            axis.texts[0].set_visible(False)
            axis.texts[0].set_verticalalignment("top")
        plot_height = (height_mm - 32.0) / 2.0
        top_bottom = height_mm - 3.0 - plot_height
        lower_bottom = 14.0

        def place_axes():
            roc_axis.set_position([17.5 / width_mm, top_bottom / height_mm,
                                   (width_mm - 19.5) / width_mm, plot_height / height_mm])
            pr_axis.set_position([17.5 / width_mm, lower_bottom / height_mm,
                                  (width_mm - 19.5) / width_mm, plot_height / height_mm])

        place_axes()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        gap_mm = (roc_axis.get_tightbbox(renderer).y0 -
                  pr_axis.get_tightbbox(renderer).y1) * 25.4 / fig.dpi
        removed_space = gap_mm - 3.0
        top_bottom -= removed_space
        height_mm -= removed_space
        height_mm += 3.0  # Headroom for the shared publication labels.
        fig.set_size_inches(width_mm / 25.4, height_mm / 25.4, forward=True)
        place_axes()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        final_gap = (roc_axis.get_tightbbox(renderer).y0 -
                     pr_axis.get_tightbbox(renderer).y1) * 25.4 / fig.dpi
        print(f"Canvas: {width_mm:.3f} x {height_mm:.3f} mm; panel-content gap: {final_gap:.3f} mm")
        roc_box = roc_axis.get_legend().get_window_extent(renderer).transformed(roc_axis.transData.inverted())
        pr_box = pr_axis.get_legend().get_window_extent(renderer).transformed(pr_axis.transData.inverted())
        print(f"ROC legend bounds (data coordinates): {roc_box.bounds}")
        if roc_box.y1 >= roc_box.x0:
            raise ValueError("ROC legend intersects the y=x reference line; reposition it.")
        if pr_box.y0 <= positive_rate <= pr_box.y1:
            raise ValueError("PR legend intersects the prevalence reference line; reposition it.")

    if vertical:
        add_panel_labels(fig, list(axes), ["(a)", "(b)"])

    # Save both PNG and PDF.
    # PDF is recommended for LaTeX because text and lines remain vector-based.
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_output_path = output_path.with_suffix(".pdf")

    fig.savefig(
        output_path,
        dpi=dpi,
    )

    fig.savefig(
        pdf_output_path,
    )

    print(f"Output PNG: {output_path}")
    print(f"Output PDF: {pdf_output_path}")

    plt.close(fig)


def main() -> None:
    args = parse_args()

    if args.cat_weight < 0 or args.alignn_weight < 0:
        raise ValueError(
            "Fusion weights must be non-negative."
        )

    if args.cat_weight + args.alignn_weight <= 0:
        raise ValueError(
            "At least one fusion weight must be positive."
        )

    cat_path = Path(args.cat_oof)
    alignn_path = Path(args.alignn_oof)
    fusion_path = Path(args.fusion_oof)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cat_rows = _load_model_rows(
        cat_path,
        args.cat_prob_column,
        "catboost",
    )

    alignn_rows = _load_model_rows(
        alignn_path,
        args.alignn_prob_column,
        "alignn",
    )

    fusion_rows_from_file = _read_csv(fusion_path)

    if not fusion_rows_from_file:
        raise ValueError(
            f"Fusion CSV contains no rows: {fusion_path}"
        )

    fusion_probability_column = (
        args.fusion_prob_column
        if args.fusion_prob_column in fusion_rows_from_file[0]
        else "p_cat_gate_a"
    )

    fusion_rows = _load_model_rows(
        fusion_path,
        fusion_probability_column,
        "fusion_reference",
    )

    merged = _merge_predictions(
        cat_rows,
        alignn_rows,
        fusion_rows,
        args.cat_weight,
        args.alignn_weight,
    )

    targets = np.asarray(
        [int(row["target"]) for row in merged],
        dtype=int,
    )

    model_probabilities = {
        "CatBoost Gate A": np.asarray(
            [float(row["p_catboost"]) for row in merged],
            dtype=float,
        ),
        "ALIGNN": np.asarray(
            [float(row["p_alignn"]) for row in merged],
            dtype=float,
        ),
        "Weighted Fusion": np.asarray(
            [float(row["p_fusion"]) for row in merged],
            dtype=float,
        ),
    }

    model_metrics = {
        name: _metrics(
            targets,
            probabilities,
        )
        for name, probabilities in model_probabilities.items()
    }

    aligned_path = output_dir / "aligned_oof_predictions.csv"

    _write_csv(
        aligned_path,
        merged,
        [
            "sample_id",
            "target",
            "p_catboost",
            "p_alignn",
            "p_fusion",
        ],
    )

    roc_rows: list[dict[str, object]] = []
    pr_rows: list[dict[str, object]] = []

    for name, probabilities in model_probabilities.items():
        model_roc, model_pr = _curve_rows(
            targets,
            probabilities,
            name,
        )

        roc_rows.extend(model_roc)
        pr_rows.extend(model_pr)

    _write_csv(
        output_dir / "roc_curve_points.csv",
        roc_rows,
        [
            "model",
            "fpr",
            "tpr",
            "threshold",
        ],
    )

    _write_csv(
        output_dir / "pr_curve_points.csv",
        pr_rows,
        [
            "model",
            "recall",
            "precision",
            "threshold",
        ],
    )

    report = {
        "data_source": (
            "five_fold_out_of_fold_predictions"
        ),
        "n_common_samples": len(merged),
        "positive_count": int(np.sum(targets)),
        "negative_count": int(
            len(targets) - np.sum(targets)
        ),
        "fusion_formula": (
            f"{args.cat_weight:.12g} * p_cat_gate_a + "
            f"{args.alignn_weight:.12g} * p_alignn"
        ),
        "inputs": {
            "catboost_oof": str(cat_path),
            "alignn_oof": str(alignn_path),
            "fusion_oof_reference": str(fusion_path),
        },
        "metrics": model_metrics,
        "outputs": {
            "plot_png": str(
                output_dir / "classifier_roc_pr.png"
            ),
            "plot_pdf": str(
                output_dir / "classifier_roc_pr.pdf"
            ),
            "aligned_oof_predictions": str(
                aligned_path
            ),
            "roc_curve_points": str(
                output_dir / "roc_curve_points.csv"
            ),
            "pr_curve_points": str(
                output_dir / "pr_curve_points.csv"
            ),
        },
    }

    (
        output_dir / "classifier_roc_pr_metrics.json"
    ).write_text(
        json.dumps(
            report,
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    _plot(
        targets,
        model_probabilities,
        model_metrics,
        output_dir / "classifier_roc_pr.png",
        args.dpi,
        layout=args.layout,
        width_mm=args.width_mm,
        height_mm=args.height_mm,
    )

    print(
        json.dumps(
            report,
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
