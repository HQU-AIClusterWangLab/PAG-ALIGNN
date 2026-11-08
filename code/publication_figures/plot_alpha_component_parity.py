#!/usr/bin/env python
"""Plot double-column OOF density/parity panels and residual strips for xx/yy/zz.

The in-plane source is a five-fold directory containing evaluated CSV OOF
predictions (under ``fold_evaluations``).
The out-of-plane zz source points to the corresponding dataset directory.  That
directory contains the fold data, while the associated reconstructed OOF
target/prediction table is stored in ``comparison_outputs``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator, FixedLocator
import numpy as np
from publication_panel_labels import add_panel_labels


plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": False,
    }
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPLANE_SOURCE = ROOT / "supplementary_data/D07_tensor_predictions.csv"
DEFAULT_ZZ_SOURCE = DEFAULT_INPLANE_SOURCE
DEFAULT_ZZ_OOF_TABLE = DEFAULT_INPLANE_SOURCE
DEFAULT_OUTPUT = ROOT / "work/figures/F06.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot true-versus-predicted OOF alpha component parity plots."
    )
    parser.add_argument(
        "--inplane-predictions",
        type=Path,
        default=DEFAULT_INPLANE_SOURCE,
        help="Five-fold directory or prediction file for xx, yy, and xy.",
    )
    parser.add_argument(
        "--zz-predictions",
        type=Path,
        default=DEFAULT_ZZ_SOURCE,
        help="Out-of-plane zz dataset directory or prediction file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PNG file.",
    )
    parser.add_argument("--dpi", type=int, default=600, help="PNG resolution.")
    parser.add_argument("--width-mm", type=float, default=177.0, help="Final figure width in mm (default: double-column 177).")
    parser.add_argument("--height-mm", type=float, default=85.0, help="Final figure height in mm.")
    parser.add_argument("--gridsize", type=int, default=28, help="Hexagonal density-bin resolution.")
    parser.add_argument(
        "--title",
        default="",
        help="Figure title.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also display the figure after writing the PNG.",
    )
    return parser.parse_args()


def _finite_pairs(
    pairs: list[tuple[float, float]], source: Path, component: str
) -> tuple[np.ndarray, np.ndarray]:
    if not pairs:
        raise ValueError(f"No finite {component} target/prediction pairs in {source}")
    targets, predictions = zip(*pairs)
    return np.asarray(targets), np.asarray(predictions)


def _load_csv_component(csv_path: Path, component: str) -> tuple[np.ndarray, np.ndarray]:
    """Load finite target/prediction values for one component from a CSV."""
    target_key = f"target_{component}"
    prediction_key = f"prediction_{component}"
    pairs: list[tuple[float, float]] = []
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Prediction CSV has no header: {csv_path}")
        required = {target_key, prediction_key}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{csv_path} is missing columns: {', '.join(sorted(missing))}"
            )
        for row_index, row in enumerate(reader, start=2):
            try:
                target = float(row[target_key])
                prediction = float(row[prediction_key])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid {component} value in {csv_path} at row {row_index}."
                ) from exc
            if math.isfinite(target) and math.isfinite(prediction):
                pairs.append((target, prediction))
    return _finite_pairs(pairs, csv_path, component)


def _flatten_values(values: object) -> list[float]:
    """Flatten the one-row/list variants emitted by ALIGNN evaluators."""
    while isinstance(values, list) and len(values) == 1 and isinstance(values[0], list):
        values = values[0]
    if not isinstance(values, list):
        raise ValueError("Prediction record values must be lists.")
    return [float(value) for value in values]


def _load_json_component(json_path: Path, component: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one component from ALIGNN multi-output/Test_results JSON."""
    component_index = {"xx": 0, "yy": 1, "xy": 2, "zz": 0}[component]
    records = json.loads(json_path.read_text(encoding="utf-8"))
    pairs: list[tuple[float, float]] = []
    for record in records:
        target_values = record.get("target", record.get("target_out"))
        prediction_values = record.get("predictions", record.get("pred_out"))
        if target_values is None or prediction_values is None:
            continue
        target_values = _flatten_values(target_values)
        prediction_values = _flatten_values(prediction_values)
        if component_index >= len(target_values) or component_index >= len(prediction_values):
            raise ValueError(f"{json_path} has no {component} component in a prediction record.")
        target = target_values[component_index]
        prediction = prediction_values[component_index]
        if math.isfinite(target) and math.isfinite(prediction):
            pairs.append((target, prediction))
    return _finite_pairs(pairs, json_path, component)


def _resolve_prediction_csv(source: Path, component: str) -> Path:
    """Resolve a prediction table for an out-of-plane dataset directory.

    The requested zz source is a raw data directory.  Prefer its associated
    reconstructed OOF table and never silently substitute the legacy
    ``runs/outplane_zz_best`` result (which is a different evaluation).
    """
    candidates: list[Path] = []
    if source == DEFAULT_ZZ_SOURCE.resolve():
        candidates.append(DEFAULT_ZZ_OOF_TABLE)
    candidates.append(source / "submodel_evaluation" / "predictions.csv")
    candidates.extend(sorted(source.rglob("predictions.csv")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"No prediction CSV found for {component} source {source}. Searched:\n{searched}"
    )


def load_component(source: Path, component: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a component from a CSV, JSON file, or model fold directory."""
    source = source.resolve()
    if source.is_dir():
        # Prefer the explicit fold-evaluation OOF tables.  The same run also
        # contains ``fold_*/multi_out_predictions.json`` from a different
        # evaluation stage; using those would produce a different metric set.
        evaluation_root = source / "fold_evaluations"
        # The supplied in-plane path ends in ``.../folds``; in this run the
        # evaluated OOF tables are in the sibling ``.../fold_evaluations``.
        if not evaluation_root.is_dir() and source.name.lower() == "folds":
            evaluation_root = source.parent / "fold_evaluations"
        fold_csvs = sorted(evaluation_root.glob("fold_*/predictions.csv"))
        if fold_csvs:
            all_targets: list[np.ndarray] = []
            all_predictions: list[np.ndarray] = []
            for csv_path in fold_csvs:
                targets, predictions = _load_csv_component(csv_path, component)
                all_targets.append(targets)
                all_predictions.append(predictions)
            return np.concatenate(all_targets), np.concatenate(all_predictions)

        fold_jsons = sorted(source.glob("fold_*/multi_out_predictions.json"))
        if fold_jsons:
            all_targets: list[np.ndarray] = []
            all_predictions: list[np.ndarray] = []
            for json_path in fold_jsons:
                targets, predictions = _load_json_component(json_path, component)
                all_targets.append(targets)
                all_predictions.append(predictions)
            return np.concatenate(all_targets), np.concatenate(all_predictions)
        source = _resolve_prediction_csv(source, component)

    if not source.is_file():
        raise FileNotFoundError(f"Prediction source not found: {source}")
    if source.suffix.lower() == ".json":
        return _load_json_component(source, component)
    return _load_csv_component(source, component)


def calculate_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    residuals = predictions - targets
    mse = float(np.mean(residuals**2))
    return {
        "mae": float(np.mean(np.abs(residuals))),
        "rmse": float(np.sqrt(mse)),
    }


def parity_limits(targets: np.ndarray, predictions: np.ndarray) -> tuple[float, float]:
    lower = float(min(np.min(targets), np.min(predictions)))
    upper = float(max(np.max(targets), np.max(predictions)))
    span = upper - lower
    padding = 0.05 * span if span > 0 else max(abs(lower) * 0.05, 1e-6)
    return lower - padding, upper + padding


def draw_density_panel(axis, residual_axis, component, targets, predictions,
                       limits, residual_limit, gridsize):
    """Retain every finite observation; density colors are counts, not uncertainty."""
    lower, upper = limits
    density = axis.hexbin(
        targets, predictions, gridsize=gridsize,
        extent=(lower, upper, lower, upper), mincnt=1,
        cmap="YlGnBu", linewidths=0, rasterized=True, zorder=2,
    )
    # Recover sparsely occupied hexagons and show their observations as points.
    centers = density.get_offsets()
    counts = np.asarray(density.get_array())
    span = upper - lower
    distances = ((targets[:, None] - centers[None, :, 0]) / span) ** 2
    distances += ((predictions[:, None] - centers[None, :, 1]) / span) ** 2
    sparse = counts[np.argmin(distances, axis=1)] <= 2
    axis.scatter(targets[sparse], predictions[sparse], s=4.5,
                 color="#287a8e", linewidths=0, alpha=0.75,
                 rasterized=True, zorder=3)
    axis.plot([lower, upper], [lower, upper], color="#383838",
              ls="--", lw=0.8, zorder=4)
    axis.set(xlim=limits, ylim=limits)
    axis.set_aspect("equal", adjustable="box")
    axis.tick_params(labelbottom=False, direction="out", length=2.5, pad=2)
    axis.grid(False)
    axis.set_facecolor("white")
    metrics = calculate_metrics(targets, predictions)
    axis.text(0.045, 0.955,
              f"MAE  {metrics['mae']:.4f} Å\nRMSE  {metrics['rmse']:.4f} Å",
              transform=axis.transAxes, va="top", fontsize=7,
              bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    residuals = predictions - targets
    residual_axis.scatter(targets, residuals, s=3.5, color="#287a8e",
                          alpha=0.38, linewidths=0, rasterized=True, zorder=2)
    residual_axis.axhline(0, color="#bd5a32", ls="--", lw=0.75, zorder=3)
    residual_axis.set(xlim=limits, ylim=(-residual_limit, residual_limit),
                      xlabel="")
    residual_axis.set_yticks([-residual_limit, 0, residual_limit])
    residual_axis.tick_params(direction="out", length=2.5, pad=2)
    residual_axis.grid(False)
    residual_axis.set_facecolor("white")
    for ax in (axis, residual_axis):
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.get_offset_text().set_fontsize(6)
    return density, metrics, int(sparse.sum())


def main() -> None:
    args = parse_args()
    if args.dpi <= 0 or args.width_mm <= 0 or args.height_mm <= 0:
        raise ValueError("DPI and figure dimensions must be positive.")
    if args.gridsize < 8:
        raise ValueError("--gridsize must be at least 8.")
    components = ("xx", "yy", "zz")
    sources = {"xx": args.inplane_predictions, "yy": args.inplane_predictions,
               "zz": args.zz_predictions}
    data = {c: load_component(sources[c], c) for c in components}
    inplane_limits = parity_limits(
        np.concatenate([data[c][0] for c in ("xx", "yy")]),
        np.concatenate([data[c][1] for c in ("xx", "yy")]))
    limits = {"xx": inplane_limits, "yy": inplane_limits,
              "zz": parity_limits(*data["zz"])}
    # Use common in-plane scales. Keep all zz observations on its own scale.
    residual_limits = {}
    for group in (("xx", "yy"), ("zz",)):
        extreme = max(float(np.max(np.abs(data[c][1] - data[c][0]))) for c in group)
        step = 10 ** math.floor(math.log10(extreme)) if extreme > 0 else 1
        rounded = math.ceil(extreme * 1.06 / step) * step
        for c in group:
            residual_limits[c] = rounded
    residual_limits["zz"] = 1.0
    figure = plt.figure(figsize=(args.width_mm / 25.4, args.height_mm / 25.4))
    grid = figure.add_gridspec(2, 3, left=0.075, right=0.925,
                              bottom=0.16, top=0.865,
                              height_ratios=(3.25, 1), hspace=0.12, wspace=0.36)
    densities, axes, residual_axes = [], [], []
    metrics_by_component = {}
    for i, c in enumerate(components):
        ax = figure.add_subplot(grid[0, i])
        residual_ax = figure.add_subplot(grid[1, i], sharex=ax)
        density, metrics, sparse_n = draw_density_panel(
            ax, residual_ax, c, *data[c], limits[c], residual_limits[c], args.gridsize)
        ax.set_title(rf"$\alpha_{{{c}}}$", loc="left", fontsize=9, pad=5)
        metrics_by_component[c] = dict(metrics, n=len(data[c][0]), sparse_n=sparse_n)
        densities.append(density)
        axes.append(ax)
        residual_axes.append(residual_ax)
    axes[0].set_ylabel("Predicted (Å)")
    residual_axes[0].set_ylabel(r"$\Delta\alpha$ (Å)", fontsize=8)
    maximum = max(float(np.max(h.get_array())) for h in densities)
    norm = LogNorm(vmin=1, vmax=max(2, maximum))
    for h in densities:
        h.set_norm(norm)
    color_axis = figure.add_axes([0.945, 0.445, 0.012, 0.31])
    colorbar = figure.colorbar(densities[0], cax=color_axis)
    colorbar.set_label("Samples / hexagon", fontsize=12, labelpad=3)
    colorbar.ax.tick_params(labelsize=11, length=2, pad=2)
    colorbar.minorticks_off()
    # Sample counts are documented in the LaTeX caption and metadata.
    if args.title:
        figure.text(0.075, 0.965, args.title, fontsize=8, va="top")
    # Match strip widths to square parity axes without changing the canvas size.
    figure.canvas.draw()
    for ax, strip in zip(axes, residual_axes):
        box, strip_box = ax.get_position(), strip.get_position()
        strip.set_position([box.x0, strip_box.y0, box.width, strip_box.height])
    # Enlarge text only, after establishing the reference figure geometry.
    # Freeze tick locations: AutoLocator would otherwise remove ticks when
    # the font grows, even though no axis limits or positions were changed.
    for ax in (*axes, *residual_axes):
        ax.xaxis.set_major_locator(FixedLocator(ax.get_xticks()))
        ax.yaxis.set_major_locator(FixedLocator(ax.get_yticks()))
        ax.tick_params(axis="both", labelsize=11)
        ax.xaxis.label.set_fontsize(12)
        ax.yaxis.label.set_fontsize(12)
        ax.xaxis.get_offset_text().set_fontsize(11)
        ax.yaxis.get_offset_text().set_fontsize(11)
    for ax in axes:
        ax.set_title(ax.get_title(loc="left"), loc="left", fontsize=12, pad=5)
        for annotation in ax.texts:
            annotation.set_fontsize(11)
    for annotation in figure.texts:
        annotation.set_fontsize(12)
    # Compact equal-sized panels with a 5 mm gap at the final manuscript size.
    scale = 165.1 / args.width_mm
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    # Measure the next panel's tick text so 5 mm is clear space, not text overlap.
    tick_indent = max((ax.bbox.x0 - tick.get_window_extent(renderer).x0)
                      for ax in [*axes[1:], *residual_axes[1:]]
                      for tick in ax.get_yticklabels() if tick.get_visible()) * 25.4 / figure.dpi
    gap = 5.0 / scale + tick_indent
    side = 39.0
    start = 14.0
    for i, (ax, strip) in enumerate(zip(axes, residual_axes)):
        x = start + i * (side + gap)
        ax.set_position([x/args.width_mm, 27.5/args.height_mm,
                         side/args.width_mm, side/args.height_mm])
        strip.set_position([x/args.width_mm, 8.5/args.height_mm,
                            side/args.width_mm, 13./args.height_mm])
    # Four readable zz ticks replace the crowded quarter-unit labels.
    residual_axes[2].set_xticks([0., .4, .8, 1.2])
    end = start + 3*side + 2*gap
    color_axis.set_position([(end+2.8)/args.width_mm, 27.5/args.height_mm,
                             2.1/args.width_mm, side/args.height_mm])
    add_panel_labels(figure, axes, ["(a)", "(b)", "(c)"], insertion_scale=scale)
    output = args.output.resolve()
    if output.suffix.lower() not in (".png", ".pdf"):
        raise ValueError("--output must end in .png or .pdf.")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Do not use bbox_inches='tight': it changes the requested publication size.
    figure.savefig(output.with_suffix(".png"), dpi=args.dpi, facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), dpi=args.dpi, facecolor="white")
    metadata = {
        "width_mm": args.width_mm, "height_mm": args.height_mm, "dpi": args.dpi,
        "components": list(components), "gridsize": args.gridsize,
        "panel_gap_at_insertion_mm": 5., "panel_label_final_pt": 12.,
        "density": "Samples per hexagon; shared logarithmic color scale",
        "sparse_observations": "Counts <= 2 in nearest hexagon",
        "residual": "prediction - reference", "metrics": metrics_by_component,
        "sources": {c: str(p.resolve()) for c, p in sources.items()},
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved: {output.with_suffix('.png')}")
    print(f"Saved: {output.with_suffix('.pdf')}")
    print(f"Canvas: {args.width_mm:g} × {args.height_mm:g} mm at {args.dpi} dpi")
    for c, metrics in metrics_by_component.items():
        print(f"alpha_{c}: N={metrics['n']}, MAE={metrics['mae']:.8g}, RMSE={metrics['rmse']:.8g}")
    if args.show:
        plt.show()
    plt.close(figure)


if __name__ == "__main__":
    main()
