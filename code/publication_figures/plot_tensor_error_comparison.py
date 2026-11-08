#!/usr/bin/env python
"""Figure 7: paired OOF tensor errors, paired differences and grouped boxes.

Run from this project with its plotting Python environment:
    python visualization_tools/plot_tensor_error_comparison.py

Compact stacked single-column version (85 mm wide, 9 pt, 3 mm panel-content gap):
    python visualization_tools/plot_tensor_error_comparison.py --layout vertical --output outputs/F07_single_column.png

The two CSVs must contain unique material IDs and the six target/prediction
columns. Alignment is by ID, never CSV row order. All IDs must match; missing
samples, nonfinite entries and inconsistent reference tensors are errors.
If a sibling EATGNN training directory is present, its five-fold test JSONs
are included after recovering the recorded split and checking every tensor.

Frobenius errors use weights (1,1,1,2,2,2) for (xx,yy,zz,xy,xz,yz).
Response bins use LABEL alpha_parallel=(target_xx+target_yy)/2, not predictions.
The default bins are descriptive response ranges, not classifier labels.
Boxes summarize identical paired cohorts for both models: median, quartiles,
1.5-IQR whiskers, and visible outliers. No confidence intervals are implied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator
import numpy as np
from publication_panel_labels import add_panel_labels
import pandas as pd

COMPONENTS = ("xx", "yy", "zz", "xy", "xz", "yz")
WEIGHTS = np.array((1., 1., 1., 2., 2., 2.))
ROOT = Path(__file__).resolve().parents[1]
COLORS = {"baseline": "#F57C00", "eatgnn": "#A64DFF", "pag": "#0066FF"}
NAMES = {"baseline": "ALIGNN baseline", "eatgnn": "EATGNN", "pag": "PAG-ALIGNN"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument("--pag-predictions", type=Path)
    p.add_argument("--baseline-predictions", type=Path)
    p.add_argument("--eatgnn-root", type=Path,
                   help="EATGNN training directory; defaults to the sibling EATGNN directory if present.")
    p.add_argument("--no-eatgnn", action="store_true", help="Draw only the original two models.")
    p.add_argument("--output", type=Path, help="PNG/PDF output path; both formats are written.")
    p.add_argument("--bin-edges", type=float, nargs="+", default=[1., 5., 10.],
                   help="Upper response-bin boundaries in angstroms (default: 1 5 10).")
    p.add_argument("--layout", choices=("horizontal", "vertical"), default="horizontal",
                   help="Side-by-side double-column or stacked single-column panels.")
    p.add_argument("--width-mm", type=float)
    p.add_argument("--height-mm", type=float)
    p.add_argument("--font-size", type=float,
                   help="Defaults to 9 pt for vertical layout and 12 pt for horizontal layout.")
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--error-scale", choices=("log", "linear", "symlog"), default="log",
                   help="Absolute-error scale in panel (b).")
    p.add_argument("--delta-linthresh", type=float, default=.25,
                   help="Panel (a) symmetric-log linear half-width in angstroms.")
    return p.parse_args()


def load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"id": str, "sample_id": str})
    if "id" not in frame and "sample_id" in frame:
        frame = frame.rename(columns={"sample_id": "id"})
    columns = [f"{prefix}_{c}" for prefix in ("target", "prediction") for c in COMPONENTS]
    missing = set(["id", *columns]) - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if frame["id"].isna().any() or frame["id"].str.strip().eq("").any():
        raise ValueError(f"{path}: blank material IDs")
    if frame["id"].duplicated().any():
        raise ValueError(f"{path}: duplicate material IDs")
    frame[columns] = frame[columns].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[columns].to_numpy()).all():
        raise ValueError(f"{path}: nonfinite tensor values")
    if frame.empty:
        raise ValueError(f"{path}: empty predictions")
    return frame.set_index("id")


def align_and_measure(pag_path, baseline_path):
    pag, baseline = load_predictions(pag_path), load_predictions(baseline_path)
    if set(pag.index) != set(baseline.index):
        raise ValueError("The models have different material IDs: "
                         f"PAG-only={len(set(pag.index)-set(baseline.index))}, "
                         f"baseline-only={len(set(baseline.index)-set(pag.index))}.")
    baseline = baseline.loc[pag.index]
    targets = ["target_" + c for c in COMPONENTS]
    if not np.allclose(pag[targets], baseline[targets], rtol=1e-7, atol=1e-8):
        raise ValueError("Aligned material IDs have inconsistent reference tensors.")
    result = pd.DataFrame(index=pag.index)
    result["label_alpha_parallel"] = (pag.target_xx + pag.target_yy) / 2.
    for key, frame in (("pag", pag), ("baseline", baseline)):
        errors = np.column_stack([frame["prediction_"+c]-frame["target_"+c]
                                  for c in COMPONENTS])
        result["error_"+key] = np.sqrt(np.sum(WEIGHTS * errors**2, axis=1))
    result["paired_delta_error"] = result.error_pag - result.error_baseline
    return result


def bin_labels(edges):
    labels = [rf"$\leq {edges[0]:g}$"]
    labels.extend(rf"${a:g}\!\mathrm{{-}}\!{b:g}$" for a, b in zip(edges[:-1], edges[1:]))
    labels.append(rf"$>{edges[-1]:g}$")
    return labels


def load_eatgnn_oof(training_root: Path, canonical: pd.DataFrame):
    """Recover IDs from the recorded CV protocol, checking every exported tensor.

    This does not match predictions by nearest tensor or assume CSV row order.
    It reproduces sorted structure/label UID intersection and outer split from
    EATGNN/eatgnn/data.py. The JSON test order is then verified against labels.
    Neither training nor changes to the EATGNN directory are performed.
    """
    from sklearn.model_selection import KFold, StratifiedKFold
    cv = training_root / "checkpoints_eatgnn_5fold"
    first_config = json.loads((cv / "fold_01/training_config_resolved.json").read_text())
    label_path = Path(first_config["label_path"])
    structure_path = Path(first_config["structure_path"])
    if not label_path.exists():
        label_path = training_root / label_path.name
    if not structure_path.exists():
        structure_path = training_root / structure_path.name
    uid_key = first_config.get("label_uid_key", "uid")
    labels = pd.read_csv(label_path, dtype={uid_key: str}).set_index(uid_key)
    if labels.index.has_duplicates:
        raise ValueError("EATGNN training labels contain duplicate IDs.")
    structures = json.loads(structure_path.read_text())
    structure_key = first_config.get("structure_uid_key", "uid")
    if not isinstance(structures, list):
        raise ValueError("Expected the recorded EATGNN list-of-structures format.")
    uids = sorted(set(labels.index) & {str(r[structure_key]) for r in structures})
    if first_config.get("n") is not None:
        uids = uids[:int(first_config["n"])]
    if set(uids) != set(canonical.index):
        raise ValueError("EATGNN training cohort does not match the canonical OOF cohort.")
    matrix_columns = first_config.get("target_matrix_columns", [
        ["xx", "xy", "xz"], ["yx", "yy", "yz"], ["zx", "zy", "zz"]])
    raw = labels.loc[uids, [c for row in matrix_columns for c in row]].to_numpy(float).reshape(-1, 3, 3)
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite EATGNN training targets.")
    nfold = int(first_config["_cv_folds"])
    seed = int(first_config["_cv_split_seed"])
    # Match training's float32 classification before StratifiedKFold.
    t = raw.astype(np.float32)
    xx, yy, zz, xy = t[:, 0, 0], t[:, 1, 1], t[:, 2, 2], t[:, 0, 1]
    denom = xx + yy
    diag = np.divide(abs(xx-yy), denom, out=np.full_like(denom, np.inf), where=abs(denom)>1e-12)
    shear = np.divide(2*abs(xy), denom, out=np.full_like(denom, np.inf), where=abs(denom)>1e-12)
    clusters = np.where(xx+yy+zz < first_config.get("cluster_trace_threshold", .001), 3,
                        np.where(shear > first_config.get("cluster_xy_threshold", .3), 1,
                                 np.where(diag > first_config.get("cluster_diag_threshold", .3), 2, 0)))
    counts = np.bincount(clusters)
    stratified = (first_config.get("stratified_split", True)
                  and first_config.get("stratify_by") == "tensor_cluster"
                  and counts[counts>0].min() >= nfold)
    splitter = (StratifiedKFold(n_splits=nfold, shuffle=True, random_state=seed) if stratified
                else KFold(n_splits=nfold, shuffle=True, random_state=seed))
    folds = list(splitter.split(np.arange(len(uids)), clusters) if stratified
                 else splitter.split(np.arange(len(uids))))
    pairs = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
    canonical_cols = ["target_"+c for c in COMPONENTS]
    records_out, audit_folds = [], []
    for fold_index, (_, test_indices) in enumerate(folds):
        folder = cv / f"fold_{fold_index+1:02d}"
        config = json.loads((folder / "training_config_resolved.json").read_text())
        if (config["_cv_fold_index"] != fold_index or config["_cv_folds"] != nfold
                or config["_cv_split_seed"] != seed):
            raise ValueError(f"{folder}: inconsistent recorded fold protocol.")
        prediction_path = folder / "test_predictions_best.json"
        records = sorted(json.loads(prediction_path.read_text()), key=lambda r:r["test_order_index"])
        if len(records) != len(test_indices) or [r["test_order_index"] for r in records] != list(range(len(records))):
            raise ValueError(f"{prediction_path}: incomplete or duplicated test-order indices.")
        target = np.array([r["target_tensor"] for r in records], dtype=float)
        pred = np.array([r["pred_tensor"] for r in records], dtype=float)
        mask = np.array([r["mask_tensor"] for r in records], dtype=float)
        if target.shape != (len(records), 3, 3) or pred.shape != target.shape or mask.shape != target.shape:
            raise ValueError(f"{prediction_path}: invalid tensor shapes.")
        if not np.isfinite(target).all() or not np.isfinite(pred).all() or not (mask == 1).all():
            raise ValueError(f"{prediction_path}: nonfinite predictions or incomplete tensor masks.")
        if not np.allclose(target, raw[test_indices], rtol=1e-5, atol=1e-6):
            raise ValueError(f"{prediction_path}: recovered IDs fail exported-label verification.")
        if not np.allclose(pred, pred.transpose(0, 2, 1), rtol=1e-7, atol=1e-8):
            raise ValueError(f"{prediction_path}: asymmetric predicted tensors.")
        fold_uids = [uids[i] for i in test_indices]
        exported6 = np.column_stack([target[:, i, j] for i, j in pairs])
        canonical6 = canonical.loc[fold_uids, canonical_cols].to_numpy(float)
        if not np.allclose(exported6, canonical6, rtol=1e-6, atol=1e-7):
            raise ValueError(f"{prediction_path}: reference tensors differ beyond float32 precision.")
        full_errors = np.linalg.norm(pred-target, axis=(1, 2))
        saved_metrics = json.loads((folder / "test_metrics_best.json").read_text())
        if not np.isclose(full_errors.mean(), saved_metrics["frobenius_absolute_error"], rtol=1e-5, atol=1e-7):
            raise ValueError(f"{prediction_path}: recomputed errors disagree with saved test metrics.")
        for k, uid in enumerate(fold_uids):
            row = {"id": uid, "fold": fold_index+1, "test_order_index": k}
            for c, (i, j), reference in zip(COMPONENTS, pairs, canonical6[k]):
                # Use a common high-precision label for errors across all models;
                # retain the exported float32 label separately for auditing.
                row["target_"+c] = reference
                row["exported_target_"+c] = target[k, i, j]
                row["prediction_"+c] = pred[k, i, j]
            records_out.append(row)
        audit_folds.append({"fold": fold_index+1, "n": len(records), "prediction_file": str(prediction_path),
                            "export_vs_training_target_max_abs": float(abs(target-raw[test_indices]).max()),
                            "export_vs_canonical_target_max_abs": float(abs(exported6-canonical6).max()),
                            "saved_frobenius_mean_A": saved_metrics["frobenius_absolute_error"]})
    frame = pd.DataFrame(records_out).set_index("id")
    if frame.index.has_duplicates or set(frame.index) != set(canonical.index):
        raise ValueError("EATGNN OOF IDs are not unique and exhaustive.")
    audit = {"training_root": str(training_root), "label_file": str(label_path),
             "split_seed": seed, "fold_count": nfold, "stratified": bool(stratified),
             "matched_n": len(frame), "folds": audit_folds,
             "mapping": "Reproduced sorted UID intersection and recorded outer CV; every exported target checked",
             "error_reference": "Canonical PAG high-precision labels after float32-consistency checks"}
    return frame.loc[canonical.index], audit


def make_figure(data, args, edges):
    plt.rcParams.update({
        "font.family": "Times New Roman", "mathtext.fontset": "stix",
        "font.size": args.font_size, "axes.labelsize": args.font_size,
        "axes.titlesize": args.font_size, "xtick.labelsize": args.font_size-1,
        "ytick.labelsize": args.font_size-1, "legend.fontsize": args.font_size-1,
        "axes.linewidth": .7, "axes.grid": False, "axes.facecolor": "white",
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False,
    })
    vertical = args.layout == "vertical"
    grid = ({"left": .205, "right": .975, "bottom": .20,
             "top": .95, "hspace": .58} if vertical else
            {"left": .095, "right": .978, "bottom": .32,
             "top": .88, "wspace": .32})
    fig, (left, right) = plt.subplots(
        2 if vertical else 1, 1 if vertical else 2,
        figsize=(args.width_mm/25.4, args.height_mm/25.4), gridspec_kw=grid)
    model_keys = ["baseline"] + (["eatgnn"] if "error_eatgnn" in data else []) + ["pag"]
    comparators = [key for key in model_keys if key != "pag"]
    errors_all = data[["error_"+key for key in model_keys]].to_numpy()
    if args.error_scale == "log" and (errors_all <= 0).any():
        raise ValueError("Log scale cannot display zero errors. Use --error-scale symlog or linear.")
    positive = errors_all[errors_all > 0]
    scale_options = {}
    if args.error_scale == "symlog":
        scale_options = {"linthresh": max(float(positive.min()) if positive.size else 1e-4, 1e-6)}
    right.set_yscale(args.error_scale, **scale_options)
    # Each line is the quantile profile of paired differences against one
    # comparator. Lines are sorted independently: a given percentile does not
    # identify the same material across curves.
    left.set_yscale("symlog", linthresh=args.delta_linthresh)
    left.text(.035, .95, "Lower-error samples", color="#333333",
              transform=left.transAxes, va="top", fontsize=args.font_size-1)
    largest = 0.
    for index, key in enumerate(comparators):
        column = "paired_delta_error" if key == "baseline" else "paired_delta_eatgnn"
        delta = np.sort(data[column].to_numpy())
        percentile = np.linspace(0., 100., len(delta))
        largest = max(largest, float(np.max(np.abs(delta))))
        left.fill_between(percentile, delta, 0., color=COLORS[key], alpha=.10, linewidth=0)
        left.plot(percentile, delta, color=COLORS[key], lw=1.5,
                  ls="-")
        short_name = NAMES[key]
        left.text(.035, .84-.11*index,
                  f"vs {short_name}: {(delta<0).mean():.1%} ({(delta<0).sum():,})",
                  color=COLORS[key], transform=left.transAxes, va="top",
                  fontsize=args.font_size-1)
    left.axhline(0., color="#555555", ls="-", lw=.75)
    limit = max(1., largest) * 1.12
    left.set(xlabel="Ranked percentile (%)",
              ylabel=r"$\Delta e=e_{\mathrm{PAG}}-e_{\mathrm{other}}$ (Å)",
              xlim=(0., 100.), ylim=(-limit, limit))
    extreme_tick = 10. if limit >= 10. else 5.
    ticks = np.array([-extreme_tick, -1., -.25, 0., .25, 1., extreme_tick])
    ticks = ticks[np.abs(ticks) <= limit]
    left.set_yticks(ticks)
    left.set_yticklabels([f"{tick:g}" for tick in ticks])
    left.yaxis.set_minor_locator(NullLocator())
    left.set_xticks([0., 25., 50., 75., 100.])
    if vertical:
        pass  # Panel identifiers are placed after the final layout.
    else:
        left.set_title("(a)  Paired error change", loc="left", pad=7)

    groups = np.searchsorted(edges, data.label_alpha_parallel.to_numpy(), side="left")
    labels = bin_labels(edges)
    centers = np.arange(len(labels))
    counts = [int(np.sum(groups == i)) for i in centers]
    offsets = [-.18, .18] if len(model_keys) == 2 else [-.26, 0., .26]
    box_width = .28 if len(model_keys) == 2 else .20
    for model_index, key in enumerate(model_keys):
        for i, center in enumerate(centers):
            values = data.loc[groups == i, "error_"+key].to_numpy()
            if not len(values):
                continue
            position = float(center + offsets[model_index])
            if len(values) == 1:
                right.plot(position, values[0], "o", color=COLORS[key], ms=3.)
                continue
            right.boxplot(
                [values], positions=[position], widths=box_width, patch_artist=True,
                manage_ticks=False, showfliers=True, whis=1.5,
                boxprops={"facecolor": to_rgba(COLORS[key], .20), "edgecolor": COLORS[key], "linewidth": .9},
                medianprops={"color": COLORS[key], "linewidth": 1.5},
                whiskerprops={"color": COLORS[key], "linewidth": .8},
                capprops={"color": COLORS[key], "linewidth": .8},
                flierprops={"marker": "o", "markersize": 1.7, "markerfacecolor": COLORS[key],
                            "markeredgewidth": 0, "alpha": .45})
    right.xaxis.set_major_locator(FixedLocator(centers))
    right.set_xticklabels([f"{label}\nn = {n:,}" for label, n in zip(labels, counts)])
    right.set(xlim=(-.6, len(labels)-.4), xlabel=r"Label $\alpha_{\parallel}$ (Å)",
              ylabel="Frobenius error (Å)")
    if vertical:
        pass  # Panel identifiers are placed after the final layout.
    else:
        right.set_title("(b)  Response-stratified error", loc="left", pad=7)
    right.xaxis.set_minor_locator(NullLocator())
    for ax in (left, right):
        ax.grid(False)
        ax.tick_params(direction="out", length=3, width=.7, pad=3)
    if vertical:
        # Work in physical millimetres. Keep fonts and both plotting areas
        # unchanged while removing only the excess space between their labels.
        width, height = args.width_mm, args.height_mm
        plot_height = (height - 42.) / 2.
        top_bottom, lower_bottom = height - 3. - plot_height, 20.
        def place_axes():
            left.set_position([17.5/width, top_bottom/height,
                               (width-19.5)/width, plot_height/height])
            right.set_position([17.5/width, lower_bottom/height,
                                (width-19.5)/width, plot_height/height])
        place_axes()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        measured_gap = (left.get_tightbbox(renderer).y0 -
                        right.get_tightbbox(renderer).y1) * 25.4 / fig.dpi
        removed_space = measured_gap - 3.
        top_bottom -= removed_space
        height -= removed_space
        height += 3.0  # Headroom for labels aligned above the y-axis captions.
        fig.set_size_inches(width/25.4, height/25.4, forward=True)
        place_axes()
        args.height_mm = height
        args.panel_content_gap_mm = 3.
    handles = [Line2D([], [], color=COLORS[k], lw=1.7,
                      ls="-", label=NAMES[k])
               for k in model_keys]
    fig.legend(handles=handles, ncol=len(model_keys), loc="lower center",
               bbox_to_anchor=(.52 if vertical else .54, 1./args.height_mm if vertical else .01),
               frameon=False, handlelength=1.5 if vertical else 2.,
               columnspacing=.9 if vertical else 1.6)
    if vertical:
        add_panel_labels(fig, [left, right], ["(a)", "(b)"])
    return fig, groups, counts


def main():
    args = parse_args()
    if args.width_mm is None:
        args.width_mm = 85. if args.layout == "vertical" else 177.
    if args.height_mm is None:
        args.height_mm = 114. if args.layout == "vertical" else 82.
    if args.font_size is None:
        args.font_size = 9. if args.layout == "vertical" else 12.
    if min(args.width_mm, args.height_mm, args.font_size, args.dpi, args.delta_linthresh) <= 0:
        raise ValueError("Figure dimensions, font size and DPI must be positive.")
    edges = np.array(args.bin_edges, dtype=float)
    if not np.isfinite(edges).all() or np.any(np.diff(edges) <= 0):
        raise ValueError("--bin-edges must be finite and strictly increasing.")
    root = args.project_root.resolve()
    package = root / "submission/PAG_ALIGNN_Supplementary"
    pag = (args.pag_predictions or package / "supplementary_data/D07_tensor_predictions.csv").resolve()
    baseline = (args.baseline_predictions or package / "results/baselines/six_component/predictions.csv").resolve()
    output = (args.output or root / "outputs/model_comparison/F07_tensor_error_comparison.png").resolve()
    if output.suffix.lower() not in (".png", ".pdf"):
        raise ValueError("--output must end in .png or .pdf.")
    data = align_and_measure(pag, baseline)
    eatgnn_root = (args.eatgnn_root or root.parent / "EATGNN").resolve()
    eat_frame, eat_audit = None, None
    if not args.no_eatgnn and (args.eatgnn_root is not None or eatgnn_root.exists()):
        canonical = load_predictions(pag)
        eat_frame, eat_audit = load_eatgnn_oof(eatgnn_root, canonical)
        error_components = np.column_stack([
            eat_frame["prediction_"+c]-canonical["target_"+c] for c in COMPONENTS])
        data["error_eatgnn"] = np.sqrt(np.sum(WEIGHTS*error_components**2, axis=1))
        data["paired_delta_eatgnn"] = data.error_pag-data.error_eatgnn
    model_keys = ["baseline"] + (["eatgnn"] if eat_frame is not None else []) + ["pag"]
    fig, groups, counts = make_figure(data, args, edges)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Fixed physical canvas. Tight cropping would change the requested size.
    for extension in (".png", ".pdf"):
        fig.savefig(output.with_suffix(extension), dpi=args.dpi, facecolor="white")
    plt.close(fig)
    data["response_bin"] = groups
    data.to_csv(output.with_name(output.stem+"_paired_errors.csv"), index_label="id")
    if eat_frame is not None:
        eat_frame.to_csv(output.with_name(output.stem+"_eatgnn_aligned.csv"), index_label="id")
    summaries = []
    for group in [-1, *range(len(counts))]:
        selection = data if group == -1 else data.loc[groups == group]
        for key in model_keys:
            values = selection["error_"+key]
            summaries.append({"bin": "all" if group == -1 else group,
                              "model": NAMES[key], "n": len(values),
                              "mean_error_A": float(values.mean()), "median_error_A": float(values.median()),
                              "q1_error_A": float(values.quantile(.25)), "q3_error_A": float(values.quantile(.75)),
                              "fraction_error_le_1A": float((values <= 1).mean()) if len(values) else None})
    pd.DataFrame(summaries).to_csv(output.with_name(output.stem+"_summary.csv"), index=False)
    metadata = {"pag_source": str(pag), "baseline_source": str(baseline), "paired_n": len(data),
                "target_consistency": "PAG/ALIGNN labels agree within rtol=1e-7, atol=1e-8; EATGNN exported float32 labels checked in eatgnn_oof_audit",
                "frobenius_weights": dict(zip(COMPONENTS, WEIGHTS.tolist())),
                "response_bin_edges_A": edges.tolist(), "response_bin_counts": counts,
                "bin_rule": "Upper edges inclusive; final bin above last edge; label alpha_parallel",
                "pag_lower_error_fraction": float((data.paired_delta_error < 0).mean()),
                "layout": args.layout, "width_mm": args.width_mm, "height_mm": args.height_mm,
                "panel_content_gap_mm": getattr(args, "panel_content_gap_mm", None),
                "font_size_pt": args.font_size, "dpi": args.dpi, "error_scale": args.error_scale,
                "boxplot": "Median, Q1/Q3, 1.5-IQR whiskers and all outliers; same material cohorts",
                "panel_a": "Each comparator has an independently sorted profile of error_PAG - error_comparator; negative favors PAG",
                "panel_a_xaxis": "Within-comparison percentile, not common sample identity",
                "models": [NAMES[k] for k in model_keys], "eatgnn_oof_audit": eat_audit,
                "panel_a_yscale": "symlog", "panel_a_linthresh_A": args.delta_linthresh,
                "paired_delta_mean_A": float(data.paired_delta_error.mean()),
                "paired_delta_median_A": float(data.paired_delta_error.median())}
    metadata["paired_comparisons"] = {}
    for comparator in [k for k in model_keys if k != "pag"]:
        delta = data.error_pag-data["error_"+comparator]
        metadata["paired_comparisons"][comparator] = {
            "mean_delta_A": float(delta.mean()), "median_delta_A": float(delta.median()),
            "pag_lower_fraction": float((delta<0).mean()), "pag_higher_fraction": float((delta>0).mean())}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Matched materials: {len(data)}; response-bin counts: {counts}")
    for key in model_keys:
        print(f"{NAMES[key]}: mean Frobenius error = {data['error_'+key].mean():.8f} A")
    print(f"PAG lower paired error: {(data.paired_delta_error < 0).mean():.2%}")
    if eat_frame is not None:
        print(f"PAG lower paired error vs EATGNN: {(data.paired_delta_eatgnn < 0).mean():.2%}")
    print(f"Saved PNG/PDF: {output.with_suffix('')}; canvas {args.width_mm} x {args.height_mm} mm")


if __name__ == "__main__":
    main()
