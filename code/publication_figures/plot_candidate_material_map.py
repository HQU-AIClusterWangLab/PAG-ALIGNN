"""Double-column candidate map, taking candidate values and membership from main.tex.

python visualization_tools/plot_candidate_material_map.py

Context points: tensor-routed model04 lattice predictions plus C2DB electronic
polarizability and HSE06/stability metadata. Eta-only predictions are excluded.
Highlighted points: exactly the candidates and rounded numbers in the manuscript
table labelled tab:tensor-route-ranking. No Pareto/rank is inferred for this map.
An audit records missing data and differences from numerical screening of raw data.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = Path(r"C:\Users\admin\Desktop\JCTC_PAG-ALIGNN_latex_project\main.tex")
BLUE = "#0066FF"


def manuscript_candidates(path: Path) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8-sig")
    start = text.index(r"\label{tab:tensor-route-ranking}")
    block = text[start:text.index(r"\end{longtable}", start)]
    rows = []
    for line in block.splitlines():
        clean = line.replace(r"\allowbreak", "")
        clean = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", clean)
        fields = [part.strip() for part in clean.split("&")]
        if len(fields) != 8 or not fields[0].isdigit():
            continue
        rows.append({
            "manuscript_rank": int(fields[0]), "sample_id": fields[1].replace(" ", ""),
            **dict(zip(["alpha_parallel_lat", "alpha_parallel_el", "alpha_parallel_stat",
                        "alpha_zz_stat", "gap_hse_eV", "ehull_eV_per_atom"],
                       [float(re.match(r"[-+0-9.eE]+", value)[0]) for value in fields[2:]])),
        })
    frame = pd.DataFrame(rows)
    if frame.empty or frame.sample_id.duplicated().any():
        raise ValueError("No valid unique candidates found in the manuscript table")
    if not np.allclose(frame.alpha_parallel_stat,
                       frame.alpha_parallel_lat + frame.alpha_parallel_el, atol=.0011, rtol=0):
        raise ValueError("Manuscript electronic + lattice sum exceeds rounding tolerance")
    return frame


def load_metadata(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    return {"sample_id": d["uid"], "gap_hse_eV": d.get("gap_hse"),
            "ehull_eV_per_atom": d.get("ehull"), "dyn_stab": d.get("dyn_stab"),
            "alphax_el": d.get("alphax_el"), "alphay_el": d.get("alphay_el"),
            "metadata_source": str(path)}


def prepare(args, candidates):
    prediction_path = args.predictions or args.project_root / "outputs/routed_polarizability_predictions/prediction_model_04/interband_routed_predictions_model04.csv"
    pred = pd.read_csv(prediction_path, dtype={"sample_id": str})
    pred = pred[pred.prediction_branch.eq("tensor_model")].copy()
    if pred.sample_id.duplicated().any():
        raise ValueError("Duplicate tensor prediction IDs")
    if args.metadata_table:
        meta = pd.read_csv(args.metadata_table, dtype={"sample_id": str})
    else:
        paths = sorted((args.project_root / "interband_only").glob("*/*/*/data.json"))
        with ThreadPoolExecutor(max_workers=16) as pool:
            meta = pd.DataFrame(pool.map(load_metadata, paths))
    if meta.sample_id.duplicated().any():
        raise ValueError("Duplicate metadata IDs")
    joined = pred.merge(meta, on="sample_id", how="left", validate="one_to_one")
    joined["alpha_parallel_lat"] = (joined.final_xx + joined.final_yy) / 2
    joined["alpha_parallel_el"] = (joined.alphax_el + joined.alphay_el) / 2
    joined["alpha_parallel_stat"] = joined.alpha_parallel_lat + joined.alpha_parallel_el
    required = ["gap_hse_eV", "ehull_eV_per_atom", "alpha_parallel_stat"]
    valid = np.isfinite(joined[required]).all(axis=1) & (joined.alpha_parallel_stat > 0)
    points = joined[valid].copy()
    points["passes_raw_screen"] = (points.gap_hse_eV.ge(5) & points.ehull_eV_per_atom.le(.2)
                                    & points.dyn_stab.eq("Yes"))
    points["in_manuscript_table"] = points.sample_id.isin(candidates.sample_id)
    matched = candidates.merge(points, on="sample_id", how="left", suffixes=("_paper", "_raw"),
                               validate="one_to_one", indicator=True)
    if not matched._merge.eq("both").all():
        raise ValueError("Some manuscript candidates lack complete tensor-route data")
    if not matched.passes_raw_screen.all():
        raise ValueError("A manuscript candidate fails the raw-data physical screen")
    rounding_audit = {}
    for column in required + ["alpha_parallel_lat", "alpha_parallel_el"]:
        delta = abs(matched[column + "_paper"] - matched[column + "_raw"])
        rounding_audit[column] = float(delta.max())
        # The manuscript prints three decimals; retain its values, not replacements.
        if (delta > .0011).any():
            raise ValueError(f"Substantial manuscript/raw discrepancy in {column}")
    excluded = joined.loc[~valid, ["sample_id"] + required].copy()
    audit = {"manuscript": str(args.manuscript.resolve()),
             "manuscript_sha256": hashlib.sha256(args.manuscript.read_bytes()).hexdigest(),
             "prediction_source": str(prediction_path.resolve()),
             "tensor_route_count": len(pred), "plotted_context_count": len(points),
             "missing_or_invalid_count": len(excluded), "table_candidate_count": len(candidates),
             "thresholds": {"gap_hse_min_eV": 5, "ehull_max_eV_per_atom": .2, "dyn_stab": "Yes"},
             "raw_screen_count": int(points.passes_raw_screen.sum()),
             "raw_screen_ids_absent_from_table": points.loc[points.passes_raw_screen & ~points.in_manuscript_table, "sample_id"].tolist(),
             "max_manuscript_raw_deviation": rounding_audit,
             "candidate_coordinates": "Printed manuscript table values; context uses unrounded local data",
             "y_definition": "(predicted alpha_xx_lat + predicted alpha_yy_lat + alpha_x_el + alpha_y_el) / 2",
             "ranking": "Manuscript membership only; in-plane response is not a replacement for its trace ranking"}
    return points, meta, excluded, audit


def draw(points, candidates, args):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix", "font.size": 11, "axes.labelsize": 12,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "axes.linewidth": .8,
        "axes.grid": False, "axes.facecolor": "white", "figure.facecolor": "white",
        "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(args.width_mm / 25.4, args.height_mm / 25.4))
    ax = fig.add_axes([.09, .21, .73, .66])
    # Share the main axes bottom/top and reduce the horizontal separation.
    cax = fig.add_axes([.84, .21, .021, .66])
    norm = colors.PowerNorm(gamma=.5, vmin=0, vmax=max(.21, np.ceil(points.ehull_eV_per_atom.max() * 10) / 10))
    cmap = "viridis_r"
    xmax = max(9., np.ceil(points.gap_hse_eV.max()))
    ymax = np.ceil(points.alpha_parallel_stat.max() / 5) * 5 + 1
    ax.set(xlim=(-.1, xmax), ylim=(0, ymax), xlabel="Band gap (eV)",
           ylabel=r"Total static in-plane polarizability ($\mathrm{\AA}$)")
    ax.axvspan(5, xmax, color="#F0F6FC", zorder=0)
    ax.axvline(5, color="#87A6C2", lw=.8, zorder=1)
    inset = ax.inset_axes([.53, .47, .44, .44])
    inset.set(xlim=(5, 8.6), ylim=(1.15, 4.65))
    inset.set_facecolor("#F0F6FC")
    inset.set_xticks([5, 6, 7, 8])
    inset.set_yticks([2, 3, 4])
    inset.tick_params(labelsize=8, length=2)
    inset.set_title("Wide-gap candidates", loc="right", fontsize=9, pad=7)
    mappable = None
    for axis in (ax, inset):
        for stable, marker in [(False, "x"), (True, "o")]:
            subset = points[points.dyn_stab.eq("Yes") == stable]
            mappable = axis.scatter(subset.gap_hse_eV, subset.alpha_parallel_stat,
                c=subset.ehull_eV_per_atom, cmap=cmap, norm=norm, s=15 if stable else 14,
                marker=marker, linewidths=.5, alpha=.80, zorder=2)
        # Do not draw context duplicates of the rounded highlighted table positions.
        axis.scatter(candidates.gap_hse_eV, candidates.alpha_parallel_stat,
                     s=46 if axis is inset else 28, facecolors="none", edgecolors=BLUE,
                     linewidths=1.0, zorder=4)
        key = candidates.set_index("sample_id").loc["2LiMnCl3-1"]
        axis.scatter([key.gap_hse_eV], [key.alpha_parallel_stat], s=90 if axis is inset else 65,
                     marker="*", color="#E64B35", edgecolor="white", linewidth=.45, zorder=6)
        axis.grid(False)
    lookup = candidates.set_index("sample_id")
    labels = {"2LiMnCl3-1": (6.0, 4.34)}
    for uid, location in labels.items():
        p = lookup.loc[uid]
        inset.annotate(uid, xy=(p.gap_hse_eV, p.alpha_parallel_stat), xytext=location,
            fontsize=8, ha="left" if uid != "2CaF2O2H4-1" else "center",
            arrowprops={"arrowstyle": "-", "color": ".35", "lw": .55}, zorder=7)
    ax.text(.975, .15, r"$E_g\geq 5$ eV", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color="#526E86")
    cb = fig.colorbar(mappable, cax=cax)
    cb.set_ticks([0, .05, .2, .5, 1.0])
    cb.set_ticklabels(["0", "0.05", "0.2", "0.5", "1.0"])
    cb.ax.tick_params(labelsize=10, length=2)
    cb.set_label(r"Energy above hull (eV/atom)", fontsize=12, labelpad=6)
    cb.ax.axhline(.2, color="black", lw=1.2)
    handles = [Line2D([], [], ls="", marker="o", color=".5", markersize=4, label="Dynamically stable"),
               Line2D([], [], ls="", marker="x", color=".5", markersize=4, label="Unstable / unconfirmed"),
               Line2D([], [], ls="", marker="o", markerfacecolor="none", markeredgecolor=BLUE,
                      markersize=6, label="Screened candidates"),
               Line2D([], [], ls="", marker="*", color="#E64B35", markersize=8, label="DFT validation candidate")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.50, .01), ncol=4,
               fontsize=10, frameon=False, columnspacing=1.4, handletextpad=.4)
    for suffix in (".png", ".pdf"):
        fig.savefig(args.output.with_suffix(suffix), dpi=args.dpi)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=ROOT)
    p.add_argument("--manuscript", type=Path, default=MANUSCRIPT)
    p.add_argument("--predictions", type=Path)
    p.add_argument("--candidates-table", type=Path, help="Previously audited candidate table CSV")
    p.add_argument("--metadata-table", type=Path, help="Optional previously exported metadata CSV")
    p.add_argument("--output", type=Path)
    p.add_argument("--width-mm", type=float, default=177.)
    p.add_argument("--height-mm", type=float, default=100.)
    p.add_argument("--dpi", type=int, default=600)
    args = p.parse_args()
    args.output = args.output or args.project_root / "outputs/candidate_map/F09_candidate_map.png"
    candidates = (pd.read_csv(args.candidates_table) if args.candidates_table
                  else manuscript_candidates(args.manuscript))
    points, meta, excluded, audit = prepare(args, candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    draw(points, candidates, args)
    for suffix, frame in [("_context.csv", points), ("_metadata.csv", meta),
                          ("_manuscript_candidates.csv", candidates), ("_excluded.csv", excluded)]:
        frame.to_csv(args.output.with_name(args.output.stem + suffix), index=False)
    audit.update(width_mm=args.width_mm, height_mm=args.height_mm)
    args.output.with_suffix(".json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
