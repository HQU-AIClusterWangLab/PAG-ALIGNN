"""Real HSE06+SOC bands flanking a common-vacuum band-offset diagram.

Uses plot_c2db_real_bands.read_material for checked original mesh eigenvalues.
Lines connect actual sampled points; no interpolation or extra DFT is performed.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
from publication_panel_labels import add_panel_labels

from plot_c2db_real_bands import read_material, MATERIALS

ROOT=Path(__file__).resolve().parents[1]
BLUE="#0066FF"
PURPLE="#A64DFF"


def draw(args):
    li=read_material(args.materials_root/"ABC3/2LiMnCl3/1")
    mo=read_material(args.materials_root/"AB2/1MoS2/1")
    plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman"],
        "mathtext.fontset":"stix","font.size":10,"axes.labelsize":11,
        "xtick.labelsize":10,"ytick.labelsize":10,"axes.grid":False,
        "axes.facecolor":"white","figure.facecolor":"white","savefig.facecolor":"white",
        "axes.linewidth":.8,"pdf.fonttype":42})
    fig=plt.figure(figsize=(177/25.4,100/25.4))
    grid=fig.add_gridspec(1,3,left=.095,right=.98,bottom=.17,top=.83,
                         width_ratios=[1.,1.05,1.],wspace=.12)
    left=fig.add_subplot(grid[0,0])
    mid=fig.add_subplot(grid[0,1],sharey=left)
    right=fig.add_subplot(grid[0,2],sharey=left)
    lower=-9.;upper=.5
    edges=[]
    for axis,m,color,panel in [(left,li,BLUE,"a"),
                                (right,mo,PURPLE,"c")]:
        e=m["energies"][:,m["indices"]]-m["vacuum"]
        for values in e:
            if values.max()<lower or values.min()>upper:continue
            axis.plot(m["distance"],values,color=color,lw=.7,marker="o",ms=1.4,
                      markeredgewidth=0,alpha=.85,zorder=2)
        for tick in m["ticks"][1:-1]:axis.axvline(tick,color=".88",lw=.6,zorder=0)
        axis.set(xlim=(0,m["distance"][-1]),ylim=(lower,upper),
                 xticks=m["ticks"],xticklabels=[r"$\Gamma$","M","K",r"$\Gamma$"],
                 xlabel="Wave vector")

        axis.tick_params(direction="out",length=3)
        axis.grid(False)
        vbm=m["vbm"]-m["vacuum"];cbm=m["cbm"]-m["vacuum"]
        edges.append({"sample_id":m["meta"]["uid"],"vbm_eV":vbm,"cbm_eV":cbm,
                      "gap_eV":cbm-vbm,"path_points":len(m["indices"]),
                      "source":str(m["folder"]),"hashes":m["hashes"]})
        axis.text(.5,(vbm+cbm)/2,rf"$E_g={cbm-vbm:.3f}$ eV",
                  transform=axis.get_yaxis_transform(),ha="center",va="center",fontsize=9)
    right.tick_params(labelleft=False)
    left.set_ylabel("Energy relative to vacuum (eV)")
    left.set_yticks([-8,-6,-4,-2,0])
    mid.set(xlim=(0,1),ylim=(lower,upper),xticks=[])
    mid.tick_params(left=False,labelleft=False,bottom=False)
    for spine in mid.spines.values():spine.set_visible(False)
    # A common zero across all three panels, including the narrow gaps.
    vacuum=ConnectionPatch(xyA=(0,0),coordsA=left.transData,
        xyB=(right.get_xlim()[1],0),coordsB=right.transData,
        color=".6",lw=.8,clip_on=False,zorder=1)
    fig.add_artist(vacuum)
    mid.text(.5,.16,"Vacuum = 0",ha="center",va="bottom",fontsize=8.5)
    xli=(.04,.29);xmo=(.72,.97)
    for edge,span,col in [(edges[0],xli,BLUE),(edges[1],xmo,PURPLE)]:
        for name in ["vbm_eV","cbm_eV"]:
            y=edge[name]
            mid.plot(span,[y,y],color=col,lw=2,zorder=4)
        # Narrow pale gap bars identify each material without filling the full panel.
        mid.fill_between(span,edge["vbm_eV"],edge["cbm_eV"],color=col,alpha=.06,zorder=0)
    # Fine guides join each material's band edge to its band plot, at the same energy.
    for edge in ["cbm_eV","vbm_eV"]:
        yl=edges[0][edge];yr=edges[1][edge]
        fig.add_artist(ConnectionPatch(xyA=(left.get_xlim()[1],yl),coordsA=left.transData,
            xyB=(xli[0],yl),coordsB=mid.transData,color=BLUE,alpha=.45,lw=.7,
            clip_on=False,zorder=1))
        fig.add_artist(ConnectionPatch(xyA=(xmo[1],yr),coordsA=mid.transData,
            xyB=(0,yr),coordsB=right.transData,color=PURPLE,alpha=.45,lw=.7,
            clip_on=False,zorder=1))
        mid.plot([xli[1],.46],[yl,yl],color=".65",lw=.6)
        mid.plot([.46,xmo[0]],[yr,yr],color=".65",lw=.6)
        mid.annotate("",xy=(.46,max(yl,yr)),xytext=(.46,min(yl,yr)),
                     arrowprops={"arrowstyle":"<->","color":".2","lw":.9,
                                 "shrinkA":0,"shrinkB":0})
        symbol="C" if edge=="cbm_eV" else "V"
        mid.text(.53,(yl+yr)/2,rf"$\Delta E_{{{symbol}}}$"+f"\n{abs(yl-yr):.2f} eV",
                 ha="left",va="center",fontsize=8.5,linespacing=1.15,zorder=5)
    for edge,span,col in [(edges[0],xli,BLUE),(edges[1],xmo,PURPLE)]:
        x=np.mean(span)
        mid.text(x,edge["cbm_eV"]+.22,f"{edge['cbm_eV']:.2f}",color=col,ha="center",fontsize=8)
        mid.text(x,edge["vbm_eV"]-.48,f"{edge['vbm_eV']:.2f}",color=col,ha="center",fontsize=8)
    mid.text(np.mean(xli),-8.45,r"LiMnCl$_3$",color=BLUE,ha="center",fontsize=8)
    mid.text(np.mean(xmo),-8.45,r"MoS$_2$",color=PURPLE,ha="center",fontsize=8)
    panel_labels = add_panel_labels(fig, [left, mid, right], ["(a)", "(b)", "(c)"],
                                    insertion_scale=165.1/177.)
    # Keep (c) above the right panel and center (b) between the outer labels.
    panel_labels[2].set_x(right.get_position().x0 + .02*right.get_position().width)
    panel_labels[2].set_ha("left")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outer_boxes = [panel_labels[i].get_window_extent(renderer) for i in (0, 2)]
    center_px = sum((box.x0 + box.x1)/2 for box in outer_boxes)/2
    panel_labels[1].set_x(fig.transFigure.inverted().transform((center_px, 0))[0])
    panel_labels[1].set_ha("center")
    # Final optical adjustment: move only (c) left by 1 mm at manuscript size.
    panel_labels[2].set_x(panel_labels[2].get_position()[0] - 1.0/165.1)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for suffix in [".png",".pdf"]:fig.savefig(args.output.with_suffix(suffix),dpi=600)
    plt.close(fig)
    result={"materials":edges,"conduction_offset_eV":edges[0]["cbm_eV"]-edges[1]["cbm_eV"],
            "valence_offset_eV":edges[1]["vbm_eV"]-edges[0]["vbm_eV"],
            "reference":"Vacuum = 0 eV; raw eig_soc_mk minus evac",
            "scope":"Natural isolated-monolayer band offsets, not a calculated interface",
            "interpolation":"None; actual mesh-path points joined by straight segments",
            "width_mm":177,"height_mm":100}
    args.output.with_suffix(".json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--materials-root",type=Path,default=MATERIALS)
    p.add_argument("--output",type=Path,default=ROOT/"outputs/real_bands/C2DB_bands_with_offsets.png")
    draw(p.parse_args())


if __name__=="__main__":main()
