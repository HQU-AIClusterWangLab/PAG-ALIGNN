"""Preview actual C2DB HSE06+SOC bands from sampled mesh points on Gamma-M-K-Gamma.

Only eigenvalues already present in hse.json are used. Straight lines join
sampled points; no synthetic or spline-interpolated band values are generated.
The path is defined for the verified hexagonal cells of these two materials.
Vacuum reference: raw eig_soc_mk minus data.json evac, checked against stored
vacuum-referenced band edges. --reference vbm is also supported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MATERIALS = ROOT / "data/publication_figures/materials"
PATH_VERTICES = np.array([[0.,0.,0.],[.5,0.,0.],[1/3,1/3,0.],[0.,0.,0.]])
VALENCE = "#0066FF"
CONDUCTION = "#E87524"


def read_material(folder):
    hse = json.loads((folder/"hse.json").read_text(encoding="utf-8"))
    meta = json.loads((folder/"data.json").read_text(encoding="utf-8"))
    raw_structure = json.loads((folder/"structure.json").read_text(encoding="utf-8"))
    structure = next(v for v in raw_structure.values() if isinstance(v,dict) and "cell" in v)
    cell = np.asarray(structure["cell"],dtype=float)
    a,b = cell[0],cell[1]
    angle=np.degrees(np.arccos(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))))
    if abs(angle-120)>1 or abs(np.linalg.norm(a)/np.linalg.norm(b)-1)>.02:
        raise ValueError("This Gamma-M-K path requires an approximately hexagonal 120-degree cell")
    k=np.asarray(hse["bzkpts"],dtype=float)
    energies=np.asarray(hse["eig_soc_mk"],dtype=float)
    if energies.ndim!=2 or energies.shape[1]!=len(k) or not np.isfinite(energies).all():
        raise ValueError("Unexpected eigenvalue shape or missing eigenvalues")
    ef=float(hse["efermi"])
    vbm=float(energies[energies<ef].max())
    cbm=float(energies[energies>ef].min())
    vacuum=float(meta["evac"])
    checks={"vbm_hse":vbm-vacuum,"cbm_hse":cbm-vacuum,"gap_hse":cbm-vbm}
    for key,value in checks.items():
        if not np.isclose(value,float(meta[key]),atol=2e-5,rtol=0):
            raise ValueError(f"Eigenvalues disagree with data.json: {key}")
    reduced=k%1
    reduced[np.isclose(reduced,1,atol=1e-8)]=0
    indices=[]; segment_lengths=[]; segment_counts=[]
    reciprocal=2*np.pi*np.linalg.inv(cell).T
    for start,end in zip(PATH_VERTICES[:-1],PATH_VERTICES[1:]):
        delta=end-start
        t=(reduced-start)@delta/np.dot(delta,delta)
        residual=np.linalg.norm(reduced-(start+t[:,None]*delta),axis=1)
        selected=np.where((residual<1e-7)&(t>=-1e-7)&(t<=1+1e-7))[0]
        selected=selected[np.argsort(t[selected])]
        if len(selected)<2 or not np.isclose(t[selected[0]],0) or not np.isclose(t[selected[-1]],1):
            raise ValueError("Mesh lacks a required high-symmetry path endpoint")
        segment_counts.append(len(selected))
        indices.extend(selected.tolist() if not indices else selected[1:].tolist())
        segment_lengths.append(float(np.linalg.norm(delta@reciprocal)))
    indices=np.asarray(indices)
    path_k=reduced[indices]
    distance=np.r_[0,np.linalg.norm(np.diff(path_k,axis=0)@reciprocal,axis=1).cumsum()]
    ticks=np.r_[0,np.cumsum(segment_lengths)]
    assert np.isclose(distance[-1],ticks[-1])
    source_hashes={name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
                   for name in ["data.json","hse.json","structure.json"]}
    return {"folder":folder,"meta":meta,"energies":energies,"ef":ef,"vbm":vbm,"cbm":cbm,
            "vacuum":vacuum,"k":k,"indices":indices,"distance":distance,"ticks":ticks,
            "path_k":path_k,"segment_counts":segment_counts,"hashes":source_hashes}


def draw(materials,args):
    plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman"],
        "mathtext.fontset":"stix","font.size":11,"axes.labelsize":12,
        "xtick.labelsize":11,"ytick.labelsize":11,"axes.grid":False,
        "axes.facecolor":"white","figure.facecolor":"white","savefig.facecolor":"white",
        "axes.linewidth":.8,"pdf.fonttype":42})
    fig,axes=plt.subplots(1,2,figsize=(177/25.4,100/25.4),sharey=True)
    fig.subplots_adjust(left=.11,right=.98,bottom=.16,top=.82,wspace=.16)
    reference=lambda m:m["vacuum"] if args.reference=="vacuum" else m["vbm"]
    lower=min(m["vbm"]-reference(m) for m in materials)-2.0
    upper=max(m["cbm"]-reference(m) for m in materials)+1.8
    if args.reference=="vacuum":upper=max(.6,upper)
    rows=[]; audit=[]
    for panel,(ax,m) in enumerate(zip(axes,materials)):
        offset=reference(m)
        path_energy=m["energies"][:,m["indices"]]-offset
        vbm,cbm=m["vbm"]-offset,m["cbm"]-offset
        shown=[]
        for band,values in enumerate(path_energy):
            if values.max()<lower or values.min()>upper:continue
            occupied=bool(np.max(m["energies"][band])<m["ef"])
            # Neither material has a partially occupied band; refuse a misleading classification.
            if np.min(m["energies"][band])<m["ef"]<np.max(m["energies"][band]):
                raise ValueError("A band crosses E_F; occupied/unoccupied plotting needs review")
            col=VALENCE if occupied else CONDUCTION
            ax.plot(m["distance"],values,color=col,lw=.8,marker="o",ms=1.6,
                    markeredgewidth=0,alpha=.85,zorder=2)
            shown.append(band)
            for rank,(x,y,kk) in enumerate(zip(m["distance"],values,m["path_k"])):
                rows.append({"sample_id":m["meta"]["uid"],"band_index":band,
                             "path_index":rank,"distance_inverse_angstrom":x,
                             "kx":kk[0],"ky":kk[1],"kz":kk[2],
                             "energy_eV":y,"occupied":occupied,"reference":args.reference})
        for tick in m["ticks"][1:-1]:ax.axvline(tick,color=".85",lw=.7,zorder=1)
        if args.reference=="vacuum":ax.axhline(0,color=".45",lw=.8,zorder=1)
        ax.text(.5,(vbm+cbm)/2,rf"$E_g={m['cbm']-m['vbm']:.3f}$ eV",
                transform=ax.get_yaxis_transform(),ha="center",va="center",fontsize=11)
        ax.set(xlim=(0,m["distance"][-1]),ylim=(lower,upper),
               xticks=m["ticks"],xticklabels=[r"$\Gamma$","M","K",r"$\Gamma$"],
               xlabel="Wave vector")
        label="2LiMnCl3-1" if panel==0 else r"MoS$_2$ (1MoS2-1)"
        ax.set_title(f"({'ab'[panel]}) {label}",loc="left",fontsize=12,pad=10)
        ax.grid(False);ax.tick_params(direction="out",length=3)
        path_vbm=path_energy[path_energy<m["ef"]-offset].max()
        path_cbm=path_energy[path_energy>m["ef"]-offset].min()
        audit.append({"sample_id":m["meta"]["uid"],"source_folder":str(m["folder"]),
            "file_sha256":m["hashes"],"mesh_kpoints":len(m["k"]),
            "mesh_shape_in_plane":[len(np.unique(m["k"][:,i].round(8))) for i in [0,1]],
            "path_sample_count":len(m["indices"]),"segment_point_counts":m["segment_counts"],
            "shown_band_indices":shown,"full_mesh_gap_eV":m["cbm"]-m["vbm"],
            "sampled_path_gap_eV":float(path_cbm-path_vbm),
            "vbm_vacuum_eV":m["vbm"]-m["vacuum"],"cbm_vacuum_eV":m["cbm"]-m["vacuum"]})
    axes[0].set_ylabel("Energy relative to vacuum (eV)" if args.reference=="vacuum" else "Energy relative to VBM (eV)")
    handles=[Line2D([],[],color=VALENCE,lw=1.2,marker="o",ms=2,label="Occupied bands"),
             Line2D([],[],color=CONDUCTION,lw=1.2,marker="o",ms=2,label="Unoccupied bands")]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.54,1.),
               ncol=2,frameon=False,fontsize=10)
    fig.text(.54,.905,"HSE06 + SOC | Calculated mesh points on the symmetry path",
             ha="center",fontsize=9,color=".35")
    for suffix in [".png",".pdf"]:fig.savefig(args.output.with_suffix(suffix),dpi=600)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(args.output.with_name(args.output.stem+"_path_bands.csv"),index=False)
    result={"reference":args.reference,"path":"Gamma-M-K-Gamma",
            "fractional_vertices":PATH_VERTICES.tolist(),"interpolation":"None: straight joins of actual sampled points",
            "gap_labels":"Full original mesh gaps, verified against data.json", "materials":audit}
    args.output.with_suffix(".json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--materials-root",type=Path,default=MATERIALS)
    p.add_argument("--reference",choices=["vacuum","vbm"],default="vacuum")
    p.add_argument("--output",type=Path,default=ROOT/"outputs/real_bands/C2DB_HSE_SOC_bands_preview.png")
    args=p.parse_args()
    materials=[read_material(args.materials_root/relative)
               for relative in ["ABC3/2LiMnCl3/1","AB2/1MoS2/1"]]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    draw(materials,args)


if __name__=="__main__":main()
