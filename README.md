# PAG-ALIGNN: code, models, and supporting data

Supporting materials for **PAG-ALIGNN: Full Lattice Polarizability Tensor Prediction for Screening Two-Dimensional Dielectric Candidates**.

PAG-ALIGNN is a physics-guided attention-gated framework based on ALIGNN. This package contains the modified model implementation, trained checkpoints, material structures, numerical descriptors, archived training inputs and split IDs, and prediction results. All paths and examples below refer to the files currently included in this directory.

## 1. Contents

```text
code/
  alignn/                    Modified ALIGNN implementation and training module
  scripts/                   Preparation, training, inference, and evaluation scripts
  publication_figures/       Plotting scripts
configs/                     Archived configurations and model selection information
data/
  structures/static_2001.json Labeled-set structures (2,001 materials)
  features/                  Numerical training descriptors
  training/                  Archived labels, split IDs, weights, and structure maps
  inference/                 Structures and descriptors for 3,351 inference materials
models/                      Regression ensembles and classification models
results/                     Archived training, evaluation, and comparison outputs
supplementary_data/          Machine-readable labels and prediction tables
environment/                 Recorded Python and package versions
licenses/                    Upstream ALIGNN terms of use
predict.py                   Routed ensemble inference entry point
train_fold.py                Archived training wrapper; see Section 6
```

The primary targets are intrinsic two-dimensional **lattice** polarizabilities. Tensor entries have units of angstrom (Å); in-plane anisotropy and classification probabilities are dimensionless. The mean in-plane lattice response is `(alpha_xx + alpha_yy) / 2`. Electronic and total static polarizabilities are identified separately in the candidate table.

## 2. Models and inputs

| Model directory | Checkpoints | Deployed outputs |
| --- | ---: | --- |
| `models/eta/folds/` | Five folds | In-plane anisotropy, eta_parallel |
| `models/inplane/folds/` | Five folds | xx, yy, xy |
| `models/outplane_zz/folds/` | Five folds | zz |
| `models/outplane_aux/folds/` | Five folds | xz, yz |
| `models/classification/alignn_gate_full/` | One | ALIGNN gate probability |
| `models/classification/catboost_gate_a/` | One CatBoost model | CatBoost gate probability |

Each regression fold contains `best_model.pt`, `config.json`, and `extra_feature_names.json`. All four regression branches use **101 global physical descriptors**. Local site-symmetry node features are used by the main eta model; they are distinct from the 101 global descriptors. Training outputs can include auxiliary targets beyond the deployed outputs listed above; use the archived configuration to determine their order and transformations.

The classification manifest is `models/classification/weighted_fusion/fusion_classifier_manifest.json`. It specifies:

```text
p_final = 0.96 * p_CatBoost + 0.04 * p_ALIGNN
high-response route: p_final >= 0.06761944
training positive class: mean in-plane lattice polarizability > 16.1 Å
```

Numerical descriptor inputs are provided in addition to feature-name lists:

| File | Materials | Descriptor columns, excluding ID |
| --- | ---: | ---: |
| `data/features/training_aux.csv` | 2,001 | 101 |
| `data/features/catboost_training.csv` | 2,001 | 57 |
| `data/inference/aux.csv` | 3,351 | 101 |
| `data/inference/catboost.csv` | 3,351 | 57 |

The two ALIGNN tables use `uid`; the CatBoost tables use `sample_id`. Their material IDs match the corresponding packaged structure sets. Keep descriptor column names and their recorded order unchanged. `D05_graph_features.csv` and `D13_catboost_features.csv` describe the columns; they do not replace the numerical inputs above.

## 3. Environment and local imports

The recorded working environment used Python **3.10.19**, PyTorch **2.1.2+cu121**, DGL **2.0.0+cu121**, and NumPy **1.26.4**. See:

- `environment/requirements-recorded.txt`: principal package versions.
- `environment/inspected_environment.json`: the fuller installed-package inventory.

Create an isolated Python 3.10 environment, for example:

```powershell
conda create -n pag-alignn python=3.10 pip -y
conda activate pag-alignn
```

Install compatible PyTorch, DGL, and torchdata builds for the selected operating system and CPU/CUDA runtime. The recorded GPU setup uses CUDA 12.1 builds and `torchdata==0.7.1`. The `+cu121` packages require their corresponding binary distribution channels; the recorded requirements file is an environment record, not a universal installation lockfile. This directory does not contain an offline wheel bundle.

After installing these framework packages, the other principal recorded dependencies can be installed with:

```powershell
python -m pip install numpy==1.26.4 pandas==2.3.3 scikit-learn==1.7.2 jarvis-tools==2026.1.10 catboost==1.2.10 spglib==2.7.0 pydantic==2.12.5 pydantic-settings==2.13.1 matplotlib==3.10.6 ase==3.27.0 pymatgen==2025.10.7 lmdb==1.7.5 scipy==1.15.3 tqdm==4.67.3 networkx==3.4.2
```

Run the following examples from the **package root**, using PowerShell. Set the import path so Python uses the modified implementation supplied here:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) "code")
$env:PYTHONDONTWRITEBYTECODE = "1"
python -c "import alignn, torch, dgl, jarvis, catboost, spglib; print(alignn.__file__); print(torch.__version__, dgl.__version__)"
```

The printed ALIGNN path should point inside this package's `code/alignn/`. Using an unrelated installed ALIGNN version can omit the changes required by these checkpoints. Bash users can instead set `export PYTHONPATH="$PWD/code"` and `export PYTHONDONTWRITEBYTECODE=1`; the short inference commands below also work in Bash. PowerShell here-strings in later examples can be copied into a temporary `.py` file and run with Python on other platforms.

## 4. Inference with supplied checkpoints

Run the two packaged validation candidates as a small input check:

```powershell
python predict.py --smoke --device cpu --output work/smoke/predictions.csv
```

`--smoke` selects `2LiMnCl3-1` and `4FIn-2` from the packaged inference structures. It still loads the classification models and all regression ensembles; it is not a replacement model or a shortened ensemble.

Run all 3,351 inference structures:

```powershell
python predict.py --device cpu --output work/inference/predictions.csv
```

Use `--device cuda` with a compatible GPU environment. CPU execution can take substantially longer. The wrapper writes predictions and intermediate calculation files under the selected output directory. Use separate output paths for separate runs.

For custom materials:

```powershell
python predict.py --structures custom/structures.json --aux custom/aux.csv --cat custom/catboost.csv --device cpu --output work/custom/predictions.csv
```

Follow the schemas of `data/inference/structures.json`, `aux.csv`, and `catboost.csv`. Each structure entry contains its material ID and a `structure` object; all three inputs must refer to the same materials. Provide the same descriptor columns used by the archived models, including CatBoost categorical columns. Do not refit descriptor encodings independently for a new prediction set.

In the routed output, tensor-route materials retain all six tensor components. Eta-route materials retain eta_parallel and the three out-of-plane components; xx/yy/xy are intentionally absent from the compact routed table. This is not missing prediction data. `D02_eta_tensor_complement.csv` provides a separate supplementary tensor analysis for the eta-routed materials.

## 5. Labels, predictions, and metric checks

The current supplementary filenames are:

| File in `supplementary_data/` | Records | Contents |
| --- | ---: | --- |
| `D01_routed_predictions.csv` | 3,351 | Gate probabilities, route decisions, and retained outputs |
| `D02_eta_tensor_complement.csv` | 2,515 | Supplementary tensor analysis of eta-route materials |
| `D03_eta_ablation.csv` | 8 | Eta-model comparison records |
| `D04_screened_candidates_16.csv` | 16 | Ranked tensor-route candidates |
| `D05_graph_features.csv` | 101 | Global descriptor names and order |
| `D06_split_overlap.csv` | 25 | Split-size and overlap summaries |
| `D07_tensor_predictions.csv` | 1,803 | Tensor targets and OOF predictions |
| `D08_eta_predictions.csv` | 2,001 | Eta targets and OOF predictions, including raw/clipped values |
| `D9_static_labels_2001.csv` | 2,001 | Base static lattice-polarizability labels |
| `D10_tensor_labels_1803.csv` | 1,803 | Cleaned tensor-task labels |
| `D11_classifier_oof.csv` | 1,998 | Common-sample classifier comparison and fusion predictions |
| `D12_fusion_search.csv` | 101 | Fusion-weight search records |
| `D13_catboost_features.csv` | 57 | CatBoost descriptor definitions |

`D9_static_labels_2001.csv` intentionally appears above with its actual, unpadded filename. `model_records.json` contains additional archived model/configuration/split/metric records. `results/` contains further outputs; it is not a replacement for the main CSV tables.

The following check reads the existing CSVs without loading model weights, installing machine-learning packages, or writing output files. It uses only the Python standard library:

```powershell
@'
from pathlib import Path
import csv, math

base = Path("supplementary_data")
def read(name):
    with (base / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

tensor = read("D07_tensor_predictions.csv")
f_error, trace_error = [], []
for row in tensor:
    d = {c: float(row["prediction_" + c]) - float(row["target_" + c])
         for c in ("xx", "yy", "zz", "xy", "xz", "yz")}
    f_error.append(math.sqrt(sum(d[c]**2 for c in ("xx", "yy", "zz"))
                            + 2 * sum(d[c]**2 for c in ("xy", "xz", "yz"))))
    trace_error.append(abs(d["xx"] + d["yy"] + d["zz"]))
print("Tensor samples:", len(tensor))
print("Frobenius MAE (angstrom):", sum(f_error) / len(f_error))
print("Trace MAE (angstrom):", sum(trace_error) / len(trace_error))

eta = read("D08_eta_predictions.csv")
eta_mae = sum(abs(float(r["prediction_eta_parallel"])
                  - float(r["target_eta_parallel"])) for r in eta) / len(eta)
print("Eta samples:", len(eta), "MAE:", eta_mae)

gate = read("D11_classifier_oof.csv")
counts = {(y, p): 0 for y in (0, 1) for p in (0, 1)}
for row in gate:
    counts[int(row["target"]), int(row["prediction_fused"]) ] += 1
print("Classifier TN, FP, FN, TP:",
      counts[0, 0], counts[0, 1], counts[1, 0], counts[1, 1])
'@ | python -
```

Expected values, subject to displayed rounding:

```text
Tensor samples: 1803
Frobenius MAE: 1.6493906140278884 Å
Trace MAE: 1.7580296191742681 Å
Eta samples: 2001; MAE: 0.07885892849921351
Classifier TN, FP, FN, TP: 1587, 215, 17, 179
```

The factor of two on off-diagonal squared errors accounts for both symmetric entries in the full 3-by-3 tensor. These checks reproduce metrics from the archived predictions; they do not rerun model inference. Do not apply inverse target transformations again to the physical-space values in these CSVs.

For predictions generated by a new training run, the supplied evaluation entry points include `code/scripts/evaluate_eta_parallel_model.py`, `evaluate_polarizability_split.py`, and `evaluate_polarizability_tensor.py`. Their run-directory inputs must contain the applicable prediction files and matching configuration; use `--help` to inspect the arguments. Tensor reconstruction must combine xx/yy/xy from the in-plane branch, zz from the primary out-of-plane branch, and xz/yz from the auxiliary branch.

## 6. Retraining from the existing files

### Training inputs are present; the root wrapper is incomplete

The root-level `train_fold.py` currently expects `configs/experiment_index.json` and `data/structure_pool/`, neither of which is included in this copy. Consequently, even `python train_fold.py --list` fails at the missing index. This does **not** mean the supplied model implementation cannot be trained: the 2,001 full structures are present in `data/structures/static_2001.json`, and the branch-specific labels, split IDs, weights, and configurations are archived separately.

The existing `code/scripts/prepare_polarizability_alignn.py` includes `load_structures` and `write_poscar` functions. They can reconstruct the structure files from the packaged JSON. Exporting all 2,001 structures with that function reproduced the recorded eta-fold structure hashes in a local preparation check. No missing structure IDs were found across the four regression branches' five-fold label files.

### Prepare one archived fold without the root wrapper

The example below prepares **eta, fold 01** in a new `work/retraining/` directory. It calls the existing export function and reuses the archived labels and split IDs instead of generating a new random split. It also retains the checkpoint's descriptor order and the archived hyperparameters.

```powershell
@'
from pathlib import Path
import csv, hashlib, json, runpy, shutil

root = Path.cwd()
branch, fold = "eta", "fold_01"
source = root / "data/training/main" / branch / fold
out = root / "work/retraining" / branch / fold
if out.exists() and any(out.iterdir()):
    raise RuntimeError("Choose a new, empty output directory before preparing this fold.")
prepared = out / "input"
prepared.mkdir(parents=True, exist_ok=True)

tools = runpy.run_path(str(root / "code/scripts/prepare_polarizability_alignn.py"),
                       run_name="prepare_existing_fold")
structures = tools["load_structures"](root / "data/structures/static_2001.json")
mapping = json.loads((source / "structures.json").read_text(encoding="utf-8"))
for uid, expected_hash in mapping.items():
    target = prepared / uid
    if target.parent.resolve() != prepared.resolve():
        raise ValueError("Unexpected structure filename: " + uid)
    tools["write_poscar"](target, uid, structures[uid])
    text = target.read_bytes().replace(b"\r\n", b"\n")
    hashes = {hashlib.sha256(text).hexdigest(),
              hashlib.sha256(text.replace(b"\n", b"\r\n")).hexdigest()}
    if expected_hash not in hashes:
        raise ValueError("Structure hash mismatch: " + uid)

for name in ("id_prop.csv", "ids_train_val_test.json",
             "sample_weights.csv", "stratify_labels.csv"):
    if (source / name).exists():
        shutil.copyfile(source / name, prepared / name)

cfg = json.loads((root / "configs/main" / branch / fold / "config.json")
                 .read_text(encoding="utf-8"))
cfg["output_dir"] = str(out / "run")
if cfg.get("sample_weight_csv"):
    cfg["sample_weight_csv"] = str(prepared / "sample_weights.csv")
(out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

names = json.loads((root / "models" / branch / "folds" / fold
                    / "extra_feature_names.json").read_text(encoding="utf-8"))
with (root / "data/features/training_aux.csv").open(
        encoding="utf-8-sig", newline="") as f:
    records = list(csv.DictReader(f))
assert set(mapping) <= {r["uid"] for r in records}
with (out / "features.csv").open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["uid"] + names)
    writer.writeheader()
    writer.writerows({"uid": r["uid"], **{k: r[k] for k in names}}
                     for r in records if r["uid"] in mapping)
print("Prepared:", branch, fold, "samples:", len(mapping), "features:", len(names))
'@ | python -
```

With the environment and local import path configured as in Section 3, start the bottom-level training module:

```powershell
python -m alignn.train_alignn --root_dir work/retraining/eta/fold_01/input --config_name work/retraining/eta/fold_01/config.json --output_dir work/retraining/eta/fold_01/run --extra_features_csv work/retraining/eta/fold_01/features.csv --extra_features_id_key uid
```

For the other regression tasks, change `branch` to `inplane`, `outplane_zz`, or `outplane_aux`; for the other folds, use `fold_02` through `fold_05`. Change the training command paths accordingly. The preparation code deliberately leaves the original package data, configurations, and checkpoints unchanged. New results are written under `work/retraining/`.

The archived main configurations specify 300 epochs, batch size 16, learning rate 0.001, and random seed 123, along with branch-specific settings. The saved configuration is authoritative. The general preparation script also contains defaults for other or historical experiments; running it with only a target name is not equivalent to reproducing the archived main model.

This README's preparation route has been checked using the included data. A complete training run and checkpoint-based inference have not been rerun as part of writing this README. Floating-point behavior, hardware, and library builds may affect newly trained weights and exact numerical results.

## 7. Five-fold records and their interpretation

Detailed material ID lists are provided in:

```text
data/training/main/eta/fold_01/ids_train_val_test.json
data/training/main/inplane/fold_01/ids_train_val_test.json
data/training/main/outplane_zz/fold_01/ids_train_val_test.json
data/training/main/outplane_aux/fold_01/ids_train_val_test.json
data/training/classification_oof/fold_01/ids_train_val_test.json
```

Each group also has folds 02-05. The JSON keys include `id_train`, `id_val`, and `id_test`; some records contain additional selection/prediction lists. Each outer test material occurs once across the five folds: 2,001 materials for eta and classification, and 1,803 for each tensor branch. `data/training/classification/full/` belongs to the full-data deployment classifier, not an extra OOF fold.

The archived eta and classification OOF splits have disjoint training, validation, and test IDs within each fold. For the archived in-plane and both out-of-plane branches, `id_val` is a subset of `id_test`, while `id_train` and `id_test` are disjoint. Thus their archived validation and test sets must not be described as three mutually exclusive partitions. `D06_split_overlap.csv` records these overlaps. Replaying the archived split preserves this protocol; it does not create a new independent test set.

The common classifier comparison CSV contains 1,998 aligned records, whereas the base classifier split lists cover 2,001. Use the supplied common-record table when reproducing the reported fusion comparison rather than silently substituting a different evaluation population. Historical cleaning experiments with names such as `cleaned_1801` are additional comparisons, not the current main 2,001-sample eta training set.

## 8. Plotting, computational verification, and scope

`code/publication_figures/` contains plotting code, but not every script is self-contained with its current defaults:

- The candidate-map script defaults to a local manuscript path. Its `--manuscript` or `--candidates-table` argument must be supplied with an available input when used elsewhere.
- The real-band and band-offset scripts expect material files under `data/publication_figures/materials/`; that directory is absent from this copy.
- Tensor-error comparison plots need the comparison prediction inputs selected by their arguments; inspect `--help` and supply those inputs explicitly.

Do not treat the presence of plotting code as confirmation that every manuscript figure can be regenerated from this folder alone. It also does not contain raw VASP input/output sets such as INCAR, KPOINTS, OUTCAR, or vasprun.xml. The manuscript and Supporting Information describe the first-principles verification protocol; this folder primarily supplies the machine-learning materials.

CSV filename references in the configuration, plotting scripts, and classification records use the current names `D07_tensor_predictions.csv`, `D08_eta_predictions.csv`, and `D12_fusion_search.csv`.

## 9. Attribution and terms

Please cite the associated manuscript when using these materials and acknowledge the upstream ALIGNN and C2DB resources as appropriate. The upstream ALIGNN notice is included in `licenses/ALIGNN_LICENSE.rst`. This README does not assign a new license to the authors' additions, data, or third-party resources; retain the applicable notices.
