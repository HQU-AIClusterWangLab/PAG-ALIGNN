#!/usr/bin/env python

"""Module to train for a folder with formatted dataset."""
import os
import shutil
import torch.distributed as dist
import csv
import sys
import json
import zipfile
from alignn.data import get_train_val_loaders
from alignn.train import train_dgl
from alignn.config import TrainingConfig
from jarvis.db.jsonutils import loadjson
import argparse
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig
import torch
import time
from jarvis.core.atoms import Atoms
import random
import math
from ase.stress import voigt_6_to_full_3x3_stress

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")


def formula_key_from_atoms(atoms):
    """Return a stable composition key from a JARVIS Atoms object."""
    counts = {}
    for element in atoms.elements:
        counts[element] = counts.get(element, 0) + 1
    divisor = 0
    for count in counts.values():
        divisor = count if divisor == 0 else math.gcd(divisor, count)
    divisor = max(divisor, 1)
    return "".join(
        f"{element}{counts[element] // divisor}"
        for element in sorted(counts)
    )


def setup(rank=0, world_size=0, port="12356"):
    """Set up multi GPU rank."""
    # "12356"
    if port == "":
        port = str(random.randint(10000, 99999))
    if world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = port
        # os.environ["MASTER_PORT"] = "12355"
        # Initialize the distributed environment.
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

def cleanup(world_size):
    """Clean up distributed process."""
    if world_size > 1:
        dist.destroy_process_group()


parser = argparse.ArgumentParser(
    description="Atomistic Line Graph Neural Network"
)
parser.add_argument(
    "--root_dir",
    default="./",
    help="Folder with id_props.csv, structure files",
)
parser.add_argument(
    "--config_name",
    default="alignn/examples/sample_data/config_example.json",
    help="Name of the config file",
)

parser.add_argument(
    "--file_format", default="poscar", help="poscar/cif/xyz/pdb file format."
)

# parser.add_argument(
#    "--keep_data_order",
#    default=True,
#    help="Whether to randomly shuffle samples",
# )

parser.add_argument(
    "--classification_threshold",
    default=None,
    help="Floating point threshold for converting into 0/1 class"
    + ", use only for classification tasks",
)

parser.add_argument(
    "--batch_size", default=None, help="Batch size, generally 64"
)

parser.add_argument(
    "--epochs", default=None, help="Number of epochs, generally 300"
)


parser.add_argument(
    "--target_key",
    default="total_energy",
    help="Name of the key for graph level data such as total_energy",
)

parser.add_argument(
    "--id_key",
    default="jid",
    help="Name of the key for graph level id such as id",
)

parser.add_argument(
    "--force_key",
    default="forces",
    help="Name of key for gradient level data such as forces, (Natoms x p)",
)

parser.add_argument(
    "--atomwise_key",
    default="forces",
    help="Name of key for atomwise level data: forces, charges (Natoms x p)",
)


parser.add_argument(
    "--stresswise_key",
    default="stresses",
    help="Name of the key for stress (3x3) level data such as forces",
)

parser.add_argument(
    "--additional_output_key",
    default="additional_output",
    help="Name of the key for extra global output eg DOS",
)


parser.add_argument(
    "--output_dir",
    default="./",
    help="Folder to save outputs",
)


parser.add_argument(
    "--extra_features_csv",
    default=None,
    help=(
        "Optional descriptor CSV with a header row. "
        "Example columns: id,f1,f2,... where each row maps to one sample."
    ),
)

parser.add_argument(
    "--extra_features_id_key",
    default=None,
    help=(
        "Column name in extra_features_csv used for joining rows with "
        "samples (e.g. id or jid). If omitted, use config value."
    ),
)


parser.add_argument(
    "--restart_model_path",
    default=None,
    help="Checkpoint file path for model",
)


parser.add_argument(
    "--device",
    default=None,
    help="set device for training the model [e.g. cpu, cuda, cuda:2]",
)


def train_for_folder(
    rank=0,
    world_size=0,
    root_dir="examples/sample_data",
    config_name="config.json",
    classification_threshold=None,
    batch_size=None,
    epochs=None,
    id_key="jid",
    target_key="total_energy",
    atomwise_key="forces",
    gradwise_key="forces",
    stresswise_key="stresses",
    additional_output_key="additional_output",
    file_format="poscar",
    restart_model_path=None,
    output_dir=None,
    extra_features_csv=None,
    extra_features_id_key=None,
):
    """Train for a folder."""
    setup(rank=rank, world_size=world_size)
    print("root_dir", root_dir)
    id_prop_json = os.path.join(root_dir, "id_prop.json")
    id_prop_json_zip = os.path.join(root_dir, "id_prop.json.zip")
    id_prop_csv = os.path.join(root_dir, "id_prop.csv")
    id_prop_csv_file = False
    multioutput = False
    # lists_length_equal = True
    if os.path.exists(id_prop_json_zip):
        dat = json.loads(
            zipfile.ZipFile(id_prop_json_zip).read("id_prop.json")
        )
    elif os.path.exists(id_prop_json):
        dat = loadjson(os.path.join(root_dir, "id_prop.json"))
    elif os.path.exists(id_prop_csv):
        id_prop_csv_file = True
        with open(id_prop_csv, "r") as f:
            reader = csv.reader(f)
            dat = [row for row in reader]
        print("id_prop_csv_file exists", id_prop_csv_file)
    else:
        print("Check dataset file.")
    config_dict = loadjson(config_name)
    # Keep external-descriptor options backward-compatible with older
    # TrainingConfig schemas (which may forbid unknown keys).
    extra_cfg = {
        "use_extra_features_csv": config_dict.pop(
            "use_extra_features_csv", None
        ),
        "extra_features_csv": config_dict.pop("extra_features_csv", None),
        "extra_features_id_key": config_dict.pop(
            "extra_features_id_key", None
        ),
    }
    # These keys are consumed by the local data-preparation workflow only.
    # TrainingConfig forbids unknown keys, so strip them if an older generated
    # config still contains them.
    for key in list(config_dict):
        if key.startswith("stratify_"):
            config_dict.pop(key)
    config = TrainingConfig(**config_dict)
    if type(config) is dict:
        try:
            config = TrainingConfig(**config)
        except Exception as exp:
            print("Check", exp)

    # config.keep_data_order = keep_data_order
    if classification_threshold is not None:
        config.classification_threshold = float(classification_threshold)
    if output_dir is not None:
        config.output_dir = output_dir
    if batch_size is not None:
        config.batch_size = int(batch_size)
    if epochs is not None:
        config.epochs = int(epochs)
    root_split_file = os.path.join(root_dir, "ids_train_val_test.json")
    if os.path.exists(root_split_file):
        os.makedirs(config.output_dir, exist_ok=True)
        shutil.copyfile(
            root_split_file,
            os.path.join(config.output_dir, "ids_train_val_test.json"),
        )
        root_stratify_file = os.path.join(root_dir, "stratify_labels.csv")
        if os.path.exists(root_stratify_file):
            shutil.copyfile(
                root_stratify_file,
                os.path.join(config.output_dir, "stratify_labels.csv"),
            )
        print("Using split file from root_dir:", root_split_file)
    cfg_extra_id_key = (
        extra_cfg["extra_features_id_key"]
        if extra_cfg["extra_features_id_key"] is not None
        else getattr(config, "extra_features_id_key", "id")
    )
    cfg_extra_csv = (
        extra_cfg["extra_features_csv"]
        if extra_cfg["extra_features_csv"] is not None
        else getattr(config, "extra_features_csv", None)
    )
    cfg_use_extra_csv = (
        extra_cfg["use_extra_features_csv"]
        if extra_cfg["use_extra_features_csv"] is not None
        else getattr(config, "use_extra_features_csv", False)
    )
    if extra_features_id_key is None:
        extra_features_id_key = cfg_extra_id_key
    if extra_features_csv is None:
        if cfg_extra_csv not in (None, ""):
            extra_features_csv = cfg_extra_csv
        elif cfg_use_extra_csv:
            raise ValueError(
                "Config sets use_extra_features_csv=True but "
                "extra_features_csv is empty."
            )

    # Optional path: load per-sample global descriptors from a standalone CSV.
    # This keeps the original label CSV unchanged while enabling extra features.
    extra_features_map = {}
    extra_feature_names = []
    if extra_features_csv is not None:
        if not os.path.exists(extra_features_csv):
            raise FileNotFoundError(
                f"extra_features_csv not found: {extra_features_csv}"
            )
        with open(extra_features_csv, "r") as ef:
            # Header-aware reader so descriptor names can be tracked.
            reader = csv.DictReader(ef)
            if reader.fieldnames is None:
                raise ValueError(
                    "extra_features_csv must include a header row."
                )
            id_col = extra_features_id_key
            if id_col not in reader.fieldnames:
                raise ValueError(
                    f"Column '{id_col}' not in extra_features_csv header: "
                    + ",".join(reader.fieldnames)
                )
            # Keep original header order for descriptor columns.
            extra_feature_names = [
                c for c in reader.fieldnames if c != extra_features_id_key
            ]
            if len(extra_feature_names) == 0:
                raise ValueError(
                    "No descriptor columns found in extra_features_csv."
                )
            for row in reader:
                key = str(row[id_col]).strip()
                if key == "":
                    continue
                # Convert descriptor columns to float vectors.
                extra_features_map[key] = [
                    float(row[c]) for c in extra_feature_names
                ]
        # Ensure model input dimension matches descriptor count.
        if config.model.extra_features not in (0, len(extra_feature_names)):
            raise ValueError(
                "config.model.extra_features must be 0 or match descriptor "
                + f"count. Found config={config.model.extra_features}, "
                + f"csv={len(extra_feature_names)}"
            )
        config.model.extra_features = len(extra_feature_names)
        if not os.path.exists(config.output_dir):
            os.makedirs(config.output_dir)
        # Persist descriptor names for downstream analysis/reproducibility.
        feat_name_file = os.path.join(
            config.output_dir, "extra_feature_names.json"
        )
        with open(feat_name_file, "w") as f:
            f.write(json.dumps(extra_feature_names, indent=2))
        print(
            "Loaded extra features:",
            len(extra_features_map),
            "rows and",
            len(extra_feature_names),
            "columns.",
        )

    sample_weight_map = {}
    if config.use_sample_weights:
        if config.sample_weight_csv in (None, ""):
            raise ValueError(
                "use_sample_weights=True but sample_weight_csv is empty."
            )
        if not os.path.exists(config.sample_weight_csv):
            raise FileNotFoundError(
                f"sample_weight_csv not found: {config.sample_weight_csv}"
            )
        with open(config.sample_weight_csv, "r") as sw:
            reader = csv.DictReader(sw)
            if reader.fieldnames is None:
                raise ValueError("sample_weight_csv must include a header row.")
            id_col = config.sample_weight_id_key
            value_col = config.sample_weight_value_key
            if id_col not in reader.fieldnames:
                raise ValueError(
                    f"Column '{id_col}' not in sample_weight_csv header."
                )
            if value_col not in reader.fieldnames:
                raise ValueError(
                    f"Column '{value_col}' not in sample_weight_csv header."
                )
            for row in reader:
                key = str(row[id_col]).strip()
                if key == "":
                    continue
                sample_weight_map[key] = float(row[value_col])
        sample_weight_values = list(sample_weight_map.values())
        print(
            "Loaded sample weights:",
            len(sample_weight_map),
            "rows.",
            "min",
            min(sample_weight_values),
            "max",
            max(sample_weight_values),
            "mean",
            sum(sample_weight_values) / len(sample_weight_values),
        )

    train_grad = False
    train_stress = False
    train_additional_output = False
    train_atom = False
    try:
        if (
            config.model.calculate_gradient
            and config.model.gradwise_weight != 0
        ):
            train_grad = True
        else:
            train_grad = False
        if (
            config.model.calculate_gradient
            and config.model.stresswise_weight != 0
        ):
            train_stress = True
        else:
            train_stress = False
        if config.model.atomwise_weight != 0:
            train_atom = True
        else:
            train_atom = False
        if (
            config.model.additional_output_features > 0
            and config.model.additional_output_weight != 0
        ):
            train_additional_output = True
        else:
            train_additional_output = False
    except Exception as exp:
        print("exp", exp)
        pass
    # if config.model.atomwise_weight == 0:
    #    train_atom = False
    # if config.model.gradwise_weight == 0:
    #    train_grad = False
    # if config.model.stresswise_weight == 0:
    #    train_stress = False
    target_atomwise = None  # "atomwise_target"
    target_grad = None  # "atomwise_grad"
    target_stress = None  # "stresses"
    target_additional_output = None  # "stresses"

    # mem = []
    # enp = []
    n_outputs = []
    dataset = []
    missing_extra = []
    missing_weights = []
    use_same_formula_pair_losses = bool(
        getattr(config, "eta_pairwise_loss_enabled", False)
        or getattr(config, "eta_embedding_contrastive_loss_enabled", False)
    )
    for i in dat:
        info = {}
        if id_prop_csv_file:
            file_name = i[0]
            tmp = [float(j) for j in i[1:]]  # float(i[1])
            info["jid"] = file_name

            if len(tmp) == 1:
                tmp = tmp[0]
            else:
                multioutput = True
                n_outputs.append(tmp)
            info["target"] = tmp
            file_path = os.path.join(root_dir, file_name)
            if file_format == "poscar":
                atoms = Atoms.from_poscar(file_path)
            elif file_format == "cif":
                atoms = Atoms.from_cif(file_path)
            elif file_format == "xyz":
                atoms = Atoms.from_xyz(file_path, box_size=500)
            elif file_format == "pdb":
                # Note using 500 angstrom as box size
                # Recommended install pytraj
                # conda install -c ambermd pytraj
                atoms = Atoms.from_pdb(file_path, max_lat=500)
            else:
                raise NotImplementedError(
                    "File format not implemented", file_format
                )
            info["atoms"] = atoms.to_dict()
            if use_same_formula_pair_losses:
                info["pairwise_formula_key"] = formula_key_from_atoms(atoms)
            sample_key = str(file_name)
        else:
            info["target"] = i[target_key]
            info["atoms"] = i["atoms"]
            info["jid"] = i[id_key]
            atoms = Atoms.from_dict(i["atoms"])
            if use_same_formula_pair_losses:
                info["pairwise_formula_key"] = formula_key_from_atoms(atoms)
            sample_key = str(i[id_key])
        if train_atom:
            target_atomwise = "atomwise_target"
            info["atomwise_target"] = i[atomwise_key]  # such as charges
        if train_grad:
            target_grad = "atomwise_grad"
            info["atomwise_grad"] = i[gradwise_key]  # - mean_force
        if train_stress:
            if len(i[stresswise_key]) == 6:

                stress = voigt_6_to_full_3x3_stress(i[stresswise_key])
            else:
                stress = i[stresswise_key]
            info["stresses"] = stress  # - mean_force
            target_stress = "stresses"

        if train_additional_output:
            target_additional_output = "additional"
            info["additional"] = i[additional_output_key]  # - mean_force
        # Priority: external descriptor CSV > in-record extra_features key.
        if sample_key in extra_features_map:
            info["extra_features"] = extra_features_map[sample_key]
        elif "extra_features" in i:
            info["extra_features"] = i["extra_features"]
        elif extra_features_csv is not None:
            # If descriptor CSV is provided, each sample must have a match.
            missing_extra.append(sample_key)
        if config.use_sample_weights:
            if sample_key in sample_weight_map:
                info["sample_weight"] = sample_weight_map[sample_key]
            else:
                missing_weights.append(sample_key)
        dataset.append(info)
    if len(missing_extra) > 0:
        miss = ",".join(missing_extra[:10])
        raise ValueError(
            "Could not find extra descriptors for some samples. "
            + f"Missing count={len(missing_extra)}. Examples: {miss}"
        )
    if len(missing_weights) > 0:
        miss = ",".join(missing_weights[:10])
        raise ValueError(
            "Could not find sample weights for some samples. "
            + f"Missing count={len(missing_weights)}. Examples: {miss}"
        )
    print("len dataset", len(dataset))
    print("train_stress", train_stress)
    del dat
    # multioutput = False
    lists_length_equal = True
    line_graph = False
    # alignn_models = {
    #    # "alignn",
    #    # "alignn_layernorm",
    #    "alignn_atomwise",
    # }

    if config.compute_line_graph > 0:
        # if config.model.alignn_layers > 0:
        line_graph = True

    if multioutput:
        print("multioutput", multioutput)
        lists_length_equal = False not in [
            len(i) == len(n_outputs[0]) for i in n_outputs
        ]
        print("lists_length_equal", lists_length_equal, len(n_outputs[0]))
        if lists_length_equal:
            config.model.output_features = len(n_outputs[0])

        else:
            raise ValueError("Make sure the outputs are of same size.")
    model = None
    if restart_model_path is not None:
        # Should be best_model.pt file
        print("Restarting the model training:", restart_model_path)
        if config.model.name == "alignn_atomwise":
            rest_config = loadjson(
                restart_model_path.replace("current_model.pt", "config.json")
                # restart_model_path.replace("best_model.pt", "config.json")
            )

            tmp = ALIGNNAtomWiseConfig(**rest_config["model"])
            print("Rest config", tmp)
            model = ALIGNNAtomWise(tmp)  # config.model)
            print("model", model)
            model.load_state_dict(
                torch.load(restart_model_path, map_location=device)
            )
            model = model.to(device)

    # print ('n_outputs',n_outputs[0])
    # if multioutput and classification_threshold is not None:
    #    raise ValueError("Classification for multi-output not implemented.")
    # if multioutput and lists_length_equal:
    #    config.model.output_features = len(n_outputs[0])
    # else:
    #    # TODO: Pad with NaN
    #    if not lists_length_equal:
    #        raise ValueError("Make sure the outputs are of same size.")
    #    else:
    #        config.model.output_features = 1
    # print('config.neighbor_strategy',config.neighbor_strategy)
    # import sys
    # sys.exit()
    (
        train_loader,
        val_loader,
        test_loader,
        prepare_batch,
    ) = get_train_val_loaders(
        dataset_array=dataset,
        target="target",
        target_atomwise=target_atomwise,
        target_grad=target_grad,
        target_stress=target_stress,
        target_additional_output=target_additional_output,
        n_train=config.n_train,
        n_val=config.n_val,
        n_test=config.n_test,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        line_graph=line_graph,
        batch_size=config.batch_size,
        atom_features=config.atom_features,
        neighbor_strategy=config.neighbor_strategy,
        standardize=config.atom_features != "cgcnn",
        id_tag=config.id_tag,
        pin_memory=config.pin_memory,
        workers=config.num_workers,
        save_dataloader=config.save_dataloader,
        use_canonize=config.use_canonize,
        filename=config.filename,
        cutoff=config.cutoff,
        cutoff_extra=config.cutoff_extra,
        max_neighbors=config.max_neighbors,
        output_features=config.model.output_features,
        classification_threshold=config.classification_threshold,
        target_multiplication_factor=config.target_multiplication_factor,
        standard_scalar_and_pca=config.standard_scalar_and_pca,
        keep_data_order=config.keep_data_order,
        output_dir=config.output_dir,
        use_lmdb=config.use_lmdb,
        dtype=config.dtype,
        use_site_symmetry_node_features=(
            config.model.use_site_symmetry_node_features
        ),
        site_symmetry_symprec=config.model.site_symmetry_symprec,
        site_symmetry_angle_tolerance=(
            config.model.site_symmetry_angle_tolerance
        ),
        site_symmetry_node_features_dim=(
            config.model.site_symmetry_node_features
        ),
    )
    # print("dataset", dataset[0])
    t1 = time.time()
    # world_size = torch.cuda.device_count()
    print("rank", rank)
    print("world_size", world_size)
    train_dgl(
        config,
        model=model,
        train_val_test_loaders=[
            train_loader,
            val_loader,
            test_loader,
            prepare_batch,
        ],
        rank=rank,
        world_size=world_size,
    )
    t2 = time.time()
    print("Time taken (s)", t2 - t1)

    # train_data = get_torch_dataset(


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    world_size = int(torch.cuda.device_count())
    print("world_size", world_size)
    if world_size > 1:
        torch.multiprocessing.spawn(
            train_for_folder,
            args=(
                world_size,
                args.root_dir,
                args.config_name,
                args.classification_threshold,
                args.batch_size,
                args.epochs,
                args.id_key,
                args.target_key,
                args.atomwise_key,
                args.force_key,
                args.stresswise_key,
                args.additional_output_key,
                args.file_format,
                args.restart_model_path,
                args.output_dir,
                args.extra_features_csv,
                args.extra_features_id_key,
            ),
            nprocs=world_size,
        )
    else:
        train_for_folder(
            0,
            world_size,
            args.root_dir,
            args.config_name,
            args.classification_threshold,
            args.batch_size,
            args.epochs,
            args.id_key,
            args.target_key,
            args.atomwise_key,
            args.force_key,
            args.stresswise_key,
            args.additional_output_key,
            args.file_format,
            args.restart_model_path,
            args.output_dir,
            args.extra_features_csv,
            args.extra_features_id_key,
        )
    try:
        cleanup(world_size)
    except Exception:
        pass
