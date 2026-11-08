"""Module for training script."""

from torch.nn.parallel import DistributedDataParallel as DDP
from functools import partial
from typing import Any, Dict, Union
import csv
import torch
import random
from sklearn.metrics import mean_absolute_error
import pickle as pk
import numpy as np
from torch import nn
from alignn.data import get_train_val_loaders
from alignn.config import TrainingConfig
from alignn.models.alignn_atomwise import ALIGNNAtomWise
from alignn.models.ealignn_atomwise import eALIGNNAtomWise
from alignn.models.alignn import ALIGNN
from jarvis.db.jsonutils import dumpjson
import json
import pprint
import os
import warnings
import time
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from alignn.utils import (
    # activated_output_transform,
    # make_standard_scalar_and_pca,
    # thresholded_output_transform,
    group_decay,
    setup_optimizer,
    print_train_val_loss,
)
import dgl

# from sklearn.metrics import log_loss

warnings.filterwarnings("ignore", category=RuntimeWarning)

# torch.autograd.detect_anomaly()

figlet_alignn = """
    _    _     ___ ____ _   _ _   _
   / \  | |   |_ _/ ___| \ | | \ | |
  / _ \ | |    | | |  _|  \| |  \| |
 / ___ \| |___ | | |_| | |\  | |\  |
/_/   \_\_____|___\____|_| \_|_| \_|
"""


def _graph_sample_weights(graph, device):
    """Return one sample weight per graph in a batched DGL graph."""
    if "sample_weight" not in graph.ndata:
        return None
    weights = [g.ndata["sample_weight"][0] for g in dgl.unbatch(graph)]
    return torch.stack(weights).to(device).view(-1)


def _component_loss_weights(config, n_outputs, device):
    """Return one configured loss weight per graph-level output."""
    configured = getattr(config, "component_loss_weights", None)
    if not configured:
        return None
    target_columns = getattr(config, "target_columns", None)
    if not target_columns:
        raise ValueError(
            "component_loss_weights requires target_columns in the training config."
        )
    if len(target_columns) != n_outputs:
        raise ValueError(
            "target_columns length does not match model output width: "
            f"{len(target_columns)} != {n_outputs}."
        )

    values = []
    for column in target_columns:
        weight = float(configured.get(column, 1.0))
        if weight < 0:
            raise ValueError(f"Loss weight for {column} must be >= 0.")
        values.append(weight)
    weights = torch.tensor(values, dtype=torch.float32, device=device)
    if torch.sum(weights) <= 0:
        raise ValueError("At least one component loss weight must be > 0.")
    return weights


def _configured_component_weight(config, column, default=1.0):
    configured = getattr(config, "component_loss_weights", None)
    if not configured:
        return float(default)
    weight = float(configured.get(column, default))
    if weight < 0:
        raise ValueError(f"Loss weight for {column} must be >= 0.")
    return weight


def _base_regression_element_loss(pred, target, criterion):
    if criterion == "mse":
        return (pred - target) ** 2
    if criterion == "l1":
        return torch.abs(pred - target)
    if criterion == "smooth_l1":
        return torch.nn.functional.smooth_l1_loss(
            pred, target, reduction="none"
        )
    return None


def _inplane_trace_reconstruction_loss(pred, target, config):
    """Return SmoothL1 loss for trace reconstructed from predicted xx and yy."""
    target_columns = getattr(config, "target_columns", None)
    if not target_columns or "inplane_trace_raw" not in target_columns:
        return None
    required = {"xx", "yy", "inplane_trace_raw"}
    if not required.issubset(set(target_columns)):
        return None

    xx_idx = target_columns.index("xx")
    yy_idx = target_columns.index("yy")
    trace_idx = target_columns.index("inplane_trace_raw")
    if max(xx_idx, yy_idx, trace_idx) >= pred.shape[1]:
        return None

    trace_pred_from_components = torch.expm1(pred[:, xx_idx]) + torch.expm1(
        pred[:, yy_idx]
    )
    trace_true = torch.expm1(target[:, xx_idx]) + torch.expm1(target[:, yy_idx])
    pred_trace_log = torch.log1p(trace_pred_from_components.clamp_min(1e-12))
    true_trace_log = torch.log1p(trace_true.clamp_min(1e-12))
    return torch.nn.functional.smooth_l1_loss(
        pred_trace_log,
        true_trace_log,
        reduction="none",
    )


def _inplane_eta_reconstruction_loss(pred, target, config):
    """Return SmoothL1 loss for eta reconstructed from xx, yy, and xy."""
    target_columns = getattr(config, "target_columns", None)
    if not target_columns or "eta_parallel" not in target_columns:
        return None
    required = {"xx", "yy", "xy", "eta_parallel"}
    if not required.issubset(set(target_columns)):
        return None

    xx_idx = target_columns.index("xx")
    yy_idx = target_columns.index("yy")
    xy_idx = target_columns.index("xy")
    eta_idx = target_columns.index("eta_parallel")
    if max(xx_idx, yy_idx, xy_idx, eta_idx) >= pred.shape[1]:
        return None

    pred_xx_raw = torch.expm1(pred[:, xx_idx])
    pred_yy_raw = torch.expm1(pred[:, yy_idx])
    pred_xy_raw = torch.sinh(pred[:, xy_idx])
    trace_recon = pred_xx_raw + pred_yy_raw
    eta_recon = torch.sqrt(
        (pred_xx_raw - pred_yy_raw) ** 2 + 4.0 * pred_xy_raw**2
    ) / trace_recon.clamp_min(1e-12)
    eta_recon_asinh = torch.asinh(
        torch.nan_to_num(eta_recon, nan=0.0, posinf=0.0, neginf=0.0)
    )
    return torch.nn.functional.smooth_l1_loss(
        eta_recon_asinh,
        target[:, eta_idx],
        reduction="none",
    )


def _inplane_raw_tail_loss(pred, target, config):
    """Raw-space SmoothL1 auxiliary loss for high-alpha in-plane samples."""
    if not getattr(config, "raw_tail_loss_enabled", False):
        return None
    target_columns = getattr(config, "target_columns", None)
    if not target_columns or not {"xx", "yy"}.issubset(set(target_columns)):
        return None

    threshold = getattr(config, "raw_tail_loss_threshold", None)
    scale = getattr(config, "raw_tail_loss_scale", None)
    if threshold is None or scale is None or float(scale) <= 0:
        return None

    xx_idx = target_columns.index("xx")
    yy_idx = target_columns.index("yy")
    if max(xx_idx, yy_idx) >= pred.shape[1]:
        return None

    pred_xx_raw = torch.expm1(pred[:, xx_idx])
    pred_yy_raw = torch.expm1(pred[:, yy_idx])
    true_xx_raw = torch.expm1(target[:, xx_idx])
    true_yy_raw = torch.expm1(target[:, yy_idx])
    alpha_pred = 0.5 * (pred_xx_raw + pred_yy_raw)
    alpha_true = 0.5 * (true_xx_raw + true_yy_raw)
    tail_mask = alpha_true > float(threshold)
    if not torch.any(tail_mask):
        return torch.zeros(pred.shape[0], dtype=pred.dtype, device=pred.device)

    scale = float(scale)
    loss_xx = torch.nn.functional.smooth_l1_loss(
        (pred_xx_raw - true_xx_raw) / scale,
        torch.zeros_like(pred_xx_raw),
        reduction="none",
    )
    loss_yy = torch.nn.functional.smooth_l1_loss(
        (pred_yy_raw - true_yy_raw) / scale,
        torch.zeros_like(pred_yy_raw),
        reduction="none",
    )
    alpha_err = (alpha_pred - alpha_true) / scale
    loss_alpha = torch.nn.functional.smooth_l1_loss(
        alpha_err,
        torch.zeros_like(alpha_err),
        reduction="none",
    )
    under_factor = float(getattr(config, "raw_tail_loss_under_factor", 1.5))
    if under_factor < 1.0:
        raise ValueError("raw_tail_loss_under_factor must be >= 1.")
    loss_alpha = torch.where(alpha_err < 0, under_factor * loss_alpha, loss_alpha)
    alpha_weight = float(getattr(config, "raw_tail_loss_alpha_weight", 1.0))
    if alpha_weight < 0:
        raise ValueError("raw_tail_loss_alpha_weight must be >= 0.")
    raw_tail = loss_xx + loss_yy + alpha_weight * loss_alpha
    return torch.where(tail_mask, raw_tail, torch.zeros_like(raw_tail))


def _eta_underestimate_penalty_loss(pred, target, config):
    """Extra loss for under-predicting high eta-like graph targets."""
    if not getattr(config, "eta_underestimate_penalty_enabled", False):
        return None
    target_columns = getattr(config, "target_columns", None)
    penalty_target = getattr(
        config, "eta_underestimate_penalty_target", "eta_parallel"
    )
    if not target_columns or penalty_target not in target_columns:
        return None

    idx = target_columns.index(penalty_target)
    if idx >= pred.shape[1]:
        return None

    threshold = float(getattr(config, "eta_underestimate_penalty_threshold", 0.8))
    penalty_lambda = float(
        getattr(config, "eta_underestimate_penalty_lambda", 0.5)
    )
    if penalty_lambda < 0:
        raise ValueError("eta_underestimate_penalty_lambda must be >= 0.")
    if penalty_lambda == 0:
        return torch.zeros(pred.shape[0], dtype=pred.dtype, device=pred.device)

    power = float(getattr(config, "eta_underestimate_penalty_power", 1.0))
    if power <= 0:
        raise ValueError("eta_underestimate_penalty_power must be > 0.")

    configured_scale = getattr(config, "eta_underestimate_penalty_scale", None)
    if configured_scale is None:
        scale = max(1.0 - threshold, 1e-6)
    else:
        scale = float(configured_scale)
        if scale <= 0:
            raise ValueError("eta_underestimate_penalty_scale must be > 0.")

    target_eta = target[:, idx]
    pred_eta = pred[:, idx]
    tail_strength = torch.clamp((target_eta - threshold) / scale, min=0.0)
    tail_strength = tail_strength.pow(power)
    under_error = torch.clamp(target_eta - pred_eta, min=0.0)

    if config.criterion == "mse":
        under_penalty = under_error**2
    elif config.criterion == "l1":
        under_penalty = under_error
    elif config.criterion == "smooth_l1":
        under_penalty = torch.nn.functional.smooth_l1_loss(
            under_error,
            torch.zeros_like(under_error),
            reduction="none",
        )
    else:
        under_penalty = under_error

    return penalty_lambda * tail_strength * under_penalty


def _target_column_index(config, target_name):
    """Return configured graph-output column index, or None if unavailable."""
    target_columns = getattr(config, "target_columns", None)
    if not target_columns or target_name not in target_columns:
        return None
    return list(target_columns).index(target_name)


def _graph_formula_groups(graph, device):
    """Return one same-composition group id per graph in a DGL batch."""
    if "pairwise_formula_group" not in graph.ndata:
        return None
    groups = [
        g.ndata["pairwise_formula_group"][0]
        for g in dgl.unbatch(graph)
        if g.num_nodes() > 0
    ]
    if not groups:
        return None
    return torch.stack(groups).to(device).long().view(-1)


def _limited_pair_indices(mask, scores, max_pairs):
    """Return upper-triangle pair indices selected by mask and score."""
    selected = torch.nonzero(mask, as_tuple=False)
    if selected.numel() == 0:
        return None, None
    if max_pairs is not None and int(max_pairs) > 0 and selected.shape[0] > int(max_pairs):
        selected_scores = scores[selected[:, 0], selected[:, 1]]
        _, order = torch.topk(selected_scores, k=int(max_pairs), largest=True)
        selected = selected[order]
    return selected[:, 0], selected[:, 1]


def _same_formula_high_delta_pairs(target_values, formula_groups, config, prefix):
    """Return same-formula pair indices filtered by target delta."""
    if formula_groups is None or formula_groups.shape[0] != target_values.shape[0]:
        return None, None
    n_graphs = target_values.shape[0]
    if n_graphs < 2:
        return None, None
    rows, cols = torch.triu_indices(
        n_graphs, n_graphs, offset=1, device=target_values.device
    )
    same_formula = (
        (formula_groups[rows] >= 0)
        & (formula_groups[rows] == formula_groups[cols])
    )
    target_delta = target_values[rows] - target_values[cols]
    min_delta = float(getattr(config, f"{prefix}_min_delta", 0.5))
    high_delta = torch.abs(target_delta) >= min_delta
    mask_flat = same_formula & high_delta
    mask = torch.zeros(
        (n_graphs, n_graphs), dtype=torch.bool, device=target_values.device
    )
    mask[rows, cols] = mask_flat
    score = torch.zeros_like(mask, dtype=target_values.dtype)
    score[rows, cols] = torch.abs(target_delta)
    max_pairs = int(getattr(config, f"{prefix}_max_pairs", 512))
    return _limited_pair_indices(mask, score, max_pairs)


def _eta_pairwise_loss(pred, target, graph, config, device):
    """Prediction-difference loss for same-formula high-delta eta pairs."""
    if not getattr(config, "eta_pairwise_loss_enabled", False):
        return None, 0
    loss_lambda = float(getattr(config, "eta_pairwise_loss_lambda", 0.0))
    if loss_lambda < 0:
        raise ValueError("eta_pairwise_loss_lambda must be >= 0.")
    target_name = str(getattr(config, "eta_pairwise_loss_target", "eta_parallel"))
    idx = _target_column_index(config, target_name)
    if idx is None:
        return None, 0
    pred = pred.to(device)
    target = target.to(device)
    pred, target = _reshape_graph_outputs(pred, target)
    if pred.ndim == 1:
        if idx != 0:
            return None, 0
        pred_values = pred.view(-1)
        target_values = target.view(-1)
    else:
        if idx >= pred.shape[1]:
            return None, 0
        pred_values = pred[:, idx]
        target_values = target[:, idx]
    groups = _graph_formula_groups(graph, device)
    pair_i, pair_j = _same_formula_high_delta_pairs(
        target_values, groups, config, "eta_pairwise_loss"
    )
    if pair_i is None:
        zero = torch.zeros((), dtype=pred_values.dtype, device=device)
        return zero, 0
    pred_delta = pred_values[pair_i] - pred_values[pair_j]
    target_delta = target_values[pair_i] - target_values[pair_j]
    if bool(getattr(config, "eta_pairwise_loss_use_abs_delta", False)):
        pred_delta = torch.abs(pred_delta)
        target_delta = torch.abs(target_delta)
    loss = torch.mean((pred_delta - target_delta) ** 2)
    return loss_lambda * loss, int(pair_i.shape[0])


def _eta_pairwise_explicit_loss(pred, target, n_pairs, config, device):
    """Prediction-difference loss for an explicitly ordered pair batch."""
    if not getattr(config, "eta_pairwise_loss_enabled", False):
        return None, 0
    n_pairs = int(n_pairs)
    if n_pairs <= 0:
        return None, 0
    loss_lambda = float(getattr(config, "eta_pairwise_loss_lambda", 0.0))
    if loss_lambda < 0:
        raise ValueError("eta_pairwise_loss_lambda must be >= 0.")
    target_name = str(getattr(config, "eta_pairwise_loss_target", "eta_parallel"))
    idx = _target_column_index(config, target_name)
    if idx is None:
        return None, 0
    pred = pred.to(device)
    target = target.to(device)
    pred, target = _reshape_graph_outputs(pred, target)
    if pred.ndim == 1:
        if idx != 0:
            return None, 0
        pred_values = pred.view(-1)
        target_values = target.view(-1)
    else:
        if idx >= pred.shape[1]:
            return None, 0
        pred_values = pred[:, idx]
        target_values = target[:, idx]
    n_pairs = min(n_pairs, pred_values.shape[0] // 2)
    if n_pairs <= 0:
        return None, 0
    pair_i = torch.arange(0, 2 * n_pairs, 2, device=device)
    pair_j = pair_i + 1
    pred_delta = pred_values[pair_i] - pred_values[pair_j]
    target_delta = target_values[pair_i] - target_values[pair_j]
    if bool(getattr(config, "eta_pairwise_loss_use_abs_delta", False)):
        pred_delta = torch.abs(pred_delta)
        target_delta = torch.abs(target_delta)
    loss = torch.mean((pred_delta - target_delta) ** 2)
    return loss_lambda * loss, int(n_pairs)


def _dataset_pairwise_formula_groups(dataset):
    """Return one same-composition group id per dataset item."""
    graphs = getattr(dataset, "graphs", None)
    if graphs is None:
        return None
    groups = []
    for graph in graphs:
        if "pairwise_formula_group" not in graph.ndata:
            return None
        groups.append(int(graph.ndata["pairwise_formula_group"][0].item()))
    return groups


def _dataset_pairwise_target_values(dataset, config):
    """Return target values used to build the explicit pair buffer."""
    labels = getattr(dataset, "labels", None)
    if labels is None:
        return None
    target_name = str(getattr(config, "eta_pairwise_loss_target", "eta_parallel"))
    idx = _target_column_index(config, target_name)
    if idx is None:
        return None
    labels = torch.as_tensor(labels).detach().cpu()
    if labels.ndim == 1:
        if idx != 0:
            return None
        values = labels.view(-1)
    else:
        if idx >= labels.shape[1]:
            return None
        values = labels[:, idx]
    return [float(value) for value in values.tolist()]


def _build_eta_pairwise_pair_buffer(dataset, config):
    """Build train-set same-formula high-delta pairs for explicit sampling."""
    if not getattr(config, "eta_pairwise_loss_enabled", False):
        return []
    if not getattr(config, "eta_pairwise_loss_sampler_enabled", True):
        return []
    target_values = _dataset_pairwise_target_values(dataset, config)
    formula_groups = _dataset_pairwise_formula_groups(dataset)
    if target_values is None or formula_groups is None:
        return []
    if len(target_values) != len(formula_groups):
        return []

    min_delta = float(getattr(config, "eta_pairwise_loss_min_delta", 0.5))
    if min_delta < 0:
        raise ValueError("eta_pairwise_loss_min_delta must be >= 0.")
    grouped = {}
    for idx, group_id in enumerate(formula_groups):
        if group_id < 0:
            continue
        grouped.setdefault(group_id, []).append(idx)

    pair_buffer = []
    for indices in grouped.values():
        if len(indices) < 2:
            continue
        for pos, idx_i in enumerate(indices):
            for idx_j in indices[pos + 1 :]:
                delta = abs(target_values[idx_i] - target_values[idx_j])
                if delta >= min_delta:
                    pair_buffer.append((idx_i, idx_j, float(delta)))
    pair_buffer.sort(key=lambda item: item[2], reverse=True)
    max_pairs = int(getattr(config, "eta_pairwise_loss_buffer_max_pairs", 2048))
    if max_pairs > 0 and len(pair_buffer) > max_pairs:
        pair_buffer = pair_buffer[:max_pairs]
    return pair_buffer


def _write_same_formula_pair_buffer_summary(
    pair_buffer, dataset, output_dir, filename
):
    """Write a compact summary of explicit pairwise training pairs."""
    if not pair_buffer:
        return
    ids = list(getattr(dataset, "ids", []))
    examples = []
    for idx_i, idx_j, delta in pair_buffer[:50]:
        examples.append(
            {
                "id_i": ids[idx_i] if idx_i < len(ids) else int(idx_i),
                "id_j": ids[idx_j] if idx_j < len(ids) else int(idx_j),
                "index_i": int(idx_i),
                "index_j": int(idx_j),
                "target_delta_abs": float(delta),
            }
        )
    dumpjson(
        filename=os.path.join(output_dir, filename),
        data={
            "n_pairs": int(len(pair_buffer)),
            "top_pairs": examples,
        },
    )


def _write_eta_pairwise_pair_buffer_summary(pair_buffer, dataset, output_dir):
    """Write a compact summary of explicit output-pairwise training pairs."""
    _write_same_formula_pair_buffer_summary(
        pair_buffer,
        dataset,
        output_dir,
        "eta_pairwise_pair_buffer_summary.json",
    )


def _sample_eta_pairwise_pair_batch(dataset, pair_buffer, config):
    """Collate an explicit pair batch ordered as i,j,i,j,..."""
    if not pair_buffer:
        return None, 0
    n_pairs = int(getattr(config, "eta_pairwise_loss_pairs_per_batch", 2))
    if n_pairs <= 0:
        return None, 0
    n_pairs = min(n_pairs, len(pair_buffer))
    sampled_pairs = random.sample(pair_buffer, n_pairs)
    samples = []
    for idx_i, idx_j, _ in sampled_pairs:
        samples.append(dataset[idx_i])
        samples.append(dataset[idx_j])
    if getattr(dataset, "line_graph", False):
        return dataset.collate_line_graph(samples), n_pairs
    return dataset.collate(samples), n_pairs


def _eta_embedding_contrastive_loss(embedding, target, graph, config, device):
    """Embedding margin loss for same-formula high-delta eta pairs."""
    if not getattr(config, "eta_embedding_contrastive_loss_enabled", False):
        return None, 0
    push_lambda = float(
        getattr(config, "eta_embedding_contrastive_loss_lambda", 0.0)
    )
    pull_lambda = float(
        getattr(config, "eta_embedding_contrastive_pull_lambda", 0.0)
    )
    if push_lambda < 0 or pull_lambda < 0:
        raise ValueError("eta embedding contrastive lambdas must be >= 0.")
    if embedding is None or embedding.numel() == 0:
        zero = torch.zeros((), dtype=target.dtype, device=device)
        return zero, 0
    target_name = str(
        getattr(config, "eta_embedding_contrastive_loss_target", "eta_parallel")
    )
    idx = _target_column_index(config, target_name)
    if idx is None:
        return None, 0
    target = target.to(device)
    if target.ndim == 1:
        if idx != 0:
            return None, 0
        target_values = target.view(-1)
    else:
        if idx >= target.shape[1]:
            return None, 0
        target_values = target[:, idx]
    z = embedding.to(device)
    if z.ndim == 1:
        z = z.view(1, -1)
    if bool(getattr(config, "eta_embedding_contrastive_loss_normalize", True)):
        z = torch.nn.functional.normalize(z, p=2, dim=1)
    groups = _graph_formula_groups(graph, device)
    pair_i, pair_j = _same_formula_high_delta_pairs(
        target_values, groups, config, "eta_embedding_contrastive_loss"
    )
    zero = torch.zeros((), dtype=z.dtype, device=device)
    total_loss = zero
    pair_count = 0
    if pair_i is not None and push_lambda > 0:
        distance = torch.linalg.norm(z[pair_i] - z[pair_j], dim=1)
        margin = float(getattr(config, "eta_embedding_contrastive_loss_margin", 1.0))
        if margin <= 0:
            raise ValueError("eta_embedding_contrastive_loss_margin must be > 0.")
        push_loss = torch.mean(torch.clamp(margin - distance, min=0.0) ** 2)
        total_loss = total_loss + push_lambda * push_loss
        pair_count += int(pair_i.shape[0])

    if (
        bool(getattr(config, "eta_embedding_contrastive_pull_enabled", False))
        and pull_lambda > 0
        and groups is not None
        and groups.shape[0] == target_values.shape[0]
    ):
        n_graphs = target_values.shape[0]
        rows, cols = torch.triu_indices(
            n_graphs, n_graphs, offset=1, device=target_values.device
        )
        same_formula = (
            (groups[rows] >= 0) & (groups[rows] == groups[cols])
        )
        target_delta = torch.abs(target_values[rows] - target_values[cols])
        pull_max_delta = float(
            getattr(config, "eta_embedding_contrastive_pull_max_delta", 0.05)
        )
        mask_flat = same_formula & (target_delta <= pull_max_delta)
        mask = torch.zeros(
            (n_graphs, n_graphs), dtype=torch.bool, device=target_values.device
        )
        mask[rows, cols] = mask_flat
        score = torch.zeros_like(mask, dtype=target_values.dtype)
        score[rows, cols] = -target_delta
        max_pairs = int(
            getattr(config, "eta_embedding_contrastive_pull_max_pairs", 512)
        )
        pull_i, pull_j = _limited_pair_indices(mask, score, max_pairs)
        if pull_i is not None:
            pull_distance = torch.linalg.norm(z[pull_i] - z[pull_j], dim=1)
            total_loss = total_loss + pull_lambda * torch.mean(pull_distance**2)
            pair_count += int(pull_i.shape[0])

    return total_loss, pair_count


def _reshape_graph_outputs(pred, target):
    if pred.ndim == 1 and target.ndim == 2 and target.shape[1] == 1:
        pred = pred.view(-1, 1)
    elif target.ndim == 1 and pred.ndim == 2 and pred.shape[1] == 1:
        target = target.view(-1, 1)
    elif pred.shape != target.shape and pred.numel() == target.numel():
        target = target.reshape_as(pred)
    return pred, target


def _target_column_validation_loss(
    pred, target, graph, config, device, target_column
):
    """Graph validation loss for checkpointing by one output column.

    This intentionally excludes auxiliary target columns while preserving the
    selected target's graph-level weighting and eta-specific penalty terms.
    """
    target_columns = getattr(config, "target_columns", None)
    if not target_columns:
        raise ValueError(
            "best_model_selection='target_column' requires target_columns."
        )
    if target_column not in target_columns:
        raise ValueError(
            f"best_model_target={target_column!r} is not in target_columns: "
            f"{list(target_columns)}"
        )
    pred = pred.to(device)
    target = target.to(device)
    pred, target = _reshape_graph_outputs(pred, target)
    idx = list(target_columns).index(target_column)
    if pred.ndim == 1:
        if idx != 0:
            raise ValueError(
                f"Cannot select target column {target_column!r} from scalar output."
            )
        pred_column = pred
        target_column_values = target
    else:
        if idx >= pred.shape[1]:
            raise ValueError(
                f"best_model_target index {idx} is outside prediction width "
                f"{pred.shape[1]}."
            )
        pred_column = pred[:, idx]
        target_column_values = target[:, idx]
    per_element = _base_regression_element_loss(
        pred_column, target_column_values, config.criterion
    )
    if per_element is None:
        per_element = torch.abs(pred_column - target_column_values)
    per_sample = per_element.reshape(-1)

    penalty_target = str(
        getattr(config, "eta_underestimate_penalty_target", "eta_parallel")
    )
    if str(target_column) == penalty_target:
        pred_for_penalty = pred if pred.ndim > 1 else pred.view(-1, 1)
        target_for_penalty = target if target.ndim > 1 else target.view(-1, 1)
        eta_under_penalty = _eta_underestimate_penalty_loss(
            pred_for_penalty, target_for_penalty, config
        )
        if eta_under_penalty is not None:
            per_sample = per_sample + eta_under_penalty.reshape_as(per_sample)

    if not getattr(config, "use_sample_weights", False):
        return per_sample.mean()

    weights = _graph_sample_weights(graph, device)
    if weights is None:
        return per_sample.mean()
    weights = weights.reshape_as(per_sample)
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-12)


def _best_model_selection_loss(
    pred, target, graph, config, device, fallback_loss=None
):
    """Return scalar validation loss used for checkpoint/early-stop selection."""
    selection = str(getattr(config, "best_model_selection", "total_loss"))
    if selection in {"total_loss", "validation_loss", "val_loss"}:
        if fallback_loss is None:
            return None
        return fallback_loss
    if selection in {"target_column", "component", "target"}:
        target_column = getattr(config, "best_model_target", None)
        if not target_column:
            target_columns = getattr(config, "target_columns", None)
            target_column = (
                target_columns[0] if target_columns else None
            )
        if not target_column:
            raise ValueError(
                "best_model_selection='target_column' requires "
                "best_model_target or target_columns."
            )
        return _target_column_validation_loss(
            pred, target, graph, config, device, str(target_column)
        )
    raise ValueError(
        "Unknown best_model_selection. Use 'total_loss' or 'target_column'."
    )


def _graphwise_regression_loss(pred, target, graph, config, device):
    """Graph-level loss with optional per-sample tail weighting."""
    target = target.to(device)
    pred = pred.to(device)
    pred, target = _reshape_graph_outputs(pred, target)

    per_element = _base_regression_element_loss(pred, target, config.criterion)
    if per_element is None:
        # Keep legacy behavior for less common criteria.
        return nn.L1Loss()(pred, target)

    if per_element.ndim == 1:
        per_sample = per_element
    else:
        per_element = per_element.reshape(per_element.shape[0], -1)
        target_for_components = target.reshape(target.shape[0], -1)
        pred_for_components = pred.reshape(pred.shape[0], -1)
        component_weights = _component_loss_weights(
            config, per_element.shape[1], device
        )
        if component_weights is None:
            weighted_loss_sum = per_element.sum(dim=1)
            loss_weight_sum = float(per_element.shape[1])
        else:
            weighted_loss_sum = (
                per_element * component_weights.view(1, -1)
            ).sum(dim=1)
            loss_weight_sum = component_weights.sum()
        trace_recon = _inplane_trace_reconstruction_loss(
            pred_for_components, target_for_components, config
        )
        if trace_recon is not None:
            trace_weight = _configured_component_weight(
                config, "inplane_trace_recon", default=1.0
            )
            weighted_loss_sum = weighted_loss_sum + trace_recon * trace_weight
            loss_weight_sum = loss_weight_sum + trace_weight
        eta_recon = _inplane_eta_reconstruction_loss(
            pred_for_components, target_for_components, config
        )
        if eta_recon is not None:
            eta_weight = _configured_component_weight(
                config, "inplane_eta_recon", default=1.0
            )
            weighted_loss_sum = weighted_loss_sum + eta_recon * eta_weight
            loss_weight_sum = loss_weight_sum + eta_weight
        if isinstance(loss_weight_sum, torch.Tensor):
            per_sample = weighted_loss_sum / loss_weight_sum.clamp_min(1e-12)
        else:
            per_sample = weighted_loss_sum / max(loss_weight_sum, 1e-12)
        raw_tail = _inplane_raw_tail_loss(
            pred_for_components, target_for_components, config
        )
        if raw_tail is not None:
            per_sample = per_sample + float(
                getattr(config, "raw_tail_loss_lambda", 0.05)
            ) * raw_tail
        eta_under_penalty = _eta_underestimate_penalty_loss(
            pred_for_components, target_for_components, config
        )
        if eta_under_penalty is not None:
            per_sample = per_sample + eta_under_penalty

    if not getattr(config, "use_sample_weights", False):
        return per_sample.mean()

    weights = _graph_sample_weights(graph, device)
    if weights is None:
        return per_sample.mean()
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-12)


def _graphwise_binary_classification_loss(pred, target, graph, device):
    """Graph-level binary classification loss with optional sample weights."""
    target = target.to(device).view(-1)
    pred = pred.to(device)

    if pred.ndim == 2 and pred.shape[-1] == 2:
        per_sample = torch.nn.functional.nll_loss(
            pred, target.long(), reduction="none"
        )
    else:
        prob = pred.reshape(-1).clamp(1e-7, 1.0 - 1e-7)
        per_sample = torch.nn.functional.binary_cross_entropy(
            prob, target.float(), reduction="none"
        )

    weights = _graph_sample_weights(graph, device)
    if weights is None:
        return per_sample.mean()
    weights = weights.reshape_as(per_sample)
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-12)


def _classification_positive_probability(out):
    """Return P(class=1) from supported classification output formats."""
    if out.ndim == 0:
        out = out.view(1)
    if out.ndim == 2 and out.shape[-1] == 2:
        return torch.exp(out)[:, 1]
    return out.reshape(-1).clamp(0.0, 1.0)


def _classification_predictions_from_loader(model, loader, config, device):
    """Collect ids, binary targets, and positive-class probabilities."""
    model.eval()
    ids = list(loader.dataset.ids)
    sample_ids = []
    targets = []
    probabilities = []
    id_offset = 0
    with torch.no_grad():
        for dat in loader:
            out_data = model(_model_input_from_batch(dat, config, device))[
                "out"
            ]
            prob = _classification_positive_probability(out_data).cpu()
            target = dat[-1].view(-1).long().cpu()
            batch_size = target.shape[0]
            batch_ids = ids[id_offset : id_offset + batch_size]
            id_offset += batch_size
            sample_ids.extend(batch_ids)
            targets.extend(target.tolist())
            probabilities.extend(prob.tolist())
    return sample_ids, targets, probabilities


def _classification_metrics_at_threshold(targets, probabilities, threshold):
    """Return binary classification metrics for a probability threshold."""
    targets_arr = np.array(targets, dtype=int)
    probabilities_arr = np.array(probabilities, dtype=float)
    predictions_arr = (probabilities_arr >= float(threshold)).astype(int)
    cm = confusion_matrix(targets_arr, predictions_arr, labels=[0, 1])
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    negative_total = tn + fp
    positive_total = tp + fn
    specificity = None
    false_positive_rate = None
    if negative_total > 0:
        specificity = float(tn / negative_total)
        false_positive_rate = float(fp / negative_total)
    metrics = {
        "n_samples": int(len(targets)),
        "threshold": float(threshold),
        "decision_threshold": float(threshold),
        "positive_label": 1,
        "negative_label": 0,
        "class_counts": {
            "negative": int(np.sum(targets_arr == 0)),
            "positive": int(np.sum(targets_arr == 1)),
        },
        "accuracy": float(accuracy_score(targets_arr, predictions_arr)),
        "precision": float(
            precision_score(
                targets_arr,
                predictions_arr,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                targets_arr,
                predictions_arr,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                targets_arr,
                predictions_arr,
                zero_division=0,
            )
        ),
        "specificity": specificity,
        "negative_retention": specificity,
        "false_positive_rate": false_positive_rate,
        "roc_auc": None,
        "average_precision": None,
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": cm.astype(int).tolist(),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        },
    }
    if positive_total == 0:
        metrics["recall"] = 0.0
    try:
        metrics["roc_auc"] = float(
            roc_auc_score(targets_arr, probabilities_arr)
        )
    except ValueError as exp:
        print("Cannot compute Test ROCAUC:", exp)
    try:
        metrics["average_precision"] = float(
            average_precision_score(targets_arr, probabilities_arr)
        )
    except ValueError as exp:
        print("Cannot compute Test average precision:", exp)
    return metrics


def _classification_threshold_curve(targets, probabilities):
    """Evaluate all distinct probability thresholds plus common sentinels."""
    thresholds = sorted(
        {0.0, 0.5, 1.0}
        | {float(probability) for probability in probabilities}
    )
    return [
        _classification_metrics_at_threshold(targets, probabilities, threshold)
        for threshold in thresholds
    ]


def _select_classification_decision_threshold(
    targets, probabilities, config
):
    """Select a probability threshold from validation predictions."""
    fixed_threshold = getattr(config, "classification_decision_threshold", None)
    threshold_curve = _classification_threshold_curve(targets, probabilities)
    if fixed_threshold is not None:
        threshold = float(fixed_threshold)
        selected_metrics = _classification_metrics_at_threshold(
            targets, probabilities, threshold
        )
        return threshold, {
            "strategy": "fixed",
            "selected_threshold": threshold,
            "selected_metrics": selected_metrics,
            "candidate_count": len(threshold_curve),
            "fallback_used": False,
        }, threshold_curve

    strategy = getattr(
        config,
        "classification_threshold_strategy",
        "max_recall_at_min_specificity",
    )
    supported_strategies = {
        "max_recall_at_min_specificity",
        "max_specificity_at_min_recall",
    }
    if strategy not in supported_strategies:
        raise ValueError(
            "Unsupported classification_threshold_strategy: "
            + str(strategy)
        )

    min_specificity = float(
        getattr(config, "classification_min_specificity", 0.9)
    )
    min_recall = float(getattr(config, "classification_min_recall", 0.9))
    if strategy == "max_recall_at_min_specificity":
        candidates = [
            item
            for item in threshold_curve
            if item["specificity"] is not None
            and item["specificity"] >= min_specificity
        ]
        selection_key = lambda item: (
            item["recall"],
            item["specificity"] or 0.0,
            item["precision"],
            item["f1"],
            item["threshold"],
        )
    else:
        candidates = [
            item
            for item in threshold_curve
            if item["recall"] >= min_recall
        ]
        selection_key = lambda item: (
            item["specificity"] or 0.0,
            item["recall"],
            item["precision"],
            item["f1"],
            item["threshold"],
        )
    fallback_used = False
    if not candidates:
        candidates = [
            item
            for item in threshold_curve
            if item["specificity"] is not None
        ]
        fallback_used = True
    if not candidates:
        threshold = 0.5
        selected_metrics = _classification_metrics_at_threshold(
            targets, probabilities, threshold
        )
        return threshold, {
            "strategy": strategy,
            "min_specificity": min_specificity,
            "min_negative_retention": min_specificity,
            "min_recall": min_recall,
            "min_positive_recall": min_recall,
            "selected_threshold": threshold,
            "selected_metrics": selected_metrics,
            "candidate_count": len(threshold_curve),
            "eligible_candidate_count": 0,
            "fallback_used": True,
        }, threshold_curve
    selected_metrics = max(candidates, key=selection_key)
    threshold = float(selected_metrics["threshold"])
    return threshold, {
        "strategy": strategy,
        "min_specificity": min_specificity,
        "min_negative_retention": min_specificity,
        "min_recall": min_recall,
        "min_positive_recall": min_recall,
        "selected_threshold": threshold,
        "selected_metrics": selected_metrics,
        "candidate_count": len(threshold_curve),
        "eligible_candidate_count": len(candidates),
        "fallback_used": fallback_used,
    }, threshold_curve


def _model_input_from_batch(dats, config, device):
    """Build model input while tolerating legacy 3-item line-graph batches."""
    if (config.compute_line_graph) > 0:
        if len(dats) == 4:
            return [
                dats[0].to(device),
                dats[1].to(device),
                dats[2].to(device),
            ]
        if len(dats) == 3 and isinstance(dats[1], dgl.DGLGraph):
            return [
                dats[0].to(device),
                dats[1].to(device),
                dats[-1].to(device),
            ]
    return [dats[0].to(device), dats[1].to(device)]


def _extra_feature_fusion_gamma_snapshot(model, epoch):
    """Return current bounded graph descriptor fusion scale, when available."""
    module = model.module if hasattr(model, "module") else model
    if not hasattr(module, "extra_feature_fusion_gamma_value"):
        return None
    if not hasattr(module, "extra_feature_gated_fusion"):
        return None
    fusion = module.extra_feature_gated_fusion
    gamma = module.extra_feature_fusion_gamma_value().detach().cpu().item()
    gamma_raw = fusion.gamma_raw.detach().cpu().item()
    gamma_max = float(getattr(fusion, "gamma_max", 1.0))
    return {
        "epoch": int(epoch),
        "gamma": float(gamma),
        "gamma_raw": float(gamma_raw),
        "gamma_max": gamma_max,
    }


def _site_symmetry_gamma_snapshot(model, epoch):
    """Return current bounded site-symmetry feature scale, when available."""
    module = model.module if hasattr(model, "module") else model
    if not hasattr(module, "site_symmetry_gamma_value"):
        return None
    if not hasattr(module, "site_symmetry_gamma_raw"):
        return None
    gamma = module.site_symmetry_gamma_value().detach().cpu().item()
    gamma_raw = module.site_symmetry_gamma_raw.detach().cpu().item()
    gamma_max = float(getattr(module, "site_symmetry_gamma_max", 1.0))
    return {
        "epoch": int(epoch),
        "gamma": float(gamma),
        "gamma_raw": float(gamma_raw),
        "gamma_max": gamma_max,
    }


def _inplane_direction_attention_gamma_snapshot(model, epoch):
    """Return current bounded in-plane direction attention scale."""
    module = model.module if hasattr(model, "module") else model
    if not hasattr(module, "inplane_direction_attention_gamma_value"):
        return None
    if not hasattr(module, "inplane_direction_attention"):
        return None
    attention = module.inplane_direction_attention
    gamma = module.inplane_direction_attention_gamma_value().detach().cpu().item()
    gamma_raw = attention.gamma_raw.detach().cpu().item()
    gamma_max = float(getattr(attention, "gamma_max", 1.0))
    return {
        "epoch": int(epoch),
        "gamma": float(gamma),
        "gamma_raw": float(gamma_raw),
        "gamma_max": gamma_max,
    }


def train_dgl(
    config: Union[TrainingConfig, Dict[str, Any]],
    model: nn.Module = None,
    # checkpoint_dir: Path = Path("./"),
    train_val_test_loaders=[],
    rank=0,
    world_size=0,
    # log_tensorboard: bool = False,
):
    """Training entry point for DGL networks.

    `config` should conform to alignn.conf.TrainingConfig, and
    if passed as a dict with matching keys, pydantic validation is used
    """
    # print("rank", rank)
    # setup(rank, world_size)
    if rank == 0:
        # print(config)
        if type(config) is dict:
            try:
                print("Trying to convert dictionary.")
                config = TrainingConfig(**config)
            except Exception as exp:
                print("Check", exp)
    print("config:", config.dict())

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)
    # checkpoint_dir = os.path.join(config.output_dir)
    # deterministic = False
    classification = False
    tmp = config.dict()
    f = open(os.path.join(config.output_dir, "config.json"), "w")
    f.write(json.dumps(tmp, indent=4))
    f.close()
    global tmp_output_dir
    tmp_output_dir = config.output_dir
    pprint.pprint(tmp)  # , sort_dicts=False)
    if config.classification_threshold is not None:
        classification = True
    TORCH_DTYPES = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat": torch.bfloat16,
    }
    torch.set_default_dtype(TORCH_DTYPES[config.dtype])
    line_graph = False
    if config.compute_line_graph > 0:
        # if config.model.alignn_layers > 0:
        line_graph = True
    if world_size > 1:
        use_ddp = True
    else:
        use_ddp = False
        device = "cpu"
        if torch.cuda.is_available():
            device = torch.device("cuda")
    if not train_val_test_loaders:
        # use input standardization for all real-valued feature sets
        # print("config.neighbor_strategy",config.neighbor_strategy)
        # import sys
        # sys.exit()
        (
            train_loader,
            val_loader,
            test_loader,
            prepare_batch,
        ) = get_train_val_loaders(
            dataset=config.dataset,
            target=config.target,
            n_train=config.n_train,
            n_val=config.n_val,
            n_test=config.n_test,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            batch_size=config.batch_size,
            atom_features=config.atom_features,
            neighbor_strategy=config.neighbor_strategy,
            standardize=config.atom_features != "cgcnn",
            line_graph=line_graph,
            id_tag=config.id_tag,
            pin_memory=config.pin_memory,
            workers=config.num_workers,
            save_dataloader=config.save_dataloader,
            use_canonize=config.use_canonize,
            filename=config.filename,
            cutoff=config.cutoff,
            max_neighbors=config.max_neighbors,
            output_features=config.model.output_features,
            classification_threshold=config.classification_threshold,
            target_multiplication_factor=config.target_multiplication_factor,
            standard_scalar_and_pca=config.standard_scalar_and_pca,
            keep_data_order=config.keep_data_order,
            output_dir=config.output_dir,
            use_lmdb=config.use_lmdb,
            dtype=config.dtype,
        )
    else:
        train_loader = train_val_test_loaders[0]
        val_loader = train_val_test_loaders[1]
        test_loader = train_val_test_loaders[2]
        prepare_batch = train_val_test_loaders[3]
    # rank=0
    if use_ddp:
        device = torch.device(f"cuda:{rank}")
    prepare_batch = partial(prepare_batch, device=device)
    if classification:
        config.model.classification = True
    _model = {
        "alignn_atomwise": ALIGNNAtomWise,
        "ealignn_atomwise": eALIGNNAtomWise,
        "alignn": ALIGNN,
    }
    if config.random_seed is not None:
        random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)
        try:
            import torch_xla.core.xla_model as xm

            xm.set_rng_state(config.random_seed)
        except ImportError:
            pass
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(config.random_seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(":4096:8")
        torch.use_deterministic_algorithms(True)
    if model is None:
        net = _model.get(config.model.name)(config.model)
    else:
        net = model
    print(figlet_alignn)
    print("Model parameters", sum(p.numel() for p in net.parameters()))
    print("CUDA available", torch.cuda.is_available())
    print("CUDA device count", int(torch.cuda.device_count()))
    try:
        gpu_stats = torch.cuda.get_device_properties(0)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        from platform import system as platform_system

        platform_system = platform_system()
        statistics = (
            f"   GPU: {gpu_stats.name}. Max memory: {max_memory} GB"
            + f". Platform = {platform_system}.\n"
            f"   Pytorch: {torch.__version__}. CUDA = "
            + f"{gpu_stats.major}.{gpu_stats.minor}."
            + f" CUDA Toolkit = {torch.version.cuda}.\n"
        )
        print(statistics)
    except Exception:
        pass
    # print("device", device)
    net.to(device)
    if use_ddp:
        net = DDP(net, device_ids=[rank], find_unused_parameters=True)
    # group parameters to skip weight decay for bias and batchnorm
    params = group_decay(net)
    optimizer = setup_optimizer(params, config)
    if config.scheduler == "none":
        # always return multiplier of 1 (i.e. do nothing)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: 1.0
        )

    elif config.scheduler == "onecycle":
        steps_per_epoch = len(train_loader)
        # pct_start = config.warmup_steps / (config.epochs * steps_per_epoch)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
            # pct_start=pct_start,
            pct_start=0.3,
        )
    elif config.scheduler == "step":
        # pct_start = config.warmup_steps / (config.epochs * steps_per_epoch)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
        )

    # if (
    #    config.model.name == "alignn_atomwise"
    #    or config.model.name == "alignn_ff2"
    # ):
    if "alignn_" in config.model.name:
        best_loss = np.inf
        epochs_without_improvement = 0
        early_stop_epoch = None
        criterion = nn.L1Loss()
        if classification:
            criterion = nn.NLLLoss()
        params = group_decay(net)
        optimizer = setup_optimizer(params, config)
        # optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        history_train = []
        history_val = []
        extra_feature_fusion_gamma_history = []
        site_symmetry_gamma_history = []
        inplane_direction_attention_gamma_history = []
        eta_pairwise_loss_history = []
        best_model_selection = str(
            getattr(config, "best_model_selection", "total_loss")
        )
        target_column_selection = best_model_selection in {
            "target_column",
            "component",
            "target",
        }
        best_model_target = getattr(config, "best_model_target", None)
        if target_column_selection and not best_model_target:
            target_columns = getattr(config, "target_columns", None)
            best_model_target = target_columns[0] if target_columns else None
        eta_pairwise_pair_buffer = _build_eta_pairwise_pair_buffer(
            train_loader.dataset, config
        )
        if eta_pairwise_pair_buffer:
            _write_eta_pairwise_pair_buffer_summary(
                eta_pairwise_pair_buffer,
                train_loader.dataset,
                config.output_dir,
            )
            print(
                "Eta pairwise explicit pair buffer:",
                len(eta_pairwise_pair_buffer),
                "pairs",
            )
        elif (
            getattr(config, "eta_pairwise_loss_enabled", False)
            and getattr(config, "eta_pairwise_loss_sampler_enabled", True)
        ):
            print(
                "Eta pairwise explicit pair buffer: 0 pairs "
                "(check formula groups and min_delta)."
            )
        for e in range(config.epochs):
            # optimizer.zero_grad()
            train_init_time = time.time()
            running_loss = 0
            running_loss1 = 0
            running_loss2 = 0
            running_loss3 = 0
            running_loss4 = 0
            running_loss5 = 0
            running_eta_pairwise_loss = 0.0
            running_eta_pairwise_pairs = 0
            running_eta_embedding_contrastive_loss = 0.0
            running_eta_embedding_contrastive_pairs = 0
            train_result = []
            for dats, jid in zip(train_loader, train_loader.dataset.ids):
                info = {}
                # info["id"] = jid
                optimizer.zero_grad()
                result = net(_model_input_from_batch(dats, config, device))
                # info = {}
                info["target_out"] = []
                info["pred_out"] = []
                info["target_atomwise_pred"] = []
                info["pred_atomwise_pred"] = []
                info["target_grad"] = []
                info["pred_grad"] = []
                info["target_stress"] = []
                info["pred_stress"] = []
                info["target_additional"] = []
                info["pred_additional"] = []

                loss1 = 0  # Such as energy
                loss2 = 0  # Such as bader charges
                loss3 = 0  # Such as forces
                loss4 = 0  # Such as stresses
                loss5 = 0  # Such as dos
                if config.model.output_features is not None:
                    if classification:
                        loss1 = (
                            config.model.graphwise_weight
                            * _graphwise_binary_classification_loss(
                                result["out"],
                                dats[-1],
                                dats[0],
                                device,
                            )
                        )
                    else:
                        loss1 = (
                            config.model.graphwise_weight
                            * _graphwise_regression_loss(
                                result["out"],
                                dats[-1].to(device),
                                dats[0],
                                config,
                                device,
                                # result["out"], dats[2].to(device)
                            )
                        )
                    info["target_out"] = dats[-1].cpu().numpy().tolist()
                    # info["target_out"] = dats[2].cpu().numpy().tolist()
                    info["pred_out"] = (
                        result["out"].cpu().detach().numpy().tolist()
                    )
                    running_loss1 += loss1.item()
                if (
                    config.model.atomwise_output_features > 0
                    # config.model.atomwise_output_features is not None
                    and config.model.atomwise_weight != 0
                ):
                    loss2 = config.model.atomwise_weight * criterion(
                        result["atomwise_pred"].to(device),
                        dats[0].ndata["atomwise_target"].to(device),
                    )
                    info["target_atomwise_pred"] = (
                        dats[0].ndata["atomwise_target"].cpu().numpy().tolist()
                    )
                    info["pred_atomwise_pred"] = (
                        result["atomwise_pred"].cpu().detach().numpy().tolist()
                    )
                    running_loss2 += loss2.item()

                if config.model.calculate_gradient:
                    loss3 = config.model.gradwise_weight * criterion(
                        result["grad"].to(device),
                        dats[0].ndata["atomwise_grad"].to(device),
                    )
                    info["target_grad"] = (
                        dats[0].ndata["atomwise_grad"].cpu().numpy().tolist()
                    )
                    info["pred_grad"] = (
                        result["grad"].cpu().detach().numpy().tolist()
                    )
                    running_loss3 += loss3.item()
                if config.model.stresswise_weight != 0:
                    # print('unbatch',dgl.unbatch(dats[0]))

                    targ_stress = torch.stack(
                        [
                            gg.ndata["stresses"][0]
                            for gg in dgl.unbatch(dats[0])
                        ]
                    ).to(device)
                    pred_stress = result["stresses"]
                    # print('targ_stress',targ_stress,targ_stress.shape)
                    # print('pred_stress',pred_stress,pred_stress.shape)
                    loss4 = config.model.stresswise_weight * criterion(
                        pred_stress.to(device),
                        targ_stress.to(device),
                    )
                    info["target_stress"] = (
                        targ_stress.cpu()
                        .numpy()
                        .tolist()
                        # dats[0].ndata["stresses"][0].cpu().numpy().tolist()
                    )
                    info["pred_stress"] = (
                        result["stresses"].cpu().detach().numpy().tolist()
                    )
                    running_loss4 += loss4.item()
                if config.model.additional_output_weight != 0:
                    # print('unbatch',dgl.unbatch(dats[0]))
                    additional_dat = [
                        gg.ndata["additional"][0]
                        for gg in dgl.unbatch(dats[0])
                    ]
                    # print('additional_dat',additional_dat,len(additional_dat))
                    targ = torch.stack(additional_dat).to(device)
                    # targ=torch.tensor(additional_dat).to( dats[0].device)
                    # print('result["additional"]',result["additional"],result["additional"].shape)
                    # print('targ',targ,targ.shape)
                    # print('targ device',targ.device)
                    loss5 = config.model.additional_output_weight * criterion(
                        (result["additional"]).to(device),
                        targ,
                        # (dats[0].ndata["additional"]).to(device),
                    )
                    info["target_additional"] = targ.cpu().numpy().tolist()
                    info["pred_additional"] = (
                        result["additional"].cpu().detach().numpy().tolist()
                    )
                    running_loss5 += loss5.item()
                    # print("target_stress", info["target_stress"][0])
                    # print("pred_stress", info["pred_stress"][0])
                train_result.append(info)
                eta_pairwise_loss = torch.zeros((), device=device)
                eta_pairwise_pairs = 0
                if getattr(config, "eta_pairwise_loss_batch_enabled", False):
                    (
                        batch_eta_pairwise_loss,
                        batch_eta_pairwise_pairs,
                    ) = _eta_pairwise_loss(
                        result["out"],
                        dats[-1].to(device),
                        dats[0],
                        config,
                        device,
                    )
                    if batch_eta_pairwise_loss is not None:
                        eta_pairwise_loss = (
                            eta_pairwise_loss + batch_eta_pairwise_loss
                        )
                        eta_pairwise_pairs += int(batch_eta_pairwise_pairs)
                (
                    explicit_pair_batch,
                    explicit_pair_count,
                ) = _sample_eta_pairwise_pair_batch(
                    train_loader.dataset,
                    eta_pairwise_pair_buffer,
                    config,
                )
                if explicit_pair_batch is not None:
                    explicit_pair_result = net(
                        _model_input_from_batch(
                            explicit_pair_batch, config, device
                        )
                    )
                    (
                        explicit_eta_pairwise_loss,
                        explicit_eta_pairwise_pairs,
                    ) = _eta_pairwise_explicit_loss(
                        explicit_pair_result["out"],
                        explicit_pair_batch[-1].to(device),
                        explicit_pair_count,
                        config,
                        device,
                    )
                    if explicit_eta_pairwise_loss is not None:
                        eta_pairwise_loss = (
                            eta_pairwise_loss + explicit_eta_pairwise_loss
                        )
                        eta_pairwise_pairs += int(explicit_eta_pairwise_pairs)
                if eta_pairwise_pairs:
                    running_eta_pairwise_loss += eta_pairwise_loss.item()
                    running_eta_pairwise_pairs += int(eta_pairwise_pairs)
                eta_embedding_contrastive_loss = torch.zeros(
                    (), device=device
                )
                eta_embedding_contrastive_pairs = 0
                (
                    eta_embedding_contrastive_loss_candidate,
                    eta_embedding_contrastive_pairs,
                ) = _eta_embedding_contrastive_loss(
                    result.get("graph_embedding", None),
                    dats[-1].to(device),
                    dats[0],
                    config,
                    device,
                )
                if eta_embedding_contrastive_loss_candidate is not None:
                    eta_embedding_contrastive_loss = (
                        eta_embedding_contrastive_loss
                        + eta_embedding_contrastive_loss_candidate
                    )
                if eta_embedding_contrastive_pairs:
                    running_eta_embedding_contrastive_loss += (
                        eta_embedding_contrastive_loss.item()
                    )
                    running_eta_embedding_contrastive_pairs += int(
                        eta_embedding_contrastive_pairs
                    )
                loss = (
                    loss1
                    + loss2
                    + loss3
                    + loss4
                    + loss5
                    + eta_pairwise_loss
                    + eta_embedding_contrastive_loss
                )
                loss.backward()
                optimizer.step()
                # optimizer.zero_grad() #never
                running_loss += loss.item()
            # mean_out, mean_atom, mean_grad, mean_stress = get_batch_errors(
            #    train_result
            # )
            # dumpjson(filename="Train_results.json", data=train_result)
            scheduler.step()
            train_final_time = time.time()
            train_ep_time = train_final_time - train_init_time
            # if rank == 0: # or world_size == 1:
            history_train.append(
                [
                    running_loss,
                    running_loss1,
                    running_loss2,
                    running_loss3,
                    running_loss4,
                    running_loss5,
                ]
            )
            dumpjson(
                filename=os.path.join(config.output_dir, "history_train.json"),
                data=history_train,
            )
            if (
                getattr(config, "eta_pairwise_loss_enabled", False)
                or getattr(
                    config, "eta_embedding_contrastive_loss_enabled", False
                )
            ):
                eta_pairwise_loss_history.append(
                    {
                        "epoch": int(e),
                        "eta_pairwise_loss": float(
                            running_eta_pairwise_loss
                        ),
                        "eta_pairwise_pairs": int(
                            running_eta_pairwise_pairs
                        ),
                        "eta_embedding_contrastive_loss": float(
                            running_eta_embedding_contrastive_loss
                        ),
                        "eta_embedding_contrastive_pairs": int(
                            running_eta_embedding_contrastive_pairs
                        ),
                    }
                )
                dumpjson(
                    filename=os.path.join(
                        config.output_dir,
                        "eta_pairwise_loss_history.json",
                    ),
                    data=eta_pairwise_loss_history,
                )
            extra_feature_fusion_gamma = _extra_feature_fusion_gamma_snapshot(
                net, e
            )
            if extra_feature_fusion_gamma is not None:
                extra_feature_fusion_gamma_history.append(
                    extra_feature_fusion_gamma
                )
                dumpjson(
                    filename=os.path.join(
                        config.output_dir,
                        "extra_feature_fusion_gamma_history.json",
                    ),
                    data=extra_feature_fusion_gamma_history,
                )
            site_symmetry_gamma = _site_symmetry_gamma_snapshot(net, e)
            if site_symmetry_gamma is not None:
                site_symmetry_gamma_history.append(site_symmetry_gamma)
                dumpjson(
                    filename=os.path.join(
                        config.output_dir,
                        "site_symmetry_gamma_history.json",
                    ),
                    data=site_symmetry_gamma_history,
                )
            inplane_direction_attention_gamma = (
                _inplane_direction_attention_gamma_snapshot(net, e)
            )
            if inplane_direction_attention_gamma is not None:
                inplane_direction_attention_gamma_history.append(
                    inplane_direction_attention_gamma
                )
                dumpjson(
                    filename=os.path.join(
                        config.output_dir,
                        "inplane_direction_attention_gamma_history.json",
                    ),
                    data=inplane_direction_attention_gamma_history,
                )
            val_loss = 0
            val_loss1 = 0
            val_loss2 = 0
            val_loss3 = 0
            val_loss4 = 0
            val_loss5 = 0
            val_selection_loss = 0
            val_result = []
            # for dats in val_loader:
            val_init_time = time.time()
            for dats, jid in zip(val_loader, val_loader.dataset.ids):
                info = {}
                info["id"] = jid
                optimizer.zero_grad()
                # result = net([dats[0].to(device), dats[1].to(device)])
                # if (config.model.alignn_layers) > 0:
                # if (config.create_line_graph) > 0:
                result = net(_model_input_from_batch(dats, config, device))
                    # result = net(dats[0].to(device))
                # info = {}
                info["target_out"] = []
                info["pred_out"] = []
                info["target_atomwise_pred"] = []
                info["pred_atomwise_pred"] = []
                info["target_grad"] = []
                info["pred_grad"] = []
                info["target_stress"] = []
                info["pred_stress"] = []
                loss1 = 0  # Such as energy
                loss2 = 0  # Such as bader charges
                loss3 = 0  # Such as forces
                loss4 = 0  # Such as stresses
                loss5 = 0  # Such as stresses
                if config.model.output_features is not None:
                    if classification:
                        loss1 = (
                            config.model.graphwise_weight
                            * _graphwise_binary_classification_loss(
                                result["out"],
                                dats[-1],
                                dats[0],
                                device,
                            )
                        )
                    else:
                        loss1 = (
                            config.model.graphwise_weight
                            * _graphwise_regression_loss(
                                result["out"],
                                dats[-1].to(device),
                                dats[0],
                                config,
                                device,
                            )
                        )
                        if target_column_selection:
                            val_selection_loss += _best_model_selection_loss(
                                result["out"],
                                dats[-1].to(device),
                                dats[0],
                                config,
                                device,
                            ).item()
                    info["target_out"] = dats[-1].cpu().numpy().tolist()
                    info["pred_out"] = (
                        result["out"].cpu().detach().numpy().tolist()
                    )
                    val_loss1 += loss1.item()

                if (
                    config.model.atomwise_output_features > 0
                    and config.model.atomwise_weight != 0
                ):
                    loss2 = config.model.atomwise_weight * criterion(
                        result["atomwise_pred"].to(device),
                        dats[0].ndata["atomwise_target"].to(device),
                    )
                    info["target_atomwise_pred"] = (
                        dats[0].ndata["atomwise_target"].cpu().numpy().tolist()
                    )
                    info["pred_atomwise_pred"] = (
                        result["atomwise_pred"].cpu().detach().numpy().tolist()
                    )
                    val_loss2 += loss2.item()
                if config.model.calculate_gradient:
                    loss3 = config.model.gradwise_weight * criterion(
                        result["grad"].to(device),
                        dats[0].ndata["atomwise_grad"].to(device),
                    )
                    info["target_grad"] = (
                        dats[0].ndata["atomwise_grad"].cpu().numpy().tolist()
                    )
                    info["pred_grad"] = (
                        result["grad"].cpu().detach().numpy().tolist()
                    )
                    val_loss3 += loss3.item()
                if config.model.stresswise_weight != 0:
                    # loss4 = config.model.stresswise_weight * criterion(
                    #    result["stress"].to(device),
                    #    dats[0].ndata["stresses"][0].to(device),
                    # )

                    targ_stress = torch.stack(
                        [
                            gg.ndata["stresses"][0]
                            for gg in dgl.unbatch(dats[0])
                        ]
                    ).to(device)
                    pred_stress = result["stresses"]
                    # print('targ_stress',targ_stress,targ_stress.shape)
                    # print('pred_stress',pred_stress,pred_stress.shape)
                    loss4 = config.model.stresswise_weight * criterion(
                        pred_stress.to(device),
                        targ_stress.to(device),
                    )
                    info["target_stress"] = (
                        targ_stress.cpu()
                        .numpy()
                        .tolist()
                        # dats[0].ndata["stresses"][0].cpu().numpy().tolist()
                    )
                    info["pred_stress"] = (
                        result["stresses"].cpu().detach().numpy().tolist()
                    )

                    val_loss4 += loss4.item()
                if config.model.additional_output_weight != 0:
                    additional_dat = [
                        gg.ndata["additional"][0]
                        for gg in dgl.unbatch(dats[0])
                    ]
                    # print('additional_dat',additional_dat,len(additional_dat))
                    targ = torch.stack(additional_dat).to(device)
                    # targ=torch.tensor(additional_dat).to( dats[0].device)
                    # print('result["additional"]',result["additional"],result["additional"].shape)
                    # print('targ',targ,targ.shape)
                    # print('targ device',targ.device)
                    loss5 = config.model.additional_output_weight * criterion(
                        (result["additional"]).to(device),
                        targ,
                        # (dats[0].ndata["additional"]).to(device),
                    )
                    info["target_additional"] = targ.cpu().numpy().tolist()
                    info["pred_additional"] = (
                        result["additional"].cpu().detach().numpy().tolist()
                    )

                    val_loss5 += loss5.item()
                loss = loss1 + loss2 + loss3 + loss4 + loss5
                val_result.append(info)
                val_loss += loss.item()
            # mean_out, mean_atom, mean_grad, mean_stress = get_batch_errors(
            #    val_result
            # )
            val_fin_time = time.time()
            val_ep_time = val_fin_time - val_init_time
            current_model_name = "current_model.pt"
            torch.save(
                net.state_dict(),
                os.path.join(config.output_dir, current_model_name),
            )
            checkpoint_loss = (
                val_selection_loss if target_column_selection else val_loss
            )
            saving_msg = ""
            if checkpoint_loss < best_loss:
                best_loss = checkpoint_loss
                epochs_without_improvement = 0
                best_model_name = "best_model.pt"
                torch.save(
                    net.state_dict(),
                    os.path.join(config.output_dir, best_model_name),
                )
                # print("Saving data for epoch:", e)
                saving_msg = "Saving model"
                dumpjson(
                    filename=os.path.join(
                        config.output_dir, "Train_results.json"
                    ),
                    data=train_result,
                )
                dumpjson(
                    filename=os.path.join(
                        config.output_dir, "Val_results.json"
                    ),
                    data=val_result,
                )
                best_model = net
            else:
                epochs_without_improvement += 1
            history_val.append(
                [
                    val_loss,
                    val_loss1,
                    val_loss2,
                    val_loss3,
                    val_loss4,
                    val_loss5,
                ]
            )
            # history_val.append([mean_out, mean_atom, mean_grad, mean_stress])
            dumpjson(
                filename=os.path.join(config.output_dir, "history_val.json"),
                data=history_val,
            )
            if rank == 0:
                print_train_val_loss(
                    e,
                    running_loss,
                    running_loss1,
                    running_loss2,
                    running_loss3,
                    running_loss4,
                    running_loss5,
                    val_loss,
                    val_loss1,
                    val_loss2,
                    val_loss3,
                    val_loss4,
                    val_loss5,
                    train_ep_time,
                    val_ep_time,
                    saving_msg=saving_msg,
                )
            if (
                config.n_early_stopping is not None
                and epochs_without_improvement >= config.n_early_stopping
            ):
                early_stop_epoch = e
                print(
                    "Early stopping at epoch",
                    e,
                    "after",
                    epochs_without_improvement,
                    "epochs without validation improvement.",
                )
                break

        dumpjson(
            filename=os.path.join(config.output_dir, "early_stopping.json"),
            data={
                "enabled": config.n_early_stopping is not None,
                "patience": config.n_early_stopping,
                "stopped": early_stop_epoch is not None,
                "stopped_epoch": early_stop_epoch,
                "best_loss": best_loss,
                "best_model_selection": best_model_selection,
                "best_model_target": best_model_target,
                "best_model_selection_excludes_auxiliary_targets": (
                    target_column_selection
                ),
                "best_model_selection_uses_sample_weights": bool(
                    target_column_selection
                    and getattr(config, "use_sample_weights", False)
                ),
                "best_model_selection_uses_eta_underestimate_penalty": bool(
                    target_column_selection
                    and getattr(
                        config,
                        "eta_underestimate_penalty_enabled",
                        False,
                    )
                    and str(best_model_target)
                    == str(
                        getattr(
                            config,
                            "eta_underestimate_penalty_target",
                            "eta_parallel",
                        )
                    )
                ),
                "epochs_without_improvement": epochs_without_improvement,
                "epochs_completed": len(history_val),
            },
        )

        if rank == 0 or world_size == 1:
            test_loss = 0
            test_result = []
            for dats, jid in zip(test_loader, test_loader.dataset.ids):
                # for dats in test_loader:
                info = {}
                info["id"] = jid
                optimizer.zero_grad()
                # if (config.model.alignn_layers) > 0:
                # if (config.create_line_graph) > 0:
                result = net(_model_input_from_batch(dats, config, device))
                    # result = net(dats[0].to(device))
                loss1 = 0  # Such as energy
                loss2 = 0  # Such as bader charges
                loss3 = 0  # Such as forces
                loss4 = 0  # Such as stresses
                if (
                    config.model.output_features is not None
                    and not classification
                ):
                    # print('result["out"]',result["out"])
                    # print('dats[2]',dats[2])
                    loss1 = config.model.graphwise_weight * _graphwise_regression_loss(
                        result["out"],
                        dats[-1].to(device),
                        dats[0],
                        config,
                        device,
                    )
                    info["target_out"] = dats[-1].cpu().numpy().tolist()
                    info["pred_out"] = (
                        result["out"].cpu().detach().numpy().tolist()
                    )

                if config.model.atomwise_output_features > 0:
                    loss2 = config.model.atomwise_weight * criterion(
                        result["atomwise_pred"].to(device),
                        dats[0].ndata["atomwise_target"].to(device),
                    )
                    info["target_atomwise_pred"] = (
                        dats[0].ndata["atomwise_target"].cpu().numpy().tolist()
                    )
                    info["pred_atomwise_pred"] = (
                        result["atomwise_pred"].cpu().detach().numpy().tolist()
                    )

                if config.model.calculate_gradient:
                    loss3 = config.model.gradwise_weight * criterion(
                        result["grad"].to(device),
                        dats[0].ndata["atomwise_grad"].to(device),
                    )
                    info["target_grad"] = (
                        dats[0].ndata["atomwise_grad"].cpu().numpy().tolist()
                    )
                    info["pred_grad"] = (
                        result["grad"].cpu().detach().numpy().tolist()
                    )
                if config.model.stresswise_weight != 0:

                    targ_stress = torch.stack(
                        [
                            gg.ndata["stresses"][0]
                            for gg in dgl.unbatch(dats[0])
                        ]
                    ).to(device)
                    pred_stress = result["stresses"]
                    # print('targ_stress',targ_stress,targ_stress.shape)
                    # print('pred_stress',pred_stress,pred_stress.shape)
                    loss4 = config.model.stresswise_weight * criterion(
                        pred_stress.to(device),
                        targ_stress.to(device),
                    )
                    info["target_stress"] = (
                        targ_stress.cpu()
                        .numpy()
                        .tolist()
                        # dats[0].ndata["stresses"][0].cpu().numpy().tolist()
                    )
                    info["pred_stress"] = (
                        result["stresses"].cpu().detach().numpy().tolist()
                    )

                test_result.append(info)
                loss = loss1 + loss2 + loss3 + loss4
                if not classification:
                    test_loss += loss.item()
            print("TestLoss", e, test_loss)
            dumpjson(
                filename=os.path.join(config.output_dir, "Test_results.json"),
                data=test_result,
            )
            last_model_name = "last_model.pt"
            torch.save(
                net.state_dict(),
                os.path.join(config.output_dir, last_model_name),
            )
            # return test_result
    if rank == 0 or world_size == 1:
        if config.write_predictions and classification:
            _, val_targets, val_probabilities = (
                _classification_predictions_from_loader(
                    best_model, val_loader, config, device
                )
            )
            (
                decision_threshold,
                threshold_selection,
                threshold_curve,
            ) = _select_classification_decision_threshold(
                val_targets, val_probabilities, config
            )
            dumpjson(
                filename=os.path.join(
                    config.output_dir,
                    "classification_threshold_selection.json",
                ),
                data=threshold_selection,
            )
            with open(
                os.path.join(
                    config.output_dir,
                    "classification_threshold_curve_val.csv",
                ),
                "w",
                newline="",
            ) as curve_file:
                fieldnames = [
                    "threshold",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "specificity",
                    "negative_retention",
                    "false_positive_rate",
                    "tn",
                    "fp",
                    "fn",
                    "tp",
                ]
                writer = csv.DictWriter(curve_file, fieldnames=fieldnames)
                writer.writeheader()
                for item in threshold_curve:
                    cm_item = item["confusion_matrix"]
                    writer.writerow(
                        {
                            "threshold": item["threshold"],
                            "accuracy": item["accuracy"],
                            "precision": item["precision"],
                            "recall": item["recall"],
                            "f1": item["f1"],
                            "specificity": item["specificity"],
                            "negative_retention": item[
                                "negative_retention"
                            ],
                            "false_positive_rate": item[
                                "false_positive_rate"
                            ],
                            "tn": cm_item["tn"],
                            "fp": cm_item["fp"],
                            "fn": cm_item["fn"],
                            "tp": cm_item["tp"],
                        }
                    )

            ids, targets, probabilities = (
                _classification_predictions_from_loader(
                    best_model, test_loader, config, device
                )
            )
            predictions = [
                int(probability >= decision_threshold)
                for probability in probabilities
            ]
            with open(
                os.path.join(
                    config.output_dir, "prediction_results_test_set.csv"
                ),
                "w",
                newline="",
            ) as pred_file:
                writer = csv.writer(pred_file)
                writer.writerow(
                    [
                        "id",
                        "target",
                        "probability",
                        "prediction",
                        "decision_threshold",
                    ]
                )
                for sample_id, target_i, prob_i, pred_i in zip(
                    ids, targets, probabilities, predictions
                ):
                    writer.writerow(
                        [
                            sample_id,
                            int(target_i),
                            f"{float(prob_i):.8f}",
                            int(pred_i),
                            f"{float(decision_threshold):.8f}",
                        ]
                    )

            metrics = _classification_metrics_at_threshold(
                targets, probabilities, decision_threshold
            )
            metrics["threshold_selection"] = threshold_selection
            dumpjson(
                filename=os.path.join(
                    config.output_dir, "classification_metrics.json"
                ),
                data=metrics,
            )
            with open(
                os.path.join(
                    config.output_dir, "classification_metrics.csv"
                ),
                "w",
            ) as metrics_file:
                metrics_file.write("metric,value\n")
                for key in [
                    "n_samples",
                    "threshold",
                    "decision_threshold",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "specificity",
                    "negative_retention",
                    "false_positive_rate",
                    "roc_auc",
                    "average_precision",
                ]:
                    metrics_file.write(f"{key},{metrics[key]}\n")
                cm = metrics["confusion_matrix"]
                metrics_file.write(f"tn,{cm['tn']}\n")
                metrics_file.write(f"fp,{cm['fp']}\n")
                metrics_file.write(f"fn,{cm['fn']}\n")
                metrics_file.write(f"tp,{cm['tp']}\n")
            print(
                "Selected classification decision threshold:",
                decision_threshold,
            )
            print("Test metrics:", metrics)

        if (
            config.write_predictions
            and not classification
            and config.model.output_features > 1
        ):
            best_model.eval()
            # net.eval()
            mem = []
            with torch.no_grad():
                ids = list(test_loader.dataset.ids)
                id_offset = 0
                for dat in test_loader:
                    if len(dat) == 4:
                        g, lg, lat, target = dat
                        model_input = [
                            g.to(device),
                            lg.to(device),
                            lat.to(device),
                        ]
                    elif len(dat) == 3:
                        if isinstance(dat[1], dgl.DGLGraph):
                            g, lg, target = dat
                            # Backward compatibility for older
                            # collate_line_graph batches that omitted lattice.
                            model_input = [
                                g.to(device),
                                lg.to(device),
                                target.to(device),
                            ]
                        else:
                            g, lat, target = dat
                            model_input = [g.to(device), lat.to(device)]
                    else:
                        raise ValueError(
                            f"Unexpected batch format with {len(dat)} items."
                        )
                    out_data = best_model(model_input)["out"]
                    # out_data = net([g.to(device), lg.to(device)])["out"]
                    out_data = out_data.detach().cpu().numpy()
                    if config.standard_scalar_and_pca:
                        sc = pk.load(open("sc.pkl", "rb"))
                        out_data = list(
                            sc.transform(np.array(out_data).reshape(1, -1))[0]
                        )  # [0][0]
                    target = target.cpu().numpy()
                    out_data = np.atleast_2d(out_data)
                    target = np.atleast_2d(target)
                    batch_size = target.shape[0]
                    batch_ids = ids[id_offset : id_offset + batch_size]
                    id_offset += batch_size
                    for sample_id, sample_target, sample_prediction in zip(
                        batch_ids, target, out_data
                    ):
                        info = {}
                        info["id"] = sample_id
                        info["target"] = sample_target.flatten().tolist()
                        info["predictions"] = (
                            sample_prediction.flatten().tolist()
                        )
                        mem.append(info)
            dumpjson(
                filename=os.path.join(
                    config.output_dir, "multi_out_predictions.json"
                ),
                data=mem,
            )
        if (
            config.write_predictions
            and not classification
            and config.model.output_features == 1
            and config.model.gradwise_weight == 0
        ):
            best_model.eval()
            # net.eval()
            f = open(
                os.path.join(
                    config.output_dir, "prediction_results_test_set.csv"
                ),
                "w",
            )
            f.write("id,target,prediction\n")
            targets = []
            predictions = []
            with torch.no_grad():
                ids = list(test_loader.dataset.ids)
                id_offset = 0
                for dat in test_loader:
                    target = dat[-1]
                    model_input = _model_input_from_batch(dat, config, device)
                    out_data = best_model(model_input)["out"]
                    # out_data = net([g.to(device), lg.to(device)])["out"]
                    out_data = out_data.detach().cpu().numpy().reshape(-1)
                    if config.standard_scalar_and_pca:
                        sc = pk.load(
                            open(os.path.join(tmp_output_dir, "sc.pkl"), "rb")
                        )
                        out_data = sc.transform(
                            np.array(out_data).reshape(-1, 1)
                        ).reshape(-1)
                    target = target.detach().cpu().numpy().reshape(-1)
                    batch_ids = ids[id_offset : id_offset + len(target)]
                    id_offset += len(target)
                    for sample_id, target_value, pred_value in zip(
                        batch_ids, target, out_data
                    ):
                        f.write(
                            "%s, %6f, %6f\n"
                            % (sample_id, float(target_value), float(pred_value))
                        )
                        targets.append(float(target_value))
                        predictions.append(float(pred_value))
            f.close()

            print(
                "Test MAE:",
                mean_absolute_error(np.array(targets), np.array(predictions)),
            )
            best_model.eval()
            # net.eval()
            f = open(
                os.path.join(
                    config.output_dir, "prediction_results_train_set.csv"
                ),
                "w",
            )
            f.write("target,prediction\n")
            targets = []
            predictions = []
            with torch.no_grad():
                for dat in train_loader:
                    target = dat[-1]
                    model_input = _model_input_from_batch(dat, config, device)
                    out_data = best_model(model_input)["out"]
                    # out_data = net([g.to(device), lg.to(device)])["out"]
                    out_data = out_data.detach().cpu().numpy().reshape(-1)
                    if config.standard_scalar_and_pca:
                        sc = pk.load(
                            open(os.path.join(tmp_output_dir, "sc.pkl"), "rb")
                        )
                        out_data = sc.transform(
                            np.array(out_data).reshape(-1, 1)
                        ).reshape(-1)
                    target = target.detach().cpu().numpy().reshape(-1)
                    for ii, jj in zip(target, out_data):
                        f.write("%6f, %6f\n" % (float(ii), float(jj)))
                        targets.append(float(ii))
                        predictions.append(float(jj))
            f.close()
        if config.use_lmdb:
            print("Closing LMDB.")
            train_loader.dataset.close()
            val_loader.dataset.close()
            test_loader.dataset.close()


if __name__ == "__main__":
    config = TrainingConfig(
        random_seed=123, epochs=10, n_train=32, n_val=32, batch_size=16
    )
    history = train_dgl(config)
