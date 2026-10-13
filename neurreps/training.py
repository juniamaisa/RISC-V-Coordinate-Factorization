from __future__ import annotations

import copy
import json
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .models import build_model, count_trainable_parameters


@dataclass
class TrainResult:
    model: torch.nn.Module
    metrics: dict
    latents: dict[str, np.ndarray]
    probabilities: dict[str, np.ndarray]
    labels: dict[str, np.ndarray]
    indices: dict[str, np.ndarray]
    test_layer_features: dict[str, np.ndarray] | None = None




def resolve_device(device: str | torch.device | None = None) -> str:
    """Resolve device policy.

    ``None`` and ``"auto"`` mean CUDA when available, otherwise CPU.
    Explicit CPU is respected. Explicit CUDA falls back to CPU when CUDA
    is unavailable, which keeps the configuration portable across laptops,
    workstations, and CI environments.
    """
    if isinstance(device, torch.device):
        device = str(device)
    if device is None:
        requested = "auto"
    else:
        requested = str(device).strip().lower()
    if requested in {"", "none", "null", "auto", "cuda_if_available", "cuda-if-available"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        warnings.warn(
            f"Requested device={device!r}, but CUDA is unavailable; falling back to CPU.",
            RuntimeWarning,
        )
        return "cpu"
    return str(device)

def configure_torch_determinism(enabled: bool = True) -> None:
    if bool(enabled):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(False)


def seed_everything(seed: int, *, deterministic_algorithms: bool = True) -> None:
    configure_torch_determinism(deterministic_algorithms)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _has_both_classes(y: np.ndarray, idx: np.ndarray) -> bool:
    return set(np.unique(np.asarray(y)[np.asarray(idx, dtype=np.int64)]).tolist()) == {0, 1}


def grouped_split(
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    max_attempts: int = 100,
):
    """Group-disjoint split with both classes in every partition."""
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups)
    idx = np.arange(len(y))
    for attempt in range(int(max_attempts)):
        rs = int(seed) + 104729 * attempt
        outer = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=rs)
        train_idx, temp_idx = next(outer.split(idx, y, groups))
        relative_val = val_ratio / (1.0 - train_ratio)
        inner = GroupShuffleSplit(n_splits=1, train_size=relative_val, random_state=rs + 1)
        val_local, test_local = next(inner.split(temp_idx, y[temp_idx], groups[temp_idx]))
        val_idx, test_idx = temp_idx[val_local], temp_idx[test_local]
        if all(_has_both_classes(y, part) for part in (train_idx, val_idx, test_idx)):
            return (
                np.sort(train_idx.astype(np.int64)),
                np.sort(val_idx.astype(np.int64)),
                np.sort(test_idx.astype(np.int64)),
            )
    raise RuntimeError("Unable to create a group-disjoint split containing both classes")


def _loader(
    X, y, idx, batch_size, shuffle, mask: np.ndarray | None = None, *, pin_memory: bool = False
):
    idx = np.asarray(idx, dtype=np.int64)
    tensors = [
        torch.from_numpy(np.asarray(X)[idx]),
        torch.from_numpy(np.asarray(y)[idx].astype(np.float32)),
    ]
    if mask is not None:
        tensors.append(torch.from_numpy(np.asarray(mask)[idx].astype(np.float32)))
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=bool(pin_memory),
    )


def _unpack_batch(batch, device):
    non_blocking = torch.device(device).type == "cuda"
    if len(batch) == 2:
        xb, yb = batch
        mb = None
    elif len(batch) == 3:
        xb, yb, mb = batch
        mb = mb.to(device, non_blocking=non_blocking)
    else:
        raise RuntimeError(f"Unexpected batch length: {len(batch)}")
    return (
        xb.to(device, non_blocking=non_blocking),
        yb.to(device, non_blocking=non_blocking),
        mb,
    )


def _predict(model, loader, device, *, collect_latent: bool = True):
    model.eval()
    probs, labels = [], []
    latents = [] if collect_latent else None
    with torch.inference_mode():
        for batch in loader:
            xb, yb, mb = _unpack_batch(batch, device)
            if collect_latent:
                logits, z = model(xb, mask=mb, return_latent=True)
                latents.append(z.cpu().numpy())
            else:
                logits = model(xb, mask=mb)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(yb.cpu().numpy())
    return (
        np.concatenate(probs),
        np.concatenate(labels).astype(int),
        np.concatenate(latents) if collect_latent else None,
    )


def _predict_with_intermediates(model, loader, device):
    model.eval()
    probs, labels, latents = [], [], []
    collected: dict[str, list[np.ndarray]] = {}
    with torch.inference_mode():
        for batch in loader:
            xb, yb, mb = _unpack_batch(batch, device)
            z, features = model.encode_with_intermediates(xb, mask=mb)
            logits = model.classifier(z).squeeze(-1)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(yb.cpu().numpy())
            latents.append(z.cpu().numpy())
            for name, value in features.items():
                collected.setdefault(name, []).append(value.cpu().numpy())
    return (
        np.concatenate(probs),
        np.concatenate(labels).astype(int),
        np.concatenate(latents),
        {name: np.concatenate(parts) for name, parts in collected.items()},
    )


def _threshold_at_target_fpr(y: np.ndarray, prob: np.ndarray, target_fpr: float) -> float:
    negative = np.asarray(prob)[np.asarray(y).astype(int) == 0]
    if len(negative) == 0:
        return 0.5
    return float(np.quantile(negative, 1.0 - float(target_fpr), method="higher"))


def _error_rates(y: np.ndarray, prob: np.ndarray, threshold: float) -> tuple[float, float]:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(prob) >= float(threshold)).astype(int)
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    return float(fn / max(fn + tp, 1)), float(fp / max(fp + tn, 1))


def _expected_calibration_error(y: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    y, prob = np.asarray(y).astype(int), np.asarray(prob, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (prob >= lo) & (prob < hi if hi < 1.0 else prob <= hi)
        if mask.any():
            ece += (mask.sum() / max(len(y), 1)) * abs(prob[mask].mean() - y[mask].mean())
    return float(ece)


def binary_metrics(y: np.ndarray, prob: np.ndarray, prefix: str = "") -> dict:
    y, prob = np.asarray(y).astype(int), np.asarray(prob, dtype=float)
    out = {f"{prefix}samples": int(len(y))}
    if len(y) == 0:
        return out
    out[f"{prefix}positive_samples"] = int((y == 1).sum())
    out[f"{prefix}negative_samples"] = int((y == 0).sum())
    if set(np.unique(y).tolist()) == {0, 1}:
        out[f"{prefix}pr_auc"] = float(average_precision_score(y, prob))
        out[f"{prefix}roc_auc"] = float(roc_auc_score(y, prob))
    fnr, fpr = _error_rates(y, prob, 0.5)
    out[f"{prefix}fnr"] = fnr
    out[f"{prefix}fpr"] = fpr
    out[f"{prefix}ece"] = _expected_calibration_error(y, prob)
    return out


def _weighted_rank_metrics_from_counts(
    y_sorted: np.ndarray,
    group_starts: np.ndarray,
    sample_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized AP/ROC-AUC for bootstrap multiplicities at fixed scores.

    Prediction scores never change across bootstrap replicates, so sorting once
    is sufficient. ``sample_counts`` has shape ``(B, n_samples)`` and contains
    the multiplicity of each sorted sample in every bootstrap replicate.
    Tied prediction scores are reduced before metric calculation, matching the
    threshold grouping used by rank-based binary metrics.
    """
    counts = np.asarray(sample_counts, dtype=np.float64)
    y = np.asarray(y_sorted, dtype=np.float64)
    pos = counts * y[None, :]
    neg = counts * (1.0 - y)[None, :]
    pos_g = np.add.reduceat(pos, group_starts, axis=1)
    neg_g = np.add.reduceat(neg, group_starts, axis=1)
    total_pos = pos_g.sum(axis=1)
    total_neg = neg_g.sum(axis=1)
    valid = (total_pos > 0) & (total_neg > 0)

    cum_pos = np.cumsum(pos_g, axis=1)
    cum_total = np.cumsum(pos_g + neg_g, axis=1)
    precision = np.divide(cum_pos, cum_total, out=np.zeros_like(cum_pos), where=cum_total > 0)
    ap = np.divide(
        np.sum(precision * pos_g, axis=1),
        total_pos,
        out=np.full(len(counts), np.nan, dtype=np.float64),
        where=total_pos > 0,
    )

    neg_below = total_neg[:, None] - np.cumsum(neg_g, axis=1)
    concordant = np.sum(pos_g * (neg_below + 0.5 * neg_g), axis=1)
    auc = np.divide(
        concordant,
        total_pos * total_neg,
        out=np.full(len(counts), np.nan, dtype=np.float64),
        where=valid,
    )
    return ap, auc, valid


def bootstrap_binary_metrics(
    y: np.ndarray,
    prob: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    n_boot: int = 2000,
    seed: int = 42,
    batch_replicates: int = 64,
) -> dict[str, float]:
    """Cluster-aware bootstrap intervals for threshold-free test metrics.

    Independent split components are sampled when ``groups`` is supplied. This
    respects matched-pair/duplicate dependence. The implementation sorts model
    scores once and evaluates bootstrap multiplicities in vectorized batches,
    avoiding thousands of repeated sklearn sorts without changing the metric
    definitions.
    """
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, dtype=float)
    if len(y) != len(prob) or len(y) == 0:
        return {}
    if groups is None:
        groups = np.arange(len(y), dtype=np.int64)
    groups = np.asarray(groups)
    if len(groups) != len(y):
        raise ValueError("Bootstrap groups must align with y/prob")

    _, group_inverse = np.unique(groups.astype(str), return_inverse=True)
    n_groups = int(group_inverse.max()) + 1 if len(group_inverse) else 0
    if n_groups == 0:
        return {}

    order = np.argsort(-prob, kind="mergesort")
    y_sorted = y[order]
    prob_sorted = prob[order]
    group_sorted = group_inverse[order]
    group_starts = np.r_[0, np.flatnonzero(prob_sorted[1:] != prob_sorted[:-1]) + 1].astype(np.int64)

    rng = np.random.default_rng(seed)
    ap_values: list[np.ndarray] = []
    auc_values: list[np.ndarray] = []
    remaining = int(n_boot)
    probabilities = np.full(n_groups, 1.0 / n_groups, dtype=np.float64)
    while remaining > 0:
        batch = min(int(batch_replicates), remaining)
        group_counts = rng.multinomial(n_groups, probabilities, size=batch)
        sample_counts = group_counts[:, group_sorted]
        ap, auc, valid = _weighted_rank_metrics_from_counts(y_sorted, group_starts, sample_counts)
        if np.any(valid):
            ap_values.append(ap[valid])
            auc_values.append(auc[valid])
        remaining -= batch

    out: dict[str, float] = {}
    for name, chunks in (("pr_auc", ap_values), ("roc_auc", auc_values)):
        if chunks:
            vals = np.concatenate(chunks)
            out[f"test_{name}_bootstrap_low"] = float(np.quantile(vals, 0.025))
            out[f"test_{name}_bootstrap_high"] = float(np.quantile(vals, 0.975))
            out[f"test_{name}_bootstrap_valid_replicates"] = int(len(vals))
    out["test_bootstrap_clusters"] = int(n_groups)
    return out


def predict_numpy(
    model: torch.nn.Module,
    X: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    batch_size: int = 512,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if len(X) == 0:
        latent_dim = int(model.classifier.in_features)
        return np.empty(0), np.empty((0, latent_dim), dtype=np.float32)
    device = resolve_device(device or next(model.parameters()).device)
    model.eval()
    probabilities, latents = [], []
    with torch.inference_mode():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(np.asarray(X)[start : start + batch_size]).to(device)
            mb = None
            if mask is not None:
                mb = torch.from_numpy(np.asarray(mask)[start : start + batch_size].astype(np.float32)).to(device)
            logits, z = model(xb, mask=mb, return_latent=True)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            latents.append(z.cpu().numpy())
    return np.concatenate(probabilities), np.concatenate(latents)


def extract_layer_features_numpy(
    model: torch.nn.Module,
    X: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    batch_size: int = 512,
    device: str | None = None,
) -> dict[str, np.ndarray]:
    """Return pooled adapter/block features and the final latent in row order."""
    device = resolve_device(device or next(model.parameters()).device)
    model.eval()
    collected: dict[str, list[np.ndarray]] = {}
    with torch.inference_mode():
        for start in range(0, len(X), batch_size):
            xb = torch.from_numpy(np.asarray(X)[start : start + batch_size]).to(device)
            mb = None
            if mask is not None:
                mb = torch.from_numpy(np.asarray(mask)[start : start + batch_size].astype(np.float32)).to(device)
            _, features = model.encode_with_intermediates(xb, mask=mb)
            for name, value in features.items():
                collected.setdefault(name, []).append(value.cpu().numpy())
    return {name: np.concatenate(parts) for name, parts in collected.items()}


def train_representation_model(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    representation: str,
    *,
    seed: int = 42,
    epochs: int = 120,
    max_epochs: int | None = None,
    epoch_extension: int = 60,
    auto_extend_epochs: bool = True,
    batch_size: int = 256,
    eval_batch_size: int | None = None,
    patience: int = 15,
    min_epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    scheduler_patience: int = 5,
    scheduler_factor: float = 0.5,
    min_lr: float = 1e-6,
    convergence_tail_epochs: int = 10,
    require_convergence: bool = True,
    early_stopping_min_delta: float = 1e-5,
    width: int = 32,
    latent_dim: int = 64,
    depth: int = 3,
    device: str | None = None,
    fixed_indices: dict[str, np.ndarray] | None = None,
    mask: np.ndarray | None = None,
    mask_aware: bool = False,
    input_channels: int | None = None,
    bootstrap_replicates: int = 1000,
    collect_test_intermediates: bool = False,
    deterministic_algorithms: bool = True,
) -> TrainResult:
    requested_device = "auto" if device is None else str(device)
    seed_everything(seed, deterministic_algorithms=deterministic_algorithms)
    device = resolve_device(device)
    if fixed_indices is None:
        train_idx, val_idx, test_idx = grouped_split(y, groups, seed)
        indices = {"train": train_idx, "val": val_idx, "test": test_idx}
    else:
        indices = {key: np.asarray(value, dtype=np.int64) for key, value in fixed_indices.items()}
    if mask_aware and mask is None:
        raise ValueError("mask_aware=True requires mask")
    if mask is not None and (len(mask) != len(X) or mask.shape[-1] != X.shape[-1]):
        raise ValueError(f"Mask {mask.shape} incompatible with X {X.shape}")

    model = build_model(
        representation,
        width=width,
        latent_dim=latent_dim,
        depth=depth,
        input_channels=input_channels if input_channels is not None else int(X.shape[1]),
        mask_aware=mask_aware,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(scheduler_factor),
        patience=int(scheduler_patience),
        min_lr=float(min_lr),
    )
    criterion = nn.BCEWithLogitsLoss()
    pin_memory = torch.device(device).type == "cuda"
    eval_batch_size = int(eval_batch_size if eval_batch_size is not None else batch_size)
    if eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive")
    train_loader = _loader(
        X, y, indices["train"], batch_size, True, mask=mask, pin_memory=pin_memory
    )
    val_loader = _loader(
        X, y, indices["val"], eval_batch_size, False, mask=mask, pin_memory=pin_memory
    )

    initial_epoch_limit = int(epochs)
    hard_max_epochs = int(max_epochs if max_epochs is not None else epochs)
    if initial_epoch_limit < int(min_epochs):
        raise ValueError("epochs must be >= min_epochs")
    if hard_max_epochs < initial_epoch_limit:
        raise ValueError("max_epochs must be >= epochs")
    if int(epoch_extension) <= 0:
        raise ValueError("epoch_extension must be positive")
    if int(patience) <= int(scheduler_patience):
        raise ValueError("early-stopping patience must exceed scheduler_patience")

    best_state, best_ap, best_epoch, wait = None, -np.inf, 0, 0
    stopped_early = False
    history = []
    current_limit = min(initial_epoch_limit, hard_max_epochs)
    extension_count = 0
    epoch = 0
    while epoch < current_limit:
        model.train()
        loss_sum = 0.0
        batch_count = 0
        for batch in train_loader:
            xb, yb, mb = _unpack_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb, mask=mb), yb)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())
            batch_count += 1
        val_prob, val_y, _ = _predict(model, val_loader, device, collect_latent=False)
        val_ap = average_precision_score(val_y, val_prob) if set(np.unique(val_y)) == {0, 1} else float("nan")
        current_lr = float(optimizer.param_groups[0]["lr"])
        epoch += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(loss_sum / max(batch_count, 1)),
                "val_pr_auc": float(val_ap),
                "lr": current_lr,
            }
        )
        if np.isfinite(val_ap):
            scheduler.step(float(val_ap))
        if np.isfinite(val_ap) and val_ap > best_ap + float(early_stopping_min_delta):
            best_ap = float(val_ap)
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = int(epoch)
            wait = 0
        else:
            wait += 1
        if epoch >= int(min_epochs) and wait >= int(patience):
            stopped_early = True
            break
        if epoch >= current_limit and not stopped_early:
            if bool(auto_extend_epochs) and current_limit < hard_max_epochs:
                new_limit = min(current_limit + int(epoch_extension), hard_max_epochs)
                extension_count += 1
                print(
                    f"[TRAIN] Extending {representation} seed={seed}: "
                    f"{current_limit}->{new_limit} epochs (best_epoch={best_epoch}, wait={wait})"
                )
                current_limit = new_limit
            else:
                break

    epochs_completed = int(len(history))
    converged = bool(stopped_early)
    near_cap = bool(best_epoch > max(0, epochs_completed - int(convergence_tail_epochs)))
    if not converged:
        convergence_message = (
            f"Training did not satisfy the early-stopping convergence criterion for "
            f"{representation} seed={seed}: best_epoch={best_epoch}, "
            f"epochs_completed={epochs_completed}, tail={convergence_tail_epochs}."
        )
        if bool(require_convergence):
            raise RuntimeError(
                convergence_message + " Increase epochs or inspect optimization."
            )
        warnings.warn(
            convergence_message
            + " Continuing with the checkpoint having the best validation PR-AUC.",
            RuntimeWarning,
        )
    if best_state is not None:
        model.load_state_dict(best_state)

    # Critical correction: all post-training extraction loaders are unshuffled,
    # so latent row i and label row i correspond exactly.
    eval_loaders = {
        split: _loader(
            X, y, idx, eval_batch_size, False, mask=mask, pin_memory=pin_memory
        )
        for split, idx in indices.items()
    }
    metrics = {
        "representation": representation,
        "seed": int(seed),
        "parameters": count_trainable_parameters(model),
        "best_val_pr_auc": float(best_ap),
        "best_epoch": int(best_epoch),
        "epochs_completed": int(epochs_completed),
        "initial_epoch_limit": int(initial_epoch_limit),
        "hard_max_epochs": int(hard_max_epochs),
        "epoch_extension": int(epoch_extension),
        "extension_count": int(extension_count),
        "deterministic_algorithms": bool(deterministic_algorithms),
        "requested_device": requested_device,
        "resolved_device": str(device),
        "cuda_available_at_training": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "auto_extend_epochs": bool(auto_extend_epochs),
        "stopped_early": bool(stopped_early),
        "converged": bool(converged),
        "best_epoch_in_final_tail": bool(near_cap),
        "final_lr": float(optimizer.param_groups[0]["lr"]),
        "scheduler_patience": int(scheduler_patience),
        "scheduler_factor": float(scheduler_factor),
        "min_lr": float(min_lr),
        "early_stopping_patience": int(patience),
        "min_epochs": int(min_epochs),
        "history": history,
        "mask_aware": bool(mask_aware),
        "input_channels": int(X.shape[1]),
        "train_batch_size": int(batch_size),
        "eval_batch_size": int(eval_batch_size),
        "width": int(width),
        "latent_dim": int(latent_dim),
        "depth": int(depth),
    }
    latents: dict[str, np.ndarray] = {}
    probabilities: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    test_layer_features: dict[str, np.ndarray] | None = None
    for split, loader in eval_loaders.items():
        if split == "test" and collect_test_intermediates:
            prob, yy, z, test_layer_features = _predict_with_intermediates(model, loader, device)
        else:
            prob, yy, z = _predict(model, loader, device, collect_latent=True)
        probabilities[split], labels[split], latents[split] = prob, yy, z
        expected = np.asarray(y)[indices[split]].astype(int)
        if not np.array_equal(yy, expected):
            raise RuntimeError(f"Post-training extraction order mismatch in {split}")
        metrics.update(binary_metrics(yy, prob, prefix=f"{split}_"))

    val_prob, val_y = probabilities["val"], labels["val"]
    test_prob, test_y = probabilities["test"], labels["test"]
    for target in (0.01, 0.05):
        threshold = _threshold_at_target_fpr(val_y, val_prob, target)
        val_fnr, val_fpr = _error_rates(val_y, val_prob, threshold)
        test_fnr, test_fpr = _error_rates(test_y, test_prob, threshold)
        key = f"val_fpr_{int(target * 100)}pct"
        metrics[f"threshold_at_{key}"] = threshold
        metrics[f"val_fnr_at_{key}"] = val_fnr
        metrics[f"val_fpr_at_{key}"] = val_fpr
        metrics[f"test_fnr_at_{key}"] = test_fnr
        metrics[f"test_fpr_at_{key}"] = test_fpr
    if bootstrap_replicates > 0:
        metrics.update(
            bootstrap_binary_metrics(
                test_y,
                test_prob,
                groups=np.asarray(groups)[indices["test"]],
                n_boot=bootstrap_replicates,
                seed=seed,
            )
        )
    return TrainResult(
        model=model,
        metrics=metrics,
        latents=latents,
        probabilities=probabilities,
        labels=labels,
        indices=indices,
        test_layer_features=test_layer_features,
    )


def save_train_result(
    result: TrainResult,
    out_dir: str | Path,
    representation: str,
    seed: int,
    sample_id: np.ndarray,
    sample_hash: np.ndarray,
    y: np.ndarray,
    *,
    compress: bool = True,
    fingerprint: dict | None = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(result.model.state_dict(), out_dir / f"model_{representation}_seed{seed}.pt")
    with (out_dir / f"metrics_{representation}_seed{seed}.json").open("w", encoding="utf-8") as handle:
        json.dump(result.metrics, handle, indent=2)
    for split, z in result.latents.items():
        idx = result.indices[split]
        saver = np.savez_compressed if compress else np.savez
        saver(
            out_dir / f"latents_{representation}_{split}_seed{seed}.npz",
            Z=z,
            probability=result.probabilities[split],
            y=result.labels[split],
            sample_id=np.asarray(sample_id)[idx],
            sample_hash=np.asarray(sample_hash)[idx],
            index=idx,
        )
    if result.test_layer_features is not None:
        saver = np.savez_compressed if compress else np.savez
        saver(
            out_dir / f"intermediates_{representation}_test_seed{seed}.npz",
            **{name: value for name, value in result.test_layer_features.items()},
        )
    if fingerprint is not None:
        payload = {"fingerprint": fingerprint}
        (out_dir / f"fingerprint_{representation}_seed{seed}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )


def load_train_result_if_complete(
    out_dir: str | Path,
    representation: str,
    seed: int,
    *,
    model: torch.nn.Module,
    expected_fingerprint: dict,
    require_intermediates: bool = False,
    device: str | None = None,
) -> TrainResult | None:
    """Load an exactly fingerprinted completed training run, otherwise miss."""
    out_dir = Path(out_dir)
    fingerprint_path = out_dir / f"fingerprint_{representation}_seed{seed}.json"
    metrics_path = out_dir / f"metrics_{representation}_seed{seed}.json"
    model_path = out_dir / f"model_{representation}_seed{seed}.pt"
    latent_paths = {
        split: out_dir / f"latents_{representation}_{split}_seed{seed}.npz"
        for split in ("train", "val", "test")
    }
    intermediate_path = out_dir / f"intermediates_{representation}_test_seed{seed}.npz"
    required = [fingerprint_path, metrics_path, model_path, *latent_paths.values()]
    if require_intermediates:
        required.append(intermediate_path)
    if not all(path.exists() for path in required):
        return None
    try:
        stored = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if stored.get("fingerprint") != expected_fingerprint:
        return None
    target_device = resolve_device(device)
    try:
        state = torch.load(model_path, map_location=target_device, weights_only=True)
    except TypeError:  # older torch without weights_only
        state = torch.load(model_path, map_location=target_device)
    model.load_state_dict(state)
    model = model.to(target_device)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    latents, probabilities, labels, indices = {}, {}, {}, {}
    for split, path in latent_paths.items():
        saved = np.load(path, allow_pickle=False)
        latents[split] = saved["Z"]
        probabilities[split] = saved["probability"]
        labels[split] = saved["y"].astype(int)
        indices[split] = saved["index"].astype(np.int64)
    features = None
    if intermediate_path.exists():
        saved = np.load(intermediate_path, allow_pickle=False)
        features = {name: saved[name] for name in saved.files}
    return TrainResult(
        model=model, metrics=metrics, latents=latents, probabilities=probabilities,
        labels=labels, indices=indices, test_layer_features=features,
    )


def encode_numpy(model, X: np.ndarray, *, mask: np.ndarray | None = None, batch_size: int = 512, device: str | None = None) -> np.ndarray:
    _, z = predict_numpy(model, X, mask=mask, batch_size=batch_size, device=device)
    return z
