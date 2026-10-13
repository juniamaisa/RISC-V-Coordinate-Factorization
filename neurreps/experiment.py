from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch

from processing.dataset_registry import _prepare_common_clean_samples

from .audit import distribution_balance, encoding_artifact_baseline, nuisance_only_baseline
from .cost import analyze_model_cost
from .counterfactual import (
    generate_counterfactual_and_control_pairs,
    load_weak_blocks,
    save_pairs,
)
from .data import (
    AlignedBundle,
    _aligned_cache_key,
    build_aligned_bundle,
    build_aligned_bundle_cached,
    build_arrays_from_chains,
    common_sample_ids,
    connected_match_groups,
    indices_for_sample_ids,
    nuisance_matched_pairs,
)
from .geometry import (
    class_geometry,
    counterfactual_direction_geometry,
    counterfactual_score_metrics,
    dimension_sensitivity,
    input_linear_probe,
    layerwise_alignment,
    linear_cka,
    linear_probe,
    mean_direction_cosine,
    optional_persistent_homology,
    procrustes_error,
    rsa_spearman,
    shuffled_counterfactual_null,
    validate_intrinsic_dimension_synthetic,
)
from .models import build_model, count_trainable_parameters, find_width_for_parameter_budget
from .reproducibility import sha256_json, sha256_strings, write_reproducibility_manifest
from .runtime import RuntimeProfiler
from .training import (
    configure_torch_determinism,
    grouped_split,
    predict_numpy,
    resolve_device,
    load_train_result_if_complete,
    save_train_result,
    seed_everything,
    train_representation_model,
)


def _load_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _stable_nuisance_matched_selections(
    bundles: Mapping[int, AlignedBundle], config: Mapping
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray | None], set[str]]:
    """Create one deterministic nuisance match shared by every context budget.

    The previous implementation repeatedly matched each ``S`` independently
    and intersected the selected identities. Greedy nearest-neighbour matching
    is not idempotent: after identities are removed, the nearest available
    partner can change, so the fixed-point iteration may keep shrinking and
    never stabilize.

    This implementation aligns the shared identities once, standardizes the
    nuisance variables inside each context budget, concatenates the standardized
    feature blocks, and performs a single one-to-one match in the joint space.
    Dividing each block by ``sqrt(number_of_budgets)`` makes the Euclidean
    distance an RMS distance across budgets, preserving the interpretation of
    the configured caliper. The resulting identity pairs are mapped back to
    every bundle, so all ``S`` values use exactly the same samples and pairs.
    """
    identifiers = common_sample_ids(bundles)
    if not identifiers:
        raise RuntimeError("No sample identities are shared across context budgets")

    ordered_ids = np.asarray(sorted(identifiers), dtype=str)
    if not config.get("nuisance_match", False):
        selections = {
            S: indices_for_sample_ids(bundle, set(ordered_ids.tolist()))
            for S, bundle in bundles.items()
        }
        return selections, {S: None for S in bundles}, set(ordered_ids.tolist())

    from sklearn.neighbors import NearestNeighbors

    features = tuple(
        config.get(
            "matching_features",
            [
                "raw_instruction_count",
                "raw_byte_count",
                "compressed_ratio",
                "raw_zero_byte_ratio",
                "pattern_0013_ratio",
                "padding_rate",
                "preserved_ratio",
                "trunc_removed",
            ],
        )
    )
    data_seed = int(config.get("data_seed", 42))
    caliper = float(config.get("matching_caliper", 0.35))
    max_neighbors = int(config.get("matching_max_neighbors", 50))

    per_s_indices: dict[int, np.ndarray] = {}
    standardized_blocks: list[np.ndarray] = []
    reference_labels: np.ndarray | None = None

    for S in sorted(bundles):
        bundle = bundles[S]
        id_to_index = {str(value): i for i, value in enumerate(bundle.sample_id)}
        if len(id_to_index) != len(bundle.sample_id):
            raise RuntimeError(f"Sample identifiers are not unique for S={S}")
        try:
            aligned_idx = np.asarray([id_to_index[value] for value in ordered_ids], dtype=np.int64)
        except KeyError as exc:
            raise RuntimeError(f"Shared sample identity missing from S={S}: {exc}") from exc
        per_s_indices[S] = aligned_idx

        labels = bundle.labels[aligned_idx].astype(int)
        if reference_labels is None:
            reference_labels = labels
        elif not np.array_equal(reference_labels, labels):
            raise RuntimeError(f"Label alignment differs across context budgets at S={S}")

        missing = [name for name in features if name not in bundle.metadata.columns]
        if missing:
            raise KeyError(f"Nuisance features missing for S={S}: {missing}")
        frame = (
            bundle.metadata.iloc[aligned_idx]
            .loc[:, list(features)]
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
        )
        frame = frame.fillna(frame.median(numeric_only=True))
        scale = frame.std(ddof=0).replace(0.0, 1.0)
        standardized_blocks.append(((frame - frame.mean()) / scale).to_numpy(dtype=float))

    assert reference_labels is not None
    if set(np.unique(reference_labels)) != {0, 1}:
        raise ValueError("Joint nuisance matching requires both labels")

    joint = np.concatenate(standardized_blocks, axis=1) / np.sqrt(len(standardized_blocks))
    idx0 = np.flatnonzero(reference_labels == 0).astype(np.int64)
    idx1 = np.flatnonzero(reference_labels == 1).astype(np.int64)
    rng = np.random.default_rng(data_seed)
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    query, pool = (idx0, idx1) if len(idx0) <= len(idx1) else (idx1, idx0)

    n_neighbors = min(max(1, max_neighbors), len(pool))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(joint[pool])
    distances, neighbors = nn.kneighbors(joint[query])
    used: set[int] = set()
    local_pairs: list[tuple[int, int]] = []
    for q, ds, ns in zip(query, distances, neighbors):
        chosen = None
        for distance, local_neighbor in zip(ds, ns):
            candidate = int(pool[int(local_neighbor)])
            if candidate not in used and float(distance) <= caliper:
                chosen = candidate
                break
        if chosen is None:
            continue
        used.add(chosen)
        if reference_labels[int(q)] == 0:
            local_pairs.append((int(q), chosen))
        else:
            local_pairs.append((chosen, int(q)))

    if not local_pairs:
        raise RuntimeError(
            "Joint nuisance matching found no pairs; increase matching_caliper "
            "or inspect the selected matching_features"
        )
    local_pairs_array = np.asarray(local_pairs, dtype=np.int64)
    selected_local = np.asarray(sorted(local_pairs_array.reshape(-1).tolist()), dtype=np.int64)
    selected_ids = set(ordered_ids[selected_local].tolist())

    selections: dict[int, np.ndarray] = {}
    pairs_by_s: dict[int, np.ndarray] = {}
    for S, aligned_idx in per_s_indices.items():
        global_pairs = aligned_idx[local_pairs_array]
        selected = np.asarray(sorted(global_pairs.reshape(-1).tolist()), dtype=np.int64)
        ids_at_s = set(bundles[S].sample_id[selected].astype(str).tolist())
        if ids_at_s != selected_ids:
            raise RuntimeError(f"Joint matched identity mapping failed for S={S}")
        selections[S] = selected
        pairs_by_s[S] = global_pairs.astype(np.int64)

    # Enforce a post-match balance gate on every feature actually used for
    # matching. This explicitly controls pattern_0013_ratio instead of merely
    # reporting its residual imbalance after training.
    balance_limit = float(config.get("matching_max_abs_smd", 0.10))
    require_balance = bool(config.get("matching_require_balance", True))
    worst = []
    for S, selected in selections.items():
        selected_meta = bundles[S].metadata.iloc[selected].reset_index(drop=True)
        balance = distribution_balance(selected_meta, features=features)
        if balance.empty:
            continue
        row = balance.loc[balance["abs_smd"].idxmax()]
        worst.append((int(S), str(row["feature"]), float(row["abs_smd"])))
    if worst:
        worst_S, worst_feature, worst_smd = max(worst, key=lambda value: value[2])
        print(
            f"[MATCH] Worst post-match |SMD|={worst_smd:.4f} "
            f"for {worst_feature} at S={worst_S}"
        )
        if require_balance and worst_smd > balance_limit:
            raise RuntimeError(
                f"Nuisance balance gate failed: |SMD|={worst_smd:.4f} for "
                f"{worst_feature} at S={worst_S}, limit={balance_limit:.4f}. "
                "Tighten matching_caliper or inspect matching_features."
            )

    print(
        f"[MATCH] Joint nuisance matching across S={sorted(bundles)} retained "
        f"{len(local_pairs_array):,} pairs / {len(ordered_ids):,} shared identities"
    )
    return selections, pairs_by_s, selected_ids


def _raw_input_matrix(X: np.ndarray, representation: str) -> np.ndarray:
    """Flatten the presented coordinates without changing their information."""
    rep = str(representation).upper()
    values = np.asarray(X, dtype=np.float32).reshape(len(X), -1)
    return values if rep == "32C" else values / 255.0


def _raw_input_alignment_rows(
    arrays: Mapping[str, np.ndarray],
    *,
    S: int,
    seed: int,
    max_samples: int = 1500,
) -> list[dict]:
    """Describe pre-training geometry on one paired deterministic subset.

    CKA and RSA are appropriate across unequal coordinate dimensions. A raw
    Procrustes score is deliberately omitted here because SC/4C and 32C have
    different dimensions; forcing PCA before training would introduce an
    additional learned coordinate transform into this control.
    """
    reps = sorted(arrays)
    n = len(next(iter(arrays.values()))) if arrays else 0
    if any(len(arrays[rep]) != n for rep in reps):
        raise ValueError("Raw-input alignment requires paired rows")
    idx = np.arange(n, dtype=np.int64)
    if n > int(max_samples):
        rng = np.random.default_rng(int(seed))
        idx = np.sort(rng.choice(n, int(max_samples), replace=False))
    flat = {rep: _raw_input_matrix(arrays[rep][idx], rep) for rep in reps}
    rows = []
    for i, a in enumerate(reps):
        for b in reps[i + 1:]:
            rows.append({
                "S": int(S),
                "representation_a": a,
                "representation_b": b,
                "samples": int(len(idx)),
                "linear_cka": linear_cka(flat[a], flat[b]),
                "rsa_spearman": rsa_spearman(flat[a], flat[b], max_samples=len(idx), seed=int(seed)),
                "scope": "pretraining_selected_dataset",
            })
    return rows


def _cached_joint_nuisance_selections(
    bundles: Mapping[int, AlignedBundle], config: Mapping
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray | None], set[str]]:
    """Persist the deterministic joint nuisance match for exact reruns."""
    if not bool(config.get("cache_joint_matching", True)):
        return _stable_nuisance_matched_selections(bundles, config)
    mode_folder = Path(config["mode_folder"])
    software = str(config["software"])
    payload = {
        "version": "joint-nuisance-v1",
        "S_values": sorted(int(S) for S in bundles),
        "bundle_keys": {
            str(S): _aligned_cache_key(
                mode_folder, software, int(S),
                max_per_class=config.get("max_per_class"),
                seed=int(config.get("data_seed", 42)),
                remove_cross_label_tensor_collisions=True,
            )
            for S in bundles
        },
        "nuisance_match": bool(config.get("nuisance_match", False)),
        "matching_caliper": float(config.get("matching_caliper", 0.35)),
        "matching_max_neighbors": int(config.get("matching_max_neighbors", 50)),
        "matching_features": list(config.get("matching_features", [])),
        "data_seed": int(config.get("data_seed", 42)),
    }
    key = sha256_json(payload)[:20]
    root = Path(config.get("joint_matching_cache_dir") or (mode_folder / ".neurreps_cache"))
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"joint_match_{software}_{key}.npz"
    if path.exists():
        saved = np.load(path, allow_pickle=False)
        selections = {int(S): saved[f"selected_{int(S)}"].astype(np.int64) for S in bundles}
        pairs_by_s = {
            int(S): (saved[f"pairs_{int(S)}"].astype(np.int64) if f"pairs_{int(S)}" in saved.files else None)
            for S in bundles
        }
        identity_sets = {
            int(S): set(bundles[int(S)].sample_id[selections[int(S)]].astype(str).tolist())
            for S in bundles
        }
        first = next(iter(identity_sets.values()))
        if not all(values == first for values in identity_sets.values()):
            raise RuntimeError(f"Joint matching cache identity mismatch: {path}")
        print(f"[CACHE] Joint nuisance matching loaded: {path}")
        return selections, pairs_by_s, set(first)

    selections, pairs_by_s, shared = _stable_nuisance_matched_selections(bundles, config)
    arrays = {}
    for S in bundles:
        arrays[f"selected_{int(S)}"] = np.asarray(selections[int(S)], dtype=np.int64)
        if pairs_by_s[int(S)] is not None:
            arrays[f"pairs_{int(S)}"] = np.asarray(pairs_by_s[int(S)], dtype=np.int64)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)
    print(f"[CACHE] Joint nuisance matching saved: {path}")
    return selections, pairs_by_s, shared


MAIN_TRAINING_PROTOCOL_KEYS = (
    "epochs", "max_epochs", "epoch_extension", "auto_extend_epochs",
    "batch_size", "eval_batch_size", "patience", "min_epochs",
    "lr", "weight_decay", "scheduler_patience", "scheduler_factor",
    "min_lr", "convergence_tail_epochs", "require_convergence",
    "early_stopping_min_delta", "bootstrap_replicates", "device",
    "deterministic_torch",
)


def _main_training_protocol_signature(config: Mapping) -> dict:
    """Fields that can change learned weights/checkpoint selection or saved test metrics."""
    return {key: config.get(key) for key in MAIN_TRAINING_PROTOCOL_KEYS}


def _model_spec(
    rep: str,
    budget: str,
    config: Mapping,
    *,
    input_channels: int,
) -> dict[str, int]:
    override = config.get("model_overrides", {}).get(rep, {})
    base_width = int(override.get("width", config.get("width", 32)))
    latent_dim = int(override.get("latent_dim", config.get("latent_dim", 64)))
    depth = int(override.get("depth", config.get("depth", 3)))
    if budget == "fixed_core":
        width = base_width
    elif budget == "matched_total":
        target_rep = str(config.get("parameter_target_representation", "4C")).upper()
        target_channels = {"SC": 1, "4C": 4, "32C": 32}[target_rep]
        target_model = build_model(
            target_rep,
            width=int(config.get("width", 32)),
            latent_dim=int(config.get("latent_dim", 64)),
            depth=int(config.get("depth", 3)),
            input_channels=target_channels,
        )
        target = count_trainable_parameters(target_model)
        width, achieved = find_width_for_parameter_budget(
            rep,
            target,
            latent_dim=latent_dim,
            depth=depth,
            input_channels=input_channels,
        )
        tolerance = float(config.get("parameter_match_tolerance", 0.03))
        if abs(achieved - target) / max(target, 1) > tolerance:
            raise RuntimeError(
                f"Could not match {rep} parameter budget within {tolerance:.1%}: "
                f"target={target}, achieved={achieved}"
            )
    else:
        raise ValueError(f"Unknown model budget: {budget}")
    return {"width": int(width), "latent_dim": latent_dim, "depth": depth}


def _initialization_geometry_control(
    X: np.ndarray,
    y: np.ndarray,
    test_idx: np.ndarray,
    *,
    rep: str,
    spec: Mapping,
    seed: int,
    device: str | None,
    batch_size: int,
    max_samples: int,
    deterministic_algorithms: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Geometry induced by coordinates + architecture before learning.

    ``train_representation_model`` reseeds before constructing the trained
    model. Reseeding here with the same seed constructs the identical initial
    model distribution without consuming RNG state from the subsequent
    training run. Only a deterministic subset of the test split is encoded.
    """
    test_idx = np.asarray(test_idx, dtype=np.int64)
    local = np.arange(len(test_idx), dtype=np.int64)
    if len(local) > int(max_samples):
        rng = np.random.default_rng(int(seed) + 9001)
        local = np.sort(rng.choice(local, size=int(max_samples), replace=False))
    global_idx = test_idx[local]
    seed_everything(int(seed), deterministic_algorithms=deterministic_algorithms)
    resolved_device = resolve_device(device)
    model = build_model(
        rep,
        width=int(spec["width"]),
        latent_dim=int(spec["latent_dim"]),
        depth=int(spec["depth"]),
        input_channels=int(X.shape[1]),
    ).to(resolved_device)
    _, initial_z = predict_numpy(
        model,
        X[global_idx],
        device=resolved_device,
        batch_size=int(batch_size),
    )
    yy = np.asarray(y)[global_idx].astype(int)
    metrics = class_geometry(initial_z, yy, seed=int(seed))
    return local, initial_z, metrics


def _counterfactual_rows(
    pairs: list[dict],
    rep: str,
    benign_prob: np.ndarray,
    modified_prob: np.ndarray,
    benign_z: np.ndarray,
    modified_z: np.ndarray,
    *,
    threshold: float,
    S: int,
    seed: int,
    budget: str,
    null_repeats: int = 200,
) -> tuple[list[dict], list[dict], list[dict]]:
    detail_rows: list[dict] = []
    summary_rows: list[dict] = []
    contrast_rows: list[dict] = []
    meta = pd.DataFrame(pairs)
    for (control_type, attack_type), part in meta.groupby(["control_type", "attack_type"]):
        idx = part.index.to_numpy(dtype=np.int64)
        geometry = counterfactual_direction_geometry(benign_z[idx], modified_z[idx])
        score = counterfactual_score_metrics(
            benign_prob[idx], modified_prob[idx], threshold=threshold
        )
        summary_rows.append(
            {
                "S": int(S),
                "seed": int(seed),
                "model_budget": budget,
                "representation": rep,
                "control_type": control_type,
                "attack_type": attack_type,
                "pairs": int(len(idx)),
                **geometry,
                **score,
                **shuffled_counterfactual_null(
                    benign_z[idx], modified_z[idx], repeats=int(null_repeats), seed=seed + len(idx)
                ),
            }
        )
        deltas = modified_z[idx] - benign_z[idx]
        distances = np.linalg.norm(deltas, axis=1)
        for local, global_index in enumerate(idx):
            detail_rows.append(
                {
                    "S": int(S),
                    "seed": int(seed),
                    "model_budget": budget,
                    "representation": rep,
                    "pair_id": pairs[int(global_index)]["pair_id"],
                    "comparison_group_id": pairs[int(global_index)].get("comparison_group_id", ""),
                    "control_type": control_type,
                    "attack_type": attack_type,
                    "length_difference": int(pairs[int(global_index)].get("length_difference", 0)),
                    "distance": float(distances[local]),
                    "benign_probability": float(benign_prob[global_index]),
                    "modified_probability": float(modified_prob[global_index]),
                    "probability_shift": float(modified_prob[global_index] - benign_prob[global_index]),
                }
            )
    # Angular relation between entry and exit mean invalid directions.
    invalid_entry = meta.index[(meta.control_type == "invalid") & (meta.attack_type == "entry")].to_numpy()
    invalid_exit = meta.index[(meta.control_type == "invalid") & (meta.attack_type == "exit")].to_numpy()
    if len(invalid_entry) and len(invalid_exit):
        cosine = mean_direction_cosine(
            modified_z[invalid_entry] - benign_z[invalid_entry],
            modified_z[invalid_exit] - benign_z[invalid_exit],
        )
        for row in summary_rows:
            row["invalid_entry_exit_mean_direction_cosine"] = cosine

    # Base-matched invalid-versus-valid contrasts. Each comparison group uses
    # the same benign context and changed position, isolating CFG invalidity
    # from generic block substitution.
    detail = pd.DataFrame(detail_rows)
    if not detail.empty and "comparison_group_id" in detail:
        for (attack_type, group_id), part in detail.groupby(["attack_type", "comparison_group_id"]):
            by_type = {row.control_type: row for row in part.itertuples(index=False)}
            if set(by_type) != {"invalid", "valid"}:
                continue
            invalid, valid = by_type["invalid"], by_type["valid"]
            invalid_index = int(meta.index[(meta.comparison_group_id == group_id) & (meta.control_type == "invalid")][0])
            valid_index = int(meta.index[(meta.comparison_group_id == group_id) & (meta.control_type == "valid")][0])
            invalid_delta = modified_z[invalid_index] - benign_z[invalid_index]
            valid_delta = modified_z[valid_index] - benign_z[valid_index]
            cosine = float(
                np.dot(invalid_delta, valid_delta)
                / max(np.linalg.norm(invalid_delta) * np.linalg.norm(valid_delta), 1e-12)
            )
            contrast_rows.append(
                {
                    "S": int(S),
                    "seed": int(seed),
                    "model_budget": budget,
                    "representation": rep,
                    "attack_type": attack_type,
                    "comparison_group_id": group_id,
                    "invalid_minus_valid_distance": float(invalid.distance - valid.distance),
                    "invalid_minus_valid_probability_shift": float(
                        invalid.probability_shift - valid.probability_shift
                    ),
                    "invalid_minus_valid_modified_probability": float(
                        invalid.modified_probability - valid.modified_probability
                    ),
                    "invalid_outscores_valid": float(
                        invalid.modified_probability > valid.modified_probability
                    ),
                    "invalid_valid_delta_cosine": cosine,
                }
            )
    return summary_rows, detail_rows, contrast_rows


def run(config: dict) -> Path:
    configure_torch_determinism(bool(config.get("deterministic_torch", True)))
    out = Path(config.get("output_dir", "results/neurreps_geometry"))
    out.mkdir(parents=True, exist_ok=True)
    profiler = RuntimeProfiler(out)
    integrity_rows: list[dict] = []
    with profiler.stage("reproducibility_manifest"):
        reproducibility_manifest = write_reproducibility_manifest(
            out,
            config=config,
            project_root=Path(__file__).resolve().parents[1],
            mode_folder=config["mode_folder"],
            software=config["software"],
            use_hash_cache=bool(config.get("cache_file_hashes", True)),
        )
    if bool(config.get("validate_intrinsic_dimension", True)):
        with profiler.stage("intrinsic_dimension_validation"):
            validation_rows = validate_intrinsic_dimension_synthetic(
            dimensions=config.get("intrinsic_validation_dimensions", [2, 5, 10, 20]),
            n_samples=int(config.get("intrinsic_validation_samples", 4000)),
            repeats=int(config.get("intrinsic_validation_repeats", 3)),
            relative_tolerance=float(config.get("intrinsic_validation_relative_tolerance", 0.30)),
            seed=int(config.get("intrinsic_validation_seed", 1729)),
        )
        validation_frame = pd.DataFrame(validation_rows)
        validation_frame.to_csv(out / "intrinsic_dimension_synthetic_validation.csv", index=False)
        validation_summary = {
            "passed": bool(validation_frame["passed"].all()),
            "max_relative_error": float(validation_frame["relative_error"].max()),
            "rows": validation_rows,
        }
        (out / "intrinsic_dimension_synthetic_validation.json").write_text(
            json.dumps(validation_summary, indent=2), encoding="utf-8"
        )
        if bool(config.get("require_intrinsic_validation", True)) and not validation_summary["passed"]:
            raise RuntimeError(
                "Intrinsic-dimension synthetic validation failed; "
                "do not run the scientific experiment until the estimator is fixed."
            )
        print(
            f"[DIM] Synthetic MLE validation passed={validation_summary['passed']} "
            f"max_relative_error={validation_summary['max_relative_error']:.3f}"
        )
    reps = [str(rep).upper() for rep in config.get("representations", ["SC", "4C", "32C"])]
    seeds = [int(seed) for seed in config.get("seeds", [11, 23, 37, 53, 71])]
    S_values = [int(value) for value in config.get("S_values", [32])]
    budgets = list(config.get("model_budgets", ["fixed_core", "matched_total"]))

    with profiler.stage("common_dataset_loading"):
        common = _prepare_common_clean_samples(config["mode_folder"], config["software"])
    bundles: dict[int, AlignedBundle] = {}
    for S in S_values:
        with profiler.stage("aligned_bundle", S=int(S)):
            bundles[S] = build_aligned_bundle_cached(
                config["mode_folder"],
                config["software"],
                S,
                max_per_class=config.get("max_per_class"),
                seed=int(config.get("data_seed", 42)),
                remove_cross_label_tensor_collisions=True,
                use_cache=bool(config.get("cache_aligned_bundles", True)),
                cache_dir=config.get("aligned_bundle_cache_dir"),
                common=common,
            )
    if config.get("common_identities_across_S", True):
        with profiler.stage("joint_nuisance_matching"):
            selections, pairs_by_s, shared_ids = _cached_joint_nuisance_selections(bundles, config)
    else:
        selections, pairs_by_s, shared_ids = {}, {}, set()
        for S, bundle in bundles.items():
            candidate = np.arange(len(bundle.labels), dtype=np.int64)
            pairs = None
            if config.get("nuisance_match", False):
                pairs = nuisance_matched_pairs(
                    bundle.metadata,
                    bundle.labels,
                    candidate_indices=candidate,
                    seed=int(config.get("data_seed", 42)),
                    caliper=float(config.get("matching_caliper", 0.35)),
                    features=tuple(config.get("matching_features", [
                        "raw_instruction_count", "raw_byte_count", "compressed_ratio",
                        "raw_zero_byte_ratio", "pattern_0013_ratio", "padding_rate",
                        "preserved_ratio", "trunc_removed",
                    ])),
                )
                candidate = np.asarray(sorted(pairs.reshape(-1).tolist()), dtype=np.int64)
            selections[S], pairs_by_s[S] = candidate, pairs

    geometry_rows: list[dict] = []
    alignment_rows: list[dict] = []
    layer_rows: list[dict] = []
    counterfactual_rows: list[dict] = []
    counterfactual_detail_rows: list[dict] = []
    counterfactual_contrast_rows: list[dict] = []
    cost_rows: list[dict] = []
    cost_cache: dict[tuple, dict] = {}
    dimension_rows: list[dict] = []
    input_probe_rows: list[dict] = []
    raw_input_alignment_rows: list[dict] = []
    initialization_control_rows: list[dict] = []
    reused_equivalent_training_rows: list[dict] = []
    selection_audit_rows: list[pd.DataFrame] = []
    selection_baseline_rows: list[dict] = []
    encoding_artifact_rows: list[dict] = []
    collision_frames: list[pd.DataFrame] = []

    with profiler.stage("weak_block_metadata_loading"):
        weak_blocks = load_weak_blocks(config.get("gadget_file"))

    for S in S_values:
        bundle = bundles[S]
        selected = selections[S]
        matched_pairs = pairs_by_s[S]
        y = bundle.labels[selected]
        with profiler.stage("selected_tensor_materialization", S=int(S)):
            selected_arrays = {rep: bundle.arrays[rep][selected] for rep in reps}
            selected_masks = {rep: bundle.masks[rep][selected] for rep in reps}
            selected_tensor_digests = {
                rep: sha256_strings(bundle.tensor_hashes[rep][selected].astype(str).tolist())
                for rep in reps
            }
        with profiler.stage("raw_input_alignment", S=int(S)):
            raw_input_alignment_rows.extend(
                _raw_input_alignment_rows(
                    selected_arrays,
                    S=int(S),
                    seed=int(config.get("data_seed", 42)) + int(S),
                    max_samples=int(config.get("raw_input_alignment_max_samples", 1500)),
                )
            )
        duplicate_keys = [bundle.tensor_hashes[rep] for rep in reps]
        with profiler.stage("group_components_and_nuisance_audit", S=int(S)):
            groups = connected_match_groups(
                bundle.sample_id,
                selected,
                matched_pairs,
                duplicate_keys=duplicate_keys,
            )
            selected_meta = bundle.metadata.iloc[selected].reset_index(drop=True)
            balance = distribution_balance(selected_meta)
            balance["S"] = S
            balance["selected_samples"] = len(selected)
            selection_audit_rows.append(balance)
            nuisance_result = nuisance_only_baseline(selected_meta, groups=groups, seeds=seeds)
            for nuisance_run in nuisance_result["runs"]:
                selection_baseline_rows.append(
                    {
                        "S": int(S),
                        "selected_samples": int(len(selected)),
                        **nuisance_run,
                    }
                )
            for artifact_run in encoding_artifact_baseline(
                selected_arrays, selected_masks, y, groups, seeds=seeds
            ):
                encoding_artifact_rows.append(
                    {
                        "S": int(S),
                        "selected_samples": int(len(selected)),
                        **artifact_run,
                    }
                )
        collision = bundle.collision_report.copy()
        collision["S"] = S
        collision_frames.append(collision)
        integrity_rows.append(
            {
                "kind": "selection",
                "S": int(S),
                "seed": None,
                "split": None,
                "sha256": sha256_strings(bundle.sample_id[selected].astype(str).tolist()),
                "canonical_content_sha256": sha256_strings(
                    bundle.tensor_hashes["4C"][selected].astype(str).tolist()
                ),
                "samples": int(len(selected)),
            }
        )

        for seed in seeds:
            with profiler.stage("grouped_split", S=int(S), seed=int(seed)):
                train_idx, val_idx, test_idx = grouped_split(y, groups, seed)
            fixed = {"train": train_idx, "val": val_idx, "test": test_idx}
            for split_name, split_idx in fixed.items():
                split_ids = bundle.sample_id[selected[split_idx]].astype(str)
                integrity_rows.append(
                    {
                        "kind": "split",
                        "S": int(S),
                        "seed": int(seed),
                        "split": split_name,
                        "sha256": sha256_strings(sorted(split_ids.tolist())),
                        "canonical_content_sha256": None,
                        "samples": int(len(split_idx)),
                    }
                )
            test_global = selected[test_idx]
            test_meta = bundle.metadata.iloc[test_global]
            test_benign = test_meta[test_meta.label == 0]
            with profiler.stage("counterfactual_generation", S=int(S), seed=int(seed)):
                pairs = generate_counterfactual_and_control_pairs(
                    common,
                    max_pairs_per_stratum=int(config.get("counterfactual_pairs_per_stratum", 500)),
                    seed=int(config.get("data_seed", 42)) + S + seed,
                    weak_blocks=weak_blocks,
                    length_tolerance=int(config.get("counterfactual_length_tolerance", 0)),
                    exact_length=bool(config.get("counterfactual_exact_length", True)),
                    allowed_sample_hashes=set(test_benign.sample_hash.astype(str).tolist()),
                    allowed_source_indices=set(test_benign.source_index.astype(int).tolist()),
                    match_entry_weak_status=bool(config.get("counterfactual_match_entry_weak_status", True)),
                )
            pair_dir = out / f"S{S}" / f"seed{seed}"
            save_pairs(pairs, pair_dir / "counterfactual_and_control_pairs.jsonl")
            integrity_rows.append(
                {
                    "kind": "counterfactual_pairs",
                    "S": int(S),
                    "seed": int(seed),
                    "split": "test",
                    "sha256": sha256_json(pairs),
                    "canonical_content_sha256": None,
                    "samples": int(len(pairs)),
                }
            )
            with profiler.stage("counterfactual_encoding", S=int(S), seed=int(seed)):
                pair_bundle = build_arrays_from_chains(
                    common, [row["benign_chain"] for row in pairs], S
                )
                modified_bundle = build_arrays_from_chains(
                    common, [row["modified_chain"] for row in pairs], S
                )

            input_probe_cache: dict[str, dict[str, float]] = {}
            equivalent_training_cache: dict[tuple, object] = {}
            equivalent_training_source: dict[tuple, str] = {}
            for budget in budgets:
                layers_by_rep: dict[str, dict[str, np.ndarray]] = {}
                for rep in reps:
                    X = selected_arrays[rep]
                    mask = selected_masks[rep]
                    spec = _model_spec(rep, budget, config, input_channels=X.shape[1])
                    model_dir = pair_dir / budget
                    with profiler.stage(
                        "initialization_geometry_control",
                        S=int(S), seed=int(seed), budget=str(budget), representation=rep,
                    ):
                        init_local_idx, initial_z, initial_geometry = _initialization_geometry_control(
                            X, y, test_idx, rep=rep, spec=spec, seed=seed,
                            device=config.get("device"),
                            batch_size=int(config.get("inference_batch_size", 1024)),
                            max_samples=int(config.get("initialization_control_max_samples", 1500)),
                            deterministic_algorithms=bool(config.get("deterministic_torch", True)),
                        )
                    fingerprint = {
                        "schema": "neurreps-train-v2",
                        "code_sha256": reproducibility_manifest["code"]["code_sha256"],
                        "training_protocol": _main_training_protocol_signature(config),
                        "S": int(S),
                        "seed": int(seed),
                        "model_budget": str(budget),
                        "representation": rep,
                        "model_spec": spec,
                        "selected_tensor_sha256": selected_tensor_digests[rep],
                        "split_sha256": {
                            name: sha256_strings(
                                sorted(bundle.sample_id[selected[idx]].astype(str).tolist())
                            )
                            for name, idx in fixed.items()
                        },
                    }
                    # A capacity label is not itself a distinct training condition.
                    # If two budgets resolve to the exact same architecture for the
                    # same representation, seed, tensor and split, the optimization
                    # trajectory is identical. Reuse it and retain both budget rows.
                    equivalent_key = (
                        rep, int(spec["width"]), int(spec["latent_dim"]), int(spec["depth"]),
                        selected_tensor_digests[rep],
                        tuple((name, fingerprint["split_sha256"][name]) for name in sorted(fingerprint["split_sha256"])),
                        sha256_json(fingerprint["training_protocol"]),
                        reproducibility_manifest["code"]["code_sha256"],
                    )
                    result = equivalent_training_cache.get(equivalent_key)
                    reused_from_budget = equivalent_training_source.get(equivalent_key)
                    if result is not None:
                        reused_equivalent_training_rows.append(
                            {
                                "S": int(S),
                                "seed": int(seed),
                                "representation": rep,
                                "model_budget": str(budget),
                                "reused_from_budget": str(reused_from_budget),
                                "width": int(spec["width"]),
                                "latent_dim": int(spec["latent_dim"]),
                                "depth": int(spec["depth"]),
                                "reason": "identical_training_condition",
                            }
                        )
                        print(
                            f"[CACHE] Equivalent training reused: S={S} seed={seed} "
                            f"rep={rep} {reused_from_budget}->{budget}"
                        )
                    if result is None and bool(config.get("resume_completed_models", True)):
                        resume_model = build_model(
                            rep,
                            width=spec["width"],
                            latent_dim=spec["latent_dim"],
                            depth=spec["depth"],
                            input_channels=X.shape[1],
                        )
                        with profiler.stage(
                            "model_resume_lookup", S=int(S), seed=int(seed),
                            budget=str(budget), representation=rep,
                        ):
                            result = load_train_result_if_complete(
                                model_dir, rep, seed, model=resume_model,
                                expected_fingerprint=fingerprint,
                                require_intermediates=True,
                                device=config.get("device"),
                            )
                        if result is not None:
                            print(
                                f"[CACHE] Completed model loaded: S={S} seed={seed} "
                                f"budget={budget} rep={rep}"
                            )
                    if result is None:
                        with profiler.stage(
                            "model_training",
                            S=int(S),
                            seed=int(seed),
                            budget=str(budget),
                            representation=rep,
                        ):
                            result = train_representation_model(
                                X,
                                y,
                                groups,
                                rep,
                                seed=seed,
                                epochs=int(config.get("epochs", 120)),
                                max_epochs=int(config.get("max_epochs", config.get("epochs", 120))),
                                epoch_extension=int(config.get("epoch_extension", 60)),
                                auto_extend_epochs=bool(config.get("auto_extend_epochs", True)),
                                batch_size=int(config.get("batch_size", 256)),
                                eval_batch_size=int(config.get("eval_batch_size", config.get("batch_size", 256))),
                                patience=int(config.get("patience", 15)),
                                min_epochs=int(config.get("min_epochs", 30)),
                                lr=float(config.get("lr", 1e-3)),
                                weight_decay=float(config.get("weight_decay", 1e-4)),
                                scheduler_patience=int(config.get("scheduler_patience", 5)),
                                scheduler_factor=float(config.get("scheduler_factor", 0.5)),
                                min_lr=float(config.get("min_lr", 1e-6)),
                                convergence_tail_epochs=int(config.get("convergence_tail_epochs", 10)),
                                require_convergence=bool(config.get("require_convergence", True)),
                                early_stopping_min_delta=float(config.get("early_stopping_min_delta", 1e-5)),
                                width=spec["width"],
                                latent_dim=spec["latent_dim"],
                                depth=spec["depth"],
                                device=config.get("device"),
                                fixed_indices=fixed,
                                bootstrap_replicates=int(config.get("bootstrap_replicates", 1000)),
                                collect_test_intermediates=True,
                                deterministic_algorithms=bool(config.get("deterministic_torch", True)),
                            )
                        with profiler.stage(
                            "model_artifact_serialization",
                            S=int(S), seed=int(seed), budget=str(budget), representation=rep,
                        ):
                            save_train_result(
                                result,
                                model_dir,
                                rep,
                                seed,
                                bundle.sample_id[selected],
                                bundle.sample_hash[selected],
                                y,
                                compress=bool(config.get("compress_latent_artifacts", False)),
                                fingerprint=fingerprint,
                            )
                    elif reused_from_budget is not None:
                        # Persist a complete artifact for traceability under the
                        # second budget label. The checkpoint is identical by
                        # construction; only its provenance fingerprint differs.
                        with profiler.stage(
                            "model_artifact_serialization_reuse",
                            S=int(S), seed=int(seed), budget=str(budget), representation=rep,
                        ):
                            save_train_result(
                                result, model_dir, rep, seed,
                                bundle.sample_id[selected], bundle.sample_hash[selected], y,
                                compress=bool(config.get("compress_latent_artifacts", False)),
                                fingerprint=fingerprint,
                            )
                    equivalent_training_cache[equivalent_key] = result
                    equivalent_training_source.setdefault(equivalent_key, str(budget))
                    z_train, z_test = result.latents["train"], result.latents["test"]
                    yy_train, yy_test = result.labels["train"], result.labels["test"]
                    trained_subset_z = z_test[init_local_idx]
                    trained_subset_y = yy_test[init_local_idx]
                    trained_subset_geometry = class_geometry(
                        trained_subset_z, trained_subset_y, seed=int(seed)
                    )
                    init_row = {
                        "S": int(S),
                        "seed": int(seed),
                        "model_budget": str(budget),
                        "representation": rep,
                        "samples": int(len(init_local_idx)),
                        "initial_to_trained_cka": linear_cka(initial_z, trained_subset_z),
                        "initial_to_trained_rsa": rsa_spearman(
                            initial_z, trained_subset_z,
                            max_samples=len(init_local_idx), seed=int(seed),
                        ),
                        "initial_to_trained_procrustes_error": procrustes_error(
                            initial_z, trained_subset_z
                        ),
                    }
                    for name, value in initial_geometry.items():
                        init_row[f"initial_{name}"] = value
                    for name, value in trained_subset_geometry.items():
                        init_row[f"trained_same_subset_{name}"] = value
                        if name in initial_geometry and np.isscalar(value):
                            try:
                                init_row[f"learning_delta_{name}"] = float(value) - float(initial_geometry[name])
                            except (TypeError, ValueError):
                                pass
                    initialization_control_rows.append(init_row)
                    metrics = dict(result.metrics)
                    # The full per-epoch history remains in the per-model JSON.
                    # Removing it from the table avoids a very large redundant CSV cell.
                    metrics.pop("history", None)
                    with profiler.stage(
                        "geometry_and_probes",
                        S=int(S), seed=int(seed), budget=str(budget), representation=rep,
                    ):
                        metrics.update(class_geometry(z_test, yy_test, seed=seed))
                        metrics.update(linear_probe(z_train, yy_train, z_test, yy_test, seed=seed))
                        if rep not in input_probe_cache:
                            input_probe_cache[rep] = input_linear_probe(
                                X[train_idx], yy_train, X[test_idx], yy_test, seed=seed
                            )
                        metrics.update(input_probe_cache[rep])
                        if config.get("persistent_homology", False):
                            metrics.update(optional_persistent_homology(z_test, seed=seed))
                    metrics.update(
                        {
                            "S": int(S),
                            "model_budget": budget,
                            "selected_samples": int(len(selected)),
                            "shared_identity_count": int(len(shared_ids)) if shared_ids else int(len(selected)),
                            "post_tensor_collision_rows_removed": int(
                                bundle.collision_report.rows_removed_union.max()
                            ),
                        }
                    )
                    geometry_rows.append(metrics)
                    input_probe_rows.append(
                        {
                            "S": S,
                            "seed": seed,
                            "model_budget": budget,
                            "representation": rep,
                            "input_probe_pr_auc": metrics["input_probe_pr_auc"],
                            "input_probe_roc_auc": metrics["input_probe_roc_auc"],
                        }
                    )
                    with profiler.stage(
                        "dimension_sensitivity",
                        S=int(S), seed=int(seed), budget=str(budget), representation=rep,
                    ):
                        for row in dimension_sensitivity(
                            z_test,
                            sample_sizes=config.get(
                                "dimension_sample_sizes", [250, 500, 1000, 2500, 5000, 10000]
                            ),
                            repeats=int(config.get("dimension_repeats", 3)),
                            seed=seed,
                        ):
                            dimension_rows.append(
                                {
                                    "S": S,
                                    "seed": seed,
                                    "model_budget": budget,
                                    "representation": rep,
                                    **row,
                                }
                            )

                    if result.test_layer_features is None:
                        raise RuntimeError("Test intermediates were requested but not collected")
                    layers_by_rep[rep] = result.test_layer_features

                    with profiler.stage(
                        "counterfactual_inference_and_geometry",
                        S=int(S), seed=int(seed), budget=str(budget), representation=rep,
                    ):
                        benign_X = pair_bundle["arrays"][rep]
                        modified_X = modified_bundle["arrays"][rep]
                        combined_X = np.concatenate([benign_X, modified_X], axis=0)
                        combined_prob, combined_z = predict_numpy(
                            result.model,
                            combined_X,
                            device=config.get("device"),
                            batch_size=int(config.get("inference_batch_size", 1024)),
                        )
                        n_pairs = len(benign_X)
                        benign_prob, modified_prob = combined_prob[:n_pairs], combined_prob[n_pairs:]
                        benign_z, modified_z = combined_z[:n_pairs], combined_z[n_pairs:]
                        summary, detail, contrasts = _counterfactual_rows(
                            pairs,
                            rep,
                            benign_prob,
                            modified_prob,
                            benign_z,
                            modified_z,
                            threshold=float(result.metrics["threshold_at_val_fpr_5pct"]),
                            S=S,
                            seed=seed,
                            budget=budget,
                            null_repeats=int(config.get("counterfactual_null_repeats", 200)),
                        )
                    counterfactual_rows.extend(summary)
                    counterfactual_detail_rows.extend(detail)
                    counterfactual_contrast_rows.extend(contrasts)

                    # Cost is an architecture/input-shape property. Repeating a
                    # 100-iteration latency benchmark for every random seed adds
                    # runtime without adding an independent scientific observation.
                    # Measure each architecture condition once and record which seed
                    # supplied the weights used for the timing benchmark.
                    cost_key = (int(S), str(budget), rep, int(X.shape[1]), int(X.shape[-1]))
                    if cost_key not in cost_cache:
                        with profiler.stage(
                            "cost_analysis",
                            S=int(S), seed=int(seed), budget=str(budget), representation=rep,
                        ):
                            cost_cache[cost_key] = analyze_model_cost(
                                result.model,
                                X[test_idx[:1]],
                                representation=rep,
                                latency_warmup=int(config.get("latency_warmup", 10)),
                                latency_repeats=int(config.get("latency_repeats", 30)),
                            )
                        cost_rows.append(
                            {
                                "S": S,
                                "measurement_seed": seed,
                                "model_budget": budget,
                                **cost_cache[cost_key],
                            }
                        )

                with profiler.stage(
                    "layerwise_alignment", S=int(S), seed=int(seed), budget=str(budget)
                ):
                    alignment_for_layers = layerwise_alignment(
                        layers_by_rep,
                        seed=seed,
                        rsa_max_samples=int(config.get("rsa_max_samples", 2500)),
                    )
                for row in alignment_for_layers:
                    layer_rows.append({"S": S, "seed": seed, "model_budget": budget, **row})
                    if row.get("layer") == "latent":
                        alignment_rows.append(
                            {
                                "S": S,
                                "seed": seed,
                                "model_budget": budget,
                                "representation_a": row["representation_a"],
                                "representation_b": row["representation_b"],
                                "linear_cka": row["linear_cka"],
                                "rsa_spearman": row["rsa_spearman"],
                                "procrustes_error": row["procrustes_error"],
                            }
                        )

    pd.DataFrame(geometry_rows).to_csv(out / "geometry_results.csv", index=False)
    pd.DataFrame(alignment_rows).to_csv(out / "alignment_results.csv", index=False)
    pd.DataFrame(layer_rows).to_csv(out / "layerwise_alignment_results.csv", index=False)
    pd.DataFrame(counterfactual_rows).to_csv(out / "counterfactual_summary.csv", index=False)
    pd.DataFrame(counterfactual_detail_rows).to_csv(out / "counterfactual_pairs_results.csv", index=False)
    pd.DataFrame(counterfactual_contrast_rows).to_csv(
        out / "counterfactual_control_contrasts.csv", index=False
    )
    pd.DataFrame(cost_rows).to_csv(out / "cost_results.csv", index=False)
    pd.DataFrame(dimension_rows).to_csv(out / "dimension_sensitivity.csv", index=False)
    pd.DataFrame(input_probe_rows).to_csv(out / "input_probe_results.csv", index=False)
    pd.DataFrame(raw_input_alignment_rows).to_csv(out / "raw_input_alignment_results.csv", index=False)
    pd.DataFrame(initialization_control_rows).to_csv(
        out / "initialization_geometry_control.csv", index=False
    )
    pd.DataFrame(reused_equivalent_training_rows).to_csv(
        out / "reused_equivalent_training.csv", index=False
    )
    pd.concat(selection_audit_rows, ignore_index=True).to_csv(out / "selected_nuisance_balance.csv", index=False)
    pd.DataFrame(selection_baseline_rows).to_csv(
        out / "selected_nuisance_baseline.csv", index=False
    )
    pd.DataFrame(encoding_artifact_rows).to_csv(
        out / "selected_encoding_artifact_baseline.csv", index=False
    )
    pd.concat(collision_frames, ignore_index=True).to_csv(out / "tensor_collision_report.csv", index=False)
    pd.DataFrame(integrity_rows).to_csv(out / "integrity_hashes.csv", index=False)
    (out / "integrity_hashes.json").write_text(
        json.dumps(integrity_rows, indent=2, default=str), encoding="utf-8"
    )
    (out / "config_used.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    profiler.save()
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run corrected NeurReps representation experiments")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    output = run(_load_config(args.config))
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main()
