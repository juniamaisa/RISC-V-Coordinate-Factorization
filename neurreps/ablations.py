from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .audit import distribution_balance
from .cost import analyze_model_cost
from .data import (
    REPRESENTATIONS,
    build_aligned_bundle,
    build_aligned_bundle_cached,
    connected_match_groups,
    exact_length_matched_pairs,
    no_padding_indices,
    no_truncation_indices,
    nuisance_matched_pairs,
)
from .experiment import _cached_joint_nuisance_selections, _model_spec
from .models import build_model
from .geometry import class_geometry, input_linear_probe, linear_probe, validate_intrinsic_dimension_synthetic
from .reproducibility import sha256_strings, write_reproducibility_manifest
from .runtime import RuntimeProfiler
from .training import (
    configure_torch_determinism,
    binary_metrics, grouped_split, load_train_result_if_complete, predict_numpy,
    save_train_result, train_representation_model,
)

ABLATION_VARIANTS = (
    "original_padding",
    "no_padding_samples",
    "exact_real_length_matched",
    "explicit_mask_channel",
    "randomized_padding_positions",
    "zero_cue_removed",
    "eval_no_truncation",
    "mask_aware_encoder",
)


def append_explicit_mask_channel(X: np.ndarray, mask: np.ndarray, representation: str) -> np.ndarray:
    X, mask = np.asarray(X), np.asarray(mask)
    if mask.shape != (len(X), 1, X.shape[-1]):
        raise ValueError(f"Mask {mask.shape} incompatible with X {X.shape}")
    scale = 1 if representation.upper() == "32C" else 255
    return np.concatenate([X, (mask > 0).astype(X.dtype) * scale], axis=1)


def _slot_view(X: np.ndarray, mask: np.ndarray, representation: str) -> tuple[np.ndarray, np.ndarray, callable]:
    """Expose sequence positions as complete instruction slots for every view."""
    rep = representation.upper()
    if rep == "SC":
        if X.shape[-1] % 4:
            raise ValueError("Aligned SC length must be divisible by four")
        slots = X.reshape(len(X), 1, -1, 4).transpose(0, 2, 1, 3).reshape(len(X), -1, 4)
        slot_mask = mask.reshape(len(X), 1, -1, 4)[:, 0, :, 0]

        def restore(values, masks):
            restored = values.reshape(len(X), -1, 1, 4).transpose(0, 2, 1, 3).reshape(X.shape)
            restored_mask = np.repeat(masks[:, None, :, None], 4, axis=3).reshape(mask.shape)
            return restored, restored_mask

        return slots, slot_mask, restore
    slots = X.transpose(0, 2, 1)
    slot_mask = mask[:, 0]

    def restore(values, masks):
        return values.transpose(0, 2, 1), masks[:, None]

    return slots, slot_mask, restore


def randomize_padding_positions(
    X: np.ndarray,
    mask: np.ndarray,
    *,
    representation: str | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Move complete padded instruction slots; valid-slot order is preserved."""
    if representation is None:
        X, mask = np.asarray(X), np.asarray(mask)
        rng = np.random.default_rng(seed)
        out, out_mask = np.empty_like(X), np.empty_like(mask)
        positions = np.arange(X.shape[-1])
        for i in range(len(X)):
            valid = np.flatnonzero(mask[i, 0] > 0)
            padded = np.flatnonzero(mask[i, 0] == 0)
            if not len(valid) or not len(padded):
                out[i], out_mask[i] = X[i], mask[i]
                continue
            target_valid = np.sort(rng.choice(positions, size=len(valid), replace=False))
            target_pad = np.setdiff1d(positions, target_valid, assume_unique=True)
            out[i, :, target_valid] = X[i, :, valid]
            out[i, :, target_pad] = X[i, :, padded]
            out_mask[i] = 0
            out_mask[i, 0, target_valid] = 1
        return out, out_mask
    slots, slot_mask, restore = _slot_view(np.asarray(X), np.asarray(mask), representation)
    rng = np.random.default_rng(seed)
    out, out_mask = np.empty_like(slots), np.empty_like(slot_mask)
    positions = np.arange(slots.shape[1])
    for i in range(len(slots)):
        valid = np.flatnonzero(slot_mask[i] > 0)
        padded = np.flatnonzero(slot_mask[i] == 0)
        if not len(valid) or not len(padded):
            out[i], out_mask[i] = slots[i], slot_mask[i]
            continue
        target_valid = np.sort(rng.choice(positions, size=len(valid), replace=False))
        target_pad = np.setdiff1d(positions, target_valid, assume_unique=True)
        out[i, target_valid] = slots[i, valid]
        out[i, target_pad] = slots[i, padded]
        out_mask[i] = 0
        out_mask[i, target_valid] = 1
    return restore(out, out_mask)


def replace_padding_with_training_content(
    X: np.ndarray,
    mask: np.ndarray,
    train_indices: np.ndarray,
    *,
    representation: str | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Replace complete PAD slots with train-only valid instruction slots."""
    if representation is None:
        X, mask = np.asarray(X), np.asarray(mask)
        train_indices = np.asarray(train_indices, dtype=np.int64)
        train_tokens = X[train_indices].transpose(0, 2, 1)
        train_valid = mask[train_indices, 0].reshape(-1) > 0
        pool = train_tokens.reshape(-1, X.shape[1])[train_valid]
        if len(pool) == 0:
            raise RuntimeError("Training split has no valid tokens for PAD replacement")
        rng = np.random.default_rng(seed)
        out = X.copy()
        for i in range(len(out)):
            padded = np.flatnonzero(mask[i, 0] == 0)
            if len(padded):
                out[i][:, padded] = pool[rng.integers(0, len(pool), size=len(padded))].T
        return out
    slots, slot_mask, restore = _slot_view(np.asarray(X), np.asarray(mask), representation)
    train_indices = np.asarray(train_indices, dtype=np.int64)
    pool = slots[train_indices][slot_mask[train_indices] > 0]
    if len(pool) == 0:
        raise RuntimeError("Training split has no valid tokens for PAD replacement")
    rng = np.random.default_rng(seed)
    out = slots.copy()
    for i in range(len(out)):
        padded = np.flatnonzero(slot_mask[i] == 0)
        if len(padded):
            out[i, padded] = pool[rng.integers(0, len(pool), size=len(padded))]
    restored, _ = restore(out, slot_mask)
    return restored


def _select_base(bundle, config: Mapping) -> tuple[np.ndarray, np.ndarray | None]:
    candidate = np.arange(len(bundle.labels), dtype=np.int64)
    pairs = None
    if config.get("nuisance_match", True):
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
    return candidate, pairs


def _variant_selection(
    bundle,
    base_selected: np.ndarray,
    base_pairs: np.ndarray | None,
    variant: str,
    reps: Sequence[str],
    config: Mapping,
) -> tuple[np.ndarray, np.ndarray | None]:
    seed = int(config.get("data_seed", 42))
    caliper = float(config.get("matching_caliper", 0.35))
    if variant == "no_padding_samples":
        candidate = no_padding_indices(bundle.masks, representations=reps, candidate_indices=base_selected)
        # Critical correction: matching is re-run after filtering. We never
        # balance individual rows or retain half of an earlier nuisance pair.
        pairs = nuisance_matched_pairs(
            bundle.metadata,
            bundle.labels,
            candidate_indices=candidate,
            seed=seed,
            caliper=caliper,
            features=tuple(config.get("matching_features", [
                "raw_instruction_count", "raw_byte_count", "compressed_ratio",
                "raw_zero_byte_ratio", "pattern_0013_ratio", "padding_rate",
                "preserved_ratio", "trunc_removed",
            ])),
        ) if config.get("nuisance_match", True) else exact_length_matched_pairs(
            bundle.metadata, bundle.labels, candidate_indices=candidate, seed=seed
        )
        return np.asarray(sorted(pairs.reshape(-1).tolist()), dtype=np.int64), pairs
    if variant == "exact_real_length_matched":
        pairs = exact_length_matched_pairs(
            bundle.metadata,
            bundle.labels,
            candidate_indices=base_selected,
            seed=seed,
            fields=tuple(config.get("exact_length_fields", ["slot_real_length"])),
        )
        return np.asarray(sorted(pairs.reshape(-1).tolist()), dtype=np.int64), pairs
    # Original/mask/randomized/value variants preserve the exact base match.
    # Re-running a greedy matcher on an already matched set can change partners
    # and wastes substantial time; it is not a scientific ablation.
    return np.asarray(base_selected, dtype=np.int64), (
        None if base_pairs is None else np.asarray(base_pairs, dtype=np.int64)
    )


def _split(bundle, selected: np.ndarray, pairs: np.ndarray | None, reps: Sequence[str], seed: int):
    y = bundle.labels[selected]
    groups = connected_match_groups(
        bundle.sample_id,
        selected,
        pairs,
        duplicate_keys=[bundle.tensor_hashes[rep] for rep in reps],
    )
    train, val, test = grouped_split(y, groups, seed)
    return y, groups, {"train": train, "val": val, "test": test}


def _prepare_variant_input(bundle, selected, rep, variant, seed, budget, config):
    X = bundle.arrays[rep][selected].copy()
    mask = bundle.masks[rep][selected].copy()
    model_mask = None
    mask_aware = False
    if variant == "explicit_mask_channel":
        X = append_explicit_mask_channel(X, mask, rep)
    elif variant == "randomized_padding_positions":
        X, mask = randomize_padding_positions(
            X, mask, representation=rep,
            seed=int(config.get("padding_randomization_seed", 991)) + seed,
        )
    elif variant == "zero_cue_removed":
        # The train split is needed for the replacement pool and is applied by
        # the caller after the grouped split has been fixed.
        pass
    elif variant == "mask_aware_encoder":
        model_mask, mask_aware = mask, True
    spec = _model_spec(rep, budget, config, input_channels=X.shape[1])
    return X, mask, model_mask, mask_aware, spec


def _train_prepared_variant(X, mask, model_mask, mask_aware, y, groups, fixed, rep, seed, spec, config):
    return train_representation_model(
        X, y, groups, rep, seed=seed,
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
        width=spec["width"], latent_dim=spec["latent_dim"], depth=spec["depth"],
        device=config.get("device"), fixed_indices=fixed, mask=model_mask,
        mask_aware=mask_aware, input_channels=X.shape[1],
        bootstrap_replicates=int(config.get("bootstrap_replicates", 500)),
        deterministic_algorithms=bool(config.get("deterministic_torch", True)),
    )

TRAINING_EQUIVALENCE_KEYS = (
    "epochs", "max_epochs", "epoch_extension", "auto_extend_epochs",
    "batch_size", "eval_batch_size", "patience", "min_epochs", "lr", "weight_decay",
    "scheduler_patience", "scheduler_factor", "min_lr",
    "convergence_tail_epochs", "require_convergence", "early_stopping_min_delta",
    "bootstrap_replicates", "width", "latent_dim", "depth", "device",
    "deterministic_torch",
)


def _training_config_signature(config: Mapping) -> dict:
    return {key: config.get(key) for key in TRAINING_EQUIVALENCE_KEYS}


def _reuse_main_original_if_compatible(
    *,
    config: Mapping,
    main_config: Mapping | None,
    main_results_dir: Path | None,
    S: int, seed: int, budget: str, rep: str, spec: dict, input_channels: int,
    selected_tensor_sha256: str, split_sha256: dict[str, str], code_sha256: str,
):
    if not bool(config.get("reuse_main_original_padding", True)):
        return None
    if main_results_dir is None or main_config is None:
        return None
    if _training_config_signature(config) != _training_config_signature(main_config):
        return None
    source_dir = main_results_dir / f"S{int(S)}" / f"seed{int(seed)}" / str(budget)
    fp_path = source_dir / f"fingerprint_{rep}_seed{int(seed)}.json"
    if not fp_path.exists():
        return None
    try:
        stored = json.loads(fp_path.read_text(encoding="utf-8")).get("fingerprint", {})
    except (OSError, json.JSONDecodeError):
        return None
    required = {
        "schema": "neurreps-train-v2",
        "code_sha256": code_sha256,
        "S": int(S),
        "seed": int(seed),
        "model_budget": str(budget),
        "representation": rep,
        "model_spec": spec,
        "selected_tensor_sha256": selected_tensor_sha256,
        "split_sha256": split_sha256,
    }
    if any(stored.get(key) != value for key, value in required.items()):
        return None
    model = build_model(
        rep, width=spec["width"], latent_dim=spec["latent_dim"], depth=spec["depth"],
        input_channels=input_channels, mask_aware=False,
    )
    return load_train_result_if_complete(
        source_dir, rep, seed, model=model, expected_fingerprint=stored,
        require_intermediates=False, device=config.get("device"),
    )


def _geometry(result, X, y, fixed, seed):
    metrics = dict(result.metrics)
    metrics.pop("history", None)
    metrics.update(class_geometry(result.latents["test"], result.labels["test"], seed=seed))
    metrics.update(
        linear_probe(
            result.latents["train"],
            result.labels["train"],
            result.latents["test"],
            result.labels["test"],
            seed=seed,
        )
    )
    metrics.update(
        input_linear_probe(
            X[fixed["train"]],
            result.labels["train"],
            X[fixed["test"]],
            result.labels["test"],
            seed=seed,
        )
    )
    return metrics


def _no_truncation_row(bundle, selected, y, fixed, rep, seed, budget, original_result, original_X, model_mask, config):
    test_global = selected[fixed["test"]]
    allowed = set(no_truncation_indices(bundle.metadata).tolist())
    keep = np.asarray([i for i, global_i in enumerate(test_global) if int(global_i) in allowed], dtype=np.int64)
    row = {
        "variant": "eval_no_truncation",
        "representation": rep,
        "seed": seed,
        "model_budget": budget,
        "selected_samples": len(selected),
        "test_no_truncation_samples": len(keep),
        "training_variant": "original_padding",
    }
    if not len(keep):
        row["status"] = "empty"
        return row
    test_local = fixed["test"][keep]
    prob, z = predict_numpy(
        original_result.model,
        original_X[test_local],
        mask=model_mask[test_local] if model_mask is not None else None,
        device=config.get("device"),
    )
    yy = y[test_local]
    row.update(binary_metrics(yy, prob, prefix="test_"))
    if set(np.unique(yy)) == {0, 1}:
        row.update(class_geometry(z, yy, seed=seed))
    for pct in (1, 5):
        threshold = original_result.metrics.get(f"threshold_at_val_fpr_{pct}pct")
        if threshold is not None:
            pred = prob >= float(threshold)
            row[f"test_fnr_at_val_fpr_{pct}pct"] = float(np.mean(~pred[yy == 1]))
            row[f"test_fpr_at_val_fpr_{pct}pct"] = float(np.mean(pred[yy == 0]))
    row["status"] = "ok"
    return row


def run(config: dict) -> Path:
    configure_torch_determinism(bool(config.get("deterministic_torch", True)))
    out = Path(config.get("output_dir", "results/neurreps_padding_ablations"))
    out.mkdir(parents=True, exist_ok=True)
    profiler = RuntimeProfiler(out)
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
        validation_rows = validate_intrinsic_dimension_synthetic(
            dimensions=config.get("intrinsic_validation_dimensions", [2, 5, 10, 20]),
            n_samples=int(config.get("intrinsic_validation_samples", 4000)),
            repeats=int(config.get("intrinsic_validation_repeats", 3)),
            relative_tolerance=float(config.get("intrinsic_validation_relative_tolerance", 0.30)),
            seed=int(config.get("intrinsic_validation_seed", 1729)),
        )
        validation_frame = pd.DataFrame(validation_rows)
        validation_frame.to_csv(out / "intrinsic_dimension_synthetic_validation.csv", index=False)
        if bool(config.get("require_intrinsic_validation", True)) and not bool(validation_frame["passed"].all()):
            raise RuntimeError(
                "Intrinsic-dimension synthetic validation failed; aborting experiment."
            )
    reps = [str(rep).upper() for rep in config.get("representations", REPRESENTATIONS)]
    seeds = [int(seed) for seed in config.get("seeds", [11, 23, 37, 53, 71])]
    S_values = [int(S) for S in config.get("S_values", [32])]
    variants = list(config.get("padding_ablation_variants", ABLATION_VARIANTS))
    budgets = list(config.get("model_budgets", ["fixed_core", "matched_total"]))
    unknown = sorted(set(variants) - set(ABLATION_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    train_variants = [variant for variant in variants if variant != "eval_no_truncation"]
    rows, count_rows = [], []
    cost_cache: dict[tuple, dict] = {}
    reused_rows: list[dict] = []
    main_results_dir = Path(config["main_results_dir"]) if config.get("main_results_dir") else None
    main_config = None
    if main_results_dir is not None and (main_results_dir / "config_used.json").exists():
        main_config = json.loads((main_results_dir / "config_used.json").read_text(encoding="utf-8"))

    bundles = {}
    for S in S_values:
        with profiler.stage("aligned_bundle", S=int(S)):
            bundles[int(S)] = build_aligned_bundle_cached(
                config["mode_folder"],
                config["software"],
                S,
                max_per_class=config.get("max_per_class"),
                seed=int(config.get("data_seed", 42)),
                use_cache=bool(config.get("cache_aligned_bundles", True)),
                cache_dir=config.get("aligned_bundle_cache_dir"),
            )
    if bool(config.get("common_identities_across_S", True)):
        with profiler.stage("ablation_joint_base_matching"):
            base_selections, base_pairs_by_s, _ = _cached_joint_nuisance_selections(bundles, config)
    else:
        base_selections, base_pairs_by_s = {}, {}
        for S, bundle in bundles.items():
            base_selections[S], base_pairs_by_s[S] = _select_base(bundle, config)

    for S in S_values:
        bundle = bundles[int(S)]
        base_selected, base_pairs = base_selections[int(S)], base_pairs_by_s[int(S)]
        selection_cache = {}
        needed = set(train_variants + (["original_padding"] if "eval_no_truncation" in variants else []))
        for variant in needed:
            selected, pairs = _variant_selection(
                bundle, base_selected, base_pairs, variant, reps, config
            )
            selection_cache[variant] = selected, pairs
            matching_features = tuple(config.get("matching_features", [
                "raw_instruction_count", "raw_byte_count", "compressed_ratio",
                "raw_zero_byte_ratio", "pattern_0013_ratio", "padding_rate",
                "preserved_ratio", "trunc_removed",
            ]))
            balance = distribution_balance(
                bundle.metadata.iloc[selected].reset_index(drop=True),
                features=matching_features,
            )
            max_abs_smd = float(balance["abs_smd"].max()) if len(balance) else float("nan")
            # Variants whose selection is nuisance-matched must satisfy the
            # same balance gate as the main experiment. The exact-length-only
            # ablation intentionally changes the matching criterion and is
            # reported rather than rejected here.
            nuisance_matched_variant = bool(
                config.get("nuisance_match", True)
                and variant != "exact_real_length_matched"
            )
            if (
                nuisance_matched_variant
                and bool(config.get("matching_require_balance", True))
                and np.isfinite(max_abs_smd)
                and max_abs_smd > float(config.get("matching_max_abs_smd", 0.10))
            ):
                raise RuntimeError(
                    f"Ablation nuisance balance gate failed for S={S}, variant={variant}: "
                    f"max |SMD|={max_abs_smd:.4f}"
                )
            count_rows.append(
                {
                    "S": S,
                    "variant": variant,
                    "samples": len(selected),
                    "pairs": len(pairs) if pairs is not None else np.nan,
                    "benign": int((bundle.labels[selected] == 0).sum()),
                    "cra_like": int((bundle.labels[selected] == 1).sum()),
                    "mean_padding_rate": float(bundle.metadata.iloc[selected].padding_rate.mean()),
                    "no_truncation_fraction": float(bundle.metadata.iloc[selected].no_truncation_all.mean()),
                    "max_abs_smd_matching_features": max_abs_smd,
                }
            )
        for seed in seeds:
            split_cache = {
                variant: _split(bundle, selected, pairs, reps, seed)
                for variant, (selected, pairs) in selection_cache.items()
            }
            for budget in budgets:
                originals = {}
                for variant in train_variants:
                    selected, _ = selection_cache[variant]
                    y, groups, fixed = split_cache[variant]
                    for rep in reps:
                        X, mask, model_mask, mask_aware, spec = _prepare_variant_input(
                            bundle, selected, rep, variant, seed, budget, config
                        )
                        if variant == "zero_cue_removed":
                            X = replace_padding_with_training_content(
                                X, mask, fixed["train"], representation=rep,
                                seed=int(config.get("zero_cue_seed", 1231)) + seed,
                            )
                        model_dir = out / f"S{S}" / f"seed{seed}" / budget / variant
                        fingerprint = {
                            "schema": "neurreps-ablation-train-v1",
                            "code_sha256": reproducibility_manifest["code"]["code_sha256"],
                            "config_sha256": reproducibility_manifest["config_sha256"],
                            "S": int(S), "seed": int(seed), "model_budget": str(budget),
                            "variant": str(variant), "representation": rep, "model_spec": spec,
                            "selected_tensor_sha256": sha256_strings(
                                bundle.tensor_hashes[rep][selected].astype(str).tolist()
                            ),
                            "split_sha256": {
                                name: sha256_strings(
                                    sorted(bundle.sample_id[selected[idx]].astype(str).tolist())
                                )
                                for name, idx in fixed.items()
                            },
                        }
                        result = None
                        if variant == "original_padding":
                            with profiler.stage(
                                "ablation_main_model_reuse_lookup", S=int(S), seed=int(seed),
                                budget=str(budget), variant=str(variant), representation=rep,
                            ):
                                result = _reuse_main_original_if_compatible(
                                    config=config, main_config=main_config,
                                    main_results_dir=main_results_dir, S=int(S), seed=int(seed),
                                    budget=str(budget), rep=rep, spec=spec, input_channels=int(X.shape[1]),
                                    selected_tensor_sha256=fingerprint["selected_tensor_sha256"],
                                    split_sha256=fingerprint["split_sha256"],
                                    code_sha256=reproducibility_manifest["code"]["code_sha256"],
                                )
                            if result is not None:
                                reused_rows.append({
                                    "S": int(S), "seed": int(seed), "model_budget": str(budget),
                                    "variant": str(variant), "representation": rep,
                                    "source": "main_experiment",
                                })
                                print(
                                    f"[CACHE] Reusing main original-padding model: S={S} seed={seed} "
                                    f"budget={budget} rep={rep}"
                                )
                        if result is None and bool(config.get("resume_completed_models", True)):
                            resume_model = build_model(
                                rep, width=spec["width"], latent_dim=spec["latent_dim"],
                                depth=spec["depth"], input_channels=X.shape[1],
                                mask_aware=mask_aware,
                            )
                            with profiler.stage(
                                "ablation_model_resume_lookup", S=int(S), seed=int(seed),
                                budget=str(budget), variant=str(variant), representation=rep,
                            ):
                                result = load_train_result_if_complete(
                                    model_dir, rep, seed, model=resume_model,
                                    expected_fingerprint=fingerprint, require_intermediates=False,
                                    device=config.get("device"),
                                )
                            if result is not None:
                                print(
                                    f"[CACHE] Completed ablation model loaded: S={S} seed={seed} "
                                    f"budget={budget} variant={variant} rep={rep}"
                                )
                        if result is None:
                            with profiler.stage(
                                "ablation_model_training", S=int(S), seed=int(seed),
                                budget=str(budget), variant=str(variant), representation=rep,
                            ):
                                result = _train_prepared_variant(
                                    X, mask, model_mask, mask_aware, y, groups, fixed,
                                    rep, seed, spec, config,
                                )
                            with profiler.stage(
                                "ablation_model_artifact_serialization", S=int(S), seed=int(seed),
                                budget=str(budget), variant=str(variant), representation=rep,
                            ):
                                save_train_result(
                                    result, model_dir, rep, seed,
                                    bundle.sample_id[selected], bundle.sample_hash[selected], y,
                                    compress=bool(config.get("compress_latent_artifacts", False)),
                                    fingerprint=fingerprint,
                                )
                        metrics = _geometry(result, X, y, fixed, seed)
                        cost_key = (
                            int(S), str(budget), rep, int(X.shape[1]), int(X.shape[-1]),
                            bool(model_mask is not None),
                        )
                        if cost_key not in cost_cache:
                            cost_cache[cost_key] = analyze_model_cost(
                                result.model,
                                X[fixed["test"][:1]],
                                mask=model_mask[fixed["test"][:1]] if model_mask is not None else None,
                                representation=rep,
                                latency_warmup=int(config.get("ablation_latency_warmup", 3)),
                                latency_repeats=int(config.get("ablation_latency_repeats", 10)),
                            )
                        metrics.update(cost_cache[cost_key])
                        metrics.update(
                            {
                                "variant": variant,
                                "S": S,
                                "model_budget": budget,
                                "selected_samples": len(selected),
                                "mean_padding_rate_selected": float(1.0 - mask.mean()),
                                "status": "ok",
                            }
                        )
                        rows.append(metrics)
                        if variant == "original_padding":
                            originals[rep] = result, X, model_mask, selected, y, fixed
                if "eval_no_truncation" in variants:
                    if not originals:
                        raise RuntimeError("eval_no_truncation requires original_padding")
                    for rep, values in originals.items():
                        result, X, model_mask, selected, y, fixed = values
                        row = _no_truncation_row(
                            bundle,
                            selected,
                            y,
                            fixed,
                            rep,
                            seed,
                            budget,
                            result,
                            X,
                            model_mask,
                            config,
                        )
                        row["S"] = S
                        rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(out / "padding_ablation_results.csv", index=False)
    pd.DataFrame(count_rows).drop_duplicates().to_csv(out / "padding_ablation_counts.csv", index=False)
    pd.DataFrame(reused_rows).to_csv(out / "reused_main_models.csv", index=False)
    numeric = results.select_dtypes(include=[np.number]).columns.difference(["seed"])
    summary = results.groupby(["variant", "representation", "S", "model_budget"], dropna=False)[list(numeric)].agg(["mean", "std"])
    summary.columns = ["_".join(column).strip("_") for column in summary.columns]
    summary.reset_index().to_csv(out / "padding_ablation_summary.csv", index=False)
    (out / "config_used.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    profiler.save()
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run corrected padding/length ablations")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    output = run(json.loads(Path(args.config).read_text(encoding="utf-8")))
    print(f"Padding ablation results saved to {output}")


if __name__ == "__main__":
    main()
