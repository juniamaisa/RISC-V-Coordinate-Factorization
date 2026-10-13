from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import (
    AlignedBundle,
    build_aligned_bundle,
    connected_match_groups,
    nuisance_matched_pairs,
    save_aligned_bundle,
)
from .training import grouped_split

DEFAULT_NUISANCE_FEATURES = [
    "raw_instruction_count",
    "raw_byte_count",
    "compressed_ratio",
    "raw_zero_byte_ratio",
    "pattern_0013_ratio",
    "padding_rate",
    "preserved_ratio",
    "trunc_removed",
]


def nuisance_only_baseline(
    meta: pd.DataFrame,
    *,
    groups: np.ndarray,
    seeds: Sequence[int] = (11, 23, 37, 53, 71),
    features: Sequence[str] = DEFAULT_NUISANCE_FEATURES,
) -> dict:
    """Repeated group-disjoint nuisance-only classification."""
    X = meta.loc[:, list(features)].astype(float).replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).to_numpy(dtype=float)
    y = meta["label"].to_numpy(dtype=int)
    rows = []
    for seed in seeds:
        train, _, test = grouped_split(y, groups, int(seed), train_ratio=0.8, val_ratio=0.1)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", random_state=int(seed)),
        )
        model.fit(X[train], y[train])
        prob = model.predict_proba(X[test])[:, 1]
        rows.append(
            {
                "seed": int(seed),
                "pr_auc": float(average_precision_score(y[test], prob)),
                "roc_auc": float(roc_auc_score(y[test], prob)),
                "test_samples": int(len(test)),
            }
        )
    frame = pd.DataFrame(rows)
    return {
        "runs": rows,
        "pr_auc_mean": float(frame.pr_auc.mean()),
        "pr_auc_std": float(frame.pr_auc.std(ddof=1)),
        "roc_auc_mean": float(frame.roc_auc.mean()),
        "roc_auc_std": float(frame.roc_auc.std(ddof=1)),
    }


def encoding_artifact_baseline(
    arrays: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    seeds: Sequence[int] = (11, 23, 37, 53, 71),
) -> list[dict]:
    """Test whether simple marginal tensor statistics predict the label.

    This is a shortcut audit, not a semantic classifier. It intentionally uses
    no ordering-sensitive features: global mean/std/zero rate, valid fraction,
    and per-channel means. A strong score signals an encoding-level marginal
    cue that must be reported and investigated.
    """
    y = np.asarray(labels).astype(int)
    groups = np.asarray(groups)
    rows: list[dict] = []
    for rep in sorted(arrays):
        X = np.asarray(arrays[rep], dtype=np.float32)
        M = np.asarray(masks[rep], dtype=np.float32)
        flat = X.reshape(len(X), -1)
        features = [
            flat.mean(axis=1, keepdims=True),
            flat.std(axis=1, keepdims=True),
            (flat == 0).mean(axis=1, keepdims=True),
            M.reshape(len(M), -1).mean(axis=1, keepdims=True),
            X.mean(axis=2),
        ]
        stats = np.concatenate(features, axis=1).astype(np.float64)
        for seed in seeds:
            train, _, test = grouped_split(y, groups, int(seed), train_ratio=0.8, val_ratio=0.1)
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=3000, class_weight="balanced", random_state=int(seed)),
            )
            model.fit(stats[train], y[train])
            prob = model.predict_proba(stats[test])[:, 1]
            rows.append({
                "representation": rep,
                "seed": int(seed),
                "features": int(stats.shape[1]),
                "test_samples": int(len(test)),
                "pr_auc": float(average_precision_score(y[test], prob)),
                "roc_auc": float(roc_auc_score(y[test], prob)),
            })
    return rows


def distribution_balance(meta: pd.DataFrame, features: Sequence[str] = DEFAULT_NUISANCE_FEATURES) -> pd.DataFrame:
    rows = []
    y = meta["label"].to_numpy(dtype=int)
    for feature in features:
        a = pd.to_numeric(meta.loc[y == 0, feature], errors="coerce").dropna().to_numpy(dtype=float)
        b = pd.to_numeric(meta.loc[y == 1, feature], errors="coerce").dropna().to_numpy(dtype=float)
        pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
        smd = (np.mean(b) - np.mean(a)) / pooled if pooled > 0 else 0.0
        ks = ks_2samp(a, b, method="auto")
        rows.append(
            {
                "feature": feature,
                "benign_mean": float(np.mean(a)),
                "cra_like_mean": float(np.mean(b)),
                "smd": float(smd),
                "abs_smd": float(abs(smd)),
                "ks_statistic": float(ks.statistic),
                "ks_p_value": float(ks.pvalue),
                "wasserstein": float(wasserstein_distance(a, b)),
                "median_difference": float(np.median(b) - np.median(a)),
                "q10_difference": float(np.quantile(b, 0.1) - np.quantile(a, 0.1)),
                "q90_difference": float(np.quantile(b, 0.9) - np.quantile(a, 0.9)),
            }
        )
    return pd.DataFrame(rows)


def audit_bundle(
    bundle: AlignedBundle,
    out_dir: str | Path,
    *,
    seeds: Sequence[int] = (11, 23, 37, 53, 71),
    matching_seed: int = 42,
    matching_caliper: float = 0.35,
    candidate_indices: np.ndarray | None = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = (
        np.arange(len(bundle.labels), dtype=np.int64)
        if candidate_indices is None
        else np.asarray(candidate_indices, dtype=np.int64)
    )
    meta = bundle.metadata.iloc[selected].reset_index(drop=True)
    duplicate_keys = [bundle.tensor_hashes[rep][selected] for rep in sorted(bundle.tensor_hashes)]
    raw_groups = connected_match_groups(
        bundle.sample_id,
        selected,
        pairs=None,
        duplicate_keys=[bundle.tensor_hashes[rep] for rep in sorted(bundle.tensor_hashes)],
    )
    raw_baseline = nuisance_only_baseline(meta, groups=raw_groups, seeds=seeds)
    raw_balance = distribution_balance(meta)
    raw_balance["subset"] = "raw"

    global_pairs = nuisance_matched_pairs(
        bundle.metadata,
        bundle.labels,
        candidate_indices=selected,
        seed=matching_seed,
        caliper=matching_caliper,
    )
    matched_global = np.asarray(sorted(global_pairs.reshape(-1).tolist()), dtype=np.int64)
    matched_meta = bundle.metadata.iloc[matched_global].reset_index(drop=True)
    matched_groups = connected_match_groups(
        bundle.sample_id,
        matched_global,
        global_pairs,
        duplicate_keys=[bundle.tensor_hashes[rep] for rep in sorted(bundle.tensor_hashes)],
    )
    matched_baseline = nuisance_only_baseline(matched_meta, groups=matched_groups, seeds=seeds)
    matched_balance = distribution_balance(matched_meta)
    matched_balance["subset"] = "matched"

    balance = pd.concat([raw_balance, matched_balance], ignore_index=True)
    balance.to_csv(out_dir / "nuisance_balance.csv", index=False)
    raw_balance.to_csv(out_dir / "nuisance_balance_raw.csv", index=False)
    matched_balance.to_csv(out_dir / "nuisance_balance_matched.csv", index=False)
    raw_runs = pd.DataFrame(raw_baseline["runs"])
    raw_runs["subset"] = "raw"
    matched_runs = pd.DataFrame(matched_baseline["runs"])
    matched_runs["subset"] = "matched"
    pd.concat([raw_runs, matched_runs], ignore_index=True).to_csv(
        out_dir / "nuisance_baseline_repeated.csv", index=False
    )
    pd.DataFrame(global_pairs, columns=["benign_index", "cra_like_index"]).to_csv(
        out_dir / "nuisance_matched_pairs.csv", index=False
    )
    bundle.collision_report.to_csv(out_dir / "tensor_collision_report.csv", index=False)
    class_summary = (
        meta.groupby("label")
        .agg(
            samples=("sample_id", "count"),
            raw_instruction_mean=("raw_instruction_count", "mean"),
            raw_bytes_mean=("raw_byte_count", "mean"),
            compressed_ratio_mean=("compressed_ratio", "mean"),
            padding_rate_mean=("padding_rate", "mean"),
            preserved_ratio_mean=("preserved_ratio", "mean"),
            trunc_removed_mean=("trunc_removed", "mean"),
        )
        .reset_index()
    )
    class_summary.to_csv(out_dir / "class_audit.csv", index=False)
    summary = {
        "software": bundle.software,
        "S": int(bundle.S),
        "raw_samples": int(len(selected)),
        "matched_samples": int(len(matched_global)),
        "matched_pairs": int(len(global_pairs)),
        "nuisance_only_raw": raw_baseline,
        "nuisance_only_matched": matched_baseline,
        "max_abs_smd_raw": float(raw_balance.abs_smd.max()),
        "max_abs_smd_matched": float(matched_balance.abs_smd.max()),
        "post_tensor_collision_rows_removed": int(
            bundle.collision_report["rows_removed_union"].max()
            if len(bundle.collision_report)
            else 0
        ),
    }
    (out_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit aligned SC/4C/32C data")
    parser.add_argument("--mode-folder", required=True)
    parser.add_argument("--software", required=True)
    parser.add_argument("--S", type=int, default=16)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--matching-seed", type=int, default=42)
    parser.add_argument("--matching-caliper", type=float, default=0.35)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37, 53, 71])
    parser.add_argument("--out", default="results/neurreps_audit")
    parser.add_argument("--save-bundle", action="store_true")
    args = parser.parse_args(argv)
    bundle = build_aligned_bundle(
        args.mode_folder,
        args.software,
        args.S,
        max_per_class=args.max_per_class,
        seed=args.matching_seed,
    )
    out = Path(args.out)
    if args.save_bundle:
        save_aligned_bundle(bundle, out / "aligned_data")
    summary = audit_bundle(
        bundle,
        out,
        seeds=args.seeds,
        matching_seed=args.matching_seed,
        matching_caliper=args.matching_caliper,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
