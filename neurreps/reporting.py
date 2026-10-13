from __future__ import annotations

import argparse
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .statistics import (
    bootstrap_mean_ci,
    holm_adjust,
    paired_permutation_test,
)

PRIMARY_METRICS = [
    "test_pr_auc",
    "test_fnr_at_val_fpr_5pct",
    "fisher_ratio",
    "participation_ratio",
    "effective_rank",
    "mle_id",
    "linear_probe_pr_auc",
    "input_probe_pr_auc",
]


def _fmt(value: float, digits: int = 3) -> str:
    return "--" if value is None or not np.isfinite(value) else f"{float(value):.{digits}f}"


def aggregate_geometry(df: pd.DataFrame, *, seed: int = 42) -> pd.DataFrame:
    group_cols = [column for column in ["S", "model_budget", "representation"] if column in df.columns]
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys))
        row["runs"] = int(len(group))
        for metric in PRIMARY_METRICS:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy()
            mean, low, high = bootstrap_mean_ci(values, n_boot=5000, seed=seed)
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = mean
            row[f"{metric}_seed_bootstrap_low"] = low
            row[f"{metric}_seed_bootstrap_high"] = high
            row[f"{metric}_seed_std"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
            row[f"{metric}_seed_min"] = float(np.min(finite)) if len(finite) else np.nan
            row[f"{metric}_seed_max"] = float(np.max(finite)) if len(finite) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_representation_tests(df: pd.DataFrame, *, seed: int = 42) -> pd.DataFrame:
    if "representation" not in df:
        return pd.DataFrame()
    condition_cols = [column for column in ["S", "model_budget"] if column in df.columns]
    rows = []
    reps = sorted(df.representation.dropna().unique())
    for condition, part in df.groupby(condition_cols, dropna=False):
        condition = condition if isinstance(condition, tuple) else (condition,)
        condition_values = dict(zip(condition_cols, condition))
        for metric in PRIMARY_METRICS:
            if metric not in part:
                continue
            wide = part.pivot_table(index=["seed"], columns="representation", values=metric, aggfunc="first")
            for a, b in itertools.combinations(reps, 2):
                if a not in wide or b not in wide:
                    continue
                pair = wide[[a, b]].dropna().reset_index()
                if pair.empty:
                    continue
                test = paired_permutation_test(pair[a], pair[b], seed=seed)
                _, diff_low, diff_high = bootstrap_mean_ci(
                    pair[a].to_numpy(dtype=float) - pair[b].to_numpy(dtype=float),
                    n_boot=5000, seed=seed,
                )
                rows.append(
                    {
                        **condition_values,
                        "metric": metric,
                        "representation_a": a,
                        "representation_b": b,
                        "difference_seed_bootstrap_low": diff_low,
                        "difference_seed_bootstrap_high": diff_high,
                        **test,
                    }
                )
    output = pd.DataFrame(rows)
    if not output.empty:
        output["inference_unit"] = "random_seed_run"
        output["inference_scope"] = "optimization_stability_not_dataset_generalization"
        output["p_holm"] = np.nan
        family_cols = [column for column in ["model_budget", "metric"] if column in output]
        for _, idx in output.groupby(family_cols, dropna=False).groups.items():
            output.loc[idx, "p_holm"] = holm_adjust(output.loc[idx, "p_value"].to_numpy())
    return output



def context_length_tests(df: pd.DataFrame, *, seed: int = 42) -> pd.DataFrame:
    """Compare context budgets with one paired observation per seed/fold.

    This function treats repeated S values conservatively; V8 uses S=32 only
    observations.
    """
    if "S" not in df or df["S"].nunique() < 2:
        return pd.DataFrame()
    condition_cols = [column for column in ["model_budget", "representation"] if column in df]
    rows = []
    S_values = sorted(pd.to_numeric(df["S"], errors="coerce").dropna().astype(int).unique())
    for condition, part in df.groupby(condition_cols, dropna=False):
        condition = condition if isinstance(condition, tuple) else (condition,)
        condition_values = dict(zip(condition_cols, condition))
        for metric in PRIMARY_METRICS:
            if metric not in part:
                continue
            wide = part.pivot_table(index=["seed"], columns="S", values=metric, aggfunc="first")
            for a, b in itertools.combinations(S_values, 2):
                if a not in wide or b not in wide:
                    continue
                pair = wide[[a, b]].dropna().reset_index()
                if pair.empty:
                    continue
                test = paired_permutation_test(pair[b], pair[a], seed=seed)
                _, diff_low, diff_high = bootstrap_mean_ci(
                    pair[b].to_numpy(dtype=float) - pair[a].to_numpy(dtype=float),
                    n_boot=5000, seed=seed,
                )
                rows.append(
                    {
                        **condition_values,
                        "metric": metric,
                        "S_a": int(a),
                        "S_b": int(b),
                        "difference_definition": f"S{b}-S{a}",
                        "difference_seed_bootstrap_low": diff_low,
                        "difference_seed_bootstrap_high": diff_high,
                        **test,
                    }
                )
    output = pd.DataFrame(rows)
    if not output.empty:
        output["inference_unit"] = "random_seed_run"
        output["inference_scope"] = "optimization_stability_not_dataset_generalization"
        output["p_holm"] = np.nan
        family_cols = [column for column in ["model_budget", "metric"] if column in output]
        for _, idx in output.groupby(family_cols, dropna=False).groups.items():
            output.loc[idx, "p_holm"] = holm_adjust(output.loc[idx, "p_value"].to_numpy())
    return output

def geometry_performance_delta_associations(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive co-variation of paired representation effects across seeds.

    This is intentionally not an inferential population test. For each S/model
    budget and representation pair, it correlates seed-wise performance deltas
    with seed-wise geometry deltas. With five seeds the statistic is reported as
    a stability diagnostic only.
    """
    if "representation" not in df or "seed" not in df:
        return pd.DataFrame()
    geometry_metrics = [
        metric for metric in (
            "fisher_ratio", "silhouette", "participation_ratio", "effective_rank",
            "mle_id", "linear_probe_pr_auc",
        ) if metric in df.columns
    ]
    performance_metrics = [
        metric for metric in ("test_pr_auc", "test_fnr_at_val_fpr_5pct")
        if metric in df.columns
    ]
    condition_cols = [column for column in ("S", "model_budget") if column in df.columns]
    rows = []
    reps = sorted(df["representation"].dropna().unique())
    for condition, part in df.groupby(condition_cols, dropna=False):
        condition = condition if isinstance(condition, tuple) else (condition,)
        condition_values = dict(zip(condition_cols, condition))
        for a, b in itertools.combinations(reps, 2):
            pair = part[part["representation"].isin([a, b])]
            for perf in performance_metrics:
                perf_wide = pair.pivot_table(index="seed", columns="representation", values=perf, aggfunc="first")
                if a not in perf_wide or b not in perf_wide:
                    continue
                for geom in geometry_metrics:
                    geom_wide = pair.pivot_table(index="seed", columns="representation", values=geom, aggfunc="first")
                    joined = pd.DataFrame({
                        "performance_delta": perf_wide[b] - perf_wide[a],
                        "geometry_delta": geom_wide[b] - geom_wide[a],
                    }).dropna()
                    if len(joined) < 3 or joined["performance_delta"].nunique() < 2 or joined["geometry_delta"].nunique() < 2:
                        rho = np.nan
                    else:
                        rho = float(spearmanr(joined["performance_delta"], joined["geometry_delta"]).statistic)
                    rows.append({
                        **condition_values,
                        "representation_a": a,
                        "representation_b": b,
                        "performance_metric": perf,
                        "geometry_metric": geom,
                        "spearman_rho": rho,
                        "n_seed_pairs": int(len(joined)),
                        "interpretation": "descriptive_seed_stability_only",
                    })
    return pd.DataFrame(rows)


def _latex_table(frame: pd.DataFrame, columns: list[tuple[str, str]], caption: str, label: str) -> str:
    if frame.empty:
        return f"% No data for {label}.\n"
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{" + "l" * sum(not pd.api.types.is_numeric_dtype(frame[col]) for col, _ in columns) + "r" * sum(pd.api.types.is_numeric_dtype(frame[col]) for col, _ in columns) + "}",
        "\\toprule",
        " & ".join(title for _, title in columns) + " \\\\",
        "\\midrule",
    ]
    for _, row in frame.iterrows():
        cells = []
        for column, _ in columns:
            value = row.get(column)
            cells.append(_fmt(value) if isinstance(value, (float, np.floating)) else str(value))
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    return "\n".join(lines)


def _geometry_latex(agg: pd.DataFrame) -> str:
    if agg.empty:
        return "% No geometry results.\n"
    rows = []
    keys = [column for column in ["S", "model_budget", "representation"] if column in agg]
    metrics = [
        ("test_pr_auc", "PR--AUC"),
        ("test_fnr_at_val_fpr_5pct", "FNR@5\\% FPR"),
        ("fisher_ratio", "Fisher"),
        ("participation_ratio", "PR-dim"),
        ("mle_id", "MLE-ID"),
        ("linear_probe_pr_auc", "Latent probe"),
        ("input_probe_pr_auc", "Input probe"),
    ]
    for _, row in agg.iterrows():
        output = {key: row[key] for key in keys}
        for metric, _ in metrics:
            output[metric] = (
                f"{_fmt(row.get(metric + '_mean'))} "
                f"[{_fmt(row.get(metric + '_seed_bootstrap_low'))}, {_fmt(row.get(metric + '_seed_bootstrap_high'))}]"
            )
        rows.append(output)
    frame = pd.DataFrame(rows)
    columns = [(key, key.replace("_", " ")) for key in keys] + metrics
    # All metrics are strings after CI formatting.
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Detection and representation geometry. Values are means with descriptive 95\\% seed-bootstrap intervals. Seeds characterize optimization variability and are not treated as independent datasets.}",
        "\\label{tab:geometry-summary}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{" + "l" * len(keys) + "r" * len(metrics) + "}",
        "\\toprule",
        " & ".join(title for _, title in columns) + " \\\\",
        "\\midrule",
    ]
    for _, row in frame.iterrows():
        lines.append(" & ".join(str(row[column]) for column, _ in columns) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}}", "\\end{table*}", ""])
    return "\n".join(lines)


def _aggregate_simple(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    agg = df.groupby(group_cols, dropna=False)[metrics].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join([str(v) for v in column if str(v)]) if isinstance(column, tuple) else column for column in agg.columns]
    return agg


def _read_csv_safe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def generate_report(
    results_dir: str | Path,
    out_dir: str | Path | None = None,
    *,
    seed: int = 42,
    ablation_results_dir: str | Path | None = None,
) -> Path:
    results_dir = Path(results_dir)
    out_dir = Path(out_dir) if out_dir else results_dir / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    geometry_path = results_dir / "geometry_results.csv"
    if not geometry_path.exists():
        geometry_path = None
    if geometry_path is None:
        raise FileNotFoundError(f"No geometry results found in {results_dir}")
    geometry = pd.read_csv(geometry_path)
    aggregate = aggregate_geometry(geometry, seed=seed)
    tests = pairwise_representation_tests(geometry, seed=seed)
    S_tests = context_length_tests(geometry, seed=seed)
    associations = geometry_performance_delta_associations(geometry)
    aggregate.to_csv(out_dir / "geometry_aggregate.csv", index=False)
    tests.to_csv(out_dir / "pairwise_representation_tests.csv", index=False)
    S_tests.to_csv(out_dir / "context_length_tests.csv", index=False)
    associations.to_csv(out_dir / "geometry_performance_delta_associations.csv", index=False)
    (out_dir / "geometry_summary_table.tex").write_text(_geometry_latex(aggregate), encoding="utf-8")

    generated = ["geometry_summary_table.tex", "geometry_performance_delta_associations.csv"]
    table_specs = [
        (
            "raw_input_alignment_results.csv",
            ["S", "representation_a", "representation_b"],
            ["linear_cka", "rsa_spearman", "procrustes_error"],
            "raw_input_alignment_aggregate.csv",
        ),
        (
            "initialization_geometry_control.csv",
            ["S", "model_budget", "representation"],
            [
                "initial_fisher_ratio",
                "trained_same_subset_fisher_ratio",
                "learning_delta_fisher_ratio",
                "initial_silhouette",
                "trained_same_subset_silhouette",
                "learning_delta_silhouette",
                "initial_effective_rank",
                "trained_same_subset_effective_rank",
                "initial_to_trained_cka",
                "initial_to_trained_rsa",
                "initial_to_trained_procrustes_error",
            ],
            "initialization_geometry_control_aggregate.csv",
        ),
        (
            "alignment_results.csv",
            ["S", "model_budget", "representation_a", "representation_b"],
            ["linear_cka", "rsa_spearman", "procrustes_error"],
            "alignment_aggregate.csv",
        ),
        (
            "layerwise_alignment_results.csv",
            ["S", "model_budget", "layer", "representation_a", "representation_b"],
            ["linear_cka", "rsa_spearman", "procrustes_error"],
            "layerwise_alignment_aggregate.csv",
        ),
        (
            "counterfactual_summary.csv",
            ["S", "model_budget", "representation", "control_type", "attack_type"],
            [
                "direction_concentration",
                "pair_distance_mean",
                "delta_effective_rank",
                "delta_pca1_variance",
                "probability_shift_mean",
                "paired_detection_success",
            ],
            "counterfactual_aggregate.csv",
        ),
        (
            "counterfactual_control_contrasts.csv",
            ["S", "model_budget", "representation", "attack_type"],
            [
                "invalid_minus_valid_distance",
                "invalid_minus_valid_probability_shift",
                "invalid_minus_valid_modified_probability",
                "invalid_outscores_valid",
                "invalid_valid_delta_cosine",
            ],
            "counterfactual_control_contrasts_aggregate.csv",
        ),
        (
            "cost_results.csv",
            ["S", "model_budget", "representation"],
            [
                "parameters",
                "macs_per_sample",
                "weights_fp32_bytes",
                "weights_int8_bytes",
                "input_theoretical_bytes",
                "peak_single_activation_bytes",
                "latency_ms_median",
            ],
            "cost_aggregate.csv",
        ),
    ]
    for filename, group_cols, metrics, output_name in table_specs:
        path = results_dir / filename
        if not path.exists():
            continue
        frame = _read_csv_safe(path)
        available_groups = [column for column in group_cols if column in frame]
        available_metrics = [column for column in metrics if column in frame]
        aggregated = _aggregate_simple(frame, available_groups, available_metrics)
        aggregated.to_csv(out_dir / output_name, index=False)
        generated.append(output_name)

    for filename in [
        "selected_nuisance_balance.csv",
        "selected_nuisance_baseline.csv",
        "selected_encoding_artifact_baseline.csv",
        "tensor_collision_report.csv",
        "dimension_sensitivity.csv",
        "intrinsic_dimension_synthetic_validation.csv",
        "input_probe_results.csv",
        "raw_input_alignment_results.csv",
        "initialization_geometry_control.csv",
        "reused_equivalent_training.csv",
        "integrity_hashes.csv",
        "runtime_profile.csv",
        "runtime_profile_summary.csv",
    ]:
        source = results_dir / filename
        if source.exists():
            _read_csv_safe(source).to_csv(out_dir / filename, index=False)
            generated.append(filename)

    for filename in ["reproducibility_manifest.json", "integrity_hashes.json"]:
        source = results_dir / filename
        if source.exists():
            shutil.copy2(source, out_dir / filename)
            generated.append(filename)

    ablation_source = Path(ablation_results_dir) if ablation_results_dir is not None else results_dir
    for filename in [
        "padding_ablation_results.csv",
        "padding_ablation_summary.csv",
        "padding_ablation_counts.csv",
        "reused_main_models.csv",
    ]:
        source = ablation_source / filename
        if source.exists():
            _read_csv_safe(source).to_csv(out_dir / filename, index=False)
            generated.append(filename)

    manifest = {
        "geometry_source": geometry_path.name,
        "ablation_source": ablation_source.name if ablation_results_dir is not None else None,
        "geometry_rows": int(len(geometry)),
        "aggregate_rows": int(len(aggregate)),
        "pairwise_tests": int(len(tests)),
        "context_length_tests": int(len(S_tests)),
        "generated_files": generated,
        "statistical_note": (
            "Representation tests are conducted separately for each S and model budget. "
            "Seeds are paired units; with five seeds, exact two-sided sign-permutation "
            "p-values cannot be smaller than 0.0625."
        ),
    }
    (out_dir / "report_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate corrected NeurReps experiments")
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ablations", default=None)
    args = parser.parse_args(argv)
    output = generate_report(
        args.results, args.out, seed=args.seed, ablation_results_dir=args.ablations
    )
    print(f"Report saved to {output}")


if __name__ == "__main__":
    main()
