from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from .ablations import run as run_ablations
from .experiment import run as run_experiment
from .reporting import generate_report


def _copy_capacity_artifacts(capacity_out: Path, report_out: Path) -> list[str]:
    """Copy capacity-control artifacts into the integrated report directory.

    The focused protocol trains only the non-reference matched-capacity models
    (SC and 32C at S=32). The 4C reference is read from the main fixed-core run
    because it is the parameter-matching anchor. This helper does not alter any
    scientific result; it only makes the capacity control easy to audit from the
    report folder.
    """
    copied: list[str] = []
    for name in [
        "geometry_results.csv",
        "cost_results.csv",
        "input_probe_results.csv",
        "selected_nuisance_balance.csv",
        "selected_encoding_artifact_baseline.csv",
        "runtime_profile.csv",
        "runtime_profile_summary.csv",
        "reproducibility_manifest.json",
        "integrity_hashes.json",
        "config_used.json",
    ]:
        src = capacity_out / name
        if src.exists():
            dst = report_out / f"capacity_{name}"
            shutil.copy2(src, dst)
            copied.append(dst.name)
    # Add a compact aggregate for table construction.
    geom = capacity_out / "geometry_results.csv"
    if geom.exists():
        frame = pd.read_csv(geom)
        metrics = [
            c for c in [
                "pr_auc", "fnr_at_5pct_fpr", "roc_auc", "accuracy",
                "fisher_ratio", "silhouette", "participation_ratio",
                "effective_rank", "mle_intrinsic_dim", "latent_linear_probe_auc",
            ] if c in frame.columns
        ]
        groups = [c for c in ["S", "model_budget", "representation"] if c in frame.columns]
        if groups and metrics:
            agg = frame.groupby(groups, dropna=False)[metrics].agg(["mean", "std"]).reset_index()
            agg.columns = [
                "_".join([str(v) for v in col if str(v)]) if isinstance(col, tuple) else str(col)
                for col in agg.columns
            ]
            agg.to_csv(report_out / "capacity_geometry_aggregate.csv", index=False)
            copied.append("capacity_geometry_aggregate.csv")
    return copied


def run(
    main_config_path: str | Path,
    ablation_config_path: str | Path | None = None,
    *,
    capacity_config_path: str | Path | None = None,
    padding_config_path: str | Path | None = None,
) -> dict[str, str]:
    """Run the in-program NeurReps pipeline.

    Two modes are supported.

    1. Legacy V6 mode: provide ``main_config_path`` and ``ablation_config_path``.
    2. Focused V10 focused mode: provide ``main_config_path``, ``capacity_config_path``,
       and ``padding_config_path``. This evaluates the paper-sized protocol:
       9 main S=32 fixed-core trainings, 6 non-reference S=32 capacity-control
       trainings, and 9 S=32 padding-free trainings.
    """
    main_config = json.loads(Path(main_config_path).read_text(encoding="utf-8"))
    main_out = Path(run_experiment(main_config))

    summary: dict[str, str] = {"main_results": str(main_out)}

    if capacity_config_path is not None or padding_config_path is not None:
        if capacity_config_path is None or padding_config_path is None:
            raise ValueError("Focused mode requires both --capacity-config and --padding-config")
        capacity_config = json.loads(Path(capacity_config_path).read_text(encoding="utf-8"))
        padding_config = json.loads(Path(padding_config_path).read_text(encoding="utf-8"))
        capacity_out = Path(run_experiment(capacity_config))
        padding_config["main_results_dir"] = str(main_out)
        padding_config["reuse_main_original_padding"] = False
        padding_config["validate_intrinsic_dimension"] = False
        padding_config["intrinsic_validation_reused_from"] = str(
            main_out / "intrinsic_dimension_synthetic_validation.json"
        )
        padding_out = Path(run_ablations(padding_config))
        report_out = generate_report(main_out, ablation_results_dir=padding_out)
        copied = _copy_capacity_artifacts(capacity_out, report_out)
        focused = {
            "protocol": "focused_in_program_v10_s32_three_seed_cuda_auto",
            "main_fixed_core_unique_trainings": 9,
            "capacity_control_unique_trainings": 6,
            "padding_free_control_unique_trainings": 9,
            "total_unique_cnn_trainings": 24,
            "main_results": str(main_out),
            "capacity_results": str(capacity_out),
            "padding_results": str(padding_out),
            "report": str(report_out),
            "capacity_artifacts_copied_to_report": copied,
            "scientific_scope": "strictly in-program sha-test at S=32 only; three paired seeds; auto CUDA if available else CPU; information-equivalent SC/4C/32C; leakage-resistant grouped splits; collision and shortcut audits; no S=16, no cross-program, no cross-workload, no leave-one-program-out",
        }
        (main_out / "focused_pipeline_outputs.json").write_text(
            json.dumps(focused, indent=2), encoding="utf-8"
        )
        summary.update({
            "capacity_results": str(capacity_out),
            "padding_results": str(padding_out),
            "report": str(report_out),
        })
        return summary

    if ablation_config_path is None:
        raise ValueError("Legacy mode requires --ablation-config when focused configs are not provided")
    ablation_config = json.loads(Path(ablation_config_path).read_text(encoding="utf-8"))
    ablation_config["main_results_dir"] = str(main_out)
    ablation_config["reuse_main_original_padding"] = True
    ablation_config["validate_intrinsic_dimension"] = False
    ablation_config["intrinsic_validation_reused_from"] = str(
        main_out / "intrinsic_dimension_synthetic_validation.json"
    )
    ablation_out = Path(run_ablations(ablation_config))
    report_out = generate_report(main_out, ablation_results_dir=ablation_out)
    summary.update({
        "ablation_results": str(ablation_out),
        "report": str(report_out),
    })
    (main_out / "pipeline_outputs.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the in-program NeurReps pipeline")
    parser.add_argument("--main-config", required=True)
    parser.add_argument("--ablation-config", default=None, help="Legacy V6 full-ablation config")
    parser.add_argument("--capacity-config", default=None, help="Focused V9 S=32 capacity-control config")
    parser.add_argument("--padding-config", default=None, help="Focused V9 S=32 padding-free config")
    args = parser.parse_args(argv)
    summary = run(
        args.main_config,
        args.ablation_config,
        capacity_config_path=args.capacity_config,
        padding_config_path=args.padding_config,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
