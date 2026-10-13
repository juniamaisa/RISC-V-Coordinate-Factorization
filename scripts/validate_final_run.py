#!/usr/bin/env python
"""Validate that a V10 final run produced the required paper artifacts.

This script does not recompute metrics. It checks protocol completeness and
artifact presence before the results are used in a manuscript.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED_MAIN_FILES = [
    "geometry_results.csv",
    "input_probe_results.csv",
    "layerwise_alignment_results.csv",
    "counterfactual_summary.csv",
    "counterfactual_control_contrasts.csv",
    "selected_nuisance_balance.csv",
    "selected_nuisance_baseline.csv",
    "selected_encoding_artifact_baseline.csv",
    "tensor_collision_report.csv",
    "reproducibility_manifest.json",
    "runtime_profile_summary.csv",
]
REQUIRED_CAPACITY_FILES = [
    "geometry_results.csv",
    "input_probe_results.csv",
    "selected_nuisance_balance.csv",
    "reproducibility_manifest.json",
]
REQUIRED_PADDING_FILES = [
    "padding_ablation_results.csv",
    "padding_ablation_summary.csv",
    "padding_ablation_counts.csv",
]


def check_dir(path: Path, files: list[str]) -> list[str]:
    return [str(path / f) for f in files if not (path / f).exists()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", default="results/neurreps_sha_test_s32_main")
    parser.add_argument("--capacity", default="results/neurreps_sha_test_s32_capacity")
    parser.add_argument("--padding", default="results/neurreps_sha_test_s32_padding_free")
    args = parser.parse_args()

    main_dir = Path(args.main)
    capacity_dir = Path(args.capacity)
    padding_dir = Path(args.padding)
    missing = []
    missing += check_dir(main_dir, REQUIRED_MAIN_FILES)
    missing += check_dir(capacity_dir, REQUIRED_CAPACITY_FILES)
    missing += check_dir(padding_dir, REQUIRED_PADDING_FILES)

    config_path = main_dir / "config_used.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        if cfg.get("S_values") != [32]:
            missing.append("protocol error: main config S_values must be [32]")
        if cfg.get("seeds") != [11, 37, 71]:
            missing.append("protocol error: main seeds must be [11, 37, 71]")
        if cfg.get("device") != "auto":
            missing.append("protocol error: main device must be auto")

    metric_files = sorted(main_dir.rglob("metrics_*_seed*.json"))
    if len(metric_files) < 9:
        missing.append(f"expected at least 9 main metrics files, found {len(metric_files)}")
    for path in metric_files:
        try:
            metrics = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            missing.append(f"could not read {path}: {exc}")
            continue
        if "resolved_device" not in metrics:
            missing.append(f"missing resolved_device in {path}")
        if metrics.get("converged") is not True:
            missing.append(f"non-converged model: {path}")

    if missing:
        print("V10 final-run validation: FAIL")
        for item in missing:
            print("-", item)
        return 1
    print("V10 final-run validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
