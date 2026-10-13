"""Convenience entry point.

Examples:
  python run_neurreps.py audit --mode-folder output/Coremark-pro/sha-test/dynamic_and_static --software sha-test --S 32
  python run_neurreps.py experiment --config configs/neurreps_sha_test.json
  python run_neurreps.py ablations --config configs/neurreps_padding_ablations_sha_test.json
  python run_neurreps.py report --results results/neurreps_geometry_sha_test
  python run_neurreps.py pipeline --main-config configs/neurreps_sha_test_focused_main.json --capacity-config configs/neurreps_sha_test_focused_capacity_s32.json --padding-config configs/neurreps_sha_test_focused_padding_s32.json
"""
from __future__ import annotations

import sys


def main():
    commands = {"audit", "experiment", "ablations", "report", "pipeline"}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        raise SystemExit(f"Usage: python run_neurreps.py {{{'|'.join(sorted(commands))}}} [arguments]")
    command = sys.argv[1]
    argv = sys.argv[2:]
    if command == "audit":
        from neurreps.audit import main as target
    elif command == "experiment":
        from neurreps.experiment import main as target
    elif command == "ablations":
        from neurreps.ablations import main as target
    elif command == "pipeline":
        from neurreps.pipeline import main as target
    else:
        from neurreps.reporting import main as target
    target(argv)


if __name__ == "__main__":
    main()
