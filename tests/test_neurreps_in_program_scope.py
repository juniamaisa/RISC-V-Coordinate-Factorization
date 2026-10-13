from pathlib import Path


def test_framework_is_in_program_only():
    root = Path(__file__).resolve().parents[1]
    entrypoint = (root / "run_neurreps.py").read_text(encoding="utf-8")

    assert '"cross-program"' not in entrypoint
    assert not (root / "neurreps" / "cross_program.py").exists()
    assert not (root / "configs" / "neurreps_cross_program.example.json").exists()
