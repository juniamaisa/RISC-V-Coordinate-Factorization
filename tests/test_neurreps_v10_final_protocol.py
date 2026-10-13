import json
from pathlib import Path


def test_v10_protocol_scope_and_counts():
    cfgs = {
        "configs/neurreps_sha_test_focused_main.json": ("results/neurreps_sha_test_s32_main", ["SC", "4C", "32C"], ["fixed_core"], 9),
        "configs/neurreps_sha_test_focused_capacity_s32.json": ("results/neurreps_sha_test_s32_capacity", ["SC", "32C"], ["matched_total"], 6),
        "configs/neurreps_sha_test_focused_padding_s32.json": ("results/neurreps_sha_test_s32_padding_free", ["SC", "4C", "32C"], ["fixed_core"], 9),
    }
    for path, (out_dir, reps, budgets, expected_trainings) in cfgs.items():
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
        assert cfg["S_values"] == [32]
        assert cfg["seeds"] == [11, 37, 71]
        assert cfg["device"] == "auto"
        assert cfg["output_dir"] == out_dir
        assert cfg["representations"] == reps
        assert cfg["model_budgets"] == budgets
        assert cfg["expected_unique_cnn_trainings"] == expected_trainings
        assert "S=32 only" in cfg["paper_protocol_scope"]


def test_v10_final_methodology_doc_excludes_removed_scope():
    text = Path("docs/V10_FINAL_METHOD_PROTOCOL_PT.md").read_text(encoding="utf-8")
    required = [
        "Representation-independent execution-context",
        "Static + dynamic CFG",
        "Counterfactual pair",
        "Leakage-resistant",
        "Structural fingerprints",
        "Duplicate detection",
        "Representation collision",
        "Representation-induced shortcut",
    ]
    for item in required:
        assert item in text
    assert "opcode-semantic" in text
    assert "random-split" in text


def test_v10_validator_script_exists():
    assert Path("scripts/validate_final_run.py").exists()
