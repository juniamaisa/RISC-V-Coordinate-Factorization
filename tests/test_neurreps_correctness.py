import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from neurreps.ablations import randomize_padding_positions
from neurreps.cost import analyze_model_cost
from neurreps.counterfactual import generate_counterfactual_and_control_pairs
from neurreps.data import (
    _cross_label_collision_indices,
    connected_match_groups,
)
from neurreps.geometry import mle_intrinsic_dimension, validate_intrinsic_dimension_synthetic
from neurreps.models import (
    build_model,
    count_trainable_parameters,
    find_width_for_parameter_budget,
)
from neurreps.statistics import paired_permutation_test
from neurreps.training import save_train_result, train_representation_model


def test_post_tensor_cross_label_collision_union():
    labels = np.array([0, 1, 0, 1, 0])
    hashes = {
        "SC": np.array(["x", "x", "a", "b", "c"]),
        "4C": np.array(["u", "v", "y", "y", "z"]),
        "32C": np.array(["1", "2", "3", "4", "5"]),
    }
    remove, report = _cross_label_collision_indices(hashes, labels)
    np.testing.assert_array_equal(remove, np.array([0, 1, 2, 3]))
    assert report.rows_removed_union.nunique() == 1
    assert int(report.rows_removed_union.iloc[0]) == 4


def test_mle_intrinsic_dimension_synthetic_validation():
    rows = validate_intrinsic_dimension_synthetic(
        dimensions=(2, 5, 10),
        n_samples=1800,
        repeats=1,
        relative_tolerance=0.35,
        seed=17,
    )
    assert len(rows) == 3
    assert all(row["passed"] for row in rows)
    # Direct estimator sanity check on a known 5-D Gaussian.
    rng = np.random.default_rng(4)
    estimate = mle_intrinsic_dimension(rng.normal(size=(2000, 5)), max_samples=2000, seed=2)
    assert 3.5 <= estimate <= 6.5


def test_training_extraction_and_saved_hashes_are_aligned(tmp_path):
    rng = np.random.default_rng(3)
    X = rng.integers(0, 256, size=(60, 4, 8), dtype=np.uint8)
    y = np.array([0, 1] * 30, dtype=np.uint8)
    groups = np.array([f"g{i:03d}" for i in range(len(y))])
    fixed = {
        "train": np.arange(0, 40),
        "val": np.arange(40, 50),
        "test": np.arange(50, 60),
    }
    result = train_representation_model(
        X,
        y,
        groups,
        "4C",
        seed=5,
        epochs=1,
        batch_size=7,
        patience=1,
        min_epochs=1,
        scheduler_patience=0,
        require_convergence=False,
        width=4,
        latent_dim=4,
        depth=1,
        device="cpu",
        fixed_indices=fixed,
        bootstrap_replicates=0,
    )
    for split, idx in fixed.items():
        np.testing.assert_array_equal(result.labels[split], y[idx])
    sample_id = np.array([f"id_{i}" for i in range(len(y))])
    sample_hash = np.array([f"hash_{i}" for i in range(len(y))])
    save_train_result(result, tmp_path, "4C", 5, sample_id, sample_hash, y)
    saved = np.load(tmp_path / "latents_4C_train_seed5.npz")
    np.testing.assert_array_equal(saved["y"], y[fixed["train"]])
    np.testing.assert_array_equal(saved["sample_hash"], sample_hash[fixed["train"]])
    assert not any(str(value).startswith("component_") for value in saved["sample_hash"])


def test_exact_sign_permutation_does_not_overstate_five_seeds():
    result = paired_permutation_test(np.ones(5), np.zeros(5))
    assert result["exact"] is True
    assert result["n"] == 5
    assert abs(result["p_value"] - 0.0625) < 1e-12


def test_counterfactual_valid_control_is_base_matched():
    common = {
        "records_s1": [
            {"sample_hash": "base", "source_index": 7, "chain_blocks": ["A", "B", "C"]}
        ],
        "bb2instr": {key: [[1, 0, 0, 0]] for key in ["A", "B", "C", "D", "E", "F", "G"]},
        "SUCC_MAP": {
            "A": {"B"},
            "B": {"C", "D"},
            "C": set(),
            "D": set(),
            "E": set(),
            "F": {"B"},
            "G": set(),
        },
    }
    pairs = generate_counterfactual_and_control_pairs(
        common,
        max_pairs_per_stratum=1,
        seed=1,
        weak_blocks={"B", "F", "G"},
        length_tolerance=0,
        exact_length=True,
    )
    assert len(pairs) == 4
    frame = pd.DataFrame(pairs)
    for (_, group), part in frame.groupby(["attack_type", "comparison_group_id"]):
        assert set(part.control_type) == {"valid", "invalid"}
        assert len({tuple(chain) for chain in part.benign_chain}) == 1
        assert set(part.changed_position) in ({0}, {2})
        invalid = part[part.control_type == "invalid"].iloc[0]
        valid = part[part.control_type == "valid"].iloc[0]
        assert len(invalid.invalid_edges) == 1
        assert len(valid.invalid_edges) == 0
        assert invalid.length_difference == 0
        assert valid.length_difference == 0
        if invalid.attack_type == "entry":
            assert bool(invalid.replacement_is_weak)
            assert bool(valid.replacement_is_weak)


def test_sc_padding_randomization_moves_complete_four_byte_slots():
    # Three valid slots and two padded slots.
    slots = np.array(
        [[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [0, 0, 0, 0], [0, 0, 0, 0]]]],
        dtype=np.uint8,
    ).reshape(1, 1, 20)
    mask = np.repeat(np.array([[[1, 1, 1, 0, 0]]], dtype=np.uint8), 4, axis=2)
    out, out_mask = randomize_padding_positions(slots, mask, representation="SC", seed=9)
    restored_slots = out.reshape(1, 5, 4)[0]
    slot_mask = out_mask.reshape(1, 5, 4)[0, :, 0]
    valid = restored_slots[slot_mask == 1]
    np.testing.assert_array_equal(valid, np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]], dtype=np.uint8))
    assert np.all(restored_slots[slot_mask == 0] == 0)


def test_parameter_matching_and_cost_fields():
    target_model = build_model("4C", width=16, latent_dim=16, depth=2, input_channels=4)
    target = count_trainable_parameters(target_model)
    width, achieved = find_width_for_parameter_budget(
        "32C", target, latent_dim=16, depth=2, input_channels=32
    )
    assert width > 0
    assert abs(achieved - target) / target < 0.03
    model = build_model("32C", width=width, latent_dim=16, depth=2, input_channels=32)
    cost = analyze_model_cost(
        model,
        np.zeros((1, 32, 16), dtype=np.uint8),
        representation="32C",
        latency_warmup=1,
        latency_repeats=2,
    )
    required = {
        "parameters", "macs_per_sample", "weights_fp32_bytes", "weights_int8_bytes",
        "input_theoretical_bytes", "peak_single_activation_bytes", "latency_ms_median",
    }
    assert required.issubset(cost)
    assert all(float(cost[key]) >= 0 for key in required)


def test_connected_groups_keep_matched_pairs_indivisible():
    sample_ids = np.array(["a", "b", "c", "d"])
    selected = np.arange(4)
    pairs = np.array([[0, 1], [2, 3]])
    duplicate = np.array(["x", "y", "x", "z"])
    groups = connected_match_groups(sample_ids, selected, pairs, duplicate_keys=[duplicate])
    assert len(set(groups)) == 1


def test_joint_nuisance_matching_is_single_pass_and_shared_across_s():
    from neurreps.data import AlignedBundle
    from neurreps.experiment import _stable_nuisance_matched_selections

    ids = np.array([f"id_{i}" for i in range(12)])
    labels = np.array([0] * 6 + [1] * 6, dtype=np.uint8)

    def make_bundle(S: int, reverse: bool = False):
        order = np.arange(len(ids))[::-1] if reverse else np.arange(len(ids))
        local_ids = ids[order]
        local_y = labels[order]
        # Every benign item has a close malicious partner, while S-dependent
        # padding/truncation values differ enough to exercise joint matching.
        base = np.tile(np.arange(6, dtype=float), 2)[order]
        meta = pd.DataFrame(
            {
                "raw_instruction_count": 10 + base,
                "raw_byte_count": 40 + 4 * base,
                "compressed_ratio": 0.01 * base,
                "raw_zero_byte_ratio": 0.005 * base,
                "pattern_0013_ratio": 0.004 * base,
                "padding_rate": (0.02 * base + (0.03 if S == 16 else 0.01)),
                "preserved_ratio": 1.0 - 0.01 * base,
                "trunc_removed": (base > (3 if S == 16 else 4)).astype(float),
                "label": local_y,
            }
        )
        arrays = {rep: np.zeros((len(ids), 1, 1), dtype=np.uint8) for rep in ("SC", "4C", "32C")}
        masks = {rep: np.ones((len(ids), 1, 1), dtype=np.uint8) for rep in arrays}
        hashes = {rep: np.array([f"{rep}_{value}" for value in local_ids]) for rep in arrays}
        return AlignedBundle(
            arrays=arrays,
            masks=masks,
            labels=local_y,
            sample_id=local_ids,
            sample_hash=np.array([f"h_{value}" for value in local_ids]),
            tensor_hashes=hashes,
            metadata=meta,
            collision_report=pd.DataFrame(),
            S=S,
            software="toy",
        )

    bundles = {16: make_bundle(16), 32: make_bundle(32, reverse=True)}
    selections, pairs_by_s, selected_ids = _stable_nuisance_matched_selections(
        bundles,
        {
            "nuisance_match": True,
            "data_seed": 7,
            "matching_caliper": 0.5,
            "matching_max_neighbors": 6,
        },
    )
    assert len(selected_ids) == 12
    ids16 = set(bundles[16].sample_id[selections[16]].tolist())
    ids32 = set(bundles[32].sample_id[selections[32]].tolist())
    assert ids16 == ids32 == selected_ids
    pair_ids16 = {
        tuple(bundles[16].sample_id[pair].tolist()) for pair in pairs_by_s[16]
    }
    pair_ids32 = {
        tuple(bundles[32].sample_id[pair].tolist()) for pair in pairs_by_s[32]
    }
    assert pair_ids16 == pair_ids32
