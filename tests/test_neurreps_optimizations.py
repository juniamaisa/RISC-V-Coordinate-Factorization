import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from neurreps.geometry import (
    counterfactual_direction_geometry,
    layerwise_alignment,
    linear_cka,
    procrustes_error,
    rsa_spearman,
    shuffled_counterfactual_null,
)
from neurreps.training import _weighted_rank_metrics_from_counts, bootstrap_binary_metrics, train_representation_model


def test_layerwise_cached_metrics_match_direct_definitions():
    rng = np.random.default_rng(123)
    features = {
        "SC": {"block1": rng.normal(size=(96, 8)), "latent": rng.normal(size=(96, 6))},
        "4C": {"block1": rng.normal(size=(96, 8)), "latent": rng.normal(size=(96, 6))},
        "32C": {"block1": rng.normal(size=(96, 8)), "latent": rng.normal(size=(96, 6))},
    }
    rows = layerwise_alignment(features, seed=7, rsa_max_samples=1000)
    for row in rows:
        a, b, layer = row["representation_a"], row["representation_b"], row["layer"]
        x, y = features[a][layer], features[b][layer]
        assert np.isclose(row["linear_cka"], linear_cka(x, y), atol=1e-12, rtol=1e-12)
        assert np.isclose(row["rsa_spearman"], rsa_spearman(x, y, max_samples=1000, seed=7), atol=1e-12, rtol=1e-12)
        assert np.isclose(row["procrustes_error"], procrustes_error(x, y), atol=1e-12, rtol=1e-12)


def test_counterfactual_fast_null_uses_same_direction_statistic():
    rng = np.random.default_rng(9)
    benign = rng.normal(size=(80, 12))
    modified = benign + rng.normal(scale=0.2, size=benign.shape)
    seed = 5
    perm = np.random.default_rng(seed).permutation(len(modified))
    expected = counterfactual_direction_geometry(benign, modified[perm])["direction_concentration"]
    actual = shuffled_counterfactual_null(benign, modified, repeats=1, seed=seed)["shuffled_concentration_mean"]
    assert np.isclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_weighted_rank_metrics_match_explicit_resampling():
    y = np.array([0, 1, 1, 0, 1, 0], dtype=int)
    prob = np.array([0.9, 0.9, 0.7, 0.3, 0.3, 0.1], dtype=float)
    counts = np.array([[1, 2, 0, 1, 3, 2], [2, 1, 2, 3, 1, 1]], dtype=int)
    order = np.argsort(-prob, kind="mergesort")
    ys, ps, cs = y[order], prob[order], counts[:, order]
    starts = np.r_[0, np.flatnonzero(ps[1:] != ps[:-1]) + 1]
    ap, auc, valid = _weighted_rank_metrics_from_counts(ys, starts, cs)
    assert valid.all()
    for i in range(len(counts)):
        yy = np.repeat(ys, cs[i])
        pp = np.repeat(ps, cs[i])
        assert np.isclose(ap[i], average_precision_score(yy, pp), atol=1e-12, rtol=1e-12)
        assert np.isclose(auc[i], roc_auc_score(yy, pp), atol=1e-12, rtol=1e-12)


def test_bootstrap_respects_cluster_count():
    rng = np.random.default_rng(10)
    y = np.array([0, 1] * 20)
    p = rng.random(len(y))
    groups = np.array([f"g{i // 2}" for i in range(len(y))])
    out = bootstrap_binary_metrics(y, p, groups=groups, n_boot=50, seed=3)
    assert out["test_bootstrap_clusters"] == 20
    assert out["test_pr_auc_bootstrap_valid_replicates"] > 0


def test_adaptive_epoch_extension_reaches_hard_cap_when_not_converged():
    rng = np.random.default_rng(4)
    X = rng.integers(0, 256, size=(36, 4, 8), dtype=np.uint8)
    y = np.array([0, 1] * 18, dtype=np.uint8)
    groups = np.array([f"g{i}" for i in range(len(y))])
    fixed = {"train": np.arange(0, 24), "val": np.arange(24, 30), "test": np.arange(30, 36)}
    result = train_representation_model(
        X, y, groups, "4C", seed=2,
        epochs=1, max_epochs=2, epoch_extension=1, auto_extend_epochs=True,
        batch_size=8, patience=5, min_epochs=1, scheduler_patience=0,
        require_convergence=False, width=4, latent_dim=4, depth=1,
        fixed_indices=fixed, bootstrap_replicates=0, device="cpu",
    )
    assert result.metrics["initial_epoch_limit"] == 1
    assert result.metrics["hard_max_epochs"] == 2
    assert result.metrics["epochs_completed"] == 2
    assert result.metrics["extension_count"] == 1
    assert result.metrics["converged"] is False


def test_raw_sc_4c_geometry_is_isometric_under_coordinate_permutation():
    from neurreps.experiment import _raw_input_alignment_rows
    rng = np.random.default_rng(77)
    canonical = rng.integers(0, 256, size=(80, 12, 4), dtype=np.uint8)
    x4 = canonical.transpose(0, 2, 1)
    sc = canonical.reshape(len(canonical), 1, -1)
    x32 = np.unpackbits(canonical, axis=2).transpose(0, 2, 1).astype(np.uint8)
    rows = _raw_input_alignment_rows({"SC": sc, "4C": x4, "32C": x32}, S=12, seed=3, max_samples=1000)
    row = next(r for r in rows if {r["representation_a"], r["representation_b"]} == {"SC", "4C"})
    assert np.isclose(row["linear_cka"], 1.0, atol=1e-12)
    assert np.isclose(row["rsa_spearman"], 1.0, atol=1e-12)


def test_shared_spectral_computation_matches_public_metrics():
    from neurreps.geometry import _spectral_geometry, effective_rank, participation_ratio
    rng = np.random.default_rng(91)
    x = rng.normal(size=(120, 17))
    shared = _spectral_geometry(x)
    assert np.isclose(shared["participation_ratio"], participation_ratio(x), atol=1e-12, rtol=1e-12)
    assert np.isclose(shared["effective_rank"], effective_rank(x), atol=1e-12, rtol=1e-12)


def test_completed_model_resume_roundtrip(tmp_path):
    from neurreps.models import build_model
    from neurreps.training import load_train_result_if_complete, save_train_result
    rng = np.random.default_rng(19)
    X = rng.integers(0, 256, size=(36, 4, 8), dtype=np.uint8)
    y = np.array([0, 1] * 18, dtype=np.uint8)
    groups = np.array([f"g{i}" for i in range(len(y))])
    fixed = {"train": np.arange(0, 24), "val": np.arange(24, 30), "test": np.arange(30, 36)}
    result = train_representation_model(
        X, y, groups, "4C", seed=8, epochs=1, max_epochs=1,
        batch_size=8, patience=2, min_epochs=1, scheduler_patience=0,
        require_convergence=False, width=4, latent_dim=4, depth=1,
        fixed_indices=fixed, bootstrap_replicates=0, device="cpu",
        collect_test_intermediates=True,
    )
    fingerprint = {"schema": "test", "value": 3}
    sample_id = np.array([f"id{i}" for i in range(len(y))])
    sample_hash = np.array([f"h{i}" for i in range(len(y))])
    save_train_result(
        result, tmp_path, "4C", 8, sample_id, sample_hash, y,
        compress=False, fingerprint=fingerprint,
    )
    model = build_model("4C", width=4, latent_dim=4, depth=1, input_channels=4)
    loaded = load_train_result_if_complete(
        tmp_path, "4C", 8, model=model, expected_fingerprint=fingerprint,
        require_intermediates=True, device="cpu",
    )
    assert loaded is not None
    np.testing.assert_array_equal(loaded.indices["test"], fixed["test"])
    np.testing.assert_allclose(loaded.latents["test"], result.latents["test"])
    assert loaded.test_layer_features is not None
    assert set(loaded.test_layer_features) == set(result.test_layer_features)
    miss = load_train_result_if_complete(
        tmp_path, "4C", 8,
        model=build_model("4C", width=4, latent_dim=4, depth=1, input_channels=4),
        expected_fingerprint={"schema": "different"}, require_intermediates=True, device="cpu",
    )
    assert miss is None


def test_inference_batch_size_preserves_predictions():
    from neurreps.models import build_model
    from neurreps.training import predict_numpy
    rng = np.random.default_rng(211)
    X = rng.integers(0, 256, size=(41, 4, 16), dtype=np.uint8)
    model = build_model("4C", width=8, latent_dim=7, depth=2, input_channels=4)
    p_small, z_small = predict_numpy(model, X, batch_size=5, device="cpu")
    p_large, z_large = predict_numpy(model, X, batch_size=41, device="cpu")
    np.testing.assert_allclose(p_small, p_large, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(z_small, z_large, atol=1e-6, rtol=1e-6)


def test_initialization_geometry_control_does_not_change_training_result():
    from neurreps.experiment import _initialization_geometry_control

    rng = np.random.default_rng(333)
    X = rng.integers(0, 256, size=(40, 4, 8), dtype=np.uint8)
    y = np.array([0, 1] * 20, dtype=np.uint8)
    groups = np.array([f"g{i}" for i in range(len(y))])
    fixed = {
        "train": np.arange(0, 24),
        "val": np.arange(24, 32),
        "test": np.arange(32, 40),
    }
    kwargs = dict(
        seed=17,
        epochs=1,
        max_epochs=1,
        batch_size=8,
        eval_batch_size=16,
        patience=2,
        min_epochs=1,
        scheduler_patience=0,
        require_convergence=False,
        width=4,
        latent_dim=4,
        depth=1,
        fixed_indices=fixed,
        bootstrap_replicates=0,
        device="cpu",
    )
    baseline = train_representation_model(X, y, groups, "4C", **kwargs)

    local, initial_z, metrics = _initialization_geometry_control(
        X,
        y,
        fixed["test"],
        rep="4C",
        spec={"width": 4, "latent_dim": 4, "depth": 1},
        seed=17,
        device="cpu",
        batch_size=16,
        max_samples=8,
    )
    assert len(local) == len(initial_z) == 8
    assert np.isfinite(metrics["fisher_ratio"])

    after_control = train_representation_model(X, y, groups, "4C", **kwargs)
    for key, value in baseline.model.state_dict().items():
        np.testing.assert_allclose(
            value.detach().cpu().numpy(),
            after_control.model.state_dict()[key].detach().cpu().numpy(),
            atol=0,
            rtol=0,
        )
    np.testing.assert_allclose(
        baseline.probabilities["test"], after_control.probabilities["test"], atol=0, rtol=0
    )
    np.testing.assert_allclose(
        baseline.latents["test"], after_control.latents["test"], atol=0, rtol=0
    )


def test_reference_representation_has_identical_fixed_and_matched_model_spec():
    from neurreps.experiment import _model_spec

    config = {
        "width": 32,
        "latent_dim": 64,
        "depth": 3,
        "parameter_target_representation": "4C",
        "parameter_match_tolerance": 0.03,
    }
    fixed = _model_spec("4C", "fixed_core", config, input_channels=4)
    matched = _model_spec("4C", "matched_total", config, input_channels=4)
    assert fixed == matched


def test_layerwise_rsa_nonfinite_fallback_matches_direct_definition():
    rng = np.random.default_rng(991)
    features = {
        "SC": {"latent": rng.normal(size=(24, 5))},
        "4C": {"latent": rng.normal(size=(24, 5))},
    }
    features["SC"]["latent"][0] = 0.0
    features["4C"]["latent"][1] = 0.0
    row = layerwise_alignment(features, seed=2, rsa_max_samples=100)[0]
    expected = rsa_spearman(
        features["SC"]["latent"], features["4C"]["latent"], max_samples=100, seed=2
    )
    assert np.isclose(row["rsa_spearman"], expected, equal_nan=True, atol=1e-12, rtol=1e-12)
