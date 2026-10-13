from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import pdist
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    davies_bouldin_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2-D matrix, got {x.shape}")
    return x


def _finite_rows(x: np.ndarray) -> np.ndarray:
    x = _as_2d(x)
    return x[np.all(np.isfinite(x), axis=1)]


def _unique_rows(x: np.ndarray) -> np.ndarray:
    x = _finite_rows(x)
    if len(x) == 0:
        return x
    return np.unique(x, axis=0)


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x, y = _as_2d(x), _as_2d(y)
    if x.shape[0] != y.shape[0]:
        raise ValueError("CKA requires paired rows")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    numerator = np.linalg.norm(y.T @ x, ord="fro") ** 2
    denominator = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")
    return float(numerator / max(denominator, 1e-12))


def rsa_spearman(x: np.ndarray, y: np.ndarray, max_samples: int = 2500, seed: int = 42) -> float:
    x, y = _as_2d(x), _as_2d(y)
    if x.shape[0] != y.shape[0]:
        raise ValueError("RSA requires paired rows")
    if len(x) > max_samples:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(x), max_samples, replace=False))
        x, y = x[idx], y[idx]
    dx, dy = pdist(x, metric="cosine"), pdist(y, metric="cosine")
    keep = np.isfinite(dx) & np.isfinite(dy)
    if keep.sum() < 3:
        return float("nan")
    return float(spearmanr(dx[keep], dy[keep]).statistic)


def procrustes_error(x: np.ndarray, y: np.ndarray) -> float:
    x, y = _as_2d(x), _as_2d(y)
    if x.shape[0] != y.shape[0]:
        raise ValueError("Procrustes comparison requires paired rows")
    d = min(x.shape[1], y.shape[1], max(1, x.shape[0] - 1))
    if x.shape[1] != d:
        x = PCA(n_components=d, random_state=0).fit_transform(x)
    else:
        x = x.copy()
    if y.shape[1] != d:
        y = PCA(n_components=d, random_state=0).fit_transform(y)
    else:
        y = y.copy()
    x -= x.mean(axis=0, keepdims=True)
    y -= y.mean(axis=0, keepdims=True)
    x /= max(np.linalg.norm(x), 1e-12)
    y /= max(np.linalg.norm(y), 1e-12)
    rotation, _ = orthogonal_procrustes(x, y)
    return float(np.linalg.norm(x @ rotation - y, ord="fro") ** 2)


def _spectral_geometry(x: np.ndarray) -> dict[str, float]:
    """Exact PR/effective-rank/PCA1 statistics from one centered SVD."""
    values = _finite_rows(x)
    if len(values) < 2:
        return {
            "participation_ratio": float("nan"),
            "effective_rank": float("nan"),
            "pca1_variance": float("nan"),
        }
    s = np.linalg.svd(values - values.mean(axis=0, keepdims=True), compute_uv=False)
    s2 = np.square(s)
    total2 = float(s2.sum())
    pr = float(total2 * total2 / max(float(np.square(s2).sum()), 1e-12))
    total = float(s.sum())
    if total <= 1e-12:
        erank = 1.0
        pca1 = float("nan")
    else:
        p = s / total
        p = p[p > 0]
        erank = float(np.exp(-(p * np.log(p)).sum()))
        pca1 = float(s2[0] / total2) if total2 > 1e-12 else float("nan")
    return {"participation_ratio": pr, "effective_rank": erank, "pca1_variance": pca1}


def participation_ratio(x: np.ndarray) -> float:
    return _spectral_geometry(x)["participation_ratio"]


def effective_rank(x: np.ndarray) -> float:
    return _spectral_geometry(x)["effective_rank"]

def _prepared_knn_points(x: np.ndarray, max_samples: int, seed: int) -> tuple[np.ndarray, int, int]:
    original = _finite_rows(x)
    unique = np.unique(original, axis=0) if len(original) else original
    unique_count = len(unique)
    if len(unique) > max_samples:
        rng = np.random.default_rng(seed)
        unique = unique[np.sort(rng.choice(len(unique), max_samples, replace=False))]
    if len(unique) >= 2:
        unique = StandardScaler().fit_transform(unique)
    return unique, len(original), unique_count


def mle_intrinsic_dimension(
    x: np.ndarray,
    *,
    k_values: Sequence[int] = (5, 8, 10, 12, 15),
    max_samples: int = 10000,
    seed: int = 42,
) -> float:
    """Levina--Bickel nearest-neighbour MLE averaged over stable k values."""
    points, _, _ = _prepared_knn_points(x, max_samples, seed)
    if len(points) < 8:
        return float("nan")
    max_k = min(max(k_values), len(points) - 1)
    distances, _ = NearestNeighbors(n_neighbors=max_k + 1).fit(points).kneighbors(points)
    neighbor_distances = distances[:, 1:]
    estimates: list[float] = []
    for k in k_values:
        if k > max_k or k < 3:
            continue
        d = neighbor_distances[:, :k]
        valid = np.all(d > 1e-12, axis=1)
        if not np.any(valid):
            continue
        dv = d[valid]
        tk = dv[:, -1]
        denom = np.sum(np.log(tk[:, None] / dv[:, :-1]), axis=1)
        keep = denom > 1e-12
        if np.any(keep):
            local = (float(k - 2) / denom[keep]).astype(np.float64, copy=False)
            estimates.append(float(np.median(local)))
    return float(np.median(estimates)) if estimates else float("nan")


def validate_intrinsic_dimension_synthetic(
    *,
    dimensions: Sequence[int] = (2, 5, 10, 20),
    n_samples: int = 4000,
    repeats: int = 3,
    relative_tolerance: float = 0.30,
    seed: int = 1729,
) -> list[dict]:
    """Validate the reported MLE intrinsic-dimension estimator on known Gaussians.

    TwoNN is intentionally not reported by the experiment because near-duplicate
    latent vectors made it unstable on this dataset. This validation can stop
    the scientific run before any model training if the MLE estimator fails on
    synthetic data with known intrinsic dimension.
    """
    rows: list[dict] = []
    for dimension in [int(value) for value in dimensions]:
        if dimension < 1:
            raise ValueError("Synthetic intrinsic dimensions must be positive")
        estimates = []
        for repeat in range(int(repeats)):
            rng = np.random.default_rng(int(seed) + 1009 * dimension + repeat)
            points = rng.normal(size=(int(n_samples), dimension))
            estimate = mle_intrinsic_dimension(
                points,
                max_samples=int(n_samples),
                seed=int(seed) + repeat,
            )
            estimates.append(float(estimate))
        mean_estimate = float(np.nanmean(estimates))
        relative_error = float(abs(mean_estimate - dimension) / dimension)
        rows.append(
            {
                "true_dimension": int(dimension),
                "samples": int(n_samples),
                "repeats": int(repeats),
                "mle_mean": mean_estimate,
                "mle_std": float(np.nanstd(estimates, ddof=1)) if len(estimates) > 1 else 0.0,
                "relative_error": relative_error,
                "relative_tolerance": float(relative_tolerance),
                "passed": bool(np.isfinite(mean_estimate) and relative_error <= float(relative_tolerance)),
            }
        )
    return rows


def dimension_sensitivity(
    x: np.ndarray,
    *,
    sample_sizes: Sequence[int] = (250, 500, 1000, 2500, 5000, 10000),
    repeats: int = 5,
    seed: int = 42,
) -> list[dict]:
    unique = _unique_rows(x)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for n in sample_sizes:
        if len(unique) < min(int(n), 20):
            continue
        size = min(int(n), len(unique))
        for repeat in range(int(repeats)):
            idx = np.sort(rng.choice(len(unique), size, replace=False))
            subset = unique[idx]
            spectral = _spectral_geometry(subset)
            rows.append(
                {
                    "sample_size": int(size),
                    "repeat": int(repeat),
                    "mle_id": mle_intrinsic_dimension(subset, max_samples=size, seed=seed + repeat),
                    "participation_ratio": spectral["participation_ratio"],
                    "effective_rank": spectral["effective_rank"],
                }
            )
    return rows


def class_geometry(x: np.ndarray, y: np.ndarray, max_silhouette: int = 10000, seed: int = 42) -> dict[str, float]:
    x, y = _as_2d(x), np.asarray(y).astype(int)
    if len(x) != len(y) or set(np.unique(y)) != {0, 1}:
        raise ValueError("Class geometry requires paired binary labels")
    z0, z1 = x[y == 0], x[y == 1]
    mu0, mu1 = z0.mean(axis=0), z1.mean(axis=0)
    within = 0.5 * (
        np.mean(np.sum((z0 - mu0) ** 2, axis=1)) + np.mean(np.sum((z1 - mu1) ** 2, axis=1))
    )
    centroid_distance = float(np.linalg.norm(mu1 - mu0))
    if len(x) > max_silhouette:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(x), max_silhouette, replace=False))
        sx, sy = x[idx], y[idx]
    else:
        sx, sy = x, y
    unique_count = len(_unique_rows(x))
    spectral = _spectral_geometry(x)
    return {
        "centroid_distance": centroid_distance,
        "within_class_scatter": float(within),
        "fisher_ratio": float(centroid_distance**2 / max(within, 1e-12)),
        "silhouette": float(silhouette_score(sx, sy, metric="euclidean")),
        "davies_bouldin": float(davies_bouldin_score(sx, sy)),
        "participation_ratio": spectral["participation_ratio"],
        "effective_rank": spectral["effective_rank"],
        "mle_id": mle_intrinsic_dimension(x, seed=seed),
        "latent_rows": int(len(x)),
        "latent_unique_rows": int(unique_count),
        "latent_duplicate_fraction": float(1.0 - unique_count / max(len(x), 1)),
    }


def linear_probe(train_x, train_y, test_x, test_y, seed: int = 42, prefix: str = "linear_probe_") -> dict[str, float]:
    train_x, test_x = _as_2d(train_x), _as_2d(test_x)
    train_y, test_y = np.asarray(train_y).astype(int), np.asarray(test_y).astype(int)
    if len(train_x) != len(train_y) or len(test_x) != len(test_y):
        raise ValueError("Linear probe labels are not aligned with features")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed, solver="lbfgs"),
    )
    model.fit(train_x, train_y)
    prob = model.predict_proba(test_x)[:, 1]
    return {
        f"{prefix}pr_auc": float(average_precision_score(test_y, prob)),
        f"{prefix}roc_auc": float(roc_auc_score(test_y, prob)),
    }


def input_linear_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    seed: int = 42,
    max_train: int = 30000,
) -> dict[str, float]:
    """Linear accessibility of the raw input before neural training."""
    xtr = np.asarray(train_x).reshape(len(train_x), -1).astype(np.float32)
    xte = np.asarray(test_x).reshape(len(test_x), -1).astype(np.float32)
    ytr = np.asarray(train_y).astype(int)
    if len(xtr) > max_train:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(xtr), max_train, replace=False))
        xtr, ytr = xtr[idx], ytr[idx]
    return linear_probe(xtr, ytr, xte, test_y, seed=seed, prefix="input_probe_")


def counterfactual_direction_geometry(benign_z: np.ndarray, modified_z: np.ndarray) -> dict[str, float]:
    benign_z, modified_z = _as_2d(benign_z), _as_2d(modified_z)
    if benign_z.shape != modified_z.shape:
        raise ValueError("Counterfactual matrices must have equal shape")
    delta = modified_z - benign_z
    norm = np.linalg.norm(delta, axis=1)
    unit = delta / np.maximum(norm[:, None], 1e-12)
    mean_direction = unit.mean(axis=0)
    spectral = _spectral_geometry(delta)
    return {
        "pair_distance_mean": float(norm.mean()),
        "pair_distance_median": float(np.median(norm)),
        "direction_concentration": float(np.linalg.norm(mean_direction)),
        "delta_effective_rank": spectral["effective_rank"],
        "delta_participation_ratio": spectral["participation_ratio"],
        "delta_pca1_variance": spectral["pca1_variance"],
        "mean_direction_norm": float(np.linalg.norm(mean_direction)),
    }


def counterfactual_score_metrics(
    benign_prob: np.ndarray,
    modified_prob: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    benign_prob = np.asarray(benign_prob, dtype=float)
    modified_prob = np.asarray(modified_prob, dtype=float)
    eps = 1e-7
    b_logit = np.log(np.clip(benign_prob, eps, 1 - eps) / np.clip(1 - benign_prob, eps, 1 - eps))
    m_logit = np.log(np.clip(modified_prob, eps, 1 - eps) / np.clip(1 - modified_prob, eps, 1 - eps))
    return {
        "probability_shift_mean": float(np.mean(modified_prob - benign_prob)),
        "probability_shift_median": float(np.median(modified_prob - benign_prob)),
        "logit_shift_mean": float(np.mean(m_logit - b_logit)),
        "score_increase_fraction": float(np.mean(modified_prob > benign_prob)),
        "paired_detection_success": float(np.mean((benign_prob < threshold) & (modified_prob >= threshold))),
        "benign_correct_fraction": float(np.mean(benign_prob < threshold)),
        "modified_detected_fraction": float(np.mean(modified_prob >= threshold)),
    }


def mean_direction_cosine(a_delta: np.ndarray, b_delta: np.ndarray) -> float:
    a = _as_2d(a_delta).mean(axis=0)
    b = _as_2d(b_delta).mean(axis=0)
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def _direction_concentration_only(benign_z: np.ndarray, modified_z: np.ndarray) -> float:
    """Compute only the statistic needed by the shuffled null.

    This is algebraically identical to the direction-concentration field from
    ``counterfactual_direction_geometry`` and intentionally avoids its SVD,
    covariance eigendecomposition, and PCA, which are irrelevant for the null.
    """
    delta = np.asarray(modified_z, dtype=np.float64) - np.asarray(benign_z, dtype=np.float64)
    norm = np.linalg.norm(delta, axis=1)
    unit = delta / np.maximum(norm[:, None], 1e-12)
    return float(np.linalg.norm(unit.mean(axis=0)))


def shuffled_counterfactual_null(
    benign_z: np.ndarray,
    modified_z: np.ndarray,
    *,
    repeats: int = 500,
    seed: int = 42,
) -> dict[str, float]:
    benign_z, modified_z = _as_2d(benign_z), _as_2d(modified_z)
    if benign_z.shape != modified_z.shape:
        raise ValueError("Counterfactual matrices must have equal shape")
    rng = np.random.default_rng(seed)
    concentrations = np.empty(int(repeats), dtype=np.float64)
    for repeat in range(int(repeats)):
        perm = rng.permutation(len(modified_z))
        concentrations[repeat] = _direction_concentration_only(benign_z, modified_z[perm])
    return {
        "shuffled_concentration_mean": float(np.mean(concentrations)),
        "shuffled_concentration_p95": float(np.quantile(concentrations, 0.95)),
    }


def _rsa_distance_vector(x: np.ndarray, *, indices: np.ndarray | None = None) -> np.ndarray:
    x = _as_2d(x)
    if indices is not None:
        x = x[np.asarray(indices, dtype=np.int64)]
    return pdist(x, metric="cosine")


def layerwise_alignment(
    features_by_rep: Mapping[str, Mapping[str, np.ndarray]],
    *,
    seed: int = 42,
    rsa_max_samples: int = 2500,
) -> list[dict]:
    """Pairwise layer alignment with exact reusable sufficient statistics.

    RSA pairwise distances, CKA centering/self-normalizers, and normalized
    Procrustes inputs are cached once per representation/layer. Pairwise metric
    definitions remain identical to ``rsa_spearman``, ``linear_cka``, and
    ``procrustes_error``.
    """
    reps = sorted(features_by_rep)
    if not reps:
        return []
    common_layers = sorted(set.intersection(*(set(features_by_rep[rep]) for rep in reps)))
    rows: list[dict] = []
    for layer in common_layers:
        arrays = {rep: _as_2d(features_by_rep[rep][layer]) for rep in reps}
        n = len(arrays[reps[0]])
        if any(len(arrays[rep]) != n for rep in reps):
            raise ValueError(f"Layer {layer} has unpaired rows across representations")

        indices = None
        if n > int(rsa_max_samples):
            rng = np.random.default_rng(seed)
            indices = np.sort(rng.choice(n, int(rsa_max_samples), replace=False))
        rsa_cache = {rep: _rsa_distance_vector(arrays[rep], indices=indices) for rep in reps}
        rsa_all_finite = all(np.all(np.isfinite(values)) for values in rsa_cache.values())
        # In the normal latent-space case all cosine distances are finite.
        # Spearman is Pearson correlation of average ranks, so ranking each
        # representation once per layer is exactly reusable across pairs.
        # If any vector contains non-finite values, pair-specific filtering can
        # change the ranking population and we conservatively fall back below.
        rsa_rank_cache = (
            {rep: rankdata(values, method="average") for rep, values in rsa_cache.items()}
            if rsa_all_finite else {}
        )

        centered = {rep: arr - arr.mean(axis=0, keepdims=True) for rep, arr in arrays.items()}
        cka_self = {
            rep: float(np.linalg.norm(value.T @ value, ord="fro"))
            for rep, value in centered.items()
        }
        proc_cache: dict[tuple[str, int], np.ndarray] = {}

        def proc_input(rep: str, d: int) -> np.ndarray:
            key = (rep, int(d))
            if key in proc_cache:
                return proc_cache[key]
            arr = arrays[rep]
            if arr.shape[1] != d:
                value = PCA(n_components=d, random_state=0).fit_transform(arr)
                value -= value.mean(axis=0, keepdims=True)
            else:
                value = centered[rep].copy()
            value /= max(np.linalg.norm(value), 1e-12)
            proc_cache[key] = value
            return value

        for i, a in enumerate(reps):
            for b in reps[i + 1 :]:
                dx, dy = rsa_cache[a], rsa_cache[b]
                if rsa_all_finite:
                    rx, ry = rsa_rank_cache[a], rsa_rank_cache[b]
                    rx_centered = rx - rx.mean()
                    ry_centered = ry - ry.mean()
                    denom = np.linalg.norm(rx_centered) * np.linalg.norm(ry_centered)
                    rsa = float(
                        np.dot(rx_centered, ry_centered) / denom
                    ) if denom > 1e-12 else float("nan")
                else:
                    keep = np.isfinite(dx) & np.isfinite(dy)
                    rsa = (
                        float("nan")
                        if keep.sum() < 3
                        else float(spearmanr(dx[keep], dy[keep]).statistic)
                    )

                xa, yb = centered[a], centered[b]
                numerator = np.linalg.norm(yb.T @ xa, ord="fro") ** 2
                cka = float(numerator / max(cka_self[a] * cka_self[b], 1e-12))

                d = min(arrays[a].shape[1], arrays[b].shape[1], max(1, n - 1))
                px, py = proc_input(a, d), proc_input(b, d)
                rotation, _ = orthogonal_procrustes(px, py)
                proc = float(np.linalg.norm(px @ rotation - py, ord="fro") ** 2)

                rows.append(
                    {
                        "representation_a": a,
                        "representation_b": b,
                        "layer": layer,
                        "linear_cka": cka,
                        "rsa_spearman": rsa,
                        "procrustes_error": proc,
                    }
                )
    return rows

def optional_persistent_homology(x: np.ndarray, max_samples: int = 1500, seed: int = 42) -> dict:
    try:
        from ripser import ripser
    except ImportError:
        return {"topology_available": False}
    x = _unique_rows(x)
    if len(x) > max_samples:
        rng = np.random.default_rng(seed)
        x = x[np.sort(rng.choice(len(x), max_samples, replace=False))]
    if len(x) < 5:
        return {"topology_available": True, "topology_samples": int(len(x))}
    d = min(10, x.shape[1], len(x) - 1)
    projected = PCA(n_components=d, whiten=True, random_state=seed).fit_transform(x)
    diagrams = ripser(projected, maxdim=1)["dgms"]
    out = {"topology_available": True, "topology_samples": int(len(projected))}
    for dim, diagram in enumerate(diagrams[:2]):
        finite = diagram[np.isfinite(diagram[:, 1])]
        persistence = finite[:, 1] - finite[:, 0] if len(finite) else np.array([])
        out[f"h{dim}_count"] = int(len(finite))
        out[f"h{dim}_total_persistence"] = float(persistence.sum()) if len(persistence) else 0.0
        out[f"h{dim}_max_persistence"] = float(persistence.max()) if len(persistence) else 0.0
    return out
