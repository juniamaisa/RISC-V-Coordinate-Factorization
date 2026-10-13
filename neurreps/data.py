from __future__ import annotations

import hashlib
import json
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from processing.dataset_registry import _prepare_common_clean_samples, _source_paths
from .windowing import build_sample_from_chain

REPRESENTATIONS = ("SC", "4C", "32C")


@dataclass(frozen=True)
class AlignedBundle:
    """Aligned representations of exactly the same canonical instruction window."""

    arrays: dict[str, np.ndarray]
    masks: dict[str, np.ndarray]
    labels: np.ndarray
    sample_id: np.ndarray
    sample_hash: np.ndarray
    tensor_hashes: dict[str, np.ndarray]
    metadata: pd.DataFrame
    collision_report: pd.DataFrame
    S: int
    software: str



ALIGNED_CACHE_VERSION = "aligned-v1-canonical-Sx4-content-equivalent"


def _aligned_cache_key(
    mode_folder: str | Path,
    software: str,
    S: int,
    *,
    max_per_class: int | None,
    seed: int,
    remove_cross_label_tensor_collisions: bool,
) -> str:
    payload = {
        "version": ALIGNED_CACHE_VERSION,
        "software": str(software),
        "S": int(S),
        "max_per_class": max_per_class,
        "seed": int(seed),
        "remove_cross_label_tensor_collisions": bool(remove_cross_label_tensor_collisions),
        "sources": {},
    }
    for name, path in _source_paths(mode_folder, software).items():
        path = Path(path)
        payload["sources"][name] = (
            {"size": int(path.stat().st_size), "mtime_ns": int(path.stat().st_mtime_ns)}
            if path.exists()
            else {"missing": True}
        )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


def build_aligned_bundle_cached(
    mode_folder: str | Path,
    software: str,
    S: int,
    *,
    max_per_class: int | None = None,
    seed: int = 42,
    force_rebuild_common: bool = False,
    remove_cross_label_tensor_collisions: bool = True,
    use_cache: bool = True,
    cache_dir: str | Path | None = None,
    common: Mapping | None = None,
) -> AlignedBundle:
    """Load/store an exact AlignedBundle cache without changing its contents."""
    if not use_cache or force_rebuild_common:
        if common is not None and not force_rebuild_common:
            return _build_aligned_bundle_from_common(
                common, software, S, max_per_class=max_per_class, seed=seed,
                remove_cross_label_tensor_collisions=remove_cross_label_tensor_collisions,
            )
        return build_aligned_bundle(
            mode_folder, software, S, max_per_class=max_per_class, seed=seed,
            force_rebuild_common=force_rebuild_common,
            remove_cross_label_tensor_collisions=remove_cross_label_tensor_collisions,
        )
    key = _aligned_cache_key(
        mode_folder, software, S, max_per_class=max_per_class, seed=seed,
        remove_cross_label_tensor_collisions=remove_cross_label_tensor_collisions,
    )
    root = Path(cache_dir) if cache_dir is not None else Path(mode_folder) / ".neurreps_cache"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"aligned_{software}_S{int(S)}_{key}.pickle"
    if path.exists():
        with path.open("rb") as handle:
            bundle = pickle.load(handle)
        if not isinstance(bundle, AlignedBundle):
            raise RuntimeError(f"Invalid aligned-bundle cache: {path}")
        print(f"[CACHE] Aligned bundle loaded: {path}")
        return bundle
    bundle = (
        _build_aligned_bundle_from_common(
            common, software, S, max_per_class=max_per_class, seed=seed,
            remove_cross_label_tensor_collisions=remove_cross_label_tensor_collisions,
        )
        if common is not None and not force_rebuild_common
        else build_aligned_bundle(
            mode_folder, software, S, max_per_class=max_per_class, seed=seed,
            force_rebuild_common=force_rebuild_common,
            remove_cross_label_tensor_collisions=remove_cross_label_tensor_collisions,
        )
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)
    print(f"[CACHE] Aligned bundle saved: {path}")
    return bundle

def _central_block(chain: Sequence[str]) -> str:
    return str(chain[1]) if len(chain) >= 2 else (str(chain[0]) if chain else "")


def _slot_mask_from_meta(meta: Mapping, S: int) -> np.ndarray:
    """Create a valid-slot mask from builder metadata, never from token values."""
    mask = np.ones((1, int(S)), dtype=np.uint8)
    pad = int(meta.get("pad_added", 0))
    end = int(meta.get("end_token_count", 0))
    if pad <= 0:
        return mask
    if end > 0:
        start, stop = max(int(S) - 1 - pad, 0), max(int(S) - 1, 0)
    else:
        start, stop = max(int(S) - pad, 0), int(S)
    mask[:, start:stop] = 0
    return mask


def _sample_identifier(record: Mapping, software: str) -> str:
    return (
        f"{software}|{int(record.get('true_label', -1))}|"
        f"{int(record.get('source_index', -1))}|{record.get('sample_hash', '')}"
    )


def _tensor_hash(x: np.ndarray) -> str:
    arr = np.ascontiguousarray(x)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("ascii"))
    h.update(str(arr.dtype).encode("ascii"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _tensor_hash_vector(x: np.ndarray) -> np.ndarray:
    return np.asarray([_tensor_hash(row) for row in x], dtype="U64")


def _cross_label_collision_indices(
    tensor_hashes: Mapping[str, np.ndarray], labels: np.ndarray
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return the union of post-tensorization cross-label collisions.

    A collision in any primary representation is removed from every
    representation so comparisons retain identical sample identities.
    """
    y = np.asarray(labels).astype(int)
    remove: set[int] = set()
    rows: list[dict] = []
    for rep, hashes in tensor_hashes.items():
        buckets: dict[str, list[int]] = defaultdict(list)
        for i, value in enumerate(np.asarray(hashes).astype(str)):
            buckets[value].append(i)
        groups = 0
        involved = 0
        for value, idx in buckets.items():
            labs = set(y[idx].tolist())
            if labs == {0, 1}:
                groups += 1
                involved += len(idx)
                remove.update(idx)
        rows.append(
            {
                "representation": rep,
                "cross_label_collision_groups": int(groups),
                "rows_in_rep_collisions": int(involved),
            }
        )
    union = np.asarray(sorted(remove), dtype=np.int64)
    for row in rows:
        row["rows_removed_union"] = int(len(union))
    return union, pd.DataFrame(rows)


def _record_metadata(record: Mapping, meta: Mapping, S: int, sample_id: str) -> dict:
    chain = list(record.get("chain_blocks", []))
    slot_real_length = int(S) - int(meta.get("pad_added", 0))
    trunc_removed = int(meta.get("trunc_removed", 0))
    return {
        "sample_id": sample_id,
        "sample_hash": str(record.get("sample_hash", "")),
        "label": int(record.get("true_label", -1)),
        "origin": str(record.get("sample_origin", "")),
        "software": str(record.get("software", "")),
        "source_index": int(record.get("source_index", -1)),
        "chain_blocks": json.dumps(chain),
        "central_block": _central_block(chain),
        "raw_instruction_count": int(record.get("raw_instruction_count", 0)),
        "raw_byte_count": int(record.get("raw_byte_count", len(record.get("raw_bytes", b"")))),
        "compressed_instruction_count": int(record.get("compressed_instruction_count", 0)),
        "normal_instruction_count": int(record.get("normal_instruction_count", 0)),
        "compressed_ratio": float(record.get("compressed_ratio", 0.0)),
        "raw_zero_byte_ratio": float(record.get("raw_zero_byte_ratio", 0.0)),
        "pattern_0013_ratio": float(record.get("pattern_0013_ratio", 0.0)),
        "selected_instruction_count": int(meta.get("selected_instruction_count_before_trunc", 0)),
        "kept_instruction_count": int(meta.get("kept_original_instruction_count", 0)),
        "padding_rate": float(meta.get("padding_rate", 0.0)),
        "preserved_ratio": float(meta.get("preserved_ratio", 1.0)),
        "trunc_removed": trunc_removed,
        "invalid_transition_count": int(meta.get("selected_invalid_transition_count", 0)),
        "window_rule": str(meta.get("rule", "")),
        "S": int(S),
        # SC is now the flattened canonical slot window; all three inputs retain
        # exactly the same selected content.
        "sc_real_length": int(slot_real_length * 4),
        "sc_padding_count": int((S - slot_real_length) * 4),
        "sc_padding_rate": float((S - slot_real_length) / max(S, 1)),
        "sc_trunc_removed_bytes": int(trunc_removed * 4),
        "slot_real_length": slot_real_length,
        "slot_padding_count": int(meta.get("pad_added", 0)),
        "has_padding_sc": bool(slot_real_length < S),
        "has_padding_slots": bool(slot_real_length < S),
        "has_truncation_sc": bool(trunc_removed > 0),
        "has_truncation_slots": bool(trunc_removed > 0),
        "no_padding_all": bool(slot_real_length == S),
        "no_truncation_all": bool(trunc_removed == 0),
    }


def _sample_records(records: Sequence[Mapping], max_per_class: int | None, seed: int) -> list[Mapping]:
    if max_per_class is None:
        return list(records)
    rng = np.random.default_rng(seed)
    out: list[Mapping] = []
    for label in (0, 1):
        idx = np.asarray(
            [i for i, r in enumerate(records) if int(r["true_label"]) == label], dtype=np.int64
        )
        chosen = rng.choice(idx, size=min(int(max_per_class), len(idx)), replace=False)
        out.extend(records[int(i)] for i in chosen)
    out.sort(key=lambda r: (int(r["true_label"]), int(r.get("source_index", -1))))
    return out


def _build_aligned_bundle_from_common(
    common: Mapping,
    software: str,
    S: int,
    *,
    max_per_class: int | None = None,
    seed: int = 42,
    remove_cross_label_tensor_collisions: bool = True,
) -> AlignedBundle:
    """Build aligned views from an already loaded common dataset."""
    if S <= 0:
        raise ValueError("S must be positive")
    records = _sample_records(
        list(common["records_s1"]) + list(common["records_s2"]), max_per_class, seed
    )
    n = len(records)
    X_4c = np.empty((n, 4, S), dtype=np.uint8)
    M_slots = np.zeros((n, 1, S), dtype=np.uint8)
    labels = np.empty(n, dtype=np.uint8)
    sample_ids = np.empty(n, dtype="U160")
    sample_hashes = np.empty(n, dtype="U64")
    rows: list[dict] = []

    for i, record in enumerate(records):
        chain = list(record.get("chain_blocks", []))
        window, meta = build_sample_from_chain(
            chain, S, common["bb2instr"], common["bb2indirect"], common["SUCC_MAP"]
        )
        slot_window = np.asarray(window, dtype=np.uint8)
        if slot_window.shape != (S, 4):
            raise RuntimeError(f"Canonical window has shape {slot_window.shape}; expected {(S, 4)}")
        X_4c[i] = slot_window.T
        M_slots[i] = _slot_mask_from_meta(meta, S)
        labels[i] = int(record["true_label"])
        sample_ids[i] = _sample_identifier(record, software)
        sample_hashes[i] = str(record.get("sample_hash", ""))
        rows.append(_record_metadata(record, meta, S, sample_ids[i]))

    X_sc = X_4c.transpose(0, 2, 1).reshape(n, 1, 4 * S).copy()
    M_sc = np.repeat(M_slots.transpose(0, 2, 1), 4, axis=2).reshape(n, 1, 4 * S)
    X_32c = np.unpackbits(X_4c.transpose(0, 2, 1), axis=2).transpose(0, 2, 1).astype(np.uint8)
    arrays = {"SC": X_sc, "4C": X_4c, "32C": X_32c}
    masks = {"SC": M_sc, "4C": M_slots, "32C": M_slots.copy()}
    tensor_hashes = {rep: _tensor_hash_vector(X) for rep, X in arrays.items()}
    collision_idx, collision_report = _cross_label_collision_indices(tensor_hashes, labels)

    keep = np.ones(n, dtype=bool)
    if remove_cross_label_tensor_collisions and len(collision_idx):
        keep[collision_idx] = False
    arrays = {k: v[keep] for k, v in arrays.items()}
    masks = {k: v[keep] for k, v in masks.items()}
    tensor_hashes = {k: v[keep] for k, v in tensor_hashes.items()}
    labels = labels[keep]
    sample_ids = sample_ids[keep]
    sample_hashes = sample_hashes[keep]
    metadata = pd.DataFrame(rows).loc[keep].reset_index(drop=True)
    for rep, hashes in tensor_hashes.items():
        metadata[f"tensor_hash_{rep}"] = hashes
    if not np.array_equal(metadata["label"].to_numpy(dtype=np.uint8), labels):
        raise RuntimeError("metadata/label alignment failure")
    if not np.array_equal(arrays["SC"].reshape(len(labels), S, 4), arrays["4C"].transpose(0, 2, 1)):
        raise RuntimeError("SC and 4C are not exact views of the same canonical bytes")
    reconstructed_32c = np.unpackbits(arrays["4C"].transpose(0, 2, 1), axis=2).transpose(0, 2, 1)
    if not np.array_equal(reconstructed_32c, arrays["32C"]):
        raise RuntimeError("32C is not an exact bit expansion of 4C")

    collision_report["removed"] = bool(remove_cross_label_tensor_collisions)
    collision_report["samples_before"] = int(n)
    collision_report["samples_after"] = int(len(labels))
    return AlignedBundle(
        arrays=arrays,
        masks=masks,
        labels=labels,
        sample_id=sample_ids,
        sample_hash=sample_hashes,
        tensor_hashes=tensor_hashes,
        metadata=metadata,
        collision_report=collision_report,
        S=int(S),
        software=software,
    )


def build_aligned_bundle(
    mode_folder: str | Path,
    software: str,
    S: int,
    *,
    max_per_class: int | None = None,
    seed: int = 42,
    force_rebuild_common: bool = False,
    remove_cross_label_tensor_collisions: bool = True,
) -> AlignedBundle:
    """Build SC, 4C and 32C from one canonical ``S x 4`` slot window."""
    common = _prepare_common_clean_samples(
        Path(mode_folder), software, force_rebuild_common=force_rebuild_common
    )
    return _build_aligned_bundle_from_common(
        common,
        software,
        S,
        max_per_class=max_per_class,
        seed=seed,
        remove_cross_label_tensor_collisions=remove_cross_label_tensor_collisions,
    )

def save_aligned_bundle(bundle: AlignedBundle, out_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for rep, X in bundle.arrays.items():
        path = out_dir / f"aligned_{bundle.software}_{rep}_S{bundle.S}.npz"
        np.savez_compressed(
            path,
            X=X,
            mask=bundle.masks[rep],
            y=bundle.labels,
            sample_id=bundle.sample_id,
            sample_hash=bundle.sample_hash,
            tensor_hash=bundle.tensor_hashes[rep],
        )
        paths[rep] = path
    meta_path = out_dir / f"aligned_{bundle.software}_metadata_S{bundle.S}.csv"
    bundle.metadata.to_csv(meta_path, index=False)
    bundle.collision_report.to_csv(out_dir / f"tensor_collision_report_S{bundle.S}.csv", index=False)
    paths["metadata"] = meta_path
    return paths


def nuisance_matched_pairs(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    *,
    candidate_indices: np.ndarray | None = None,
    seed: int = 42,
    caliper: float = 0.35,
    features: Sequence[str] = (
        "raw_instruction_count",
        "raw_byte_count",
        "compressed_ratio",
        "raw_zero_byte_ratio",
        "pattern_0013_ratio",
        "padding_rate",
        "preserved_ratio",
        "trunc_removed",
    ),
) -> np.ndarray:
    """One-to-one matching without replacement in standardized nuisance space."""
    y = np.asarray(labels).astype(int)
    candidate = (
        np.arange(len(y), dtype=np.int64)
        if candidate_indices is None
        else np.asarray(candidate_indices, dtype=np.int64)
    )
    candidate = np.unique(candidate)
    if set(np.unique(y[candidate])) != {0, 1}:
        raise ValueError("nuisance matching requires both labels")
    missing = [c for c in features if c not in metadata.columns]
    if missing:
        raise KeyError(f"Nuisance features missing: {missing}")
    X = metadata.loc[:, list(features)].astype(float).replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    base = X.iloc[candidate]
    scale = base.std(ddof=0).replace(0.0, 1.0)
    Z = ((X - base.mean()) / scale).to_numpy(dtype=float)
    idx0 = candidate[y[candidate] == 0].copy()
    idx1 = candidate[y[candidate] == 1].copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    query, pool = (idx0, idx1) if len(idx0) <= len(idx1) else (idx1, idx0)
    nn = NearestNeighbors(n_neighbors=min(50, len(pool)), metric="euclidean").fit(Z[pool])
    distances, neighbors = nn.kneighbors(Z[query])
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for q, ds, ns in zip(query, distances, neighbors):
        chosen = None
        for d, local in zip(ds, ns):
            p = int(pool[int(local)])
            if p not in used and float(d) <= float(caliper):
                chosen = p
                break
        if chosen is None:
            continue
        used.add(chosen)
        pairs.append((int(q), chosen) if y[int(q)] == 0 else (chosen, int(q)))
    if not pairs:
        raise RuntimeError("No nuisance-matched pairs found; relax the caliper")
    return np.asarray(pairs, dtype=np.int64)


def nuisance_matched_indices(metadata: pd.DataFrame, labels: np.ndarray, **kwargs) -> np.ndarray:
    pairs = nuisance_matched_pairs(metadata, labels, **kwargs)
    return np.asarray(sorted(pairs.reshape(-1).tolist()), dtype=np.int64)


def exact_length_matched_pairs(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    *,
    candidate_indices: np.ndarray | None = None,
    seed: int = 42,
    fields: Sequence[str] = ("slot_real_length",),
) -> np.ndarray:
    y = np.asarray(labels).astype(int)
    candidate = np.arange(len(y), dtype=np.int64) if candidate_indices is None else np.asarray(candidate_indices, dtype=np.int64)
    missing = [f for f in fields if f not in metadata.columns]
    if missing:
        raise KeyError(f"Exact-length fields missing: {missing}")
    rng = np.random.default_rng(seed)
    buckets: dict[tuple, dict[int, list[int]]] = defaultdict(lambda: {0: [], 1: []})
    for i in candidate:
        signature = tuple(int(metadata.iloc[int(i)][field]) for field in fields)
        buckets[signature][int(y[int(i)])].append(int(i))
    pairs: list[tuple[int, int]] = []
    for signature in sorted(buckets):
        a = np.asarray(buckets[signature][0], dtype=np.int64)
        b = np.asarray(buckets[signature][1], dtype=np.int64)
        rng.shuffle(a)
        rng.shuffle(b)
        pairs.extend((int(a[j]), int(b[j])) for j in range(min(len(a), len(b))))
    if not pairs:
        raise RuntimeError("No exact real-length matches found")
    return np.asarray(pairs, dtype=np.int64)


def filter_complete_pairs(pairs: np.ndarray, allowed_indices: np.ndarray) -> np.ndarray:
    allowed = set(np.asarray(allowed_indices, dtype=np.int64).tolist())
    kept = [(int(a), int(b)) for a, b in np.asarray(pairs, dtype=np.int64) if int(a) in allowed and int(b) in allowed]
    return np.asarray(kept, dtype=np.int64).reshape(-1, 2) if kept else np.empty((0, 2), dtype=np.int64)


def cap_matched_pairs(pairs: np.ndarray, max_pairs: int, seed: int) -> np.ndarray:
    pairs = np.asarray(pairs, dtype=np.int64)
    if len(pairs) <= int(max_pairs):
        return pairs
    rng = np.random.default_rng(seed)
    return pairs[np.sort(rng.choice(len(pairs), int(max_pairs), replace=False))]


def no_padding_indices(
    masks: Mapping[str, np.ndarray],
    *,
    representations: Sequence[str] = REPRESENTATIONS,
    candidate_indices: np.ndarray | None = None,
) -> np.ndarray:
    n = len(next(iter(masks.values())))
    keep = np.ones(n, dtype=bool)
    for rep in representations:
        keep &= np.all(np.asarray(masks[rep]).reshape(n, -1) > 0, axis=1)
    idx = np.flatnonzero(keep)
    if candidate_indices is not None:
        idx = np.intersect1d(idx, np.asarray(candidate_indices, dtype=np.int64))
    return idx.astype(np.int64)


def no_truncation_indices(metadata: pd.DataFrame, *, candidate_indices: np.ndarray | None = None) -> np.ndarray:
    idx = np.flatnonzero(metadata["no_truncation_all"].astype(bool).to_numpy())
    if candidate_indices is not None:
        idx = np.intersect1d(idx, np.asarray(candidate_indices, dtype=np.int64))
    return idx.astype(np.int64)


def connected_match_groups(
    sample_id: np.ndarray,
    selected: np.ndarray,
    pairs: np.ndarray | None = None,
    *,
    duplicate_keys: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    """Make matched pairs and duplicate tensors indivisible split components."""
    selected = np.asarray(selected, dtype=np.int64)
    pos = {int(global_i): local_i for local_i, global_i in enumerate(selected)}
    parent = np.arange(len(selected), dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return int(x)

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    if pairs is not None:
        for a, b in np.asarray(pairs, dtype=np.int64).reshape(-1, 2):
            if int(a) in pos and int(b) in pos:
                union(pos[int(a)], pos[int(b)])
    keys = [np.asarray(sample_id).astype(str)]
    if duplicate_keys:
        keys.extend(np.asarray(k).astype(str) for k in duplicate_keys)
    for key_values in keys:
        first: dict[str, int] = {}
        for local_i, global_i in enumerate(selected):
            key = key_values[int(global_i)]
            if key in first:
                union(local_i, first[key])
            else:
                first[key] = local_i
    roots = [find(i) for i in range(len(selected))]
    remap = {root: j for j, root in enumerate(sorted(set(roots)))}
    return np.asarray([f"component_{remap[root]:08d}" for root in roots], dtype="U32")


def stratified_cap_indices(labels: np.ndarray, max_per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.asarray(labels).astype(int)
    chosen: list[int] = []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        chosen.extend(rng.choice(idx, min(len(idx), int(max_per_class)), replace=False).tolist())
    return np.asarray(sorted(chosen), dtype=np.int64)


def common_sample_ids(bundles: Mapping[int, AlignedBundle]) -> set[str]:
    sets = [set(bundle.sample_id.astype(str).tolist()) for bundle in bundles.values()]
    return set.intersection(*sets) if sets else set()


def indices_for_sample_ids(bundle: AlignedBundle, identifiers: set[str]) -> np.ndarray:
    return np.flatnonzero(np.isin(bundle.sample_id.astype(str), list(identifiers))).astype(np.int64)


def build_arrays_from_chains(common: Mapping, chains: Sequence[Sequence[str]], S: int) -> dict[str, np.ndarray]:
    """Encode arbitrary chains using the same canonical content for every view."""
    n = len(chains)
    X_4c = np.empty((n, 4, S), dtype=np.uint8)
    M_slots = np.empty((n, 1, S), dtype=np.uint8)
    for i, chain in enumerate(chains):
        window, meta = build_sample_from_chain(
            list(chain), S, common["bb2instr"], common["bb2indirect"], common["SUCC_MAP"]
        )
        X_4c[i] = np.asarray(window, dtype=np.uint8).T
        M_slots[i] = _slot_mask_from_meta(meta, S)
    X_sc = X_4c.transpose(0, 2, 1).reshape(n, 1, 4 * S).copy()
    M_sc = np.repeat(M_slots.transpose(0, 2, 1), 4, axis=2).reshape(n, 1, 4 * S)
    X_32c = np.unpackbits(X_4c.transpose(0, 2, 1), axis=2).transpose(0, 2, 1).astype(np.uint8)
    return {
        "arrays": {"SC": X_sc, "4C": X_4c, "32C": X_32c},
        "masks": {"SC": M_sc, "4C": M_slots, "32C": M_slots.copy()},
    }
