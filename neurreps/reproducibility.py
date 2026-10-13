from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from processing.dataset_registry import _source_paths


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_json(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_strings(values: Iterable[str]) -> str:
    h = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        h.update(len(encoded).to_bytes(8, "little"))
        h.update(encoded)
    return h.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("ascii"))
    h.update(str(arr.dtype).encode("ascii"))
    h.update(arr.tobytes())
    return h.hexdigest()


def environment_manifest() -> dict:
    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "python_executable_name": Path(sys.executable).name,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "packages": packages,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "torch_cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else [],
        "torch_deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_deterministic": bool(getattr(torch.backends.cudnn, "deterministic", False)),
        "cudnn_benchmark": bool(getattr(torch.backends.cudnn, "benchmark", False)),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def source_dataset_manifest(
    mode_folder: str | Path,
    software: str,
    *,
    use_hash_cache: bool = True,
) -> dict:
    """Hash source files without leaking author-specific absolute paths.

    SHA-256 is cached by file size and nanosecond mtime under the dataset's
    local ``.neurreps_cache`` directory. This makes the first provenance pass
    cryptographic and later main/ablation runs inexpensive while preserving
    the recorded content hash for unchanged files.
    """
    mode_folder = Path(mode_folder)
    paths = _source_paths(mode_folder, software)
    cache_dir = mode_folder / ".neurreps_cache"
    cache_path = cache_dir / "file_hashes.json"
    cache: dict[str, dict] = {}
    if use_hash_cache and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cache = {}
    rows = {}
    changed = False
    for name, path in paths.items():
        path = Path(path)
        exists = path.exists()
        size = int(path.stat().st_size) if exists else None
        mtime = int(path.stat().st_mtime_ns) if exists else None
        key = str(path)
        cached = cache.get(key, {})
        hit = bool(
            exists
            and use_hash_cache
            and cached.get("size_bytes") == size
            and cached.get("mtime_ns") == mtime
            and cached.get("sha256")
        )
        digest = str(cached["sha256"]) if hit else (sha256_file(path) if exists else None)
        if exists and not hit:
            cache[key] = {"size_bytes": size, "mtime_ns": mtime, "sha256": digest}
            changed = True
        try:
            display_path = str(path.relative_to(mode_folder.parent))
        except ValueError:
            display_path = path.name
        rows[name] = {
            "path": display_path,
            "exists": exists,
            "size_bytes": size,
            "mtime_ns": mtime,
            "sha256": digest,
            "hash_cache_used": hit,
        }
    if use_hash_cache and changed:
        cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(cache_path)
    return rows

def code_manifest(root: str | Path) -> dict:
    root = Path(root)
    files = []
    for folder in (root / "neurreps", root / "processing"):
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*.py")):
            files.append({"path": str(path.relative_to(root)), "sha256": sha256_file(path)})
    return {"files": files, "code_sha256": sha256_json(files)}


def write_reproducibility_manifest(
    out_dir: str | Path,
    *,
    config: dict,
    project_root: str | Path,
    mode_folder: str | Path,
    software: str,
    use_hash_cache: bool = True,
) -> dict:
    out_dir = Path(out_dir)
    manifest = {
        "config_sha256": sha256_json(config),
        "environment": environment_manifest(),
        "dataset_sources": source_dataset_manifest(mode_folder, software, use_hash_cache=use_hash_cache),
        "code": code_manifest(project_root),
    }
    (out_dir / "reproducibility_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest
