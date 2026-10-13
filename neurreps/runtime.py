from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd


class RuntimeProfiler:
    """Low-overhead stage timer persisted during long experiments."""

    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict[str, Any]] = []
        self.jsonl_path = self.out_dir / "runtime_profile.jsonl"
        if self.jsonl_path.exists():
            self.jsonl_path.unlink()

    @contextmanager
    def stage(self, name: str, **metadata: Any):
        start = time.perf_counter()
        status = "ok"
        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            row = {
                "stage": str(name),
                "seconds": float(time.perf_counter() - start),
                "status": status,
                **metadata,
            }
            self.rows.append(row)
            with self.jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")

    def save(self) -> None:
        frame = pd.DataFrame(self.rows)
        frame.to_csv(self.out_dir / "runtime_profile.csv", index=False)
        if not frame.empty:
            summary = (
                frame.groupby("stage", dropna=False)["seconds"]
                .agg(["count", "sum", "mean", "max"])
                .sort_values("sum", ascending=False)
                .reset_index()
            )
            summary.to_csv(self.out_dir / "runtime_profile_summary.csv", index=False)
