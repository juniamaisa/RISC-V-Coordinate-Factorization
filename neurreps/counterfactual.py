from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def load_weak_blocks(path: str | Path | None) -> set[str] | None:
    if path is None:
        return None
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and "cra" in payload:
        return {str(value) for value in payload["cra"]}
    raise ValueError("gadget file must be a dict containing the 'cra' list")


def _block_len(common: Mapping, block: str) -> int:
    return len(common["bb2instr"].get(str(block), []))


def _length_buckets(common: Mapping, candidates: Sequence[str]) -> dict[int, list[str]]:
    buckets: dict[int, list[str]] = defaultdict(list)
    for block in candidates:
        length = _block_len(common, str(block))
        if length > 0:
            buckets[length].append(str(block))
    return buckets


def _candidate_lengths(target: int, tolerance: int):
    yield target
    for delta in range(1, int(tolerance) + 1):
        if target - delta > 0:
            yield target - delta
        yield target + delta


def _record_allowed(record: Mapping, allowed_sample_hashes: set[str] | None, allowed_source_indices: set[int] | None) -> bool:
    if allowed_sample_hashes is not None and str(record.get("sample_hash", "")) not in allowed_sample_hashes:
        return False
    if allowed_source_indices is not None and int(record.get("source_index", -1)) not in allowed_source_indices:
        return False
    return True


def generate_counterfactual_and_control_pairs(
    common: Mapping,
    *,
    max_pairs_per_stratum: int = 1000,
    seed: int = 42,
    weak_blocks: set[str] | None = None,
    length_tolerance: int = 0,
    exact_length: bool = True,
    allowed_sample_hashes: set[str] | None = None,
    allowed_source_indices: set[int] | None = None,
    require_all_strata: bool = True,
    match_entry_weak_status: bool = True,
) -> list[dict]:
    """Generate base-matched invalid and valid substitutions for entry/exit.

    For each retained benign context and changed position, the generator only
    accepts the record when it can construct both a CFG-invalid substitution
    and a CFG-valid substitution from the same benign base. By default,
    ``exact_length=True`` requires the original block, invalid replacement and
    valid replacement to have exactly the same instruction count. The legacy
    tolerance mode is retained only for explicit sensitivity experiments.
    """
    rng = np.random.default_rng(seed)
    records = [
        record
        for record in common["records_s1"]
        if _record_allowed(record, allowed_sample_hashes, allowed_source_indices)
    ]
    rng.shuffle(records)
    succ = {
        str(src): {str(dst) for dst in destinations}
        for src, destinations in common["SUCC_MAP"].items()
    }
    all_blocks = sorted(
        str(block) for block, instructions in common["bb2instr"].items() if len(instructions) > 0
    )
    weak = set(all_blocks) if weak_blocks is None else set(all_blocks).intersection(weak_blocks)
    all_buckets = _length_buckets(common, all_blocks)
    weak_buckets = _length_buckets(common, sorted(weak))
    predecessors: dict[str, set[str]] = defaultdict(set)
    for src, destinations in succ.items():
        for dst in destinations:
            predecessors[dst].add(src)

    matched_groups: dict[str, list[dict[str, dict]]] = {"entry": [], "exit": []}

    def choose(buckets, target_len: int, predicate, forbidden: set[str] | None = None) -> str | None:
        forbidden = forbidden or set()
        candidate_lengths = (target_len,) if exact_length else tuple(
            _candidate_lengths(target_len, length_tolerance)
        )
        for length in candidate_lengths:
            pool = buckets.get(length, [])
            if not pool:
                continue
            for local in rng.permutation(len(pool))[: min(len(pool), 500)]:
                candidate = str(pool[int(local)])
                if candidate not in forbidden and predicate(candidate):
                    return candidate
        return None

    def make_row(record, chain, modified, control_type, attack_type, changed_position, group_id):
        invalid_edges = [
            [src, dst]
            for src, dst in zip(modified[:-1], modified[1:])
            if dst not in succ.get(src, set())
        ]
        expected = 1 if control_type == "invalid" else 0
        if len(invalid_edges) != expected:
            return None
        replacement = modified[changed_position]
        return {
            "comparison_group_id": group_id,
            "benign_sample_hash": str(record.get("sample_hash", "")),
            "benign_source_index": int(record.get("source_index", -1)),
            "control_type": control_type,
            "attack_type": attack_type,
            "benign_chain": chain,
            "modified_chain": modified,
            "malicious_chain": modified,
            "central_block": chain[1],
            "changed_position": int(changed_position),
            "invalid_edges": invalid_edges,
            "original_block_length": _block_len(common, chain[changed_position]),
            "replacement_block_length": _block_len(common, replacement),
            "length_difference": int(
                _block_len(common, replacement) - _block_len(common, chain[changed_position])
            ),
            "exact_length_match": bool(
                _block_len(common, replacement) == _block_len(common, chain[changed_position])
            ),
            "replacement_is_weak": bool(replacement in weak),
        }

    for record in records:
        chain = [str(value) for value in record.get("chain_blocks", [])]
        if len(chain) != 3:
            continue
        b1, b2, b3 = chain
        base_token = f"{int(record.get('source_index', -1))}_{str(record.get('sample_hash', ''))[:16]}"

        # Exit substitutions share b1,b2 and replace b3. When gadget metadata
        # is supplied, the central transfer point b2 must be weak for the whole
        # invalid/control comparison.
        if len(matched_groups["exit"]) < int(max_pairs_per_stratum):
            if weak_blocks is None or b2 in weak:
                target_len = _block_len(common, b3)
                invalid_replacement = choose(
                    all_buckets,
                    target_len,
                    lambda candidate: candidate not in succ.get(b2, set()),
                    {b3},
                )
                valid_replacement = choose(
                    all_buckets,
                    target_len,
                    lambda candidate: candidate in succ.get(b2, set()),
                    {b3, invalid_replacement} if invalid_replacement else {b3},
                )
                if invalid_replacement is not None and valid_replacement is not None:
                    if exact_length:
                        lengths = {
                            _block_len(common, b3),
                            _block_len(common, invalid_replacement),
                            _block_len(common, valid_replacement),
                        }
                        if len(lengths) != 1:
                            continue
                    group_id = f"exit_{base_token}"
                    invalid_chain = [b1, b2, invalid_replacement]
                    valid_chain = [b1, b2, valid_replacement]
                    invalid_row = make_row(
                        record, chain, invalid_chain, "invalid", "exit", 2, group_id
                    )
                    valid_row = make_row(record, chain, valid_chain, "valid", "exit", 2, group_id)
                    if invalid_row is not None and valid_row is not None:
                        matched_groups["exit"].append({"invalid": invalid_row, "valid": valid_row})

        # Entry substitutions share b2,b3 and replace b1. The invalid entry
        # candidate may be restricted to weak blocks; its valid control is
        # drawn from the same length neighbourhood and the same base context.
        if len(matched_groups["entry"]) < int(max_pairs_per_stratum):
            target_len = _block_len(common, b1)
            invalid_replacement = choose(
                weak_buckets,
                target_len,
                lambda candidate: b2 not in succ.get(candidate, set()),
                {b1},
            )
            entry_control_buckets = (
                weak_buckets
                if weak_blocks is not None and bool(match_entry_weak_status)
                else all_buckets
            )
            valid_replacement = choose(
                entry_control_buckets,
                target_len,
                lambda candidate: candidate in predecessors.get(b2, set()),
                {b1, invalid_replacement} if invalid_replacement else {b1},
            )
            if invalid_replacement is not None and valid_replacement is not None:
                if exact_length:
                    lengths = {
                        _block_len(common, b1),
                        _block_len(common, invalid_replacement),
                        _block_len(common, valid_replacement),
                    }
                    if len(lengths) != 1:
                        continue
                group_id = f"entry_{base_token}"
                invalid_chain = [invalid_replacement, b2, b3]
                valid_chain = [valid_replacement, b2, b3]
                invalid_row = make_row(
                    record, chain, invalid_chain, "invalid", "entry", 0, group_id
                )
                valid_row = make_row(record, chain, valid_chain, "valid", "entry", 0, group_id)
                if invalid_row is not None and valid_row is not None:
                    matched_groups["entry"].append({"invalid": invalid_row, "valid": valid_row})

        if all(len(groups) >= int(max_pairs_per_stratum) for groups in matched_groups.values()):
            break

    available = {attack: len(groups) for attack, groups in matched_groups.items()}
    if require_all_strata and min(available.values(), default=0) == 0:
        missing = [attack for attack, count in available.items() if count == 0]
        raise RuntimeError(f"Could not construct base-matched counterfactual/control pairs: {missing}")
    active_counts = [count for count in available.values() if count > 0]
    if not active_counts:
        raise RuntimeError("No counterfactual/control pairs could be constructed")
    n = min(min(active_counts), int(max_pairs_per_stratum))
    output: list[dict] = []
    for attack_type in ("entry", "exit"):
        groups = matched_groups[attack_type]
        if not groups:
            continue
        chosen = (
            np.sort(rng.choice(len(groups), n, replace=False))
            if len(groups) > n
            else np.arange(n)
        )
        for group_index in chosen:
            group = groups[int(group_index)]
            for control_type in ("invalid", "valid"):
                row = dict(group[control_type])
                row["pair_id"] = (
                    f"{row['comparison_group_id']}_{control_type}_{len(output):06d}"
                )
                output.append(row)
    return output

def generate_matched_counterfactual_pairs(common: Mapping, **kwargs) -> list[dict]:
    """Backward-compatible wrapper returning only invalid pairs."""
    max_pairs = int(kwargs.pop("max_pairs", 2000))
    pairs = generate_counterfactual_and_control_pairs(
        common,
        max_pairs_per_stratum=max(1, max_pairs // 2),
        require_all_strata=False,
        **kwargs,
    )
    return [row for row in pairs if row["control_type"] == "invalid"][:max_pairs]


def save_pairs(pairs: Sequence[Mapping], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in pairs:
            handle.write(json.dumps(dict(row)) + "\n")
    return path
