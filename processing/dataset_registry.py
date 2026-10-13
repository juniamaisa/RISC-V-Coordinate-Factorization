from dataclasses import dataclass
import csv
import json
import os
import hashlib
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    """
    Especificação de um dataset processado.

    Campos:
        key:
            Nome único usado nos registries.

        name:
            Tipo lógico do dataset.
            Valores esperados:
                - single_channel
                - four_channel
                - bit32_channel
                - embedding_seq

        S:
            Número de instruções no contexto.
            Cada instrução tem 4 bytes.
            Ex:
                S=32 -> 128 bytes
                S=16 -> 64 bytes

        max_len:
            Comprimento em bytes para representação single-channel.
            Ex:
                max_len=128
                max_len=64

        channels:
            Número de canais do tensor final.

        shape_type:
            Organização do tensor final.
            Valores esperados:
                - channels_first
                - sequence_first
    """

    key: str
    name: str
    S: int | None
    max_len: int | None
    channels: int
    shape_type: str




# ============================================================
# Basic-block instruction parsing and CFG utilities
# ============================================================
# =========================
# Cell 2 — Tokens + parsing (R1/R2/R3)
# =========================
PAD_INST   = [0x13, 0x00, 0x00, 0x00]  # addi x0, x0, 0
TOKEN_CUT  = [0x00, 0x00, 0xFF, 0xFF]
TOKEN_END  = [0x00, 0xFF, 0x00, 0xFF] 
#TOKEN_CUT = [0xFE, 0xFE, 0xFE, 0xFE]
#TOKEN_END = [0xFD, 0xFD, 0xFD, 0xFD]


INDIRECT_MNEM = {"jalr", "jr", "ret", "c.jr", "c.jalr"}

def hex_line_to_raw_bytes(line: str):
    """
    Extract the original byte sequence from one ``hex ...`` line.

    This function intentionally preserves the real instruction size.
    For example, a compressed RISC-V instruction represented by two bytes
    remains a 2-byte list. No padding, truncation, or conversion to a
    4-byte instruction slot is applied here.
    """
    if not (isinstance(line, str) and line.startswith("hex ")):
        return None

    hex_part = line.split("hex ", 1)[1].strip()
    toks = hex_part.split() if hex_part else []

    out = []
    for t in toks:
        try:
            out.append(int(t, 16) & 0xFF)
        except ValueError:
            continue

    return out if out else None


def hex_line_to_inst4(line: str):
    """
    R1 — one 'hex ...' line becomes exactly one instruction slot with 4 bytes:
      - 2 bytes, compressed instruction: [b0,b1,0,0]
      - 4 bytes: [b0,b1,b2,b3]
      - >4 bytes: truncate
      - <4 bytes: zero-pad
    """
    if not (isinstance(line, str) and line.startswith("hex ")):
        return None
    hex_part = line.split("hex ", 1)[1].strip()
    toks = hex_part.split() if hex_part else []

    b = []
    for t in toks:
        try:
            b.append(int(t, 16) & 0xFF)
        except ValueError:
            continue

    if len(b) >= 4:
        return b[:4]
    if len(b) == 3:
        return [b[0], b[1], b[2], 0x00]
    if len(b) == 2:
        return [b[0], b[1], 0x00, 0x00]
    if len(b) == 1:
        return [b[0], 0x00, 0x00, 0x00]
    return PAD_INST[:]  # PAD_INST


# 


def extract_bb_raw_instructions(bb_id: str, blocks_data: dict):
    """
    Preserve ordered raw instruction bytes from ``hex ...`` lines.

    Output example:
        [[0x82, 0x80], [0x13, 0x05, 0x00, 0x00], ...]

    This is the correct extraction for the common stage and for
    ``single_channel`` before max_len padding/truncation.
    """
    out = []
    for ln in blocks_data.get(bb_id, []):
        raw = hex_line_to_raw_bytes(ln)
        if raw is not None:
            out.append(raw)
    return out


def extract_bb_instructions(bb_id: str, blocks_data: dict):
    """
    Preserve order of ``hex ...`` lines and convert each instruction to a
    fixed 4-byte slot. This is used by ``four_channel`` and by
    ``bit32_channel`` generation only.
    """
    out = []
    for ln in blocks_data.get(bb_id, []):
        inst4 = hex_line_to_inst4(ln)
        if inst4 is not None:
            out.append(inst4)
    return out

def bb_has_indirect_jump(bb_id: str, blocks_data: dict) -> bool:
    """
    R3 — mark BB as indirect if any opcode mnemonic is:
      jalr/jr/ret/c.jr/c.jalr or endswith 'jalr'
    """
    for ln in blocks_data.get(bb_id, []):
        if isinstance(ln, str) and ln.startswith("opcode "):
            rest = ln.split("opcode ", 1)[1].strip()
            if not rest:
                continue
            mnemonic = rest.split()[0].strip()
            if mnemonic in INDIRECT_MNEM:
                return True
            if mnemonic.endswith("jalr"):
                return True
            low = rest.lower()
            if ("jalr" in low) or low.startswith("jr") or low.startswith("ret") or ("c.jr" in low) or ("c.jalr" in low):
                return True
    return False

# =========================
# NEW CELL — Metrics helpers
# =========================
def build_successor_map(adjlist):
    succ = {}
    for src, edges in adjlist.items():
        succ[src] = set(dst for _et, dst in edges)
    return succ

def count_invalid_transitions(chain, succ_map):
    """
    Conta transições inválidas consecutivas em uma cadeia:
      (b1->b2), (b2->b3), ...
    Uma transição é inválida se dst não pertence aos sucessores de src no CFG final.
    """
    invalid_pairs = []
    chain = [str(x) for x in chain]

    for a, b in zip(chain[:-1], chain[1:]):
        if b not in succ_map.get(a, set()):
            invalid_pairs.append((a, b))

    return len(invalid_pairs), invalid_pairs

def count_chain_instructions(chain_blocks, bb2instr):
    """
    Soma o número de instruções (já extraídas como vetores de 4 bytes) dos blocos da cadeia.
    """
    total = 0
    per_block = []
    for bb in chain_blocks:
        n = len(bb2instr.get(bb, []))
        per_block.append((bb, n))
        total += n
    return total, per_block

def extract_bb_mnemonics(bb_id, blocks_data):
    """
    Extrai apenas o mnemônico das linhas 'opcode ...' de um basic block.
    """
    out = []
    for ln in blocks_data.get(bb_id, []):
        if isinstance(ln, str) and ln.startswith("opcode "):
            rest = ln.split("opcode ", 1)[1].strip()
            if not rest:
                continue
            mnemonic = rest.split()[0].strip().lower()
            out.append(mnemonic)
    return out

def extract_chain_mnemonics(chain_blocks, blocks_data):
    """
    Extrai mnemônicos concatenados dos blocos da cadeia, preservando ordem.
    """
    out = []
    for bb in chain_blocks:
        out.extend(extract_bb_mnemonics(bb, blocks_data))
    return out

# ============================================================
# CRA DATASET GENERATION FLOWCHART
# ============================================================
#
#                    ┌──────────────────────────────┐
#                    │ blocksdata{software}.pickle  │
#                    │ s1{software}.pickle          │
#                    │ s2{software}.pickle          │
#                    │ adjlist{software}.pickle     │
#                    └───────────────┬──────────────┘
#                                    │
#                                    v
#                    ┌──────────────────────────────┐
#                    │ Load basic blocks, S1, S2    │
#                    │ and final CFG/eCFG           │
#                    └───────────────┬──────────────┘
#                                    │
#                                    v
#                    ┌──────────────────────────────┐
#                    │ For each S1/S2 chain:        │
#                    │ [BB1, BB2, BB3, ...]         │
#                    └───────────────┬──────────────┘
#                                    │
#                                    v
#                    ┌──────────────────────────────┐
#                    │ Extract ordered "hex ..."    │
#                    │ instructions from blocksdata │
#                    └───────────────┬──────────────┘
#                                    │
#                                    v
#                    ┌──────────────────────────────┐
#                    │ COMMON BYTE-STAGE FILTERING  │
#                    │ BEFORE ANY INPUT PADDING     │
#                    │                              │
#                    │ 1. Convert chains to raw     │
#                    │    byte representation       │
#                    │ 2. Save equal S1/S2 samples  │
#                    │    to CSV and remove both    │
#                    │ 3. Preserve duplicates       │
#                    │    inside the same class     │
#                    │ 4. Do not balance classes    │
#                    └───────────────┬──────────────┘
#                                    │
#              ┌─────────────────────┴─────────────────────┐
#              │                                           │
#              v                                           v
# ┌─────────────────────────────┐             ┌─────────────────────────────┐
# │ INPUT 1: SINGLE CHANNEL     │             │ INPUT 2: FOUR CHANNEL       │
# └──────────────┬──────────────┘             └──────────────┬──────────────┘
#                │                                           │
#                v                                           v
# ┌─────────────────────────────┐             ┌─────────────────────────────┐
# │ Flatten raw instruction     │             │ Apply four-channel window   │
# │ bytes into one byte vector  │             │ rules using instruction     │
# │                             │             │ slots of 4 bytes            │
# │ [b0,b1,b2,...]             │             │                             │
# │                             │             │ inst=[b0,b1,b2,b3]          │
# └──────────────┬──────────────┘             └──────────────┬──────────────┘
#                │                                           │
#                v                                           v
# ┌─────────────────────────────┐             ┌─────────────────────────────┐
# │ Single-channel padding:     │             │ Four-channel padding:       │
# │ pad/truncate to max_len     │             │ pad/truncate to S slots     │
# │ with byte value 0           │             │ using PAD/CUT/END tokens    │
# └──────────────┬──────────────┘             └──────────────┬──────────────┘
#                │                                           │
#                v                                           v
# ┌─────────────────────────────┐             ┌─────────────────────────────┐
# │ Final tensor:               │             │ Final tensor:               │
# │ X = (N, 1, L)               │             │ X = (N, 4, S)               │
# │ L = 64 or 128               │             │ S = 16 or 32                │
# └─────────────────────────────┘             └──────────────┬──────────────┘
#                                                            │
#                                                            v
#                                            ┌─────────────────────────────┐
#                                            │ INPUT 3: BIT32 CHANNEL      │
#                                            │ derived from four_channel   │
#                                            └──────────────┬──────────────┘
#                                                           │
#                                                           v
#                                            ┌─────────────────────────────┐
#                                            │ (N,4,S) -> (N,S,4)          │
#                                            │ np.unpackbits -> (N,S,32)   │
#                                            │ transpose -> (N,32,S)       │
#                                            └──────────────┬──────────────┘
#                                                           │
#                                                           v
#                                            ┌─────────────────────────────┐
#                                            │ Final tensor:               │
#                                            │ X = (N, 32, S)              │
#                                            │ S = 16 or 32                │
#                                            └─────────────────────────────┘
#
# ============================================================

DATASET_SPECS = {
    # ============================================================
    # Single-channel input: (N, 1, L)
    # ============================================================
    # S=32 -> 128 bytes
    "single_channel_128": DatasetSpec(
        key="single_channel_128",
        name="single_channel",
        S=None,
        max_len=128,
        channels=1,
        shape_type="channels_first",
    ),

    # S=16 -> 64 bytes
    "single_channel_64": DatasetSpec(
        key="single_channel_64",
        name="single_channel",
        S=None,
        max_len=64,
        channels=1,
        shape_type="channels_first",
    ),

    # ============================================================
    # Four-channel input: (N, 4, S)
    # ============================================================
    # S=32 -> 128 bytes
    "four_channel_32": DatasetSpec(
        key="four_channel_32",
        name="four_channel",
        S=32,
        max_len=None,
        channels=4,
        shape_type="channels_first",
    ),

    # S=16 -> 64 bytes
    "four_channel_16": DatasetSpec(
        key="four_channel_16",
        name="four_channel",
        S=16,
        max_len=None,
        channels=4,
        shape_type="channels_first",
    ),

    # ============================================================
    # Bit-unpacked input: (N, 32, S)
    # ============================================================
    # S=32 -> 128 bytes unpacked into 32 bit-channels
    "bit32_channel_32": DatasetSpec(
        key="bit32_channel_32",
        name="bit32_channel",
        S=32,
        max_len=None,
        channels=32,
        shape_type="channels_first",
    ),

    # S=16 -> 64 bytes unpacked into 32 bit-channels
    "bit32_channel_16": DatasetSpec(
        key="bit32_channel_16",
        name="bit32_channel",
        S=16,
        max_len=None,
        channels=32,
        shape_type="channels_first",
    ),

    # ============================================================
    # Embedding sequence-first input: (N, S, 4)
    # ============================================================
    # S=32 -> 128 bytes organized as 32 instructions x 4 bytes
    "embedding_seq_32": DatasetSpec(
        key="embedding_seq_32",
        name="embedding_seq",
        S=32,
        max_len=None,
        channels=4,
        shape_type="sequence_first",
    ),

    # S=16 -> 64 bytes organized as 16 instructions x 4 bytes
    "embedding_seq_16": DatasetSpec(
        key="embedding_seq_16",
        name="embedding_seq",
        S=16,
        max_len=None,
        channels=4,
        shape_type="sequence_first",
    ),
}




# ---------------------------------------------------------------------------
# Dataset-shape semantics moved from processing.main
# ---------------------------------------------------------------------------
def bytes_from_S(S):
    """Converte número de instruções em bytes. Cada instrução tem 4 bytes."""
    return int(S) * 4


def describe_input_representation(dataset_spec):
    """Returns human-readable input semantics for the result table."""
    import math

    if dataset_spec.name == "single_channel":
        tensor_shape = f"1 x {dataset_spec.max_len}"
        temporal_axis = "byte position"
        per_position_content = "one raw byte"
        equivalent_raw_context_bytes = dataset_spec.max_len
    elif dataset_spec.name == "four_channel":
        tensor_shape = f"4 x {dataset_spec.S}"
        temporal_axis = "instruction slot"
        per_position_content = "four instruction bytes"
        equivalent_raw_context_bytes = dataset_spec.S * 4
    elif dataset_spec.name == "bit32_channel":
        tensor_shape = f"32 x {dataset_spec.S}"
        temporal_axis = "instruction slot"
        per_position_content = "32 bits from four instruction bytes"
        equivalent_raw_context_bytes = dataset_spec.S * 4
    elif dataset_spec.name == "embedding_seq":
        tensor_shape = f"{dataset_spec.S} x 4"
        temporal_axis = "instruction slot"
        per_position_content = "four instruction bytes"
        equivalent_raw_context_bytes = dataset_spec.S * 4
    else:
        tensor_shape = "unknown"
        temporal_axis = "unknown"
        per_position_content = "unknown"
        equivalent_raw_context_bytes = math.nan

    return {
        "tensor_shape": tensor_shape,
        "temporal_axis": temporal_axis,
        "per_position_content": per_position_content,
        "equivalent_raw_context_bytes": equivalent_raw_context_bytes,
    }


def infer_seq_len(dataset_spec, dataset_info):
    """Define seq_len preferindo a especificação do dataset."""
    from processing.dataloaders import infer_sequence_length_from_shape

    if dataset_spec.S is not None:
        return dataset_spec.S
    if dataset_spec.max_len is not None:
        return dataset_spec.max_len
    return infer_sequence_length_from_shape(dataset_info["X_shape"])



# ============================================================
# Preprocessing artifact loading and common sample preparation
# ============================================================
def _metadata_jsonl_path(out_path):
    return str(out_path).replace(".npz", ".metadata.jsonl")


def _safe_json_value(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, bytes):
        return v.hex()
    return v


def _safe_write_sample_metadata(out_path, records):
    """
    Save dataset-level metadata in exactly the same order as X/y.
    This file is consumed by FP/FN post-inference analysis.
    """
    meta_path = _metadata_jsonl_path(out_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        for rec in records:
            safe = {k: _safe_json_value(v) for k, v in rec.items()}
            f.write(json.dumps(safe, ensure_ascii=False) + "\n")
    print("[SAVED] Metadata:", meta_path)
    return meta_path


def _finalize_metadata_ids(records):
    for i, rec in enumerate(records):
        rec["sample_id"] = int(i)
        rec["dataset_index"] = int(i)
    return records


def _write_rows_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        keys = set()
        for row in rows:
            keys.update(row.keys())
        fieldnames = sorted(keys)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _safe_json_value(row.get(k, "")) for k in fieldnames})

    print(f"[SAVED] {path}")
    return path


def _artifact_filename_map(software):
    return {
        "blocksdata": f"blocksdata{software}.pickle",
        "s1": f"s1{software}.pickle",
        "s2": f"s2{software}.pickle",
        "adjlist": f"adjlist{software}.pickle",
    }


def _derive_artifact_context(mode_folder, software):
    mode_folder = Path(mode_folder).resolve()
    s1_mode = mode_folder.name
    workload = mode_folder.parent.name if mode_folder.parent.name else software
    benchmark = mode_folder.parent.parent.name if mode_folder.parent.parent.name else "Coremark-pro"
    return benchmark, workload, s1_mode


def _artifact_candidate_dirs(mode_folder, software):
    """
    Candidate artifact roots.

    Dataset files are generated under output/, but preprocessing artifacts may
    already exist under data/, as in data/Coremark-pro/<workload>/<S1_MODE>/.
    """
    mode_folder = Path(mode_folder).resolve()
    project_root = Path(__file__).resolve().parents[1]
    benchmark, workload, s1_mode = _derive_artifact_context(mode_folder, software)
    candidates = [
        mode_folder,
        project_root / "data" / str(benchmark) / str(workload) / str(s1_mode),
        project_root / "output" / str(benchmark) / str(workload) / str(s1_mode),
    ]
    unique = []
    seen = set()
    for path in candidates:
        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _source_paths(mode_folder, software):
    filenames = _artifact_filename_map(software)
    candidates = _artifact_candidate_dirs(mode_folder, software)
    resolved = {}
    for name, filename in filenames.items():
        found = None
        for folder in candidates:
            candidate = folder / filename
            if candidate.exists():
                found = candidate
                break
        resolved[name] = found if found is not None else candidates[0] / filename
    return resolved


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _source_mtimes(paths):
    return {name: os.path.getmtime(path) for name, path in paths.items()}


def _sample_hash_from_bytes(raw_bytes: bytes) -> str:
    """Stable identity of the real sample before padding/truncation."""
    return hashlib.sha256(bytes(raw_bytes)).hexdigest()


def _flatten_instruction_byte_sequences(instrs: Sequence[Sequence[int]]) -> bytes:
    """
    Convert ordered instruction byte sequences into a compact immutable bytes key.

    Unlike the previous common-stage implementation, this function does not
    normalize each instruction to 4 bytes. A compressed 2-byte instruction
    contributes exactly 2 bytes to the single-channel raw sequence.
    """
    flat = bytearray()
    for inst in instrs:
        if inst is None:
            continue
        for b in list(inst):
            flat.append(int(b) & 0xFF)
    return bytes(flat)


def _extract_chain_instrs(chain, bb2raw_instr):
    """
    Common extraction point used before all input-specific encodings.

    Input:
        chain: [BB1, BB2, BB3, ...]
        bb2raw_instr[BB] -> list of original byte sequences extracted from
                            ordered ``hex ...`` lines.

    Output:
        instrs: ordered raw instruction byte-sequence list for the whole chain
        missing_blocks: BBs not available in blocksdata/bb2raw_instr
    """
    instrs = []
    missing_blocks = []

    for bb in chain:
        bb = str(bb)
        if bb not in bb2raw_instr:
            missing_blocks.append(bb)
            continue
        instrs.extend(bb2raw_instr.get(bb, []))

    return instrs, missing_blocks


def _make_raw_records(chains, label, origin, software, bb2raw_instr):
    records = []
    for i, chain in enumerate(chains):
        chain_blocks = [str(x) for x in chain] if isinstance(chain, (list, tuple)) else []
        instrs, missing_blocks = _extract_chain_instrs(chain_blocks, bb2raw_instr)
        raw_bytes = _flatten_instruction_byte_sequences(instrs)

        compressed_instruction_count = int(sum(1 for inst in instrs if len(inst) == 2))
        normal_instruction_count = int(sum(1 for inst in instrs if len(inst) == 4))
        other_instruction_count = int(max(0, len(instrs) - compressed_instruction_count - normal_instruction_count))
        raw_zero_byte_count = int(sum(1 for b in raw_bytes if b == 0))
        pattern_0013_count = int(sum(
            1 for j in range(max(0, len(raw_bytes) - 1))
            if raw_bytes[j] == 0x00 and raw_bytes[j + 1] == 0x13
        ))

        records.append(
            {
                "source_index": int(i),
                "class_index": int(i),
                "true_label": int(label),
                "sample_origin": origin,
                "software": software,
                "chain_blocks": chain_blocks,
                "missing_blocks": missing_blocks,
                "raw_instruction_count": int(len(instrs)),
                "raw_byte_count": int(len(raw_bytes)),
                "compressed_instruction_count": compressed_instruction_count,
                "normal_instruction_count": normal_instruction_count,
                "other_instruction_count": other_instruction_count,
                "compressed_ratio": float(compressed_instruction_count / len(instrs)) if instrs else 0.0,
                "raw_zero_byte_count": raw_zero_byte_count,
                "raw_zero_byte_ratio": float(raw_zero_byte_count / len(raw_bytes)) if raw_bytes else 0.0,
                "pattern_0013_count": pattern_0013_count,
                "pattern_0013_ratio": float(pattern_0013_count / max(1, len(raw_bytes) - 1)) if len(raw_bytes) > 1 else 0.0,
                "raw_bytes": raw_bytes,
                "raw_key": raw_bytes,
                "sample_hash": _sample_hash_from_bytes(raw_bytes),
            }
        )
    return records


def _hex_preview(raw_bytes: bytes, max_bytes=32):
    toks = [f"{b:02x}" for b in raw_bytes[:max_bytes]]
    suffix = " ..." if len(raw_bytes) > max_bytes else ""
    return " ".join(toks) + suffix


def _remove_cross_class_overlaps(records_s1, records_s2, out_csv_path):
    """
    Remove equal raw samples appearing in both classes.
    All overlapped representatives are removed from both S1 and S2.
    """
    s1_by_key = defaultdict(list)
    s2_by_key = defaultdict(list)

    for rec in records_s1:
        s1_by_key[rec["raw_key"]].append(rec)
    for rec in records_s2:
        s2_by_key[rec["raw_key"]].append(rec)

    overlap_keys = sorted(set(s1_by_key).intersection(s2_by_key), key=lambda k: (len(k), k))
    rows = []

    for dup_id, key in enumerate(overlap_keys):
        s1_recs = s1_by_key[key]
        s2_recs = s2_by_key[key]
        rows.append(
            {
                "overlap_id": dup_id,
                "raw_instruction_count": s1_recs[0].get("raw_instruction_count", None),
                "raw_byte_count": len(key),
                "s1_count": len(s1_recs),
                "s2_count": len(s2_recs),
                "s1_source_indices": json.dumps([r["source_index"] for r in s1_recs]),
                "s2_source_indices": json.dumps([r["source_index"] for r in s2_recs]),
                "s1_origin_addresses": json.dumps([r["chain_blocks"] for r in s1_recs]),
                "s2_origin_addresses": json.dumps([r["chain_blocks"] for r in s2_recs]),
                "sample_hex_preview": _hex_preview(key),
            }
        )

    _write_rows_csv(
        out_csv_path,
        rows,
        fieldnames=[
            "overlap_id",
            "raw_instruction_count",
            "raw_byte_count",
            "s1_count",
            "s2_count",
            "s1_source_indices",
            "s2_source_indices",
            "s1_origin_addresses",
            "s2_origin_addresses",
            "sample_hex_preview",
        ],
    )

    overlap_set = set(overlap_keys)
    records_s1_no_overlap = [r for r in records_s1 if r["raw_key"] not in overlap_set]
    records_s2_no_overlap = [r for r in records_s2 if r["raw_key"] not in overlap_set]

    return records_s1_no_overlap, records_s2_no_overlap, rows


def _summarize_common_stage(records_s1_raw, records_s2_raw, records_s1_final, records_s2_final,
                            overlap_rows, out_csv_path, software):
    """
    Summarize the common processing stage used by this version.

    Only S1/S2 cross-class overlaps are removed. Duplicates inside the same
    class are intentionally preserved, and no class balancing/downsampling is
    applied here.
    """
    removed_s1_overlap = sum(int(r.get("s1_count", 0)) for r in overlap_rows)
    removed_s2_overlap = sum(int(r.get("s2_count", 0)) for r in overlap_rows)

    rows = [
        {
            "software": software,
            "stage": "raw_loaded",
            "s1_samples": len(records_s1_raw),
            "s2_samples": len(records_s2_raw),
            "removed_s1": 0,
            "removed_s2": 0,
            "note": "After extracting ordered hex instructions from blocksdata into raw bytes without padding.",
        },
        {
            "software": software,
            "stage": "cross_class_overlap_removed",
            "s1_samples": len(records_s1_final),
            "s2_samples": len(records_s2_final),
            "removed_s1": removed_s1_overlap,
            "removed_s2": removed_s2_overlap,
            "note": "Equal raw-byte samples found in both S1 and S2 were saved to CSV and removed from both classes before padding/truncation.",
        },
        {
            "software": software,
            "stage": "within_class_duplicates_preserved",
            "s1_samples": len(records_s1_final),
            "s2_samples": len(records_s2_final),
            "removed_s1": 0,
            "removed_s2": 0,
            "note": "Duplicated samples inside S1 and inside S2 were intentionally preserved.",
        },
        {
            "software": software,
            "stage": "final_no_balancing",
            "s1_samples": len(records_s1_final),
            "s2_samples": len(records_s2_final),
            "removed_s1": 0,
            "removed_s2": 0,
            "note": "No class balancing/downsampling was applied in the common stage.",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out_csv_path, index=False)
    print(f"[SAVED] {out_csv_path}")
    return df


def _records_to_metadata(records, dataset_name):
    out = []
    for rec in records:
        out.append(
            {
                "sample_id": None,
                "dataset_index": None,
                "source_index": int(rec["source_index"]),
                "class_index": int(rec["class_index"]),
                "true_label": int(rec["true_label"]),
                "sample_origin": rec["sample_origin"],
                "software": rec["software"],
                "dataset_name": dataset_name,
                "chain_blocks": list(rec["chain_blocks"]),
                "missing_blocks": list(rec["missing_blocks"]),
                "raw_instruction_count": int(rec["raw_instruction_count"]),
                "raw_byte_count": int(rec["raw_byte_count"]),
                "compressed_instruction_count": int(rec.get("compressed_instruction_count", 0)),
                "normal_instruction_count": int(rec.get("normal_instruction_count", 0)),
                "other_instruction_count": int(rec.get("other_instruction_count", 0)),
                "compressed_ratio": float(rec.get("compressed_ratio", 0.0)),
                "raw_zero_byte_count": int(rec.get("raw_zero_byte_count", 0)),
                "raw_zero_byte_ratio": float(rec.get("raw_zero_byte_ratio", 0.0)),
                "pattern_0013_count": int(rec.get("pattern_0013_count", 0)),
                "pattern_0013_ratio": float(rec.get("pattern_0013_ratio", 0.0)),
                "sample_hash": rec.get("sample_hash", _sample_hash_from_bytes(rec.get("raw_key", b""))),
            }
        )
    return out


def _common_cache_path(mode_folder, software):
    # Versioned cache name avoids reusing older caches that removed within-class
    # duplicates or performed common-stage balancing.
    return Path(mode_folder) / f"common_overlap_filtered_samples_v2_{software}.pickle"


def _cache_is_fresh(cache_path, paths):
    if not Path(cache_path).exists():
        return False
    try:
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        return payload.get("source_mtimes") == _source_mtimes(paths)
    except Exception:
        return False


def _prepare_common_clean_samples(mode_folder, software1, force_rebuild_common=False):
    """
    The single common processing stage for all input representations.

    This version uses the original S1/S2 chains directly, with only one
    removal rule: raw-byte overlaps between S1 and S2 are removed from both
    classes before padding/truncation. Duplicates inside the same class are
    preserved, and no common-stage balancing is applied.

    Returns:
        payload with:
            blocks_data, adjlist, bb2instr, bb2indirect, SUCC_MAP,
            records_s1, records_s2
    """
    mode_folder = Path(mode_folder)
    mode_folder.mkdir(parents=True, exist_ok=True)
    paths = _source_paths(mode_folder, software1)

    missing_paths = {name: path for name, path in paths.items() if not path.exists()}
    if missing_paths:
        candidate_text = "\n".join(str(p) for p in _artifact_candidate_dirs(mode_folder, software1))
        missing_text = "\n".join(f"{name}: {path}" for name, path in missing_paths.items())
        raise FileNotFoundError(
            f"Missing required preprocessing artifact(s) for '{software1}'.\n"
            f"Missing files:\n{missing_text}\n"
            f"Searched folders:\n{candidate_text}"
        )

    cache_path = _common_cache_path(mode_folder, software1)
    if not force_rebuild_common and _cache_is_fresh(cache_path, paths):
        print(f"[CACHE] Common overlap-filtered samples loaded: {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("[BUILD] Common overlap-filtered S1/S2 samples")
    blocks_data = _load_pickle(paths["blocksdata"])
    s1 = _load_pickle(paths["s1"])
    s2 = _load_pickle(paths["s2"])
    adjlist = _load_pickle(paths["adjlist"])

    print(
        "Loaded:",
        f"{len(blocks_data)} blocks | {len(s1)} S1 chains | {len(s2)} S2 chains | adjlist nodes={len(adjlist)}",
    )

    # ------------------------------------------------------------
    # Common processing up to the user's requested split point:
    # blocksdata/S1/S2/adjlist -> load -> chain -> ordered hex instruction extraction
    # ------------------------------------------------------------
    # Common stage: preserve original byte length per instruction.
    # This prevents compressed RISC-V instructions from being normalized
    # before the split between single_channel and four_channel.
    bb2raw_instr = {bb: extract_bb_raw_instructions(bb, blocks_data) for bb in blocks_data.keys()}

    # Fixed-slot extraction is representation-specific. It is used only by
    # four_channel and by bit32_channel, which is derived from four_channel.
    bb2instr = {bb: extract_bb_instructions(bb, blocks_data) for bb in blocks_data.keys()}
    bb2indirect = {bb: bb_has_indirect_jump(bb, blocks_data) for bb in blocks_data.keys()}
    SUCC_MAP = build_successor_map(adjlist)

    records_s1_raw = _make_raw_records(s1, label=0, origin="S1", software=software1, bb2raw_instr=bb2raw_instr)
    records_s2_raw = _make_raw_records(s2, label=1, origin="S2", software=software1, bb2raw_instr=bb2raw_instr)

    # Cross-class equal raw samples: save origins, then remove from both classes.
    # This is the only common-stage removal. Within-class duplicates are preserved,
    # and no class balancing/downsampling is applied here.
    records_s1_final, records_s2_final, overlap_rows = _remove_cross_class_overlaps(
        records_s1_raw,
        records_s2_raw,
        mode_folder / f"common_overlap_s1_s2_{software1}.csv",
    )

    if not records_s1_final or not records_s2_final:
        raise RuntimeError(
            "Common overlap filtering produced an empty class. Check S1/S2 overlap and blocksdata coverage."
        )

    _summarize_common_stage(
        records_s1_raw,
        records_s2_raw,
        records_s1_final,
        records_s2_final,
        overlap_rows,
        mode_folder / f"common_cleaning_summary_{software1}.csv",
        software1,
    )

    payload = {
        "source_mtimes": _source_mtimes(paths),
        "software": software1,
        "blocks_data": blocks_data,
        "adjlist": adjlist,
        "bb2raw_instr": bb2raw_instr,
        "bb2instr": bb2instr,
        "bb2indirect": bb2indirect,
        "SUCC_MAP": SUCC_MAP,
        "processing_policy": "original_s1_s2_minus_cross_class_raw_byte_overlap_only; within_class_duplicates_preserved; no_common_stage_balancing",
        "records_s1": records_s1_final,
        "records_s2": records_s2_final,
    }

    with open(cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[SAVED] Common overlap-filtered samples cache: {cache_path}")

    return payload

