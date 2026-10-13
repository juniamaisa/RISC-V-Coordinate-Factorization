"""Canonical context-window construction used by all NeurReps inputs.

This module contains only the window-selection policy needed by the experiment.
SC, 4C, and 32C are derived from the same returned ``S x 4`` byte matrix.
"""
from __future__ import annotations

from processing.dataset_registry import (
    PAD_INST,
    TOKEN_CUT,
    TOKEN_END,
    count_chain_instructions,
    count_invalid_transitions,
)


def take_head(instrs, n):
    return instrs[: max(n, 0)] if n > 0 else []


def take_tail(instrs, n):
    if n <= 0:
        return []
    return instrs[-n:] if n <= len(instrs) else instrs[:]


def finalize_no_end(content, S):
    """No cut: pad at the end without an END token."""
    if len(content) >= S:
        return content[:S]
    return content + [PAD_INST] * (S - len(content))


def finalize_with_end(content, S):
    """Cut: reserve the final slot for TOKEN_END and pad before it."""
    budget = S - 1
    if budget < 0:
        return []
    if len(content) > budget:
        content = content[:budget]
    if len(content) < budget:
        content = content + [PAD_INST] * (budget - len(content))
    return content + [TOKEN_END]


def filter_chain_valid(chain_raw, bb2instr):
    valid, missing = [], []
    for bb in chain_raw:
        bb = str(bb)
        if bb in bb2instr:
            valid.append(bb)
        else:
            missing.append(bb)
    return valid, missing


def select_triple_focus_indirect(chain_valid, bb2indirect):
    n = len(chain_valid)
    if n < 3:
        return chain_valid
    indirect_idx = [i for i, bb in enumerate(chain_valid) if bb2indirect.get(bb, False)]
    for i in indirect_idx:
        if 1 <= i <= n - 2:
            return [chain_valid[i - 1], chain_valid[i], chain_valid[i + 1]]
    if indirect_idx:
        i = indirect_idx[0]
        if i == 0:
            return chain_valid[:3]
        if i == n - 1:
            return chain_valid[-3:]
    return chain_valid[:3]


RULE_EXPECTED_BLOCKS = {
    "EMPTY_PAD": 0,
    "ONE_BB_NO_CUT": 1,
    "T2_NO_CUT_FITS": 2,
    "T2_CASE5_SS_NO_CUT": 2,
    "T2_CASE6_LL_CUT": 2,
    "T2_CASE7_LS_CUT": 2,
    "T2_CASE8_SL_CUT": 2,
    "T3_NO_CUT_FITS": 3,
    "T3_CASE4_SSS_NO_CUT": 3,
    "T3_CASE2_LLS_CUT": 3,
    "T3_CASE3_LSS_CUT": 3,
    "T3_CASE1_LLL_CUT": 3,
    "T3_MIXED_KEEP_SMALL_CUT_LARGE": 3,
    "T3_FALLBACK_CUT": 3,
}


def repr_bb2_head_cut_tail_with_counts(bb2_instrs, total_budget):
    if total_budget <= 0:
        return [], 0, 0
    if total_budget == 1:
        return [TOKEN_CUT], 0, 1
    content_budget = total_budget - 1
    head = content_budget // 2
    tail = content_budget - head
    seq = take_head(bb2_instrs, head) + [TOKEN_CUT] + take_tail(bb2_instrs, tail)
    return seq, head + tail, 1


def finalize_meta_counts(
    meta,
    selected_instr_count,
    kept_original_instr_count,
    cut_token_count,
    end_token_count,
    S,
):
    meta["selected_instruction_count_before_trunc"] = int(selected_instr_count)
    meta["kept_original_instruction_count"] = int(kept_original_instr_count)
    meta["cut_token_count"] = int(cut_token_count)
    meta["end_token_count"] = int(end_token_count)
    meta["final_length"] = int(S)
    meta["trunc_removed"] = int(max(selected_instr_count - kept_original_instr_count, 0))
    meta["pad_added"] = int(
        max(S - kept_original_instr_count - cut_token_count - end_token_count, 0)
    )
    meta["padding_rate"] = float(meta["pad_added"] / S) if S > 0 else 0.0
    meta["preserved_ratio"] = (
        float(kept_original_instr_count / selected_instr_count)
        if selected_instr_count > 0
        else 1.0
    )
    return meta


def build_sample_from_chain(chain_raw, S, bb2instr, bb2indirect, successor_map):
    """Return one canonical length-S sequence of four-byte instruction slots."""
    meta = {
        "orig_chain_len": len(chain_raw),
        "missing_blocks": [],
        "valid_chain_len": 0,
        "used_blocks": [],
        "rule": None,
        "cut_applied": False,
        "expected_blocks": None,
        "n1": None,
        "n2": None,
        "n3": None,
        "T2": None,
        "T3": None,
        "raw_instruction_count": 0,
        "raw_invalid_transition_count": 0,
        "raw_invalid_pairs": [],
        "selected_instruction_count_before_trunc": 0,
        "kept_original_instruction_count": 0,
        "cut_token_count": 0,
        "end_token_count": 0,
        "final_length": 0,
        "trunc_removed": 0,
        "pad_added": 0,
        "padding_rate": 0.0,
        "preserved_ratio": 1.0,
        "selected_invalid_transition_count": 0,
        "selected_invalid_pairs": [],
    }
    chain_valid, missing = filter_chain_valid(chain_raw, bb2instr)
    meta["missing_blocks"] = missing
    meta["valid_chain_len"] = len(chain_valid)
    raw_count, _ = count_chain_instructions(chain_valid, bb2instr)
    invalid_count, invalid_pairs = count_invalid_transitions(chain_valid, successor_map)
    meta["raw_instruction_count"] = raw_count
    meta["raw_invalid_transition_count"] = invalid_count
    meta["raw_invalid_pairs"] = invalid_pairs

    if not chain_valid:
        meta["rule"] = "EMPTY_PAD"
        meta["expected_blocks"] = 0
        return [PAD_INST] * S, finalize_meta_counts(meta, 0, 0, 0, 0, S)

    if len(chain_valid) == 1:
        bb1 = chain_valid[0]
        B1 = bb2instr[bb1]
        meta.update(rule="ONE_BB_NO_CUT", expected_blocks=1, used_blocks=[bb1], n1=len(B1))
        meta["selected_invalid_transition_count"], meta["selected_invalid_pairs"] = (
            count_invalid_transitions(meta["used_blocks"], successor_map)
        )
        kept = min(len(B1), S)
        return finalize_no_end(take_head(B1, S), S), finalize_meta_counts(
            meta, len(B1), kept, 0, 0, S
        )

    if len(chain_valid) == 2:
        bb1, bb2 = chain_valid[:2]
        B1, B2 = bb2instr[bb1], bb2instr[bb2]
        ind2 = bb2indirect.get(bb2, False)
        n1, n2 = len(B1), len(B2)
        selected_count = n1 + n2
        meta.update(n1=n1, n2=n2, used_blocks=[bb1, bb2])
        meta["selected_invalid_transition_count"], meta["selected_invalid_pairs"] = (
            count_invalid_transitions(meta["used_blocks"], successor_map)
        )
        if selected_count <= S:
            meta.update(rule="T2_NO_CUT_FITS", expected_blocks=2)
            return finalize_no_end(B1 + B2, S), finalize_meta_counts(
                meta, selected_count, selected_count, 0, 0, S
            )
        T2 = S // 2
        meta["T2"] = T2
        small1, small2 = n1 <= T2, n2 <= T2
        if small1 and small2:
            meta.update(rule="T2_CASE5_SS_NO_CUT", expected_blocks=2)
            kept = min(selected_count, S)
            return finalize_no_end(B1 + B2, S), finalize_meta_counts(
                meta, selected_count, kept, 0, 0, S
            )
        if not small1 and not small2:
            meta.update(rule="T2_CASE6_LL_CUT", cut_applied=True, expected_blocks=2)
            half = S // 2
            part1 = take_tail(B1, half)
            budget2 = (S - 1) - len(part1)
            part2 = take_tail(B2, budget2) if ind2 else take_head(B2, budget2)
            kept = len(part1) + len(part2)
            return finalize_with_end(part1 + part2, S), finalize_meta_counts(
                meta, selected_count, kept, 0, 1, S
            )
        if not small1 and small2:
            meta.update(rule="T2_CASE7_LS_CUT", cut_applied=True, expected_blocks=2)
            part1 = take_tail(B1, max((S - 1) - len(B2), 0))
            kept = len(part1) + len(B2)
            return finalize_with_end(part1 + B2, S), finalize_meta_counts(
                meta, selected_count, kept, 0, 1, S
            )
        if small1 and not small2:
            meta.update(rule="T2_CASE8_SL_CUT", cut_applied=True, expected_blocks=2)
            budget2 = max((S - 1) - len(B1), 0)
            part2 = take_tail(B2, budget2) if ind2 else take_head(B2, budget2)
            kept = len(B1) + len(part2)
            return finalize_with_end(B1 + part2, S), finalize_meta_counts(
                meta, selected_count, kept, 0, 1, S
            )
        meta.update(rule="T2_CASE6_LL_CUT", cut_applied=True, expected_blocks=2)
        return finalize_with_end(B1 + B2, S), finalize_meta_counts(
            meta, selected_count, min(selected_count, S - 1), 0, 1, S
        )

    trio = select_triple_focus_indirect(chain_valid, bb2indirect)
    bb1, bb2, bb3 = trio[:3]
    B1, B2, B3 = bb2instr[bb1], bb2instr[bb2], bb2instr[bb3]
    ind3 = bb2indirect.get(bb3, False)
    n1, n2, n3 = len(B1), len(B2), len(B3)
    selected_count = n1 + n2 + n3
    meta.update(n1=n1, n2=n2, n3=n3, used_blocks=[bb1, bb2, bb3])
    meta["selected_invalid_transition_count"], meta["selected_invalid_pairs"] = (
        count_invalid_transitions(meta["used_blocks"], successor_map)
    )
    if selected_count <= S:
        meta.update(rule="T3_NO_CUT_FITS", expected_blocks=3)
        return finalize_no_end(B1 + B2 + B3, S), finalize_meta_counts(
            meta, selected_count, selected_count, 0, 0, S
        )

    T3 = S // 3
    meta["T3"] = T3
    small1, small2, small3 = n1 <= T3, n2 <= T3, n3 <= T3
    if small1 and small2 and small3:
        meta.update(rule="T3_CASE4_SSS_NO_CUT", expected_blocks=3)
        kept = min(selected_count, S)
        return finalize_no_end(B1 + B2 + B3, S), finalize_meta_counts(
            meta, selected_count, kept, 0, 0, S
        )
    if not small1 and not small2 and small3:
        meta.update(rule="T3_CASE2_LLS_CUT", cut_applied=True, expected_blocks=3)
        fixed3 = B3[:]
        remaining = (S - 1) - len(fixed3)
        if remaining <= 0:
            return finalize_with_end(fixed3, S), finalize_meta_counts(
                meta, selected_count, len(fixed3), 0, 1, S
            )
        k = remaining // 2
        part1 = take_tail(B1, k)
        part2, kept2, cut2 = repr_bb2_head_cut_tail_with_counts(B2, remaining - k)
        kept = len(part1) + kept2 + len(fixed3)
        return finalize_with_end(part1 + part2 + fixed3, S), finalize_meta_counts(
            meta, selected_count, kept, cut2, 1, S
        )
    if not small1 and small2 and small3:
        meta.update(rule="T3_CASE3_LSS_CUT", cut_applied=True, expected_blocks=3)
        fixed23 = B2 + B3
        rem = (S - 1) - len(fixed23)
        if rem <= 0:
            return finalize_with_end(fixed23, S), finalize_meta_counts(
                meta, selected_count, len(fixed23), 0, 1, S
            )
        part1 = take_tail(B1, rem)
        kept = len(part1) + len(fixed23)
        return finalize_with_end(part1 + fixed23, S), finalize_meta_counts(
            meta, selected_count, kept, 0, 1, S
        )
    if not small1 and not small2 and not small3:
        meta.update(rule="T3_CASE1_LLL_CUT", cut_applied=True, expected_blocks=3)
        bb1_tail = T3
        bb3_budget = T3
        bb2_budget = (S - 1) - bb1_tail - bb3_budget
        part1 = take_tail(B1, bb1_tail)
        part2, kept2, cut2 = repr_bb2_head_cut_tail_with_counts(B2, bb2_budget)
        part3 = take_tail(B3, bb3_budget) if ind3 else take_head(B3, bb3_budget)
        kept = len(part1) + kept2 + len(part3)
        return finalize_with_end(part1 + part2 + part3, S), finalize_meta_counts(
            meta, selected_count, kept, cut2, 1, S
        )

    meta.update(rule="T3_MIXED_KEEP_SMALL_CUT_LARGE", cut_applied=True, expected_blocks=3)
    budget = S - 1
    parts = [B1 if small1 else [], B2 if small2 else [], B3 if small3 else []]
    kept = sum(len(part) for part in parts)
    cut_tokens = 0
    rem = budget - kept
    if rem <= 0:
        return finalize_with_end(parts[0] + parts[1] + parts[2], S), finalize_meta_counts(
            meta, selected_count, kept, cut_tokens, 1, S
        )
    large_pos = [i for i, small in enumerate([small1, small2, small3]) if not small]
    if not large_pos:
        return finalize_with_end(B1 + B2 + B3, S), finalize_meta_counts(
            meta, selected_count, min(selected_count, S - 1), 0, 1, S
        )
    base, extra = divmod(rem, len(large_pos))
    alloc = {pos: base + (1 if j < extra else 0) for j, pos in enumerate(large_pos)}
    if 0 in alloc:
        parts[0] = take_tail(B1, alloc[0])
        kept += len(parts[0])
    if 1 in alloc:
        parts[1], kept2, cut2 = repr_bb2_head_cut_tail_with_counts(B2, alloc[1])
        kept += kept2
        cut_tokens += cut2
    if 2 in alloc:
        parts[2] = take_tail(B3, alloc[2]) if ind3 else take_head(B3, alloc[2])
        kept += len(parts[2])
    return finalize_with_end(parts[0] + parts[1] + parts[2], S), finalize_meta_counts(
        meta, selected_count, kept, cut_tokens, 1, S
    )
