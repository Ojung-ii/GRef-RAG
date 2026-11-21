"""Build fixed-budget contexts from ranked evidence and rendered inputs."""

from __future__ import annotations

import hashlib

from ..rendering import render_evidence

BUDGETS = (512, 1024, 2048)
METHODS = ("gref-rag", "kg2rag")
POLICIES = ("packing", "truncation")


def pack_context(units, separator, token_count, cap):
    """Keep complete units in rank order, skipping units that exceed the cap."""
    selected = []
    context = ""
    for unit_id, text in units:
        if not text:
            continue
        trial = separator.join((context, text)) if context else text
        if token_count(trial) <= cap:
            selected.append(unit_id)
            context = trial
    return context, selected


def evidence_units(row, method):
    if method == "gref-rag":
        return [
            (bundle["bundle_id"], render_evidence([bundle], 1, 16000, "common"))
            for bundle in row["evidence_bundles"]
        ], "\n\n"
    return [
        (chunk["chunk_id"], f"{chunk.get('source_id') or chunk['title']}: {chunk['text']}")
        for chunk in row["final_ordered_chunks"]
    ], "\n"


def budget_contexts(row, method, tokenizer):
    """Return all caps for the two matched-budget controls for one question."""
    units, separator = evidence_units(row, method)
    rendered = row.get("context", row.get("rendered_context"))
    if not isinstance(rendered, str):
        raise ValueError("Matched-budget controls require the original rendered context")

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    tokens = encode(rendered)
    for cap in BUDGETS:
        packed, ids = pack_context(units, separator, lambda text: len(encode(text)), cap)
        prefix = tokenizer.decode(tokens[:cap], skip_special_tokens=True)
        for policy, context in (("packing", packed), ("truncation", prefix)):
            count = len(encode(context))
            if count > cap:
                raise ValueError(f"{policy} context exceeds its {cap}-token cap")
            yield {
                "policy": policy,
                "budget": cap,
                "context": context,
                "context_tokens": count,
                "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
                **({"selected_evidence_ids": ids} if policy == "packing" else {}),
            }
