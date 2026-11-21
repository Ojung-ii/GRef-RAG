"""Render selected evidence bundles without reference to gold evidence."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .text import normalize_title_for_recall, safe_truncate, sentence_split


def _ordered_bundles(bundles: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    order_rank = {
        "anchor_connected_chain_complete": 0,
        "multi_anchor": 1,
        "anchor": 2,
        "chain_complete_v2": 3,
        "bridge_connected": 4,
        "same_title": 5,
        "generic_relation": 6,
        "fallback": 7,
        "complete_bridge_chain": 3,
        "exact_query_anchor": 2,
        "bridge_candidate": 4,
    }
    return [
        item[1]
        for item in sorted(
            list(enumerate(bundles)),
            key=lambda item: (
                order_rank.get(str(item[1].get("ordering_group", "same_title")), 8),
                not bool(item[1].get("chain_complete_v2", item[1].get("chain_complete"))),
                not bool(item[1].get("anchor_connected")),
                bool(item[1].get("is_generic_relation_title")),
                not bool(item[1].get("exact_anchor_match")),
                -float(item[1].get("residual_coverage_count", 0.0) or 0.0),
                -float(item[1].get("score", 0.0) or 0.0),
                item[0],
            ),
        )
    ]


def _bridge_paths(bundle: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        x
        for x in bundle.get("evidence_path", []) or []
        if isinstance(x, Mapping) and x.get("path_type") == "mention_bridge"
    ]


def _sentence_norm(text: Any) -> str:
    value = str(text or "").lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())


def _record(
    title: Any, text: Any, role: str, priority: int, source_id: Any = None
) -> dict[str, Any] | None:
    sentence = str(text or "").strip()
    if not sentence:
        return None
    return {
        "title": str(title or "").strip(),
        "text": sentence,
        "role": role,
        "priority": int(priority),
        "source_id": str(source_id or "").strip(),
        "norm": _sentence_norm(sentence),
    }


def _chain_records(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = []
    for path in _bridge_paths(bundle):
        seed = _record(
            path.get("source_title") or bundle.get("anchor_title"),
            path.get("seed_prop"),
            "chain_seed",
            0,
            path.get("seed_prop_id"),
        )
        bridge = _record(
            path.get("bridge_title"),
            path.get("bridge_prop"),
            "chain_bridge",
            1,
            path.get("bridge_prop_id"),
        )
        if seed:
            records.append(seed)
        if bridge:
            records.append(bridge)
    return records


def _prop_records(
    bundle: Mapping[str, Any], chain_norms: set[str] | None = None
) -> list[dict[str, Any]]:
    chain_norms = chain_norms or set()
    records = []
    for p in bundle.get("propositions", []) or []:
        rec = _record(
            p.get("title") or bundle.get("anchor_title"),
            p.get("text"),
            "support",
            3,
            p.get("prop_id"),
        )
        if not rec or rec["norm"] in chain_norms:
            continue
        for score_key in (
            "residual_score",
            "original_relevance_score",
            "score",
            "relevance_score",
        ):
            if p.get(score_key) is not None:
                try:
                    rec["score"] = float(p.get(score_key) or 0.0)
                except Exception:
                    rec["score"] = 0.0
                break
        if (
            bundle.get("answer_slot_aligned")
            or int(bundle.get("residual_coverage_count", 0) or 0) > 0
        ):
            rec["role"] = "answer_slot_support"
            rec["priority"] = 2
        elif bundle.get("anchor_connected"):
            rec["role"] = "anchor_support"
            rec["priority"] = 3
        records.append(rec)
    return records


def _source_records(
    bundle: Mapping[str, Any],
    chain_norms: set[str] | None = None,
    limit_per_chunk: int = 2,
) -> list[dict[str, Any]]:
    chain_norms = chain_norms or set()
    records = []
    for c in bundle.get("source_chunks", []) or []:
        for sent in sentence_split(str(c.get("text", "")))[:limit_per_chunk]:
            rec = _record(
                c.get("title") or bundle.get("anchor_title"),
                sent,
                "source",
                5,
                c.get("chunk_id"),
            )
            if rec and rec["norm"] not in chain_norms:
                records.append(rec)
    return records


def _append_unique_source_title(titles: list[str], seen: set[str], title: Any) -> None:
    clean = str(title or "").strip()
    norm = normalize_title_for_recall(clean)
    if not norm or norm in seen:
        return
    seen.add(norm)
    titles.append(clean)


def _bundle_source_titles_in_render_order(bundle: Mapping[str, Any]) -> list[str]:
    titles: list[str] = []
    seen: set[str] = set()
    if bundle.get("bundle_type") == "multi_anchor":
        for rec in _prop_records(bundle):
            _append_unique_source_title(titles, seen, rec.get("title"))
        for rec in _source_records(bundle, limit_per_chunk=1):
            _append_unique_source_title(titles, seen, rec.get("title"))
    else:
        for rec in _chain_records(bundle):
            _append_unique_source_title(titles, seen, rec.get("title"))
        chain_norms = {r["norm"] for r in _chain_records(bundle) if r.get("norm")}
        for rec in _prop_records(bundle, chain_norms):
            _append_unique_source_title(titles, seen, rec.get("title"))
        for rec in _source_records(
            bundle,
            {
                r["norm"]
                for r in _chain_records(bundle) + _prop_records(bundle, chain_norms)
                if r.get("norm")
            },
            limit_per_chunk=1,
        ):
            _append_unique_source_title(titles, seen, rec.get("title"))
    if not titles:
        _append_unique_source_title(titles, seen, bundle.get("anchor_title") or bundle.get("title"))
    return titles


def rendered_source_titles_for_bundles(
    evidence_bundles: Sequence[Mapping[str, Any]], max_bundles: int | None = None
) -> list[str]:
    ordered = _ordered_bundles(evidence_bundles or [])
    if max_bundles is not None:
        ordered = ordered[: max(0, int(max_bundles))]
    titles: list[str] = []
    seen: set[str] = set()
    for bundle in ordered:
        for title in _bundle_source_titles_in_render_order(bundle):
            _append_unique_source_title(titles, seen, title)
    return titles


def _append_record(
    lines: list[str], seen: set[str], rec: Mapping[str, Any], prefix: str = "- "
) -> bool:
    norm = str(rec.get("norm") or _sentence_norm(rec.get("text")))
    if not norm or norm in seen:
        return False
    seen.add(norm)
    title = str(rec.get("title") or "").strip()
    text = str(rec.get("text") or "").strip()
    if title:
        lines.append(f"{prefix}{title}: {text}")
    else:
        lines.append(f"{prefix}{text}")
    return True


def _source_refs(bundle: Mapping[str, Any]) -> str:
    refs = []
    for c in bundle.get("source_chunks", []) or []:
        title = str(c.get("title") or "").strip()
        chunk_id = str(c.get("chunk_id") or "").strip()
        if title or chunk_id:
            refs.append(f"{title} | {chunk_id}".strip(" |"))
    return "; ".join(dict.fromkeys(refs))


def _render_chain_dedup_ordered(bundles: Sequence[Mapping[str, Any]], max_chars: int) -> str:
    parts = []
    seen: set[str] = set()
    for i, bundle in enumerate(bundles, start=1):
        if bundle.get("bundle_type") == "multi_anchor":
            anchors = "; ".join(str(x) for x in bundle.get("anchor_titles", []) or [])
            parts.append(
                f"[Multi-Anchor Evidence {i} | anchors={anchors} | complete={bool(bundle.get('multi_anchor_complete'))}]"
            )
            appended = False
            for rec in _prop_records(bundle):
                appended = _append_record(parts, seen, rec) or appended
            if not appended:
                for rec in _source_records(bundle, limit_per_chunk=1):
                    _append_record(parts, seen, rec)
            parts.append("")
            continue
        bridge = ", ".join(str(x) for x in bundle.get("bridge_titles", []) or [])
        if _bridge_paths(bundle):
            parts.append(
                f"[Evidence Chain {i} | anchor_connected={bool(bundle.get('anchor_connected'))} | chain_complete_v2={bool(bundle.get('chain_complete_v2', bundle.get('chain_complete')))}]"
            )
            parts.append(f"Anchor: {bundle.get('anchor_title')}")
            if bridge:
                parts.append(f"Bridge: {bridge}")
            parts.append("Chain:")
            for rec in _chain_records(bundle):
                _append_record(parts, seen, rec)
            support_records = _prop_records(
                bundle, {r.get("norm", "") for r in _chain_records(bundle)}
            )
            if support_records:
                parts.append("Supporting Propositions:")
                for rec in support_records:
                    _append_record(parts, seen, rec)
        else:
            parts.append(
                f"[Supporting Evidence {i} | anchor={bundle.get('anchor_title')} | relation_title={bool(bundle.get('is_relation_title_bundle'))}]"
            )
            appended = False
            for rec in _prop_records(bundle):
                appended = _append_record(parts, seen, rec) or appended
            if not appended:
                for rec in _source_records(bundle, limit_per_chunk=1):
                    _append_record(parts, seen, rec)
        refs = _source_refs(bundle)
        if refs:
            parts.append(f"Sources: {refs}")
        parts.append("")
    return safe_truncate("\n".join(parts).strip(), max_chars)


def _bundle_unique_key(bundle: Mapping[str, Any], fallback_index: int) -> str:
    for key in (
        "bundle_id",
        "source_candidate_id",
        "parent_title",
        "anchor_title",
        "title",
    ):
        value = bundle.get(key)
        if value is not None and str(value).strip():
            return f"{key}:{value}"
    return f"index:{fallback_index}"


def topn_chain_dedup_bundles(
    bundles: Sequence[Mapping[str, Any]], max_blocks: int
) -> list[Mapping[str, Any]]:
    limit = max(0, int(max_blocks or 0))
    source = list(bundles)
    if not source or limit <= 0:
        return []
    if limit <= 3:
        return list(_ordered_bundles(source[:limit])[:limit])
    selected = list(_ordered_bundles(source[:3])[:3])
    seen = {_bundle_unique_key(bundle, i) for i, bundle in enumerate(selected)}
    for k in range(4, min(limit, len(source)) + 1):
        for bundle in _ordered_bundles(source[:k]):
            key = _bundle_unique_key(bundle, source.index(bundle) if bundle in source else k)
            if key in seen:
                continue
            selected.append(bundle)
            seen.add(key)
            break
    return selected[: min(limit, len(source))]


def render_topn_chain_dedup_context(
    bundles: Sequence[Mapping[str, Any]],
    max_chars: int,
    max_blocks: int,
) -> str:
    return _render_chain_dedup_ordered(topn_chain_dedup_bundles(bundles, max_blocks), max_chars)


def _clean_records_for_bundle(
    bundle: Mapping[str, Any], include_source_fallback: bool = True
) -> list[dict[str, Any]]:
    chain = _chain_records(bundle)
    chain_norms = {r.get("norm", "") for r in chain if r.get("norm")}
    records = chain + _prop_records(bundle, chain_norms)
    if include_source_fallback and not records:
        records.extend(_source_records(bundle, chain_norms, limit_per_chunk=1))
    return records


def _append_plain_sentence(lines: list[str], seen: set[str], rec: Mapping[str, Any]) -> bool:
    norm = str(rec.get("norm") or _sentence_norm(rec.get("text")))
    text = str(rec.get("text") or "").strip()
    if not norm or norm in seen or not text:
        return False
    seen.add(norm)
    lines.append(text)
    return True


def _render_multi_anchor_hybrid(
    parts: list[str], seen: set[str], bundle: Mapping[str, Any], idx: int
) -> bool:
    anchors = [str(x) for x in bundle.get("anchor_titles", []) or [] if str(x).strip()]
    if not anchors and bundle.get("anchor_title"):
        anchors = [str(bundle.get("anchor_title"))]
    records = _clean_records_for_bundle(bundle)
    if not records:
        return False
    parts.append(f"Multi-Anchor Evidence {idx}")
    if anchors:
        parts.append("Anchors: " + "; ".join(anchors))
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    order: list[str] = []
    for rec in records:
        title = str(rec.get("title") or "").strip() or (anchors[0] if anchors else "Evidence")
        if title not in grouped:
            grouped[title] = []
            order.append(title)
        grouped[title].append(rec)
    for title in order:
        local_before = len(seen)
        local: list[str] = [f"Anchor: {title}"]
        for rec in grouped[title]:
            _append_plain_sentence(local, seen, rec)
        if len(seen) > local_before:
            parts.extend(local)
    parts.append("")
    return True


def _render_chain_paragraph_hybrid(bundles: Sequence[Mapping[str, Any]], max_chars: int) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    idx = 1
    for bundle in bundles:
        if bundle.get("bundle_type") == "multi_anchor":
            if _render_multi_anchor_hybrid(parts, seen, bundle, idx):
                idx += 1
            continue
        chain = _chain_records(bundle)
        chain_norms = {r.get("norm", "") for r in chain if r.get("norm")}
        support = _prop_records(bundle, chain_norms)
        if chain:
            before = len(seen)
            block = [f"Evidence Chain {idx}"]
            anchor = str(bundle.get("anchor_title") or "").strip()
            bridge = ", ".join(
                str(x) for x in bundle.get("bridge_titles", []) or [] if str(x).strip()
            )
            if anchor:
                block.append(f"Question anchor: {anchor}")
            if bridge:
                block.append(f"Bridge entity: {bridge}")
            connection = [r for r in chain if str(r.get("role")) == "chain_seed"]
            answer = [r for r in chain if str(r.get("role")) == "chain_bridge"] + support
            if connection:
                block.append("Connection evidence:")
                for rec in connection:
                    _append_plain_sentence(block, seen, rec)
            if answer:
                block.append("Answer evidence:")
                for rec in answer:
                    _append_plain_sentence(block, seen, rec)
            if len(seen) > before:
                parts.extend(block)
                parts.append("")
                idx += 1
            continue
        records = support or _source_records(bundle, limit_per_chunk=1)
        if not records:
            continue
        before = len(seen)
        anchor = str(bundle.get("anchor_title") or bundle.get("title") or "").strip()
        block = [f"Supporting Evidence {idx}"]
        if anchor:
            block.append(f"Anchor: {anchor}")
        block.append("Evidence:")
        for rec in records:
            _append_plain_sentence(block, seen, rec)
        if len(seen) > before:
            parts.extend(block)
            parts.append("")
            idx += 1
    return safe_truncate("\n".join(parts).strip(), max_chars)


def render_evidence(
    bundles: Sequence[Mapping[str, Any]], budget: int, max_chars: int, layout: str
) -> str:
    """Render the first B bundles using the layout for the selected protocol."""
    bundles = bundles[:budget]
    if layout == "common":
        return render_topn_chain_dedup_context(bundles, max_chars, budget)
    if layout == "chain_paragraph":
        return _render_chain_paragraph_hybrid(bundles, max_chars)
    if layout == "chain_dedup":
        return _render_chain_dedup_ordered(_ordered_bundles(bundles), max_chars)
    raise ValueError(f"Unknown evidence layout: {layout}")


def source_titles(bundles: Sequence[Mapping[str, Any]], context: str) -> list[str]:
    """Map emitted evidence lines to source titles in their rendered order."""
    plain_lines: dict[str, set[str]] = {}
    tagged_lines: dict[str, str] = {}
    for bundle in bundles:
        for record in _chain_records(bundle) + _prop_records(bundle) + _source_records(bundle):
            text = record["text"]
            title = record["title"]
            if title:
                plain_lines.setdefault(text, set()).add(title)
                tagged_lines[f"- {title}: {text}"] = title
    titles: list[str] = []
    seen: set[str] = set()
    for line in context.splitlines():
        if line in tagged_lines:
            _append_unique_source_title(titles, seen, tagged_lines[line])
        elif not line.startswith(("Sources:", "Anchor:")):
            matches = plain_lines.get(line, set())
            if len(matches) == 1:
                _append_unique_source_title(titles, seen, next(iter(matches)))
    return titles
