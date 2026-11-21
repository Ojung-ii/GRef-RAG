"""Chain Expansion and Bundle Reranking in Local Refinement."""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..embedding import encode_query
from ..rendering import render_evidence
from ..text import jaccard, token_count, tokenize
from .lexical import (
    COMPARISON_TERMS,
    GENERIC_RELATION_TITLES,
    RESIDUAL_STOPWORDS,
    _bridge_rank_text,
    _coverage_count,
    _has_date_signal,
    _lexical_bm25_score,
    _norm_phrase,
    _norm_tokens,
    build_residual_query,
)


class LocalRefinement:
    """Chain Expansion and Bundle Reranking in Local Refinement."""

    def _comparison_like(self, question: str) -> bool:
        return bool(set(_norm_tokens(question)) & COMPARISON_TERMS)

    def _multi_anchor_residual_terms(
        self, question: str, anchor_titles: Sequence[str]
    ) -> list[str]:
        remove = set()
        for title in anchor_titles:
            remove.update(_norm_tokens(title))
        terms = []
        seen = set()
        for term in _norm_tokens(question):
            if term in remove or term in RESIDUAL_STOPWORDS or len(term) <= 1:
                continue
            if term not in seen:
                seen.add(term)
                terms.append(term)
        return terms or [t for t in _norm_tokens(question) if t not in RESIDUAL_STOPWORDS]

    def _semantic_prop_rank_key(
        self, prop: Mapping[str, Any], candidate_scores: Mapping[tuple[str, str], float]
    ) -> tuple[float, float, str]:
        pid = str(prop.get("prop_id", ""))
        score = float(candidate_scores.get(("proposition", pid), 0.0) or 0.0)
        return (
            score,
            -float(prop.get("token_count", token_count(prop.get("text", ""))) or 0.0),
            pid,
        )

    def _semantic_chunk_rank_key(
        self,
        chunk: Mapping[str, Any],
        candidate_scores: Mapping[tuple[str, str], float],
    ) -> tuple[float, float, str]:
        cid = str(chunk.get("chunk_id", ""))
        score = float(candidate_scores.get(("chunk", cid), 0.0) or 0.0)
        return (
            score,
            -float(chunk.get("token_count", token_count(chunk.get("text", ""))) or 0.0),
            cid,
        )

    def _semantic_same_title_bundle(
        self,
        question: str,
        seed: Mapping[str, Any],
        candidate_scores: Mapping[tuple[str, str], float],
        bundle_id: str,
    ) -> dict[str, Any] | None:
        title = str(seed.get("title") or seed.get("parent_title") or "")
        if not title:
            return None
        per_entity_props = int(self.refine_cfg.get("per_entity_propositions", 4))
        per_entity_chunks = int(self.refine_cfg.get("per_entity_chunks", 2))
        local_props = list(self.props_by_title.get(title, []))
        local_props.sort(
            key=lambda p: self._semantic_prop_rank_key(p, candidate_scores),
            reverse=True,
        )
        selected_props = [dict(p) for p in local_props[:per_entity_props]]
        if str(seed.get("unit")) == "proposition":
            seed_pid = str(seed.get("id") or seed.get("source_candidate_id") or "")
            if (
                seed_pid
                and seed_pid in self.prop_by_id
                and not any(str(p.get("prop_id")) == seed_pid for p in selected_props)
            ):
                selected_props.insert(0, dict(self.prop_by_id[seed_pid]))
                selected_props = selected_props[:per_entity_props]
        chunk_ids = set()
        selected_chunks = []
        for prop in selected_props:
            cid = str(prop.get("chunk_id", ""))
            if cid and cid not in chunk_ids and cid in self.chunk_by_id:
                selected_chunks.append(dict(self.chunk_by_id[cid]))
                chunk_ids.add(cid)
        if len(selected_chunks) < per_entity_chunks:
            chunks = list(self.chunks_by_title.get(title, []))
            chunks.sort(
                key=lambda c: self._semantic_chunk_rank_key(c, candidate_scores),
                reverse=True,
            )
            for chunk in chunks:
                cid = str(chunk.get("chunk_id", ""))
                if cid and cid not in chunk_ids:
                    selected_chunks.append(dict(chunk))
                    chunk_ids.add(cid)
                if len(selected_chunks) >= per_entity_chunks:
                    break
        if not selected_props and not selected_chunks:
            return None
        seed_type = self._seed_unit_type(seed)
        mentioned_titles = sorted(
            {
                t
                for p in selected_props
                for t in self.prop_to_mentioned_titles.get(str(p.get("prop_id", "")), [])
            }
        )
        path = [
            {"type": "query", "id": "q", "text": question},
            {"type": "entity", "id": title, "text": title},
        ]
        path.extend(
            {"type": "proposition", "id": p.get("prop_id"), "text": p.get("text", "")}
            for p in selected_props
        )
        score = float(seed.get("score", 0.0) or 0.0) + sum(
            float(candidate_scores.get(("proposition", str(p.get("prop_id", ""))), 0.0) or 0.0)
            for p in selected_props
        )
        self._remember_bundle_candidate(
            "same_title",
            anchor=title,
            seed=seed.get("id") or seed.get("source_candidate_id"),
            props=",".join(str(p.get("prop_id", "")) for p in selected_props if p.get("prop_id")),
            chunks=",".join(
                str(c.get("chunk_id", "")) for c in selected_chunks if c.get("chunk_id")
            ),
        )
        return {
            "bundle_id": bundle_id,
            "anchor_title": title,
            "seed": dict(seed),
            "seed_unit_type": seed_type,
            "original_candidate_type": str(
                seed.get("original_candidate_type") or seed.get("unit") or seed_type
            ),
            "selected_bundle_source_type": seed_type,
            "source_candidate_id": seed.get("source_candidate_id", seed.get("id")),
            "parent_title": seed.get("parent_title", title),
            "mentioned_titles": mentioned_titles,
            "propositions": selected_props,
            "source_chunks": selected_chunks,
            "evidence_path": path,
            "score": round(score, 6),
            "bridge_titles": [],
            "has_bridge": False,
            "bridge_connected": False,
            "answer_slot_aligned": False,
            "chain_complete_v2": False,
            "chain_complete": False,
            "residual_query": "",
            "residual_terms": [],
            "residual_coverage_count": 0,
            "ordering_group": "same_title",
            "diagnostics": {
                "refinement": "semantic same-title base",
                "seed_unit_type": seed_type,
            },
        }

    def _semantic_multi_anchor_bundle(
        self,
        question: str,
        query_anchor_info: Mapping[str, Any],
        candidate_scores: Mapping[tuple[str, str], float],
        bundle_id: str,
    ) -> dict[str, Any] | None:
        if not bool(self.bridge_cfg.get("multi_anchor_bundle", True)):
            return None
        anchors = [
            str(t) for t in query_anchor_info.get("query_anchor_titles", []) or [] if str(t).strip()
        ]
        if len(anchors) < 2:
            return None
        max_titles = int(self.bridge_cfg.get("max_multi_anchor_titles", 4))
        max_props = int(self.bridge_cfg.get("max_multi_anchor_props_per_title", 2))
        anchors = anchors[:max_titles]
        selected_props = []
        source_chunks = []
        chunk_ids = set()
        covered = []
        score = 0.0
        evidence_path = [{"type": "query", "id": "q", "text": question}]
        for title in anchors:
            props = list(self.props_by_title.get(title, []))
            props.sort(
                key=lambda p: self._semantic_prop_rank_key(p, candidate_scores),
                reverse=True,
            )
            picked = [dict(p) for p in props[:max_props]]
            if not picked and self.chunks_by_title.get(title):
                chunks = list(self.chunks_by_title.get(title, []))
                chunks.sort(
                    key=lambda c: self._semantic_chunk_rank_key(c, candidate_scores),
                    reverse=True,
                )
                c = dict(chunks[0])
                source_chunks.append(c)
                chunk_ids.add(str(c.get("chunk_id", "")))
                picked = [
                    {
                        "prop_id": None,
                        "title": title,
                        "text": c.get("text", ""),
                        "chunk_id": c.get("chunk_id"),
                    }
                ]
            if not picked:
                continue
            covered.append(title)
            for p in picked:
                if p.get("prop_id"):
                    selected_props.append(p)
                    score += float(
                        candidate_scores.get(("proposition", str(p.get("prop_id", ""))), 0.0) or 0.0
                    )
                    cid = str(p.get("chunk_id", ""))
                    if cid and cid not in chunk_ids and cid in self.chunk_by_id:
                        source_chunks.append(dict(self.chunk_by_id[cid]))
                        chunk_ids.add(cid)
                evidence_path.append(
                    {
                        "type": "multi_anchor",
                        "anchor_title": title,
                        "prop_id": p.get("prop_id"),
                        "text": p.get("text", ""),
                    }
                )
        if len(set(covered)) < 2:
            return None
        self._remember_bundle_candidate(
            "multi_anchor",
            anchors=";".join(anchors),
            props=",".join(str(p.get("prop_id", "")) for p in selected_props if p.get("prop_id")),
            chunks=",".join(str(c.get("chunk_id", "")) for c in source_chunks if c.get("chunk_id")),
        )
        return {
            "bundle_id": bundle_id,
            "bundle_type": "multi_anchor",
            "anchor_title": "; ".join(anchors),
            "anchor_titles": anchors,
            "seed": {
                "unit": "multi_anchor",
                "id": bundle_id,
                "title": "; ".join(anchors),
                "text": question,
                "seed_unit_type": "multi_anchor",
            },
            "seed_unit_type": "multi_anchor",
            "original_candidate_type": "multi_anchor",
            "selected_bundle_source_type": "multi_anchor",
            "source_candidate_id": bundle_id,
            "parent_title": "; ".join(anchors),
            "mentioned_titles": [],
            "propositions": selected_props,
            "source_chunks": source_chunks,
            "evidence_path": evidence_path,
            "score": round(score, 6),
            "bridge_titles": [],
            "has_bridge": False,
            "bridge_connected": False,
            "answer_slot_aligned": False,
            "chain_complete_v2": False,
            "chain_complete": False,
            "residual_query": "",
            "residual_terms": [],
            "residual_coverage_count": 0,
            "multi_anchor_complete": True,
            "covered_query_anchor_norms": [_norm_phrase(t) for t in covered],
            "diagnostics": {
                "refinement": "semantic multi-anchor candidate grouping",
                "anchors": len(anchors),
                "anchors_with_evidence": len(set(covered)),
            },
        }

    def _semantic_mention_rows_for_prop(
        self, prop_id: str, source_title: str
    ) -> list[Mapping[str, Any]]:
        if not self.bridge_enabled:
            return []
        skip_ambiguous = bool(self.bridge_cfg.get("skip_ambiguous_aliases", True))
        remove_self = bool(self.bridge_cfg.get("remove_self_mentions", True))
        rows = list(self.title_mentions_by_prop.get(str(prop_id), []))
        if not rows:
            rows = [
                {
                    "prop_id": str(prop_id),
                    "source_title": source_title,
                    "mentioned_title": title,
                    "mention": title,
                    "mention_norm": str(title).lower(),
                    "ambiguous": False,
                }
                for title in self.prop_to_mentioned_titles.get(str(prop_id), [])
            ]
        out = []
        for row in rows:
            if skip_ambiguous and row.get("ambiguous"):
                continue
            title = str(row.get("mentioned_title", ""))
            if remove_self and title == source_title:
                continue
            if not title:
                continue
            out.append(row)
        return out

    def _semantic_select_bridge_props(
        self,
        prefix_bundle: Mapping[str, Any],
        bridge_title: str,
        candidate_scores: Mapping[tuple[str, str], float],
        demand: Sequence[Mapping[str, Any]],
    ) -> list[tuple[Mapping[str, Any], dict[str, Any]]]:
        max_props = int(self.bridge_cfg.get("max_bridge_props_per_title", 2))
        candidates = list(self.props_by_title.get(str(bridge_title), []) or [])
        selected = []
        selected_ids = set()
        prefix = [prefix_bundle]
        current_coverage = self._semantic_coverage_vector(prefix, demand)
        for _ in range(max_props):
            best = None
            for prop in candidates:
                pid = str(prop.get("prop_id", ""))
                if not pid or pid in selected_ids:
                    continue
                tmp_bundle = {"propositions": [dict(prop)], "source_chunks": []}
                gain = self._semantic_marginal_gain(tmp_bundle, prefix, demand, current_coverage)
                score = float(candidate_scores.get(("proposition", pid), 0.0) or 0.0)
                tok = float(prop.get("token_count", token_count(prop.get("text", ""))) or 0.0)
                key = (gain, score, -tok, pid)
                if best is None or key > best[0]:
                    best = (key, prop, gain, score, tok)
            if best is None:
                break
            _, prop, gain, score, tok = best
            selected_ids.add(str(prop.get("prop_id", "")))
            info = {
                "semantic_selection_variant": self.local_refinement_strategy,
                "semantic_marginal_gain": round(float(gain), 8),
                "original_relevance_score": round(float(score), 6),
                "token_count": int(tok),
                "selected_bridge_prop_rank": len(selected) + 1,
            }
            selected.append((prop, info))
            prefix.append({"propositions": [dict(prop)], "source_chunks": []})
            current_coverage = self._semantic_coverage_vector(prefix, demand)
        return selected

    def _lexical_select_bridge_props(
        self,
        question: str,
        anchor: str,
        source_prop: Mapping[str, Any],
        bridge_title: str,
        existing_prop_ids: set[str],
    ) -> tuple[list[tuple[Mapping[str, Any], dict[str, Any]]], list[str]]:
        max_props = int(self.bridge_cfg.get("max_bridge_props_per_title", 2))
        residual_terms = build_residual_query(
            question, anchor, bridge_title, str(source_prop.get("text", ""))
        )
        candidates = list(self.props_by_title.get(str(bridge_title), []) or [])
        self._bump("bridge_prop_candidate_count", len(candidates))
        t0 = time.perf_counter()
        ranked = self._rank_bridge_props(
            candidates,
            residual_terms,
            _norm_tokens(question),
            bridge_title=bridge_title,
            variant="residual_lexical",
        )
        self._active_stage_durations["residual_bridge_selection"] += time.perf_counter() - t0
        self._active_stage_counts["residual_bridge_selection"] += 1
        selected = []
        seen = set(existing_prop_ids)
        for prop, info in ranked:
            pid = str(prop.get("prop_id", ""))
            if not pid or pid in seen:
                continue
            selected.append((prop, info))
            seen.add(pid)
            if len(selected) >= max_props:
                break
        return selected, residual_terms

    def _bundle_target_props(self, bundle: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        props = [p for p in bundle.get("propositions", []) or [] if isinstance(p, Mapping)]
        bridge_props = [p for p in props if isinstance(p.get("bridge_selection"), Mapping)]
        return bridge_props or props

    def _bundle_target_text(self, bundle: Mapping[str, Any]) -> str:
        parts = [str(p.get("text", "")) for p in self._bundle_target_props(bundle)]
        if not parts:
            parts = [
                str(c.get("text", ""))
                for c in bundle.get("source_chunks", []) or []
                if isinstance(c, Mapping)
            ]
        return " ".join(x for x in parts if x)

    def _bundle_global_relevance(
        self,
        bundle: Mapping[str, Any],
        candidate_scores: Mapping[tuple[str, str], float],
    ) -> float:
        total = 0.0
        seen = set()
        target_props = self._bundle_target_props(bundle)
        for prop in target_props:
            pid = str(prop.get("prop_id", ""))
            if pid and pid not in seen:
                seen.add(pid)
                total += float(candidate_scores.get(("proposition", pid), 0.0) or 0.0)
        if target_props:
            return float(total)
        for chunk in bundle.get("source_chunks", []) or []:
            cid = str(chunk.get("chunk_id", ""))
            if cid and cid not in seen:
                seen.add(cid)
                total += float(candidate_scores.get(("chunk", cid), 0.0) or 0.0)
        if not seen:
            total = float(bundle.get("score", 0.0) or 0.0)
        return float(total)

    def _bundle_residual_terms(self, question: str, bundle: Mapping[str, Any]) -> list[str]:
        terms = [str(t) for t in bundle.get("residual_terms", []) or [] if str(t)]
        if terms:
            return terms
        bridge_path = next(
            (
                p
                for p in bundle.get("evidence_path", []) or []
                if isinstance(p, Mapping) and p.get("path_type") == "mention_bridge"
            ),
            {},
        )
        if bridge_path:
            return build_residual_query(
                question,
                str(bundle.get("anchor_title", "")),
                str(bridge_path.get("bridge_title", "")),
                str(bridge_path.get("seed_prop", "")),
            )
        seed = bundle.get("seed") if isinstance(bundle.get("seed"), Mapping) else {}
        return build_residual_query(
            question, str(bundle.get("anchor_title", "")), "", str(seed.get("text", ""))
        )

    def _lexical_bundle_rank_key(
        self,
        question: str,
        bundle: Mapping[str, Any],
        candidate_scores: Mapping[tuple[str, str], float],
        incremental_tokens: int | None = None,
    ) -> tuple[float, float, float, float, str]:
        self._bump("lexical_selector_score_call_count")
        residual_terms = self._bundle_residual_terms(question, bundle)
        target_text = self._bundle_target_text(bundle)
        coverage = float(_coverage_count(residual_terms, target_text))
        query_terms = [
            t for t in _norm_tokens(question) if t not in RESIDUAL_STOPWORDS and len(t) > 1
        ]
        query_relevance = _lexical_bm25_score(query_terms, target_text) + jaccard(
            query_terms, _norm_tokens(target_text)
        )
        global_relevance = self._bundle_global_relevance(bundle, candidate_scores)
        inc = float(
            incremental_tokens
            if incremental_tokens is not None
            else max(1, token_count(target_text))
        )
        stable = str(
            bundle.get("semantic_bundle_signature") or self._semantic_bundle_signature(bundle)
        )
        return (coverage, float(query_relevance), float(global_relevance), -inc, stable)

    def _attach_lexical_bundle_score(
        self,
        question: str,
        bundle: Mapping[str, Any],
        candidate_scores: Mapping[tuple[str, str], float],
        incremental_tokens: int | None = None,
    ) -> dict[str, Any]:
        b = dict(bundle)
        residual_terms = self._bundle_residual_terms(question, b)
        target_text = self._bundle_target_text(b)
        query_terms = [
            t for t in _norm_tokens(question) if t not in RESIDUAL_STOPWORDS and len(t) > 1
        ]
        coverage = float(_coverage_count(residual_terms, target_text))
        query_relevance = _lexical_bm25_score(query_terms, target_text) + jaccard(
            query_terms, _norm_tokens(target_text)
        )
        global_relevance = self._bundle_global_relevance(b, candidate_scores)
        inc = int(
            incremental_tokens
            if incremental_tokens is not None
            else max(1, token_count(target_text))
        )
        key = (
            coverage,
            float(query_relevance),
            float(global_relevance),
            -float(inc),
            str(b.get("semantic_bundle_signature") or self._semantic_bundle_signature(b)),
        )
        b["bundle_selector_variant"] = "lexical_residual"
        b["lexical_residual_terms"] = residual_terms
        b["lexical_residual_coverage"] = round(float(coverage), 6)
        b["lexical_query_relevance"] = round(float(query_relevance), 6)
        b["lexical_global_relevance"] = round(float(global_relevance), 6)
        b["lexical_incremental_tokens"] = inc
        b["lexical_bundle_rank_key"] = [x if not isinstance(x, float) else round(x, 8) for x in key]
        diag = dict(b.get("diagnostics", {}) or {})
        diag.update(
            {
                "bundle_selector_variant": "lexical_residual",
                "lexical_residual_coverage": b["lexical_residual_coverage"],
                "lexical_query_relevance": b["lexical_query_relevance"],
                "lexical_global_relevance": b["lexical_global_relevance"],
                "lexical_incremental_tokens": inc,
            }
        )
        b["diagnostics"] = diag
        return b

    def _lexical_order_bundles(
        self,
        question: str,
        bundles: Sequence[Mapping[str, Any]],
        candidate_scores: Mapping[tuple[str, str], float],
        query_anchor_info: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        finalized = [self._finalize_bundle(question, dict(b), query_anchor_info) for b in bundles]
        attached = [
            self._attach_lexical_bundle_score(question, b, candidate_scores) for b in finalized
        ]
        return sorted(
            attached,
            key=lambda b: self._lexical_bundle_rank_key(
                question,
                b,
                candidate_scores,
                int(b.get("lexical_incremental_tokens", 0) or 0),
            ),
            reverse=True,
        )

    def _semantic_bridge_candidate_bundles(
        self,
        base_bundle: Mapping[str, Any],
        question: str,
        candidate_scores: Mapping[tuple[str, str], float],
        demand: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        if not self.bridge_enabled:
            return []
        anchor = str(base_bundle.get("anchor_title", ""))
        base_props = list(base_bundle.get("propositions", []) or [])
        if not base_props:
            return []
        max_scan = int(self.bridge_cfg.get("max_bridge_source_props_per_seed", 16))
        source_props = []
        seen_source = set()
        for prop in base_props:
            pid = str(prop.get("prop_id", ""))
            if pid:
                source_props.append(prop)
                seen_source.add(pid)
        for prop in self.props_by_title.get(anchor, [])[:max_scan]:
            pid = str(prop.get("prop_id", ""))
            if pid and pid not in seen_source and self.prop_to_mentioned_titles.get(pid):
                source_props.append(prop)
                seen_source.add(pid)
        bridge_by_title: dict[str, tuple[tuple[float, float, float, str], dict[str, Any]]] = {}
        mention_t0 = time.perf_counter()
        for source_prop in source_props:
            for row in self._semantic_mention_rows_for_prop(
                str(source_prop.get("prop_id", "")), str(source_prop.get("title", ""))
            ):
                title = str(row.get("mentioned_title", ""))
                if not title or title not in self.props_by_title:
                    continue
                prefix = dict(base_bundle)
                existing_prop_ids = {
                    str(p.get("prop_id"))
                    for p in prefix.get("propositions", [])
                    if p.get("prop_id")
                }
                existing_chunk_ids = {
                    str(c.get("chunk_id"))
                    for c in prefix.get("source_chunks", [])
                    if c.get("chunk_id")
                }
                src_pid = str(source_prop.get("prop_id", ""))
                bridge_source_props = []
                if src_pid and src_pid not in existing_prop_ids:
                    bridge_source_props.append(dict(source_prop))
                    existing_prop_ids.add(src_pid)
                source_chunks = [dict(c) for c in prefix.get("source_chunks", []) or []]
                source_cid = str(source_prop.get("chunk_id", ""))
                if (
                    source_cid
                    and source_cid not in existing_chunk_ids
                    and source_cid in self.chunk_by_id
                ):
                    source_chunks.append(dict(self.chunk_by_id[source_cid]))
                    existing_chunk_ids.add(source_cid)
                target_candidates = list(self.props_by_title.get(title, []) or [])
                for target_prop in target_candidates:
                    self._remember_bundle_candidate(
                        "bridge_target",
                        anchor=anchor,
                        base=self._semantic_bundle_signature(base_bundle),
                        source_prop=src_pid,
                        bridge_title=title,
                        target_prop=target_prop.get("prop_id"),
                        target_chunk=target_prop.get("chunk_id"),
                    )
                semantic_prefix = {
                    **prefix,
                    "propositions": list(prefix.get("propositions", []) or [])
                    + bridge_source_props,
                    "source_chunks": source_chunks,
                }
                if self.bundle_selector_variant == "lexical_residual":
                    selected, residual_terms = self._lexical_select_bridge_props(
                        question, anchor, source_prop, title, existing_prop_ids
                    )
                else:
                    selected = self._semantic_select_bridge_props(
                        semantic_prefix, title, candidate_scores, demand
                    )
                    residual_terms = []
                if not selected:
                    continue
                bridge_props = []
                bridge_paths = []
                score = float(prefix.get("score", 0.0) or 0.0)
                total_gain = 0.0
                total_residual_coverage = 0
                residual_terms_all = []
                for prop, info in selected:
                    pid = str(prop.get("prop_id", ""))
                    if pid in existing_prop_ids:
                        continue
                    prop_dict = dict(prop)
                    prop_dict["bridge_selection"] = {
                        k: v for k, v in info.items() if k != "rank_key"
                    }
                    bridge_props.append(prop_dict)
                    existing_prop_ids.add(pid)
                    if self.bundle_selector_variant == "lexical_residual":
                        score += float(candidate_scores.get(("proposition", pid), 0.0) or 0.0)
                        total_residual_coverage += int(info.get("residual_coverage_count", 0) or 0)
                    else:
                        score += float(info.get("original_relevance_score", 0.0) or 0.0)
                    total_gain += float(info.get("semantic_marginal_gain", 0.0) or 0.0)
                    cid = str(prop.get("chunk_id", ""))
                    if cid and cid not in existing_chunk_ids and cid in self.chunk_by_id:
                        source_chunks.append(dict(self.chunk_by_id[cid]))
                        existing_chunk_ids.add(cid)
                    if self.bundle_selector_variant == "lexical_residual":
                        for term in residual_terms:
                            if term not in residual_terms_all:
                                residual_terms_all.append(term)
                    path_info = {
                        "path_type": "mention_bridge",
                        "source_title": str(source_prop.get("title", "")),
                        "seed_prop": str(source_prop.get("text", "")),
                        "mention": row.get("mention"),
                        "bridge_title": title,
                        "bridge_prop": str(prop.get("text", "")),
                        "seed_prop_id": source_prop.get("prop_id"),
                        "bridge_prop_id": prop.get("prop_id"),
                        "bundle_selector_variant": self.bundle_selector_variant,
                        "semantic_selection_variant": self.local_refinement_strategy,
                        "semantic_marginal_gain": info.get("semantic_marginal_gain"),
                        "original_relevance_score": info.get("original_relevance_score"),
                        "selected_bridge_prop_rank": info.get("selected_bridge_prop_rank"),
                        "bridge_prop_candidate_count": len(target_candidates),
                    }
                    if self.bundle_selector_variant == "lexical_residual":
                        path_info.update(
                            {
                                "residual_query": " ".join(residual_terms),
                                "residual_terms": residual_terms,
                                "residual_selection_variant": "residual_lexical",
                                "residual_coverage_count": info.get("residual_coverage_count", 0),
                                "residual_score": info.get("residual_score", 0.0),
                                "residual_dense_score": info.get("residual_dense_score", 0.0),
                            }
                        )
                    bridge_paths.append(path_info)
                if not bridge_props:
                    continue
                bundle = dict(prefix)
                bundle["propositions"] = (
                    list(prefix.get("propositions", []) or []) + bridge_source_props + bridge_props
                )
                bundle["source_chunks"] = source_chunks
                bundle["bridge_titles"] = [title]
                bundle["mentioned_titles"] = sorted(
                    set(list(prefix.get("mentioned_titles", []) or []) + [title])
                )
                bundle["has_bridge"] = True
                bundle["bridge_prop_count"] = len(bridge_props)
                bundle["bridge_connected"] = True
                bundle["answer_slot_aligned"] = (
                    total_residual_coverage > 0
                    if self.bundle_selector_variant == "lexical_residual"
                    else False
                )
                bundle["semantic_bridge_aligned"] = total_gain > 0.0
                bundle["chain_complete_v2"] = bool(
                    bundle["answer_slot_aligned"] and bundle["bridge_connected"]
                )
                bundle["chain_complete"] = bundle["chain_complete_v2"]
                bundle["ordering_group"] = (
                    "chain_complete_v2" if bundle["chain_complete_v2"] else "bridge_connected"
                )
                bundle["evidence_path"] = list(prefix.get("evidence_path", []) or []) + bridge_paths
                bundle["score"] = round(
                    score
                    + (total_gain if self.bundle_selector_variant != "lexical_residual" else 0.0),
                    6,
                )
                bundle["semantic_marginal_gain"] = round(total_gain, 8)
                if self.bundle_selector_variant != "lexical_residual":
                    bundle["semantic_facility_value"] = round(
                        self._semantic_facility_value([bundle], demand), 8
                    )
                bundle["residual_terms"] = residual_terms_all
                bundle["residual_query"] = " ".join(residual_terms_all)
                bundle["residual_coverage_count"] = total_residual_coverage
                bundle["bundle_selector_variant"] = self.bundle_selector_variant
                diag = dict(bundle.get("diagnostics", {}) or {})
                diag.update(
                    {
                        "refinement": "semantic mention-bridge expansion",
                        "bundle_selector_variant": self.bundle_selector_variant,
                        "semantic_selection_variant": self.local_refinement_strategy,
                        "semantic_marginal_gain": round(total_gain, 8),
                        "bridge_connected": True,
                        "bridge_titles": 1,
                        "bridge_prop_count": len(bridge_props),
                        "answer_slot_aligned": bundle["answer_slot_aligned"],
                        "answer_slot_alignment_not_used": self.bundle_selector_variant
                        != "lexical_residual",
                        "residual_coverage_count": total_residual_coverage,
                    }
                )
                bundle["diagnostics"] = diag
                if self.bundle_selector_variant == "lexical_residual":
                    key = self._lexical_bundle_rank_key(question, bundle, candidate_scores)
                else:
                    key = (
                        float(bundle.get("semantic_marginal_gain", 0.0) or 0.0),
                        float(bundle.get("score", 0.0) or 0.0),
                        -float(
                            sum(
                                token_count(p.get("text", ""))
                                for p in bundle.get("propositions", []) or []
                            )
                        ),
                        self._semantic_bundle_signature(bundle),
                    )
                if title not in bridge_by_title or key > bridge_by_title[title][0]:
                    bridge_by_title[title] = (key, bundle)
        mention_elapsed = time.perf_counter() - mention_t0
        self._active_stage_durations["mention_bridge_expansion"] += mention_elapsed
        self._active_stage_counts["mention_bridge_expansion"] += 1
        out = [bundle for _, bundle in bridge_by_title.values()]
        self._bump("semantic_refinement_candidate_bundle_count", len(out))
        return out

    def _semantic_direct_candidate_bundle(
        self, question: str, cand: Mapping[str, Any], bundle_id: str
    ) -> dict[str, Any] | None:
        title = str(cand.get("title") or cand.get("parent_title") or "")
        if not title:
            return None
        chunk = self.chunk_by_id.get(cand.get("chunk_id")) if cand.get("chunk_id") else None
        prop = self.prop_by_id.get(cand.get("id")) if cand.get("unit") == "proposition" else None
        bprops = [dict(prop)] if prop else []
        bchunks = [dict(chunk)] if chunk else ([dict(cand)] if cand.get("unit") == "chunk" else [])
        if not bprops and not bchunks:
            return None
        seed_type = self._seed_unit_type(cand)
        self._remember_bundle_candidate(
            "direct_candidate",
            anchor=title,
            candidate=cand.get("id") or cand.get("source_candidate_id"),
            props=",".join(str(p.get("prop_id", "")) for p in bprops if p.get("prop_id")),
            chunks=",".join(str(c.get("chunk_id", "")) for c in bchunks if c.get("chunk_id")),
        )
        return {
            "bundle_id": bundle_id,
            "anchor_title": title,
            "seed": dict(cand),
            "seed_unit_type": seed_type,
            "original_candidate_type": str(
                cand.get("original_candidate_type") or cand.get("unit") or seed_type
            ),
            "selected_bundle_source_type": seed_type,
            "source_candidate_id": cand.get("source_candidate_id", cand.get("id")),
            "parent_title": cand.get("parent_title", title),
            "mentioned_titles": list(cand.get("mentioned_titles") or []),
            "propositions": bprops,
            "source_chunks": bchunks,
            "evidence_path": [
                {"type": "query", "id": "q", "text": question},
                {
                    "type": "candidate",
                    "id": cand.get("id"),
                    "text": cand.get("text", ""),
                },
            ],
            "score": round(float(cand.get("score", 0.0) or 0.0), 6),
            "bridge_titles": [],
            "has_bridge": False,
            "bridge_connected": False,
            "answer_slot_aligned": False,
            "chain_complete_v2": False,
            "chain_complete": False,
            "residual_query": "",
            "residual_terms": [],
            "residual_coverage_count": 0,
            "ordering_group": "fallback",
            "diagnostics": {
                "refinement": "semantic direct-candidate fallback",
                "seed_unit_type": seed_type,
            },
        }

    def _dedup_semantic_bundles(self, bundles: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        out = []
        seen = set()
        for bundle in bundles:
            sig = self._semantic_bundle_signature(bundle)
            if sig in seen:
                continue
            seen.add(sig)
            b = dict(bundle)
            b["semantic_bundle_signature"] = sig
            out.append(b)
        return out

    def _semantic_order_bundles(
        self,
        question: str,
        bundles: Sequence[Mapping[str, Any]],
        demand: Sequence[Mapping[str, Any]],
        query_anchor_info: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        finalized = [self._finalize_bundle(question, dict(b), query_anchor_info) for b in bundles]
        for b in finalized:
            b["semantic_facility_value"] = round(self._semantic_facility_value([b], demand), 8)
        return sorted(
            finalized,
            key=lambda b: (
                float(b.get("semantic_facility_value", 0.0) or 0.0),
                float(b.get("semantic_marginal_gain", 0.0) or 0.0),
                float(b.get("score", 0.0) or 0.0),
                -float(
                    sum(token_count(p.get("text", "")) for p in b.get("propositions", []) or [])
                    or sum(token_count(c.get("text", "")) for c in b.get("source_chunks", []) or [])
                ),
                str(b.get("semantic_bundle_signature") or b.get("bundle_id") or ""),
            ),
            reverse=True,
        )

    def _expand_and_rerank_bundles(
        self,
        question: str,
        seeds: Sequence[Mapping[str, Any]],
        candidates: Sequence[Mapping[str, Any]],
        query_anchor_info: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        if not seeds:
            return []
        max_bundles = int(self.refine_cfg.get("max_bundles", 6))
        q_toks = tokenize(question)
        _ = q_toks
        candidate_scores = {
            (str(c.get("unit")), str(c.get("id"))): float(c.get("score", 0.0)) for c in candidates
        }
        demand = (
            []
            if self.bundle_selector_variant == "lexical_residual"
            else self._semantic_demand(candidates)
        )
        bundles = []
        multi = self._semantic_multi_anchor_bundle(
            question, query_anchor_info, candidate_scores, f"b{len(bundles)}"
        )
        if multi is not None:
            bundles.append(multi)
        seen_titles = set()
        for seed in seeds:
            title = str(seed.get("title") or seed.get("parent_title") or "")
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)
            base = self._semantic_same_title_bundle(
                question, seed, candidate_scores, f"b{len(bundles)}"
            )
            if base is None:
                continue
            bridge_candidates = self._semantic_bridge_candidate_bundles(
                base, question, candidate_scores, demand
            )
            if bridge_candidates:
                if self.bundle_selector_variant == "lexical_residual":
                    bridge_candidates = [
                        self._attach_lexical_bundle_score(question, b, candidate_scores)
                        for b in bridge_candidates
                    ]
                    best = max(
                        bridge_candidates,
                        key=lambda b: self._lexical_bundle_rank_key(question, b, candidate_scores),
                    )
                else:
                    current_cov = self._semantic_coverage_vector([base], demand)
                    for cand_bundle in bridge_candidates:
                        cand_bundle["semantic_marginal_gain"] = round(
                            self._semantic_marginal_gain(cand_bundle, [base], demand, current_cov),
                            8,
                        )
                    best = max(
                        bridge_candidates,
                        key=lambda b: (
                            float(b.get("semantic_marginal_gain", 0.0) or 0.0),
                            float(b.get("semantic_facility_value", 0.0) or 0.0),
                            float(b.get("score", 0.0) or 0.0),
                            -float(
                                sum(
                                    token_count(p.get("text", ""))
                                    for p in b.get("propositions", []) or []
                                )
                            ),
                            self._semantic_bundle_signature(b),
                        ),
                    )
                best["bundle_id"] = f"b{len(bundles)}"
                best["semantic_v2_selected_from_candidates"] = len(bridge_candidates)
                bundles.append(best)
            else:
                bundles.append(base)
        for cand in candidates:
            if len(bundles) >= max_bundles:
                break
            title = str(cand.get("title") or cand.get("parent_title") or "")
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)
            direct = self._semantic_direct_candidate_bundle(question, cand, f"b{len(bundles)}")
            if direct is not None:
                bundles.append(direct)
        bundles = self._dedup_semantic_bundles(bundles)
        order_t0 = time.perf_counter()
        if self.bundle_selector_variant == "lexical_residual":
            ordered = self._lexical_order_bundles(
                question, bundles, candidate_scores, query_anchor_info
            )
        else:
            ordered = self._semantic_order_bundles(question, bundles, demand, query_anchor_info)
        order_elapsed = time.perf_counter() - order_t0
        self._active_stage_durations["anchor_chain_ordering"] += order_elapsed
        self._active_stage_counts["anchor_chain_ordering"] += 1
        self._bump("semantic_refinement_selected_count", min(max_bundles, len(ordered)))
        return ordered[:max_bundles]

    def _rendered_context_tokens_for_bundles(self, bundles: Sequence[Mapping[str, Any]]) -> int:
        configuration = self.cfg["_generation"]
        context = render_evidence(
            bundles,
            configuration["top_bundles"],
            configuration["max_context_chars"],
            configuration["layout"],
        )
        return self.context_token_counter(context)

    def _candidate_bridge_idf(self, candidates: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        df: Counter[str] = Counter()
        for prop in candidates:
            df.update(set(_norm_tokens(prop.get("text", ""))))
        n = max(1, len(candidates))
        return {term: math.log(1.0 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def _residual_dense_query_vec(self, residual_text: str):
        text = str(residual_text or "").strip()
        if not text:
            return None
        if self._residual_dense_query_vector is not None:
            return self._residual_dense_query_vector
        vec = None
        try:
            if self.dense_indexes:
                vec = encode_query(
                    {"embedding": self.cfg.get("embedding", {})},
                    text,
                    logger=self.logger,
                    cache=self._embed_cache,
                )
        except Exception as exc:
            if self.logger:
                self.logger.log(f"residual.dense.warning: failed to encode residual query: {exc!r}")
            vec = None
        if vec is not None:
            self._bump("residual_dense_embedding_calls")
            self._residual_dense_query_vector = np.asarray(vec, dtype=np.float32).reshape(-1)
            self._residual_dense_query_text = text
        return self._residual_dense_query_vector

    def _residual_dense_score(self, prop: Mapping[str, Any], residual_text: str) -> float:
        pid = str(prop.get("prop_id", ""))
        if not pid:
            return 0.0
        key = (pid, str(residual_text or ""))
        cache = getattr(self, "_residual_dense_score_cache", None)
        if cache is not None and key in cache:
            self._bump("residual_dense_similarity_cache_hits")
            return float(cache[key])
        qv = self._residual_dense_query_vec(residual_text)
        if qv is None:
            return 0.0
        score = 0.0
        if self._dense_prop_matrix is not None:
            pos = self.prop_dense_pos.get(pid)
            if pos is not None and pos < self._dense_prop_matrix.matrix.shape[0]:
                pv = np.asarray(self._dense_prop_matrix.matrix[pos], dtype=np.float32).reshape(-1)
                score = float(pv @ qv)
        elif "proposition" in self.dense_indexes:
            dense = self.dense_indexes["proposition"]
            pos = self.sentence_vector_positions.get(pid)
            if pos is not None and pos < dense.matrix.shape[0]:
                pv = np.asarray(dense.matrix[pos], dtype=np.float32).reshape(-1)
                score = float(pv @ qv)
        self._bump("residual_dense_used")
        self._bump("residual_dense_similarity_computations")
        if cache is not None:
            cache[key] = score
        return score

    def _rank_bridge_props(
        self,
        candidates: Sequence[Mapping[str, Any]],
        residual_terms: Sequence[str],
        q_toks: Sequence[str],
        *,
        bridge_title: str = "",
        full_query_terms: Sequence[str] | None = None,
        variant: str | None = None,
    ) -> list[tuple[Mapping[str, Any], dict[str, Any]]]:
        if not candidates:
            return []
        variant = str(variant or self.residual_selection_variant)
        scoring_terms = list(q_toks if variant == "bridge_fullquery" else residual_terms)
        residual_text = " ".join(scoring_terms)
        idf = self._candidate_bridge_idf(candidates)
        avgdl = sum(token_count(p.get("text", "")) for p in candidates) / max(1, len(candidates))
        ranked = []
        precomputed = []
        for idx, prop in enumerate(candidates):
            text = prop.get("text", "")
            rank_text = _bridge_rank_text(prop, candidates[idx - 1] if idx > 0 else None)
            coverage = _coverage_count(scoring_terms, rank_text)
            residual_score = _lexical_bm25_score(scoring_terms, rank_text, idf, avgdl)
            original_score = jaccard(q_toks, _norm_tokens(rank_text))
            dense_score = 0.0
            if variant in {
                "residual_dense_only",
                "residual_hybrid_lex_first",
                "residual_dense_fallback",
                "residual_unified_alignment",
            }:
                dense_score = self._residual_dense_score(prop, residual_text)
            tok_count = int(prop.get("token_count", token_count(text)) or 0)
            orientation_valid = bool(
                not bridge_title
                or _norm_phrase(prop.get("title", "")) == _norm_phrase(bridge_title)
            )
            precomputed.append(
                (
                    coverage,
                    residual_score,
                    dense_score,
                    original_score,
                    tok_count,
                    orientation_valid,
                )
            )
        lexical_available = any(x[0] > 0 for x in precomputed)
        for idx, prop in enumerate(candidates):
            text = prop.get("text", "")
            rank_text = _bridge_rank_text(prop, candidates[idx - 1] if idx > 0 else None)
            (
                coverage,
                residual_score,
                dense_score,
                original_score,
                tok_count,
                orientation_valid,
            ) = precomputed[idx]
            if variant in {"residual_lexical", "bridge_fullquery"}:
                rank_key = (
                    coverage > 0,
                    coverage,
                    residual_score,
                    original_score,
                    -tok_count,
                )
            elif variant == "residual_dense_only":
                rank_key = (orientation_valid, dense_score, original_score, -tok_count)
            elif variant == "residual_dense_fallback":
                if lexical_available:
                    rank_key = (
                        coverage > 0,
                        coverage,
                        residual_score,
                        original_score,
                        -tok_count,
                    )
                else:
                    rank_key = (
                        orientation_valid,
                        dense_score,
                        original_score,
                        -tok_count,
                    )
            elif variant in {"residual_hybrid_lex_first", "residual_unified_alignment"}:
                rank_key = (
                    orientation_valid,
                    coverage > 0,
                    coverage,
                    residual_score,
                    dense_score,
                    original_score,
                    -tok_count,
                )
            else:
                rank_key = (
                    coverage > 0,
                    coverage,
                    residual_score,
                    original_score,
                    -tok_count,
                )
            info = {
                "residual_selection_variant": variant,
                "residual_coverage_count": coverage,
                "residual_score": round(float(residual_score), 6),
                "residual_dense_score": round(float(dense_score), 6),
                "original_relevance_score": round(float(original_score), 6),
                "orientation_valid": orientation_valid,
                "token_count": tok_count,
                "rank_key": rank_key,
            }
            if rank_text != text:
                info["display_text"] = rank_text
            ranked.append((prop, info))
        ranked.sort(key=lambda x: x[1]["rank_key"], reverse=True)
        for rank, (_, info) in enumerate(ranked, start=1):
            info["selected_bridge_prop_rank"] = rank
        return ranked

    def _rank_multi_anchor_props(
        self,
        candidates: Sequence[Mapping[str, Any]],
        residual_terms: Sequence[str],
        q_toks: Sequence[str],
        comparison_like: bool,
    ) -> list[tuple[Mapping[str, Any], dict[str, Any]]]:
        ranked = self._rank_bridge_props(
            candidates, residual_terms, q_toks, variant="residual_lexical"
        )
        if comparison_like:
            ranked.sort(
                key=lambda x: (
                    _has_date_signal(x[0].get("text", "")),
                    x[1]["rank_key"],
                ),
                reverse=True,
            )
        return ranked

    def _is_generic_relation_title(self, question: str, title: str) -> bool:
        if not bool(self.bridge_cfg.get("demote_generic_relation_titles", True)):
            return False
        norm_title = _norm_phrase(title)
        if not norm_title:
            return False
        if norm_title in GENERIC_RELATION_TITLES:
            return True
        q_norm = _norm_phrase(question)
        return norm_title in q_norm and any(
            phrase in norm_title for phrase in GENERIC_RELATION_TITLES
        )

    def _bundle_evidence_title_norms(self, bundle: Mapping[str, Any]) -> set[str]:
        norms = set()
        for key in ("anchor_title",):
            if bundle.get(key):
                norms.add(_norm_phrase(bundle.get(key)))
        for title in bundle.get("anchor_titles", []) or []:
            norms.add(_norm_phrase(title))
        for path in bundle.get("evidence_path", []) or []:
            if not isinstance(path, Mapping):
                continue
            for key in ("source_title", "bridge_title", "anchor_title"):
                if path.get(key):
                    norms.add(_norm_phrase(path.get(key)))
        return {n for n in norms if n}

    def _finalize_bundle(
        self,
        question: str,
        bundle: Mapping[str, Any],
        query_anchor_info: Mapping[str, Any],
    ) -> dict[str, Any]:
        b = dict(bundle)
        is_generic = self._is_generic_relation_title(question, str(b.get("anchor_title", "")))
        exact = self._exact_anchor_match(question, str(b.get("anchor_title", "")))
        query_anchor_titles = [
            str(t) for t in query_anchor_info.get("query_anchor_titles", []) or []
        ]
        query_relation_titles = [
            str(t) for t in query_anchor_info.get("query_relation_titles", []) or []
        ]
        query_anchor_norms = {_norm_phrase(t) for t in query_anchor_titles}
        query_relation_norms = {_norm_phrase(t) for t in query_relation_titles}
        anchor_norm = _norm_phrase(b.get("anchor_title", ""))
        evidence_norms = self._bundle_evidence_title_norms(b)
        covered_norms = sorted(evidence_norms & query_anchor_norms)
        is_query_anchor = bool(anchor_norm and anchor_norm in query_anchor_norms)
        is_relation = bool(is_generic or (anchor_norm and anchor_norm in query_relation_norms))
        anchor_connected = bool(is_query_anchor or covered_norms)
        bridge_connected = bool(b.get("bridge_connected", b.get("has_bridge", False)))
        aligned = bool(
            b.get(
                "answer_slot_aligned",
                float(b.get("residual_coverage_count", 0.0) or 0.0) > 0.0,
            )
        )
        chain_v2 = bridge_connected and aligned
        anchor_connected_chain_complete = bool(chain_v2 and anchor_connected)
        anchor_mismatch_chain = bool(chain_v2 and not anchor_connected)
        b["query_anchor_titles"] = query_anchor_titles
        b["query_relation_titles"] = query_relation_titles
        b["is_query_anchor_bundle"] = is_query_anchor
        b["is_relation_title_bundle"] = is_relation
        b["anchor_connected"] = anchor_connected
        b["anchor_connected_chain_complete"] = anchor_connected_chain_complete
        b["anchor_mismatch_chain"] = anchor_mismatch_chain
        b["covered_query_anchor_norms"] = covered_norms
        b["is_generic_relation_title"] = is_generic
        b["exact_anchor_match"] = exact
        b["bridge_connected"] = bridge_connected
        b["answer_slot_aligned"] = aligned
        b["chain_complete_v2"] = chain_v2
        b["old_chain_complete"] = bool(b.get("old_chain_complete", b.get("chain_complete", False)))
        b["chain_complete"] = chain_v2
        if b.get("bundle_type") == "multi_anchor":
            b["ordering_group"] = "multi_anchor"
        elif anchor_connected_chain_complete:
            b["ordering_group"] = "anchor_connected_chain_complete"
        elif is_relation:
            b["ordering_group"] = "generic_relation"
        elif chain_v2:
            b["ordering_group"] = "chain_complete_v2"
        elif is_query_anchor:
            b["ordering_group"] = "anchor"
        elif bridge_connected:
            b["ordering_group"] = "bridge_connected"
        elif exact:
            b["ordering_group"] = "anchor"
        elif b.get("propositions"):
            b["ordering_group"] = "same_title"
        else:
            b["ordering_group"] = "fallback"
        return b

    def _exact_anchor_match(self, question: str, title: str) -> bool:
        q = set(_norm_tokens(question))
        t = set(_norm_tokens(title))
        return bool(t and t.issubset(q))
