"""Global Seed Search across document anchors, chunks and sentences."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from ..embedding import encode_query
from ..text import token_count
from .lexical import (
    GENERIC_RELATION_TITLES,
    RESIDUAL_STOPWORDS,
    _alias_spans,
    _is_generic_relation_norm,
    _norm_phrase,
    _norm_tokens,
    _title_aliases_for_query,
)


def _rank_norm(keys: Sequence[Any]) -> dict[Any, float]:
    return {key: 1.0 / (rank + 1) for rank, key in enumerate(keys)}


class GlobalSeedSearch:
    """Global Seed Search across document anchors, chunks and sentences."""

    def _build_query_title_aliases(self) -> dict[str, list[str]]:
        alias_to_titles: dict[str, set[str]] = defaultdict(set)
        for alias, titles in self.title_aliases.items():
            alias_norm = _norm_phrase(alias)
            if not alias_norm:
                continue
            for title in titles or []:
                if str(title).strip():
                    alias_to_titles[alias_norm].add(str(title))
        for ent in self.entities:
            title = str(ent.get("title", "")).strip()
            if not title:
                continue
            for alias in _title_aliases_for_query(title):
                if (
                    alias
                    and not _is_generic_relation_norm(alias)
                    and not (len(alias.split()) == 1 and alias in RESIDUAL_STOPWORDS)
                ):
                    alias_to_titles[alias].add(title)
        return {alias: sorted(titles) for alias, titles in alias_to_titles.items()}

    def _extract_query_anchor_info(self, question: str) -> dict[str, Any]:
        q_tokens = _norm_tokens(question)
        candidates = []
        for n in range(self.query_alias_max_len, 0, -1):
            if n > len(q_tokens):
                continue
            aliases = self.query_aliases_by_len.get(n, {})
            if not aliases:
                continue
            for i in range(0, len(q_tokens) - n + 1):
                alias = " ".join(q_tokens[i : i + n])
                titles = aliases.get(alias)
                if not titles or len(titles) != 1:
                    continue
                title = titles[0]
                title_norm = _norm_phrase(title)
                relation = _is_generic_relation_norm(title_norm) or _is_generic_relation_norm(alias)
                candidates.append(
                    {
                        "alias": alias,
                        "title": title,
                        "kind": "relation" if relation else "anchor",
                        "start": i,
                        "end": i + n,
                        "length": n,
                    }
                )
        candidates.sort(key=lambda x: (int(x["length"]), len(str(x["title"]))), reverse=True)
        occupied = set()
        anchor_titles = []
        relation_titles = []
        matched = []
        for cand in candidates:
            span = set(range(int(cand["start"]), int(cand["end"])))
            if span & occupied:
                continue
            occupied.update(span)
            matched.append({"alias": cand["alias"], "title": cand["title"], "kind": cand["kind"]})
            target = relation_titles if cand["kind"] == "relation" else anchor_titles
            if _norm_phrase(cand["title"]) not in {_norm_phrase(x) for x in target}:
                target.append(cand["title"])
        generic = {_norm_phrase(x) for x in GENERIC_RELATION_TITLES}
        for phrase in sorted(generic, key=lambda x: len(x.split()), reverse=True):
            if phrase and _alias_spans(q_tokens, phrase):
                label = " ".join(w.capitalize() if len(w) > 2 else w for w in phrase.split())
                if _norm_phrase(label) not in {_norm_phrase(x) for x in relation_titles}:
                    relation_titles.append(label)
                if not any(
                    m.get("alias") == phrase and m.get("kind") == "relation" for m in matched
                ):
                    matched.append({"alias": phrase, "title": label, "kind": "relation"})
        return {
            "query_anchor_titles": anchor_titles,
            "query_relation_titles": relation_titles,
            "matched_aliases": matched,
        }

    def _add_candidate(
        self,
        merged: dict[tuple[str, str], dict[str, Any]],
        cand: dict[str, Any],
        component: str,
        score: float,
    ) -> None:
        key = (cand["unit"], cand["id"])
        self._bump("num_candidate_additions")
        if key not in merged:
            cand["score_components"] = {}
            merged[key] = cand
        else:
            self._bump("duplicate_candidate_count")
            merged[key].setdefault("raw_scores", {}).update(cand.get("raw_scores", {}))
        merged[key]["score_components"][component] = max(
            float(score), merged[key]["score_components"].get(component, float("-inf"))
        )

    def _candidate_score(
        self, cand: Mapping[str, Any], weights: tuple[float, float, float]
    ) -> float:
        comps = cand.get("score_components", {}) or {}
        key = (
            str(cand.get("unit", "")),
            str(cand.get("id", "")),
            tuple(sorted((str(k), round(float(v), 12)) for k, v in comps.items())),
        )
        cache = getattr(self, "_candidate_score_cache", None)
        if cache is not None and key in cache:
            self._bump("num_candidate_score_cache_hits")
            return cache[key]
        self._bump("num_candidate_score_computations")
        w_bm25, w_dense, w_entity = weights
        value = (
            w_bm25 * float(comps.get("bm25", 0.0))
            + w_dense * float(comps.get("dense", 0.0))
            + w_entity * float(comps.get("entity", 0.0))
        )
        if cache is not None:
            cache[key] = float(value)
        return float(value)

    def _stable_dedup_before_seed(
        self, candidates: Sequence[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        opt_cfg = self.cfg.get("optimization", {}) or {}
        enabled = bool(opt_cfg.get("stable_dedup_before_seed", False))
        self._diag_counters["stable_dedup_before_seed"] = int(enabled)
        if not enabled:
            return list(candidates)
        self._diag_counters["stable_dedup_input_count"] = len(candidates)
        out: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for cand in candidates:
            key = (str(cand.get("unit", "")), str(cand.get("id", "")))
            if key in seen:
                continue
            seen.add(key)
            out.append(cand)
        self._diag_counters["stable_dedup_output_count"] = len(out)
        return out

    def _apply_candidate_cap(
        self, candidates: Sequence[dict[str, Any]], default_total: int
    ) -> list[dict[str, Any]]:
        cap_cfg = self.cfg.get("candidate_cap", {}) or {}
        enabled = bool(cap_cfg.get("enabled", False))
        self._diag_counters["candidate_cap_enabled"] = int(enabled)
        self._diag_counters["candidate_cap_input_count"] = len(candidates)
        if not enabled:
            self._diag_counters["candidate_cap_total_candidates"] = 0
            self._diag_counters["candidate_cap_output_count"] = len(candidates)
            return list(candidates)
        total = int(cap_cfg.get("total_candidates") or default_total)
        per_type_keys = {
            "entity": cap_cfg.get("title_candidates"),
            "chunk": cap_cfg.get("chunk_candidates"),
            "proposition": cap_cfg.get("proposition_candidates"),
        }
        per_type_caps: dict[str, int] = {}
        for unit, value in per_type_keys.items():
            if value is not None:
                per_type_caps[unit] = max(0, int(value))
        counts: Counter[str] = Counter()
        capped: list[dict[str, Any]] = []
        for cand in candidates:
            unit = str(cand.get("unit", ""))
            cap = per_type_caps.get(unit)
            if cap is not None and counts[unit] >= cap:
                continue
            capped.append(cand)
            counts[unit] += 1
            if len(capped) >= total:
                break
        self._diag_counters["candidate_cap_total_candidates"] = total
        self._diag_counters["candidate_cap_output_count"] = len(capped)
        return capped

    def _retrieve_seed_candidates(
        self, question: str, metadata: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        top_k = int(self.cfg.get("candidate_top_k", 80))
        cap_cfg = self.cfg.get("candidate_cap", {}) or {}
        if bool(cap_cfg.get("enabled", False)) and cap_cfg.get("total_candidates") is not None:
            top_k = min(top_k, max(1, int(cap_cfg.get("total_candidates") or top_k)))
        bm25_top_k = int(self.cfg.get("bm25", {}).get("top_k", top_k))
        dense_top_k = int(self.cfg.get("dense", {}).get("top_k", top_k))
        if bool(cap_cfg.get("enabled", False)) and cap_cfg.get("total_candidates") is not None:
            cap_total = max(1, int(cap_cfg.get("total_candidates") or top_k))
            bm25_top_k = min(bm25_top_k, cap_total)
            dense_top_k = min(dense_top_k, cap_total)
        search_units = set(self.cfg.get("search_units", ["proposition", "chunk", "entity"]))
        if self.retrieval_variant != "full_hetero":
            search_units = {"proposition"}
        weights = self.cfg.get("weights", {})
        w_bm25, w_dense, w_entity = (
            float(weights.get("bm25", 0.35)),
            float(weights.get("dense", 0.55)),
            float(weights.get("entity", 0.10)),
        )
        merged: dict[tuple[str, str], dict[str, Any]] = {}
        query_anchor_norms = {
            _norm_phrase(t) for t in metadata.get("_query_anchor_titles", []) or []
        }

        if "proposition" in search_units:
            self._bump("num_bm25_search_calls")
            self._bump("num_proposition_search_calls")
            prop_pairs = self.prop_bm25.topk(question, bm25_top_k)
            norm = _rank_norm([("proposition", self.props[i]["prop_id"]) for i, _ in prop_pairs])
            for idx, raw in prop_pairs:
                p = self.props[idx]
                self._add_candidate(
                    merged,
                    self._prop_candidate(p, {"bm25": raw}),
                    "bm25",
                    norm[("proposition", p["prop_id"])],
                )
        if "chunk" in search_units:
            self._bump("num_bm25_search_calls")
            self._bump("num_chunk_search_calls")
            chunk_pairs = self.chunk_bm25.topk(question, max(1, bm25_top_k // 2))
            norm = _rank_norm([("chunk", self.chunks[i]["chunk_id"]) for i, _ in chunk_pairs])
            for idx, raw in chunk_pairs:
                c = self.chunks[idx]
                self._add_candidate(
                    merged,
                    {
                        "unit": "chunk",
                        "id": c["chunk_id"],
                        "title": c["title"],
                        "text": c["text"],
                        "chunk_id": c["chunk_id"],
                        "tokens": c.get("token_count", token_count(c.get("text", ""))),
                        "raw_scores": {"bm25": raw},
                    },
                    "bm25",
                    norm[("chunk", c["chunk_id"])],
                )

        if self.dense_indexes:
            self._bump("num_query_embedding_calls")
            qvec = encode_query(
                {"embedding": self.cfg.get("embedding", {})},
                question,
                logger=self.logger,
                cache=self._embed_cache,
            )
            if "proposition" in search_units and "proposition" in self.dense_indexes:
                self._bump("num_dense_search_calls")
                self._bump("num_proposition_search_calls")
                pairs = self.dense_indexes["proposition"].search(qvec, dense_top_k)
                norm = _rank_norm([("proposition", pid) for pid, _ in pairs])
                for pid, raw in pairs:
                    p = self.prop_by_id.get(pid)
                    if p:
                        self._add_candidate(
                            merged,
                            self._prop_candidate(p, {"dense": raw}),
                            "dense",
                            norm[("proposition", pid)],
                        )
            if "chunk" in search_units and "chunk" in self.dense_indexes:
                self._bump("num_dense_search_calls")
                self._bump("num_chunk_search_calls")
                pairs = self.dense_indexes["chunk"].search(qvec, max(1, dense_top_k // 2))
                norm = _rank_norm([("chunk", cid) for cid, _ in pairs])
                for cid, raw in pairs:
                    c = self.chunk_by_id.get(cid)
                    if c:
                        self._add_candidate(
                            merged,
                            {
                                "unit": "chunk",
                                "id": cid,
                                "title": c["title"],
                                "text": c["text"],
                                "chunk_id": cid,
                                "tokens": c.get("token_count", token_count(c.get("text", ""))),
                                "raw_scores": {"dense": raw},
                            },
                            "dense",
                            norm[("chunk", cid)],
                        )

        if "entity" in search_units and self.retrieval_variant == "full_hetero":
            anchors: list[str] = []
            anchors.extend(str(x) for x in metadata.get("_query_anchor_titles", []) or [])
            if bool(self.cfg.get("scan_titles_in_question", False)):
                q_l = question.lower()
                max_scan = int(self.cfg.get("title_scan_limit", 50000))
                for i, (title_l, e) in enumerate(self.entity_by_title.items()):
                    if i >= max_scan:
                        break
                    if title_l and title_l in q_l:
                        anchors.append(str(e["title"]))
            anchors = list(dict.fromkeys(anchors))
            if anchors:
                self._bump("num_title_search_calls")
            for a in anchors:
                e = self.entity_by_title.get(a.lower())
                if not e:
                    continue
                anchor_score = float(self.cfg.get("entity_anchor_score", 5.0))
                self._add_candidate(
                    merged,
                    {
                        "unit": "entity",
                        "id": e["entity_id"],
                        "title": e["title"],
                        "text": e["title"],
                        "chunk_id": None,
                        "tokens": 1,
                        "raw_scores": {"entity": anchor_score},
                    },
                    "entity",
                    anchor_score,
                )
                for pid in e.get("proposition_ids", [])[
                    : int(self.cfg.get("entity_anchor_prop_limit", 16))
                ]:
                    p = self.prop_by_id.get(pid)
                    if p:
                        self._add_candidate(
                            merged,
                            self._prop_candidate(p, {"entity": anchor_score}),
                            "entity",
                            anchor_score,
                        )

        candidates = []
        self._diag_counters["unique_candidate_count"] = len(merged)
        self._diag_counters["raw_candidate_count"] = int(
            self._diag_counters.get("num_candidate_additions", 0)
        )
        for c in merged.values():
            c["score"] = self._candidate_score(c, (w_bm25, w_dense, w_entity))
            c["original_candidate_type"] = str(
                c.get("original_candidate_type") or c.get("unit") or "fallback"
            )
            c["seed_unit_type"] = self._seed_unit_type(c)
            c["source_candidate_id"] = c.get("id")
            c["parent_title"] = c.get("parent_title") or c.get("title")
            mentioned = list(
                c.get("mentioned_titles")
                or (
                    self.prop_to_mentioned_titles.get(str(c.get("id")), [])
                    if c.get("unit") == "proposition"
                    else []
                )
                or []
            )
            c["mentioned_titles"] = mentioned
            parent_match = _norm_phrase(c.get("parent_title", "")) in query_anchor_norms
            mention_match = bool({_norm_phrase(t) for t in mentioned} & query_anchor_norms)
            c["query_anchor_parent_match"] = parent_match
            c["query_anchor_mention_match"] = mention_match
            c["anchor_relation_priority"] = 2 if parent_match else (1 if mention_match else 0)
            candidates.append(c)
        if self.retrieval_variant in {
            "prop_parent_anchor",
            "prop_parent_mention_bidirectional",
        }:

            def key(c: Mapping[str, Any]):
                if self.retrieval_variant == "prop_parent_anchor":
                    priority = 1 if c.get("query_anchor_parent_match") else 0
                else:
                    priority = int(c.get("anchor_relation_priority", 0) or 0)
                return (priority, float(c.get("score", 0.0)))

            candidates.sort(key=key, reverse=True)
        else:
            candidates.sort(key=lambda x: x["score"], reverse=True)
        candidates = self._stable_dedup_before_seed(candidates)
        if bool((self.cfg.get("candidate_cap", {}) or {}).get("enabled", False)):
            return self._apply_candidate_cap(
                candidates, max(top_k, int(self.cfg.get("min_candidate_pool", top_k)))
            )
        return candidates[: max(top_k, int(self.cfg.get("min_candidate_pool", top_k)))]

    def _ranked_seed_selection(
        self, candidates: Sequence[Mapping[str, Any]], variant: str
    ) -> list[dict[str, Any]]:
        seed_count = min(int(self.cfg.get("seed_count", 4)), len(candidates))

        def key(candidate):
            return float(candidate.get("score") or 0.0), -float(candidate.get("tokens") or 0.0)

        seeds = [dict(c) for c in sorted(candidates, key=key, reverse=True)[:seed_count]]
        for rank, seed in enumerate(seeds):
            seed["seed_rank"] = rank + 1
            seed["seed_selection_variant"] = variant
        return seeds

    def _select_seeds(self, candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        if not candidates:
            return []
        return self._ranked_seed_selection(candidates, self.seed_selection_variant)
