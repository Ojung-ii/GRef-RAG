"""Integrated GRef-RAG retriever."""

from __future__ import annotations

import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..embedding import DenseIndex
from ..io_utils import read_jsonl
from ..text import token_count
from .controls import (
    FullRefinementContextBuilder,
    GloballyRankedSizeMatchedContextBuilder,
    RetrievalContextBuilder,
    SeedOnlyContextBuilder,
)
from .global_seed_search import GlobalSeedSearch
from .lexical import (
    BUNDLE_SELECTOR_VARIANTS,
    CONTEXT_BUILDER_STRATEGIES,
    RESIDUAL_SELECTION_VARIANTS,
    BM25Index,
    _norm_phrase,
)
from .local_refinement import LocalRefinement
from .semantic_coverage import SemanticCoverage


@dataclass
class RetrievalDiagnostics:
    candidate_count: int
    seed_count: int
    bundle_count: int
    context_tokens: int
    timings: dict[str, float]
    unit_counts: dict[str, int]
    dense_enabled: bool


class GRefRagRetriever(GlobalSeedSearch, LocalRefinement, SemanticCoverage):
    """Run both retrieval phases with a corpus-only graph and question text."""

    def __init__(
        self,
        index: Mapping[str, Any],
        cfg: Mapping[str, Any],
        dense_indexes: Mapping[str, DenseIndex] | None = None,
        logger: Any | None = None,
        *,
        context_token_counter=token_count,
    ):
        self.context_token_counter = context_token_counter
        self.index = index
        self.cfg = cfg
        self.logger = logger
        self.retrieval_variant = str(cfg.get("retrieval_variant", "full_hetero") or "full_hetero")
        if self.retrieval_variant not in {
            "full_hetero",
            "prop_text_only",
            "prop_parent_anchor",
            "prop_parent_mention_bidirectional",
        }:
            raise ValueError(f"Unsupported retrieval_variant={self.retrieval_variant!r}")
        self.seed_selection_variant = "global_seed_search"
        self.bridge_cfg = dict(cfg.get("bridge", {}) or {})
        self.refine_cfg = dict(cfg.get("refine", {}) or {})
        self.semantic_refine_cfg = dict(cfg.get("semantic_refinement", {}) or {})
        self.bundle_selector_variant = str(
            cfg.get("bundle_selector")
            or self.semantic_refine_cfg.get("bundle_selector")
            or self.semantic_refine_cfg.get("bundle_selection")
            or "semantic_marginal"
        )
        if self.bundle_selector_variant not in BUNDLE_SELECTOR_VARIANTS:
            raise ValueError(
                f"Unsupported bundle_selector={self.bundle_selector_variant!r}; "
                f"choices={sorted(BUNDLE_SELECTOR_VARIANTS)}"
            )
        self.local_refinement_strategy = "chain_expansion_and_bundle_reranking"
        self.ablation_variant = str(
            cfg.get("_ablation_variant") or cfg.get("ablation_variant") or ""
        )
        raw_context_builder = (
            cfg.get("context_builder") or self.semantic_refine_cfg.get("context_builder") or ""
        )
        if isinstance(raw_context_builder, Mapping):
            raw_context_builder = (
                raw_context_builder.get("strategy") or raw_context_builder.get("name") or ""
            )
        self.context_builder_strategy = self._resolve_context_builder_strategy(
            str(raw_context_builder or "")
        )
        self.context_builder = self._make_context_builder(self.context_builder_strategy)
        self.residual_selection_variant = str(
            cfg.get("residual_selection")
            or self.bridge_cfg.get("residual_selection")
            or self.bridge_cfg.get("selection", "residual_lexical")
            or "residual_lexical"
        )
        if self.residual_selection_variant not in RESIDUAL_SELECTION_VARIANTS:
            raise ValueError(f"Unsupported residual_selection={self.residual_selection_variant!r}")
        self.dense_indexes = dict(dense_indexes or {}) if isinstance(dense_indexes, Mapping) else {}
        self.chunks = list(index.get("chunks", []))
        self.props = list(index.get("propositions", []))
        self.entities = list(index.get("entities", []))
        self.entity_by_title = {str(e["title"]).lower(): e for e in self.entities}
        self.chunk_by_id = {c["chunk_id"]: c for c in self.chunks}
        self.prop_by_id = {p["prop_id"]: p for p in self.props}
        self.sentence_vector_positions: dict[str, int] = {}
        if "proposition" in self.dense_indexes:
            try:
                self.sentence_vector_positions = {
                    str(pid): i for i, pid in enumerate(self.dense_indexes["proposition"].ids)
                }
            except Exception:
                self.sentence_vector_positions = {}
        self.props_by_title: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        self.chunks_by_title: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for p in self.props:
            self.props_by_title[str(p.get("title", ""))].append(p)
        for c in self.chunks:
            self.chunks_by_title[str(c.get("title", ""))].append(c)
        bm25_cfg = cfg.get("bm25", {})
        self.prop_bm25 = BM25Index(
            self.props,
            k1=float(bm25_cfg.get("k1", 1.5)),
            b=float(bm25_cfg.get("b", 0.75)),
        )
        self.chunk_bm25 = BM25Index(
            self.chunks,
            k1=float(bm25_cfg.get("k1", 1.5)),
            b=float(bm25_cfg.get("b", 0.75)),
        )
        self.rng = random.Random(int(cfg.get("random_seed", 13)))
        self._embed_cache: dict[str, Any] = {}
        raw_index_dir = index.get("index_dir") or (index.get("meta", {}) or {}).get("index_dir")
        self.bridge_index_dir = Path(raw_index_dir) if raw_index_dir else None
        self.bridge_requested = bool(self.bridge_cfg.get("enabled", False))
        self.bridge_index_loaded = False
        self.bridge_index_warning: str | None = None
        self.title_aliases: dict[str, list[str]] = {}
        self.prop_to_mentioned_titles: dict[str, list[str]] = {}
        self.title_to_mentioning_props: dict[str, list[str]] = {}
        self.title_mentions: list[Mapping[str, Any]] = []
        self._load_bridge_index(index)
        self.mention_link_perturbation_manifest = self._apply_mention_link_perturbation()
        self.query_title_aliases = self._build_query_title_aliases()
        self.query_aliases_by_len: dict[int, dict[str, list[str]]] = defaultdict(dict)
        for alias, titles in self.query_title_aliases.items():
            n = len(alias.split())
            if n > 0:
                self.query_aliases_by_len[n][alias] = titles
        self.query_alias_max_len = min(max(self.query_aliases_by_len.keys(), default=0), 8)
        self.title_mentions_by_prop: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in self.title_mentions:
            if row.get("prop_id"):
                self.title_mentions_by_prop[str(row["prop_id"])].append(row)
        self.bridge_enabled = (
            self.bridge_requested
            and self.bridge_index_loaded
            and bool(self.prop_to_mentioned_titles)
        )
        if self.bridge_requested and not self.bridge_enabled:
            self._warn_bridge(
                self.bridge_index_warning
                or "Bridge retrieval disabled because the mention bridge index is empty"
            )

    def _apply_mention_link_perturbation(self) -> dict[str, Any]:
        cfg = (
            self.cfg.get("mention_link_perturbation")
            or self.cfg.get("mention_link_robustness")
            or {}
        )
        if not isinstance(cfg, Mapping) or not cfg.get("enabled", False):
            return {"enabled": False}
        mode = str(cfg.get("mode") or cfg.get("kind") or "missing")
        rate = float(cfg.get("rate", 0.0) or 0.0)
        seed = int(cfg.get("seed", self.cfg.get("random_seed", 13)) or 13)
        if mode not in {"missing", "spurious"}:
            raise ValueError(f"Unsupported mention_link_perturbation mode={mode!r}")
        if rate < 0.0 or rate > 1.0:
            raise ValueError(f"mention_link_perturbation rate must be in [0,1], got {rate}")
        rows = [dict(row) for row in self.title_mentions]
        original_rows = len(rows)
        rng = random.Random(seed)
        mask_ids: list[str] = []
        if mode == "missing":
            scored = []
            for idx, row in enumerate(rows):
                key = f"{row.get('prop_id')}|{row.get('mentioned_title')}|{idx}|{seed}"
                scored.append((hashlib.sha256(key.encode("utf-8")).hexdigest(), idx))
            scored.sort()
            remove_n = int(round(rate * len(scored)))
            remove = {idx for _hash, idx in scored[:remove_n]}
            mask_ids = [
                str(rows[idx].get("prop_id")) + "->" + str(rows[idx].get("mentioned_title"))
                for idx in sorted(remove)
            ]
            rows = [row for idx, row in enumerate(rows) if idx not in remove]
        else:
            titles = sorted(
                {str(p.get("title") or "") for p in self.props if str(p.get("title") or "")}
            )
            by_prop = defaultdict(set)
            for row in rows:
                if row.get("prop_id") and row.get("mentioned_title"):
                    by_prop[str(row["prop_id"])].add(str(row["mentioned_title"]))
            add_n = int(round(rate * max(1, len(rows))))
            prop_ids = sorted(by_prop) or sorted(
                str(p.get("prop_id")) for p in self.props if p.get("prop_id")
            )
            added = 0
            attempts = 0
            while prop_ids and titles and added < add_n and attempts < add_n * 20 + 100:
                attempts += 1
                prop_id = prop_ids[attempts % len(prop_ids)]
                title = titles[rng.randrange(len(titles))]
                if title in by_prop[prop_id]:
                    continue
                row = {
                    "prop_id": prop_id,
                    "source_title": "",
                    "mentioned_title": title,
                    "mention": title,
                    "mention_norm": _norm_phrase(title),
                    "ambiguous": False,
                    "spurious": True,
                }
                rows.append(row)
                by_prop[prop_id].add(title)
                mask_ids.append(f"{prop_id}->{title}")
                added += 1
        prop_map: dict[str, list[str]] = defaultdict(list)
        title_map: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            prop_id = str(row.get("prop_id") or "")
            title = str(row.get("mentioned_title") or "")
            if not prop_id or not title:
                continue
            if title not in prop_map[prop_id]:
                prop_map[prop_id].append(title)
            if prop_id not in title_map[title]:
                title_map[title].append(prop_id)
        self.title_mentions = rows
        self.prop_to_mentioned_titles = dict(prop_map)
        self.title_to_mentioning_props = dict(title_map)
        return {
            "enabled": True,
            "mode": mode,
            "rate": rate,
            "seed": seed,
            "original_link_count": original_rows,
            "final_link_count": len(rows),
            "masked_or_added_count": len(mask_ids),
            "mask_ids": mask_ids[:1000],
            "true_links_preserved_for_spurious": mode == "spurious",
        }

    def _resolve_context_builder_strategy(self, configured: str) -> str:
        key = str(configured or "").strip()
        aliases = {
            "": "full_refinement",
            "full": "full_refinement",
            "full_refinement": "full_refinement",
            "without_local_refinement_seed_only": "seed_only",
            "seeds_only": "seed_only",
            "seed_only": "seed_only",
            "phase_seeds_only": "seed_only",
            "without_local_refinement_size_matched": "globally_ranked_size_matched",
            "seeds_only_budget_matched": "globally_ranked_size_matched",
            "size_matched": "globally_ranked_size_matched",
            "phase_seeds_only_budget_matched": "globally_ranked_size_matched",
        }
        if not key:
            return aliases.get(str(self.ablation_variant or "").strip(), "full_refinement")
        strategy = aliases.get(key, key)
        if strategy not in CONTEXT_BUILDER_STRATEGIES:
            raise ValueError(
                f"Unsupported context_builder strategy={strategy!r}; choices={sorted(CONTEXT_BUILDER_STRATEGIES)}"
            )
        return strategy

    def _make_context_builder(self, strategy: str) -> RetrievalContextBuilder:
        if strategy == "seed_only":
            return SeedOnlyContextBuilder()
        if strategy == "globally_ranked_size_matched":
            return GloballyRankedSizeMatchedContextBuilder()
        return FullRefinementContextBuilder()

    def retrieve(
        self,
        question: str,
        *,
        dataset: str | None = None,
        query_id: str | None = None,
        timing_recorder: Any = None,
    ) -> dict[str, Any]:
        timings: dict[str, float] = {}
        self._active_stage_durations: dict[str, float] = defaultdict(float)
        self._active_stage_counts: dict[str, int] = defaultdict(int)
        self._diag_counters: Counter[str] = Counter()
        self._pairwise_similarity_cache: dict[tuple[str, str], float] = {}
        self._bridge_title_lookup_cache: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
        self._bridge_prop_score_cache: dict[tuple[str, tuple[str, ...]], float] = {}
        self._bridge_rank_cache: dict[
            tuple[str, str, tuple[str, ...], tuple[str, ...]],
            list[tuple[Mapping[str, Any], dict[str, Any]]],
        ] = {}
        self._candidate_score_cache: dict[
            tuple[str, str, tuple[tuple[str, float], ...]], float
        ] = {}
        self._last_bundle_candidate_fingerprints: list[str] = []
        self._residual_dense_query_vector = None
        self._residual_dense_query_text = ""
        self._residual_dense_score_cache: dict[tuple[str, str], float] = {}
        self._seen_bridge_titles: set[str] = set()
        self._current_dataset = str(dataset or "")
        self._last_context_builder_diag: dict[str, Any] = {
            "context_builder": self.context_builder_strategy
        }

        def record_stage(
            stage: str,
            start: float,
            end: float,
            num_items_in: int | None = None,
            num_items_out: int | None = None,
            extra: Mapping[str, Any] | None = None,
        ) -> None:
            if timing_recorder is not None:
                timing_recorder.record(
                    dataset=str(dataset or ""),
                    query_id=str(query_id) if query_id is not None else None,
                    stage=stage,
                    start_ts=start,
                    end_ts=end,
                    num_items_in=num_items_in,
                    num_items_out=num_items_out,
                    extra=extra,
                )

        t_all = time.perf_counter()
        wall0 = time.time()
        t0 = time.perf_counter()
        query_anchor_info = self._extract_query_anchor_info(question)
        elapsed = time.perf_counter() - t0
        timings["query_preprocess_s"] = elapsed
        record_stage(
            "query_preprocess",
            wall0,
            wall0 + elapsed,
            1,
            len(query_anchor_info.get("query_anchor_titles", []) or []),
            {"retrieval_variant": self.retrieval_variant},
        )
        metadata = {
            "_query_anchor_titles": list(query_anchor_info.get("query_anchor_titles", []) or []),
            "_query_relation_titles": list(
                query_anchor_info.get("query_relation_titles", []) or []
            ),
        }
        wall0 = time.time()
        t0 = time.perf_counter()
        candidates = self._retrieve_seed_candidates(question, metadata)
        elapsed = time.perf_counter() - t0
        timings["candidate_retrieval_s"] = elapsed
        record_stage(
            "candidate_retrieval",
            wall0,
            wall0 + elapsed,
            1,
            len(candidates),
            {
                "retrieval_variant": self.retrieval_variant,
                **self._counter_subset(
                    "num_query_embedding_calls",
                    "num_dense_search_calls",
                    "num_bm25_search_calls",
                    "num_title_search_calls",
                    "num_chunk_search_calls",
                    "num_proposition_search_calls",
                    "num_candidate_score_computations",
                    "num_candidate_score_cache_hits",
                    "raw_candidate_count",
                    "unique_candidate_count",
                    "duplicate_candidate_count",
                ),
            },
        )
        wall0 = time.time()
        t0 = time.perf_counter()
        seeds = self._select_seeds(candidates)
        elapsed = time.perf_counter() - t0
        timings["seed_selection_s"] = elapsed
        timings[f"{self.seed_selection_variant}_s"] = elapsed
        record_stage(
            "seed_selection",
            wall0,
            wall0 + elapsed,
            len(candidates),
            len(seeds),
            {
                "retrieval_variant": self.retrieval_variant,
                "seed_selection_variant": self.seed_selection_variant,
                **self._counter_subset(
                    "num_pairwise_similarity_computations",
                    "num_pairwise_similarity_cache_hits",
                    "pairwise_matrix_size",
                ),
            },
        )
        wall0 = time.time()
        t0 = time.perf_counter()
        bundles = self.context_builder.build(self, question, seeds, candidates, query_anchor_info)
        elapsed = time.perf_counter() - t0
        timings["local_refinement_s"] = elapsed
        bridge_time = self._active_stage_durations.get("mention_bridge_expansion", 0.0)
        residual_time = self._active_stage_durations.get("residual_bridge_selection", 0.0)
        order_time = self._active_stage_durations.get("anchor_chain_ordering", 0.0)
        same_title = max(0.0, elapsed - bridge_time - residual_time - order_time)
        timings["same_title_refinement_s"] = same_title
        timings["mention_bridge_expansion_s"] = bridge_time
        timings["residual_bridge_selection_s"] = residual_time
        timings["anchor_chain_ordering_s"] = order_time
        record_stage(
            "same_title_refinement",
            wall0,
            wall0 + same_title,
            len(seeds),
            len(bundles),
            {"retrieval_variant": self.retrieval_variant},
        )
        record_stage(
            "mention_bridge_expansion",
            wall0 + same_title,
            wall0 + same_title + bridge_time,
            len(seeds),
            sum(1 for b in bundles if b.get("has_bridge")),
            {"retrieval_variant": self.retrieval_variant},
        )
        record_stage(
            "residual_bridge_selection",
            wall0 + same_title + bridge_time,
            wall0 + same_title + bridge_time + residual_time,
            sum(1 for b in bundles if b.get("has_bridge")),
            sum(1 for b in bundles if b.get("answer_slot_aligned")),
            {"retrieval_variant": self.retrieval_variant},
        )
        record_stage(
            "anchor_chain_ordering",
            wall0 + same_title + bridge_time + residual_time,
            wall0 + same_title + bridge_time + residual_time + order_time,
            len(bundles),
            len(bundles),
            {"retrieval_variant": self.retrieval_variant},
        )
        t0 = time.perf_counter()
        bundles, context_tokens = self._budget_context(bundles)
        timings["budgeting_s"] = time.perf_counter() - t0
        timings["total_retrieval_s"] = time.perf_counter() - t_all
        diag = RetrievalDiagnostics(
            candidate_count=len(candidates),
            seed_count=len(seeds),
            bundle_count=len(bundles),
            context_tokens=context_tokens,
            timings={k: round(v, 6) for k, v in timings.items()},
            unit_counts={
                "propositions": sum(1 for c in candidates if c["unit"] == "proposition"),
                "chunks": sum(1 for c in candidates if c["unit"] == "chunk"),
                "entities": sum(1 for c in candidates if c["unit"] == "entity"),
            },
            dense_enabled=bool(self.dense_indexes),
        )
        seed_type_distribution = dict(
            Counter(str(s.get("seed_unit_type") or self._seed_unit_type(s)) for s in seeds)
        )
        selected_bundle_source_type_distribution = dict(
            Counter(
                str(b.get("selected_bundle_source_type") or b.get("seed_unit_type") or "fallback")
                for b in bundles
            )
        )
        chain_success_by_seed_type = dict(
            Counter(
                str(b.get("selected_bundle_source_type") or "fallback")
                for b in bundles
                if b.get("chain_complete_v2")
            )
        )
        anchor_connected_chain_by_seed_type = dict(
            Counter(
                str(b.get("selected_bundle_source_type") or "fallback")
                for b in bundles
                if b.get("anchor_connected_chain_complete")
            )
        )
        answer_slot_aligned_by_seed_type = dict(
            Counter(
                str(b.get("selected_bundle_source_type") or "fallback")
                for b in bundles
                if b.get("answer_slot_aligned")
            )
        )
        selected_seed_ids = [
            f"{self._seed_unit_type(s)}:{s.get('id') or s.get('source_candidate_id') or s.get('title') or ''}"
            for s in seeds
        ]
        selected_seed_hash = hashlib.sha256(
            json.dumps(selected_seed_ids, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        bridge_titles = {t for b in bundles for t in b.get("bridge_titles", []) or []}
        bridge_connected_count = sum(1 for b in bundles if b.get("bridge_connected"))
        answer_slot_aligned_count = sum(1 for b in bundles if b.get("answer_slot_aligned"))
        chain_complete_v2_count = sum(1 for b in bundles if b.get("chain_complete_v2"))
        anchor_connected_chain_complete_count = sum(
            1 for b in bundles if b.get("anchor_connected_chain_complete")
        )
        anchor_mismatch_chain_count = sum(1 for b in bundles if b.get("anchor_mismatch_chain"))
        multi_anchor_bundle_count = sum(
            1 for b in bundles if b.get("bundle_type") == "multi_anchor"
        )
        query_anchor_norms = {
            _norm_phrase(t) for t in query_anchor_info.get("query_anchor_titles", []) or []
        }
        covered_anchor_norms = {
            norm
            for b in bundles
            for norm in b.get("covered_query_anchor_norms", []) or []
            if norm in query_anchor_norms
        }
        query_anchor_coverage = (
            len(covered_anchor_norms) / max(1, len(query_anchor_norms))
            if query_anchor_norms
            else 0.0
        )
        residual_counts = [float(b.get("residual_coverage_count", 0.0) or 0.0) for b in bundles]
        top1 = bundles[0] if bundles else {}
        diag_dict = diag.__dict__
        diag_dict.update(
            {
                "bridge_enabled": self.bridge_enabled,
                "bridge_index_loaded": self.bridge_index_loaded,
                "ordering_mode": self.bridge_cfg.get("ordering", "chain_aware")
                if self.bridge_enabled
                else "score",
                "query_anchor_titles": query_anchor_info.get("query_anchor_titles", []),
                "query_relation_titles": query_anchor_info.get("query_relation_titles", []),
                "matched_aliases": query_anchor_info.get("matched_aliases", []),
                "retrieval_input_policy": "question_only",
                "benchmark_row_metadata_used_for_retrieval": False,
                "benchmark_row_metadata_used_for_generation": False,
                "bridge_title_count": len(bridge_titles),
                "bridge_bundle_count": sum(1 for b in bundles if b.get("has_bridge")),
                "chain_complete_count": sum(1 for b in bundles if b.get("chain_complete")),
                "has_chain_complete": any(bool(b.get("chain_complete")) for b in bundles),
                "bridge_connected_count": bridge_connected_count,
                "answer_slot_aligned_count": answer_slot_aligned_count,
                "chain_complete_v2_count": chain_complete_v2_count,
                "chain_complete_v2_rate": chain_complete_v2_count / max(1, len(bundles)),
                "avg_residual_coverage_count": sum(residual_counts) / max(1, len(residual_counts)),
                "has_bridge_connected": bridge_connected_count > 0,
                "has_answer_slot_aligned": answer_slot_aligned_count > 0,
                "has_chain_complete_v2": chain_complete_v2_count > 0,
                "anchor_connected_chain_complete_count": anchor_connected_chain_complete_count,
                "anchor_mismatch_chain_count": anchor_mismatch_chain_count,
                "multi_anchor_bundle_count": multi_anchor_bundle_count,
                "has_anchor_connected_chain_complete": anchor_connected_chain_complete_count > 0,
                "has_anchor_mismatch_chain": anchor_mismatch_chain_count > 0,
                "has_multi_anchor_bundle": multi_anchor_bundle_count > 0,
                "generic_relation_top1": bool(top1.get("is_relation_title_bundle")),
                "query_anchor_coverage": query_anchor_coverage,
                "retrieval_variant": self.retrieval_variant,
                "seed_selection_variant": self.seed_selection_variant,
                "local_refinement_strategy": self.local_refinement_strategy,
                "bundle_selector_variant": self.bundle_selector_variant,
                "residual_selection_variant": self.residual_selection_variant,
                "context_builder": self.context_builder_strategy,
                "mention_link_perturbation": dict(self.mention_link_perturbation_manifest),
                "anchor_ordering_enabled": str(self.bridge_cfg.get("ordering", "chain_aware_v2"))
                == "anchor_chain_aware",
                "multi_anchor_enabled": bool(self.bridge_cfg.get("multi_anchor_bundle", True)),
                "seed_selection_ms": round(
                    float(timings.get("seed_selection_s", 0.0) or 0.0) * 1000.0, 6
                ),
                "selected_seed_count": len(seeds),
                "selected_seed_ids": selected_seed_ids,
                "selected_seed_hash": selected_seed_hash,
                "seed_unit_type_distribution": seed_type_distribution,
                "selected_bundle_source_type_distribution": selected_bundle_source_type_distribution,
                "chain_success_by_seed_type": chain_success_by_seed_type,
                "anchor_connected_chain_by_seed_type": anchor_connected_chain_by_seed_type,
                "answer_slot_aligned_by_seed_type": answer_slot_aligned_by_seed_type,
                **self._duplicate_diagnostics(candidates),
            }
        )
        first_bridge_path = next(
            (
                p
                for b in bundles
                for p in b.get("evidence_path", []) or []
                if isinstance(p, Mapping)
                and p.get("path_type") == "mention_bridge"
                and p.get("bridge_prop_id")
            ),
            {},
        )
        diag_dict.update(
            {
                "residual_query_text": " ".join(
                    next(
                        (b.get("residual_terms", []) for b in bundles if b.get("residual_terms")),
                        [],
                    )
                    or []
                ),
                "residual_query_terms": next(
                    (b.get("residual_terms", []) for b in bundles if b.get("residual_terms")),
                    [],
                ),
                "residual_dense_used": bool(self._diag_counters.get("residual_dense_used", 0)),
                "residual_dense_used_rate": 1.0
                if self._diag_counters.get("residual_dense_used", 0)
                else 0.0,
                "residual_dense_embedding_calls": int(
                    self._diag_counters.get("residual_dense_embedding_calls", 0)
                ),
                "residual_dense_similarity_computations": int(
                    self._diag_counters.get("residual_dense_similarity_computations", 0)
                ),
                "residual_dense_similarity_cache_hits": int(
                    self._diag_counters.get("residual_dense_similarity_cache_hits", 0)
                ),
                "bridge_prop_candidate_count": int(
                    self._diag_counters.get("bridge_prop_candidate_count", 0)
                ),
                "bridge_prop_selection_ms": round(
                    float(self._active_stage_durations.get("residual_bridge_selection", 0.0) or 0.0)
                    * 1000.0,
                    6,
                ),
                "semantic_refinement_demand_count": int(
                    self._diag_counters.get("semantic_refinement_demand_count", 0)
                ),
                "semantic_refinement_candidate_bundle_count": int(
                    self._diag_counters.get("semantic_refinement_candidate_bundle_count", 0)
                ),
                "semantic_refinement_vector_missing_count": int(
                    self._diag_counters.get("semantic_refinement_vector_missing_count", 0)
                ),
                "semantic_refinement_fallback_uniform_weights": bool(
                    self._diag_counters.get("semantic_refinement_fallback_uniform_weights", 0)
                ),
                "semantic_refinement_selected_count": int(
                    self._diag_counters.get("semantic_refinement_selected_count", 0)
                ),
                "semantic_refinement_token_count_source": getattr(
                    self, "_semantic_token_count_source", ""
                ),
                "bundle_candidate_set_count": len(self._bundle_candidate_set_items()),
                "bundle_candidate_set_hash": self._bundle_candidate_set_hash(),
                "bundle_candidate_ids": self._bundle_candidate_set_items()
                if bool(self.semantic_refine_cfg.get("save_bundle_candidate_ids", False))
                else [],
                "semantic_selector_score_call_count": int(
                    self._diag_counters.get("semantic_selector_score_call_count", 0)
                ),
                "lexical_selector_score_call_count": int(
                    self._diag_counters.get("lexical_selector_score_call_count", 0)
                ),
                "selected_bridge_prop_id": first_bridge_path.get("bridge_prop_id"),
                "selected_bridge_prop_title": first_bridge_path.get("bridge_title"),
                "selected_bridge_prop_text": first_bridge_path.get("bridge_prop"),
                "selected_bridge_prop_rank": first_bridge_path.get("selected_bridge_prop_rank"),
            }
        )
        diag_dict.update(dict(getattr(self, "_last_context_builder_diag", {}) or {}))
        return {
            "question": question,
            "candidates": candidates,
            "seeds": seeds,
            "evidence_bundles": bundles,
            "diagnostics": diag.__dict__,
        }

    def _bump(self, key: str, amount: int = 1) -> None:
        counters = getattr(self, "_diag_counters", None)
        if counters is not None:
            counters[key] += int(amount)

    def _remember_bundle_candidate(self, kind: str, **parts: Any) -> None:
        fingerprints = getattr(self, "_last_bundle_candidate_fingerprints", None)
        if fingerprints is None:
            return
        normalized = {
            str(k): str(v) for k, v in sorted(parts.items()) if v is not None and str(v) != ""
        }
        item = str(kind)
        if normalized:
            item += "|" + "|".join(f"{k}={v}" for k, v in normalized.items())
        fingerprints.append(item)

    def _bundle_candidate_set_items(self) -> list[str]:
        items = getattr(self, "_last_bundle_candidate_fingerprints", []) or []
        return list(dict.fromkeys(str(x) for x in items))

    def _bundle_candidate_set_hash(self) -> str:
        items = self._bundle_candidate_set_items()
        return hashlib.sha256(json.dumps(items, ensure_ascii=False).encode("utf-8")).hexdigest()

    def _counter_subset(self, *keys: str) -> dict[str, int]:
        counters = getattr(self, "_diag_counters", Counter())
        return {key: int(counters.get(key, 0)) for key in keys}

    def _duplicate_diagnostics(self, candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        counters = getattr(self, "_diag_counters", Counter())
        attempted = int(counters.get("num_candidate_additions", 0))
        unique = int(counters.get("unique_candidate_count", len(candidates)))
        duplicate = int(counters.get("duplicate_candidate_count", max(0, attempted - unique)))
        pairwise_cache = getattr(self, "_pairwise_similarity_cache", {})
        self._diag_counters["raw_candidate_count"] = attempted
        self._diag_counters["pairwise_matrix_size"] = len(pairwise_cache)
        return {
            "num_query_embedding_calls": int(counters.get("num_query_embedding_calls", 0)),
            "num_dense_search_calls": int(counters.get("num_dense_search_calls", 0)),
            "num_bm25_search_calls": int(counters.get("num_bm25_search_calls", 0)),
            "num_title_search_calls": int(counters.get("num_title_search_calls", 0)),
            "num_chunk_search_calls": int(counters.get("num_chunk_search_calls", 0)),
            "num_proposition_search_calls": int(counters.get("num_proposition_search_calls", 0)),
            "num_candidate_score_computations": int(
                counters.get("num_candidate_score_computations", 0)
            ),
            "num_candidate_score_cache_hits": int(
                counters.get("num_candidate_score_cache_hits", 0)
            ),
            "num_bridge_title_lookups": int(counters.get("num_bridge_title_lookups", 0)),
            "num_bridge_title_cache_hits": int(counters.get("num_bridge_title_cache_hits", 0)),
            "num_bridge_prop_score_computations": int(
                counters.get("num_bridge_prop_score_computations", 0)
            ),
            "num_bridge_prop_score_cache_hits": int(
                counters.get("num_bridge_prop_score_cache_hits", 0)
            ),
            "unique_bridge_title_count": int(counters.get("unique_bridge_title_count", 0)),
            "duplicate_bridge_title_count": int(counters.get("duplicate_bridge_title_count", 0)),
            "num_pairwise_similarity_computations": int(
                counters.get("num_pairwise_similarity_computations", 0)
            ),
            "num_pairwise_similarity_cache_hits": int(
                counters.get("num_pairwise_similarity_cache_hits", 0)
            ),
            "pairwise_matrix_size": len(pairwise_cache),
            "candidate_count_by_type": dict(
                Counter(str(c.get("unit", "unknown")) for c in candidates)
            ),
            "raw_candidate_count": attempted,
            "unique_candidate_count": unique,
            "duplicate_candidate_count": duplicate,
            "candidate_merge_reduction_rate": duplicate / max(1, attempted),
            "candidate_cap_enabled": bool(counters.get("candidate_cap_enabled", 0)),
            "candidate_cap_total_candidates": int(
                counters.get("candidate_cap_total_candidates", 0)
            ),
            "candidate_cap_input_count": int(
                counters.get("candidate_cap_input_count", len(candidates))
            ),
            "candidate_cap_output_count": int(
                counters.get("candidate_cap_output_count", len(candidates))
            ),
            "stable_dedup_before_seed": bool(counters.get("stable_dedup_before_seed", 0)),
            "stable_dedup_input_count": int(
                counters.get("stable_dedup_input_count", len(candidates))
            ),
            "stable_dedup_output_count": int(
                counters.get("stable_dedup_output_count", len(candidates))
            ),
        }

    def _seed_unit_type(self, cand: Mapping[str, Any]) -> str:
        unit = str(cand.get("unit") or cand.get("seed_unit_type") or "")
        if unit == "entity":
            return "title"
        if unit in {"chunk", "proposition", "fallback", "multi_anchor"}:
            return unit
        if cand.get("score_components") and set(cand.get("score_components", {})) == {"dense"}:
            return "dense_candidate"
        return unit or "fallback"

    def _prop_candidate(
        self, prop: Mapping[str, Any], raw_scores: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        pid = str(prop.get("prop_id", ""))
        mentioned = list(self.prop_to_mentioned_titles.get(pid, []) or [])
        return {
            "unit": "proposition",
            "id": pid,
            "title": prop.get("title"),
            "text": prop.get("text"),
            "chunk_id": prop.get("chunk_id"),
            "tokens": prop.get("token_count", token_count(prop.get("text", ""))),
            "raw_scores": dict(raw_scores or {}),
            "original_candidate_type": "proposition",
            "seed_unit_type": "proposition",
            "parent_title": prop.get("title"),
            "mentioned_titles": mentioned,
        }

    def _warn_bridge(self, message: str) -> None:
        self.bridge_index_warning = message
        if self.logger:
            self.logger.log(f"WARNING: {message}")
            if hasattr(self.logger, "event"):
                self.logger.event({"event": "bridge.index.warning", "message": message})

    def _load_bridge_index(self, index: Mapping[str, Any]) -> None:
        self.title_aliases = {
            str(k): list(v) for k, v in dict(index.get("title_aliases", {}) or {}).items()
        }
        self.prop_to_mentioned_titles = dict(index.get("prop_to_mentioned_titles", {}) or {})
        self.title_to_mentioning_props = dict(index.get("title_to_mentioning_props", {}) or {})
        self.title_mentions = list(index.get("title_mentions", []) or [])
        in_memory_loaded = bool(
            self.title_aliases
            or self.prop_to_mentioned_titles
            or self.title_to_mentioning_props
            or self.title_mentions
        )
        if self.bridge_index_dir is not None:
            paths = {
                "title_aliases": self.bridge_index_dir / "title_aliases.json",
                "prop_to_mentioned_titles": self.bridge_index_dir / "prop_to_mentioned_titles.json",
                "title_to_mentioning_props": self.bridge_index_dir
                / "title_to_mentioning_props.json",
                "title_mentions": self.bridge_index_dir / "title_mentions.jsonl",
            }
            missing = [str(path) for path in paths.values() if not path.exists()]
            if not missing:
                self.title_aliases = json.loads(paths["title_aliases"].read_text(encoding="utf-8"))
                self.prop_to_mentioned_titles = json.loads(
                    paths["prop_to_mentioned_titles"].read_text(encoding="utf-8")
                )
                self.title_to_mentioning_props = json.loads(
                    paths["title_to_mentioning_props"].read_text(encoding="utf-8")
                )
                self.title_mentions = read_jsonl(paths["title_mentions"])
                self.bridge_index_loaded = True
            elif self.bridge_requested:
                self.bridge_index_loaded = in_memory_loaded
                self.bridge_index_warning = f"Bridge index files missing: {missing}"
        else:
            self.bridge_index_loaded = in_memory_loaded
            if self.bridge_requested and not in_memory_loaded:
                self.bridge_index_warning = (
                    "Bridge index directory is unknown and no in-memory bridge index was provided"
                )
        if self.bridge_index_loaded:
            prop_ids = set(self.prop_by_id)
            invalid = [pid for pid in self.prop_to_mentioned_titles if pid not in prop_ids]
            if invalid:
                self.bridge_index_loaded = False
                self.bridge_index_warning = f"Bridge prop_id mismatch: first_invalid={invalid[:5]}"

    def _dispatch_full_refinement(
        self,
        question: str,
        seeds: Sequence[Mapping[str, Any]],
        candidates: Sequence[Mapping[str, Any]],
        query_anchor_info: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        return self._expand_and_rerank_bundles(question, seeds, candidates, query_anchor_info)

    def _direct_candidate_bundle(
        self, question: str, cand: Mapping[str, Any], bundle_id: str, refinement: str
    ) -> dict[str, Any] | None:
        title = str(cand.get("title") or cand.get("parent_title") or "")
        text = str(cand.get("text") or title or "")
        if not title and not text:
            return None
        source_id = str(cand.get("id") or cand.get("source_candidate_id") or title)
        seed_type = self._seed_unit_type(cand)
        prop = self.prop_by_id.get(source_id) if seed_type == "proposition" else None
        chunk = self.chunk_by_id.get(source_id) if seed_type == "chunk" else None
        propositions: list[dict[str, Any]] = []
        source_chunks: list[dict[str, Any]] = []
        if prop:
            propositions.append(dict(prop))
        elif seed_type != "chunk":
            propositions.append(
                {
                    "prop_id": source_id,
                    "title": title,
                    "text": text,
                    "token_count": token_count(text),
                }
            )
        if chunk:
            source_chunks.append(dict(chunk))
        elif seed_type == "chunk":
            source_chunks.append(
                {
                    "chunk_id": source_id,
                    "title": title,
                    "text": text,
                    "token_count": token_count(text),
                }
            )
        if not propositions and not source_chunks:
            return None
        return {
            "bundle_id": bundle_id,
            "bundle_type": "direct_evidence",
            "anchor_title": title,
            "seed": dict(cand),
            "seed_unit_type": seed_type,
            "original_candidate_type": str(
                cand.get("original_candidate_type") or cand.get("unit") or seed_type
            ),
            "selected_bundle_source_type": seed_type,
            "source_candidate_id": source_id,
            "parent_title": str(cand.get("parent_title") or title),
            "mentioned_titles": list(cand.get("mentioned_titles") or []),
            "propositions": propositions,
            "source_chunks": source_chunks,
            "evidence_path": [
                {"type": "query", "id": "q", "text": question},
                {
                    "type": "direct_candidate",
                    "id": source_id,
                    "seed_unit_type": seed_type,
                    "text": text,
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
            "ordering_group": "direct_evidence",
            "diagnostics": {
                "refinement": refinement,
                "refinement_enabled": False,
                "seed_unit_type": seed_type,
            },
        }

    def _direct_payload_key(self, bundle: Mapping[str, Any]) -> str:
        texts: list[str] = []
        for prop in bundle.get("propositions", []) or []:
            texts.append(f"{prop.get('title')}::{prop.get('text')}")
        for chunk in bundle.get("source_chunks", []) or []:
            texts.append(f"{chunk.get('title')}::{chunk.get('text')}")
        return " ".join(" ".join(str(x).split()) for x in texts).lower()

    def _bundle_text_hash(self, bundle: Mapping[str, Any]) -> str:
        return hashlib.sha256(self._direct_payload_key(bundle).encode("utf-8")).hexdigest()

    def _size_matched_budget_target(self) -> int | None:
        cfg = (
            self.semantic_refine_cfg.get("size_matched_budget")
            or self.cfg.get("size_matched_budget")
            or {}
        )
        if isinstance(cfg, Mapping):
            per_dataset = (
                cfg.get("per_dataset_context_tokens") or cfg.get("recovered_budget_caps") or {}
            )
            dataset = str(getattr(self, "_current_dataset", "") or "")
            aliases = [dataset]
            if dataset == "2wikimultihopqa":
                aliases.append("2wiki")
            if dataset == "2wiki":
                aliases.append("2wikimultihopqa")
            for key in aliases:
                if isinstance(per_dataset, Mapping) and key in per_dataset:
                    return int(per_dataset[key])
            if cfg.get("target_context_tokens") is not None:
                return int(cfg["target_context_tokens"])
        return None

    def _budget_context(
        self, bundles: Sequence[Mapping[str, Any]]
    ) -> tuple[list[dict[str, Any]], int]:
        budget = int(
            self.cfg.get(
                "context_token_budget",
                self.cfg.get("refine", {}).get("context_token_budget", 3000),
            )
        )
        out: list[dict[str, Any]] = []
        used = 0
        for b in bundles:
            b2 = dict(b)
            props = [dict(p) for p in b.get("propositions", [])]
            chunks = [dict(c) for c in b.get("source_chunks", [])]
            cost = sum(token_count(c.get("text", "")) for c in chunks) or sum(
                token_count(p.get("text", "")) for p in props
            )
            if out and used + cost > budget:
                continue
            b2["context_tokens"] = cost
            out.append(b2)
            used += cost
            if used >= budget:
                break
        return out, used
