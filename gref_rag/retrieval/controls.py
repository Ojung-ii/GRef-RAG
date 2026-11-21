"""Local Refinement controls used by the paper ablations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Protocol,
)

if TYPE_CHECKING:
    from . import GRefRagRetriever


class RetrievalContextBuilder(Protocol):
    name: str

    def build(
        self,
        retriever: GRefRagRetriever,
        question: str,
        seeds: Sequence[Mapping[str, Any]],
        candidates: Sequence[Mapping[str, Any]],
        query_anchor_info: Mapping[str, Any],
    ) -> list[dict[str, Any]]: ...


class FullRefinementContextBuilder:
    name = "full_refinement"

    def build(
        self,
        retriever: GRefRagRetriever,
        question: str,
        seeds: Sequence[Mapping[str, Any]],
        candidates: Sequence[Mapping[str, Any]],
        query_anchor_info: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        retriever._last_context_builder_diag = {
            "context_builder": self.name,
            "local_refinement_call_count": 1,
        }
        return retriever._dispatch_full_refinement(question, seeds, candidates, query_anchor_info)


class SeedOnlyContextBuilder:
    name = "seed_only"

    def build(
        self,
        retriever: GRefRagRetriever,
        question: str,
        seeds: Sequence[Mapping[str, Any]],
        candidates: Sequence[Mapping[str, Any]],
        query_anchor_info: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        del candidates, query_anchor_info
        bundles: list[dict[str, Any]] = []
        seed_ids: list[str] = []
        seed_text_hashes: list[str] = []
        seen_payloads: set[str] = set()
        for seed in seeds:
            bundle = retriever._direct_candidate_bundle(
                question,
                seed,
                f"seed{len(bundles)}",
                refinement="phase-1 seed evidence",
            )
            if bundle is None:
                continue
            key = retriever._direct_payload_key(bundle)
            if key in seen_payloads:
                continue
            seen_payloads.add(key)
            bundles.append(bundle)
            seed_ids.append(str(bundle.get("source_candidate_id") or ""))
            seed_text_hashes.append(retriever._bundle_text_hash(bundle))
        if not bundles:
            raise RuntimeError(
                "Seed-only ablation produced an empty context; this violates the paper ablation contract."
            )
        retriever._last_context_builder_diag = {
            "context_builder": self.name,
            "without_local_refinement_seed_only": True,
            "local_refinement_call_count": 0,
            "mention_expansion_call_count": 0,
            "multi_anchor_bundling_call_count": 0,
            "semantic_bundle_selection_call_count": 0,
            "seed_ids": seed_ids,
            "seed_text_hashes": seed_text_hashes,
            "rendered_seed_count": len(bundles),
        }
        return bundles


class GloballyRankedSizeMatchedContextBuilder:
    name = "globally_ranked_size_matched"

    def build(
        self,
        retriever: GRefRagRetriever,
        question: str,
        seeds: Sequence[Mapping[str, Any]],
        candidates: Sequence[Mapping[str, Any]],
        query_anchor_info: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        del seeds, query_anchor_info
        budget = retriever._size_matched_budget_target()
        ranked_ids: list[str] = []
        bundles: list[dict[str, Any]] = []
        seen_payloads: set[str] = set()
        skipped_budget = 0
        for cand in candidates:
            ranked_ids.append(
                f"{retriever._seed_unit_type(cand)}:{cand.get('id') or cand.get('source_candidate_id') or cand.get('title') or ''}"
            )
            bundle = retriever._direct_candidate_bundle(
                question,
                cand,
                f"global{len(bundles)}",
                refinement="globally ranked direct evidence",
            )
            if bundle is None:
                continue
            key = retriever._direct_payload_key(bundle)
            if key in seen_payloads:
                continue
            if budget is not None:
                trial = bundles + [bundle]
                if retriever._rendered_context_tokens_for_bundles(trial) > budget:
                    skipped_budget += 1
                    continue
            seen_payloads.add(key)
            bundles.append(bundle)
        if not bundles and candidates:
            fallback = retriever._direct_candidate_bundle(
                question,
                candidates[0],
                "global0",
                refinement="globally ranked direct evidence fallback",
            )
            if fallback is not None:
                bundles.append(fallback)
        if not bundles:
            raise RuntimeError(
                "Size-matched ablation produced an empty context; this violates the paper ablation contract."
            )
        final_tokens = retriever._rendered_context_tokens_for_bundles(bundles)
        retriever._last_context_builder_diag = {
            "context_builder": self.name,
            "without_local_refinement_size_matched": True,
            "local_refinement_call_count": 0,
            "mention_expansion_call_count": 0,
            "multi_anchor_bundling_call_count": 0,
            "semantic_bundle_selection_call_count": 0,
            "globally_ranked_candidate_ids": ranked_ids[:200],
            "globally_ranked_candidate_count": len(candidates),
            "budget_matched_target_tokens": budget,
            "budget_matched_final_context_tokens_pre_generation": final_tokens,
            "budget_matched_skipped_over_budget": skipped_budget,
            "rendered_seed_count": len(bundles),
        }
        return bundles
