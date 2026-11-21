"""Semantic coverage computations for evidence bundles."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


class SemanticCoverage:
    """Semantic coverage computations for evidence bundles."""

    def _dense_prop_vector(self, prop_id: Any):
        pid = str(prop_id or "")
        if not pid:
            return None
        if "proposition" in self.dense_indexes:
            dense = self.dense_indexes["proposition"]
            pos = self.sentence_vector_positions.get(pid)
            if pos is not None and pos < dense.matrix.shape[0]:
                return np.asarray(dense.matrix[pos], dtype=np.float32).reshape(-1)
        return None

    def _dense_chunk_vector(self, chunk_id: Any):
        cid = str(chunk_id or "")
        if not cid:
            return None
        if "chunk" in self.dense_indexes:
            dense = self.dense_indexes["chunk"]
            pos = getattr(self, "chunk_vector_positions", None)
            if pos is None:
                try:
                    pos = {str(chunk_id): i for i, chunk_id in enumerate(dense.ids)}
                    self.chunk_vector_positions = pos
                except Exception:
                    pos = {}
            i = pos.get(cid)
            if i is not None and i < dense.matrix.shape[0]:
                return np.asarray(dense.matrix[i], dtype=np.float32).reshape(-1)
        return None

    def _surrogate_title_vector(self, title: Any):
        title = str(title or "")
        for prop in self.props_by_title.get(title, []) or []:
            vec = self._dense_prop_vector(prop.get("prop_id"))
            if vec is not None:
                return vec, {
                    "unit": "proposition",
                    "id": prop.get("prop_id"),
                    "title": title,
                }
        for chunk in self.chunks_by_title.get(title, []) or []:
            vec = self._dense_chunk_vector(chunk.get("chunk_id"))
            if vec is not None:
                return vec, {
                    "unit": "chunk",
                    "id": chunk.get("chunk_id"),
                    "title": title,
                }
        return None, {}

    def _candidate_semantic_vector(self, cand: Mapping[str, Any]):
        unit = str(cand.get("unit") or "")
        if unit == "proposition":
            vec = self._dense_prop_vector(cand.get("id") or cand.get("source_candidate_id"))
            if vec is not None:
                return vec, {
                    "unit": "proposition",
                    "id": cand.get("id") or cand.get("source_candidate_id"),
                }
        if unit == "chunk":
            vec = self._dense_chunk_vector(
                cand.get("id") or cand.get("chunk_id") or cand.get("source_candidate_id")
            )
            if vec is not None:
                return vec, {
                    "unit": "chunk",
                    "id": cand.get("id") or cand.get("chunk_id") or cand.get("source_candidate_id"),
                }
        if cand.get("chunk_id"):
            vec = self._dense_chunk_vector(cand.get("chunk_id"))
            if vec is not None:
                return vec, {"unit": "chunk", "id": cand.get("chunk_id")}
        vec, src = self._surrogate_title_vector(cand.get("title") or cand.get("parent_title"))
        if vec is not None:
            src = {**src, "surrogate_for_unit": unit or "unknown"}
            return vec, src
        self._bump("semantic_refinement_vector_missing_count")
        return None, {}

    def _bundle_semantic_vectors(self, bundle: Mapping[str, Any]) -> list[tuple[str, np.ndarray]]:
        out = []
        seen = set()
        for prop in bundle.get("propositions", []) or []:
            pid = str(prop.get("prop_id") or "")
            if not pid:
                continue
            key = f"proposition:{pid}"
            if key in seen:
                continue
            vec = self._dense_prop_vector(pid)
            if vec is not None:
                seen.add(key)
                out.append((key, vec))
        for chunk in bundle.get("source_chunks", []) or []:
            cid = str(chunk.get("chunk_id") or "")
            if not cid:
                continue
            key = f"chunk:{cid}"
            if key in seen:
                continue
            vec = self._dense_chunk_vector(cid)
            if vec is not None:
                seen.add(key)
                out.append((key, vec))
        return out

    def _cosine01(self, left: np.ndarray, right: np.ndarray) -> float:
        denom = max(float(np.linalg.norm(left)) * float(np.linalg.norm(right)), 1e-12)
        value = float(np.dot(left, right) / denom)
        if value <= 0.0:
            return 0.0
        if value >= 1.0:
            return 1.0
        return value

    def _semantic_demand(self, candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        demand = []
        for idx, cand in enumerate(candidates):
            vec, src = self._candidate_semantic_vector(cand)
            if vec is None:
                continue
            demand.append(
                {
                    "index": idx,
                    "key": f"{cand.get('unit', '?')}:{cand.get('id') or cand.get('source_candidate_id') or idx}",
                    "unit": cand.get("unit"),
                    "id": cand.get("id") or cand.get("source_candidate_id"),
                    "title": cand.get("title") or cand.get("parent_title"),
                    "raw_score": float(cand.get("score", 0.0) or 0.0),
                    "vector": vec,
                    "vector_source": src,
                }
            )
        self._bump("semantic_refinement_demand_count", len(demand))
        positive = sum(max(0.0, float(x["raw_score"])) for x in demand)
        if demand and positive > 0.0:
            for x in demand:
                x["weight"] = max(0.0, float(x["raw_score"])) / positive
        elif demand:
            self._bump("semantic_refinement_fallback_uniform_weights")
            weight = 1.0 / len(demand)
            for x in demand:
                x["weight"] = weight
        return demand

    def _semantic_coverage_vector(
        self, bundles: Sequence[Mapping[str, Any]], demand: Sequence[Mapping[str, Any]]
    ) -> np.ndarray:
        if not demand:
            return np.zeros((0,), dtype=np.float32)
        evidence = []
        for bundle in bundles:
            evidence.extend(self._bundle_semantic_vectors(bundle))
        if not evidence:
            return np.zeros((len(demand),), dtype=np.float32)
        out = np.zeros((len(demand),), dtype=np.float32)
        for i, d in enumerate(demand):
            dvec = np.asarray(d["vector"], dtype=np.float32)
            best = 0.0
            for _, evec in evidence:
                best = max(best, self._cosine01(dvec, evec))
            out[i] = best
        return out

    def _semantic_facility_value(
        self, bundles: Sequence[Mapping[str, Any]], demand: Sequence[Mapping[str, Any]]
    ) -> float:
        self._bump("semantic_selector_score_call_count")
        if not demand:
            return 0.0
        covered = self._semantic_coverage_vector(bundles, demand)
        return float(
            sum(float(d.get("weight", 0.0)) * float(covered[i]) for i, d in enumerate(demand))
        )

    def _semantic_marginal_gain(
        self,
        bundle: Mapping[str, Any],
        selected: Sequence[Mapping[str, Any]],
        demand: Sequence[Mapping[str, Any]],
        current_coverage: np.ndarray | None = None,
    ) -> float:
        self._bump("semantic_selector_score_call_count")
        if not demand:
            return 0.0
        base = (
            current_coverage
            if current_coverage is not None
            else self._semantic_coverage_vector(selected, demand)
        )
        bvecs = self._bundle_semantic_vectors(bundle)
        if not bvecs:
            return 0.0
        gain = 0.0
        for i, d in enumerate(demand):
            dvec = np.asarray(d["vector"], dtype=np.float32)
            best = float(base[i]) if len(base) > i else 0.0
            cand = max((self._cosine01(dvec, evec) for _, evec in bvecs), default=0.0)
            if cand > best:
                gain += float(d.get("weight", 0.0)) * (cand - best)
        return float(gain)

    def _semantic_bundle_signature(self, bundle: Mapping[str, Any]) -> str:
        keys = []
        for prop in bundle.get("propositions", []) or []:
            if prop.get("prop_id"):
                keys.append(f"p:{prop.get('prop_id')}")
        for chunk in bundle.get("source_chunks", []) or []:
            if chunk.get("chunk_id"):
                keys.append(f"c:{chunk.get('chunk_id')}")
        if not keys:
            keys = [
                str(
                    bundle.get("source_candidate_id")
                    or bundle.get("bundle_id")
                    or bundle.get("anchor_title")
                    or "empty"
                )
            ]
        return hashlib.sha256(
            json.dumps(sorted(set(keys)), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
