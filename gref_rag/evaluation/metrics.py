"""Shared paper QA metrics and protocol-aware cost aggregation."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean

from ..data import apply_official_aliases, read_benchmark
from ..io_utils import atomic_output, read_jsonl
from .recall import ADAPTERS


def answer_scores(prediction: str, references: list[str]) -> tuple[float, float]:
    """Paper EM and maximum multiset token F1 over all accepted references."""
    import re
    import unicodedata

    def normalize(text: str) -> str:
        value = unicodedata.normalize("NFKC", text).lower()
        value = re.sub(r"\b(a|an|the)\b", " ", value)
        return " ".join(re.sub(r"[^0-9a-z가-힣]+", " ", value).split())

    gold = [normalize(value) for value in references if isinstance(value, str) and value.strip()]
    if not gold:
        raise ValueError("No benchmark answer references")
    pred = normalize(prediction)
    tokens = pred.split()
    pred_counts = Counter(tokens)
    scores = []
    for answer in gold:
        gold_tokens = answer.split()
        overlap = sum((pred_counts & Counter(gold_tokens)).values())
        scores.append(
            2 * overlap / (len(tokens) + len(gold_tokens))
            if overlap
            else float(not tokens and not gold_tokens)
        )
    return float(pred in gold), max(scores)


def measure_value(row, fields, milliseconds=()):
    for key in fields + milliseconds:
        value = row.get(key)
        if value is None:
            continue
        value = float(value) / (1000 if key in milliseconds else 1)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid metric {key}={value}")
        return value
    return None


def aggregate(rows, protocol: str, index_metrics=None):
    fields = (
        "em",
        "f1",
        "recall_at_5",
        "context_tokens",
        "full_input_tokens",
        "output_tokens",
        "retrieval_seconds",
        "generation_seconds",
    )
    report = {"n": len(rows)}
    for field in fields:
        values = [row[field] for row in rows if row[field] is not None]
        report[field] = mean(values) if values else None
        report[f"{field}_n"] = len(values)
    report["paper_token_field"] = "context_tokens" if protocol == "common" else "full_input_tokens"
    tokens = report["context_tokens"]
    report["f1_per_1k_context_tokens"] = (
        1000 * report["f1"] / tokens
        if tokens and report["context_tokens_n"] == report["f1_n"] == len(rows)
        else None
    )
    report["indexing_seconds"] = (index_metrics or {}).get("indexing_seconds")
    report["indexing"] = index_metrics or None
    return report


def keyed_rows(path: Path):
    keyed = {}
    for row in read_jsonl(path):
        qid = row.get("query_id", row.get("id"))
        if qid is None or str(qid) in keyed:
            raise ValueError(f"Missing or duplicate query ID {qid!r} in {path}")
        keyed[str(qid)] = row
    return keyed


def evaluate_file(
    predictions: Path,
    output: Path,
    data_root: Path,
    dataset: str,
    *,
    protocol: str = "common",
    method: str = "gref-rag",
    gold_policy: str = "paper",
    limit=None,
    provenance=None,
    index_metrics=None,
    retrieval_only=False,
) -> dict:
    examples, _ = read_benchmark(data_root, dataset)
    examples = examples[:limit]
    if not retrieval_only:
        apply_official_aliases(examples, dataset, gold_policy, data_root)
    expected = {example.id for example in examples}
    predictions_by_id = keyed_rows(predictions)
    if predictions_by_id.keys() != expected:
        missing = expected - predictions_by_id.keys()
        extra = predictions_by_id.keys() - expected
        raise ValueError(f"Prediction cohort mismatch: {len(missing)} missing, {len(extra)} extra")
    provenance_by_id = keyed_rows(provenance) if provenance else None
    if provenance_by_id is not None and provenance_by_id.keys() != expected:
        raise ValueError("Provenance and prediction cohorts differ")
    adapter = ADAPTERS[method]
    records = []
    for example in examples:
        row = predictions_by_id[example.id]
        for key in ("generation_error", "generator_error", "retrieval_error", "error"):
            if row.get(key):
                raise ValueError(f"Experiment error for {example.id}: {row[key]}")
        source_row = row
        if provenance_by_id is not None:
            source_row = provenance_by_id[example.id]
            hash_keys = (
                "context_sha256",
                "context_hash",
                "rendered_context_hash",
                "rendered_context_sha256",
            )
            left = next((row[key] for key in hash_keys if row.get(key)), None)
            right = next((source_row[key] for key in hash_keys if source_row.get(key)), None)
            if left and right and left != right:
                raise ValueError(f"Context/provenance hash mismatch for {example.id}")
            source_row = {**row, **source_row}
        recall = adapter.score(source_row, example, dataset, 5)
        if not retrieval_only and not isinstance(row.get("prediction"), str):
            raise ValueError(f"Missing prediction text for {example.id}")
        em, f1 = (
            (None, None) if retrieval_only else answer_scores(row["prediction"], example.answers)
        )
        records.append(
            {
                "query_id": example.id,
                "em": em,
                "f1": f1,
                "recall_at_5": recall.value,
                "recall_hits": recall.hits,
                "recall_gold": recall.total,
                "context_tokens": measure_value(
                    row,
                    (
                        "context_tokens",
                        "rendered_context_tokens",
                        "retrieved_context_tokens",
                    ),
                ),
                "full_input_tokens": measure_value(
                    row,
                    (
                        "full_input_tokens",
                        "input_prompt_tokens",
                        "prompt_tokens",
                        "input_tokens",
                    ),
                ),
                "output_tokens": measure_value(row, ("output_tokens", "completion_tokens")),
                "retrieval_seconds": measure_value(
                    row,
                    ("retrieval_seconds", "retrieval_latency"),
                    ("retrieval_ms", "retrieval_total_ms"),
                ),
                "generation_seconds": measure_value(
                    row,
                    (
                        "generation_seconds",
                        "generation_latency_s",
                        "generation_latency_sec",
                    ),
                    ("generation_ms", "generation_total_ms", "generation_B_ms"),
                ),
            }
        )
    report = {
        "dataset": dataset,
        "protocol": protocol,
        "gold_policy": gold_policy,
        "recall_adapter": method,
        "recall_retrieval_unit": adapter.retrieval_unit,
        "recall_gold_unit": adapter.gold_unit,
        **aggregate(records, protocol, index_metrics),
        "per_query": records,
    }
    with atomic_output(output) as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report
