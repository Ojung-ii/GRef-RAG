"""CPU replay of the archived supplementary score suites."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from ..data import DATASETS, MANIFESTS
from .statistics import holm_adjust


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_suite(directory: Path, name: str, cohorts) -> dict:
    result = {}
    for row in read_csv(directory / f"{name}_query_scores.csv"):
        key = row["dataset"], row["condition"], row["method"]
        cell = result.setdefault(key, {})
        query = row["query_id"]
        if query in cell or any(not 0 <= float(row[metric]) <= 1 for metric in ("em", "f1")):
            raise ValueError(f"Invalid or duplicate score for {name}/{key}/{query}")
        cell[query] = row
    for (dataset, condition, method), scores in result.items():
        if scores.keys() != cohorts[dataset]:
            raise ValueError(f"Cohort mismatch for {name}/{dataset}/{condition}/{method}")
    return result


def cohorts() -> dict:
    return {
        dataset: set((MANIFESTS / "query_ids" / f"{dataset}.txt").read_text().splitlines())
        for dataset in (*DATASETS, "news")
    }


def mean_metric(scores, metric):
    return math.fsum(float(row[metric]) for row in scores.values()) / len(scores)


def differences(treatment, control, *, sorted_ids=False):
    ids = sorted(treatment) if sorted_ids else treatment
    return np.array([float(treatment[q]["f1"]) - float(control[q]["f1"]) for q in ids])


def resample(diffs, seed, count, batch_size):
    rng = np.random.default_rng(seed)
    draws = np.empty(count)
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        indices = rng.integers(0, len(diffs), (end - start, len(diffs)))
        draws[start:end] = diffs[indices].mean(axis=1)
    return draws


def sign_flip(diffs, seed, count, batch_size, dtype):
    rng = np.random.default_rng(seed)
    extreme = 0
    observed = abs(diffs.mean()) - 1e-15
    for start in range(0, count, batch_size):
        shape = min(batch_size, count - start), len(diffs)
        signs = 2 * rng.integers(0, 2, shape, dtype=dtype) - 1
        extreme += np.count_nonzero(np.abs((signs * diffs).mean(axis=1)) >= observed)
    return (1 + int(extreme)) / (count + 1)


def pair_summary(diffs, draws):
    low, high = np.quantile(draws, (0.025, 0.975))
    return {"delta_f1": float(diffs.mean()), "ci_low": float(low), "ci_high": float(high)}


def compare_statistics(observed, reference, *, fields=None, multiplier=1):
    keys = fields or dict.fromkeys(observed)
    for key in keys:
        other = fields[key] if fields else key
        if not math.isclose(observed[key] * multiplier, float(reference[other]), abs_tol=1e-12):
            raise ValueError(
                f"Archived statistic mismatch for {other}: {observed[key]} vs {reference[other]}"
            )


def ablation_statistics(data, references):
    families: dict[str, list] = {}
    for ref in references:
        dataset, variant, control = ref["dataset"], ref["variant"], ref["control"]
        tr = data[dataset, variant, "gref-rag"]
        co = data[dataset, control, "gref-rag"]
        diff = differences(tr, co, sorted_ids=True)
        count = int(ref["resamples"])
        summary = pair_summary(diff, resample(diff, int(ref["bootstrap_seed"]), count, 250))
        summary["p_raw"] = sign_flip(diff, int(ref["randomization_seed"]), count, 250, np.int8)
        compare_statistics(summary, ref)
        families.setdefault(ref["family"], []).append(
            dict(summary, dataset=dataset, variant=variant)
        )
    refs = {(row["dataset"], row["variant"]): row for row in references}
    for rows in families.values():
        holm_adjust(rows)
        for row in rows:
            compare_statistics(
                row, refs[row["dataset"], row["variant"]], fields={"p_holm": "p_holm"}
            )
    return {(row["dataset"], row["variant"]): row for rows in families.values() for row in rows}


def llama_statistics(data, references):
    refs = {row["dataset"]: row for row in references}
    by_dataset, result = {}, {}
    for dataset in DATASETS:
        ref = refs[dataset]
        diff = differences(data[dataset, "llama", "gref-rag"], data[dataset, "llama", "kg2rag"])
        by_dataset[dataset] = diff
        draws = resample(diff, int(ref["seed"]), int(ref["resamples"]), 500)
        result[dataset] = pair_summary(diff, draws)
        compare_statistics(result[dataset], ref)
    macro = refs["Four-dataset macro"]
    rng = np.random.default_rng(int(macro["seed"]))
    draws = [
        np.mean(
            [diff[rng.integers(0, len(diff), len(diff))].mean() for diff in by_dataset.values()]
        )
        for _ in range(int(macro["resamples"]))
    ]
    result["macro"] = pair_summary(np.array([diff.mean() for diff in by_dataset.values()]), draws)
    compare_statistics(result["macro"], macro)
    return result


def news_statistics(data, references, policy):
    results = []
    for ref in references:
        method = ref["method"]
        diff = differences(data["news", "news", "gref-rag"], data["news", "news", method])
        seed, count = policy["seed"], policy["resamples"]
        row = pair_summary(diff, resample(diff, seed, count, 500))
        compare_statistics(
            row,
            ref,
            fields={"delta_f1": "delta_F1_pp", "ci_low": "ci_low_pp", "ci_high": "ci_high_pp"},
            multiplier=100,
        )
        row.update(method=method, p_raw=sign_flip(diff, seed, count, 500, np.int64))
        compare_statistics(row, ref, fields={"p_raw": "p_raw"})
        results.append(row)
    holm_adjust(results)
    for row, ref in zip(results, references, strict=True):
        compare_statistics(row, ref, fields={"p_holm": "p_holm"})
    return {row["method"]: row for row in results}


def sensitivity_statistics(data, references, suite, base, field):
    results = []
    for ref in references:
        dataset = ref["dataset"]
        condition = f"Nc{ref[field]}" if field == "Nc" else ref[field]
        diff = differences(
            data[dataset, f"{suite}:{condition}", "gref-rag"], data[dataset, base, "gref-rag"]
        )
        seed, count = int(ref["seed"]), int(ref["resamples"])
        row = pair_summary(diff, resample(diff, seed, count, 250))
        compare_statistics(
            row,
            ref,
            fields={"delta_f1": "delta_F1_pp", "ci_low": "CI95_low_pp", "ci_high": "CI95_high_pp"},
            multiplier=100,
        )
        row.update(
            dataset=dataset, condition=condition, p_raw=sign_flip(diff, seed, count, 250, np.int64)
        )
        compare_statistics(row, ref, fields={"p_raw": "p_raw"})
        results.append(row)
    holm_adjust(results)
    for row, ref in zip(results, references, strict=True):
        compare_statistics(row, ref, fields={"p_holm": "p_holm"})
    return results
