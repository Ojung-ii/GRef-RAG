"""Recompute paired F1 intervals and actual token use for matched-budget runs."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ..data import DATASETS, MANIFESTS
from ..io_utils import atomic_output
from .budgets import BUDGETS, METHODS, POLICIES

DRAWS = 10_000
SEED = 20260709


def read_scores(ledger):
    cells = {
        (policy, cap, dataset, method): {}
        for policy in POLICIES
        for cap in BUDGETS
        for dataset in DATASETS
        for method in METHODS
    }
    with ledger.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            cap = int(row["budget"])
            cell = cells[row["policy"], cap, row["dataset"], row["method"]]
            qid = row["query_id"]
            f1, tokens = float(row["f1"]), int(row["context_tokens"])
            if qid in cell or not (0 <= f1 <= 1 and 0 <= tokens <= cap):
                raise ValueError(f"Invalid or duplicate matched-budget score for {qid}")
            cell[qid] = f1, tokens
    return cells


def summarize_budget_scores(ledger: Path) -> list[dict]:
    scores = read_scores(ledger)
    cohorts = {
        dataset: (MANIFESTS / "query_ids" / f"{dataset}.txt").read_text().splitlines()
        for dataset in DATASETS
    }
    rng = np.random.default_rng(SEED)
    result = []

    def append(policy, cap, dataset, n, means, draws):
        low, high = np.percentile(draws, [2.5, 97.5])
        gref_f1, kg_f1, gref_tokens, kg_tokens = means
        result.append(
            {
                "policy": policy,
                "budget": cap,
                "dataset": dataset,
                "n_pairs": n,
                "gref_f1": float(gref_f1),
                "kg2rag_f1": float(kg_f1),
                "delta_f1": float(gref_f1 - kg_f1),
                "ci_low": float(low),
                "ci_high": float(high),
                "gref_context_tokens": float(gref_tokens),
                "kg2rag_context_tokens": float(kg_tokens),
            }
        )

    for policy in POLICIES:
        for cap in BUDGETS:
            dataset_means, dataset_draws = [], []
            for dataset, ids in cohorts.items():
                by_method = [scores[policy, cap, dataset, method] for method in METHODS]
                if any(cell.keys() != set(ids) for cell in by_method):
                    raise ValueError(f"Matched-budget cohort mismatch for {policy}/{cap}/{dataset}")
                gref, kg = (np.array([cell[qid] for qid in ids]) for cell in by_method)
                means = [gref[:, 0].mean(), kg[:, 0].mean(), gref[:, 1].mean(), kg[:, 1].mean()]
                difference = gref[:, 0] - kg[:, 0]
                draws = np.empty(DRAWS)
                for start in range(0, DRAWS, 200):
                    indices = rng.integers(0, len(ids), (200, len(ids)))
                    draws[start : start + 200] = difference[indices].mean(axis=1)
                dataset_means.append(means)
                dataset_draws.append(draws)
                append(policy, cap, dataset, len(ids), means, draws)
            append(
                policy,
                cap,
                "macro",
                sum(map(len, cohorts.values())),
                np.mean(dataset_means, axis=0),
                np.mean(dataset_draws, axis=0),
            )
    return result


def write_budget_statistics(ledger: Path, output: Path) -> list[dict]:
    rows = summarize_budget_scores(ledger)
    with atomic_output(output) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows
