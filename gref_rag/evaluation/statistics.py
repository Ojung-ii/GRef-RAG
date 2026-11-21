"""Replay the paper's common-prompt paired tests from the released F1 ledger."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ..data import DATASETS, MANIFESTS
from ..io_utils import atomic_output

BASELINES = ("bm25", "dense-rag", "raptor", "hipporag2", "lightrag", "kg2rag", "pathrag")
METHODS = (*BASELINES, "gref-rag")
DRAWS = 10_000
SEED = 20261029


def load_differences(ledger: Path) -> dict[str, np.ndarray]:
    scores = {(dataset, method): {} for dataset in DATASETS for method in METHODS}
    with ledger.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            cell = scores[row["dataset"], row["method"]]
            query_id = row["query_id"]
            score = float(row["f1"])
            if query_id in cell or not 0 <= score <= 1:
                raise ValueError(
                    f"Duplicate query ID or invalid F1 in {row['dataset']}/{row['method']}"
                )
            cell[query_id] = score

    differences = {}
    for dataset in DATASETS:
        gref = scores[dataset, "gref-rag"]
        query_file = MANIFESTS / "query_ids" / f"{dataset}.txt"
        expected = set(query_file.read_text().splitlines())
        if any(scores[dataset, method].keys() != expected for method in METHODS):
            raise ValueError(f"Ledger does not match the paper cohort for {dataset}")
        # Retain the recorded GRef-RAG row order for the original resampling stream.
        differences[dataset] = np.array(
            [[gref[qid] - scores[dataset, baseline][qid] for baseline in BASELINES] for qid in gref]
        )
    return differences


def holm_adjust(rows: list[dict]) -> None:
    adjusted = 0.0
    for ordinal, row in enumerate(sorted(rows, key=lambda value: value["p_raw"])):
        adjusted = max(adjusted, min(1.0, (len(rows) - ordinal) * row["p_raw"]))
        row["p_holm"] = adjusted
        row["holm_reject"] = adjusted < 0.05


def common_statistics(ledger: Path) -> list[dict]:
    differences = load_differences(ledger)
    rng = np.random.default_rng(SEED)
    bootstrap, null, observed = {}, {}, {}
    for dataset, diff in differences.items():
        n = len(diff)
        bootstrap[dataset] = np.empty((DRAWS, len(BASELINES)))
        null[dataset] = np.empty_like(bootstrap[dataset])
        observed[dataset] = diff.mean(axis=0)
        for start in range(0, DRAWS, 200):
            window = slice(start, min(start + 200, DRAWS))
            shape = (window.stop - start, n)
            samples = rng.integers(0, n, size=shape)
            bootstrap[dataset][window] = diff[samples].mean(axis=1)
            signs = 2 * rng.integers(0, 2, size=shape, dtype=np.int8) - 1
            null[dataset][window] = signs @ diff / n

    for estimates in (bootstrap, null, observed):
        estimates["macro"] = np.mean(list(estimates.values()), axis=0)

    rows = []
    for dataset, draws in bootstrap.items():
        for index, baseline in enumerate(BASELINES):
            delta = float(observed[dataset][index])
            low, high = map(float, np.percentile(draws[:, index], [2.5, 97.5]))
            at_least_as_extreme = np.count_nonzero(
                np.abs(null[dataset][:, index]) >= abs(delta) - 1e-15
            )
            rows.append(
                {
                    "dataset": dataset,
                    "baseline": baseline,
                    "delta_f1": delta,
                    "ci_low": low,
                    "ci_high": high,
                    "p_raw": (1 + int(at_least_as_extreme)) / (DRAWS + 1),
                }
            )
    holm_adjust(rows)
    return rows


def write_common_statistics(ledger: Path, output: Path) -> list[dict]:
    rows = common_statistics(ledger)
    with atomic_output(output) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows
