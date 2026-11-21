"""Compare fresh inference to a saved camera-ready hash and score ledger."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..io_utils import atomic_output
from .metrics import keyed_rows


def compare(predictions: Path, metrics: Path, reference: Path, output: Path) -> dict:
    live = keyed_rows(predictions)
    saved = keyed_rows(reference)
    if live.keys() != saved.keys():
        raise ValueError("Live and reference query ID sets differ")
    evaluated = json.loads(metrics.read_text())["per_query"]
    by_id = {row["query_id"]: row for row in evaluated}
    if len(by_id) != len(evaluated) or by_id.keys() != live.keys():
        raise ValueError("Metrics and prediction cohorts differ")

    difference_fields = (
        "context_sha256",
        "prompt_sha256",
        "context_tokens",
        "full_input_tokens",
        "prediction_sha256",
        "em",
        "f1",
    )
    differences = {field: [] for field in difference_fields}
    for qid, expected in saved.items():
        row = live[qid]
        values = {**row, **{key: by_id[qid][key] for key in ("em", "f1")}}
        if isinstance(row.get("prediction"), str):
            values["prediction_sha256"] = hashlib.sha256(row["prediction"].encode()).hexdigest()
        for field in difference_fields:
            target = expected.get(field)
            if target is None:
                continue
            value = values.get(field)
            if field in {"em", "f1"} and value is not None:
                equal = abs(value - target) < 1e-12
            else:
                equal = value == target
            if not equal:
                differences[field].append(qid)
    report = {
        "n": len(live),
        "first_differing_stage": next(
            (field for field in difference_fields if differences[field]), None
        ),
        "difference_counts": {field: len(qids) for field, qids in differences.items()},
        "difference_ids": {field: qids for field, qids in differences.items() if qids},
    }
    with atomic_output(output) as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report
