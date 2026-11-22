"""Verify the camera-ready numbers from repository-relative CPU evidence."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from statistics import mean

from ..data import DATASETS, MANIFESTS
from ..io_utils import atomic_output, load_yaml
from .budget_statistics import summarize_budget_scores
from .statistics import common_statistics
from .supplementary import (
    ablation_statistics,
    cohorts,
    llama_statistics,
    load_suite,
    mean_metric,
    news_statistics,
    read_csv,
    sensitivity_statistics,
)

ROOT = MANIFESTS.parent
RESULTS = ROOT / "res"
SUPPLEMENTARY = RESULTS / "supplementary"
PROVENANCE = ROOT / "reproducibility"


def common_aggregates():
    ids = cohorts()
    data = defaultdict(dict)
    for row in read_csv(RESULTS / "common_query_scores.csv"):
        scores = data[row["dataset"], row["method"]]
        qid = row["query_id"]
        if qid in scores:
            raise ValueError(f"Duplicate common score for {qid}")
        scores[qid] = row
    values = {}
    for (dataset, method), scores in data.items():
        if scores.keys() != ids[dataset]:
            raise ValueError(f"Wrong common cohort for {dataset}/{method}")
        em, f1, tokens = (mean_metric(scores, field) for field in ("em", "f1", "context_tokens"))
        values[dataset, method] = dict(
            em_pct=100 * em, f1_pct=100 * f1, context_tokens=tokens, f1_per_1k=1000 * f1 / tokens
        )
    return values


def check_run_manifest():
    rows = read_csv(PROVENANCE / "run_manifest.csv")
    ids = {row["artifact_id"] for row in rows}
    if len(ids) != len(rows):
        raise ValueError("Duplicate artifact ID in run manifest")
    config_ids = {row["artifact_id"] for row in read_csv(PROVENANCE / "config_records.csv")}
    if not config_ids <= ids:
        raise ValueError("Configuration metadata references a missing artifact")
    for row in rows:
        expected = set(row["input_artifact_ids"].split())
        if not expected <= ids or row["artifact_id"] in expected:
            raise ValueError(f"Invalid input artifact link for {row['artifact_id']}")
        if (
            row["config_path"] == "reproducibility/config_records.csv"
            and row["artifact_id"] not in config_ids
        ):
            raise ValueError(f"Missing recorded configuration for {row['artifact_id']}")
        for field in ("output_path", "metrics_path", "config_path"):
            value = row[field]
            if field == "config_path" and value == "Not retained":
                continue
            path = Path(value)
            if path.is_absolute() or ".." in path.parts or not (ROOT / path).is_file():
                raise ValueError(f"Invalid repository pointer for {row['artifact_id']}/{field}")
    return len(rows)


def native_aggregates():
    data = load_suite(SUPPLEMENTARY, "native", cohorts())
    references = read_csv(RESULTS / "native_prompt.csv")
    if {(row["dataset"], "native", row["method"]) for row in references} != data.keys():
        raise ValueError("Table 8 and native score ledger have different cells")
    for row in references:
        scores = data[row["dataset"], "native", row["method"]]
        values = {field: mean_metric(scores, field) for field in ("em", "f1")}
        tokens = [row["full_input_tokens"] for row in scores.values()]
        if all(tokens):
            values["full_input_tokens"] = mean(float(value) for value in tokens)
        for field, value in values.items():
            if not math.isclose(value, float(row[field]), abs_tol=1e-12):
                raise ValueError(
                    f"Native aggregate differs for {row['dataset']}/{row['method']}/{field}"
                )
        yield {**row, **values}


def paper_metrics():
    metrics = {}
    ids = cohorts()

    def add(reference, scope, **values):
        for field, value in values.items():
            key = reference, scope, field
            if key in metrics:
                raise ValueError(f"Duplicate result cell {key}")
            metrics[key] = float(value)

    common = common_aggregates()
    for (dataset, method), row in common.items():
        scope = f"{dataset}/{method}"
        add("Table 1", scope, **row)
        add("Figure 3", scope, f1_pct=row["f1_pct"], context_tokens=row["context_tokens"])
    sizes = {len(ids[name]) for name in DATASETS}
    if len(sizes) != 1:
        raise ValueError("The four main datasets must have equal cohort sizes")
    decoder = load_yaml(ROOT / "configs/common.yaml")["generation"]
    add(
        "Table 5",
        "common",
        temperature=decoder["temperature"],
        top_p=decoder["top_p"],
        max_output_tokens=decoder["max_new_tokens"],
        queries_per_dataset=sizes.pop(),
    )
    light_caps = {
        int(row["max_output_tokens"])
        for row in read_csv(PROVENANCE / "config_records.csv")
        if row["artifact_id"].startswith("native.") and row["artifact_id"].endswith(".lightrag")
    }
    if len(light_caps) != 1:
        raise ValueError("LightRAG native output cap differs across cohorts")
    add("Table 5", "native/lightrag", max_output_tokens=light_caps.pop())
    by_method = defaultdict(list)
    for (dataset, method), values in common.items():
        by_method[method].append(values)
    timing = defaultdict(list)
    for row in read_csv(SUPPLEMENTARY / "online_timing_components.csv"):
        timing[row["method"], row["phase"]].append(float(row["mean_ms"]))
    for method, rows in by_method.items():
        add(
            "Table 2",
            method,
            context_tokens=mean(row["context_tokens"] for row in rows),
            f1_per_1k=mean(row["f1_per_1k"] for row in rows),
            retrieval_ms=mean(timing[method, "retrieval"]),
            generation_ms=mean(timing[method, "generation"]),
        )

    common_pairs = common_statistics(RESULTS / "common_query_scores.csv")
    stored_pairs = {
        (r["dataset"], r["baseline"]): r for r in read_csv(RESULTS / "main_paired_statistics.csv")
    }
    for r in common_pairs:
        ref = stored_pairs[r["dataset"], r["baseline"]]
        for field in ("delta_f1", "ci_low", "ci_high", "p_raw", "p_holm"):
            if abs(float(ref[field]) - r[field]) > 1e-12:
                raise ValueError(
                    f"Common paired statistic differs for {r['dataset']}/{r['baseline']}/{field}"
                )
        if (ref["holm_reject"] == "True") != r["holm_reject"]:
            raise ValueError("Common Holm decision differs")
    macro = next(r for r in common_pairs if (r["dataset"], r["baseline"]) == ("macro", "kg2rag"))
    gref = by_method["gref-rag"]
    kg = by_method["kg2rag"]
    add(
        "Main Result",
        "gref-rag/kg2rag",
        gref_f1_pct=mean(r["f1_pct"] for r in gref),
        kg2rag_f1_pct=mean(r["f1_pct"] for r in kg),
        context_reduction_pct=100
        * (1 - mean(r["context_tokens"] for r in gref) / mean(r["context_tokens"] for r in kg)),
        **{f"{key}_pp": 100 * macro[key] for key in ("delta_f1", "ci_low", "ci_high")},
    )
    add(
        "Paired-Bootstrap Uncertainty",
        "common",
        comparisons=len(common_pairs),
        holm_rejections=sum(r["holm_reject"] for r in common_pairs),
    )

    budget_rows = summarize_budget_scores(RESULTS / "matched_budget_scores.csv")
    budget_refs = {
        (row["policy"], int(row["budget"]), row["dataset"]): row
        for row in read_csv(RESULTS / "matched_budget_summary.csv")
    }
    for row in budget_rows:
        ref = budget_refs[row["policy"], row["budget"], row["dataset"]]
        if any(
            not math.isclose(row[field], float(ref[field]), abs_tol=1e-12)
            for field in (
                "delta_f1",
                "ci_low",
                "ci_high",
                "gref_context_tokens",
                "kg2rag_context_tokens",
            )
        ):
            raise ValueError(
                f"Matched-budget summary differs for {row['policy']}/{row['budget']}/{row['dataset']}"
            )
        if row["dataset"] != "macro":
            continue
        add(
            "Table 4(a)",
            f"{row['policy']}/{row['budget']}",
            **{f"{key}_pp": 100 * row[key] for key in ("delta_f1", "ci_low", "ci_high")},
        )
        if row["policy"] == "packing":
            add(
                "Table 7",
                str(row["budget"]),
                gref_tokens=row["gref_context_tokens"],
                kg2rag_tokens=row["kg2rag_context_tokens"],
            )

    for row in native_aggregates():
        add(
            "Table 8",
            f"{row['dataset']}/{row['method']}",
            em_pct=100 * float(row["em"]),
            f1_pct=100 * float(row["f1"]),
            full_input_tokens=row["full_input_tokens"],
        )
    for row in read_csv(RESULTS / "offline_cost.csv"):
        add(
            "Table 11",
            row["method"],
            indexing_seconds=row["indexing_seconds"],
            storage_mib=row["storage_mib"],
            update_1pct_seconds=row["update_1pct_seconds"],
        )

    ablation = load_suite(SUPPLEMENTARY, "ablation", ids)
    ablation_refs = read_csv(SUPPLEMENTARY / "ablation_statistics_reference.csv")
    for (dataset, variant), row in ablation_statistics(ablation, ablation_refs).items():
        table = "Table 12" if variant.endswith("_view") else "Table 3"
        add(
            table,
            f"{dataset}/{variant}",
            delta_f1_pp=100 * row["delta_f1"],
            holm_reject=row["holm_reject"],
        )
    for dataset, condition, method in ablation:
        if condition.endswith("_full"):
            add(
                "Table 3",
                f"{dataset}/{condition}",
                f1_pct=100 * mean_metric(ablation[dataset, condition, method], "f1"),
            )

    llama = load_suite(SUPPLEMENTARY, "llama", ids)
    stats = llama_statistics(llama, read_csv(SUPPLEMENTARY / "llama_statistics_reference.csv"))
    for dataset in DATASETS:
        gref_pct = 100 * mean_metric(llama[dataset, "llama", "gref-rag"], "f1")
        kg_pct = 100 * mean_metric(llama[dataset, "llama", "kg2rag"], "f1")
        add(
            "Table 6",
            dataset,
            gref_f1_pct=gref_pct,
            kg2rag_f1_pct=kg_pct,
            delta_f1_pp=100 * stats[dataset]["delta_f1"],
        )
        add(
            "Table 4(b)",
            dataset,
            gref_llama_f1_pct=gref_pct,
            kg2rag_llama_f1_pct=kg_pct,
            llama_delta_f1_pp=100 * stats[dataset]["delta_f1"],
            gref_qwen_f1_pct=common[dataset, "gref-rag"]["f1_pct"],
            kg2rag_qwen_f1_pct=common[dataset, "kg2rag"]["f1_pct"],
            qwen_delta_f1_pp=(
                round(common[dataset, "gref-rag"]["f1_pct"], 1)
                - round(common[dataset, "kg2rag"]["f1_pct"], 1)
            ),
        )
    add("Second generator", "macro", **{f"{k}_pp": 100 * v for k, v in stats["macro"].items()})

    news = load_suite(SUPPLEMENTARY, "news", ids)
    policy = json.loads((PROVENANCE / "statistics_policy.json").read_text())
    stats = news_statistics(
        news, read_csv(SUPPLEMENTARY / "news_statistics_reference.csv"), policy["news"]
    )
    for (_, _, method), rows in news.items():
        f1 = mean_metric(rows, "f1")
        tokens = mean_metric(rows, "context_tokens")
        add(
            "Table 4(c)",
            method,
            f1_pct=100 * f1,
            context_tokens=tokens,
            f1_per_1k=1000 * f1 / tokens,
        )
    news_gref = news["news", "news", "gref-rag"]
    add(
        "News-domain evaluation",
        "gref-rag",
        queries=len(news_gref),
        no_usable_bridge=sum(row["usable_mention_bridge"] == "0" for row in news_gref.values()),
        no_usable_bridge_pct=100 * (1 - mean_metric(news_gref, "usable_mention_bridge")),
        rendered_bridge=sum(int(row["rendered_mention_bridge"]) for row in news_gref.values()),
        delta_f1_pp=100 * stats["kg2rag"]["delta_f1"],
        ci_low_pp=100 * stats["kg2rag"]["ci_low"],
        ci_high_pp=100 * stats["kg2rag"]["ci_high"],
        p_holm=stats["kg2rag"]["p_holm"],
    )

    robust = load_suite(SUPPLEMENTARY, "robustness", ids)
    perturbations = defaultdict(list)
    for row in read_csv(SUPPLEMENTARY / "robustness_conditions.csv"):
        suite, dataset, condition = row["suite"], row["dataset"], row["condition_id"]
        base = "delete_0_control" if suite == "missing_link" else "inject_0_control"
        f1 = mean_metric(robust[dataset, f"{suite}/{condition}", "gref-rag"], "f1")
        control = mean_metric(robust[dataset, f"{suite}/{base}", "gref-rag"], "f1")
        perturbations[suite, int(float(row["rate"]) * 100)].append(f1 - control)
    for (suite, rate), diffs in perturbations.items():
        add("Table 4(d)", f"{suite}/{rate}", delta_f1_pp=100 * mean(diffs))

    rendering = load_suite(SUPPLEMENTARY, "rendering", ids)
    for (dataset, condition, _), scores in rendering.items():
        add(
            "Table 9",
            f"{dataset}/{condition}",
            f1_pct=100 * mean_metric(scores, "f1"),
            context_tokens=mean_metric(scores, "context_tokens"),
        )

    sensitivity = load_suite(SUPPLEMENTARY, "sensitivity", ids)
    for (dataset, condition, _), scores in sensitivity.items():
        add("Table 10", f"{dataset}/{condition}", f1_pct=100 * mean_metric(scores, "f1"))
    for dataset in ("hotpotqa", "2wiki"):
        all_values = [
            row
            for (d, c, _), row in sensitivity.items()
            if d == dataset and c.startswith("SeedFusion:")
        ]
        f1 = [100 * mean_metric(row, "f1") for row in all_values]
        add("Table 10", f"{dataset}/range", min_f1_pct=min(f1), max_f1_pct=max(f1))
        fusion = [
            100 * mean_metric(row, "f1")
            for (d, c, _), row in sensitivity.items()
            if d == dataset and c.startswith("SeedFusion:") and not c.startswith("SeedFusion:ks")
        ]
        add("Seed and Fusion Sensitivity", dataset, min_f1_pct=min(fusion), max_f1_pct=max(fusion))
    seed = sensitivity_statistics(
        sensitivity,
        read_csv(SUPPLEMENTARY / "seed_fusion_statistics_reference.csv"),
        "SeedFusion",
        "SeedFusion:base",
        "condition",
    )
    cap = sensitivity_statistics(
        sensitivity,
        read_csv(SUPPLEMENTARY / "candidate_cap_statistics_reference.csv"),
        "Nc",
        "Nc:Nc80",
        "Nc",
    )
    add(
        "Seed and Fusion Sensitivity",
        "holm",
        seed_comparisons=len(seed),
        cap_comparisons=len(cap),
        rejections=sum(row["holm_reject"] for row in seed + cap),
    )
    return metrics


def verify_release(report: Path, manuscript: Path | None = None):
    snapshot = json.loads((PROVENANCE / "paper_snapshot.json").read_text())
    if manuscript and hashlib.sha256(manuscript.read_bytes()).hexdigest() != snapshot["tex_sha256"]:
        raise ValueError("The manuscript does not match the recorded camera-ready snapshot")
    manifest_entries = check_run_manifest()
    results = paper_metrics()
    cells = []
    seen = set()
    for row in read_csv(PROVENANCE / "paper_values.csv"):
        key = row["paper_reference"], row["scope"], row["metric"]
        if key in seen:
            raise ValueError(f"Duplicate manuscript value for {key}")
        seen.add(key)
        value = results[key]
        expected = Decimal(row["manuscript_value"])
        rounded = Decimal(f"{value:.{int(row['decimal_places'])}f}")
        cells.append(
            dict(row, measured=value, status="Pass" if rounded == expected else "Mismatch")
        )
    counts = defaultdict(int)
    for row in cells:
        counts[row["status"]] += 1
    result = dict(
        snapshot=snapshot,
        manifest_entries=manifest_entries,
        numeric_cells=len(cells),
        counts=dict(counts),
        cells=cells,
    )
    with atomic_output(report) as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return counts
