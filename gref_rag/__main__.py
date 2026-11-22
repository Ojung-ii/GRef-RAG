"""Index, run and evaluate the camera-ready GRef-RAG benchmarks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .data import DATASETS, data_paths, download_inputs, file_sha256, read_benchmark
from .evaluation.metrics import evaluate_file
from .evaluation.recall import ADAPTERS
from .index import build_index
from .io_utils import load_yaml


def data_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root", type=Path, default=Path(os.getenv("GREF_RAG_DATA_DIR", "data"))
    )


def dataset(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    data_root(parser)


def metric_options(parser: argparse.ArgumentParser, *, select_adapter: bool = True) -> None:
    if select_adapter:
        parser.add_argument("--recall-adapter", choices=ADAPTERS, default="gref-rag")
    parser.add_argument("--gold-policy", choices=("paper", "hipporag2"), default="paper")
    parser.add_argument("--limit", type=int, choices=range(1, 1001), metavar="N")
    parser.add_argument("--retrieval-only", action="store_true")


def print_summary(result: dict) -> None:
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "per_query"},
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("data", help="download or verify pinned benchmark files")
    data.add_argument("--dataset", choices=DATASETS, action="append", dest="datasets")
    data.add_argument("--with-2wiki-aliases", action="store_true")
    data.add_argument("--verify-only", action="store_true")
    data_root(data)

    news = commands.add_parser("news-data", help="download or verify the pinned MultiHop-RAG files")
    news.add_argument("--verify-only", action="store_true")
    data_root(news)

    index = commands.add_parser("index", help="build the evidence graph and dense indices")
    dataset(index)
    index.add_argument("--config", type=Path, default=Path("configs/common.yaml"))
    index.add_argument("--index-dir", type=Path, required=True)

    run = commands.add_parser("run", help="run GRef-RAG retrieval and QA")
    dataset(run)
    metric_options(run, select_adapter=False)
    run.add_argument("--config", type=Path, default=Path("configs/common.yaml"))
    run.add_argument("--index-dir", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--save-context", action="store_true")
    run.add_argument("--save-evidence", action="store_true")

    generate = commands.add_parser("generate", help="run fixed-prompt QA on saved contexts")
    dataset(generate)
    metric_options(generate)
    generate.add_argument("--config", type=Path, default=Path("configs/common.yaml"))
    generate.add_argument("--contexts", type=Path, required=True)
    generate.add_argument("--provenance", type=Path)
    generate.add_argument("--index-metrics", type=Path)
    generate.add_argument("--output", type=Path, required=True)

    evaluate = commands.add_parser(
        "evaluate", help="score existing predictions with a provenance adapter"
    )
    dataset(evaluate)
    metric_options(evaluate)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--provenance", type=Path)
    evaluate.add_argument("--index-metrics", type=Path)
    evaluate.add_argument("--protocol", choices=("common", "native"), default="common")
    evaluate.add_argument("--output", type=Path, required=True)

    pathrag = commands.add_parser(
        "pathrag-provenance", help="export source titles from the final guarded paths"
    )
    pathrag.add_argument("--predictions", type=Path, required=True)
    pathrag.add_argument("--index-dir", type=Path, required=True)
    pathrag.add_argument("--source-manifest", type=Path, required=True)
    pathrag.add_argument("--output", type=Path, required=True)

    check = commands.add_parser("compare", help="compare a live run with a paper reference ledger")
    check.add_argument("--predictions", type=Path, required=True)
    check.add_argument("--metrics", type=Path, required=True)
    check.add_argument("--reference", type=Path, required=True)
    check.add_argument("--output", type=Path, required=True)

    statistics = commands.add_parser(
        "statistics", help="recompute the paper's paired common-prompt F1 tests"
    )
    statistics.add_argument("--ledger", type=Path, default=Path("res/common_query_scores.csv"))
    statistics.add_argument("--output", type=Path, required=True)

    budgets = commands.add_parser("matched-budget", help="run the matched context budget controls")
    dataset(budgets)
    budgets.add_argument("--config", type=Path, default=Path("configs/common.yaml"))
    budgets.add_argument("--gref-results", type=Path, required=True)
    budgets.add_argument("--kg-results", type=Path, required=True)
    budgets.add_argument("--output", type=Path, required=True)
    budgets.add_argument("--limit", type=int, choices=range(1, 1001), metavar="N")
    budgets.add_argument("--workers", type=int, choices=range(1, 65), default=32, metavar="N")
    budgets.add_argument("--prepare-only", action="store_true")

    budget_stats = commands.add_parser(
        "matched-budget-statistics", help="recompute paired matched-budget F1 intervals"
    )
    budget_stats.add_argument("--ledger", type=Path, default=Path("res/matched_budget_scores.csv"))
    budget_stats.add_argument("--output", type=Path, required=True)

    verify = commands.add_parser("verify-release", help="check the camera-ready numbers on a CPU")
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--paper-tex", type=Path)

    args = parser.parse_args()
    if args.command == "verify-release":
        from .evaluation.release_verification import verify_release

        counts = verify_release(args.output, args.paper_tex)
        print_summary(counts)
        if counts.get("Mismatch"):
            raise SystemExit(1)
    elif args.command == "news-data":
        from .data import prepare_news_inputs

        print_summary(prepare_news_inputs(args.data_root, args.verify_only))
    elif args.command == "matched-budget":
        from .budget_experiment import run_budget_experiment

        print_summary(run_budget_experiment(args, load_yaml(args.config)))
    elif args.command == "matched-budget-statistics":
        from .evaluation.budget_statistics import write_budget_statistics

        rows = write_budget_statistics(args.ledger, args.output)
        print_summary({"comparisons": len(rows)})
    elif args.command == "statistics":
        from .evaluation.statistics import write_common_statistics

        rows = write_common_statistics(args.ledger, args.output)
        print_summary(
            {
                "comparisons": len(rows),
                "holm_rejections": sum(row["holm_reject"] for row in rows),
            }
        )
    elif args.command == "compare":
        from .evaluation.parity import compare

        result = compare(args.predictions, args.metrics, args.reference, args.output)
        print_summary(result)
        if result["first_differing_stage"] is not None:
            raise SystemExit(1)
    elif args.command == "data":
        datasets = args.datasets or list(DATASETS)
        if not args.verify_only:
            download_inputs(args.data_root, datasets, args.with_2wiki_aliases)
        for name in datasets:
            examples, documents = read_benchmark(args.data_root, name)
            print(f"{name}: {len(examples)} queries, {len(documents)} corpus units")
    elif args.command == "pathrag-provenance":
        from .evaluation.provenance import export_pathrag

        export_pathrag(args.predictions, args.index_dir, args.source_manifest, args.output)
    elif args.command == "index":
        cfg = load_yaml(args.config)
        _, documents = read_benchmark(args.data_root, args.dataset)
        corpus_hash = file_sha256(data_paths(args.data_root, args.dataset)[1])
        print_summary(build_index(documents, cfg, args.index_dir, corpus_hash))
    elif args.command == "run":
        from .pipeline import run_experiment

        run_experiment(args, load_yaml(args.config))
    elif args.command == "generate":
        from .pipeline import generate_from_contexts

        if args.retrieval_only:
            parser.error("generate always performs QA")
        print_summary(generate_from_contexts(args, load_yaml(args.config)))
    else:
        index_metrics = json.loads(args.index_metrics.read_text()) if args.index_metrics else None
        print_summary(
            evaluate_file(
                args.predictions,
                args.output,
                args.data_root,
                args.dataset,
                protocol=args.protocol,
                method=args.recall_adapter,
                limit=args.limit,
                gold_policy=args.gold_policy,
                provenance=args.provenance,
                retrieval_only=args.retrieval_only,
                index_metrics=index_metrics,
            )
        )


if __name__ == "__main__":
    main()
