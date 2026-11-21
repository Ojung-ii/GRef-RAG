"""Integrate benchmarking, question-only retrieval and generation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from time import perf_counter

from tqdm import tqdm

from .data import apply_official_aliases, data_paths, file_sha256, read_benchmark
from .evaluation.metrics import evaluate_file, keyed_rows
from .generation import Generator, TokenCounter
from .index import read_index
from .io_utils import atomic_output
from .prompts import build_prompt
from .rendering import render_evidence, source_titles
from .retrieval import GRefRagRetriever


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_experiment(args, cfg):
    protocol = cfg["protocol"]
    generation = cfg["generation"].copy()
    generation["layout"] = generation.get("dataset_layouts", {}).get(
        args.dataset, generation["layout"]
    )
    counter = TokenCounter(generation["tokenizer"])
    generator = None if args.retrieval_only else Generator(generation)
    if generator:
        generator.verify_model()
    examples, _ = read_benchmark(args.data_root, args.dataset)
    if not args.retrieval_only:
        # Validate gold availability before issuing any requests.
        apply_official_aliases(
            examples[: args.limit], args.dataset, args.gold_policy, args.data_root
        )
    corpus_hash = file_sha256(data_paths(args.data_root, args.dataset)[1])
    graph, dense, index_metrics = read_index(args.index_dir, cfg, corpus_hash)
    retrieval_cfg = dict(
        cfg["retrieval"], embedding=cfg["indexing"]["embedding"], _generation=generation
    )
    retriever = GRefRagRetriever(graph, retrieval_cfg, dense, context_token_counter=counter.context)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    rows_path = output / "predictions.jsonl"
    routes = Counter()
    with atomic_output(rows_path) as stream:
        for example in tqdm(examples[: args.limit], desc=f"{args.dataset} {protocol}"):
            started = perf_counter()
            retrieval = retriever.retrieve(
                example.question, dataset=args.dataset, query_id=example.id
            )
            bundles = retrieval["evidence_bundles"]
            context = render_evidence(
                bundles,
                generation["top_bundles"],
                generation["max_context_chars"],
                generation["layout"],
            )
            retrieval_seconds = perf_counter() - started
            prompt, route = build_prompt(
                example.question,
                context,
                protocol,
                native_role=generation.get("native_role", "GRef-RAG"),
            )
            routes[route] += 1
            row = {
                "query_id": example.id,
                "question": example.question,
                "context_sha256": text_hash(context),
                "prompt_sha256": text_hash(prompt),
                "selected_seed_sha256": retrieval["diagnostics"]["selected_seed_hash"],
                "bundle_candidate_sha256": retrieval["diagnostics"]["bundle_candidate_set_hash"],
                "context_tokens": counter.context(context),
                "retrieval_seconds": retrieval_seconds,
                "source_titles": source_titles(bundles, context),
                "route": route,
            }
            if args.save_evidence:
                row["evidence_bundles"] = bundles
            if args.save_context:
                row["context"] = context
            if generator:
                row.update(complete_row(generator, counter, prompt, route, args.dataset))
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    report = {
        "method": "gref-rag",
        "dataset": args.dataset,
        "protocol": protocol,
        "generation": not args.retrieval_only,
        "rendering": {
            "layout": generation["layout"],
            "bundle_budget": generation["top_bundles"],
        },
        "decoding": decoding_manifest(generation),
        "config_sha256": text_hash(json.dumps(cfg, sort_keys=True)),
        "retrieval_input": "question and corpus",
        "route_counts": dict(routes),
        "corpus_sha256": corpus_hash,
        "predictions_sha256": file_sha256(rows_path),
        "indexing": index_metrics or None,
        "timing_boundary": "retrieval includes query encoding through context rendering; generation includes API round-trip",
    }
    with atomic_output(output / "run_manifest.json") as stream:
        json.dump(report, stream, indent=2)
    evaluate_file(
        rows_path,
        output / "metrics.json",
        args.data_root,
        args.dataset,
        protocol=protocol,
        method="gref-rag",
        gold_policy=args.gold_policy,
        limit=args.limit,
        index_metrics=index_metrics,
        retrieval_only=args.retrieval_only,
    )


def decoding_manifest(cfg: dict) -> dict:
    keys = (
        "model",
        "seed",
        "temperature",
        "top_p",
        "max_new_tokens",
        "native_role",
        "native_default_seeds",
    )
    return {key: cfg[key] for key in keys if key in cfg}


def complete_row(
    generator: Generator, counter: TokenCounter, prompt: str, route: str, dataset: str
) -> dict:
    cfg = generator.config
    seed = cfg["seed"]
    if route in {"default", "binary_comparison"}:
        seed = cfg.get("native_default_seeds", {}).get(dataset, seed)
    result = generator.complete(prompt, seed)
    result["decode_seed"] = seed
    if result["full_input_tokens"] is None:
        result["full_input_tokens"] = counter.full_input(prompt)
        result["full_input_token_source"] = "qwen_chat_template"
    else:
        result["full_input_token_source"] = "server_usage"
    return result


def generate_from_contexts(args, cfg):
    """Run QA on frozen, method-rendered contexts with fixed question alignment."""
    protocol = cfg["protocol"]
    if protocol == "native" and args.recall_adapter not in ("gref-rag", "gref-paper"):
        raise ValueError(
            "The native prompt is specific to GRef-RAG. Import baseline native predictions with evaluate."
        )
    generation = cfg["generation"]
    examples, _ = read_benchmark(args.data_root, args.dataset)
    examples = examples[: args.limit]
    apply_official_aliases(examples, args.dataset, args.gold_policy, args.data_root)
    rows = keyed_rows(args.contexts)
    if rows.keys() != {example.id for example in examples}:
        raise ValueError("Frozen context and benchmark cohorts differ")
    counter = TokenCounter(generation["tokenizer"])
    generator = Generator(generation)
    generator.verify_model()
    args.output.mkdir(parents=True, exist_ok=False)
    output = args.output / "predictions.jsonl"
    routes = Counter()
    with atomic_output(output) as stream:
        for example in tqdm(examples, desc=f"{args.dataset} {protocol} QA"):
            row = rows[example.id]
            context = row.get("context", row.get("rendered_context"))
            if not isinstance(context, str):
                raise ValueError(f"Expected frozen context text for {example.id}")
            if "question" in row and row["question"] != example.question:
                from .text import normalize_text

                if normalize_text(row["question"]) != example.question:
                    raise ValueError(f"Wrong question for {example.id}")
            digest = text_hash(context)
            for field in (
                "context_sha256",
                "context_hash",
                "rendered_context_hash",
                "rendered_context_sha256",
            ):
                if row.get(field) and row[field] != digest:
                    raise ValueError(f"Frozen context checksum mismatch for {example.id}")
            prompt, route = build_prompt(
                example.question,
                context,
                protocol,
                native_role=generation.get("native_role", "GRef-RAG"),
            )
            routes[route] += 1
            generated = {
                "query_id": example.id,
                "context_sha256": digest,
                "prompt_sha256": text_hash(prompt),
                "context_tokens": counter.context(context),
                "route": route,
                "retrieval_seconds": row.get("retrieval_seconds"),
                **complete_row(generator, counter, prompt, route, args.dataset),
            }
            if args.recall_adapter == "gref-rag":
                generated["source_titles"] = row.get("source_titles")
                if generated["source_titles"] is None and row.get("evidence_bundles") is not None:
                    generated["source_titles"] = source_titles(row["evidence_bundles"], context)
            stream.write(json.dumps(generated, ensure_ascii=False) + "\n")
    index_metrics = json.loads(args.index_metrics.read_text()) if args.index_metrics else None
    with atomic_output(args.output / "run_manifest.json") as stream:
        json.dump(
            {
                "dataset": args.dataset,
                "protocol": protocol,
                "mode": "frozen_context_generation",
                "contexts_sha256": file_sha256(args.contexts),
                "predictions_sha256": file_sha256(output),
                "route_counts": dict(routes),
                "decoding": decoding_manifest(generation),
                "config_sha256": text_hash(json.dumps(cfg, sort_keys=True)),
            },
            stream,
            indent=2,
        )
    return evaluate_file(
        output,
        args.output / "metrics.json",
        args.data_root,
        args.dataset,
        protocol=protocol,
        method=args.recall_adapter,
        gold_policy=args.gold_policy,
        limit=args.limit,
        provenance=args.provenance or args.contexts,
        index_metrics=index_metrics,
    )
