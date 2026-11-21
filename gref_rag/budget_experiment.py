"""Run the common QA protocol on paired, fixed-token-budget contexts."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from .data import apply_official_aliases, file_sha256, read_benchmark
from .evaluation.budgets import budget_contexts
from .evaluation.metrics import answer_scores, keyed_rows
from .generation import Generator, TokenCounter
from .io_utils import atomic_output
from .pipeline import decoding_manifest, text_hash
from .prompts import build_prompt
from .text import normalize_text


def prepare_cases(examples, inputs, tokenizer, dataset, limit):
    expected = {example.id for example in examples}
    retrieval = {method: keyed_rows(path) for method, path in inputs.items()}
    for method, rows in retrieval.items():
        if not {example.id for example in examples[:limit]} <= rows.keys() <= expected:
            raise ValueError(f"{method} export and requested {dataset} cohort differ")

    cases, prompts = [], {}
    for example in examples[:limit]:
        for method, rows in retrieval.items():
            row = rows[example.id]
            if normalize_text(row["question"]) != example.question:
                raise ValueError(f"Wrong question for {method}/{example.id}")
            if any(row.get(key) for key in ("error", "generation_error", "retrieval_error")):
                raise ValueError(f"Failed retrieval for {method}/{example.id}")
            for variant in budget_contexts(row, method, tokenizer):
                prompt, _ = build_prompt(example.question, variant.pop("context"), "common")
                digest = text_hash(prompt)
                prompts.setdefault(digest, prompt)
                cases.append(
                    dict(
                        dataset=dataset,
                        method=method,
                        query_id=example.id,
                        prompt_sha256=digest,
                        **variant,
                    )
                )
    return cases, prompts


def run_budget_experiment(args, config):
    if config["protocol"] != "common":
        raise ValueError("Matched budgets require the common QA configuration")
    examples, _ = read_benchmark(args.data_root, args.dataset)
    generation = config["generation"]
    tokenizer = TokenCounter(generation["tokenizer"]).tokenizer
    inputs = {"gref-rag": args.gref_results, "kg2rag": args.kg_results}
    cases, prompts = prepare_cases(examples, inputs, tokenizer, args.dataset, args.limit)
    report = {
        "dataset": args.dataset,
        "conditions": len(cases),
        "distinct_prompts": len(prompts),
        "decoding": decoding_manifest(generation),
        "retrieval_sha256": {method: file_sha256(path) for method, path in inputs.items()},
        "workers": args.workers,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    with atomic_output(args.output / "cases.jsonl") as stream:
        for case in cases:
            stream.write(json.dumps(case) + "\n")
    if not args.prepare_only:
        apply_official_aliases(examples[: args.limit], args.dataset, "paper", args.data_root)
        answers = {example.id: example.answers for example in examples[: args.limit]}
        generator = Generator(generation)
        generator.verify_model()

        def complete(prompt):
            return generator.complete(prompt, generation["seed"])["prediction"]

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            completed = tqdm(pool.map(complete, prompts.values()), total=len(prompts), desc="QA")
            predictions = dict(zip(prompts, completed, strict=True))
        with atomic_output(args.output / "scores.jsonl") as stream:
            for case in cases:
                prediction = predictions[case["prompt_sha256"]]
                em, f1 = answer_scores(prediction, answers[case["query_id"]])
                stream.write(json.dumps(dict(case, prediction=prediction, em=em, f1=f1)) + "\n")
    with atomic_output(args.output / "run_manifest.json") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report
