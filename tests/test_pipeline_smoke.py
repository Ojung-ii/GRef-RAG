"""Exercise the release CLI with a tiny fixture and deterministic QA."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

from gref_rag import __main__, data, pipeline


class FixtureTokenCounter:
    def __init__(self, name):
        self.name = name

    def context(self, text):
        return len(text.split())

    def full_input(self, prompt):
        return len(prompt.split()) + 3


class FixtureGenerator:
    def __init__(self, config):
        self.config = config

    def verify_model(self):
        assert self.config["model"] == "Qwen/Qwen2.5-7B-Instruct"

    def complete(self, prompt, seed):
        assert "John Smith" in prompt and "1940" in prompt
        return {
            "prediction": "John Smith",
            "generation_seconds": 0.01,
            "full_input_tokens": None,
            "output_tokens": 2,
        }


def test_index_run_generate_and_compare_in_both_protocols(monkeypatch, tmp_path):
    def cli(*args):
        monkeypatch.setattr(sys, "argv", ["gref_rag", *map(str, args)])
        __main__.main()

    dataset = "hotpotqa"
    query_id = "comparison"
    title = "Birth records"
    data_root = tmp_path / "data"
    qa, corpus = data.data_paths(data_root, dataset)
    qa.parent.mkdir(parents=True)
    qa.write_text(
        json.dumps(
            [
                {
                    "_id": query_id,
                    "question": "Who was born earlier, John Smith or Jane Doe?",
                    "answer": "John Smith",
                    "supporting_facts": [[title, 0]],
                }
            ]
        )
    )
    corpus.write_text(
        json.dumps(
            [{"title": title, "text": "John Smith was born in 1940. Jane Doe was born in 1950."}]
        )
    )
    manifest_dir = tmp_path / "manifests"
    (manifest_dir / "query_ids").mkdir(parents=True)
    (manifest_dir / "query_ids" / f"{dataset}.txt").write_text(f"{query_id}\n")
    query_hash = hashlib.sha256(json.dumps([query_id]).encode()).hexdigest()
    (manifest_dir / "data_manifest.json").write_text(
        json.dumps(
            {
                "datasets": {
                    dataset: {
                        "qa_sha256": data.file_sha256(qa),
                        "corpus_sha256": data.file_sha256(corpus),
                        "query_id_set_hash": query_hash,
                        "query_order_hash": query_hash,
                    }
                }
            }
        )
    )
    monkeypatch.setattr(data, "MANIFESTS", manifest_dir)
    monkeypatch.setattr(pipeline, "TokenCounter", FixtureTokenCounter)
    monkeypatch.setattr(pipeline, "Generator", FixtureGenerator)

    configs = {}
    for protocol in ("common", "native"):
        base = Path(__file__).resolve().parents[1] / "configs" / f"{protocol}.yaml"
        configs[protocol] = tmp_path / f"{protocol}.yaml"
        configs[protocol].write_text(
            f"extends: {json.dumps(str(base))}\nretrieval:\n  dense:\n    enabled: false\n"
        )
    cohort = ("--dataset", dataset, "--data-root", data_root)
    index_dir = tmp_path / "index"
    cli("data", *cohort, "--verify-only")
    cli("index", *cohort, "--index-dir", index_dir, "--config", configs["common"])

    for protocol, config in configs.items():
        direct = tmp_path / f"{protocol}-direct"
        search = tmp_path / f"{protocol}-search"
        split = tmp_path / f"{protocol}-split"
        run = (*cohort, "--index-dir", index_dir, "--config", config, "--save-context")
        cli("run", *run, "--output", direct)
        cli("run", *run, "--output", search, "--retrieval-only")
        cli(
            "generate",
            *cohort,
            "--config",
            config,
            "--output",
            split,
            "--contexts",
            search / "predictions.jsonl",
            "--index-metrics",
            index_dir / "index_metrics.json",
        )

        metrics = json.loads((direct / "metrics.json").read_text())
        assert metrics["n"] == 1
        assert metrics["em"] == metrics["f1"] == metrics["recall_at_5"] == 1
        assert metrics["indexing_seconds"] > 0
        assert metrics["context_tokens"] > 0

        row = json.loads((direct / "predictions.jsonl").read_text())
        assert row["full_input_token_source"] == "qwen_chat_template"
        if protocol == "native":
            assert row["route"] == "entity_choice_comparison"
        reference = tmp_path / f"{protocol}-reference.jsonl"
        reference.write_text(
            json.dumps(
                {
                    **row,
                    "prediction_sha256": hashlib.sha256(row["prediction"].encode()).hexdigest(),
                    "em": 1,
                    "f1": 1,
                }
            )
            + "\n"
        )
        parity = split / "parity.json"
        cli(
            "compare",
            "--predictions",
            split / "predictions.jsonl",
            "--metrics",
            split / "metrics.json",
            "--reference",
            reference,
            "--output",
            parity,
        )
        assert json.loads(parity.read_text())["first_differing_stage"] is None


def test_verify_only_rejects_modified_2wiki_aliases(monkeypatch, tmp_path):
    alias_path = tmp_path / "2wiki" / "id_aliases.json"
    alias_path.parent.mkdir()
    alias_path.write_text("{}\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gref_rag",
            "data",
            "--data-root",
            str(tmp_path),
            "--dataset",
            "2wiki",
            "--verify-only",
            "--with-2wiki-aliases",
        ],
    )
    with pytest.raises(ValueError, match="2Wiki official aliases checksum mismatch"):
        __main__.main()
