"""Method-specific provenance diagnostics used in the GRef-RAG experiments."""

from __future__ import annotations

import re
import string
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from ..data import QAExample
from ..rendering import rendered_source_titles_for_bundles, source_titles
from ..text import normalize_answer, normalize_title_for_recall


@dataclass(frozen=True)
class Recall:
    hits: int
    total: int
    empty_value: float | None = None

    @property
    def value(self) -> float | None:
        return self.hits / self.total if self.total else self.empty_value


def path_title(text) -> str:
    return " ".join(re.sub(r"[^0-9a-z가-힣]+", " ", str(text or "").lower()).split())


def baseline_answer(text) -> str:
    value = str(text or "").lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", value).split())


def raptor_title(text) -> str:
    value = re.sub(r"\b(a|an|the)\b", " ", str(text or "").lower())
    return " ".join(re.sub(r"[^0-9a-z가-힣]+", " ", value).split())


def title_hits(titles, gold, normalizer, empty=None):
    relevant = set(map(normalizer, gold)) - {""}
    selected = set(map(normalizer, titles)) - {""}
    return Recall(len(relevant & selected), len(relevant), empty)


def source_list(row, *fields):
    for field in fields:
        if isinstance(row.get(field), list):
            return row[field]
    raise ValueError(f"Missing ranked provenance field; expected one of {', '.join(fields)}")


def benchmark_support_titles(qa: QAExample, dataset: str, *, popqa_wiki_titles=False):
    if dataset in {"hotpotqa", "2wiki"}:
        return [
            item.get("title") if isinstance(item, Mapping) else item[0]
            for item in qa.metadata.get("supporting_facts", [])
            if item
        ]
    if dataset == "popqa" and popqa_wiki_titles:
        return [qa.metadata.get("s_wiki_title"), qa.metadata.get("o_wiki_title")]
    return [
        p.get("title")
        for p in qa.metadata.get("paragraphs", [])
        if isinstance(p, Mapping) and p.get("is_supporting")
    ]


def gref(row, qa, dataset, k):
    if isinstance(row.get("source_titles"), list):
        titles = row["source_titles"]
    else:
        context = row.get("rendered_context") or row.get("context")
        if not isinstance(context, str):
            raise ValueError(
                "GRef-RAG recall requires source_titles or rendered text with evidence_bundles"
            )
        titles = source_titles(source_list(row, "evidence_bundles"), context)
    prefix = list(dict.fromkeys(filter(None, map(normalize_title_for_recall, titles))))[:k]
    return title_hits(prefix, qa.support_titles, normalize_title_for_recall)


def gref_paper(row, qa, dataset, k):
    """Original GRef provenance used in the camera-ready diagnostic figure."""
    if dataset == "popqa":
        stored = row.get("rendered_source_titles")
        if not stored:
            bundles = source_list(row, "evidence_bundles")
            stored = rendered_source_titles_for_bundles(bundles, row.get("rendered_bundle_count"))
        prefix = list(dict.fromkeys(filter(None, map(normalize_title_for_recall, stored))))[:k]
        return title_hits(prefix, qa.support_titles, normalize_title_for_recall, 0.0)
    titles = []
    for bundle in source_list(row, "evidence_bundles"):
        titles.append(bundle.get("anchor_title"))
        for field in ("propositions", "source_chunks"):
            titles.extend(item.get("title") for item in bundle.get(field, []))
    return title_hits(titles, qa.support_titles, lambda text: str(text or "").strip().lower(), 0.0)


def kg2rag(row, qa, dataset, k):
    titles = row.get("final_source_titles")
    if not isinstance(titles, list):
        titles = [item["title"] for item in source_list(row, "final_ordered_chunks")]
    return title_hits(
        titles[:k],
        benchmark_support_titles(qa, dataset, popqa_wiki_titles=True),
        normalize_answer,
        0.0,
    )


def pathrag(row, qa, dataset, k):
    # The displayed paths can be reversed. Prefer graph lineage to convenience
    # fields from the runner, whose source lists can predate the final guard.
    if "chunk_to_title" in row and "node_source_ids" in row:
        titles = path_source_titles(row, row["node_source_ids"], row["chunk_to_title"])
    else:
        titles = source_list(row, "ranked_source_titles")
    prefix = list(dict.fromkeys(filter(None, titles)))[:k]
    gold = benchmark_support_titles(qa, dataset)
    if dataset == "2wiki":
        gold.extend(item[0] for item in qa.metadata.get("evidences", []) if item)
    return title_hits(prefix, gold, path_title)


def path_source_titles(row, node_sources, chunk_to_title):
    titles = []
    for path in sorted(source_list(row, "selected_paths"), key=lambda path: path["selection_rank"]):
        sources = [node_sources[node] for node in path["nodes"]]
        sources.extend(edge.get("source_id") for edge in path["edges"])
        for source in sources:
            for chunk in str(source or "").split("<SEP>"):
                if chunk:
                    titles.append(chunk_to_title[chunk])
    return list(dict.fromkeys(titles))


def raptor(row, qa, dataset, k):
    nodes = source_list(row, "retrieved_nodes", "nodes")
    titles = [title for node in nodes[:k] for title in node["descendant_source_titles"]]
    gold = benchmark_support_titles(qa, dataset)
    if dataset == "popqa":
        gold.extend([qa.metadata.get("s_wiki_title"), qa.metadata.get("o_wiki_title")])
    return title_hits(titles, gold, raptor_title)


def split_document(value):
    text = str(value or "").strip()
    if "\n" in text:
        return tuple(part.strip() for part in text.split("\n", 1))
    return "", text


def record(value):
    if isinstance(value, Mapping):
        return value
    title, text = split_document(value)
    return {"title": title, "text": text}


def passage_matches(support, item):
    support_title = baseline_answer(support.get("title"))
    item_title = baseline_answer(item.get("title"))
    support_text = baseline_answer(support.get("text"))
    item_text = baseline_answer(item.get("text"))
    combined = baseline_answer(f"{item.get('title', '')} {item.get('text', '')}")
    if support_title and support_title == item_title:
        metadata = item.get("metadata") or {}
        sentence = item.get(
            "sent_id",
            item.get("sent_idx", metadata.get("sent_id", metadata.get("sent_idx"))),
        )
        if (
            support.get("sent_id") is not None
            and sentence is not None
            and str(sentence) == str(support["sent_id"])
        ):
            return True
        if not support_text or support_text in combined:
            return True
        tokens = support_text.split()
        if sum(token in set(item_text.split()) for token in tokens) / len(tokens) >= 0.8:
            return True
    return bool(
        support_text
        and (
            support_text in combined or (len(item_text.split()) >= 8 and item_text in support_text)
        )
    )


def passages(row, qa, dataset, k):
    items = source_list(row, "top5_documents", "docs", "retrieved", "retrieved_items")
    if "gold_supports" in row:
        supports = row["gold_supports"]
    else:
        documents = row.get("gold_docs", qa.metadata.get("gold_docs"))
        if not isinstance(documents, list):
            raise ValueError(
                "Passage recall requires gold_docs or the original gold_supports sidecar"
            )
        supports = [
            dict(record(document), support_id=str(index))
            for index, document in enumerate(documents)
        ]
    matches = {
        support["support_id"]
        for item in map(record, items[:k])
        for support in supports
        if passage_matches(support, item)
    }
    return Recall(len(matches), len(supports))


def light_matches(support, item):
    text = baseline_answer(item.get("text"))
    title = baseline_answer(item.get("title"))
    source = baseline_answer(item.get("source_id"))
    expected_text = support.get("text_norm") or baseline_answer(support.get("text"))
    expected_title = support.get("title_norm") or baseline_answer(support.get("title"))
    sentence = support.get("sent_id", -1)
    if expected_text:
        if expected_text in text or text in expected_text:
            return True
        tokens = set(baseline_answer(expected_text).split())
        if expected_title and expected_title == title and tokens:
            if len(tokens & set(text.split())) / len(tokens) >= 0.5:
                return True
    elif expected_title and (
        expected_title == title or expected_title in text or expected_title in source
    ):
        return True
    return bool(
        support.get("title")
        and sentence is not None
        and int(sentence) >= 0
        and baseline_answer(f"{support['title']}::{sentence}") == source
    )


def lightrag(row, qa, dataset, k):
    items = source_list(row, "retrieved", "retrieved_items", "retrieved_chunks")
    ranked = sorted(
        enumerate(map(record, items), 1),
        key=lambda pair: int(pair[1].get("rank", pair[0])),
    )
    supports = source_list(row, "gold_supports", "supporting_facts")
    gold = {str(support["id"]) for support in supports if support["id"]}
    matches = {
        str(support["id"])
        for _, item in ranked[:k]
        for support in supports
        if light_matches(support, item)
    }
    return Recall(len(matches & gold), len(gold), 0.0)


@dataclass(frozen=True)
class Adapter:
    score: Callable[[dict[str, Any], QAExample, str, int], Recall]
    retrieval_unit: str
    gold_unit: str


PASSAGES = Adapter(passages, "native passage prefix", "support fact ID")
ADAPTERS = {
    "gref-rag": Adapter(
        gref, "unique provenance titles of emitted evidence", "unique support title"
    ),
    "gref-paper": Adapter(
        gref_paper,
        "historical all-bundle coverage; PopQA mapped provenance prefix",
        "unique support title",
    ),
    "bm25": PASSAGES,
    "dense-rag": PASSAGES,
    "hipporag2": PASSAGES,
    "lightrag": Adapter(lightrag, "native rank-sorted chunk prefix", "support fact ID"),
    "kg2rag": Adapter(kg2rag, "final chunk prefix", "unique support title"),
    "pathrag": Adapter(
        pathrag,
        "associated source title prefix in path selection order",
        "unique support title",
    ),
    "raptor": Adapter(
        raptor,
        "tree-node prefix expanded to descendant source titles",
        "unique support title",
    ),
}
