"""Lexical query and title normalization, BM25 and residual matching."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from ..text import tokenize

RESIDUAL_STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "am",
    "do",
    "does",
    "did",
    "of",
    "for",
    "to",
    "in",
    "on",
    "at",
    "by",
    "with",
    "from",
    "and",
    "or",
    "as",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "his",
    "her",
    "their",
    "there",
    "one",
    "which",
    "who",
    "whom",
    "whose",
    "what",
    "than",
    "then",
    "also",
    "into",
    "about",
    "between",
    "among",
    "under",
    "over",
    "after",
    "before",
}


GENERIC_RELATION_TITLES = {
    "place of birth",
    "place of origin",
    "date of birth",
    "date of death",
    "occupation",
    "country",
    "county",
    "city",
    "prime minister",
    "president",
    "film",
    "song",
    "album",
    "town",
}


COMPARISON_TERMS = {
    "first",
    "earlier",
    "older",
    "same",
    "both",
    "larger",
    "smaller",
    "higher",
    "lower",
    "when",
    "which",
    "before",
    "after",
    "oldest",
    "youngest",
    "largest",
    "smallest",
    "earliest",
    "latest",
}


RESIDUAL_SELECTION_VARIANTS = {
    "residual_lexical",
    "bridge_fullquery",
    "residual_dense_only",
    "residual_hybrid_lex_first",
    "residual_dense_fallback",
    "residual_unified_alignment",
}


BUNDLE_SELECTOR_VARIANTS = {
    "semantic_marginal",
    "lexical_residual",
}


CONTEXT_BUILDER_STRATEGIES = {
    "full_refinement",
    "seed_only",
    "globally_ranked_size_matched",
}


ANSWER_SLOT_TERMS = {
    "when",
    "date",
    "year",
    "time",
    "where",
    "place",
    "location",
    "birth",
    "born",
    "birthplace",
    "die",
    "died",
    "death",
    "dead",
    "prime",
    "minister",
    "known",
    "author",
    "written",
    "wrote",
    "write",
}


TERM_VARIANTS = {
    "die": {"die", "died", "death", "dead", "deaths", "d"},
    "died": {"die", "died", "death", "dead", "deaths", "d"},
    "death": {"die", "died", "death", "dead", "deaths", "d"},
    "birth": {"birth", "born", "birthplace", "birthplaces"},
    "born": {"birth", "born", "birthplace", "birthplaces"},
    "birthplace": {"birth", "born", "birthplace", "birthplaces"},
    "place": {"place", "location", "located", "where", "born_in"},
    "when": {
        "when",
        "date",
        "year",
        "time",
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    },
    "date": {
        "when",
        "date",
        "year",
        "time",
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    },
    "minister": {"minister", "ministry"},
    "prime": {"prime"},
    "film": {"film", "films", "movie", "movies"},
    "released": {"released", "release", "published", "publication"},
    "known": {"known", "notable", "famous", "recognized", "recognised"},
    "performer": {"performer", "singer", "rapper", "artist", "vocalist"},
}


def _norm_token(token: str) -> str:
    t = str(token or "").lower().strip()
    if t.endswith("'s"):
        t = t[:-2]
    return t


def _norm_tokens(text: Any) -> list[str]:
    raw = str(text or "")
    toks = [t for t in (_norm_token(x) for x in tokenize(raw)) if t]
    if re.search(r"\bborn\s+(?:in|at)\b", raw, flags=re.I):
        toks.append("born_in")
    return toks


def _norm_phrase(text: Any) -> str:
    return " ".join(_norm_tokens(text))


def _title_aliases_for_query(title: str) -> list[str]:
    raw = str(title or "")
    aliases = {_norm_phrase(raw)}
    no_parenthetical = re.sub(r"\([^)]*\)", " ", raw)
    aliases.add(_norm_phrase(no_parenthetical))
    aliases.add(_norm_phrase(re.sub(r"\b(?:18|19|20)\d{2}\b$", " ", no_parenthetical).strip()))
    return [a for a in aliases if a]


def _term_variants(term: str) -> set[str]:
    t = _norm_token(term)
    return set(TERM_VARIANTS.get(t, {t}))


def _residual_groups(terms: Sequence[str]) -> list[set[str]]:
    groups = []
    seen = set()
    for term in terms:
        t = _norm_token(term)
        if not t or t in seen:
            continue
        seen.add(t)
        groups.append(_term_variants(t))
    return groups


def _coverage_count(terms: Sequence[str], text: Any) -> int:
    toks = set(_norm_tokens(text))
    return sum(1 for group in _residual_groups(terms) if toks & group)


def _lexical_bm25_score(
    terms: Sequence[str],
    text: Any,
    idf: Mapping[str, float] | None = None,
    avgdl: float = 20.0,
) -> float:
    toks = _norm_tokens(text)
    if not toks:
        return 0.0
    tf = Counter(toks)
    dl = len(toks)
    score = 0.0
    k1 = 1.2
    b = 0.75
    for group in _residual_groups(terms):
        freq = sum(tf.get(v, 0) for v in group)
        if freq <= 0:
            continue
        term_idf = max((float(idf.get(v, 1.0)) for v in group), default=1.0) if idf else 1.0
        denom = freq + k1 * (1 - b + b * dl / max(1e-9, avgdl))
        score += term_idf * (freq * (k1 + 1)) / max(1e-9, denom)
    return float(score)


def _bridge_rank_text(prop: Mapping[str, Any], prev_prop: Mapping[str, Any] | None) -> str:
    text = str(prop.get("text", ""))
    if not prev_prop:
        return text
    if str(prev_prop.get("chunk_id", "")) != str(prop.get("chunk_id", "")):
        return text
    prev_text = str(prev_prop.get("text", ""))
    if re.search(r"\((?:d|b|c)\.\s*$", prev_text, flags=re.I):
        return f"{prev_text} {text}".strip()
    return text


def _has_date_signal(text: Any) -> bool:
    s = str(text or "")
    return bool(
        re.search(r"\b(?:18|19|20)\d{2}\b", s)
        or re.search(
            r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\b",
            s,
            flags=re.I,
        )
    )


def _alias_spans(q_tokens: Sequence[str], alias: str) -> list[tuple[int, int]]:
    a_tokens = _norm_tokens(alias)
    n = len(a_tokens)
    if n == 0 or n > len(q_tokens):
        return []
    spans = []
    for i in range(0, len(q_tokens) - n + 1):
        if list(q_tokens[i : i + n]) == a_tokens:
            spans.append((i, i + n))
    return spans


def _is_generic_relation_norm(
    norm_title: str, generic_relation_titles: Sequence[str] | None = None
) -> bool:
    generic = {_norm_phrase(x) for x in (generic_relation_titles or GENERIC_RELATION_TITLES)}
    return bool(norm_title and norm_title in generic)


def extract_query_anchors(
    question: str,
    title_aliases: Mapping[str, Sequence[str]],
    generic_relation_titles: Sequence[str] | None = None,
) -> dict[str, Any]:
    q_tokens = _norm_tokens(question)
    generic = {_norm_phrase(x) for x in (generic_relation_titles or GENERIC_RELATION_TITLES)}
    candidates = []
    for alias, titles in (title_aliases or {}).items():
        alias_norm = _norm_phrase(alias)
        if not alias_norm:
            continue
        title_list = [str(t) for t in (titles or []) if str(t).strip()]
        if len(title_list) != 1:
            continue
        spans = _alias_spans(q_tokens, alias_norm)
        if not spans:
            continue
        title = title_list[0]
        title_norm = _norm_phrase(title)
        relation = _is_generic_relation_norm(title_norm, generic) or _is_generic_relation_norm(
            alias_norm, generic
        )
        for start, end in spans:
            candidates.append(
                {
                    "alias": alias_norm,
                    "title": title,
                    "kind": "relation" if relation else "anchor",
                    "start": start,
                    "end": end,
                    "length": end - start,
                }
            )
    candidates.sort(key=lambda x: (int(x["length"]), len(str(x["title"]))), reverse=True)
    occupied = set()
    anchor_titles = []
    relation_titles = []
    matched = []
    for cand in candidates:
        span = set(range(int(cand["start"]), int(cand["end"])))
        if span & occupied:
            continue
        occupied.update(span)
        matched.append({"alias": cand["alias"], "title": cand["title"], "kind": cand["kind"]})
        target = relation_titles if cand["kind"] == "relation" else anchor_titles
        if cand["title"] not in target:
            target.append(cand["title"])
    for phrase in sorted(generic, key=lambda x: len(x.split()), reverse=True):
        if phrase and _alias_spans(q_tokens, phrase):
            label = " ".join(w.capitalize() if len(w) > 2 else w for w in phrase.split())
            if _norm_phrase(label) not in {_norm_phrase(x) for x in relation_titles}:
                relation_titles.append(label)
            if not any(m.get("alias") == phrase and m.get("kind") == "relation" for m in matched):
                matched.append({"alias": phrase, "title": label, "kind": "relation"})
    return {
        "query_anchor_titles": anchor_titles,
        "query_relation_titles": relation_titles,
        "matched_aliases": matched,
    }


def build_residual_query(
    question: str, anchor_title: str, bridge_title: str, seed_prop_text: str
) -> list[str]:
    q_terms = _norm_tokens(question)
    seed_toks = set(_norm_tokens(seed_prop_text))
    remove = set(_norm_tokens(anchor_title)) | set(_norm_tokens(bridge_title))
    for match in re.finditer(
        r"\b[A-Z][A-Za-z0-9'_-]+(?:\s+[A-Z][A-Za-z0-9'_-]+)+", str(seed_prop_text or "")
    ):
        phrase = " ".join(_norm_tokens(match.group(0)))
        if phrase and phrase not in GENERIC_RELATION_TITLES:
            remove.update(phrase.split())
    terms = []
    seen = set()
    for term in q_terms:
        if term in RESIDUAL_STOPWORDS or term in remove or len(term) <= 1:
            continue
        if term not in ANSWER_SLOT_TERMS and seed_toks & _term_variants(term):
            continue
        if term not in seen:
            seen.add(term)
            terms.append(term)
    if not terms:
        terms = [t for t in q_terms if t not in RESIDUAL_STOPWORDS] or q_terms
    return terms


class BM25Index:
    def __init__(
        self,
        docs: Sequence[Mapping[str, Any]],
        text_key: str = "text",
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.docs = list(docs)
        self.text_key = text_key
        self.k1 = k1
        self.b = b
        self.doc_tokens = [
            tokenize(str(d.get(text_key, "")) + " " + str(d.get("title", ""))) for d in self.docs
        ]
        self.doc_lens = [len(x) for x in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))
        self.tfs = [Counter(toks) for toks in self.doc_tokens]
        df: Counter[str] = Counter()
        for toks in self.doc_tokens:
            df.update(set(toks))
        n_docs = len(self.docs)
        self.idf = {t: math.log(1.0 + (n_docs - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def topk(self, query: str, k: int) -> list[tuple[int, float]]:
        q = tokenize(query)
        if not q:
            return []
        pairs: list[tuple[int, float]] = []
        for i, (tf, dl) in enumerate(zip(self.tfs, self.doc_lens)):
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                freq = tf[t]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / max(1e-9, self.avgdl))
                s += self.idf.get(t, 0.0) * (freq * (self.k1 + 1)) / max(1e-9, denom)
            if s > 0:
                pairs.append((i, float(s)))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:k]
