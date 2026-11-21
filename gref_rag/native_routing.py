from __future__ import annotations

import re
from enum import StrEnum

from .text import normalize_answer


class ComparisonRoute(StrEnum):
    ENTITY_CHOICE_COMPARISON = "entity_choice_comparison"
    BINARY_COMPARISON = "binary_comparison"
    DEFAULT = "default"


T4_CANDIDATE_LINE_TEMPLATE = "Choices: {candidate_1} | {candidate_2}. Return exactly one."

SELECTION_INTERROGATIVE_RE = re.compile(
    r"^\s*(who|which|what\s+(?:person|film|movie|item|entity|album|book|city|country|team|organization|company|work|title))\b",
    re.I,
)
COMPARISON_CUE_RE = re.compile(
    r"\b("
    r"born\s+(?:earlier|later|first)|died\s+(?:earlier|later|first)|"
    r"released\s+(?:earlier|later|first|more\s+recently)|"
    r"older|younger|lived\s+longer|longer-lived|more\s+recent|"
    r"first|last|earlier|later"
    r")\b",
    re.I,
)
A_OR_B_SELECTION_RE = re.compile(
    r"^\s*(?:was|were)\s+(.+?)\s+or\s+(.+?)\s+"
    r"(?:(?:born|died|released)\s+)?(?:first|earlier|later|more\s+recently|older|younger|longer-lived|lived\s+longer)\b",
    re.I,
)
BINARY_CUE_RE = re.compile(r"\b(same|both|share|whether)\b", re.I)
BINARY_START_RE = re.compile(r"^\s*(are|do|does|did|is|was|were|has|have|had)\b", re.I)
IS_THAN_RE = re.compile(
    r"^\s*is\s+.+?\b(?:older|younger|earlier|later|more\s+recent|larger|smaller|longer)\s+than\b",
    re.I,
)


def clean_candidate(text: str) -> str:
    text = re.sub(r"^[\s\"'“”‘’]+|[\s\"'“”‘’?.,;:]+$", "", str(text or ""))
    return re.sub(r"\s+", " ", text).strip()


def candidate_pair(a: str, b: str, source: str) -> tuple[str, str, str]:
    return clean_candidate(a), clean_candidate(b), source


def extract_candidates_from_question(question: str) -> tuple[str, str, str]:
    q = " ".join(str(question or "").strip().split())
    between_prefix = re.search(
        r"^\s*between\s+(?:the\s+)?(?:two\s+)?(?:(?:bands?|films?|movies?|people|persons|albums?|items?|entities|groups?)\s+)?(.+?)\s+and\s+(.+?),\s*(?:which|who|what)\b",
        q,
        re.I,
    )
    if between_prefix:
        a, b, src = candidate_pair(
            between_prefix.group(1), between_prefix.group(2), "question_between_prefix"
        )
        if a and b and normalize_answer(a) != normalize_answer(b):
            return a, b, src
    comma_or = re.search(r",\s*(.+?)\s+or\s+(.+?)[\?\.]?$", q, re.I)
    if comma_or:
        a, b, src = candidate_pair(comma_or.group(1), comma_or.group(2), "question_trailing_or")
        if a and b and normalize_answer(a) != normalize_answer(b):
            return a, b, src
    patterns = [
        (
            r"^\s*(?:was|were)\s+(.+?)\s+or\s+(.+?)\s+(?:(?:born|died|released)\s+)?(?:first|earlier|later|more\s+recently|older|younger|longer-lived|lived\s+longer)\b",
            "question_a_or_b",
        ),
        (
            r"^\s*who\s+(?:is|was)\s+(?:older|younger)\s+(.+?)\s+or\s+(.+?)[\?\.]?$",
            "question_who_is_comparison_or",
        ),
        (
            r"^\s*which\s+date\s+occurred\s+first\s+(.+?)\s+or\s+(.+?)[\?\.]?$",
            "question_which_date_occurred_first_or",
        ),
        (r"\bout of\s+(.+?)\s+and\s+(.+?)[\?\.]?$", "question_out_of_and"),
        (r"\bbetween\s+(.+?)\s+and\s+(.+?)[\?\.]?$", "question_between_and"),
        (
            r"\b(?:film|movie|item|person|entity|title)\s+(.+?)\s+or\s+(.+?)[\?\.]?$",
            "question_noun_or",
        ),
    ]
    for pattern, source in patterns:
        match = re.search(pattern, q, re.I)
        if match:
            a, b, src = candidate_pair(match.group(1), match.group(2), source)
            if a and b and normalize_answer(a) != normalize_answer(b):
                return a, b, src
    return "", "", ""


def route_question(question: str) -> tuple[ComparisonRoute, str]:
    q = " ".join(str(question or "").strip().split())
    q_norm = normalize_answer(q)
    if A_OR_B_SELECTION_RE.search(q):
        return ComparisonRoute.ENTITY_CHOICE_COMPARISON, "a_or_b_selection_comparison"
    if IS_THAN_RE.search(q):
        return ComparisonRoute.BINARY_COMPARISON, "is_than_binary_comparison"
    if BINARY_CUE_RE.search(q) or re.match(
        r"^\s*(are|do|does|did)\s+.+?\s+(?:and|&)\s+.+", q, re.I
    ):
        return ComparisonRoute.BINARY_COMPARISON, "binary_comparison_cue"
    if BINARY_START_RE.search(q) and not SELECTION_INTERROGATIVE_RE.search(q):
        return ComparisonRoute.BINARY_COMPARISON, "binary_question_start"
    cand1, cand2, _source = extract_candidates_from_question(q)
    if SELECTION_INTERROGATIVE_RE.search(q) and COMPARISON_CUE_RE.search(q) and cand1 and cand2:
        return (
            ComparisonRoute.ENTITY_CHOICE_COMPARISON,
            "selection_interrogative_plus_comparison_cue",
        )
    if (
        re.search(r"\b(?:out of|between)\b", q_norm)
        and COMPARISON_CUE_RE.search(q)
        and cand1
        and cand2
    ):
        return ComparisonRoute.ENTITY_CHOICE_COMPARISON, "between_out_of_comparison"
    return ComparisonRoute.DEFAULT, "no_entity_choice_comparison_route"


def candidate_line(candidate_1: str, candidate_2: str) -> str:
    return T4_CANDIDATE_LINE_TEMPLATE.format(candidate_1=candidate_1, candidate_2=candidate_2)


def insert_before_answer(prompt: str, text: str) -> str:
    if not text:
        return prompt
    marker = "\nAnswer:"
    if marker not in prompt:
        raise RuntimeError("Prompt template does not contain the final Answer marker.")
    return prompt.replace(marker, f"\n{text}\nAnswer:", 1)


def comparison_choice_fields(question: str) -> dict[str, str]:
    cand1, cand2, source = extract_candidates_from_question(question)
    if cand1 and cand2 and source.startswith("question_"):
        return {
            "candidate_1": cand1,
            "candidate_2": cand2,
            "candidate_source": source,
            "candidate_confidence": "high_question",
            "candidate_line": candidate_line(cand1, cand2),
        }
    return {
        "candidate_1": "",
        "candidate_2": "",
        "candidate_source": "",
        "candidate_confidence": "insufficient",
        "candidate_line": "",
    }
