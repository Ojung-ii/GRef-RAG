"""Camera-ready common and method-native QA prompts."""

from __future__ import annotations

from .native_routing import (
    ComparisonRoute,
    comparison_choice_fields,
    insert_before_answer,
    route_question,
)

COMMON_PROMPT = """You are a QA assistant.
Answer the question using only the provided Context.
When possible, use a concise answer explicitly supported by the Context.
Return only the final short answer.
Do not output reasoning, explanations, citations, Markdown, or prefixes.
For yes/no questions, output exactly yes or no in lowercase.
If the answer cannot be found in the Context, output exactly: insufficient information

Question: {question}
Context:
{context}
Answer:"""


NATIVE_PROMPT = """---Role---

You are an expert reading-comprehension assistant for GRef-RAG structured evidence.

---Goal---

Answer the question using ONLY the provided Context.
Return only the final short answer span.

---Instructions---

1. The Context is organized into evidence sections.
2. Section headers identify the evidence type and anchor entity.
3. Earlier evidence sections are usually more relevant, but use later sections when they complete the answer.
4. Use Evidence Chain sections to connect a question anchor to a bridge entity or answer entity.
5. Use Multi-Anchor Evidence sections for comparison or questions involving multiple entities.
6. Use Supporting Evidence sections as direct answer evidence when they answer the question.
7. Ignore source ids, relation_title fields, and empty evidence sections.
8. Do not use outside knowledge.
9. If a supported answer span exists anywhere in the Context, output that answer span.
10. Output insufficient information only when no evidence in the Context supports an answer.
11. Output exactly one line with only the final answer.
12. Do not output reasoning, explanations, citations, markdown, JSON, or prefixes.
13. For yes/no questions, output exactly yes or no in lowercase.

Question: {question}
Context:
{context}
Answer:"""


COMPARISON_INSTRUCTION = """For who/which/A-or-B comparison questions, compare the relevant values and return exactly one compared name from the question, never yes or no. Use yes or no only for genuinely binary questions about whether a statement, shared property, or both-condition is true.

"""


NATIVE_COMPARISON_PROMPT = NATIVE_PROMPT.replace(
    "13. For yes/no questions, output exactly yes or no in lowercase.",
    COMPARISON_INSTRUCTION + "13. For yes/no questions, output exactly yes or no in lowercase.",
)


def build_prompt(
    question: str, context: str, protocol: str, *, native_role: str = "GRef-RAG"
) -> tuple[str, str]:
    """Return the exact user prompt and the routing label."""
    if protocol == "common":
        return COMMON_PROMPT.format(question=question, context=context), "common"
    if protocol != "native":
        raise ValueError(f"Unknown QA protocol: {protocol}")
    route, _ = route_question(question)
    template = NATIVE_PROMPT
    if route == ComparisonRoute.ENTITY_CHOICE_COMPARISON:
        template = NATIVE_COMPARISON_PROMPT
    # This literal differs in the original runs and the typeset appendix.
    template = template.replace("GRef-RAG", native_role)
    prompt = template.format(question=question, context=context)
    if route == ComparisonRoute.ENTITY_CHOICE_COMPARISON:
        candidates = comparison_choice_fields(question)
        if candidates["candidate_line"]:
            prompt = insert_before_answer(prompt, candidates["candidate_line"])
    return prompt, route.value
