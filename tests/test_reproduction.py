import json
from pathlib import Path

from gref_rag.data import CorpusDoc, QAExample
from gref_rag.evaluation.metrics import answer_scores
from gref_rag.evaluation.recall import ADAPTERS
from gref_rag.graph import EvidenceGraph
from gref_rag.io_utils import load_yaml
from gref_rag.prompts import build_prompt
from gref_rag.rendering import render_evidence, source_titles
from gref_rag.retrieval import GRefRagRetriever


def test_question_only_chain_renders_and_attributes_a_source():
    config = load_yaml(Path(__file__).parents[1] / "configs/common.yaml")
    document = CorpusDoc("1", "California History", "California became a state in 1850.")
    graph = EvidenceGraph({"show_progress": False}).build([document])
    retriever = GRefRagRetriever(graph, config["retrieval"])
    result = retriever.retrieve("When did California become a state?")
    bundles = result["evidence_bundles"]
    context = render_evidence(bundles, 4, 16000, "common")
    assert "- California History: California became a state in 1850." in context
    assert source_titles(bundles, context) == [document.title]
    clipped = context.split("\n")[0]
    assert source_titles(bundles, clipped) == []


def test_qa_and_native_route():
    assert answer_scores("The RED red star.", ["red star", "red red star"]) == (1.0, 1.0)
    assert answer_scores("red red star", ["red star"]) == (0.0, 0.8)
    question = "Who was born earlier, John Smith or Jane Doe?"
    prompt, route = build_prompt(question, "Evidence.", "native", native_role="ACE-RAG")
    assert route == "entity_choice_comparison"
    assert "Choices: John Smith | Jane Doe. Return exactly one.\nAnswer:" in prompt
    assert "ACE-RAG structured evidence" in prompt
    assert "Choices:" not in build_prompt("Were A and B both actors?", "Evidence.", "native")[0]


def test_pathrag_and_kg2rag_apply_cutoff_at_different_stages():
    qa = QAExample("1", "", ["answer"], metadata={"supporting_facts": [["Correct Source", 0]]})
    sources = ["Duplicate"] * 5 + ["Correct Source"]
    path = ADAPTERS["pathrag"].score({"ranked_source_titles": sources}, qa, "hotpotqa", 5)
    kg = ADAPTERS["kg2rag"].score({"final_source_titles": sources}, qa, "hotpotqa", 5)
    assert path.value == 1.0
    assert kg.value == 0.0


def test_raptor_can_credit_multiple_sources_from_one_node():
    qa = QAExample(
        "1",
        "",
        ["answer"],
        metadata={"supporting_facts": [["Two Piña Coladas", 0], ["Sevens (album)", 1]]},
    )
    provenance = {
        "retrieved_nodes": [{"descendant_source_titles": ["Two Piña Coladas", "Sevens (album)"]}]
    }
    recall = ADAPTERS["raptor"].score(provenance, qa, "hotpotqa", 1)
    assert recall.hits == recall.total == 2


def test_pathrag_prefers_guarded_paths_to_stored_convenience_fields(tmp_path):
    from gref_rag.evaluation.provenance import export_pathrag

    graph = """<graphml><key id="k" attr.name="source_id"/><graph>
      <node id="A"><data key="k">c</data></node></graph></graphml>"""
    (tmp_path / "graph_chunk_entity_relation.graphml").write_text(graph)
    (tmp_path / "kv_store_text_chunks.json").write_text('{"c":{"full_doc_id":"1"}}')
    manifest = tmp_path / "sources.jsonl"
    manifest.write_text('{"full_doc_id":"1","title":"After Guard"}\n')
    paths = tmp_path / "paths.jsonl"
    paths.write_text(
        json.dumps(
            {
                "query_id": "q",
                "source_titles": ["Before Guard"],
                "selected_paths": [{"selection_rank": 1, "nodes": ["A"], "edges": []}],
            }
        )
        + "\n"
    )
    output = tmp_path / "provenance.jsonl"
    export_pathrag(paths, tmp_path, manifest, output)
    assert json.loads(output.read_text())["ranked_source_titles"] == ["After Guard"]
