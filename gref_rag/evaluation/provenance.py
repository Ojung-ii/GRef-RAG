"""Recover PathRAG source titles from the graph and guarded retrieval paths."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from ..io_utils import atomic_output, read_jsonl
from .metrics import keyed_rows
from .recall import path_source_titles


def export_pathrag(predictions: Path, index: Path, source_manifest: Path, output: Path) -> None:
    rows = keyed_rows(predictions)
    wanted = {
        node for row in rows.values() for path in row["selected_paths"] for node in path["nodes"]
    }
    names = {}
    source_ids = {}
    path = index / "graph_chunk_entity_relation.graphml"
    for _, element in ET.iterparse(path, events=("end",)):
        tag = element.tag.rsplit("}", 1)[-1]
        if tag == "key":
            names[element.attrib["id"]] = element.attrib.get("attr.name")
        elif tag == "node" and element.attrib["id"] in wanted:
            source_ids[element.attrib["id"]] = next(
                (
                    child.text
                    for child in element
                    if names.get(child.attrib.get("key")) == "source_id"
                ),
                "",
            )
        if tag in {"node", "edge"}:
            element.clear()
    if wanted - source_ids.keys():
        raise ValueError("The saved PathRAG paths do not match the supplied index")
    documents = {str(row["full_doc_id"]): row["title"] for row in read_jsonl(source_manifest)}
    chunks = json.loads((index / "kv_store_text_chunks.json").read_text())
    chunk_to_title = {
        chunk: documents[str(row["full_doc_id"])]
        for chunk, row in chunks.items()
        if str(row["full_doc_id"]) in documents
    }
    with atomic_output(output) as stream:
        for qid, row in rows.items():
            sidecar = {
                "query_id": qid,
                "ranked_source_titles": path_source_titles(row, source_ids, chunk_to_title),
            }
            if row.get("context_hash"):
                sidecar["context_hash"] = row["context_hash"]
            stream.write(json.dumps(sidecar, ensure_ascii=False) + "\n")
