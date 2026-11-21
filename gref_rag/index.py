"""Build and load compatible graph and dense indexes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from .embedding import DenseIndex, build_or_load_dense_indexes
from .graph import EvidenceGraph, bridge_files_ready
from .io_utils import atomic_output

DENSE_UNITS = {
    "proposition": ("propositions", "prop_id"),
    "chunk": ("chunks", "chunk_id"),
}


def build_index(documents, cfg, destination: Path, corpus_sha256: str) -> dict:
    # Publish all files together after the graph and embeddings are complete.
    if destination.exists():
        raise FileExistsError(f"Index already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".building-index-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "index"
        indexing = dict(cfg["indexing"], bridge=cfg["retrieval"]["bridge"])
        retrieval = dict(cfg["retrieval"], embedding=indexing["embedding"])
        started = perf_counter()
        graph = EvidenceGraph(indexing).build(documents)
        EvidenceGraph.save(graph, staged)
        graph_seconds = perf_counter() - started
        dense_started = perf_counter()
        build_or_load_dense_indexes(graph, retrieval, staged)
        dense_seconds = perf_counter() - dense_started
        report = {
            "corpus_sha256": corpus_sha256,
            "indexing_seconds": graph_seconds + dense_seconds,
            "graph_seconds": graph_seconds,
            "mention_edge_seconds": graph["meta"].get("mention_build_seconds"),
            "dense_seconds": dense_seconds,
            "storage_bytes": sum(
                path.stat().st_size for path in staged.rglob("*") if path.is_file()
            ),
            "timing_boundary": "graph and dense construction including writes; corpus loading excluded",
        }
        with atomic_output(staged / "index_metrics.json") as stream:
            json.dump(report, stream, indent=2)
        os.rename(staged, destination)
    return report


def read_index(directory: Path, cfg, expected_corpus_sha256: str) -> tuple[dict, dict, dict]:
    if not bridge_files_ready(directory):
        raise FileNotFoundError(
            f"Missing mention provenance in {directory}. Run the index command first."
        )
    graph = EvidenceGraph.load(directory)
    path = directory / "index_metrics.json"
    metrics = json.loads(path.read_text()) if path.is_file() else {}
    if metrics.get("corpus_sha256", expected_corpus_sha256) != expected_corpus_sha256:
        raise ValueError(f"Index corpus checksum differs from evaluation corpus: {directory}")
    dense = {}
    if cfg["retrieval"]["dense"]["enabled"]:
        for unit in cfg["retrieval"]["dense"]["units"]:
            field, identifier = DENSE_UNITS[unit]
            index = DenseIndex.load(directory / "dense", unit, graph[field], identifier, unit)
            if index.ids != [str(row[identifier]) for row in graph[field]] or len(
                index.matrix
            ) != len(index.ids):
                raise ValueError(f"Graph and dense {unit} IDs do not align in {directory}")
            dense[unit] = index
    return graph, dense, metrics
