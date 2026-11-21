# Evaluation protocol

All methods in the common-prompt setting share the official answer references, QA prompt, model, and decoding settings. Each retriever constructs its own context with its own indexing, ranking, retrieval units, and context organization. GRef-RAG uses the question text and corpus graph as the retrieval inputs. Evaluation annotations enter after retrieval and generation. The four cohorts are fixed by `manifests/query_ids/`.

The paper evaluator takes the maximum EM and multiset token F1 over all accepted references following NFKC, lowercasing, article removal, and non-alphanumeric splitting. HotpotQA uses the official scalar answer, 2Wiki adds v1.1 aliases and demonyms, MuSiQue uses the release's answer and aliases, and PopQA uses `possible_answers`. The pinned HippoRAG 2 data matches the official answer sets on all 4,000 queries once the 2Wiki alias file is applied. `--gold-policy hipporag2` uses only the answers packaged with the dataset and changes the 2Wiki evaluation.

Common context tokens come from the pinned Qwen tokenizer on actual rendered text with special tokens disabled. Native full-input tokens come from the server's reported usage, or the Qwen chat template when the endpoint omits usage. Output tokens are recorded separately. F1/1K is the dataset's mean F1 fraction divided by its mean retrieved-context tokens and multiplied by 1,000. The macro result averages the four dataset ratios equally.

The live retrieval timer includes query encoding, both retrieval phases, and rendering. The generation timer includes the chat-completions round-trip. The index-build timer starts after loading the corpus and includes graph, mention edges, dense vectors, and file writes. The evaluation records how many rows have each metric. The paper timings used method-specific environments and measurement boundaries. See `res/online_efficiency.csv` and `res/offline_cost.csv` for reported descriptive comparisons.

## Method-native setting

GRef-RAG uses a six-bundle rendering budget. HotpotQA and 2Wiki use the chain/paragraph renderer. MuSiQue and PopQA reuse the chain-deduplicated renderer recorded in the original native experiment. The headline LightRAG result uses hybrid retrieval with a short-answer adaptation and a 1,024-token output cap. Dense RAG, HippoRAG2, KG²RAG, and PathRAG retain the native prompt protocols recorded in the camera-ready experiment.

The archived native PopQA cohort used an earlier retrieval path that could add the dataset's subject and Wikipedia title as anchors. Its saved outputs reproduce the published scores. The release applies the question-only boundary to both prompt settings. Re-rendering the corrected PopQA bundles with the native budget changes 437 of the 1,000 archived native contexts. This historical difference concerns retrieval input provenance. The question-only rerun is recorded separately in the results.

GRef-RAG's comparison choice line comes from the question text. The paper-native config preserves the original role string `ACE-RAG` and original route-specific decode seeds. The templates in code use `GRef-RAG` by name, matching the typeset appendix. The native configuration applies the recorded string at prompt construction and reproduces the saved prompt hashes for all four cohorts.

## Recall adapters

The adapter name, retrieval unit, gold unit, numerator, and denominator are emitted in the evaluation outputs. These diagnostics reflect method-specific provenance and retrieval units. A RAPTOR node can represent many source documents, a PathRAG path refers to graph-associated sources, and a LightRAG chunk can match a textual support fact. Read recall within the corresponding adapter's scope.

| Adapter | Rank and matching rule | Provenance fields in each JSONL row |
| --- | --- | --- |
| `gref-rag` (default) | Unique normalized titles in emitted-evidence order, first five against supporting titles | `source_titles` from a release run. Archived GRef runs can supply `rendered_context` and `evidence_bundles` |
| `gref-paper` | Original all-bundle title coverage for HotpotQA, 2Wiki, and MuSiQue. PopQA uses the historical mapper's first five unique titles | `evidence_bundles`, plus `rendered_source_titles` or `rendered_bundle_count` when available |
| `bm25` | First five native passages with text-aware SupportFact ID matching | `top5_documents` and frozen `gold_supports` |
| `dense-rag`, `hipporag2` | First five ranked passages with the same support matcher | `docs` plus `gold_docs`, or `retrieved` plus `gold_supports` |
| `lightrag` | First five rank-sorted chunks, with evidence containment, title/token overlap, or source-ID matching | `retrieved` and `gold_supports` with support IDs and text resolved from the benchmark |
| `kg2rag` | Slice the final ranked chunk list before matching the source-title set | `final_source_titles` or `final_ordered_chunks` |
| `pathrag` | Expand guarded paths in selection rank, node provenance before edge provenance, deduplicate exact titles, take five, and normalize | `ranked_source_titles` from the final graph mapping, or `selected_paths` with `node_source_ids` and `chunk_to_title` |
| `raptor` | Take five native nodes, union their descendant source titles, and compare against supporting titles | `retrieved_nodes` with `descendant_source_titles` per node in native rank order |

The historical `gref-paper` coverage is uncapped for three datasets. The PopQA branch uses a mapper whose ranking can differ from the rendered evidence. The release's default GRef adapter traces evidence text present in the rendered context. Ambiguous identical evidence without an explicit source title contributes only when it can be attributed unambiguously. RAPTOR's camera-ready full-corpus recall was unmeasured. The historical tree-lineage adapter supports source-title diagnostics when descendant provenance is supplied.

### Score a baseline export

Save one UTF-8 JSON object per line with a canonical `query_id` and the string `prediction`. Include provenance on the same rows or supply a separate `--provenance` JSONL with one row per query. Include the original `context_sha256` in both files when available. With a sidecar, the CLI verifies the query IDs and compares recorded hashes. The normalized passage records use `title`, `text`, and optional `metadata.sent_id`. BM25/Dense/Hippo text passages may also be strings of the form `title\ntext`. Support objects use `support_id` for BM25/Dense/Hippo or `id` for LightRAG, with `title`, `text`, and optional `sent_id`.

```bash
python -m gref_rag evaluate --dataset hotpotqa --protocol common \
  --recall-adapter kg2rag --predictions outputs/kg2rag.jsonl \
  --provenance outputs/kg2rag-retrieval.jsonl --output outputs/kg2rag-metrics.json
```

Use field names `context_tokens`, `full_input_tokens`, `output_tokens`, `retrieval_seconds`, and `generation_seconds` for cost accounting. Pass `--index-metrics` with an indexing JSON file containing `indexing_seconds` and its measurement boundary. The evaluator reports missing costs as null.

To recover PathRAG source titles from a stored run, supply its final path trace and the index used by that trace.

```bash
python -m gref_rag pathrag-provenance --predictions outputs/pathrag.jsonl \
  --index-dir pathrag-index --source-manifest pathrag-source-manifest.jsonl \
  --output outputs/pathrag-provenance.jsonl
```

To apply the common prompt to any baseline, use `generate --config configs/common.yaml --contexts ... --recall-adapter ...`. Context rows use `query_id` and `context` (or `rendered_context`), with the method's own rendered text. The generator builds its input from the benchmark question and that text. Baseline indexing and retrieval run in their respective repositories.

## Third-party resources

- Dataset and query cohorts. [HippoRAG 2](https://huggingface.co/datasets/osunlp/HippoRAG_2), [HotpotQA](https://hotpotqa.github.io/), [2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop), [MuSiQue](https://github.com/StonyBrookNLP/musique), [PopQA](https://huggingface.co/datasets/akariasai/PopQA)
- Embedding and generation models. NV-Embed-v2 weights use CC BY-NC 4.0. [NV-Embed-v2](https://huggingface.co/nvidia/NV-Embed-v2), [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- Baseline source repositories. [HippoRAG2](https://github.com/OSU-NLP-Group/HippoRAG), [RAPTOR](https://github.com/parthsarthi03/raptor), [LightRAG](https://github.com/HKUDS/LightRAG), [KG²RAG](https://github.com/nju-websoft/KG2RAG), [PathRAG](https://github.com/BUPT-GAMMA/PathRAG)
