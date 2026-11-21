# Evaluation protocol

The common-prompt protocol matches the QA prompt, generator model name, main decoding parameters, and official answer references. Each retriever constructs its own context with its own indexing, ranking, retrieval units, and context organization. The archival audit records a shared prompt-template hash across all 32 method–dataset cells. Submitted QA payloads independently confirm the single user message and decoding settings for RAPTOR and PathRAG on all four cohorts. Their requests differ in whether they supply a seed. KG²RAG's four prediction logs record the same model name, prompt-template hash and decoding parameters per query. The main archive records the exact historical model checkpoint in eight cells. The release pins the weights and tokenizer used for fresh runs. GRef-RAG uses the question text and corpus graph as retrieval inputs. Evaluation annotations enter after retrieval and generation. The four cohorts are fixed by `manifests/query_ids/`.

The paper evaluator takes the maximum EM and multiset token F1 over all accepted references following NFKC, lowercasing, article removal, and non-alphanumeric splitting. HotpotQA uses the official scalar answer, 2Wiki adds v1.1 aliases and demonyms, MuSiQue uses the release's answer and aliases, and PopQA uses `possible_answers`. The pinned HippoRAG 2 data matches the official answer sets on all 4,000 queries once the 2Wiki alias file is applied. `--gold-policy hipporag2` uses only the answers packaged with the dataset and changes the 2Wiki evaluation.

Common context tokens come from the pinned Qwen tokenizer on actual rendered text with special tokens disabled. Native full-input tokens come from the server's reported usage, or the Qwen chat template when the endpoint omits usage. Output tokens are recorded separately. F1/1K is the dataset's mean F1 fraction divided by its mean retrieved-context tokens and multiplied by 1,000. The macro result averages the four dataset ratios equally.

The live retrieval timer includes query encoding, both retrieval phases, and rendering. The generation timer includes the chat-completions round-trip. The index-build timer starts after loading the corpus and includes graph, mention edges, dense vectors, and file writes. The evaluation records how many rows have each metric. The paper timings used method-specific environments and measurement boundaries. See `res/online_efficiency.csv` and `res/offline_cost.csv` for reported descriptive comparisons.

## Method-native setting

GRef-RAG uses a six-bundle rendering budget. HotpotQA and 2Wiki use the chain/paragraph renderer. MuSiQue and PopQA reuse the chain-deduplicated renderer recorded in the original native experiment. The headline LightRAG result uses hybrid retrieval with a short-answer adaptation and a 1,024-token output cap. Dense RAG, HippoRAG2, KG²RAG, and PathRAG retain the native prompt protocols recorded in the camera-ready experiment.

An earlier native PopQA cohort permitted the dataset's subject and Wikipedia title as anchors. The revised manuscript and release use question-only retrieval. Re-rendering the question-only bundles with the native budget changes 437 of the 1,000 earlier native contexts. The results preserve both ledgers for verification.

GRef-RAG's comparison choice line comes from the question text. The paper-native config preserves the original role string `ACE-RAG` and original route-specific decode seeds. The templates in code use `GRef-RAG` by name, matching the typeset appendix. The native configuration applies the recorded string at prompt construction and reproduces the revised four-cohort prompt references.

## Reproduction scope

| Manuscript result | What this artifact supports |
| --- | --- |
| Main common-prompt QA and token efficiency | GRef-RAG indexing and inference on the fixed cohorts. All eight methods' saved query IDs, EM/F1, and context-token counts support aggregate and paired-statistics replay. |
| Method-native QA | GRef-RAG runtime and the revised aggregate table, including the corrected PopQA cohort. |
| Matched context budgets | Both policies and all caps. The 48,000 score records support the paired F1 intervals and the actual token-use table. |
| Online and offline cost | Reported descriptive measurements and timings from new GRef-RAG runs. Measurement boundaries and hardware allocation differ across baselines. |
| Baseline provenance and recall | Method-specific adapters for supplied retrieval outputs. Baseline indexing and inference depend on the corresponding external projects and historical run settings. |
| Other supplementary experiments | The ablation and sensitivity suites, alternate generator, and MultiHop-RAG news evaluation require their original controls and inputs. |

The stored common-prompt F1 comparisons can be replayed locally with `python -m gref_rag statistics --output outputs/main-paired-statistics.csv`. The confidence intervals describe query-sampling uncertainty. Serving variability is assessed separately in the live validation record. [Results and validation](../res/README.md)

## Matched context budgets

The revised PopQA controls reuse the question-only GRef-RAG evidence bundles and the frozen KG²RAG chunk ranking. Both methods use the same 1,000 query IDs and Qwen checkpoint, common QA prompt, and decoding. The three other cohorts use saved answers from their earlier paired runs. All records use the same release evaluator with official answer references, including the 2Wiki v1.1 aliases. The stratified query bootstrap resamples pairs within each dataset and averages the four dataset means. Its intervals capture question-sampling uncertainty conditional on the recorded outputs. [Matched-budget ledger and validation](../res/README.md#matched-budget-results)

Pass a GRef-RAG JSONL export from `run --retrieval-only --save-evidence --save-context` and a KG²RAG JSONL export with `query_id`, `question`, `rendered_context`, and ranked `final_ordered_chunks`. Each chunk includes `chunk_id`, `text`, and `source_id` or `title`. Both context policies run at 512, 1,024, and 2,048 Qwen tokens. The paired runner shares one generator response for identical prompt hashes.

```bash
python -m gref_rag matched-budget --dataset popqa \
  --gref-results outputs/popqa-common/predictions.jsonl \
  --kg-results local/kg2rag-popqa.jsonl \
  --output outputs/popqa-budget
```

Append `--prepare-only` to produce the cap and prompt-hash manifest without a QA server. Run `matched-budget-statistics` on the released score ledger to reproduce the revised table.

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

For three datasets, `gref-paper` emits its uncapped value as `support_title_coverage`. The historical PopQA branch uses a mapper whose ranking can differ from the rendered evidence. The release's default GRef adapter traces evidence text present in the rendered context. Ambiguous identical evidence without an explicit source title contributes only when it can be attributed unambiguously. RAPTOR's camera-ready full-corpus recall was unmeasured. The historical tree-lineage adapter supports source-title diagnostics when descendant provenance is supplied. The revised Figure 3 uses the shared answer F1 and context-token metrics. [Figure source](../scripts/plot_common_efficiency.py)

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
