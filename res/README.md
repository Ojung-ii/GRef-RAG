# Camera-ready results

The tables contain frozen experiment outputs and paper values. EM, F1, and recall are fractions. The ledgers contain IDs, scores, counts, and hashes. Benchmark text and model predictions are obtained or generated separately.

| File | Contents |
| --- | --- |
| `common_prompt.csv` | Common-prompt QA and token table |
| `common_query_scores.csv` | All 32,000 aligned common-prompt scores for the eight methods and four cohorts |
| `main_paired_statistics.csv` | Paired common-prompt F1 differences, bootstrap intervals, and Holm-adjusted tests |
| `native_prompt.csv` | Method-native QA table, with full-input token counts |
| `online_efficiency.csv` | Macro-average context tokens, F1/1K, retrieval and generation latency |
| `offline_cost.csv` | Reported 2Wiki indexing, storage, and +1% update costs. PathRAG fresh indexing is worker-side and its warm append is the mean of three runs |
| `*-reference.jsonl` | GRef-RAG per-query hashes, counts and scores for both protocols |
| `recall_diagnostics.csv` | Historical adapter replay values. RAPTOR rows refer to the earlier tree cohort |
| `gref_rendered_recall.csv` | GRef-RAG rendered-title recall from the saved common-prompt runs |
| `stored_parity.json` | Verification against all four saved GRef-RAG result cohorts |

## Replay the main comparisons

```bash
python -m gref_rag statistics --output outputs/main-paired-statistics.csv
```

The released common-prompt ledger reproduces all 35 recorded paired results, including 31 Holm rejections. The macro GRef-RAG minus KG²RAG F1 difference is 1.82 percentage points with a 95% paired-bootstrap interval of [0.66, 2.99]. PopQA is the only individual cohort whose difference against KG²RAG passes the Holm test. The live HotpotQA common F1 from the alternate server is 38.99, compared with the archived KG²RAG 39.07. This comparison illustrates the sensitivity of the HotpotQA point-estimate ordering to generation.

## Compare a run

```bash
python -m gref_rag compare \
  --predictions outputs/hotpotqa-common/predictions.jsonl \
  --metrics outputs/hotpotqa-common/metrics.json \
  --reference res/hotpotqa-common-reference.jsonl \
  --output outputs/hotpotqa-common/parity.json
```

Parity reports the first differing stage and counts the changed queries. It exits with status 1 for a difference. [Live validation records](live_validation.json) give the observed scope and settings. The reference ledgers retain the paper figures by construction from the saved predictions. New inference depends on the serving stack and may differ even when the rendered inputs agree. [vLLM reproducibility notes](https://docs.vllm.ai/en/v0.19.0/usage/reproducibility/).

## Live validation

The release rebuilt the HotpotQA graph from the pinned corpus and matched all seven graph files against the archived index. Fresh NV-Embed query encoding with the archived dense passage index reproduced all 1,000 common contexts, prompts, and context-token totals. HotpotQA native generation used the 1,000 saved contexts whose exact rendering and prompt hashes were independently reproduced.

An additional HotpotQA check rebuilt the graph and both dense indexes from the public inputs with `NVEMBED_BATCH_SIZE=8`. The sentence and chunk IDs matched the archived indexes, and the newly computed vectors differed numerically. The first 50 common-prompt queries matched all saved context and prompt hashes, token counts, and rendered source titles. [Dense index rebuild and comparison](live_validation.json)

| Run | Paper EM / F1 (%) | Live EM / F1 (%) | Paper / live token mean | Changed predictions |
| --- | ---: | ---: | ---: | ---: |
| HotpotQA common | 32.4 / 39.29 | 32.2 / 38.99 | 479.686 / 479.686 | 18 |
| HotpotQA native | 50.7 / 62.90 | 50.8 / 63.12 | 734.315 / 734.315 | 14 |
| PopQA native with question-only retrieval | 47.2 / 60.54 | 47.7 / 60.57 | 988.006 / 1013.928 | 63 |

HotpotQA figures above use the historical server arguments shown in the README. A second launch configuration produced more answer differences with identical prompts. Holding the server and prompts fixed, varying prefix-cache reuse changed 3 of 24 outputs tested. The probes establish that these serving choices can change outputs. The full original request and serving state are unavailable, so the probes do not identify a cause for every historical difference.

For PopQA, 437 native contexts changed when the older metadata-aware retrieval was replaced with the archived question-only retrieval. The corrected ledger is `popqa-native-question-only-reference.jsonl`. The original ledger remains available to reproduce the camera-ready table. All 32,000 stored common-prompt QA scores and 32,000 historical adapter diagnostics were recalculated and matched.
