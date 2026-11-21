# Camera-ready results

The tables contain frozen experiment outputs and paper values. EM, F1, and recall are fractions. The ledgers contain IDs, scores, counts, and hashes. Benchmark text and model predictions are obtained or generated separately.

| File | Contents |
| --- | --- |
| `common_prompt.csv` | Full common-prompt QA table, with method-specific historical recall diagnostics where available |
| `common_query_scores.csv` | All 32,000 aligned common-prompt scores for the eight methods and four cohorts |
| `native_prompt.csv` | Method-native QA table, with full-input token counts |
| `online_efficiency.csv` | Macro-average context tokens, F1/1K, retrieval and generation latency |
| `offline_cost.csv` | Reported 2Wiki indexing, storage, and +1% update costs. PathRAG fresh indexing is worker-side and its warm append is the mean of three runs |
| `*-reference.jsonl` | GRef-RAG per-query hashes, counts and scores for both protocols |
| `recall_diagnostics.csv` | Historical adapter replay values. RAPTOR rows refer to the earlier tree cohort |
| `stored_parity.json` | Verification against all four saved GRef-RAG result cohorts |

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

| Run | Paper EM / F1 (%) | Live EM / F1 (%) | Paper / live token mean | Changed predictions |
| --- | ---: | ---: | ---: | ---: |
| HotpotQA common | 32.4 / 39.29 | 32.2 / 38.99 | 479.686 / 479.686 | 18 |
| HotpotQA native | 50.7 / 62.90 | 50.8 / 63.12 | 734.315 / 734.315 | 14 |
| PopQA native with question-only retrieval | 47.2 / 60.54 | 47.7 / 60.57 | 988.006 / 1013.928 | 63 |

HotpotQA figures above use the historical server arguments shown in the README. A second launch configuration produced more answer differences with identical prompts. Holding the server and prompts fixed, varying prefix-cache reuse changed 3 of 24 outputs tested. The serving configuration and prefix reuse explain observed differences in fresh outputs. Exact recovery of every original server response remains unverified.

For PopQA, 437 native contexts changed when the older metadata-aware retrieval was replaced with the archived question-only retrieval. The corrected ledger is `popqa-native-question-only-reference.jsonl`. The original ledger remains available to reproduce the camera-ready table. All 32,000 stored common-prompt QA scores and 32,000 historical adapter diagnostics were recalculated and matched.
