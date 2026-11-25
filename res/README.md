# Camera-ready results

The tables contain frozen experiment outputs and paper values. EM, F1, and recall are fractions. The ledgers contain IDs, scores, counts, and hashes. Benchmark text and model predictions are obtained or generated separately.

The supplementary ledgers cover the ablations, alternate generator, news, mention-link robustness, native QA, and sensitivity studies. Use the [table and figure map](../reproducibility/RUN_MANIFEST.md#manuscript-map) to locate each result and its run record.

| File | Contents |
| --- | --- |
| `common_prompt.csv` | Common-prompt QA and token table |
| `common_query_scores.csv` | All 32,000 aligned common-prompt scores for the eight methods and four cohorts |
| `main_paired_statistics.csv` | Paired common-prompt F1 differences, bootstrap intervals, and Holm-adjusted tests |
| `native_prompt.csv` | Revised method-native QA table, with full-input token counts |
| `matched_budget_scores.csv` | 48,000 matched-budget records with query IDs, EM/F1, context-token counts, and cohort sources |
| `matched_budget_summary.csv` | Revised matched-budget F1 comparisons, paired intervals, and actual context-token means |
| `online_efficiency.csv` | Macro-average context tokens, F1/1K, retrieval and generation latency |
| `offline_cost.csv` | Reported 2Wiki indexing, storage, and +1% update costs. PathRAG fresh indexing is worker-side and its warm append is the mean of three runs |
| `*-reference.jsonl` | GRef-RAG per-query hashes, counts and scores for both protocols |
| `recall_diagnostics.csv` | Historical adapter replay values. RAPTOR rows refer to the earlier tree cohort |
| `gref_rendered_recall.csv` | GRef-RAG rendered-title recall from the saved common-prompt runs |
| `stored_parity.json` | Verification against the original saved GRef-RAG cohorts |

## Replay the main comparisons

```bash
python -m gref_rag statistics --output outputs/main-paired-statistics.csv
```

The released common-prompt ledger reproduces all 35 recorded paired results, including 31 Holm rejections. The macro GRef-RAG minus KG²RAG F1 difference is 1.82 percentage points with a 95% paired-bootstrap interval of [0.66, 2.99]. PopQA is the only individual cohort whose difference against KG²RAG passes the Holm test. The live HotpotQA common F1 from the alternate server is 38.99, compared with the archived KG²RAG 39.07. This comparison illustrates the sensitivity of the HotpotQA point-estimate ordering to generation.

Figure 3 uses the F1 and context-token columns in `common_prompt.csv`. To regenerate the figure, install `matplotlib==3.10.8` and run `python scripts/plot_common_efficiency.py --output outputs/common-prompt-context-f1.pdf`.

## Matched-budget results

```bash
python -m gref_rag matched-budget-statistics --output outputs/matched-budget-summary.csv
```

The revised results contain new PopQA answers for both methods, both budget policies, and all three caps. The GRef-RAG inputs come from question-only retrieval. The other three datasets reuse saved answers, which were rescored with the same official references used for the new PopQA answers. Applying the 2Wiki v1.1 aliases changed 266 archived EM or F1 records. The ledger includes both methods for all 4,000 query IDs under each budget and policy. [Run and scoring provenance](matched_budget_validation.json)

Table 7 draws its token means from the same packing rows used for the F1 comparisons. It preserves the existing three-decimal formatting of the floating-point macro averages. At exact decimal ties this can differ by 0.001 token from rounding an integer token sum divided by 4,000.

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

An additional HotpotQA check rebuilt the graph and both dense indexes with `NVEMBED_BATCH_SIZE=8`. All sentence and chunk IDs matched the archive. The newly computed vectors differ numerically. Across 1,000 queries, 995 context and prompt hashes matched and mean rendered-source recall was 0.9235, compared with 0.924 on the archive. With the query vectors and graph held fixed, switching to the archived passage vectors recovered all five changed contexts. This isolates the drift to the dense passage vectors for those queries. [Dense index rebuild and comparison](live_validation.json)

| Run | Archived reference EM / F1 (%) | Live EM / F1 (%) | Archived / live token mean | Changed predictions |
| --- | ---: | ---: | ---: | ---: |
| HotpotQA common | 32.4 / 39.29 | 32.2 / 38.99 | 479.686 / 479.686 | 18 |
| HotpotQA native | 50.7 / 62.90 | 50.8 / 63.12 | 734.315 / 734.315 | 14 |
| PopQA native with question-only retrieval | 47.2 / 60.54 | 47.7 / 60.57 | 988.006 / 1013.928 | 63 |

The HotpotQA rows above come from the archived alternate-server replay. The eager YaRN replay produced more answer differences with identical prompts. Holding the server and prompts fixed, varying prefix-cache reuse changed 3 of 24 outputs tested. The archive retains partial original request and serving state, so some answer differences remain unexplained. [Serving profiles](../docs/evaluation.md#processes-and-serving-profiles)

For PopQA, 437 native contexts changed when the older metadata-aware retrieval was replaced with the archived question-only retrieval. The revised table uses `popqa-native-reference.jsonl`. The earlier values are traceable to `popqa-native-historical-reference.jsonl`. All 32,000 stored common-prompt QA scores and 32,000 historical adapter diagnostics were recalculated and matched.

## Full HotpotQA GPU reproduction

Both protocols ran on the same fixed 1,000 HotpotQA IDs at commit `b5bdd6e`. The first stage regenerated answers on saved contexts. The second rebuilt the graph and both dense indexes using NV-Embed-v2, then performed fresh retrieval and generation. Both stages used the public vLLM settings. The fresh run used the public cap of 80 for both protocols. [Archived retrieval settings](../docs/evaluation.md#archived-retrieval-settings)

| Setting | Historical F1 (%) | Frozen-context replay F1 (%) | Fresh rebuild F1 (%) | Fresh context/prompt matches |
| --- | ---: | ---: | ---: | ---: |
| Common | 39.2944 | 38.9857 | 38.9857 | 995/1,000 |
| Native | 62.9038 | 63.1223 | 63.0889 | 991/1,000 |

The seven graph files match the archive. The two dense matrices match an earlier batch-8 rebuild byte for byte. Swapping only the passage matrices recovered all 14 changed contexts across the two protocols. Frozen inputs match the historical contexts and prompts on all 2,000 rows. The frozen answers differ from the historical answers on 18 common and 14 native rows. Frozen and fresh outputs on the same server agree at every identical prompt.

All four paired intervals for the change from the historical F1 include zero. These comparisons estimate sampling uncertainty for the observed runs. Establishing equivalence would require a prespecified equivalence margin and test. The historical KG²RAG common F1 is 39.0688, and the fresh GRef-RAG common F1 is 38.9857. KG²RAG was measured in its historical run. [Full GPU reproduction](hotpotqa-gpu-reproduction.json)
