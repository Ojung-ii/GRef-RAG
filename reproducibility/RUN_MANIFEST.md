# Run manifest

The [machine-readable manifest](run_manifest.csv) connects tables, figures, and paired analyses in the revised Findings of EACL 2027 camera-ready to repository-relative outputs. [Paper snapshot](paper_snapshot.json) lists the identifiers from the compiled manuscript and records source, PDF, and figure hashes.

## Identifiers and pointers

- **Artifact ID** identifies a released record or aggregation. Its name is a lookup key in this repository.
- **Run ID** is an identifier present in execution records. **Archived directory label** preserves an identifying directory suffix from the surviving archive. These fields describe different kinds of evidence. Different conditions can share an execution ID.
- **Output path** points to saved scores, timing components, or a specification. **Output selector** gives the matching dataset, method, and condition. The matched-budget ledger stores its policy and cap in separate columns. **Metrics path** points to the corresponding aggregate, statistical reference, or algorithm. Aggregate records list their individual inputs in **Input artifact IDs**.
- **Config path** points to the relevant release configuration or a row in [recorded configuration metadata](config_records.csv), selected by artifact ID. That metadata includes preserved historical config digests, checkpoint revisions, and decoding fields when available. The packaged YAML files are the runnable release configurations.
- **Source SHA-256** fingerprints the original evidence when retained. The integrity column records whether it was checked against stored bytes or preserved from the archival audit. The result paths expose the score projections and summaries used in this release.
- **Not retained** marks an unavailable archived field. **Unverified** marks a link that could not be established. **Not applicable** covers derived statistics, diagrams, and similar records.

## Manuscript map

| Reference | Evidence and lookup |
| --- | --- |
| Table 1 and Main Result | `common.*` and `paired.common`. [Main query scores](../res/common_query_scores.csv) · [Paired tests](../res/main_paired_statistics.csv) |
| Table 2 | `timing.*` identifies the retrieval and QA timing cohorts. `efficiency.*` links the aggregates to timings and Table 1. [Timing components](../res/supplementary/online_timing_components.csv) · [Efficiency table](../res/online_efficiency.csv) |
| Table 3 | `ablation.*`. The primary Full and semantic-comparison Full are distinct inputs. [Ablation scores](../res/supplementary/ablation_query_scores.csv) · [Paired reference](../res/supplementary/ablation_statistics_reference.csv) |
| Table 4(a), Table 7 | `budget.*` and `paired.matched.*`. [Score ledger](../res/matched_budget_scores.csv) · [Paired and token results](../res/matched_budget_summary.csv) |
| Table 4(b), Table 6 | `llama.*`, `paired.llama`, and common-prompt inputs. [Llama scores](../res/supplementary/llama_query_scores.csv) |
| Table 4(c) | `news.*` and `paired.news`. [News QA and bridge scores](../res/supplementary/news_query_scores.csv) · [Holm family](../res/supplementary/news_statistics_reference.csv) |
| Table 4(d) | `robust.*` and `paired.robust`. [Link perturbation scores](../res/supplementary/robustness_query_scores.csv) · [Control and seed map](../res/supplementary/robustness_conditions.csv) |
| Table 5 | `protocol`. [Common config](../configs/common.yaml) · [Native config](../configs/native.yaml) · [Evaluation protocol](../docs/evaluation.md) |
| Table 8 | `native.*`. [Native query scores and retained token counts](../res/supplementary/native_query_scores.csv) · [Native aggregates](../res/native_prompt.csv) · [Corrected GRef-RAG PopQA records](../res/popqa-native-reference.jsonl) |
| Table 9 | `render.*`. [Fixed-bundle rendering-budget scores](../res/supplementary/rendering_query_scores.csv) |
| Table 10 | `sensitivity.*` and `paired.sensitivity`. [Seed, fusion, and candidate-cap scores](../res/supplementary/sensitivity_query_scores.csv) |
| Table 11 | `offline.*`. [Archived offline summaries](../res/offline_cost.csv) |
| Table 12 | `ablation.*` for `*_view` variants and the corresponding Main Full controls. [Ablation scores](../res/supplementary/ablation_query_scores.csv) |
| Figure 1 | `figure1`. [Observed common and native QA scores for the illustrated question](../res/supplementary/qualitative_case_evidence.csv). The original run for the larger panel is unverified. |
| Figure 2 | `figure2`. [Method diagram](../docs/framework.png) |
| Figure 3 | `figure3` and the same 32 method–dataset inputs as Table 1. [Plot data](../res/common_prompt.csv) · [Plotting script](../scripts/plot_common_efficiency.py) |
| Historical recall | `recall.archive` and `recall.rendered.gref`. [Adapter-specific diagnostics](../res/recall_diagnostics.csv) · [GRef-RAG rendered recall](../res/gref_rendered_recall.csv) |

## CPU verification

Run the following from the repository root after installing [the client dependencies](../requirements.txt).

```bash
python -m gref_rag verify-release --output outputs/camera-ready-verification.json
```

Add `--paper-tex /path/to/main.tex` to assert the recorded manuscript checksum. The verifier derives values from the score ledgers and archived measurement records, then compares them at the precision reported in [the paper-value ledger](paper_values.csv). The output lists every compared value and exits with status 1 on any discrepancy. The [release verification summary](verification_summary.json) records the check counts and figure comparisons. [Result provenance](RESULT_PROVENANCE.md) describes the replay scope and surviving evidence.
