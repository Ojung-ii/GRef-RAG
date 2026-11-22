# Result provenance

The source of truth is the revised camera-ready recorded in [paper_snapshot.json](paper_snapshot.json). Table numbers in [the run manifest](RUN_MANIFEST.md) follow that PDF. Earlier audit labels have been joined by semantic LaTeX label and result scope, since the manuscript consolidated several experiments into Table 4.

## Main and additional experiments

| Result | Evidence used for this release |
| --- | --- |
| Main common-prompt QA | The same 32 fixed method–dataset cohorts underlie Table 1, the quality and context-token columns of Table 2, and Figure 3. The archived scores yield the +1.82 point macro F1 difference against KG²RAG. The paper reports the top point estimates and separate paired significance tests. |
| Main pairing | The original 35-test Holm family consists of seven comparisons in each of four datasets and seven macro comparisons. Macro bootstrap and randomization are stratified by dataset. The check reproduces all 35 numerical results and all 31 rejections. |
| PathRAG | The main and native rows use the path-only controlled results. News uses the previously validated QA outputs. The historical recall adapter follows source titles associated with final guarded paths. |
| Local Refinement | Primary ablations and view ablations use the recorded Main Full. The semantic coverage ablation is paired with its own Full scores. The size-matched 2Wiki comparison uses its paired control and yields −7.84 points. |
| Matched context budgets | Complete-unit packing and fixed-prefix truncation are distinct policies. The final PopQA conditions use the same question-only GRef-RAG retrieval pool and one Qwen serving stack for both methods. HotpotQA, 2Wiki, and MuSiQue reuse archived generated answers and are rescored with the release evaluator. The official 2Wiki aliases change 266 archived EM or F1 records. |
| Llama generator | The published comparison uses saved contexts and the original Llama-3.1-8B-Instruct per-question scores. The Qwen points in Table 4(b) are the main Table 1 results. The Qwen gap labels use the rounded values shown in Table 1. |
| MultiHop-RAG news | The final semantic V2 run yields 46.85 F1 and 893 context tokens. The earlier lexical comparison is a distinct run. Dense RAG, HippoRAG2, and KG²RAG use regeneration on saved contexts. PathRAG results were reused. The four News Holm tests are separate from the main 35. |
| Mention-link robustness | The score ledger preserves the dataset, perturbation rate, and saved seed for each condition. Changes use the zero-perturbation control from the corresponding suite. Table 4(d) aggregates across datasets and the stored seeds. |
| Method-native QA | PopQA uses the corrected question-only GRef-RAG output. LightRAG uses the finalized short-answer adaptation with a 1,024-token output cap. Native token counts include the full model input. All 24 EM/F1 cells have per-query evidence. The eight Dense RAG and HippoRAG2 token cells retain archived means. The earlier PopQA result is labeled historical in the results notes. |
| Sensitivity | Rendering-budget changes preserve fixed evidence inputs. The November seed/fusion and candidate-cap runs use their own matched Full references. Their controls and Holm families are separate. |
| Online and offline cost | Timing evidence comes from method-specific environments and boundaries. Online timing components are separate from the QA score cohorts. The reported offline costs are archived summaries, including PathRAG's worker-side fresh build and mean of three warm append runs. |
| Figures | Figure 1 uses a fixed 2Wiki question. GRef-RAG's archived answer has F1 0.0 under the common prompt and F1 1.0 under the native prompt. The figure's exact larger-context run and depicted generation protocol are unverified. The README's Figure 2 image and the regenerated Figure 3 match the manuscript artwork pixel for pixel when rendered at the recorded resolutions. Method-specific recall diagnostics are available separately. [Figure 1 score evidence](../res/supplementary/qualitative_case_evidence.csv) · [Figure verification summary](verification_summary.json) |

## CPU verification scope

The verifier aggregates saved EM/F1 records and context-token counts, derives dataset-level and macro F1/1K, and checks the published values at their displayed precision. It replays the main, matched-budget, ablation, Llama, News, seed/fusion, and candidate-cap statistics using the recorded 10,000-draw policies. Separate Holm families apply to the main analysis, primary ablations, heterogeneous views, News, seed/fusion, and candidate cap. Individual percentile intervals describe paired query sampling. [Verification command](RUN_MANIFEST.md#cpu-verification) · [Paper values](paper_values.csv)

The released ledgers expose query IDs, scores, counts, and hashes. Some published values use archived measurement summaries, including offline costs and parts of native token accounting. Historical request parameters, original Run IDs, and exact executed source revisions are retained where supported by the surviving records. [Manifest](run_manifest.csv)

In the manuscript, Figure 1 is described as a successful comparison case. The archive supports describing it as an illustration of evidence selection for a comparison question. A claim about answer correctness should name the generation protocol.

Fresh GRef-RAG inference uses the supplied configurations. Historical baseline indexing, generation, and some supplementary controls require their original implementations and execution metadata. The two main prompt settings and stored budget controls are the release's fresh inference scope. [Evaluation guide](../docs/evaluation.md) · [Recorded live validation](../res/live_validation.json)
