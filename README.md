# GRef-RAG

**Is Broader Retrieval Always Better for GraphRAG?**

Junghyun Oh, Jin-woo Lee, and Sungsu Lim

GRef-RAG (Graph Retrieval with Evidence Refinement for RAG) retrieves document anchors, chunks, and sentences in **Global Seed Search**. **Local Refinement** expands mention-linked evidence chains, assembles evidence bundles, and reranks them using semantic coverage before rendering the QA context.

![The GRef-RAG framework from the camera-ready paper](docs/framework.png)

This release runs GRef-RAG in the common-prompt and method-native settings on the four Wikipedia-based cohorts. It includes fixed-cohort baseline scoring adapters and saved scores for replaying the main comparisons. The evaluation guide describes how these artifacts correspond to manuscript results. [Reproduction scope](docs/evaluation.md#reproduction-scope)

## Installation

Use Python 3.11. The reference client environment used PyTorch 2.6.0 with CUDA 12.4 and Transformers 4.57.6.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

export NVEMBED_MODEL_PATH="$(hf download nvidia/NV-Embed-v2 \
  --revision 3fa59658547db50a1e8e3346cf057fd0c77ed6ef)"
```

Run Qwen2.5-7B-Instruct with an OpenAI-compatible chat-completions endpoint. For the paper configuration, install the serving requirements in a separate environment and start vLLM.

```bash
pip install -r requirements-vllm.txt
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --host 127.0.0.1 --port 8011 --max-model-len 16500 \
  --gpu-memory-utilization 0.35
```

In the client environment, set `GREF_RAG_LLM_BASE_URL` if the endpoint differs from `http://127.0.0.1:8011/v1`. Use `VLLM_API_KEY` if the endpoint requires authentication. Set `CUDA_VISIBLE_DEVICES` separately for the two processes, or run retrieval and generation in separate phases on one GPU.

Pass `NVEMBED_BATCH_SIZE=8` to the `index` command if the default batch of 32 exceeds available GPU memory. Keep the embedding model, precision, and batch size in your run record when comparing newly built indexes.

## Data

Download the same four benchmark files and corpora from the [HippoRAG 2 dataset on Hugging Face](https://huggingface.co/datasets/osunlp/HippoRAG_2/tree/488c53fb27dd28f70412e18004bd49f791600574). The command below populates `data/{hotpotqa,2wiki,musique,popqa}/` and verifies the input checksums and each ordered 1,000-query cohort.

```bash
python -m gref_rag data --with-2wiki-aliases
```

Use `--dataset hotpotqa` to download a single cohort. The paper evaluator uses 2Wiki's official answer-ID aliases and demonyms in addition to the answer field. The `--with-2wiki-aliases` option downloads the [official 2Wiki v1.1 data archive](https://github.com/Alab-NII/2wikimultihop) and extracts the verified alias file. Dataset contents, indexes, and model weights stay in local storage. Use `--data-root` or `GREF_RAG_DATA_DIR` to point to a different directory.

## Run the experiments

Build one index per dataset, then reuse it for both prompt settings.

```bash
python -m gref_rag index --dataset hotpotqa --index-dir indexes/hotpotqa

python -m gref_rag run --dataset hotpotqa --index-dir indexes/hotpotqa \
  --config configs/common.yaml --output outputs/hotpotqa-common

python -m gref_rag run --dataset hotpotqa --index-dir indexes/hotpotqa \
  --config configs/native.yaml --output outputs/hotpotqa-native
```

Add `--limit 10` to run the first ten manifest questions. Pass `--retrieval-only --save-context` to run the search stage without a generation server. Pass the resulting `predictions.jsonl` as `--contexts` to the `generate` command with the same dataset, config, and limit. Use `--save-evidence` to export the ranked evidence bundles for provenance analysis.

| Setting | QA prompt | Paper token count |
| --- | --- | --- |
| Common | Shared short-answer prompt with Qwen, temperature 0, top-p 1, and a 64-token output cap | Rendered context including evidence headers and references |
| Native | GRef-RAG evidence-aware prompt and a question-only choice constraint for entity-choice comparisons | Full chat-template model input |

Native reproduction uses the exact historical role string and per-dataset renderers from the executed runs. The appendix updates the role string to the final method name. Set `generation.native_role: GRef-RAG` to use that wording. Full settings and baseline adapter inputs are in [the evaluation guide](docs/evaluation.md).

## Evaluation and results

Each run writes `predictions.jsonl`, `metrics.json`, and `run_manifest.json`. The evaluator reports EM, token-level F1, source recall, context and full-input tokens, F1 per 1,000 context tokens, online latency, and index-build time where measured. JSON EM/F1 are fractions. Index-build timing is measured by `index` and reused with the same index.

To replay the main paired F1 comparisons on a CPU, install `requirements.txt` and run `python -m gref_rag statistics --output outputs/main-paired-statistics.csv`. This replay uses the released score ledger and fixed query IDs.

Use `evaluate` with `--recall-adapter` to score saved baseline predictions, or `generate` with the common configuration to apply the same QA prompt to saved baseline contexts. Align incoming artifacts to the fixed benchmark IDs. See [adapter formats and comparison caveats](docs/evaluation.md) and [official results and parity checks](res/README.md).

| GRef-RAG | Common EM / F1 (%) | Context tokens | Native EM / F1 (%) | Full-input tokens |
| --- | ---: | ---: | ---: | ---: |
| HotpotQA | 32.4 / 39.3 | 480 | 50.7 / 62.9 | 734 |
| 2Wiki | 15.2 / 17.3 | 484 | 49.2 / 55.5 | 761 |
| MuSiQue | 5.8 / 8.3 | 558 | 20.1 / 29.1 | 1015 |
| PopQA | 40.3 / 50.2 | 522 | 47.7 / 60.6 | 1014 |

The PopQA native row uses the question-only rerun on the same 1,000 queries. The camera-ready archive reports 47.2 EM, 60.54 F1, and 988.006 full-input tokens from the earlier run, which allowed benchmark subject names and Wikipedia titles to seed retrieval. The common-prompt PopQA result already uses question-only retrieval. [PopQA correction and validation](res/README.md#live-validation)

## Code map

| Module | Purpose |
| --- | --- |
| `graph.py`, `index.py`, `embedding.py` | Corpus graph, mention edges, dense index construction, and NV-Embed-v2 |
| `retrieval/global_seed_search.py` | BM25, dense, and document-anchor rank fusion |
| `retrieval/local_refinement.py`, `retrieval/semantic_coverage.py` | Chain Expansion and Bundle Reranking |
| `rendering.py`, `prompts.py`, `generation.py` | Context rendering, QA prompts, and generation |
| `evaluation/metrics.py`, `evaluation/recall.py` | Shared QA/cost metrics and method-specific recall |
| `pipeline.py`, `__main__.py` | Integration and experiment CLI |

The release is licensed under [MIT](LICENSE). External models and datasets carry their own terms. See [third-party resources](docs/evaluation.md#third-party-resources).
