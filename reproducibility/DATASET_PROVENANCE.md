# Dataset provenance

The main evaluation uses all 1,000 questions in each of four QA files in the pinned HippoRAG 2 release. The files are subsets or benchmark-specific releases relative to the upstream benchmarks. The paper runs apply no further question subsampling or filtering. The separate corpus file is the retrieval collection for a cohort. [Direct downloads by dataset](../README.md#data) · [Pinned HippoRAG 2 release](https://huggingface.co/datasets/osunlp/HippoRAG_2/tree/488c53fb27dd28f70412e18004bd49f791600574)

The released query manifests preserve the experiment order and canonical IDs. The loader checks the byte checksums, unique ID set, and canonical sequence before indexing or evaluation. [Machine-readable input manifest](../manifests/data_manifest.json) · [Query ID manifests](../manifests/query_ids)

| Cohort | Upstream source | QA file and questions | Retrieval file and raw corpus records |
| --- | --- | --- | --- |
| HotpotQA | [Dataset site](https://hotpotqa.github.io/) | `hotpotqa.json` · 1,000 | `hotpotqa_corpus.json` · 9,811 |
| 2WikiMultiHopQA | [Official release](https://github.com/Alab-NII/2wikimultihop) | `2wikimultihopqa.json` · 1,000 | `2wikimultihopqa_corpus.json` · 6,119 |
| MuSiQue | [Official release](https://github.com/StonyBrookNLP/musique) | `musique.json` · 1,000 | `musique_corpus.json` · 11,656 |
| PopQA | [Dataset depositor](https://huggingface.co/datasets/akariasai/PopQA) | `popqa.json` · 1,000 | `popqa_corpus.json` · 8,676 |

The main MuSiQue file contains 11,656 raw corpus entries. The release loader emits 11,655 usable corpus units. The 1,000-question MuSiQue evaluation cohort is intact.

| File | SHA-256 |
| --- | --- |
| `hotpotqa.json` | `3ad9c0bcbf93f41d7004ca6007049c904d2605046b314a2f7ecfb379c64cba6d` |
| `hotpotqa_corpus.json` | `9333647b922382776cd2cb02893b390d77984df85a91bfa8be411284978aca7d` |
| `2wikimultihopqa.json` | `895cba294064df0c3302c76847b1fc08d99b5619f7663dfaa3b65cd780f1cac4` |
| `2wikimultihopqa_corpus.json` | `9d6e352952aafb18dab22bf8195039461321a44a949df902ae83bce134ad238a` |
| `musique.json` | `98ed4e21d3076532f6388d42320fb809599c63a0d8dffca8ece5e41922be6b46` |
| `musique_corpus.json` | `73157a03ce3f0b1a5673dd5dc12bb970c24976dbffc688af9eecdd758c97ffcb` |
| `popqa.json` | `e0874255f9acff5f2de0a7b8ab9782558db31d9a364fee6c84519510955880b2` |
| `popqa_corpus.json` | `16ccbb71d84c5c6e6d4073ef9b860a79f8f2c3869ce8f1ade85d3285c440ccd8` |
| `data_ids_april7.zip` | `95df2bf56fdabe034e27aebc580e02264232203cf52552f9efe8a919e5529eef` |
| `id_aliases.json` | `f08ffcb6c2cefca9bdbe86b4248d6ad7a7743762d3f7264c14ff0bae85726fb6` |

## Answer references

- HotpotQA uses its scalar answer.
- 2WikiMultiHopQA evaluates the scalar answer and the official v1.1 answer-ID aliases and demonyms. The two alias files listed above are acquired from the official April 7 release. [2Wiki update and evaluator](https://github.com/Alab-NII/2wikimultihop#update-on-december-8-2020)
- MuSiQue uses the scalar answer and released aliases.
- PopQA uses the released set of possible answers. The final common and native GRef-RAG PopQA retrieval inputs use the question and corpus.

The Wikipedia cohorts use maximum EM and token F1 over the accepted references, with the paper evaluator's normalization. Evaluation references and support annotations are scored after retrieval and QA. [Evaluator and retrieval adapters](../docs/evaluation.md)

```bash
python -m gref_rag data --with-2wiki-aliases
python -m gref_rag data --verify-only --with-2wiki-aliases
```

## MultiHop-RAG news

The news study uses 2,556 questions and a 609-document collection from the observed Hugging Face revision `71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82`. All 2,556 questions in that snapshot are represented in the released news score ledger. The archived adapter preserves the original row index and assigns a stable `mhrag_` ID. The paper uses the news run's official answer references and scorer. [Pinned input files](https://huggingface.co/datasets/yixuantt/MultiHopRAG/tree/71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82) · [Benchmark evaluation](https://github.com/yixuantt/MultiHop-RAG#evaluation)

| File | Role | Records | SHA-256 |
| --- | --- | ---: | --- |
| `MultiHopRAG.json` | Original QA file | 2,556 | `03cfb4926461f868684903aadc8024447bdda5bb3f6804741424cce338515bff` |
| `corpus.json` | Original news corpus | 609 | `20b61b5ab84de84a927420c5d265b7ec8d859ae49980699958a787ade9e4d28f` |
| `multihoprag.json` | Archived adapted QA | 2,556 | `512dd1c2b6a2b75848c8ee572251b5b0f0389812b2a01278d45105e3b1981696` |
| `multihoprag_corpus.json` | Archived adapted retrieval corpus | 609 | `360948a38ca5b8c402aae2bb17dcc64d1e0e514af72b432745e0a0efb43c7351` |

```bash
python -m gref_rag news-data
python -m gref_rag news-data --verify-only
```

This command obtains the two upstream news files. CPU verification uses [the released news score ledger](../res/supplementary/news_query_scores.csv) and does not require the raw news corpus or the historical adapter outputs. [News input manifest](../manifests/news_data_manifest.json) · [News query IDs](../manifests/query_ids/news.txt)

## Source terms and distribution

The release contains query IDs and derived numerical metrics. Obtain benchmark text, aliases, corpora, and models from their original publishers.

| Source | Publisher's stated terms or verification status |
| --- | --- |
| HotpotQA | CC BY-SA 4.0. [Publisher terms](https://hotpotqa.github.io/#getting-started) |
| 2WikiMultiHopQA | The official source repository shows Apache-2.0. The linked dataset archive is obtained separately. [Repository license](https://github.com/Alab-NII/2wikimultihop#license) |
| MuSiQue | CC BY 4.0 for the dataset. [Publisher terms](https://github.com/StonyBrookNLP/musique#data) |
| PopQA | Separate redistribution terms were unverified in the dataset card consulted. [Dataset card](https://huggingface.co/datasets/akariasai/PopQA) |
| MultiHop-RAG | ODC-BY for the database as declared upstream. Consult the source terms for the news articles. [Publisher terms](https://github.com/yixuantt/MultiHop-RAG#license) |
