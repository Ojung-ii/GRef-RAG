"""HippoRAG2 benchmark files and evaluation-only annotations."""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .io_utils import atomic_output
from .text import normalize_text


@dataclass
class CorpusDoc:
    doc_id: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QAExample:
    id: str
    question: str
    answers: list[str]
    support_titles: list[str] = field(default_factory=list)
    support_facts: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _read(path: str | Path) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing data file: {p}")
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if p.suffix.lower() == ".jsonl":
        return [json.loads(x) for x in text.splitlines() if x.strip()]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(x) for x in text.splitlines() if x.strip()]


def _rows(obj: Any) -> list[Any]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for k in ("data", "examples", "rows"):
            if isinstance(obj.get(k), list):
                return obj[k]
        return list(obj.values())
    return list(obj)


def _answers(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        for fn in (json.loads, ast.literal_eval):
            try:
                o = fn(s)
                if isinstance(o, list):
                    return [str(x) for x in o if str(x).strip()]
                if isinstance(o, str):
                    return [o]
            except Exception:
                pass
        return [x.strip() for x in s.split("|") if x.strip()] if "|" in s else [s]
    return [str(v)]


def _unique_docs(rows: Iterable[Mapping[str, Any]], prefix: str) -> list[CorpusDoc]:
    docs = []
    seen = set()
    for i, r in enumerate(rows):
        title = normalize_text(
            r.get("title") or r.get("wiki_title") or r.get("name") or f"{prefix}_{i}"
        )
        text = normalize_text(
            r.get("text")
            or r.get("body")
            or r.get("context")
            or r.get("paragraph_text")
            or r.get("paragraph")
            or ""
        )
        if not text:
            continue
        key = (title, text)
        if key in seen:
            continue
        seen.add(key)
        docs.append(
            CorpusDoc(
                str(r.get("id") or r.get("doc_id") or r.get("idx") or f"{prefix}_{len(docs)}"),
                title,
                text,
                {
                    k: v
                    for k, v in r.items()
                    if k not in {"text", "body", "context", "paragraph_text", "paragraph"}
                },
            )
        )
    return docs


def _load_corpus(path: str | Path | None, corpus_limit: int | None, prefix: str) -> list[CorpusDoc]:
    if not path:
        raise ValueError("A separate corpus file is required.")
    rows = _rows(_read(path))
    rows = rows[:corpus_limit] if corpus_limit else rows
    return _unique_docs(rows, prefix)


def load_popqa(
    cfg: Mapping[str, Any],
    limit: int | None = None,
    corpus_limit: int | None = None,
):
    qa_rows = _rows(_read(cfg.get("qa_path") or "data/popqa/popqa.json"))
    qa_rows = qa_rows[:limit] if limit else qa_rows
    docs = _load_corpus(cfg.get("corpus_path"), corpus_limit, "popqa_corpus")
    qas = []
    for i, r in enumerate(qa_rows):
        support = []
        for p in r.get("paragraphs", []) or []:
            if p.get("is_supporting") and p.get("title"):
                support.append(normalize_text(p.get("title")))
        if not support:
            for k in ("s_wiki_title", "o_wiki_title"):
                if r.get(k):
                    support.append(normalize_text(r[k]))
        qas.append(
            QAExample(
                str(r.get("id") or i),
                normalize_text(r.get("question", "")),
                _answers(r.get("possible_answers") or r.get("answers") or r.get("answer")),
                list(dict.fromkeys(support)),
                [],
                dict(r),
            )
        )
    return qas, docs


def load_hotpotqa(
    cfg: Mapping[str, Any],
    limit: int | None = None,
    corpus_limit: int | None = None,
):
    rows = _rows(_read(cfg.get("qa_path") or "data/hotpotqa/hotpotqa.json"))
    rows = rows[:limit] if limit else rows
    qas = []
    for i, r in enumerate(rows):
        sf = r.get("supporting_facts", []) or []
        titles = []
        for x in sf:
            if isinstance(x, (list, tuple)) and x:
                titles.append(normalize_text(x[0]))
            elif isinstance(x, Mapping) and x.get("title"):
                titles.append(normalize_text(x["title"]))
        qas.append(
            QAExample(
                str(r.get("_id") or r.get("id") or i),
                normalize_text(r.get("question", "")),
                _answers(r.get("answer") or r.get("answers")),
                list(dict.fromkeys(titles)),
                sf,
                dict(r),
            )
        )
    docs = _load_corpus(cfg.get("corpus_path"), corpus_limit, "hotpotqa_corpus")
    return qas, docs


def load_2wiki(
    cfg: Mapping[str, Any],
    limit: int | None = None,
    corpus_limit: int | None = None,
):
    rows = _rows(_read(cfg.get("qa_path") or "data/2wiki/2wikimultihopqa.json"))
    rows = rows[:limit] if limit else rows
    qas = []
    for i, r in enumerate(rows):
        titles = []
        for key in ("supporting_facts", "evidences", "supporting_titles"):
            for x in r.get(key) or []:
                if isinstance(x, str):
                    titles.append(normalize_text(x))
                elif isinstance(x, (list, tuple)) and x:
                    titles.append(normalize_text(x[0]))
                elif isinstance(x, Mapping) and x.get("title"):
                    titles.append(normalize_text(x["title"]))
        qas.append(
            QAExample(
                str(r.get("_id") or r.get("id") or i),
                normalize_text(r.get("question", "")),
                _answers(r.get("answer") or r.get("answers")),
                list(dict.fromkeys(titles)),
                r.get("supporting_facts", []) or [],
                dict(r),
            )
        )
    docs = _load_corpus(cfg.get("corpus_path"), corpus_limit, "2wiki_corpus")
    return qas, docs


def load_musique(
    cfg: Mapping[str, Any],
    limit: int | None = None,
    corpus_limit: int | None = None,
):
    rows = _rows(_read(cfg.get("qa_path") or "data/musique/musique.json"))
    rows = rows[:limit] if limit else rows
    qas = []
    for i, r in enumerate(rows):
        titles = []
        for j, p in enumerate(r.get("paragraphs") or r.get("context") or []):
            if not isinstance(p, Mapping):
                continue
            title = str(p.get("title") or p.get("idx") or f"para_{j}")
            if p.get("is_supporting") or p.get("supporting"):
                titles.append(normalize_text(title))
        ans = _answers(r.get("answer") or r.get("answers"))
        for a in _answers(r.get("answer_aliases")):
            if a not in ans:
                ans.append(a)
        qas.append(
            QAExample(
                str(r.get("id") or r.get("_id") or i),
                normalize_text(r.get("question", "")),
                ans,
                list(dict.fromkeys(titles)),
                [],
                dict(r),
            )
        )
    docs = _load_corpus(cfg.get("corpus_path"), corpus_limit, "musique_corpus")
    return qas, docs


DATASETS = {
    "hotpotqa": "hotpotqa",
    "2wiki": "2wikimultihopqa",
    "musique": "musique",
    "popqa": "popqa",
}
HIPPO_REVISION = "488c53fb27dd28f70412e18004bd49f791600574"
TWOWIKI_ARCHIVE_URL = "https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?dl=1"
TWOWIKI_ARCHIVE_SHA256 = "95df2bf56fdabe034e27aebc580e02264232203cf52552f9efe8a919e5529eef"
TWOWIKI_ALIASES_SHA256 = "f08ffcb6c2cefca9bdbe86b4248d6ad7a7743762d3f7264c14ff0bae85726fb6"
LOADERS = {
    "hotpotqa": load_hotpotqa,
    "2wiki": load_2wiki,
    "musique": load_musique,
    "popqa": load_popqa,
}
MANIFESTS = Path(__file__).resolve().parents[1] / "manifests"


def data_paths(root: Path, dataset: str) -> tuple[Path, Path]:
    name = DATASETS[dataset]
    return root / dataset / f"{name}.json", root / dataset / f"{name}_corpus.json"


def read_benchmark(root: Path, dataset: str, *, verify: bool = True):
    qa_path, corpus_path = data_paths(root, dataset)
    manifest = _read(MANIFESTS / "data_manifest.json")["datasets"][dataset]
    if verify:
        for path, key in ((qa_path, "qa_sha256"), (corpus_path, "corpus_sha256")):
            if file_sha256(path) != manifest[key]:
                raise ValueError(f"Input checksum mismatch for {path}")
    examples, corpus = LOADERS[dataset]({"qa_path": qa_path, "corpus_path": corpus_path})
    if not corpus:
        raise ValueError(f"Empty retrieval corpus: {corpus_path}")
    by_id = {example.id: example for example in examples}
    if len(by_id) != len(examples):
        raise ValueError(f"Duplicate query IDs in {qa_path}")
    if verify:
        query_ids = (MANIFESTS / "query_ids" / f"{dataset}.txt").read_text().splitlines()
        order = json.dumps(query_ids, ensure_ascii=False).encode("utf-8")
        ids = json.dumps(sorted(by_id), ensure_ascii=False).encode("utf-8")
        if (
            set(query_ids) != by_id.keys()
            or len(query_ids) != len(examples)
            or hashlib.sha256(order).hexdigest() != manifest["query_order_hash"]
            or hashlib.sha256(ids).hexdigest() != manifest["query_id_set_hash"]
        ):
            raise ValueError(f"Unexpected query IDs in {qa_path}")
        examples = [by_id[qid] for qid in query_ids]
    return examples, corpus


def apply_official_aliases(
    examples: list[QAExample], dataset: str, policy: str, data_root: Path
) -> None:
    """Resolve the paper gold answers from the verified release before scoring."""
    if policy == "hipporag2":
        return
    if policy != "paper":
        raise ValueError(f"Unknown gold policy {policy!r}")
    if dataset == "hotpotqa":
        for example in examples:
            example.answers = [example.metadata["answer"]]
    elif dataset == "musique":
        for example in examples:
            row = example.metadata
            example.answers = [row["answer"], *row["answer_aliases"]]
    elif dataset == "popqa":
        for example in examples:
            value = example.metadata.get("possible_answers")
            example.answers = json.loads(value) if isinstance(value, str) else value
    elif dataset == "2wiki":
        aliases = data_root / "2wiki" / "id_aliases.json"
        if not aliases.is_file():
            raise FileNotFoundError(
                f"{aliases} is required for paper gold. Run 'python -m gref_rag data --dataset 2wiki --with-2wiki-aliases'."
            )
        if file_sha256(aliases) != TWOWIKI_ALIASES_SHA256:
            raise ValueError(f"2Wiki official aliases checksum mismatch: {aliases}")
        with aliases.open(encoding="utf-8") as stream:
            by_answer_id = {row["Q_id"]: row for row in map(json.loads, filter(str.strip, stream))}
        for example in examples:
            row = example.metadata
            entry = by_answer_id.get(row["answer_id"], {})
            example.answers = [
                row["answer"],
                *entry.get("aliases", []),
                *entry.get("demonyms", []),
            ]


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _download(url: str, path: Path, expected_hash: str) -> None:
    import shutil
    from tempfile import NamedTemporaryFile
    from urllib.request import urlopen

    if path.is_file() and file_sha256(path) == expected_hash:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent) as temporary:
        with urlopen(url, timeout=120) as response:
            shutil.copyfileobj(response, temporary)
        temporary.flush()
        if file_sha256(Path(temporary.name)) != expected_hash:
            raise ValueError(f"Download checksum mismatch: {url}")
        # Preserve existing verified files if the network or checksum check fails.
        with atomic_output(path, "wb") as target:
            temporary.seek(0)
            shutil.copyfileobj(temporary, target)


def download_inputs(root: Path, datasets: list[str], with_aliases: bool = False) -> None:
    import shutil
    from zipfile import ZipFile

    specs = _read(MANIFESTS / "data_manifest.json")["datasets"]
    for dataset in datasets:
        qa_path, corpus_path = data_paths(root, dataset)
        for path, key in ((qa_path, "qa_sha256"), (corpus_path, "corpus_sha256")):
            url = f"https://huggingface.co/datasets/osunlp/HippoRAG_2/resolve/{HIPPO_REVISION}/{path.name}"
            _download(url, path, specs[dataset][key])
    if with_aliases and "2wiki" in datasets:
        alias_file = root / "2wiki" / "id_aliases.json"
        if alias_file.is_file() and file_sha256(alias_file) == TWOWIKI_ALIASES_SHA256:
            return
        archive = root / "2wiki" / "data_ids_april7.zip"
        _download(TWOWIKI_ARCHIVE_URL, archive, TWOWIKI_ARCHIVE_SHA256)
        with ZipFile(archive) as source, source.open("id_aliases.json") as stream:
            with atomic_output(alias_file, "wb", expected_hash=TWOWIKI_ALIASES_SHA256) as target:
                shutil.copyfileobj(stream, target)
