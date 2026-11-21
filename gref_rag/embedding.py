from __future__ import annotations

import gc
import json
import time
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .io_utils import dump_json, ensure_dir, load_json

RECOMMENDED_NVEMBED_VERSIONS = {
    "torch": "2.2.0",
    "transformers": "4.42.4",
    "sentence-transformers": "2.7.0",
    "flash-attn": "2.2.0",
}


def _pkg_version(pkg: str) -> str:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "not_installed"
    except Exception:
        return "unknown"


def dependency_report() -> dict[str, str]:
    return {
        "torch": _pkg_version("torch"),
        "transformers": _pkg_version("transformers"),
        "sentence_transformers": _pkg_version("sentence-transformers"),
        "flash_attn": _pkg_version("flash-attn"),
        "numpy": _pkg_version("numpy"),
    }


def require_embedding_model_path(model_path: str) -> str:
    value = str(model_path or "").strip()
    if not value or "$" in value:
        raise ValueError(
            "NVEMBED_MODEL_PATH is unset. Download nvidia/NV-Embed-v2 from "
            "Hugging Face, then export NVEMBED_MODEL_PATH=/path/to/nvidia/NV-Embed-v2. "
        )
    return value


def nvembed_dependency_report() -> dict[str, Any]:
    installed = dependency_report()
    expected = {
        "torch": RECOMMENDED_NVEMBED_VERSIONS["torch"],
        "transformers": RECOMMENDED_NVEMBED_VERSIONS["transformers"],
        "sentence_transformers": RECOMMENDED_NVEMBED_VERSIONS["sentence-transformers"],
        "flash_attn": RECOMMENDED_NVEMBED_VERSIONS["flash-attn"],
    }
    return {
        "installed": installed,
        "recommended_for_nvembed_v2": expected,
        "strict_matches": {
            k: str(installed.get(k, "")).startswith(str(v)) for k, v in expected.items()
        },
    }


def _torch_dtype_from_string(name: str | None):
    if not name or str(name).lower() in {"none", "null", "auto"}:
        return None
    import torch

    table = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = str(name).lower()
    if key not in table:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return table[key]


def _patch_transformers_cache_compat() -> bool:
    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        return False
    if hasattr(DynamicCache, "get_usable_length") or not hasattr(DynamicCache, "get_seq_length"):
        return False

    def get_usable_length(self, new_seq_length=None, layer_idx: int = 0):  # noqa: ANN001
        try:
            return self.get_seq_length(layer_idx)
        except TypeError:
            return self.get_seq_length()

    DynamicCache.get_usable_length = get_usable_length  # type: ignore[attr-defined]
    return True


def _patch_nvembed_bidirectional_mistral_compat(model: Any, logger: Any | None = None) -> bool:
    embedding_model = getattr(model, "embedding_model", None)
    if embedding_model is None or embedding_model.__class__.__name__ != "BidirectionalMistralModel":
        return False
    if getattr(embedding_model, "_gref_rag_forward_compat", False):
        return False
    try:
        import types

        import torch
        from transformers.modeling_attn_mask_utils import (
            _prepare_4d_attention_mask,
            _prepare_4d_attention_mask_for_sdpa,
        )
        from transformers.modeling_outputs import BaseModelOutputWithPast
    except Exception:
        return False

    def forward_compat(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
            device = input_ids.device
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
            device = inputs_embeds.device
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )
        if position_ids is None:
            position_ids = (
                torch.arange(0, seq_length, dtype=torch.long, device=device)
                .unsqueeze(0)
                .view(-1, seq_length)
            )
        else:
            position_ids = position_ids.view(-1, seq_length).long()
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if attention_mask is not None and self._attn_implementation == "flash_attention_2":
            attention_mask = attention_mask if 0 in attention_mask else None
        elif self._attn_implementation == "sdpa" and not output_attentions:
            attention_mask = _prepare_4d_attention_mask_for_sdpa(
                attention_mask, inputs_embeds.dtype
            )
        else:
            attention_mask = _prepare_4d_attention_mask(attention_mask, inputs_embeds.dtype)
        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        position_embeddings = (
            self.rotary_emb(hidden_states, position_ids) if hasattr(self, "rotary_emb") else None
        )
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                output_attentions=output_attentions,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
            if output_attentions:
                all_self_attns += (None,)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if not return_dict:
            return tuple(
                v for v in (hidden_states, None, all_hidden_states, all_self_attns) if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    try:
        embedding_model.config.use_cache = False
    except Exception:
        pass
    embedding_model.forward = types.MethodType(forward_compat, embedding_model)
    embedding_model._gref_rag_forward_compat = True
    if logger:
        logger.log(
            "Applied NV-Embed BidirectionalMistralModel compatibility patch for current transformers API"
        )
    return True


def normalize_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    norms = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return (x / norms).astype(np.float32)


class LocalEmbeddingModel:
    """Local embedding wrapper tuned for NV-Embed-v2.

    Default backend is HuggingFace Transformers because NVIDIA's model card
    documents `AutoModel.from_pretrained(..., trust_remote_code=True)` and
    `model.encode(texts, instruction=..., max_length=...)`. SentenceTransformers
    remains available as a fallback via `backend: sentence_transformers`.
    """

    def __init__(self, cfg: Mapping[str, Any], logger: Any | None = None):
        self.cfg = dict(cfg or {})
        self.logger = logger
        self.enabled = bool(self.cfg.get("enabled", True))
        self.backend = str(self.cfg.get("backend", "transformers") or "transformers").lower()
        self.model_path = str(self.cfg.get("model_path") or self.cfg.get("model_name") or "")
        self.device = self.cfg.get("device") or None
        self.device_map = self.cfg.get("device_map", None)
        self.batch_size = int(self.cfg.get("batch_size", 2))
        self.normalize = bool(self.cfg.get("normalize", self.cfg.get("normalize_embeddings", True)))
        self.query_instruction = str(self.cfg.get("query_instruction", "") or "")
        self.passage_instruction = str(self.cfg.get("passage_instruction", "") or "")
        self.local_files_only = bool(self.cfg.get("local_files_only", True))
        self.trust_remote_code = bool(self.cfg.get("trust_remote_code", True))
        self.max_seq_length = int(self.cfg.get("max_seq_length", 2048) or 2048)
        self.add_eos = bool(self.cfg.get("add_eos", False))
        self.torch_dtype_name = self.cfg.get("torch_dtype") or self.cfg.get("model_kwargs", {}).get(
            "torch_dtype"
        )
        self._model: Any = None
        self._tokenizer: Any = None

    def load(self) -> None:
        if not self.enabled or self._model is not None:
            return
        self.model_path = require_embedding_model_path(self.model_path)
        if self.logger:
            self.logger.log(
                "Loading embedding model: "
                + json.dumps(
                    {
                        "backend": self.backend,
                        "model_path": self.model_path,
                        "device": self.device,
                        "device_map": self.device_map,
                        "dependency_report": nvembed_dependency_report(),
                    },
                    ensure_ascii=False,
                )
            )
        if self.backend in {"transformers", "hf", "auto"}:
            try:
                self._load_transformers()
                return
            except Exception:
                if self.backend != "auto":
                    raise
                if self.logger:
                    self.logger.log(
                        "Transformers backend failed; falling back to SentenceTransformers"
                    )
        self._load_sentence_transformers()

    def _load_transformers(self) -> None:
        from transformers import AutoConfig, AutoModel

        _patch_transformers_cache_compat()
        dtype = _torch_dtype_from_string(self.torch_dtype_name)
        kwargs: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
        }
        config = AutoConfig.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
        )
        try:
            config._name_or_path = self.model_path
            if getattr(config, "text_config", None) is not None:
                config.text_config._name_or_path = self.model_path
        except Exception:
            pass
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        if self.device_map not in {None, "", "null"}:
            kwargs["device_map"] = self.device_map
        self._model = AutoModel.from_pretrained(self.model_path, config=config, **kwargs)
        if self.device and self.device_map in {None, "", "null"}:
            self._model = self._model.to(self.device)
        self._model.eval()
        _patch_nvembed_bidirectional_mistral_compat(self._model, self.logger)
        try:
            self._tokenizer = self._model.tokenizer
            self._tokenizer.padding_side = "right"
        except Exception:
            self._tokenizer = None

    def _load_sentence_transformers(self) -> None:
        from sentence_transformers import SentenceTransformer

        model_kwargs = dict(self.cfg.get("model_kwargs", {}) or {})
        dtype = _torch_dtype_from_string(self.torch_dtype_name)
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        kwargs = {
            "device": self.device,
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
        }
        if model_kwargs:
            kwargs["model_kwargs"] = model_kwargs
        try:
            self._model = SentenceTransformer(self.model_path, **kwargs)
        except TypeError:
            self._model = SentenceTransformer(
                self.model_path,
                device=self.device,
                trust_remote_code=self.trust_remote_code,
            )
        try:
            self._model.max_seq_length = self.max_seq_length
            self._model.tokenizer.padding_side = "right"
        except Exception:
            pass

    def _query_instruction(self) -> str:
        instr = self.query_instruction.strip()
        if not instr:
            return ""
        if instr.startswith("Instruct:"):
            return instr if instr.endswith("Query: ") else instr.rstrip() + "\nQuery: "
        return f"Instruct: {instr}\nQuery: "

    def _eos_token(self) -> str:
        try:
            tok = getattr(self._model, "tokenizer", None) or self._tokenizer
            return tok.eos_token or ""
        except Exception:
            return ""

    def _append_eos(self, texts: Sequence[str]) -> list[str]:
        if not self.add_eos:
            return [str(t) for t in texts]
        eos = self._eos_token()
        return [str(t) + eos for t in texts]

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def _show_progress(self, total_batches: int, progress_desc: str | None) -> bool:
        if total_batches <= 1:
            return False
        default = bool(progress_desc)
        return bool(self.cfg.get("show_progress", self.cfg.get("show_progress_bar", default)))

    def encode(
        self,
        texts: Sequence[str],
        kind: str = "passage",
        is_query: bool | None = None,
        progress_desc: str | None = None,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        if is_query is not None:
            kind = "query" if is_query else "passage"
        self.load()
        assert self._model is not None
        instruction = self._query_instruction() if kind == "query" else self.passage_instruction
        raw_texts = self._append_eos([str(t) for t in texts])
        if self.backend in {"transformers", "hf", "auto"} and hasattr(self._model, "encode"):
            import torch

            outs = []
            total_batches = (len(raw_texts) + self.batch_size - 1) // self.batch_size
            batch_starts = range(0, len(raw_texts), self.batch_size)
            if self._show_progress(total_batches, progress_desc):
                batch_starts = tqdm(
                    batch_starts,
                    total=total_batches,
                    desc=progress_desc or f"embed.{kind}",
                    ncols=100,
                    unit="batch",
                )
            # NV-Embed enters its own CUDA autocast context. Avoid keeping a
            # second copy of every cast weight during a forward pass.
            cuda_cast = (
                torch.autocast("cuda", enabled=False, cache_enabled=False)
                if torch.cuda.is_available()
                else nullcontext()
            )
            with torch.inference_mode(), cuda_cast:
                for i in batch_starts:
                    batch = raw_texts[i : i + self.batch_size]
                    emb = self._model.encode(
                        batch, instruction=instruction, max_length=self.max_seq_length
                    )
                    if hasattr(emb, "detach"):
                        emb = emb.detach().float().cpu().numpy()
                    outs.append(np.asarray(emb, dtype=np.float32))
            arr = np.vstack(outs) if outs else np.zeros((0, 0), dtype=np.float32)
        else:
            # SentenceTransformers uses prompt for query instruction. Passage prompt remains empty.
            prompt = instruction if kind == "query" else None
            kwargs = {
                "batch_size": self.batch_size,
                "convert_to_numpy": True,
                "normalize_embeddings": self.normalize,
                "show_progress_bar": self._show_progress(
                    (len(raw_texts) + self.batch_size - 1) // self.batch_size,
                    progress_desc,
                ),
            }
            if prompt:
                kwargs["prompt"] = prompt
            arr = self._model.encode(raw_texts, **kwargs)
            arr = np.asarray(arr, dtype=np.float32)
        return normalize_matrix(arr) if self.normalize else np.asarray(arr, dtype=np.float32)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode(texts, kind="query")

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode(texts, kind="passage")


def unit_text(row: Mapping[str, Any]) -> str:
    title = str(row.get("title", ""))
    text = str(row.get("text", ""))
    return f"Title: {title}\nText: {text}" if title else text


class DenseIndex:
    def __init__(
        self,
        ids: Sequence[str],
        matrix: np.ndarray,
        id_to_row: Mapping[str, Mapping[str, Any]],
        unit: str,
    ):
        self.ids = list(ids)
        self.matrix = normalize_matrix(np.asarray(matrix, dtype=np.float32))
        self.id_to_row = id_to_row
        self.unit = unit

    @classmethod
    def build(
        cls,
        rows: Sequence[Mapping[str, Any]],
        id_key: str,
        unit: str,
        embedder: LocalEmbeddingModel,
        logger: Any | None = None,
    ) -> DenseIndex:
        ids = [str(r[id_key]) for r in rows]
        texts = [unit_text(r) for r in rows]
        if logger:
            logger.log(
                f"Encoding dense {unit} embeddings: n={len(texts)} batch_size={embedder.batch_size}"
            )
        t0 = time.perf_counter()
        if logger and hasattr(logger, "time_block"):
            with logger.time_block(
                "dense.encode", unit=unit, n=len(texts), batch_size=embedder.batch_size
            ):
                mat = embedder.encode(texts, kind="passage", progress_desc=f"dense.{unit}")
        else:
            mat = embedder.encode(texts, kind="passage", progress_desc=f"dense.{unit}")
        if logger:
            dim = int(mat.shape[1]) if mat.ndim == 2 and mat.shape[0] else 0
            logger.log(
                f"Encoded dense {unit} embeddings: n={len(texts)} dim={dim} "
                f"seconds={time.perf_counter() - t0:.3f}"
            )
        return cls(
            ids=ids,
            matrix=mat,
            id_to_row={str(r[id_key]): dict(r) for r in rows},
            unit=unit,
        )

    @staticmethod
    def exists(out_dir: str | Path, name: str) -> bool:
        p = Path(out_dir)
        return (p / f"{name}_embeddings.npy").exists() and (p / f"{name}_ids.json").exists()

    def save(self, out_dir: str | Path, name: str) -> None:
        out = ensure_dir(out_dir)
        np.save(out / f"{name}_embeddings.npy", self.matrix)
        dump_json({"unit": self.unit, "ids": self.ids}, out / f"{name}_ids.json")

    @classmethod
    def load(
        cls,
        out_dir: str | Path,
        name: str,
        rows: Sequence[Mapping[str, Any]],
        id_key: str,
        unit: str,
    ) -> DenseIndex:
        p = Path(out_dir)
        mat = np.load(p / f"{name}_embeddings.npy", mmap_mode=None)
        meta = load_json(p / f"{name}_ids.json")
        return cls(
            ids=meta["ids"],
            matrix=mat,
            id_to_row={str(r[id_key]): dict(r) for r in rows},
            unit=unit,
        )

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if self.matrix.size == 0 or not self.ids:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        if q.ndim == 2:
            q = q[0]
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        sims = self.matrix @ q
        k = min(int(top_k), len(sims))
        if k <= 0:
            return []
        idx = np.argpartition(-sims, k - 1)[:k]
        pairs = [(self.ids[int(i)], float(sims[int(i)])) for i in idx]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs


def build_or_load_dense_indexes(
    index: Mapping[str, Any],
    cfg: Mapping[str, Any],
    index_dir: str | Path,
    logger: Any | None = None,
    force: bool = False,
) -> dict[str, DenseIndex]:
    dense_cfg = cfg.get("dense", {})
    if not dense_cfg.get("enabled", False):
        return {}
    embedder = LocalEmbeddingModel(cfg.get("embedding", {}), logger=logger)
    dense_dir = ensure_dir(Path(index_dir) / "dense")
    units = dense_cfg.get("units", ["proposition", "chunk"])
    out: dict[str, DenseIndex] = {}
    try:
        if "proposition" in units:
            if not force and DenseIndex.exists(dense_dir, "proposition"):
                if logger:
                    logger.log(f"Loading dense proposition index: {dense_dir}")
                out["proposition"] = DenseIndex.load(
                    dense_dir,
                    "proposition",
                    index.get("propositions", []),
                    "prop_id",
                    "proposition",
                )
            else:
                if logger:
                    logger.log("Building dense proposition index")
                out["proposition"] = DenseIndex.build(
                    index.get("propositions", []),
                    "prop_id",
                    "proposition",
                    embedder,
                    logger,
                )
                out["proposition"].save(dense_dir, "proposition")
                if logger:
                    logger.log(f"Saved dense proposition index: {dense_dir}")
        if "chunk" in units:
            if not force and DenseIndex.exists(dense_dir, "chunk"):
                if logger:
                    logger.log(f"Loading dense chunk index: {dense_dir}")
                out["chunk"] = DenseIndex.load(
                    dense_dir, "chunk", index.get("chunks", []), "chunk_id", "chunk"
                )
            else:
                if logger:
                    logger.log("Building dense chunk index")
                out["chunk"] = DenseIndex.build(
                    index.get("chunks", []), "chunk_id", "chunk", embedder, logger
                )
                out["chunk"].save(dense_dir, "chunk")
                if logger:
                    logger.log(f"Saved dense chunk index: {dense_dir}")
        dump_json(
            {
                "enabled": True,
                "units": list(out.keys()),
                "model_path": cfg.get("embedding", {}).get("model_path"),
            },
            dense_dir / "dense_meta.json",
        )
    finally:
        embedder.close()
    return out


def encode_query(
    cfg: Mapping[str, Any],
    query: str,
    logger: Any | None = None,
    cache: dict[str, LocalEmbeddingModel] | None = None,
) -> np.ndarray:
    emb_cfg = cfg.get("embedding", {})
    key = json.dumps({k: str(v) for k, v in emb_cfg.items() if k != "model_kwargs"}, sort_keys=True)
    if cache is not None and key in cache:
        embedder = cache[key]
    else:
        embedder = LocalEmbeddingModel(emb_cfg, logger=logger)
        if cache is not None:
            cache[key] = embedder
    return embedder.encode([query], kind="query")
