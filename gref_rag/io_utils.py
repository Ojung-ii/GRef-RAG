from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import re
from collections.abc import Iterable, Mapping, MutableMapping
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _expand_env(obj: Any) -> Any:
    if isinstance(obj, str):

        def repl(m: re.Match[str]) -> str:
            return os.environ.get(m.group(1), m.group(2))

        obj = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)[:-]-(.*?)\}", repl, obj)
        return os.path.expandvars(obj)
    if isinstance(obj, list):
        return [_expand_env(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    return obj


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = _expand_env(yaml.safe_load(f) or {})
    if isinstance(data, dict) and data.get("extends"):
        base_path = Path(str(data.pop("extends")))
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        base = load_yaml(base_path)
        return dict(deep_update(base, data))
    return data


def to_jsonable(obj: Any) -> Any:
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
    except Exception:
        pass
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x) for x in obj]
    return obj


def dump_json(obj: Any, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, ensure_ascii=False, indent=2)


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def deep_update(
    base: MutableMapping[str, Any], override: Mapping[str, Any]
) -> MutableMapping[str, Any]:
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), MutableMapping):
            deep_update(base[k], v)  # type: ignore[index]
        else:
            base[k] = v
    return base


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@contextlib.contextmanager
def atomic_output(path: Path, mode: str = "w", *, expected_hash: str | None = None):
    """Publish a completed output file with a same-directory atomic rename."""
    import hashlib
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {} if "b" in mode else {"encoding": "utf-8"}
    with tempfile.NamedTemporaryFile(mode=mode, dir=path.parent, delete=False, **kwargs) as stream:
        temporary = Path(stream.name)
        try:
            yield stream
            stream.close()
            if expected_hash is not None:
                with temporary.open("rb") as data:
                    if hashlib.file_digest(data, "sha256").hexdigest() != expected_hash:
                        raise ValueError(f"Output checksum mismatch: {path}")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
