from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional

import pandas as pd
import yaml


def ensure_dir(path: str | Path) -> Path:
    """
    Ensure a directory exists.
    If 'path' looks like a file path, create its parent directory.
    If 'path' is a directory path, create it directly.
    """
    p = Path(path)
    # Heuristic: if suffix exists, treat it as a file path
    target_dir = p.parent if p.suffix else p
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def resolve_path(path_str: str | Path, project_root: str | Path) -> Path:
    """
    Resolve a possibly relative path against the project root.
    """
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (Path(project_root) / p).resolve()


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML file into a dictionary.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    """
    Save an object as JSON.
    """
    p = Path(path)
    ensure_dir(p)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def load_json(path: str | Path) -> Any:
    """
    Load a JSON file.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str | Path, items: Iterable[Dict[str, Any]]) -> None:
    """
    Write an iterable of dicts to a JSONL file.
    """
    p = Path(path)
    ensure_dir(p)
    with p.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """
    Read a JSONL file into a list of dicts.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL file not found: {p}")
    records: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def iter_jsonl(path: str | Path) -> Generator[Dict[str, Any], None, None]:
    """
    Stream a JSONL file line by line.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSONL file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def read_csv_in_chunks(
    path: str | Path,
    chunksize: int = 10000,
    usecols: Optional[List[str]] = None,
    dtype: Optional[Dict[str, Any]] = None,
) -> Generator[pd.DataFrame, None, None]:
    """
    Stream a CSV file in chunks to reduce memory usage.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV file not found: {p}")

    yield from pd.read_csv(
        p,
        chunksize=chunksize,
        usecols=usecols,
        dtype=dtype,
        low_memory=False,
    )