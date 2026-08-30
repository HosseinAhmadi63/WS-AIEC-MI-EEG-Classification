"""Atomic, deterministic artifact writing."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _atomic_replace_bytes(path: Path, payload: bytes) -> None:
    ensure_directory(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: str | Path, payload: Any) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    _atomic_replace_bytes(Path(path), encoded)


def write_csv(path: str | Path, frame: pd.DataFrame) -> None:
    _atomic_replace_bytes(Path(path), frame.to_csv(index=False, lineterminator="\n").encode("utf-8"))


def read_csv_if_exists(path: str | Path) -> pd.DataFrame:
    target = Path(path)
    return pd.read_csv(target) if target.exists() else pd.DataFrame()
