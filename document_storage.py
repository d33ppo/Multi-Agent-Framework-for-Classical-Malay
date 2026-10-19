"""
Storage for uploaded documents, rendered pages, OCR output and run records.

Callers only use the small Storage interface, so the local folder used today
can be replaced by Supabase Storage later without changing them.

Saves are write-once by default. That is how the original OCR output is kept
safe: once a key is written, a second save to it fails unless the caller asks
for overwrite (only the run status snapshot does).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

from llm_text_runtime import env_value

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_STORAGE_DIR = "storage"


class Storage(Protocol):
    def save(self, key: str, data: bytes, *, overwrite: bool = False) -> None: ...

    def load(self, key: str) -> bytes: ...

    def exists(self, key: str) -> bool: ...


class LocalStorage:
    """Files under one root folder. Keys are relative paths with forward slashes."""

    def __init__(self, root: str | Path | None = None):
        root = Path(root or env_value("STORAGE_DIR", DEFAULT_STORAGE_DIR))
        self.root = (root if root.is_absolute() else PROJECT_DIR / root).resolve()

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if self.root not in path.parents:
            raise ValueError(f"Storage key is outside the storage folder: {key!r}")
        return path

    def save(self, key: str, data: bytes, *, overwrite: bool = False) -> None:
        path = self._path(key)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Storage key already exists and is write-once: {key}")
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temporary file first so a reader never sees a half-written file.
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)

    def load(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()
