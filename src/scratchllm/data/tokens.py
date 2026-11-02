"""Turning the cleaned text into token files the training loop can read.

Each source gets one binary file per split: the token ids of all its documents, one after the
other, as unsigned 16-bit integers (the vocabulary is 16,384 ids, so two bytes are enough and a
file is half the size of int32), with the `<|endoftext|>` id written after every document. That
id is the only mark of where a document ends, and it is what lets the model learn that a text
can stop. Files are read later with `numpy.memmap`, so the operating system pages in only the
windows that are used and the whole corpus never has to fit in RAM.

Which splits exist depends on the role of the source (D-002): a training source gets `train`,
`val` and `test`; an evaluation-only source gets a single `eval` file holding every document,
whatever label the sampler gave it, so it can never reach training by accident.

Next to the files goes `<source>.tokens.json`: how many documents, characters and tokens each
file holds, its sha256, and which tokenizer (by sha256) and which cleaning rules made it. The
file is the proof that two runs trained on the same tokens.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from scratchllm.data.cleaning import rules_for
from scratchllm.data.sources import Source
from scratchllm.eda.leakage import EVAL_ONLY_SOURCES
from scratchllm.tokenizer.corpus import prepared_docs
from scratchllm.tokenizer.tokenizer import Tokenizer

END_OF_TEXT = "<|endoftext|>"
DTYPE = np.dtype("<u2")  # little-endian uint16 on every machine, so the files are portable
MAX_VOCAB = 1 << 16
TRAIN_SPLITS = ("train", "val", "test")
Progress = Callable[[str], None]


def _no_progress(_: str) -> None:
    pass


def splits_of(source_name: str) -> tuple[str, ...]:
    return ("eval",) if source_name in EVAL_ONLY_SOURCES else TRAIN_SPLITS


class _TokenFile:
    """Appends ids to a temporary file, hashing and counting as it goes; `close` renames it."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._tmp = path.with_name(path.name + ".tmp")
        self._file = self._tmp.open("wb")
        self._hash = hashlib.sha256()
        self.docs = self.tokens = self.chars = 0

    def add(self, ids: list[int], chars: int) -> None:
        data = np.asarray(ids, dtype=DTYPE).tobytes()
        self._file.write(data)
        self._hash.update(data)
        self.docs += 1
        self.tokens += len(ids)
        self.chars += chars

    def abort(self) -> None:
        self._file.close()
        self._tmp.unlink(missing_ok=True)

    def close(self) -> dict[str, Any]:
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()
        os.replace(self._tmp, self.path)
        return {
            "file": self.path.name,
            "docs": self.docs,
            "chars": self.chars,
            "tokens": self.tokens,
            "sha256": self._hash.hexdigest(),
        }


def tokenize_source(
    tokenizer: Tokenizer,
    tokenizer_sha256: str,
    source: Source,
    jsonl: Path,
    out_dir: Path,
    progress: Progress = _no_progress,
) -> dict[str, Any]:
    """Write the token files of one source and return (and save) its description."""
    if tokenizer.vocab_size > MAX_VOCAB:
        raise ValueError(f"vocabulary of {tokenizer.vocab_size} ids does not fit in uint16")
    out_dir.mkdir(parents=True, exist_ok=True)
    end_id = tokenizer.special_id(END_OF_TEXT)
    names = splits_of(source.name)
    files = {name: _TokenFile(out_dir / f"{source.name}.{name}.bin") for name in names}
    try:
        for count, (label, text) in enumerate(prepared_docs(jsonl, source), start=1):
            target = files["eval"] if "eval" in files else files.get(label)
            if target is None:
                raise ValueError(f"{jsonl.name}: unexpected split label {label!r}")
            target.add([*tokenizer.encode(text), end_id], len(text))
            if count % 10_000 == 0:
                progress(f"{source.name}: {count:,} documents")
    except BaseException:
        for f in files.values():
            f.abort()
        raise
    manifest = jsonl.with_name(f"{source.name}.manifest.json")
    meta = {
        "source": source.name,
        "role": "evaluation-only" if "eval" in files else "training",
        "tokenizer_sha256": tokenizer_sha256,
        "vocab_size": tokenizer.vocab_size,
        "end_of_text_id": end_id,
        "dtype": "uint16",
        "cleaning_rules": [r.name for r in rules_for(source)],
        "input_manifest_sha256": (
            hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else None
        ),
        "splits": {name: f.close() for name, f in files.items()},
    }
    (out_dir / f"{source.name}.tokens.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return meta


def verify_tokens(meta_path: Path) -> dict[str, Any]:
    """Check that the token files still match their description; raises ValueError if not."""
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for name, info in meta["splits"].items():
        path = meta_path.parent / info["file"]
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as f:
            while chunk := f.read(1 << 22):
                digest.update(chunk)
                size += len(chunk)
        if size != info["tokens"] * DTYPE.itemsize:
            raise ValueError(f"{path.name}: {size} bytes, expected {info['tokens']} tokens")
        if digest.hexdigest() != info["sha256"]:
            raise ValueError(f"{path.name} ({name}): sha256 differs from {meta_path.name}")
    return meta
