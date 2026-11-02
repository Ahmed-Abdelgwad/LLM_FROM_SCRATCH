"""The text a tokenizer is trained and measured on: the cleaned, normalized documents.

Training text and evaluation text go through exactly the same steps as the model's will
(cleaning rules of D-004/D-005/D-010/D-011, then `normalize` of D-013), so a tokenizer is judged on
what it will really see. Documents come from the sampled files `data/raw/eda/<source>.jsonl`.

Which documents may train the tokenizer: the train split of the sources that are not
evaluation-only (D-002). Which are measured: the val and test splits of those sources, and every
document of the evaluation-only sources, which the tokenizer must never have seen.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Iterator
from pathlib import Path

from scratchllm.data.cleaning import CleaningStats, clean_docs, rules_for
from scratchllm.data.sources import SOURCES, Source
from scratchllm.eda.leakage import EVAL_ONLY_SOURCES
from scratchllm.tokenizer.normalize import normalize

EVAL_SPLITS = frozenset({"val", "test"})


def available_sources(data_dir: Path) -> list[Source]:
    """The registered sources that have a sampled file in `data_dir`, in a fixed order."""
    return [s for name, s in sorted(SOURCES.items()) if (data_dir / f"{name}.jsonl").exists()]


def training_sources(data_dir: Path) -> list[Source]:
    return [s for s in available_sources(data_dir) if s.name not in EVAL_ONLY_SOURCES]


def prepared_docs(
    path: Path, source: Source, splits: Collection[str] | None = None
) -> Iterator[tuple[str, str]]:
    """(split label, text) of each cleaned and normalized document, in file order.

    `splits` keeps only documents with those split labels (None keeps all). A document that is
    empty after normalization is skipped.
    """
    with path.open(encoding="utf-8") as f:
        raw = (json.loads(line) for line in f)
        wanted = (doc for doc in raw if splits is None or doc["split"] in splits)
        for doc in clean_docs(wanted, rules_for(source), CleaningStats()):
            text = normalize(doc["text"])
            if text:
                yield doc["split"], text


def prepared_texts(
    path: Path,
    source: Source,
    splits: Collection[str] | None = None,
    char_budget: int | None = None,
    max_docs: int | None = None,
) -> Iterator[str]:
    """Texts of `prepared_docs`, stopping once `char_budget` characters or `max_docs` documents
    have been produced."""
    chars = 0
    for count, (_, text) in enumerate(prepared_docs(path, source, splits), start=1):
        yield text
        chars += len(text)
        if (char_budget is not None and chars >= char_budget) or (
            max_docs is not None and count >= max_docs
        ):
            return
