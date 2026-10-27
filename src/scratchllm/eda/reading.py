"""Draw documents to read by hand.

Counts say how much of a sample is mixed-script or short; only reading says whether those
documents are junk or text the model should learn (decision D-004). The draw is seeded and uses
the train split only, so two people reading with the same arguments read the same documents.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from scratchllm.eda.checks_learner import letter_script
from scratchllm.eda.runner import iter_train_docs

KINDS = ("all", "mixed", "short")
SHORT_CHARS = 300  # the A5 line


def is_mixed(text: str) -> bool:
    """At least 20% Arabic and at least 20% Latin letters: the same rule as ScriptMix."""
    scripts: Counter[str | None] = Counter(letter_script(ch) for ch in text)
    letters = sum(n for script, n in scripts.items() if script is not None)
    return letters > 0 and scripts["arabic"] * 5 >= letters and scripts["latin"] * 5 >= letters


def matches(doc: dict[str, Any], kind: str) -> bool:
    if kind == "all":
        return True
    if kind == "mixed":
        return is_mixed(doc["text"])
    if kind == "short":
        return len(doc["text"]) < SHORT_CHARS
    raise ValueError(f"unknown kind {kind!r}; expected one of {KINDS}")


def draw(docs: Iterable[dict[str, Any]], kind: str, n: int, seed: int) -> tuple[list[dict], int]:
    """Uniform random `n` of the matching documents (reservoir sampling, one pass).

    Returns the drawn documents and how many matched in total, so the reader knows what
    fraction of the matches was read.
    """
    rng = random.Random(seed)
    kept: list[dict[str, Any]] = []
    seen = 0
    for doc in docs:
        if not matches(doc, kind):
            continue
        seen += 1
        if len(kept) < n:
            kept.append(doc)
        else:
            j = rng.randrange(seen)
            if j < n:
                kept[j] = doc
    return kept, seen


def read_sample(manifest_path: Path, kind: str, n: int, seed: int) -> tuple[list[dict], int]:
    return draw(iter_train_docs(manifest_path), kind, n, seed)


def render(docs: list[dict[str, Any]], total_matching: int, chars: int, title: str) -> str:
    lines = [f"# {title}", "", f"{len(docs)} drawn of {total_matching} matching documents.", ""]
    for i, d in enumerate(docs, 1):
        text = d["text"][:chars].replace("\n", " ⏎ ")
        lines += [f"{i}. `{d['id']}` {len(d['text'])} chars, url: {d['url']}", f"   {text}", ""]
    return "\n".join(lines)
