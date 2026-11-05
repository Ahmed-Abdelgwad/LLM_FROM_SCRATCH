"""Scoring a checkpoint on every held-out token, with baselines to read the numbers against.

Training-time evaluation looks at a fixed sample of windows (cheap, comparable between steps).
This module is the thorough version: for each source and split it scores ALL non-overlapping
windows of the token file, and reports

* loss, in nats per token, and its exponential, the perplexity (per token: not comparable between
  tokenizers);
* bits per character, using the split's tokens-per-character ratio: the number to compare across
  tokenizers, languages and sources. It includes the cost of predicting each end-of-text id;
* baselines on the same tokens, so the model's number has something to be measured against:
  the uniform guess (log2 of the vocabulary), and unigram models that know only how often each
  token occurs in training data (add-one smoothing) and ignore all context. `unigram_*` counts
  the train split of every training source, both languages pooled; `unigram_same_language_*`
  counts only the training sources in the language of the scored source (D-026). The pooled one
  spends probability on the other language's tokens, so it is the weaker baseline and makes the
  model look better against it; the same-language one is the fair comparison. It is left empty
  for a source whose language is not in the source registry.

Splits: val and test of the training sources, and the single `eval` split of the evaluation-only
sources (D-002), whose dialects the tokenizer and the model never saw.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scratchllm.data.sources import SOURCES
from scratchllm.data.tokens import DTYPE
from scratchllm.model.gpt import GPT
from scratchllm.training.batches import bits_per_char
from scratchllm.training.checkpoint import load_checkpoint, model_from_state
from scratchllm.training.dataset import TokenWindows


@dataclass(frozen=True)
class SplitSpec:
    source: str
    split: str
    role: str
    path: Path
    tokens: int
    chars: int


def list_splits(tokens_dir: Path, splits: Sequence[str] = ("val", "test")) -> list[SplitSpec]:
    """The requested splits of the training sources, plus the eval split of evaluation-only ones."""
    found: list[SplitSpec] = []
    for meta_path in sorted(tokens_dir.glob("*.tokens.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        wanted = splits if meta["role"] == "training" else ("eval",)
        for split in wanted:
            info = meta["splits"].get(split)
            if info is not None and info["tokens"] > 0:
                found.append(
                    SplitSpec(
                        meta["source"],
                        split,
                        meta["role"],
                        tokens_dir / info["file"],
                        info["tokens"],
                        info["chars"],
                    )  # fmt: skip
                )
    return found


@torch.no_grad()
def full_loss(
    model: GPT,
    path: Path,
    context_length: int,
    batch_size: int,
    device: torch.device,
    autocast: Callable[[], AbstractContextManager[Any]],
) -> tuple[float, int]:
    """Mean loss over every non-overlapping window of a token file, and the windows' count."""
    dataset = TokenWindows([path], context_length)
    if len(dataset) == 0:
        raise ValueError(f"{path.name}: fewer than {context_length + 1} tokens")
    model.eval()
    total = 0.0
    for start in range(0, len(dataset), batch_size):
        chunk = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
        x = torch.stack([a for a, _ in chunk]).to(device)
        y = torch.stack([b for _, b in chunk]).to(device)
        with autocast():
            _, loss = model(x, y)
        assert loss is not None
        total += float(loss) * len(chunk)
    return total / len(dataset), len(dataset)


def token_counts(paths: Sequence[Path], vocab_size: int) -> np.ndarray:
    counts = np.zeros(vocab_size, dtype=np.int64)
    for path in paths:
        tokens = np.memmap(path, dtype=DTYPE, mode="r")
        for start in range(0, len(tokens), 1 << 24):
            counts += np.bincount(tokens[start : start + (1 << 24)], minlength=vocab_size)
    return counts


def unigram_loss(counts: np.ndarray, path: Path) -> float:
    """Cross-entropy (nats/token) of a file under the add-one-smoothed token frequencies."""
    log_p = np.log((counts + 1) / (counts.sum() + len(counts)))
    tokens = np.fromfile(path, dtype=DTYPE)
    return float(-log_p[tokens].mean())


def evaluate_checkpoint(
    model: GPT,
    specs: Sequence[SplitSpec],
    train_paths: Sequence[Path],
    batch_size: int,
    device: torch.device,
    autocast: Callable[[], AbstractContextManager[Any]],
    progress: Callable[[str], None] = lambda _: None,
    language_of: Mapping[str, str] | None = None,
    train_paths_by_source: Mapping[str, Path] | None = None,
) -> list[dict[str, Any]]:
    """One row per split. `language_of` (source -> language) and `train_paths_by_source` give the
    same-language unigram; without them, or for a source of unknown language, it is None."""
    vocab = model.cfg.vocab_size
    counts = token_counts(train_paths, vocab)
    language_of = language_of or {}
    by_language: dict[str, list[str]] = {}
    for name in sorted(train_paths_by_source or {}):
        if name in language_of:
            by_language.setdefault(language_of[name], []).append(name)
    language_counts = {
        lang: token_counts([(train_paths_by_source or {})[n] for n in names], vocab)
        for lang, names in by_language.items()
    }
    rows: list[dict[str, Any]] = []
    for spec in specs:
        loss, windows = full_loss(
            model, spec.path, model.cfg.context_length, batch_size, device, autocast
        )
        uni = unigram_loss(counts, spec.path)
        row = {
            "source": spec.source,
            "split": spec.split,
            "role": spec.role,
            "tokens": spec.tokens,
            "chars": spec.chars,
            "windows": windows,
            "loss": loss,
            "perplexity": math.exp(loss),
            "bits_per_char": bits_per_char(loss, spec.tokens, spec.chars),
            "unigram_loss": uni,
            "unigram_bits_per_char": bits_per_char(uni, spec.tokens, spec.chars),
            "uniform_bits_per_char": bits_per_char(math.log(vocab), spec.tokens, spec.chars),
            "unigram_same_language_loss": None,
            "unigram_same_language_bits_per_char": None,
            "unigram_same_language_sources": None,
        }
        lang = language_of.get(spec.source)
        if lang in language_counts:
            same = unigram_loss(language_counts[lang], spec.path)
            row["unigram_same_language_loss"] = same
            row["unigram_same_language_bits_per_char"] = bits_per_char(
                same, spec.tokens, spec.chars
            )
            row["unigram_same_language_sources"] = by_language[lang]
        rows.append(row)
        progress(
            f"{spec.source}/{spec.split}: loss {loss:.3f}, {row['bits_per_char']:.3f} bits/char"
        )
    return rows


def run_evaluation(
    checkpoint: Path,
    tokens_dir: Path | None,
    splits: Sequence[str],
    batch_size: int,
    device: torch.device,
    autocast: Callable[[], AbstractContextManager[Any]],
    progress: Callable[[str], None] = lambda _: None,
) -> dict[str, Any]:
    state = load_checkpoint(checkpoint, map_location=device)
    model = model_from_state(state, device)
    directory = tokens_dir or Path(state["config"]["data"]["tokens_dir"])
    specs = list_splits(directory, splits)
    metas = [
        json.loads(m.read_text(encoding="utf-8")) for m in sorted(directory.glob("*.tokens.json"))
    ]
    by_source = {
        m["source"]: directory / m["splits"]["train"]["file"]
        for m in metas
        if m["role"] == "training"
    }
    language_of = {
        m["source"]: SOURCES[m["source"]].language for m in metas if m["source"] in SOURCES
    }
    rows = evaluate_checkpoint(
        model, specs, list(by_source.values()), batch_size, device, autocast, progress,
        language_of=language_of, train_paths_by_source=by_source,
    )  # fmt: skip
    return {
        "checkpoint": str(checkpoint),
        "step": state["step"],
        "parameters": model.num_parameters(),
        "tokens_dir": str(directory),
        "rows": rows,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Evaluation of {report['checkpoint']}",
        "",
        f"Checkpoint step {report['step']}, {report['parameters']:,} parameters. Every "
        "non-overlapping window of each split is scored. *Bits per character* is the figure to "
        "compare across sources; *unigram* knows only token frequencies of the training data (both "
        "languages pooled), *same-language unigram* only those of the training sources in the "
        "source's own language (the fair baseline, D-026), and *uniform* knows nothing.",
        "",
        "| source | split | role | tokens | loss | perplexity | bits/char | unigram bits/char "
        "| uniform bits/char | model / unigram | same-language unigram bits/char "
        "| model / same-language unigram |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in report["rows"]:
        ratio = r["bits_per_char"] / r["unigram_bits_per_char"]
        same = r.get("unigram_same_language_bits_per_char")
        same_cells = f"{same:.3f} | {r['bits_per_char'] / same:.2f}" if same else "— | —"
        lines.append(
            f"| {r['source']} | {r['split']} | {r['role']} | {r['tokens']:,} | {r['loss']:.3f} "
            f"| {r['perplexity']:.0f} | {r['bits_per_char']:.3f} "
            f"| {r['unigram_bits_per_char']:.3f} | {r['uniform_bits_per_char']:.3f} | {ratio:.2f} "
            f"| {same_cells} |"
        )
    return "\n".join(lines) + "\n"
