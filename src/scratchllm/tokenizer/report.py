"""Training the project's tokenizer, and the fertility report that compares it with others.

`train_tokenizer` reads the train split of the training sources (cleaned and normalized, equal
character budget per source), counts pre-tokens, learns the merges and returns the tokenizer with
a provenance record. `fertility_report` measures that tokenizer, at one or more vocabulary sizes,
and outside tokenizers from the Hugging Face hub, on the same held-out text, and writes numbers
and tables that are generated, not typed (the lesson of D-012).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scratchllm.eda.leakage import EVAL_ONLY_SOURCES
from scratchllm.tokenizer.bpe import count_words, train_merges
from scratchllm.tokenizer.corpus import (
    EVAL_SPLITS,
    available_sources,
    prepared_texts,
    training_sources,
)
from scratchllm.tokenizer.fertility import Counts, measure
from scratchllm.tokenizer.pretokenize import PATTERN
from scratchllm.tokenizer.tokenizer import BASE_VOCAB, Tokenizer

DEFAULT_SPECIAL_TOKENS = ("<|endoftext|>", "<email>", "<phone>")
Progress = Callable[[str], None]


def _no_progress(_: str) -> None:
    pass


def train_tokenizer(
    data_dir: Path,
    vocab_size: int,
    chars_per_source: int,
    special_tokens: Iterable[str] = DEFAULT_SPECIAL_TOKENS,
    progress: Progress = _no_progress,
) -> tuple[Tokenizer, dict[str, Any]]:
    specials = list(special_tokens)
    if vocab_size - BASE_VOCAB - len(specials) < 0:
        raise ValueError(f"vocab_size {vocab_size} is too small for the bytes and special tokens")
    counts: Counter[bytes] = Counter()
    taken: dict[str, dict[str, int]] = {}
    for source in training_sources(data_dir):
        stats = {"docs": 0, "chars": 0}

        def tally(texts: Iterator[str], stats: dict[str, int] = stats) -> Iterator[str]:
            for text in texts:
                stats["docs"] += 1
                stats["chars"] += len(text)
                yield text

        path = data_dir / f"{source.name}.jsonl"
        texts = prepared_texts(path, source, splits={"train"}, char_budget=chars_per_source)
        counts.update(count_words(tally(texts), specials))
        taken[source.name] = stats
        progress(f"{source.name}: {stats['docs']:,} docs, {stats['chars']:,} chars")
    if not taken:
        raise ValueError(f"no training source found in {data_dir}")
    merges = train_merges(counts, vocab_size - BASE_VOCAB - len(specials))
    progress(f"{len(merges):,} merges learned from {len(counts):,} distinct pre-tokens")
    provenance = {
        "vocab_size": vocab_size,
        "special_tokens": specials,
        "chars_per_source": chars_per_source,
        "sources": taken,
        "distinct_pretokens": len(counts),
        "merges": len(merges),
        "pattern": PATTERN.pattern,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return Tokenizer(merges, specials), provenance


# ---------------------------------------------------------------------------------------------


def _special_checker(first_special: int) -> Callable[[int], bool]:
    return lambda token_id: token_id >= first_special


def _hf_encoder(repo: str) -> tuple[Callable[[str], list[int]], int]:
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer as HFTokenizer

    hf = HFTokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
    return (lambda text: hf.encode(text, add_special_tokens=False).ids), hf.get_vocab_size()


def _eval_texts(data_dir: Path, max_docs: int) -> tuple[dict[str, list[str]], dict[str, str]]:
    texts: dict[str, list[str]] = {}
    roles: dict[str, str] = {}
    for source in available_sources(data_dir):
        evaluation_only = source.name in EVAL_ONLY_SOURCES
        texts[source.name] = list(
            prepared_texts(
                data_dir / f"{source.name}.jsonl",
                source,
                splits=None if evaluation_only else EVAL_SPLITS,
                max_docs=max_docs,
            )
        )
        roles[source.name] = "evaluation-only (never seen)" if evaluation_only else "val+test"
    return texts, roles


def _summary(by_source: dict[str, dict[str, Any]], languages: dict[str, str]) -> dict[str, float]:
    """Unweighted mean over the sources of a language, so a big source does not decide alone."""
    out: dict[str, float] = {}
    for language in sorted(set(languages.values())):
        names = [n for n in by_source if languages[n] == language]
        for key in ("tokens_per_word", "chars_per_token"):
            out[f"{language}_{key}"] = sum(by_source[n][key] for n in names) / len(names)
    return out


def _derived(counts: Counts) -> dict[str, Any]:
    return {
        **counts.__dict__,
        "tokens_per_word": counts.tokens_per_word,
        "chars_per_token": counts.chars_per_token,
        "bytes_per_token": counts.bytes_per_token,
        "tokens_per_1000_chars": counts.tokens_per_1000_chars,
    }


def fertility_report(
    tokenizer_path: Path,
    data_dir: Path,
    sizes: list[int],
    compare: list[str],
    max_docs: int,
    progress: Progress = _no_progress,
) -> dict[str, Any]:
    full = Tokenizer.load(tokenizer_path)
    texts, roles = _eval_texts(data_dir, max_docs)
    languages: dict[str, str] = {s.name: s.language for s in available_sources(data_dir)}
    encoders: dict[str, tuple[Callable[[str], list[int]], Callable[[int], bool] | None, int]] = {}
    for size in sizes:
        tok = full.truncated(size)
        first_special = size - len(tok.special_tokens)
        encoders[f"ours-{size}"] = (tok.encode, _special_checker(first_special), size)
    for repo in compare:
        encode, vocab = _hf_encoder(repo)
        encoders[repo] = (encode, None, vocab)

    results: dict[str, Any] = {}
    for label, (encode, is_special, size) in encoders.items():
        by_source = {name: _derived(measure(encode, t, is_special)) for name, t in texts.items()}
        results[label] = {
            "vocab_size": size,
            "sources": by_source,
            "summary": _summary(by_source, languages),
        }
        progress(f"measured {label}")
    return {
        "tokenizer_file": str(tokenizer_path),
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "max_docs_per_source": max_docs,
        "eval_sources": {
            name: {"role": roles[name], "language": languages[name], "docs": len(t)}
            for name, t in texts.items()
        },
        "results": results,
    }


def render_markdown(report: dict[str, Any]) -> str:
    results = report["results"]
    labels = list(results)
    sources = list(report["eval_sources"])
    lines = ["# Tokenizer fertility", ""]
    lines += [
        f"Tokenizer file `{report['tokenizer_file']}` "
        f"(sha256 `{report['tokenizer_sha256'][:16]}`). "
        f"Up to {report['max_docs_per_source']} held-out documents per source, cleaned and "
        "normalized, identical for every tokenizer.",
        "",
        "- *tokens per word*: tokens / `str.split()` words. Lower is better. Arabic words carry "
        "glued prefixes, so compare tokenizers with each other, not Arabic with English.",
        "- *characters per token*: independent of the word definition. Higher is better.",
        "- Sources marked *evaluation-only* were never used to train the tokenizer.",
        "- Our special tokens (`<email>`, `<phone>`) count as one token; outside tokenizers see "
        "them as text. The count of ours is in the last table.",
        "",
    ]

    def table(title: str, key: str, fmt: str) -> list[str]:
        out = [f"## {title}", "", "| source | role | " + " | ".join(labels) + " |"]
        out.append("|---|---|" + "---:|" * len(labels))
        for name in sources:
            row = [fmt.format(results[lb]["sources"][name][key]) for lb in labels]
            out.append(
                f"| {name} | {report['eval_sources'][name]['role']} | " + " | ".join(row) + " |"
            )
        return out + [""]

    lines += table("Tokens per word", "tokens_per_word", "{:.2f}")
    lines += table("Characters per token", "chars_per_token", "{:.2f}")

    lines += ["## Mean over sources of each language (unweighted)", ""]
    lines += [
        "| tokenizer | vocab | Arabic tokens/word | Arabic chars/token "
        "| English tokens/word | English chars/token |"
    ]
    lines += ["|---|---:|---:|---:|---:|---:|"]
    for lb in labels:
        s = results[lb]["summary"]
        vocab = results[lb]["vocab_size"] or ""
        lines.append(
            f"| {lb} | {vocab} | {s['ar_tokens_per_word']:.2f} | {s['ar_chars_per_token']:.2f} "
            f"| {s['en_tokens_per_word']:.2f} | {s['en_chars_per_token']:.2f} |"
        )
    lines += ["", "## Our special tokens in the evaluation text", ""]
    ours = [lb for lb in labels if lb.startswith("ours-")]
    if ours:
        sp = sum(results[ours[0]]["sources"][n]["special"] for n in sources)
        tk = sum(results[ours[0]]["sources"][n]["tokens"] for n in sources)
        lines.append(f"{sp:,} special tokens out of {tk:,} tokens ({sp / tk:.4%}) for `{ours[0]}`.")
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], out_dir: Path, run_id: str) -> Path:
    target = out_dir / run_id
    target.mkdir(parents=True, exist_ok=True)
    (target / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (target / "report.md").write_text(render_markdown(report), encoding="utf-8")
    return target
