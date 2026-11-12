"""Scoring a model on the fine-tuning test split (D-039): what it does with a question.

Two kinds of measurement, both per variety (Modern Standard, Egyptian, English):

* Teacher-forced: the response loss, in bits per character of the reference answer (as during
  training, `trainer.response_loss`), on every test pair that fits the context. It says how
  likely the model finds the reference answer, not whether it would write it.
* Free generation: greedy replies, up to `max_new_tokens`, to a fixed seeded sample of test
  prompts, and four properties of each reply that need no judge:
  - stopped: the reply ended with end-of-text before the limit (the model knows when it is done);
  - language: at least half of its letters are in the script of the prompt's language;
  - empty: it stopped at once, with no text;
  - looping: some word 3-gram occurs three or more times in it.
  None of these says the answer is correct; at 17M parameters most are not (D-040 shows samples).

The same function scores the base model, given the same chat template, so every number has a
reference: what fine-tuning changed.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import torch

from scratchllm.data.tokens import END_OF_TEXT
from scratchllm.finetune.chat import generate_reply
from scratchllm.finetune.data import read_split, script_share
from scratchllm.finetune.trainer import encode_pairs, response_loss
from scratchllm.model.gpt import GPT
from scratchllm.tokenizer.tokenizer import Tokenizer

VARIETIES = ("msa", "egyptian", "english")


def loops(text: str, n: int = 3, times: int = 3) -> bool:
    words = text.split()
    grams = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    return any(c >= times for c in grams.values())


def evaluate_sft(
    model: GPT,
    tokenizer: Tokenizer,
    data_dir: Path,
    device: torch.device,
    autocast: Callable[[], AbstractContextManager[Any]],
    *,
    split: str = "test",
    generate_per_variety: int = 150,
    max_new_tokens: int = 256,
    seed: int = 0,
    batch_size: int = 8,
    progress: Callable[[str], None] = lambda _: None,
) -> dict[str, Any]:
    pairs = read_split(data_dir, split)
    pad = tokenizer.special_id(END_OF_TEXT)
    items, dropped = encode_pairs(pairs, tokenizer, model.cfg.context_length)
    forced = response_loss(model, items, batch_size, pad, device, autocast)
    by_group: dict[str, Any] = {}
    for group, keep in (
        ("msa_aya", lambda e: e.pair.variety == "msa" and e.pair.source == "aya"),
        ("msa_cidar", lambda e: e.pair.source == "cidar"),
        ("official_aya_test", lambda e: "-test-" in e.pair.id),
    ):
        subset = [e for e in items if keep(e)]
        if subset:
            by_group[group] = response_loss(model, subset, batch_size, pad, device, autocast)

    rng = random.Random(seed)
    generation: dict[str, Any] = {}
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for variety in VARIETIES:
        pool = [e.pair for e in items if e.pair.variety == variety]
        chosen = sorted(rng.sample(pool, min(generate_per_variety, len(pool))), key=lambda p: p.id)
        counts: Counter[str] = Counter()
        lengths = []
        for i, p in enumerate(chosen):
            with torch.no_grad(), autocast():
                reply = generate_reply(model, tokenizer, p.prompt, max_new_tokens, device=device)
            counts["stopped"] += reply.stopped_by_eos
            counts["language"] += script_share(reply.text, p.language) >= 0.5
            counts["empty"] += reply.stopped_by_eos and not reply.text.strip()
            counts["looping"] += loops(reply.text)
            lengths.append(reply.new_tokens)
            if i < 6:
                samples[variety].append(
                    {"id": p.id, "prompt": p.prompt, "reference": p.response, "reply": reply.text,
                     "stopped": reply.stopped_by_eos}
                )  # fmt: skip
        n = len(chosen)
        if n:
            generation[variety] = {
                "replies": n,
                **{f"{k}_rate": counts[k] / n for k in ("stopped", "language", "empty", "looping")},
                "mean_new_tokens": sum(lengths) / n,
            }
            progress(f"{variety}: {generation[variety]}")
    return {
        "split": split,
        "pairs": len(pairs),
        "dropped_too_long": dropped,
        "teacher_forced": forced,
        "teacher_forced_groups": by_group,
        "generation": generation,
        "generation_settings": {
            "per_variety": generate_per_variety,
            "max_new_tokens": max_new_tokens,
            "decoding": "greedy",
            "seed": seed,
        },  # fmt: skip
        "samples": dict(samples),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [f"# Fine-tuning evaluation: {report.get('checkpoint', '')}", ""]
    lines += [
        "| Variety | Test pairs | Response bits/char | Stopped | Right language | Empty | "
        "Looping | Mean new tokens |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for v in VARIETIES:
        f, g = report["teacher_forced"].get(v), report["generation"].get(v)
        if f is None or g is None:
            continue
        lines.append(
            f"| {v} | {f['pairs']} | {f['bits_per_char']:.3f} | {g['stopped_rate']:.0%} | "
            f"{g['language_rate']:.0%} | {g['empty_rate']:.0%} | {g['looping_rate']:.0%} | "
            f"{g['mean_new_tokens']:.0f} |"
        )
    lines += ["", "## Samples (greedy)", ""]
    for v, rows in report["samples"].items():
        lines.append(f"### {v}")
        for s in rows:
            lines += [
                f"- **Prompt:** {s['prompt'][:200]}",
                f"  - **Reference:** {s['reference'][:200]}",
                f"  - **Reply:** {s['reply'][:300]}{'' if s['stopped'] else ' [cut]'}",
            ]
        lines.append("")
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")
