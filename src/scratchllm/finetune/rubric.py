"""The human rubric of `docs/PROBLEM.md` (D-041): blind scoring of fine-tuned models' replies.

What the automatic measures of `evaluate.py` cannot say, whether a reply does what was asked and
whether what it says is true, needs a reader. This module makes the reader's job blind and its
result reproducible.

The rubric is the per-output part of InstructGPT's human evaluation (Ouyang et al. 2022, section
3.6, Table 3 and Figure 12a), cut to the labels that apply to a 17M model:

* overall quality, a 1 to 7 Likert score;
* fails to follow the instruction or task: yes or no;
* incorrect: the reply states something false or made up. InstructGPT's "hallucination" label is
  for closed-domain tasks (making up what is not in the input); here most prompts are open
  questions, so the label covers any false statement. Left empty when a reply says nothing that
  could be true or false (an empty or pure-markup reply).

The sheet:

* a fixed, seeded sample of test-split prompts per variety (held out from training and from
  checkpoint selection, D-038); `docs/PROBLEM.md` asked for a hand-written set, the reason for
  the change is in D-041;
* each prompt with the reply of every model, greedy and at most `max_new_tokens`, exactly as
  `sft-eval` asks for replies, and the dataset's reference answer as a help to the reader (the
  reference is itself often translated and is not ground truth);
* the replies to one prompt are shown as A, B, C... in an order shuffled per prompt, so neither
  position nor anything else on the sheet tells which model wrote a reply. Which letter is which
  model is kept in `key.json`, which the reader must not open before the scores are final.

`score_sheet` reads the filled `scores.csv` with the key and reports, per model and variety, the
mean Likert score and the two label rates, with 95% bootstrap intervals over prompts, and the
paired difference between every two models on the same prompts.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import random
import statistics
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import torch

from scratchllm.finetune.chat import generate_reply
from scratchllm.finetune.data import read_split
from scratchllm.finetune.trainer import encode_pairs
from scratchllm.model.gpt import GPT
from scratchllm.tokenizer.tokenizer import Tokenizer

DEFAULT_COUNTS = {"msa": 12, "english": 12, "egyptian": 6}
FIELDS = ("prompt_no", "reply", "overall", "fails_instruction", "incorrect", "notes")
LETTERS = "ABCDEFGH"


@dataclass(frozen=True)
class SheetItem:
    prompt_no: int
    pair_id: str
    variety: str
    prompt: str
    reference: str
    replies: dict[str, str]  # letter -> text shown
    cut: dict[str, bool]  # letter -> no end-of-text within the limit


def choose_prompts(
    data_dir: Path, tokenizer: Tokenizer, context_length: int, counts: Mapping[str, int], seed: int
) -> list[Any]:
    """The seeded sample of test pairs, `counts[variety]` of each, in a fixed order."""
    items, _ = encode_pairs(read_split(data_dir, "test"), tokenizer, context_length)
    rng = random.Random(seed)
    chosen = []
    for variety in sorted(counts):
        pool = sorted((e.pair for e in items if e.pair.variety == variety), key=lambda p: p.id)
        if len(pool) < counts[variety]:
            raise ValueError(f"only {len(pool)} {variety} test pairs, {counts[variety]} wanted")
        chosen += sorted(rng.sample(pool, counts[variety]), key=lambda p: p.id)
    return chosen


def make_sheet(
    models: Mapping[str, GPT],
    tokenizer: Tokenizer,
    data_dir: Path,
    device: torch.device,
    autocast: Callable[[], AbstractContextManager[Any]],
    *,
    counts: Mapping[str, int] = DEFAULT_COUNTS,
    seed: int = 0,
    max_new_tokens: int = 256,
    progress: Callable[[str], None] = lambda _: None,
) -> tuple[list[SheetItem], dict[str, Any]]:
    """The blind items and the key (prompt number -> letter -> model name)."""
    if not 2 <= len(models) <= len(LETTERS):
        raise ValueError(f"between 2 and {len(LETTERS)} models")
    context = min(m.cfg.context_length for m in models.values())
    pairs = choose_prompts(data_dir, tokenizer, context, counts, seed)
    replies: dict[str, list[Any]] = {}
    for name, model in models.items():
        model.eval()
        replies[name] = []
        for p in pairs:
            with torch.no_grad(), autocast():
                replies[name].append(
                    generate_reply(model, tokenizer, p.prompt, max_new_tokens, device=device)
                )
        progress(f"{name}: {len(pairs)} replies")
    shuffle = random.Random(f"rubric-order-{seed}")
    names = list(models)
    items: list[SheetItem] = []
    key: dict[str, Any] = {}
    for n, p in enumerate(pairs, start=1):
        order = names[:]
        shuffle.shuffle(order)
        letters = dict(zip(LETTERS, order, strict=False))
        key[str(n)] = {"pair_id": p.id, "variety": p.variety, "letters": letters}
        items.append(
            SheetItem(
                n, p.id, p.variety, p.prompt, p.response,
                {a: replies[m][n - 1].text for a, m in letters.items()},
                {a: not replies[m][n - 1].stopped_by_eos for a, m in letters.items()},
            )
        )  # fmt: skip
    return items, {"models": names, "prompts": key}


# ----------------------------------------------------------------------------- files


def write_sheet(
    items: Sequence[SheetItem], key: Mapping[str, Any], meta: Mapping[str, Any], out_dir: Path
) -> None:
    """`sheet.html` (score in a browser), `scores.csv` (or in a spreadsheet), `key.json`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "scores.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([*FIELDS, "variety", "prompt", "reply_text"])
        for it in items:
            for a, text in it.replies.items():
                w.writerow([it.prompt_no, a, "", "", "", "", it.variety, it.prompt, text])
    (out_dir / "key.json").write_text(
        json.dumps({**key, "meta": dict(meta)}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "sheet.html").write_text(render_html(items), encoding="utf-8")


def render_html(items: Sequence[SheetItem]) -> str:
    data = json.dumps(
        [
            {"n": it.prompt_no, "variety": it.variety, "prompt": it.prompt,
             "reference": it.reference, "replies": it.replies, "cut": it.cut}
            for it in items
        ],
        ensure_ascii=False,
    ).replace("</", "<\\/")  # fmt: skip
    sheet_id = hashlib.sha256(data.encode()).hexdigest()[:12]  # one browser store per sheet
    page = (Path(__file__).with_name("rubric_sheet.html")).read_text(encoding="utf-8")
    return (
        page.replace("__DATA__", data)
        .replace("__FIELDS__", html.escape(",".join(FIELDS)))
        .replace("__SHEET__", sheet_id)
    )


def read_scores(path: Path) -> list[dict[str, Any]]:
    """The filled rows, checked: every value in range, no row missing or repeated."""
    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        for i, raw in enumerate(csv.DictReader(f), start=2):
            try:
                overall = int(raw["overall"])
                fails = int(raw["fails_instruction"])
                incorrect = None if raw["incorrect"].strip() == "" else int(raw["incorrect"])
            except (KeyError, ValueError) as e:
                raise ValueError(f"{path}:{i}: missing or not a number ({e})") from None
            if not 1 <= overall <= 7 or fails not in (0, 1) or incorrect not in (None, 0, 1):
                raise ValueError(f"{path}:{i}: overall 1-7, fails 0/1, incorrect 0/1 or empty")
            rows.append(
                {"prompt_no": str(int(raw["prompt_no"])), "reply": raw["reply"].strip(),
                 "overall": overall, "fails_instruction": fails, "incorrect": incorrect}
            )  # fmt: skip
    return rows


def _bootstrap(values_by_prompt: Sequence[float], seed: int, reps: int = 2000) -> list[float]:
    rng = random.Random(seed)
    n = len(values_by_prompt)
    means = sorted(
        statistics.fmean(values_by_prompt[rng.randrange(n)] for _ in range(n)) for _ in range(reps)
    )
    return [means[int(0.025 * reps)], means[int(0.975 * reps) - 1]]


def summarize(rows: Sequence[Mapping[str, Any]], key: Mapping[str, Any]) -> dict[str, Any]:
    prompts = key["prompts"]
    expected = {(n, a) for n, k in prompts.items() for a in k["letters"]}
    seen = [(r["prompt_no"], r["reply"]) for r in rows]
    if len(set(seen)) != len(seen):
        raise ValueError("a reply is scored twice")
    if set(seen) != expected:
        missing = sorted(expected - set(seen))[:5]
        extra = sorted(set(seen) - expected)[:5]
        raise ValueError(f"scores do not match the sheet: missing {missing}, unknown {extra}")
    by: dict[str, dict[str, Mapping[str, Any]]] = {}  # model -> prompt -> row
    for r in rows:
        model = prompts[r["prompt_no"]]["letters"][r["reply"]]
        by.setdefault(model, {})[r["prompt_no"]] = r
    out: dict[str, Any] = {"per_model": {}, "paired": {}, "prompts": len(prompts)}
    groups = {"all": sorted(prompts, key=int)}
    for v in sorted({k["variety"] for k in prompts.values()}):
        groups[v] = [n for n in groups["all"] if prompts[n]["variety"] == v]
    for model in key["models"]:
        out["per_model"][model] = {}
        for g, ns in groups.items():
            rs = [by[model][n] for n in ns]
            overall = [float(r["overall"]) for r in rs]
            judged = [r["incorrect"] for r in rs if r["incorrect"] is not None]
            out["per_model"][model][g] = {
                "n": len(rs),
                "overall": statistics.fmean(overall),
                "overall_ci95": _bootstrap(overall, seed=1),
                "fails_instruction_rate": statistics.fmean(r["fails_instruction"] for r in rs),
                "incorrect_rate": statistics.fmean(judged) if judged else None,
                "incorrect_judged": len(judged),
            }
    for a, b in combinations(key["models"], 2):
        diffs = [float(by[a][n]["overall"] - by[b][n]["overall"]) for n in groups["all"]]
        out["paired"][f"{a} - {b}"] = {
            "overall_diff": statistics.fmean(diffs),
            "ci95": _bootstrap(diffs, seed=2),
            "wins": sum(d > 0 for d in diffs),
            "ties": sum(d == 0 for d in diffs),
            "losses": sum(d < 0 for d in diffs),
        }
    return out


def render_summary(summary: Mapping[str, Any]) -> str:
    lines = ["# Human rubric scores", "", f"{summary['prompts']} prompts, scored blind.", ""]
    lines += [
        "| Model | Prompts | Overall (1-7) | 95% interval | Fails instruction "
        "| Incorrect (judged) |",
        "|---|---|---|---|---|---|",
    ]
    for model, groups in summary["per_model"].items():
        for g, s in groups.items():
            inc = "" if s["incorrect_rate"] is None else f"{s['incorrect_rate']:.0%}"
            lines.append(
                f"| {model} ({g}) | {s['n']} | {s['overall']:.2f} | "
                f"{s['overall_ci95'][0]:.2f} to {s['overall_ci95'][1]:.2f} | "
                f"{s['fails_instruction_rate']:.0%} | {inc} ({s['incorrect_judged']}) |"
            )
    lines += ["", "| Pair | Mean difference | 95% interval | Wins / ties / losses |"]
    lines += ["|---|---|---|---|"]
    for pair, p in summary["paired"].items():
        lines.append(
            f"| {pair} | {p['overall_diff']:+.2f} | {p['ci95'][0]:+.2f} to {p['ci95'][1]:+.2f} | "
            f"{p['wins']} / {p['ties']} / {p['losses']} |"
        )
    return "\n".join(lines) + "\n"


def score_sheet(sheet_dir: Path, scores: Path | None = None) -> dict[str, Any]:
    key = json.loads((sheet_dir / "key.json").read_text(encoding="utf-8"))
    summary = summarize(read_scores(scores or sheet_dir / "scores.csv"), key)
    (sheet_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (sheet_dir / "summary.md").write_text(render_summary(summary), encoding="utf-8")
    return summary
