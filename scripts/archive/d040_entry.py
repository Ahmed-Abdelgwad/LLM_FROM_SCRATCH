# ruff: noqa: E501
"""Builds the tables of the D-040 entry of docs/DECISIONS.md: the fine-tuning results, judged by the criteria of D-039.

Needs runs/sft-sweep-*, runs/sft-{full,lora}-seed{1337,2024} and the reports:
reports/sft/final-sft-base-seed{1337,2024}, reports/sft/final-sft-{full,lora}-seed{1337,2024} (scratchllm sft-eval) and
reports/eval/final-sft-{full,lora}-seed{1337,2024} (scratchllm eval), with the base models' reports/eval/final-ablation-rope-swiglu{,-seed2024}.
Run from the repository root: uv run python scripts/archive/d040_entry.py OUT.md"""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (1337, 2024)
METHODS = ("full", "lora")
BASE_EVAL = {1337: "final-ablation-rope-swiglu", 2024: "final-ablation-rope-swiglu-seed2024"}
ARABIC = ("fw2_arb", "fw2_ars", "fw2_ary", "fw2_arz", "wiki_ar")
JUDGED = ("msa", "english")
SWEEP = {"full": ("3e-5", "1e-4", "3e-4"), "lora": ("3e-4", "1e-3", "3e-3")}


def events(run: str) -> list[dict]:
    return [
        json.loads(x) for x in (ROOT / "runs" / run / "metrics.jsonl").read_text().splitlines() if x
    ]


def sft_report(name: str) -> dict:
    return json.loads((ROOT / "reports/sft" / name / "report.json").read_text())


def full_eval(name: str) -> dict[str, float]:
    rows = json.loads((ROOT / "reports/eval" / name / "report.json").read_text())["rows"]
    by = {(r["source"], r["split"]): r["bits_per_char"] for r in rows}
    return {
        "arabic": statistics.mean(by[(s, "val")] for s in ARABIC),
        "english": by[("fineweb_edu_10bt", "val")],
        "wiki_ary": by[("wiki_ary", "eval")],
    }


def response_mean(report: dict) -> float:
    """Mean of the Standard Arabic and English test response bits/char (the comparison number of D-039)."""
    t = report["teacher_forced"]
    return 0.5 * (t["msa"]["bits_per_char"] + t["english"]["bits_per_char"])


def band(diffs: list[float]) -> str:
    mean = statistics.mean(diffs)
    same_sign = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)
    if abs(mean) < 0.005:
        return "the same"
    if abs(mean) <= 0.015 or not same_sign:
        return "unresolved"
    return "differs"


def main() -> None:
    out: list[str] = []
    # ---- stage A
    out += [
        "**Stage A, learning rate** (seed 1337, 400 steps; best validation score, bits/char):",
        "",
    ]
    out += [
        "| Method | " + " | ".join(f"lr {lr}" for lr in SWEEP["full"]) + " |",
        "|---|---|---|---|",
    ]
    chosen = {}
    for m in METHODS:
        scores = {}
        for lr in SWEEP[m]:
            ev = [e for e in events(f"sft-sweep-{m}-lr{lr}") if e["event"] == "eval"]
            scores[lr] = min(e["score"] for e in ev)
        best = min(scores, key=lambda k: scores[k])
        chosen[m] = best
        cells = " | ".join(
            f"{lr}: {s:.4f}" + (" **chosen**" if lr == best else "") for lr, s in scores.items()
        )
        out.append(f"| {m} | {cells} |")
    out.append("")
    assert chosen == {"full": "1e-4", "lora": "3e-3"}, chosen

    # ---- stage B: training facts
    out += ["**Stage B, the runs** (1,000 steps; best checkpoint by validation score):", ""]
    out += [
        "| Run | Trainable parameters | Best step | Best validation score | Median seconds per step | Peak memory (MiB) |",
        "|---|---|---|---|---|---|",
    ]
    facts = {}
    for m in METHODS:
        for s in SEEDS:
            ev = events(f"sft-{m}-seed{s}")
            start = next(e for e in ev if e["event"] == "start")
            evals = [e for e in ev if e["event"] == "eval"]
            best = min(evals, key=lambda e: e["score"])
            trains = [e for e in ev if e["event"] == "train"]
            steps = [
                (b["seconds"] - a["seconds"]) / (b["step"] - a["step"])
                for a, b in zip(trains, trains[1:], strict=False)
            ]
            peak = max(e["peak_memory_mib"] or 0 for e in trains)
            facts[(m, s)] = {
                "step_s": statistics.median(steps),
                "peak": peak,
                "trainable": start["trainable_parameters"],
            }
            out.append(
                f"| {m}, seed {s} | {start['trainable_parameters']:,} | {best['step']} | {best['score']:.4f} | {statistics.median(steps):.3f} | {peak:.0f} |"
            )
    out.append("")

    # ---- criteria
    out += [
        "**Criteria 1 to 3** (test split; replies greedy, 150 prompts per variety; Egyptian, 26 pairs, reported only):",
        "",
    ]
    out += [
        "| Model | Variety | Response bits/char | Change from base | Stopped | Right language | Empty | Looping |",
        "|---|---|---|---|---|---|---|---|",
    ]
    passes: dict[tuple[str, int], dict[str, bool]] = {}
    for s in SEEDS:
        base = sft_report(f"final-sft-base-seed{s}")
        for label, rep in [("base", base)] + [
            (m, sft_report(f"final-sft-{m}-seed{s}")) for m in METHODS
        ]:
            for v in ("msa", "egyptian", "english"):
                f, g = rep["teacher_forced"][v], rep["generation"][v]
                delta = f["bits_per_char"] - base["teacher_forced"][v]["bits_per_char"]
                change = "" if label == "base" else f"{delta:+.3f}"
                out.append(
                    f"| {label}, seed {s} | {v} | {f['bits_per_char']:.3f} | {change} | {g['stopped_rate']:.0%} | {g['language_rate']:.0%} | {g['empty_rate']:.0%} | {g['looping_rate']:.0%} |"
                )
            if label != "base":
                passes[(label, s)] = {
                    "1": all(
                        rep["teacher_forced"][v]["bits_per_char"]
                        - base["teacher_forced"][v]["bits_per_char"]
                        <= -0.10
                        for v in JUDGED
                    ),
                    "2": all(rep["generation"][v]["stopped_rate"] >= 0.80 for v in JUDGED),
                    "3": all(rep["generation"][v]["language_rate"] >= 0.90 for v in JUDGED),
                }
    out.append("")

    out += [
        "**Criterion 4, forgetting** (full pretraining validation, bits/char; change from the base model):",
        "",
    ]
    out += ["| Model | Arabic mean | English | `wiki_ary` |", "|---|---|---|---|"]
    forget = {}
    for s in SEEDS:
        b = full_eval(BASE_EVAL[s])
        out.append(
            f"| base, seed {s} | {b['arabic']:.4f} | {b['english']:.4f} | {b['wiki_ary']:.4f} |"
        )
        for m in METHODS:
            e = full_eval(f"final-sft-{m}-seed{s}")
            d = {k: e[k] - b[k] for k in e}
            forget[(m, s)] = d
            passes[(m, s)]["4"] = (
                d["arabic"] < 0.03 and d["english"] < 0.03 and d["wiki_ary"] < 0.05
            )
            out.append(
                f"| {m}, seed {s} | {e['arabic']:.4f} ({d['arabic']:+.4f}) | {e['english']:.4f} ({d['english']:+.4f}) | {e['wiki_ary']:.4f} ({d['wiki_ary']:+.4f}) |"
            )
    out.append("")

    out += ["**Verdict per criterion** (a method passes if both of its runs do):", ""]
    out += ["| Method | 1 Answers | 2 Stops | 3 Language | 4 Forgetting |", "|---|---|---|---|---|"]
    verdict = {}
    for m in METHODS:
        row = []
        for c in "1234":
            ok = all(passes[(m, s)][c] for s in SEEDS)
            verdict[(m, c)] = ok
            row.append(
                ("pass" if ok else "fail")
                + f" ({', '.join('yes' if passes[(m, s)][c] else 'no' for s in SEEDS)})"
            )
        out.append(f"| {m} | " + " | ".join(row) + " |")
    out.append("")

    out += ["**LoRA against full fine-tuning** (LoRA minus full, per seed; bands of D-031):", ""]
    out += ["| Number | Seed 1337 | Seed 2024 | Mean | Band |", "|---|---|---|---|---|"]
    comparisons = {}
    rows = [
        (
            "test response bits/char (mean of Standard Arabic and English)",
            [
                response_mean(sft_report(f"final-sft-lora-seed{s}"))
                - response_mean(sft_report(f"final-sft-full-seed{s}"))
                for s in SEEDS
            ],
        )
    ]
    for k, label in (
        ("arabic", "forgetting, Arabic mean"),
        ("english", "forgetting, English"),
        ("wiki_ary", "forgetting, `wiki_ary`"),
    ):
        rows.append((label, [forget[("lora", s)][k] - forget[("full", s)][k] for s in SEEDS]))
    for label, diffs in rows:
        comparisons[label] = band(diffs)
        out.append(
            f"| {label} | {diffs[0]:+.4f} | {diffs[1]:+.4f} | {statistics.mean(diffs):+.4f} | {comparisons[label]} |"
        )
    out.append("")
    speed = statistics.mean(
        facts[("lora", s)]["step_s"] / facts[("full", s)]["step_s"] for s in SEEDS
    )
    memory = statistics.mean(facts[("lora", s)]["peak"] / facts[("full", s)]["peak"] for s in SEEDS)
    out.append(
        f"LoRA's step time is {speed:.2f} of full fine-tuning's and its peak memory {memory:.2f} (mean of the two seeds)."
    )
    out.append("")
    print("\n".join(out))
    print(
        json.dumps(
            {
                "verdict": {f"{m}-{c}": v for (m, c), v in verdict.items()},
                "comparisons": comparisons,
            },
            indent=1,
        ),
        file=sys.stderr,
    )
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text("\n".join(out) + "\n", encoding="utf-8")


main()
