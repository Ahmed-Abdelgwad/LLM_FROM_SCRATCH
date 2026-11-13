# ruff: noqa: E501
"""Builds the tables of the D-042 results entry of docs/DECISIONS.md: replay of pretraining text in full fine-tuning.

Needs runs/sft-full-replay{0,5,10,20}-seed{1337,2024} and runs/sft-full-seed{1337,2024} (D-040), the reports
reports/sft/final-sft-full-replay*-seed* (scratchllm sft-eval) and reports/eval/final-sft-full-replay*-seed* (scratchllm eval),
with the base models' reports/eval/final-ablation-rope-swiglu{,-seed2024} and reports/sft/final-sft-base-seed*.
Writes the eval events it compares to reports/eval/final-d042-replay/evals.json.
Run from the repository root: uv run python scripts/archive/d042_entry.py OUT.md"""

import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (1337, 2024)
RATES = (0, 5, 10, 20)
BASE_EVAL = {1337: "final-ablation-rope-swiglu", 2024: "final-ablation-rope-swiglu-seed2024"}
ARABIC = ("fw2_arb", "fw2_ars", "fw2_ary", "fw2_arz", "wiki_ar")
FORGET = ("arabic", "english", "wiki_ary")
LABEL = {"arabic": "Arabic mean", "english": "English", "wiki_ary": "`wiki_ary`"}


def run(rate: int, seed: int) -> str:
    return f"sft-full-replay{rate}-seed{seed}"


def events(name: str) -> list[dict]:
    lines = (ROOT / "runs" / name / "metrics.jsonl").read_text().splitlines()
    return [json.loads(x) for x in lines if x]


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
    t = report["teacher_forced"]
    return 0.5 * (t["msa"]["bits_per_char"] + t["english"]["bits_per_char"])


def band(diffs: list[float]) -> str:
    """D-031: under 0.005 the same; 0.005 to 0.015, or seeds disagreeing, unresolved; else differs."""
    mean = statistics.mean(diffs)
    same_sign = all(d > 0 for d in diffs) or all(d < 0 for d in diffs)
    if abs(mean) < 0.005:
        return "the same"
    if abs(mean) <= 0.015 or not same_sign:
        return "unresolved"
    return "differs"


def probe_arabic(e: dict) -> float:
    f = e["forgetting_bits_per_char"]
    return statistics.mean(f[s] for s in ARABIC)


def main() -> None:
    out: list[str] = []
    evidence: dict[str, list[dict]] = {}

    # ---- the 0% reruns against D-040's runs
    out += ["**The 0% runs against D-040's full runs** (same config, new code):", ""]
    out += [
        "| Seed | Eval events equal | Weights of `merged.pt` equal | Largest difference of a logged eval number |",
        "|---|---|---|---|",
    ]
    rerun_noise: dict[int, dict[str, float]] = {}
    for s in SEEDS:
        old = [e for e in events(f"sft-full-seed{s}") if e["event"] == "eval"]
        new = [e for e in events(run(0, s)) if e["event"] == "eval"]
        evidence[f"sft-full-seed{s}"] = old

        def numbers(ev: list[dict]) -> list[float]:
            vals = []
            for e in ev:
                vals.append(e["score"])
                vals += [e["val"][v]["bits_per_char"] for v in sorted(e["val"])]
                vals += [
                    e["forgetting_bits_per_char"][k] for k in sorted(e["forgetting_bits_per_char"])
                ]
            return vals

        a, b = numbers(old), numbers(new)
        largest = max(abs(x - y) for x, y in zip(a, b, strict=True))
        wa = torch.load(
            ROOT / "runs" / f"sft-full-seed{s}" / "merged.pt",
            map_location="cpu",
            weights_only=False,
        )["model"]
        wb = torch.load(
            ROOT / "runs" / run(0, s) / "merged.pt", map_location="cpu", weights_only=False
        )["model"]
        same_weights = wa.keys() == wb.keys() and all(torch.equal(wa[k], wb[k]) for k in wa)
        out.append(
            f"| {s} | {'yes' if a == b else 'no'} | {'yes' if same_weights else 'no'} | {largest:.6f} |"
        )
        # the same numbers as the comparisons below, D-040 run against the 0% rerun
        d040, rerun = full_eval(f"final-sft-full-seed{s}"), full_eval(f"final-{run(0, s)}")
        rerun_noise[s] = {k: rerun[k] - d040[k] for k in FORGET}
        rerun_noise[s]["response"] = response_mean(
            sft_report(f"final-{run(0, s)}")
        ) - response_mean(sft_report(f"final-sft-full-seed{s}"))
    out.append("")
    out += ["| Number, 0% rerun minus D-040 | Seed 1337 | Seed 2024 |", "|---|---|---|"]
    for k in ("response", *FORGET):
        name = "test response bits/char" if k == "response" else f"forgetting, {LABEL[k]}"
        out.append(f"| {name} | {rerun_noise[1337][k]:+.4f} | {rerun_noise[2024][k]:+.4f} |")
    out.append("")
    noise = max(abs(v) for s in SEEDS for v in rerun_noise[s].values())

    # ---- training facts
    out += [
        "**The runs** (1,000 steps, 32 rows per step; best checkpoint by validation score):",
        "",
    ]
    out += [
        "| Run | Replayed rows | Replay share of scored tokens | Best step | Best validation score | Median s/step | Peak MiB |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in SEEDS:
        for r in RATES:
            ev = events(run(r, s))
            evidence[run(r, s)] = [e for e in ev if e["event"] == "eval"]
            trains = [e for e in ev if e["event"] == "train"]
            evals = [e for e in ev if e["event"] == "eval"]
            best = min(evals, key=lambda e: e["score"])
            steps = [
                (b["seconds"] - a["seconds"]) / (b["step"] - a["step"])
                for a, b in zip(trains, trains[1:], strict=False)
            ]
            rows = trains[-1].get("replay_rows", 0)
            share = trains[-1].get("replay_token_share", 0.0)
            assert rows == 32_000 * r // 100, (run(r, s), rows)
            peak = max(e["peak_memory_mib"] or 0 for e in trains)
            out.append(
                f"| {r}%, seed {s} | {rows:,} | {share:.1%} | {best['step']} | {best['score']:.4f} | {statistics.median(steps):.3f} | {peak:.0f} |"
            )
    out.append("")

    # ---- per run: answers and forgetting
    out += [
        "**Answers and forgetting** (test split; full pretraining validation, change from the base model, bits/char):",
        "",
    ]
    out += [
        "| Run | Response, Standard Arabic | Response, English | Mean | Stopped (msa, en) | Right language (msa, en) | Arabic mean | English | `wiki_ary` |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    res: dict[tuple[int, int], dict[str, float]] = {}
    for s in SEEDS:
        base_f = full_eval(BASE_EVAL[s])
        base_r = response_mean(sft_report(f"final-sft-base-seed{s}"))
        for r in RATES:
            rep = sft_report(f"final-{run(r, s)}")
            f = full_eval(f"final-{run(r, s)}")
            t, g = rep["teacher_forced"], rep["generation"]
            d = {k: f[k] - base_f[k] for k in FORGET}
            res[(r, s)] = {"response": response_mean(rep), "gain": response_mean(rep) - base_r, **d,
                           "stop_msa": g["msa"]["stopped_rate"], "stop_en": g["english"]["stopped_rate"],
                           "lang_msa": g["msa"]["language_rate"], "lang_en": g["english"]["language_rate"],
                           "gain_msa": t["msa"]["bits_per_char"] - sft_report(f"final-sft-base-seed{s}")["teacher_forced"]["msa"]["bits_per_char"],
                           "gain_en": t["english"]["bits_per_char"] - sft_report(f"final-sft-base-seed{s}")["teacher_forced"]["english"]["bits_per_char"]}  # fmt: skip
            x = res[(r, s)]
            out.append(
                f"| {r}%, seed {s} | {t['msa']['bits_per_char']:.3f} | {t['english']['bits_per_char']:.3f} | {x['response']:.4f} | "
                f"{x['stop_msa']:.0%}, {x['stop_en']:.0%} | {x['lang_msa']:.0%}, {x['lang_en']:.0%} | "
                f"{d['arabic']:+.4f} | {d['english']:+.4f} | {d['wiki_ary']:+.4f} |"
            )
    out.append("")

    # ---- registered comparisons
    out += [
        "**Against the 0% run of the same seed** (replay minus 0%; bands of D-031 on the mean of the two seeds):",
        "",
    ]
    out += ["| Rate | Number | Seed 1337 | Seed 2024 | Mean | Band |", "|---|---|---|---|---|---|"]
    bands: dict[tuple[int, str], str] = {}
    means: dict[tuple[int, str], float] = {}
    for r in RATES[1:]:
        for k in ("response", *FORGET):
            diffs = [res[(r, s)][k] - res[(0, s)][k] for s in SEEDS]
            bands[(r, k)] = band(diffs)
            means[(r, k)] = statistics.mean(diffs)
            name = "cost: test response bits/char" if k == "response" else f"forgetting, {LABEL[k]}"
            out.append(
                f"| {r}% | {name} | {diffs[0]:+.4f} | {diffs[1]:+.4f} | {means[(r, k)]:+.4f} | {bands[(r, k)]} |"
            )
    out.append("")

    # ---- the choice rule
    # as registered: cost "the same" or "unresolved"; a rate that makes the answers better has no cost
    eligible = [
        r for r in RATES[1:]
        if bands[(r, "response")] in ("the same", "unresolved") or means[(r, "response")] < 0
    ]  # fmt: skip
    chosen = min(eligible, key=lambda r: means[(r, "arabic")]) if eligible else None
    out.append(
        "**Choice rule** (D-042): rates whose cost is the same or unresolved: "
        + (", ".join(f"{r}%" for r in eligible) or "none")
        + "; chosen: "
        + (f"**{chosen}%**" if chosen is not None else "**none**")
        + "."
    )
    out.append(
        f"Largest difference between a 0% rerun and its D-040 run on these numbers: {noise:.4f} bits/char."
    )
    out.append("")

    # ---- D-039 criteria, reported only
    out += [
        "**D-039 criteria, reported only** (D-041): 1 answers (both varieties at least 0.10 lower), 2 stops (80%), 3 language (90%), 4 forgetting (0.03, `wiki_ary` 0.05):",
        "",
    ]
    out += ["| Run | 1 | 2 | 3 | 4 |", "|---|---|---|---|---|"]
    for s in SEEDS:
        for r in RATES:
            x = res[(r, s)]
            c = [x["gain_msa"] <= -0.10 and x["gain_en"] <= -0.10,
                 x["stop_msa"] >= 0.80 and x["stop_en"] >= 0.80,
                 x["lang_msa"] >= 0.90 and x["lang_en"] >= 0.90,
                 x["arabic"] < 0.03 and x["english"] < 0.03 and x["wiki_ary"] < 0.05]  # fmt: skip
            out.append(f"| {r}%, seed {s} | " + " | ".join("yes" if v else "no" for v in c) + " |")
    out.append("")

    # ---- Kalajdzievski: forgetting per unit of learning
    out += [
        "**Forgetting per unit of learning** (after Kalajdzievski 2024; Arabic-mean forgetting divided by the response gain, mean of the seeds):",
        "",
    ]
    out += [
        "| Rate | Response gain (bits/char) | Arabic-mean forgetting | Ratio |",
        "|---|---|---|---|",
    ]
    for r in RATES:
        gain = statistics.mean(-res[(r, s)]["gain"] for s in SEEDS)
        forget = statistics.mean(res[(r, s)]["arabic"] for s in SEEDS)
        out.append(f"| {r}% | {gain:.4f} | {forget:+.4f} | {forget / gain:.2f} |")
    out.append("")

    # ---- curves
    out += [
        "**Forgetting during training** (16-window probe, Arabic mean, bits/char; mean of the seeds):",
        "",
    ]
    marks = (50, 250, 500, 750, 1000)
    out += [
        "| Rate | " + " | ".join(f"step {m}" for m in marks) + " |",
        "|---" * (len(marks) + 1) + "|",
    ]
    for r in RATES:
        cells = []
        for m in marks:
            vals = [
                probe_arabic(next(e for e in evidence[run(r, s)] if e["step"] == m)) for s in SEEDS
            ]
            cells.append(f"{statistics.mean(vals):.4f}")
        out.append(f"| {r}% | " + " | ".join(cells) + " |")
    out.append("")

    print("\n".join(out))
    folder = ROOT / "reports/eval/final-d042-replay"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "evals.json").write_text(json.dumps(evidence, indent=1) + "\n")
    print(
        json.dumps(
            {"chosen": chosen, "bands": {f"{r}-{k}": v for (r, k), v in bands.items()}}, indent=1
        ),
        file=sys.stderr,
    )
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text("\n".join(out) + "\n", encoding="utf-8")


main()
