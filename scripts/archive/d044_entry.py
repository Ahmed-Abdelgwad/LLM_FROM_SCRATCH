# ruff: noqa: E501
"""Builds the tables of the D-045 entry of docs/DECISIONS.md: grouped-query attention (2 key/value heads) against the
reference at two seeds, judged by the criterion registered in D-044 and in `configs/experiments/ablation-gqa2.yaml`
before the runs. Prints the tables and the verdict; the prose of the entry is written by hand around them.

Also writes the evaluation events of the four runs to reports/eval/final-d044-gqa/evals.json.
Needs runs/{ablation-rope-swiglu,ablation-rope-swiglu-seed2024,ablation-gqa2,ablation-gqa2-seed2024} and the full
evaluations final-<same names> in reports/eval.
Run from the repository root: uv run python scripts/archive/d044_entry.py OUT.md"""

import json
import statistics
import sys
from pathlib import Path

import torch

from scratchllm.model.gpt import GPT, GPTConfig

ROOT = Path(__file__).resolve().parents[2]  # scripts/archive/ -> the repository root
EVIDENCE = ROOT / "reports/eval/final-d044-gqa"
AR = ["fw2_arb", "fw2_ars", "fw2_ary", "fw2_arz", "wiki_ar"]
EN = "fineweb_edu_10bt"
UNSEEN = ["wiki_ary", "fw2_acm", "fw2_apc"]
SAME, WIKI, SYSTEM, UNRESOLVED_HI = 0.005, 0.05, 0.10, 0.015  # D-044
SEEDS = ("1337", "2024")
RUNS = {
    "ref1337": "ablation-rope-swiglu",
    "ref2024": "ablation-rope-swiglu-seed2024",
    "gqa1337": "ablation-gqa2",
    "gqa2024": "ablation-gqa2-seed2024",
}


def events(name: str) -> list[dict]:
    path = ROOT / f"runs/{name}/metrics.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


log = {k: events(v) for k, v in RUNS.items()}
ev = {k: [e for e in v if e["event"] == "eval"] for k, v in log.items()}
notes = []
for k, v in log.items():
    end = [e for e in v if e["event"] == "end"][0]
    assert end["step"] == 5000 and end["tokens_seen"] == 163_840_000, k
    if end["skipped_steps"] or [e for e in v if e["event"] in ("resume", "interrupted")]:
        notes.append(f"{RUNS[k]}: skipped {end['skipped_steps']}, resumed or interrupted")
for seed in SEEDS:  # each pair differs in n_kv_heads only
    a, b = (json.loads(json.dumps(log[f"{x}{seed}"][0]["config"])) for x in ("gqa", "ref"))
    assert a["model"]["n_kv_heads"] == 2 and b["model"].get("n_kv_heads") is None, seed
    for c in (a, b):
        c["model"].pop("n_kv_heads", None)
        c["name"] = ""
    assert a == b, seed
EVIDENCE.mkdir(parents=True, exist_ok=True)
(EVIDENCE / "evals.json").write_text(
    json.dumps({k: {"run": RUNS[k], "evals": ev[k]} for k in RUNS}, indent=1) + "\n",
    encoding="utf-8",
)


def at(k: str) -> dict:
    return [e for e in ev[k] if e["step"] == 5000][-1]


def mean_ar(e: dict) -> float:
    return statistics.mean(e["sources"][s]["bits_per_char"] for s in AR)


def run_stats(k: str) -> tuple[float, float]:
    train = [e for e in log[k] if e["event"] == "train"]
    return statistics.median(e["tokens_per_s"] for e in train), max(
        e.get("peak_memory_mib") or 0 for e in train
    )


rep = {
    k: {
        (r["source"], r["split"]): r["bits_per_char"]
        for r in json.loads((ROOT / f"reports/eval/final-{v}/report.json").read_text())["rows"]
    }
    for k, v in RUNS.items()
}
res = {}
for seed in SEEDS:
    g, r = f"gqa{seed}", f"ref{seed}"
    tg, mg = run_stats(g)
    tr, mr = run_stats(r)
    res[seed] = {
        "ar": mean_ar(at(g)) - mean_ar(at(r)),
        "en": at(g)["sources"][EN]["bits_per_char"] - at(r)["sources"][EN]["bits_per_char"],
        **{u: rep[g][(u, "eval")] - rep[r][(u, "eval")] for u in UNSEEN},
        "speed": tg / tr - 1, "mem": mg / mr - 1, "tps": (tg, tr), "peak": (mg, mr),
        "abs_ar": (mean_ar(at(g)), mean_ar(at(r))),
        "abs_en": (at(g)["sources"][EN]["bits_per_char"], at(r)["sources"][EN]["bits_per_char"]),
        "src": {s: at(g)["sources"][s]["bits_per_char"] - at(r)["sources"][s]["bits_per_char"] for s in AR},
    }  # fmt: skip


def band(key: str) -> str:
    values = [res[s][key] for s in SEEDS]
    m = statistics.mean(values)
    if (values[0] > 0) != (values[1] > 0) and max(abs(v) for v in values) >= SAME:
        return "unresolved (seeds disagree in sign)"
    if abs(m) < SAME:
        return "the same"
    if abs(m) < UNRESOLVED_HI:
        return "unresolved"
    return "differs"


mean = {k: statistics.mean(res[s][k] for s in SEEDS) for k in ("ar", "en", "speed", "mem")}
holds = {
    "Arabic mean": mean["ar"] < SAME,
    "English": mean["en"] < SAME,
    "`wiki_ary`": all(res[s]["wiki_ary"] < WIKI for s in SEEDS),
    "speed": all(res[s]["speed"] > -SYSTEM for s in SEEDS),
    "memory": all(res[s]["mem"] < SYSTEM for s in SEEDS),
}
yes = lambda b: "yes" if b else "**no**"  # noqa: E731
out = [
    "| criterion (D-044): GQA minus the reference | required | seed 1337 | seed 2024 | mean | band (D-031) | holds |",
    "|---|---|---:|---:|---:|---|---|",
    f"| Arabic mean | mean below +{SAME} | {res['1337']['ar']:+.4f} | {res['2024']['ar']:+.4f} | {mean['ar']:+.4f} | {band('ar')} | {yes(holds['Arabic mean'])} |",
    f"| English | mean below +{SAME} | {res['1337']['en']:+.4f} | {res['2024']['en']:+.4f} | {mean['en']:+.4f} | {band('en')} | {yes(holds['English'])} |",
    f"| `wiki_ary` | below +{WIKI} at both | {res['1337']['wiki_ary']:+.4f} | {res['2024']['wiki_ary']:+.4f} | | | {yes(holds['`wiki_ary`'])} |",
    f"| speed (median tokens/s) | not more than {SYSTEM:.0%} below | {res['1337']['speed']:+.1%} | {res['2024']['speed']:+.1%} | | | {yes(holds['speed'])} |",
    f"| peak memory | not more than {SYSTEM:.0%} above | {res['1337']['mem']:+.1%} | {res['2024']['mem']:+.1%} | | | {yes(holds['memory'])} |",
    "",
    "| bits/char at step 5,000 | Arabic mean, GQA | reference | English, GQA | reference | tokens/s, GQA | reference | peak MiB, GQA | reference |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]  # fmt: skip
for s in SEEDS:
    x = res[s]
    out.append(
        f"| seed {s} | {x['abs_ar'][0]:.4f} | {x['abs_ar'][1]:.4f} | {x['abs_en'][0]:.4f} | {x['abs_en'][1]:.4f} | "
        f"{x['tps'][0]:,.0f} | {x['tps'][1]:,.0f} | {x['peak'][0]:.0f} | {x['peak'][1]:.0f} |"
    )
out += ["", "| GQA minus the reference, per source | " + " | ".join(AR) + " | " + " | ".join(f"`{u}`" for u in UNSEEN) + " |",
        "|---|" + "---:|" * (len(AR) + len(UNSEEN))]  # fmt: skip
for s in SEEDS:
    out.append(
        f"| seed {s} | "
        + " | ".join(f"{res[s]['src'][x]:+.3f}" for x in AR)
        + " | "
        + " | ".join(f"{res[s][u]:+.3f}" for u in UNSEEN)
        + " |"
    )

# parameters and the cache of one full window, by construction
sizes = []
for kv in (None, 2):
    cfg = GPTConfig(**{**log["ref1337"][0]["config"]["model"], "n_kv_heads": kv})
    model = GPT(cfg).eval()
    cache = model.new_cache()
    with torch.no_grad():
        model(torch.zeros(1, cfg.context_length, dtype=torch.long), cache=cache)
    sizes.append((model.num_parameters(), cache.nbytes()))
out += [
    "",
    "| | parameters | key/value cache, one 512-token sequence, float32 |",
    "|---|---:|---:|",
    f"| reference (6 key/value heads) | {sizes[0][0]:,} | {sizes[0][1] / 2**20:.1f} MiB |",
    f"| GQA (2 key/value heads) | {sizes[1][0]:,} | {sizes[1][1] / 2**20:.1f} MiB |",
]
adopt = all(holds.values())
out += [
    "",
    f"**Verdict (D-044):** {'all conditions hold: GQA is adopted' if adopt else 'not all conditions hold: the reference stays multi-head'}.",
]
if notes:
    out += ["", "Run notes: " + "; ".join(notes)]
Path(sys.argv[1]).write_text("\n".join(out) + "\n", encoding="utf-8")
print(
    json.dumps(
        {"holds": holds, "bands": {k: band(k) for k in ("ar", "en")}, "mean": mean}, indent=1
    )
)
