# ruff: noqa: E501
"""Builds the D-036 entry of docs/DECISIONS.md: the package without RMSNorm at the second seed, judged by the
criterion of D-034 (repeated in `configs/ablation-rope-swiglu-seed2024.yaml` before the run), and the reference
architecture that follows. Tables and numbers come from the run files and the full evaluations; prose is by hand and
comparison sentences are guarded by asserts.

Also writes the evaluation events of the four runs to reports/eval/final-d036-no-rmsnorm/evals.json.
Needs runs/{modern-17m,modern-17m-seed2024,ablation-rope-swiglu,ablation-rope-swiglu-seed2024} and the full
evaluations final-{modern-17m,modern-17m-seed2024,ablation-rope-swiglu,ablation-rope-swiglu-seed2024} in reports/eval.
Run from the repository root: uv run python scripts/d036_entry.py OUT.md"""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "reports/eval/final-d036-no-rmsnorm"
AR = ["fw2_arb", "fw2_ars", "fw2_ary", "fw2_arz", "wiki_ar"]
EN = "fineweb_edu_10bt"
DROP, WIKI, SYSTEM, UNRESOLVED_HI = 0.005, 0.05, 0.10, 0.015  # D-034
RUNS = {
    "pkg1337": "modern-17m",
    "pkg2024": "modern-17m-seed2024",
    "no1337": "ablation-rope-swiglu",
    "no2024": "ablation-rope-swiglu-seed2024",
}
REPORTS = {
    "pkg1337": "final-modern-17m",
    "pkg2024": "final-modern-17m-seed2024",
    "no1337": "final-ablation-rope-swiglu",
    "no2024": "final-ablation-rope-swiglu-seed2024",
}


def events(name: str) -> list[dict]:
    path = ROOT / f"runs/{name}/metrics.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


log = {k: events(v) for k, v in RUNS.items()}
ev = {k: [e for e in v if e["event"] == "eval"] for k, v in log.items()}
for k, v in log.items():
    end = [e for e in v if e["event"] == "end"][0]
    assert (
        end["step"] == 5000 and end["skipped_steps"] == 0 and end["tokens_seen"] == 163_840_000
    ), k
    assert not [e for e in v if e["event"] in ("resume", "interrupted")], k
defaults = {"norm": "layernorm", "position": "learned", "ffn": "gelu", "rope_theta": 10000.0}
for seed in ("1337", "2024"):  # each pair differs in the norm only
    a, b = (json.loads(json.dumps(log[f"{x}{seed}"][0]["config"])) for x in ("no", "pkg"))
    assert a["model"]["norm"] == "layernorm" and b["model"]["norm"] == "rmsnorm", seed
    a["model"]["norm"] = b["model"]["norm"] = ""
    a["name"] = b["name"] = ""
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
        e.get("peak_memory_mib", 0) for e in train
    )


rep = {
    k: {
        (r["source"], r["split"]): r["bits_per_char"]
        for r in json.loads((ROOT / f"reports/eval/{REPORTS[k]}/report.json").read_text())["rows"]
    }
    for k in REPORTS
}
res = {}
for seed in ("1337", "2024"):
    no, pkg = f"no{seed}", f"pkg{seed}"
    t_no, m_no = run_stats(no)
    t_pkg, m_pkg = run_stats(pkg)
    res[seed] = {
        "ar": mean_ar(at(no)) - mean_ar(at(pkg)),
        "en": at(no)["sources"][EN]["bits_per_char"] - at(pkg)["sources"][EN]["bits_per_char"],
        "wiki": rep[no][("wiki_ary", "eval")] - rep[pkg][("wiki_ary", "eval")],
        "acm": rep[no][("fw2_acm", "eval")] - rep[pkg][("fw2_acm", "eval")],
        "apc": rep[no][("fw2_apc", "eval")] - rep[pkg][("fw2_apc", "eval")],
        "speed": t_no / t_pkg - 1, "mem": m_no / m_pkg - 1,
        "src": {s: at(no)["sources"][s]["bits_per_char"] - at(pkg)["sources"][s]["bits_per_char"] for s in AR},
    }  # fmt: skip
    r = res[seed]
    r["holds"] = {
        "Arabic mean": r["ar"] < DROP,
        "English": r["en"] < DROP,
        "`wiki_ary`": r["wiki"] < WIKI,
        "speed": r["speed"] > -SYSTEM,
        "memory": r["mem"] < SYSTEM,
    }
    assert all(r["holds"].values()), seed
    assert abs(r["ar"]) < UNRESOLVED_HI

t = [
    "| criterion (D-034): this run minus the package | required | seed 1337 | holds | seed 2024 | holds |", "|---|---|---:|---|---:|---|",
    f"| Arabic mean | below +{DROP} | {res['1337']['ar']:+.4f} | yes | {res['2024']['ar']:+.4f} | yes |",
    f"| English | below +{DROP} | {res['1337']['en']:+.4f} | yes | {res['2024']['en']:+.4f} | yes |",
    f"| `wiki_ary` | below +{WIKI} | {res['1337']['wiki']:+.4f} | yes | {res['2024']['wiki']:+.4f} | yes |",
    f"| speed | not more than {SYSTEM:.0%} below | {res['1337']['speed']:+.1%} | yes | {res['2024']['speed']:+.1%} | yes |",
    f"| memory | not more than {SYSTEM:.0%} above | {res['1337']['mem']:+.1%} | yes | {res['2024']['mem']:+.1%} | yes |",
]  # fmt: skip
ts = ["| difference to the package, per source | " + " | ".join(AR) + " |", "|---|" + "---:|" * 5]
for seed in ("1337", "2024"):
    ts.append(f"| seed {seed} | " + " | ".join(f"{res[seed]['src'][s]:+.3f}" for s in AR) + " |")
speeds = [mean_ar(at(f"no{s}")) for s in ("1337", "2024")]

text = f"""## D-036 (2026-11-08): RMSNorm is dropped from the reference: the package without it matches the package at both seeds; `reference-17m` is rotary embeddings + SwiGLU + LayerNorm

Code: none changed. Added: `configs/ablation-rope-swiglu-seed2024.yaml` (committed with its criterion before the run), `configs/reference-17m.yaml` (the new reference), and a test that every shipped config loads and
that the reference is rotary + SwiGLU + LayerNorm with 16,913,280 parameters. Evidence kept in git: `reports/eval/final-ablation-rope-swiglu-seed2024/` (full evaluation) and `reports/eval/final-d036-no-rmsnorm/evals.json`
(the evaluation events of the four runs). Script: `scripts/d036_entry.py` computed the tables and the application of the criterion (prose by hand; comparison sentences guarded by asserts). Not in git: the runs.

**What was run.** The package without RMSNorm (rotary embeddings, SwiGLU, LayerNorm) at seed 2024, against `modern-17m-seed2024` (all three); the script checks that each pair of configs differs in the norm only.
The run was not interrupted, 0 skipped steps, 163,840,000 tokens; the machine's sleep was inhibited during it. The criterion is D-034's, unchanged: dropping RMSNorm costs nothing only if all five conditions hold.

{chr(10).join(t)}

Per source, the difference to the package (bits/char):

{chr(10).join(ts)}

**All five conditions hold at both seeds.** The package without RMSNorm is, within noise, the package: the Arabic mean differs by {res["1337"]["ar"]:+.4f} and {res["2024"]["ar"]:+.4f} (the noise between seeds at this length is 0.0033, D-029), English by {res["1337"]["en"]:+.4f} and {res["2024"]["en"]:+.4f},
`wiki_ary` by {res["1337"]["wiki"]:+.3f} and {res["2024"]["wiki"]:+.3f}, the other unseen dialects by at most {max(abs(res[s][k]) for s in res for k in ("acm", "apc")):.3f}. It runs {res["1337"]["speed"]:+.0%} and {res["2024"]["speed"]:+.0%} in tokens per second and {res["1337"]["mem"]:+.0%} and {res["2024"]["mem"]:+.0%} in peak memory
(inside the {SYSTEM:.0%} band; LayerNorm is a little slower and heavier here, but not by the margin the criterion allowed for). Neither difference in quality is distinguishable from zero here; the criterion asked for evidence of no cost and this is that evidence at the precision of two seeds.

**Decisions.**

1. RMSNorm is dropped from the reference architecture, as D-034 and the config header fixed in advance: the criterion held at both seeds.
2. **`configs/reference-17m.yaml`** is the reference for later experiments: rotary position embeddings, SwiGLU feed-forward, LayerNorm, 16,913,280 parameters. It is the recipe of `ablation-rope-swiglu` (and of the seed-2024 run), so those runs are its results: seed 1337 Arabic mean {speeds[0]:.4f}, seed 2024 {speeds[1]:.4f}.
   `local-17m` remains the baseline of record for earlier numbers and `modern-17m` the record of D-028 and D-029; neither file changed.
3. The code keeps all the options (D-027): RMSNorm is dropped from the reference, not from the code.
4. The summary of the architecture work, D-028 to D-036: rotary embeddings carry the quality gain (about 0.06 bits/char on the Arabic mean, similar on English); SwiGLU adds a small gain that stays unresolved at the 0.01 threshold, and most of the speed and memory saving; RMSNorm adds nothing measurable; the cost is
   about +0.03 (rotary) and +0.04 to +0.07 (SwiGLU) bits/char on `wiki_ary`, a source neither was trained on, which together make the package's loss there. The reference keeps both for the gain and the efficiency, and the `wiki_ary` cost remains a project decision (D-024, D-025, D-035).

**Not decided, and not tested here.** Whether the `wiki_ary` cost should change the choice (for example a learned-position variant with SwiGLU, which would give up the rotary gain); effects at other sizes or longer training; grouped-query attention (not built).
"""
out = Path(sys.argv[1])
out.write_text(text, encoding="utf-8")
print(f"wrote {out}: {len(text.splitlines())} lines")
