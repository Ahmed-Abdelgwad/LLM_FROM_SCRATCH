# ruff: noqa: E501
"""Builds the D-033 entry of docs/DECISIONS.md: SwiGLU alone at a second seed, judged by the criterion
written in `configs/ablation-swiglu-only-seed2024.yaml` before the run. Tables and numbers come from the run
files and the full evaluations; prose is by hand and comparison sentences are guarded by asserts.

Also writes the evaluation events of the four runs to reports/eval/final-d033-swiglu-seeds/evals.json.
Needs runs/{local-17m,local-17m-seed2024,ablation-swiglu-only,ablation-swiglu-only-seed2024} and the full
evaluations reports/eval/final-{local-17m,local-17m-seed2024,modern-17m,modern-17m-seed2024,
ablation-swiglu-only,ablation-swiglu-only-seed2024}.
Run from the repository root: uv run python scripts/d033_entry.py OUT.md"""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "reports/eval/final-d033-swiglu-seeds"
AR = ["fw2_arb", "fw2_ars", "fw2_ary", "fw2_arz", "wiki_ar"]
EN = "fineweb_edu_10bt"
UNSEEN = ["fw2_acm", "fw2_apc", "wiki_ary"]
SHORT = {EN: "edu", "fw2_arb": "arb", "fw2_ars": "ars", "fw2_ary": "ary", "fw2_arz": "arz",
         "wiki_ar": "wiki_ar", "fw2_acm": "acm", "fw2_apc": "apc", "wiki_ary": "wiki_ary"}  # fmt: skip
RUNS = {
    "b1337": "local-17m",
    "s1337": "ablation-swiglu-only",
    "b2024": "local-17m-seed2024",
    "s2024": "ablation-swiglu-only-seed2024",
}
REPORTS = {"b1337": "final-local-17m", "s1337": "final-ablation-swiglu-only", "m1337": "final-modern-17m", "r1337": "final-ablation-rope-only",
           "b2024": "final-local-17m-seed2024", "s2024": "final-ablation-swiglu-only-seed2024", "m2024": "final-modern-17m-seed2024"}  # fmt: skip
LO, HI, WIKI, SYSTEM = 0.005, 0.015, 0.05, 0.10  # D-031 bands, wiki_ary line, speed and memory


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
    assert v[0]["config"]["train"]["seed"] == (2024 if "2024" in k else 1337), k
defaults = {"norm": "layernorm", "position": "learned", "ffn": "gelu", "rope_theta": 10000.0}
a, b = (json.loads(json.dumps(log[x][0]["config"])) for x in ("s1337", "s2024"))
for c in (a, b):
    c["model"] = defaults | c["model"]
    c["name"] = ""
    c["train"]["seed"] = 0
assert a == b and a["model"]["ffn"] == "swiglu"  # the two SwiGLU runs differ in the seed only
assert not [e for e in log["s2024"] if e["event"] in ("resume", "interrupted")]
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


res = {}
for seed in ("1337", "2024"):
    B, S = at(f"b{seed}"), at(f"s{seed}")
    tb, mb = run_stats(f"b{seed}")
    ts, ms = run_stats(f"s{seed}")
    res[seed] = {
        "ar": mean_ar(S) - mean_ar(B), "en": S["sources"][EN]["bits_per_char"] - B["sources"][EN]["bits_per_char"],
        "src": {s: S["sources"][s]["bits_per_char"] - B["sources"][s]["bits_per_char"] for s in AR},
        "speed": ts / tb - 1, "mem": ms / mb - 1,
    }  # fmt: skip
mean_effect = statistics.mean(r["ar"] for r in res.values())
size = abs(mean_effect)
cls = (
    "neutral"
    if size < LO
    else "unresolved"
    if size <= HI
    else ("helps" if mean_effect < 0 else "hurts")
)
assert cls == "unresolved"
efficient = all(r["speed"] >= SYSTEM and r["mem"] <= -SYSTEM for r in res.values())
assert efficient

t = [
    "| seed | Arabic effect | arb | ars | ary | arz | wiki_ar | English effect | speed | peak memory |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for seed, r in res.items():
    t.append(
        f"| {seed} | {r['ar']:+.4f} | "
        + " | ".join(f"{r['src'][s]:+.3f}" for s in AR)
        + f" | {r['en']:+.4f} | {r['speed']:+.0%} | {r['mem']:+.0%} |"
    )
t.append(
    f"| **mean of the two** | **{mean_effect:+.4f}** | | | | | | {statistics.mean(r['en'] for r in res.values()):+.4f} | | |"
)

rep = {
    k: {
        (r["source"], r["split"]): r["bits_per_char"]
        for r in json.loads((ROOT / f"reports/eval/{REPORTS[k]}/report.json").read_text())["rows"]
    }
    for k in REPORTS
}


def d(k: str, base: str, s: str, split: str = "eval") -> float:
    return rep[k][(s, split)] - rep[base][(s, split)]


w = {"1337": d("s1337", "b1337", "wiki_ary"), "2024": d("s2024", "b2024", "wiki_ary")}
pkg = {"1337": d("m1337", "b1337", "wiki_ary"), "2024": d("m2024", "b2024", "wiki_ary")}
both = all(x >= WIKI for x in w.values())
assert w["1337"] >= WIKI > w["2024"] and not both
share = {k: w[k] / pkg[k] for k in w}
rope_wiki = d("r1337", "b1337", "wiki_ary")
assert 0 < rope_wiki < WIKI
assert share["1337"] > 0.7 and share["2024"] < 0.5
un = [
    "| change vs the baseline of the same seed | seed 1337 SwiGLU | seed 2024 SwiGLU | seed 1337 all three | seed 2024 all three |",
    "|---|---:|---:|---:|---:|",
]
for s in UNSEEN:
    un.append(
        f"| {SHORT[s]} | {d('s1337', 'b1337', s):+.3f} | {d('s2024', 'b2024', s):+.3f} | {d('m1337', 'b1337', s):+.3f} | {d('m2024', 'b2024', s):+.3f} |"
    )
for sp in ("val", "test"):
    un.append(
        f"| trained Arabic {sp} (mean) | "
        + " | ".join(
            f"{statistics.mean(d(k, base, s, sp) for s in AR):+.3f}"
            for k, base in (
                ("s1337", "b1337"),
                ("s2024", "b2024"),
                ("m1337", "b1337"),
                ("m2024", "b2024"),
            )
        )
        + " |"
    )

text = f"""## D-033 (2026-11-07): SwiGLU alone at a second seed: the Arabic effect is still unresolved, `wiki_ary` is not worse at both seeds, and the keep decision does not depend on the first

Code: none changed. Added: `configs/ablation-swiglu-only-seed2024.yaml`, committed before the run with its criterion in the header. Evidence kept in git:
`reports/eval/final-ablation-swiglu-only-seed2024/` (full evaluation) and `reports/eval/final-d033-swiglu-seeds/evals.json` (the evaluation events of the four runs compared).
Script: `scripts/d033_entry.py` computed the tables and the application of the criterion (prose by hand; comparison sentences guarded by asserts). Not in git: the runs.

**Why it was run.** D-032 left SwiGLU open: its Arabic effect at seed 1337 ({res["1337"]["ar"]:+.4f}) lay in the unresolved band of D-031, and `wiki_ary` was worse by {w["1337"]:+.3f}, so the rules answered differently by how the Arabic effect resolved.
**The criterion** (in the config header, before the run): the Arabic effect classified on the mean of the two seeds' effects, with the D-031 bands; `wiki_ary` worse only if {WIKI} or more at both seeds; then rules 1 to 5 of D-031; if the Arabic mean is still
unresolved, SwiGLU stays open and nothing more is run for it without a new decision.

The second run is the baseline with SwiGLU on, seed 2024, 5,000 steps, 163,840,000 tokens, not interrupted, 0 skipped steps; the script checks that it differs from the seed-1337 SwiGLU run in the seed only. Effects are against the baseline of the same seed:

{chr(10).join(t)}

**Applying the criterion.**

- *Arabic mean:* the two effects are {res["1337"]["ar"]:+.4f} and {res["2024"]["ar"]:+.4f}; their mean {mean_effect:+.4f} is {size:.4f} in size, inside the unresolved band ({LO} to {HI}). **It is still unresolved.** The sign and the order of magnitude are the same at both seeds and
  all five sources move the same way at both; the second seed's effect is the larger (the difference between the seeds, {abs(res["2024"]["ar"] - res["1337"]["ar"]):.4f}, is larger than the 0.0033 seen between baselines in D-029). The band decides nothing further here.
- *`wiki_ary`:* {w["1337"]:+.3f} at seed 1337 and {w["2024"]:+.3f} at seed 2024. It reaches {WIKI} at one seed and not at the other, so **by the criterion it is "no difference"**.
- *Speed and memory:* {res["1337"]["speed"]:+.0%} and {res["2024"]["speed"]:+.0%} in tokens per second, {res["1337"]["mem"]:+.0%} and {res["2024"]["mem"]:+.0%} in peak memory: beyond the {SYSTEM:.0%} line at both seeds.
- *Rules:* with `wiki_ary` not counted as worse, the rules give the same answer for either way the Arabic effect could resolve: if it helps, rule 1 (keep); if it is neutral, rule 2 (keep, since it is faster and lighter at both seeds).
  **Under the registered rules SwiGLU is kept whichever way the unresolved Arabic effect would fall.** The sentence in the header, "SwiGLU stays open", refers to the Arabic classification, which does stay unresolved; this entry does not rewrite it. Whether the project treats the keep as settled is its decision.

**Unseen dialects and the trained languages** (full evaluation, bits/char change against the baseline of the same seed):

{chr(10).join(un)}

**What the second seed does to D-032.** D-032 attributed most of the `wiki_ary` loss of the modern package to SwiGLU: {w["1337"]:+.3f} of {pkg["1337"]:+.3f} at seed 1337 ({share["1337"]:.0%}). At seed 2024 SwiGLU alone gives {w["2024"]:+.3f} against {pkg["2024"]:+.3f} for the package ({share["2024"]:.0%}).
The share is not stable: with unseen sources whose seed-to-seed difference reaches 0.03 (D-029), two seeds give {w["1337"]:+.3f} and {w["2024"]:+.3f}, mean {statistics.mean(w.values()):+.3f}, against a package mean of {statistics.mean(pkg.values()):+.3f}.
So **the attribution of the `wiki_ary` loss to SwiGLU is weakened, not confirmed**: SwiGLU contributes something, probably less than D-032 said. The rotary embeddings alone gave {rope_wiki:+.3f} at seed 1337 (under the {WIKI} line) and have no second seed;
nothing here says which of the two, or their combination, is responsible for the rest. D-032's decision 2 is restated as: SwiGLU contributes to the `wiki_ary` loss (about {statistics.mean(w.values()) / statistics.mean(pkg.values()):.0%} of the package's on average over two seeds); the rest is not attributed.
On the other two unseen dialects SwiGLU alone does nothing (`acm` {d("s1337", "b1337", "fw2_acm"):+.3f} and {d("s2024", "b2024", "fw2_acm"):+.3f}, `apc` {d("s1337", "b1337", "fw2_apc"):+.3f} and {d("s2024", "b2024", "fw2_apc"):+.3f}), at both seeds.

**Decisions.**

1. SwiGLU's Arabic effect stays **unresolved** ({mean_effect:+.4f}, two seeds); no further run is made for it (as registered).
2. By the registered rules SwiGLU is **kept** under either resolution of that effect (rules 1 and 2), pending the project's confirmation that the header's "stays open" does not hold the keep back: it improves the trained languages slightly at both seeds, and it is {statistics.mean(r["speed"] for r in res.values()):+.0%} faster and {statistics.mean(r["mem"] for r in res.values()):+.0%} in peak memory. `configs/modern-17m.yaml` keeps it.
3. The `wiki_ary` loss is not attributed to a single component: SwiGLU contributes part of it, and the rotary embeddings' contribution rests on one seed.
4. Rotary embeddings stay kept (D-032, rule 1); RMSNorm stays a candidate to drop, not dropped.

**Not decided, and not tested here.** A second seed for rotary embeddings alone (to see how much of `wiki_ary` they carry); a run of the package without RMSNorm; whether `wiki_ary` should matter more than a 0.01 to 0.02 gain on the trained languages, a project decision (as in D-024, D-025) now that no component is shown to cause it alone.
"""
out = Path(sys.argv[1])
out.write_text(text, encoding="utf-8")
print(f"wrote {out}: {len(text.splitlines())} lines")
