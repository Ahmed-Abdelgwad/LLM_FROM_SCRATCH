# ruff: noqa: E501
"""Builds the D-035 entry of docs/DECISIONS.md: the two follow-up runs of D-034 (the package without RMSNorm;
rotary embeddings alone at a second seed), judged by the criteria written in their configs and in D-034 before the
runs. Tables and numbers come from the run files and the full evaluations; prose is by hand and comparison sentences
are guarded by asserts.

Also writes the evaluation events of the runs compared to reports/eval/final-d035-followups/evals.json.
Needs runs/{local-17m,local-17m-seed2024,modern-17m,ablation-rope-only,ablation-rope-only-seed2024,
ablation-rope-swiglu,ablation-swiglu-only,ablation-swiglu-only-seed2024} and the full evaluations of the
same names in reports/eval (final-<name>; the modern package is final-modern-17m[-seed2024]).
Run from the repository root: uv run python scripts/archive/d035_entry.py OUT.md"""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # scripts/archive/ -> the repository root
EVIDENCE = ROOT / "reports/eval/final-d035-followups"
AR = ["fw2_arb", "fw2_ars", "fw2_ary", "fw2_arz", "wiki_ar"]
EN = "fineweb_edu_10bt"
SHORT = {
    EN: "edu",
    "fw2_arb": "arb",
    "fw2_ars": "ars",
    "fw2_ary": "ary",
    "fw2_arz": "arz",
    "wiki_ar": "wiki_ar",
}
LO, HI, WIKI, SYSTEM, DROP = 0.005, 0.015, 0.05, 0.10, 0.005  # D-031, D-034
RUNS = {
    "base1337": "local-17m", "base2024": "local-17m-seed2024", "pkg1337": "modern-17m",
    "rope1337": "ablation-rope-only", "rope2024": "ablation-rope-only-seed2024",
    "swiglu1337": "ablation-swiglu-only", "swiglu2024": "ablation-swiglu-only-seed2024",
    "norms": "ablation-rope-swiglu",
}  # fmt: skip
REPORTS = {
    "base1337": "final-local-17m", "base2024": "final-local-17m-seed2024", "pkg1337": "final-modern-17m",
    "pkg2024": "final-modern-17m-seed2024", "rope1337": "final-ablation-rope-only",
    "rope2024": "final-ablation-rope-only-seed2024", "swiglu1337": "final-ablation-swiglu-only",
    "swiglu2024": "final-ablation-swiglu-only-seed2024", "norms": "final-ablation-rope-swiglu",
    "rms1337": "final-ablation-rmsnorm-only",
}  # fmt: skip


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
resumes = {
    k: [(e["event"], e["step"]) for e in v if e["event"] in ("resume", "interrupted")]
    for k, v in log.items()
}
assert resumes["rope2024"] == [("resume", 4500)] and resumes["norms"] == []
defaults = {"norm": "layernorm", "position": "learned", "ffn": "gelu", "rope_theta": 10000.0}


def cfg(k: str) -> dict:
    c = json.loads(json.dumps(log[k][0]["config"]))
    c["model"] = defaults | c["model"]
    c["name"] = ""
    return c


a, b = cfg("rope1337"), cfg("rope2024")
a["train"]["seed"] = b["train"]["seed"] = 0
assert a == b  # the two rotary-alone runs differ in the seed only
a, b = cfg("norms"), cfg("pkg1337")
assert a["model"]["norm"] == "layernorm" and b["model"]["norm"] == "rmsnorm"
a["model"]["norm"] = b["model"]["norm"] = ""
assert a == b  # the package without RMSNorm differs from the package in the norm only
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


def wiki(k: str, base: str) -> float:
    return rep[k][("wiki_ary", "eval")] - rep[base][("wiki_ary", "eval")]


# ---------------- B: the package without RMSNorm, against the package (seed 1337)
diff_ar = mean_ar(at("norms")) - mean_ar(at("pkg1337"))
diff_en = (
    at("norms")["sources"][EN]["bits_per_char"] - at("pkg1337")["sources"][EN]["bits_per_char"]
)
diff_wiki = rep["norms"][("wiki_ary", "eval")] - rep["pkg1337"][("wiki_ary", "eval")]
tn, mn = run_stats("norms")
tp, mp = run_stats("pkg1337")
speed, mem = tn / tp - 1, mn / mp - 1
checks = {
    "Arabic mean": (diff_ar, diff_ar < DROP, f"below +{DROP}"),
    "English": (diff_en, diff_en < DROP, f"below +{DROP}"),
    "`wiki_ary`": (diff_wiki, diff_wiki < WIKI, f"below +{WIKI}"),
    "speed": (speed, speed > -SYSTEM, f"not more than {SYSTEM:.0%} below"),
    "memory": (mem, mem < SYSTEM, f"not more than {SYSTEM:.0%} above"),
}
all_hold = all(ok for _, ok, _ in checks.values())
assert all_hold
tb = [
    "| criterion (D-034) | this run minus `modern-17m` | required | holds |", "|---|---:|---|---|",
    *[f"| {n} | {v:+.4f} | {r} | {'yes' if ok else 'NO'} |" for n, (v, ok, r) in checks.items()],
]  # fmt: skip
b_per_source = {
    s: at("norms")["sources"][s]["bits_per_char"] - at("pkg1337")["sources"][s]["bits_per_char"]
    for s in AR
}
vs_base = mean_ar(at("norms")) - mean_ar(at("base1337"))

# ---------------- A: rotary embeddings alone, two seeds
eff = {}
for seed in ("1337", "2024"):
    B0, R = at(f"base{seed}"), at(f"rope{seed}")
    t_r, m_r = run_stats(f"rope{seed}")
    t_b, m_b = run_stats(f"base{seed}")
    eff[seed] = {
        "ar": mean_ar(R) - mean_ar(B0), "en": R["sources"][EN]["bits_per_char"] - B0["sources"][EN]["bits_per_char"],
        "wiki": wiki(f"rope{seed}", f"base{seed}"), "pkg_wiki": wiki(f"pkg{seed}", f"base{seed}"),
        "speed": t_r / t_b - 1, "mem": m_r / m_b - 1,
        "src": {s: R["sources"][s]["bits_per_char"] - B0["sources"][s]["bits_per_char"] for s in AR},
    }  # fmt: skip
mean_effect = statistics.mean(e["ar"] for e in eff.values())
assert abs(mean_effect) > HI and mean_effect < 0  # helps
assert all(e["wiki"] < WIKI for e in eff.values()) and all(e["wiki"] > 0 for e in eff.values())
share = {s: eff[s]["wiki"] / eff[s]["pkg_wiki"] for s in eff}
sw = {s: wiki(f"swiglu{s}", f"base{s}") for s in ("1337", "2024")}
ta = [
    "| seed | Arabic effect | English effect | `wiki_ary` effect | package `wiki_ary` | rope share of package | speed | memory |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for s, e in eff.items():
    ta.append(
        f"| {s} | {e['ar']:+.4f} | {e['en']:+.4f} | {e['wiki']:+.3f} | {e['pkg_wiki']:+.3f} | {share[s]:.0%} | {e['speed']:+.0%} | {e['mem']:+.0%} |"
    )
ta.append(
    f"| mean | {mean_effect:+.4f} | {statistics.mean(e['en'] for e in eff.values()):+.4f} | {statistics.mean(e['wiki'] for e in eff.values()):+.3f} | {statistics.mean(e['pkg_wiki'] for e in eff.values()):+.3f} | | | |"
)

# ---------------- the `wiki_ary` decomposition
dec = [
    "| `wiki_ary`, change vs the baseline of the same seed | seed 1337 | seed 2024 |",
    "|---|---:|---:|",
]
dec.append(f"| rotary embeddings alone | {eff['1337']['wiki']:+.3f} | {eff['2024']['wiki']:+.3f} |")
dec.append(f"| SwiGLU alone | {sw['1337']:+.3f} | {sw['2024']:+.3f} |")
rms = wiki("rms1337", "base1337")
dec.append(f"| RMSNorm alone | {rms:+.3f} | not run |")
dec.append(
    f"| sum of rotary and SwiGLU | {eff['1337']['wiki'] + sw['1337']:+.3f} | {eff['2024']['wiki'] + sw['2024']:+.3f} |"
)
dec.append(
    f"| the whole package | {eff['1337']['pkg_wiki']:+.3f} | {eff['2024']['pkg_wiki']:+.3f} |"
)
pair_wiki = wiki("norms", "base1337")
dec.append(f"| rotary + SwiGLU without RMSNorm (D-034 B) | {pair_wiki:+.3f} | not run |")
assert (
    abs(eff["1337"]["wiki"] + sw["1337"] - eff["1337"]["pkg_wiki"]) < 0.02
    and abs(eff["2024"]["wiki"] + sw["2024"] - eff["2024"]["pkg_wiki"]) < 0.02
)
assert abs(rms) < 0.01 and abs(pair_wiki - eff["1337"]["pkg_wiki"]) < 0.02

text = f"""## D-035 (2026-11-08): the two follow-ups: dropping RMSNorm costs nothing at seed 1337; the `wiki_ary` loss is the sum of rotary embeddings and SwiGLU, neither alone reaching the line

Code: none changed. Added: `configs/experiments/ablation-rope-swiglu.yaml` and `configs/experiments/ablation-rope-only-seed2024.yaml`, committed with their criteria before the runs (D-034). Evidence kept in git: the full evaluations
`reports/eval/final-ablation-rope-swiglu/` and `reports/eval/final-ablation-rope-only-seed2024/`, and `reports/eval/final-d035-followups/evals.json` (the evaluation events of the eight runs compared).
Script: `scripts/archive/d035_entry.py` computed the tables and the application of the criteria (prose by hand; comparison sentences guarded by asserts). Not in git: the runs.

**An interruption to disclose.** `ablation-rope-only-seed2024` was stopped once. The machine suspended at 19:34 on 2026-11-07, right after step 4,560 (systemd-logind: "The system will suspend now"); the process survived but its GPU context did not, and it made no progress
for about six and a half hours. It was killed, and resumed from `last.pt` at step 4,500 (`resume` event in its log), with sleep inhibited for the rest. The losses logged for steps 4,510 to 4,550 after the resume equal those logged before the suspend to four decimals,
so the resume reproduced the run on this GPU. It is the second run of this series that was resumed once (the baseline at seed 1337 was too, D-026); the criteria do not forbid it and are not changed. `ablation-rope-swiglu` was not interrupted.

**B. Does dropping RMSNorm cost anything?** `ablation-rope-swiglu` (rotary embeddings and SwiGLU, LayerNorm kept), seed 1337, against `modern-17m` (all three). The script checks the two configs differ in the norm only.
Arabic mean {mean_ar(at("norms")):.4f} against {mean_ar(at("pkg1337")):.4f}; English {at("norms")["sources"][EN]["bits_per_char"]:.4f} against {at("pkg1337")["sources"][EN]["bits_per_char"]:.4f}. Against the baseline the pair is {vs_base:+.4f} on the Arabic mean (the package: {mean_ar(at("pkg1337")) - mean_ar(at("base1337")):+.4f}).
Per source, the difference to the package is between {min(b_per_source.values()):+.3f} and {max(b_per_source.values()):+.3f}.

{chr(10).join(tb)}

**Every criterion of D-034 holds: RMSNorm is a candidate to drop at one seed.** Per D-034, the reference config is not changed yet: the same test must also hold at seed 2024 (a run of the package without RMSNorm at seed 2024, against `modern-17m-seed2024`), which has not been run.

**A. Rotary embeddings alone, second seed** (against the baseline of the same seed):

{chr(10).join(ta)}

- *Arabic:* the effects are {eff["1337"]["ar"]:+.4f} and {eff["2024"]["ar"]:+.4f}; the mean {mean_effect:+.4f} is well beyond the {HI} edge of the unresolved band: **helps**, at both seeds, for all five sources. English {eff["1337"]["en"]:+.4f} and {eff["2024"]["en"]:+.4f}.
- *`wiki_ary`:* {eff["1337"]["wiki"]:+.3f} and {eff["2024"]["wiki"]:+.3f}: positive at both seeds, below {WIKI} at both. **By the criterion it is "no difference" for rotary embeddings alone.** Rule 1 applies: keep. Speed {eff["1337"]["speed"]:+.0%} and {eff["2024"]["speed"]:+.0%}, memory {eff["1337"]["mem"]:+.0%} and {eff["2024"]["mem"]:+.0%}: no difference.
- *Share of the package's loss:* {share["1337"]:.0%} at seed 1337, {share["2024"]:.0%} at seed 2024.

**What the `wiki_ary` loss is made of** (full evaluation; line at {WIKI}, seed-to-seed noise on unseen sources up to 0.03):

{chr(10).join(dec)}

At both seeds the two singles add up to about the package's loss (within 0.02), RMSNorm alone does nothing, and the pair without RMSNorm has the whole package's loss. So **the loss on the Moroccan Wikipedia is the sum of two moderate parts: rotary embeddings, about {eff["1337"]["wiki"]:+.3f} and {eff["2024"]["wiki"]:+.3f}, and SwiGLU, about {sw["1337"]:+.3f} and {sw["2024"]:+.3f}**.
Neither alone reaches the {WIKI} line at both seeds, so rule 4 does not apply to either; the package as a whole is beyond it at both seeds (D-029). This replaces both D-029's suspicion (rotary embeddings) and D-032's attribution (SwiGLU): each contributes, none is the sole cause.

**Decisions.**

1. Rotary embeddings are kept (rule 1): a gain of about {-mean_effect:.3f} bits/char on the Arabic mean at two seeds, with a `wiki_ary` cost of about {statistics.mean(e["wiki"] for e in eff.values()):+.3f}, under the line.
2. SwiGLU stays as decided in D-033 (kept, pending the project's confirmation).
3. RMSNorm: a candidate to drop at one seed; **not dropped**. Dropping it would simplify the reference with no measured cost (quality, `wiki_ary`, speed within 6%, memory within 5%) but needs the seed-2024 confirmation D-034 requires.
4. The `wiki_ary` cost belongs to the package, not to one component: whether it is worth the gain on the trained languages is the project's decision, as the English share was (D-024, D-025). Dropping either component would give up some of the gain with it: at seed 1337 rotary embeddings carry about {eff["1337"]["ar"] / (mean_ar(at("pkg1337")) - mean_ar(at("base1337"))):.0%} of the package's Arabic gain.

**Not decided, and not tested here.** The seed-2024 run of the package without RMSNorm (the test D-034 requires before the reference changes); whether `wiki_ary` should weigh against the package at all, given that it is one of three unseen sources and noisy; the RMSNorm-alone cell at seed 2024 (not needed for any decision above).
"""
out = Path(sys.argv[1])
out.write_text(text, encoding="utf-8")
print(f"wrote {out}: {len(text.splitlines())} lines")
