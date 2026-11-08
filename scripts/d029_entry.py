# ruff: noqa: E501
"""Builds the D-029 entry of docs/DECISIONS.md: the modern architecture against the GPT-2 baseline, repeated
with a second seed (2024), judged by the criterion written in the configs before the runs. Tables and numbers
come from the run files and the full evaluations; the prose is written here and each sentence that depends on a
comparison is guarded by an assert, so the script stops rather than print a claim the numbers do not support.

Also writes the evaluation events of the four runs to reports/eval/final-d029-seed-replicate/evals.json.
Needs runs/{local,modern}-17m[-seed2024] and reports/eval/final-{local,modern}-17m[-seed2024].
Run from the repository root: uv run python scripts/d029_entry.py OUT.md"""

import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "reports/eval/final-d029-seed-replicate"
AR = ["fw2_arb", "fw2_ars", "fw2_ary", "fw2_arz", "wiki_ar"]
EN = "fineweb_edu_10bt"
UNSEEN = ["fw2_acm", "fw2_apc", "wiki_ary"]
SHORT = {EN: "edu", "fw2_arb": "arb", "fw2_ars": "ars", "fw2_ary": "ary", "fw2_arz": "arz",
         "wiki_ar": "wiki_ar", "fw2_acm": "acm", "fw2_apc": "apc", "wiki_ary": "wiki_ary"}  # fmt: skip
RULE_MEAN, RULE_ONE, RULE_EN = 0.03, 0.04, 0.03
RUNS = {
    "b1337": "local-17m",
    "m1337": "modern-17m",
    "b2024": "local-17m-seed2024",
    "m2024": "modern-17m-seed2024",
}
REPORTS = {
    "b1337": "final-local-17m",
    "m1337": "final-modern-17m",
    "b2024": "final-local-17m-seed2024",
    "m2024": "final-modern-17m-seed2024",
}


def events(name: str) -> list[dict]:
    path = ROOT / f"runs/{name}/metrics.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


log = {k: events(v) for k, v in RUNS.items()}
ev = {k: [e for e in v if e["event"] == "eval"] for k, v in log.items()}
for k in ("b2024", "m2024", "m1337"):  # the runs made for this comparison were not interrupted
    assert not [e for e in log[k] if e["event"] in ("resume", "interrupted")], k
for k, events_of_run in log.items():
    end = [e for e in events_of_run if e["event"] == "end"][0]
    assert (
        end["step"] == 5000 and end["skipped_steps"] == 0 and end["tokens_seen"] == 163_840_000
    ), k
    assert events_of_run[0]["config"]["train"]["seed"] == (2024 if "2024" in k else 1337), k
for a, b in (("b1337", "b2024"), ("m1337", "m2024")):  # every setting but the seed
    ca, cb = (
        json.loads(json.dumps(log[a][0]["config"])),
        json.loads(json.dumps(log[b][0]["config"])),
    )
    for c in (
        ca,
        cb,
    ):  # a run made before the D-027 switches existed lacks their keys: it ran the defaults
        c["model"] = {
            "norm": "layernorm",
            "position": "learned",
            "ffn": "gelu",
            "rope_theta": 10000.0,
        } | c["model"]
    ca["name"] = cb["name"] = ""
    ca["train"]["seed"] = cb["train"]["seed"] = 0
    assert ca == cb, (a, b)
EVIDENCE.mkdir(parents=True, exist_ok=True)
(EVIDENCE / "evals.json").write_text(
    json.dumps({k: {"run": RUNS[k], "evals": ev[k]} for k in RUNS}, indent=1) + "\n",
    encoding="utf-8",
)


def at(k: str, step: int) -> dict:
    return [e for e in ev[k] if e["step"] == step][-1]


def mean_ar(e: dict) -> float:
    return statistics.mean(e["sources"][s]["bits_per_char"] for s in AR)


def bpc(k: str, s: str, step: int = 5000) -> float:
    return at(k, step)["sources"][s]["bits_per_char"]


# ---- the rule, per seed
res: dict[str, dict[str, Any]] = {}
tab = [
    "| seed | run | " + " | ".join(SHORT[s] for s in AR) + " | Arabic mean | English |",
    "|---|---|" + "---:|" * 7,
]
for seed in ("1337", "2024"):
    b, m = f"b{seed}", f"m{seed}"
    d = {s: bpc(m, s) - bpc(b, s) for s in [*AR, EN]}
    dm = mean_ar(at(m, 5000)) - mean_ar(at(b, 5000))
    res[seed] = {"d": d, "dm": dm}
    assert dm <= -RULE_MEAN and all(d[s] <= -RULE_ONE for s in AR) and d[EN] < RULE_EN, seed
    for k, label in ((b, "baseline"), (m, "modern")):
        tab.append(
            f"| {seed} | {label} | "
            + " | ".join(f"{bpc(k, s):.3f}" for s in AR)
            + f" | {mean_ar(at(k, 5000)):.4f} | {bpc(k, EN):.4f} |"
        )
    tab.append(
        f"| {seed} | change | "
        + " | ".join(f"{d[s]:+.3f}" for s in AR)
        + f" | {dm:+.4f} | {d[EN]:+.4f} |"
    )
assert abs(mean_ar(at("b1337", 5000)) - 2.0153) < 5e-4

# ---- seed noise at the full length
noise: dict[str, dict[str, float]] = {}
for pair, label in ((("b1337", "b2024"), "baseline"), (("m1337", "m2024"), "modern")):
    noise[label] = {s: bpc(pair[1], s) - bpc(pair[0], s) for s in [*AR, EN]}
    noise[label]["mean"] = mean_ar(at(pair[1], 5000)) - mean_ar(at(pair[0], 5000))
noise_max_src = max(abs(v) for lab in noise.values() for k, v in lab.items() if k != "mean")
noise_max_mean = max(abs(lab["mean"]) for lab in noise.values())
nrows = [
    "| run pair (seed 2024 minus 1337) | "
    + " | ".join(SHORT[s] for s in AR)
    + " | Arabic mean | English |",
    "|---|" + "---:|" * 7,
]
for label, v in noise.items():
    nrows.append(
        f"| {label} | "
        + " | ".join(f"{v[s]:+.3f}" for s in AR)
        + f" | {v['mean']:+.4f} | {v[EN]:+.4f} |"
    )
assert noise_max_mean * 3 < RULE_MEAN and 2 * noise_max_src < RULE_ONE

# ---- the curve of the new seed
curve = [
    "| step | Arabic mean: modern minus baseline | English: modern minus baseline |",
    "|---:|---:|---:|",
]
cd = {}
for st in range(500, 5001, 500):
    cd[st] = (
        mean_ar(at("m2024", st)) - mean_ar(at("b2024", st)),
        bpc("m2024", EN, st) - bpc("b2024", EN, st),
    )
    curve.append(f"| {st:,} | {cd[st][0]:+.3f} | {cd[st][1]:+.3f} |")
assert all(dm <= -RULE_MEAN for dm, _ in cd.values()) and abs(cd[500][0]) > abs(cd[5000][0])

# ---- full evaluations
rep = {
    k: {
        (r["source"], r["split"]): r["bits_per_char"]
        for r in json.loads((ROOT / f"reports/eval/{REPORTS[k]}/report.json").read_text())["rows"]
    }
    for k in REPORTS
}
full = ["| source | split | seed 1337 change | seed 2024 change |", "|---|---|---:|---:|"]
for s, sp in [(s, sp) for s in [EN, *AR] for sp in ("val", "test")] + [(s, "eval") for s in UNSEEN]:
    full.append(
        f"| {SHORT[s]} | {sp} | {rep['m1337'][(s, sp)] - rep['b1337'][(s, sp)]:+.3f} | {rep['m2024'][(s, sp)] - rep['b2024'][(s, sp)]:+.3f} |"
    )
grp: dict[str, list[float]] = {}
for label, keys in (("trained Arabic, val (mean)", [(s, "val") for s in AR]), ("trained Arabic, test (mean)", [(s, "test") for s in AR]),
                    ("unseen dialects (mean)", [(s, "eval") for s in UNSEEN])):  # fmt: skip
    grp[label] = [
        statistics.mean(rep[m][k] - rep[b][k] for k in keys)
        for b, m in (("b1337", "m1337"), ("b2024", "m2024"))
    ]
    full.append(f"| **{label}** | | **{grp[label][0]:+.3f}** | **{grp[label][1]:+.3f}** |")
for seed in ("1337", "2024"):
    b, m = f"b{seed}", f"m{seed}"
    assert all(rep[m][(s, "test")] < rep[b][(s, "test")] for s in [*AR, EN])
    assert rep[m][("wiki_ary", "eval")] > rep[b][("wiki_ary", "eval")]
wiki_d = [
    rep[m][("wiki_ary", "eval")] - rep[b][("wiki_ary", "eval")]
    for b, m in (("b1337", "m1337"), ("b2024", "m2024"))
]
unseen_noise = {
    lab: [rep[b2][(s, "eval")] - rep[b1][(s, "eval")] for s in UNSEEN]
    for lab, b1, b2 in (("baseline", "b1337", "b2024"), ("modern", "m1337", "m2024"))
}
unseen_noise_max = max(abs(x) for v in unseen_noise.values() for x in v)
assert min(wiki_d) > 2 * unseen_noise_max
acm_apc = [
    rep[m][(s, "eval")] - rep[b][(s, "eval")]
    for s in ("fw2_acm", "fw2_apc")
    for b, m in (("b1337", "m1337"), ("b2024", "m2024"))
]
assert all(d < 0 for d in acm_apc)

text = f"""## D-029 (2026-11-06): the modern architecture, repeated with a second seed: the result holds on both, and the Moroccan Wikipedia is worse on both

Code: none changed. Added: `configs/experiments/local-17m-seed2024.yaml` and `configs/experiments/modern-17m-seed2024.yaml`. Evidence kept in git:
`reports/eval/final-local-17m-seed2024/`, `reports/eval/final-modern-17m-seed2024/` (full evaluations) and
`reports/eval/final-d029-seed-replicate/evals.json` (the evaluation events of the four runs). Script: `scripts/d029_entry.py`
computed the tables and numbers (prose by hand; comparisons guarded by asserts). Not in git: the runs (`runs/`, ignored).

**What was run.** D-028 left one step open: the same two configurations again with a different seed. The new pair differs from
`local-17m` and `modern-17m` only in `train.seed` (1337 -> 2024), which changes the initialization and the data order; the script checks
that the stored configurations are otherwise identical, and that all four runs ended at step 5,000 with 163,840,000 tokens and no skipped
step. The new runs were not interrupted (the first baseline was, once, at step 2,990). The criterion was written in the two new
config headers before the runs: per seed, the Arabic mean at least {RULE_MEAN} lower, a claim about one source {RULE_ONE}, English not worse by {RULE_EN} or more;
superiority is claimed only if both seeds pass; nothing about seed 1337 is touched.

{chr(10).join(tab)}

**By the rule, both seeds pass.** Seed 1337: Arabic mean {res["1337"]["dm"]:+.4f}, smallest per-source gain {min(-res["1337"]["d"][s] for s in AR):.3f}, English {res["1337"]["d"][EN]:+.4f}.
Seed 2024: Arabic mean {res["2024"]["dm"]:+.4f}, smallest per-source gain {min(-res["2024"]["d"][s] for s in AR):.3f}, English {res["2024"]["d"][EN]:+.4f}. The two seeds agree to {abs(res["1337"]["dm"] - res["2024"]["dm"]):.3f} bits/char on the Arabic mean.

**Seed noise at the full length** (the same configuration, seed 2024 minus seed 1337):

{chr(10).join(nrows)}

The largest difference between seeds is {noise_max_mean:.4f} on the Arabic mean and {noise_max_src:.3f} on any one source. The 0.013 quoted in D-023 and D-024 was measured at 800 steps;
at 5,000 steps the noise is much smaller, and the threshold of {RULE_MEAN} is at least {RULE_MEAN / noise_max_mean:.0f} times it. Two seeds are a rough estimate of noise, not a
precise one. The seed-1337 baseline was interrupted once, at step 2,990; its distance to the seed-2024 baseline ({noise["baseline"]["mean"]:+.4f}) is not larger than the modern pair's ({noise["modern"]["mean"]:+.4f}), neither of which was interrupted, so the interruption is not visible at this level (two seeds cannot exclude a small effect).

The advantage over training, seed 2024, has the same shape as seed 1337 (D-028): large early, narrowing, and flat for the last 1,500 steps:

{chr(10).join(curve)}

**Full evaluation** (modern minus baseline, bits per character; negative is lower):

{chr(10).join(full)}

On every trained source, English included, the modern model has a lower bits/char on test in both seeds. **On the unseen Moroccan Wikipedia (`wiki_ary`) it is higher in both seeds,
by {wiki_d[0]:+.3f} and {wiki_d[1]:+.3f}**. The seed-to-seed difference on the three unseen sources, measured the same way (seed 2024 minus 1337, per model), reaches {unseen_noise_max:.3f}, larger than on
the trained sources, so the unseen dialects are noisier; the loss on `wiki_ary` is {min(wiki_d) / unseen_noise_max:.1f} to {max(wiki_d) / unseen_noise_max:.1f} times that, and it appears in both seeds with the same sign. On the two other unseen dialects it is lower in both seeds
(`acm` {acm_apc[0]:+.3f} and {acm_apc[1]:+.3f}, `apc` {acm_apc[2]:+.3f} and {acm_apc[3]:+.3f}). The `acm` differences are within the unseen-source noise just given, the `apc` ones a little above it. The mean over the three unseen dialects is {grp["unseen dialects (mean)"][0]:+.3f} and {grp["unseen dialects (mean)"][1]:+.3f}, which is no improvement.
`wiki_ary` is Moroccan Wikipedia text; the training mix has Moroccan web text (`ary`) and Arabic Wikipedia (`wiki_ar`) but no Moroccan Wikipedia. Why only this
source is hurt was not investigated.

**What is claimed, and what is not.**

- Claimed, with two seeds, at this size (17M parameters, 6 layers) and this budget (164M tokens): on the Arabic sources and the English source it trains on,
  RMSNorm + rotary embeddings + SwiGLU together reach a lower bits/char than the GPT-2 parts, by about {-res["2024"]["dm"]:.2f} on the Arabic mean and {-res["2024"]["d"][EN]:.2f} on English, and the rule is met on both seeds.
- Not claimed: that it is better at other sizes or longer runs (the advantage narrowed with training and its limit is unknown); that it is better on
  text unlike the training sources (the Moroccan Wikipedia is worse in both seeds); which of the three changes does the work, or whether one of them is the cause of the `wiki_ary` loss;
  that it matters for downstream use, which has not been tested at all. Two seeds are still few.
- The model has 1.15% fewer parameters (no position table); this works against the modern one, and was not compensated.

**Decisions.**

1. The claim above is recorded. `configs/experiments/modern-17m.yaml` is the reference architecture for later experiments, as D-028 said it would be if this held; `local-17m` stays the baseline of record for earlier numbers.
2. The GPT-2 parts stay available as the code default (D-027).
3. `wiki_ary` is added to what is reported next to every later architecture comparison, because it is the one source on which this change lost, and a
   later change that fixes it would be visible.

**Not decided, and not tested here.**

- Which of the three changes helps and which hurts `wiki_ary`: three runs with one switch each (or one removed). Rotary embeddings are the first suspect for the generalization difference; nothing here shows it.
- A longer run, to see where the narrowing advantage ends.
- A key/value cache and grouped-query attention (not built).
"""
out = Path(sys.argv[1])
out.write_text(text, encoding="utf-8")
print(f"wrote {out}: {len(text.splitlines())} lines")
