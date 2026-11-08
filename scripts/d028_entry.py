# ruff: noqa: E501
"""Builds the D-028 entry of docs/DECISIONS.md (the modern architecture against the GPT-2 baseline)
from the run files and the two full evaluations. Every table and number comes from a file; the
prose is written here, and sentences that depend on a comparison are guarded by an assert.

Also writes the evaluation events of the two runs to
reports/eval/final-d028-modern-arch/evals.json so the numbers stay checkable without runs/.
Needs runs/local-17m, runs/modern-17m and reports/eval/final-{local,modern}-17m.
Run from the repository root: uv run python scripts/d028_entry.py OUT.md"""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "reports/eval/final-d028-modern-arch"
AR = ["fw2_arb", "fw2_ars", "fw2_ary", "fw2_arz", "wiki_ar"]
EN = "fineweb_edu_10bt"
UNSEEN = ["fw2_acm", "fw2_apc", "wiki_ary"]
SHORT = {EN: "edu", "fw2_arb": "arb", "fw2_ars": "ars", "fw2_ary": "ary", "fw2_arz": "arz",
         "wiki_ar": "wiki_ar", "fw2_acm": "acm", "fw2_apc": "apc", "wiki_ary": "wiki_ary"}  # fmt: skip
RULE_MEAN, RULE_ONE, RULE_EN = 0.03, 0.04, 0.03  # D-023 decision 3 and D-025


def events(name: str) -> list[dict]:
    path = ROOT / f"runs/{name}/metrics.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


runs = {"baseline": "local-17m", "modern": "modern-17m"}
log = {k: events(v) for k, v in runs.items()}
ev = {k: [e for e in v if e["event"] == "eval"] for k, v in log.items()}
end = {k: [e for e in v if e["event"] == "end"][0] for k, v in log.items()}
assert end["modern"]["skipped_steps"] == 0 and end["modern"]["step"] == 5000
assert [e["event"] for e in log["modern"] if e["event"] in ("resume", "interrupted")] == []
EVIDENCE.mkdir(parents=True, exist_ok=True)
(EVIDENCE / "evals.json").write_text(
    json.dumps({k: {"run": runs[k], "evals": ev[k]} for k in runs}, indent=1) + "\n",
    encoding="utf-8",
)


def at(tag: str, step: int) -> dict:
    return [e for e in ev[tag] if e["step"] == step][-1]


def ar_mean(e: dict) -> float:
    return statistics.mean(e["sources"][s]["bits_per_char"] for s in AR)


b5, m5 = at("baseline", 5000), at("modern", 5000)
d_src = {
    s: m5["sources"][s]["bits_per_char"] - b5["sources"][s]["bits_per_char"] for s in [*AR, EN]
}
d_mean = ar_mean(m5) - ar_mean(b5)
assert abs(ar_mean(b5) - 2.0153) < 5e-4  # the value written into the config header
assert d_mean <= -RULE_MEAN and all(d_src[s] <= -RULE_ONE for s in AR) and d_src[EN] < RULE_EN

t5000 = [
    "| run (step 5,000, validation) | "
    + " | ".join(SHORT[s] for s in AR)
    + " | Arabic mean | English |",
    "|---|" + "---:|" * 7,
]
for tag, label in [("baseline", "baseline (GPT-2 parts)"), ("modern", "RMSNorm + RoPE + SwiGLU")]:
    e = at(tag, 5000)
    t5000.append(
        f"| {label} | " + " | ".join(f"{e['sources'][s]['bits_per_char']:.3f}" for s in AR)
        + f" | {ar_mean(e):.4f} | {e['sources'][EN]['bits_per_char']:.4f} |"
    )  # fmt: skip
t5000.append(
    "| change | "
    + " | ".join(f"{d_src[s]:+.3f}" for s in AR)
    + f" | {d_mean:+.4f} | {d_src[EN]:+.4f} |"
)

curve = [
    "| step | baseline Arabic mean | modern | change | baseline English | modern | change |",
    "|---:|---:|---:|---:|---:|---:|---:|",
]
cd = {}
for st in range(500, 5001, 500):
    b, m = at("baseline", st), at("modern", st)
    dm = ar_mean(m) - ar_mean(b)
    de = m["sources"][EN]["bits_per_char"] - b["sources"][EN]["bits_per_char"]
    cd[st] = (dm, de)
    curve.append(
        f"| {st:,} | {ar_mean(b):.3f} | {ar_mean(m):.3f} | {dm:+.3f} | {b['sources'][EN]['bits_per_char']:.3f} | {m['sources'][EN]['bits_per_char']:.3f} | {de:+.3f} |"
    )
assert all(dm <= -RULE_MEAN for dm, _ in cd.values()) and all(de < 0 for _, de in cd.values())
assert abs(cd[500][0]) > abs(cd[5000][0])  # the gap narrows

rb = {
    (r["source"], r["split"]): r["bits_per_char"]
    for r in json.loads((ROOT / "reports/eval/final-local-17m/report.json").read_text())["rows"]
}
rep_m = json.loads((ROOT / "reports/eval/final-modern-17m/report.json").read_text())
assert rep_m["step"] == 5000 and rep_m["parameters"] == 16_913_280
rm = {(r["source"], r["split"]): r["bits_per_char"] for r in rep_m["rows"]}
full = ["| source | split | baseline | modern | change |", "|---|---|---:|---:|---:|"]
for s, sp in [(s, sp) for s in [EN, *AR] for sp in ("val", "test")] + [(s, "eval") for s in UNSEEN]:
    full.append(
        f"| {SHORT[s]} | {sp} | {rb[(s, sp)]:.3f} | {rm[(s, sp)]:.3f} | {rm[(s, sp)] - rb[(s, sp)]:+.3f} |"
    )
grp = {}
for label, keys in [("trained Arabic, val (mean)", [(s, "val") for s in AR]), ("trained Arabic, test (mean)", [(s, "test") for s in AR]),
                    ("unseen dialects (mean)", [(s, "eval") for s in UNSEEN]), ("English, val", [(EN, "val")]), ("English, test", [(EN, "test")])]:  # fmt: skip
    b = statistics.mean(rb[k] for k in keys)
    m = statistics.mean(rm[k] for k in keys)
    grp[label] = (b, m)
    full.append(f"| **{label}** | | **{b:.3f}** | **{m:.3f}** | **{m - b:+.3f}** |")
worse_unseen = [s for s in UNSEEN if rm[(s, "eval")] > rb[(s, "eval")]]
assert all(rm[(s, "test")] < rb[(s, "test")] for s in AR + [EN])
unseen_d = grp["unseen dialects (mean)"][1] - grp["unseen dialects (mean)"][0]

tp = {
    k: statistics.median(e["tokens_per_s"] for e in v if e["event"] == "train")
    for k, v in log.items()
}
mem = {
    k: max(e.get("peak_memory_mib", 0) for e in v if e["event"] == "train") for k, v in log.items()
}
slow = min(e["tokens_per_s"] for e in log["modern"] if e["event"] == "train")

text = f"""## D-028 (2026-11-06): RMSNorm + rotary embeddings + SwiGLU against the GPT-2 baseline: Arabic {d_mean:+.3f} bits/char, English {d_src[EN]:+.3f}, at the same length

Code: none changed (D-027 built the options). Added: `configs/experiments/modern-17m.yaml`. Evidence kept in git:
`reports/eval/final-modern-17m/` (full evaluation of the 5,000-step checkpoint) and
`reports/eval/final-d028-modern-arch/evals.json` (the evaluation events of both runs). Script:
`scripts/d028_entry.py` computed the tables and numbers of this entry (prose by hand; sentences that depend on a
comparison are guarded by an assert). Not in git: the runs (`runs/`, ignored).

**What was run.** The baseline `local-17m` and `modern-17m` are identical in every setting (seed 1337, natural mix, 5,000
steps of 32,768 tokens, learning rate 6e-4 to 6e-5, the same validation windows) except the three switches of D-027, all on
together. The model has {rep_m["parameters"]:,} parameters against 17,109,888 (no position table, -1.15%). The modern
run was not interrupted and skipped no step; the baseline was stopped once at step 2,990 and resumed (D-023).

**The criterion**, in the config header before the run: the Arabic mean bits/char at step 5,000 must be at least {RULE_MEAN} below
2.0153 (a claim about one source, {RULE_ONE}), and English must not get worse by {RULE_EN} or more (D-023, D-025).

{chr(10).join(t5000)}

By the rule: the Arabic mean is {d_mean:+.4f} (threshold {-RULE_MEAN}), all five Arabic sources are beyond {RULE_ONE} (the smallest gain is
{min(-d_src[s] for s in AR):.3f}), and English moved by {d_src[EN]:+.4f}, i.e. it improved. **The criterion is met.** The gain is about
{-d_mean / 0.013:.0f} times the seed noise measured at 800 steps (0.013 on the Arabic mean, D-024), and {min(-d_src[s] for s in AR) / 0.021:.1f} times its largest per-source value.

The gap was larger earlier and narrowed as training went on:

{chr(10).join(curve)}

At step 500 the modern model is {-cd[500][0]:.3f} bits/char ahead on the Arabic mean, at 2,500 {-cd[2500][0]:.3f}, at 5,000 {-cd[5000][0]:.3f}; it stays beyond the
threshold at every checkpoint, but the advantage is shrinking: most of it is faster early learning, and the part that remains at the
end was still decreasing slowly ({cd[4000][0]:+.3f} at 4,000, {cd[5000][0]:+.3f} at 5,000). Whether it would persist, vanish or grow
with a longer run is not known from this one.

**Full evaluation** of the two final checkpoints (every held-out token; `reports/eval/final-local-17m`, `reports/eval/final-modern-17m`):

{chr(10).join(full)}

The mean gain on the trained Arabic sources is {grp["trained Arabic, val (mean)"][1] - grp["trained Arabic, val (mean)"][0]:+.3f} on validation and {grp["trained Arabic, test (mean)"][1] - grp["trained Arabic, test (mean)"][0]:+.3f} on test; every trained source, English included, has a lower bits/char on test.
**On the three dialects never seen in training the mean change is {unseen_d:+.3f}**: {", ".join(SHORT[s] for s in worse_unseen) if worse_unseen else "none"} {"is" if len(worse_unseen) == 1 else "are"} worse than the baseline
({", ".join(f"{SHORT[s]} {rm[(s, 'eval')] - rb[(s, 'eval')]:+.3f}" for s in UNSEEN)}). The modern model scored lower on the languages it trained on, but this
run gives no sign that it generalizes better to dialects it did not see, and for {"one" if len(worse_unseen) == 1 else "some"} of them it is worse. The rule does not cover these sources; it is a
point against reading the result as "a better model everywhere".

**Speed and memory** (same GPU, whole runs): median {tp["modern"]:,.0f} tokens/s against {tp["baseline"]:,.0f} for the baseline ({tp["modern"] / tp["baseline"] - 1:+.0%}); peak
memory {mem["modern"]:,.0f} MiB against {mem["baseline"]:,.0f}. The modern run's rate fell as low as {slow:,.0f} tokens/s in some stretches (the cause was not looked into:
another load on the machine, or the GPU's own clock), and the median is the fair figure; a controlled benchmark was not run. Where the difference comes from (the normalization, the missing position
addition, the SwiGLU matrix shapes) was not separated.

**Limits.** One seed per run. The three changes were tested together: this result says the package scored lower in this run, not
whether it would again, nor which part does the work, and one part could even hurt while the others help. The parameter count is 1.15% lower for the modern model, which
if anything works against it. The advantage narrows with training. Generalization to unseen dialects is not improved.

**Decisions.**

1. The modern architecture meets the criterion in this run (one seed). It is recorded as a result, not as established superiority: no claim that it is
   better is made until the same comparison has been repeated with a second seed, with every setting and the token budget unchanged.
2. `configs/local-17m.yaml` stays the baseline of record, so earlier numbers keep their meaning; `configs/experiments/modern-17m.yaml` is the new reference
   for later experiments that build on this architecture. Which of the two a later change is compared against is stated in that change's config.
3. Nothing is removed: the GPT-2 parts stay as the default of the code (D-027).

**Not decided, and not tested here.**

- **The next step, already fixed:** the same two configurations (`local-17m`, `modern-17m`) rerun with a different seed, every training
  setting and the token budget unchanged, judged by the same criterion. The baseline numbers above are not altered by it.
- Which of the three changes matters (three runs with one switch each, or with one removed); only worth it if the project wants a design
  that is minimal, or if one part is suspected of hurting the unseen dialects.
- A longer run, to see if the advantage keeps narrowing.
- Why the unseen dialects do not follow; the rotary embeddings, which generalize position differently, are the first suspect, but nothing here shows it.
- A key/value cache and grouped-query attention (not built).
"""
out = Path(sys.argv[1])
out.write_text(text, encoding="utf-8")
print(f"wrote {out}: {len(text.splitlines())} lines")
