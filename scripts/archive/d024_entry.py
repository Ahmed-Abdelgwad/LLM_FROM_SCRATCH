# ruff: noqa: E501
"""Builds the D-024 entry of docs/DECISIONS.md (the English-share ablation) from the run files and
the full evaluations. Every table and every number in the text comes from a file; the prose around
them is written here, and the sentences that depend on a comparison are guarded by an assert, so the
script stops instead of printing a claim the numbers do not support.

It also writes the evaluation events of the runs it read to
reports/eval/final-d024-mix-ablation/evals.json, so the numbers stay checkable without runs/ (which
git ignores). Needs runs/ as it was when D-024 was written, and the reports of the two full
evaluations (reports/eval/final-local-17m and reports/eval/final-ablation-english-20-long).
Run from the repository root: uv run python scripts/archive/d024_entry.py OUT.md"""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # scripts/archive/ -> the repository root
EVIDENCE = ROOT / "reports/eval/final-d024-mix-ablation"
AR = ["fw2_arb", "fw2_ars", "fw2_ary", "fw2_arz", "wiki_ar"]
EN = "fineweb_edu_10bt"
SHORT = {EN: "edu", "fw2_arb": "arb", "fw2_ars": "ars", "fw2_ary": "ary", "fw2_arz": "arz",
         "wiki_ar": "wiki_ar", "fw2_acm": "acm", "fw2_apc": "apc", "wiki_ary": "wiki_ary"}  # fmt: skip
RULE_MEAN, RULE_ONE = 0.03, 0.04  # D-023, decision 3


def events(name: str) -> list[dict]:
    path = ROOT / f"runs/{name}/metrics.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


def last_eval(name: str, step: int) -> dict:
    return [e for e in events(name) if e["event"] == "eval" and e["step"] == step][-1]


def ar_mean(ev: dict, key: str = "bits_per_char") -> float:
    return statistics.mean(ev["sources"][s][key] for s in AR)


def verdict(delta_mean: float, deltas: list[float]) -> str:
    ok_mean = delta_mean <= -RULE_MEAN
    n_one = sum(d <= -RULE_ONE for d in deltas)
    return (
        f"mean {'meets' if ok_mean else 'does not meet'} the {RULE_MEAN} threshold; "
        f"{n_one} of {len(deltas)} sources reach {RULE_ONE}"
    )


# ------------------------------------------------------------------ the runs
RUNS = {
    "control-800": "diag-lr-6e-4",
    "control-800-seed2": "diag-lr-6e-4-seed2",
    "arabic-only-800": "ablation-arabic-only",
    "english-20-800": "ablation-english-20",
    "baseline-5000": "local-17m",
    "english-20-5000": "ablation-english-20-long",
}
ev = {}
starts = {}
for tag, name in RUNS.items():
    es = events(name)
    ends = [e for e in es if e["event"] == "end"]
    assert len(ends) == 1 and ends[0]["skipped_steps"] == 0, tag
    starts[tag] = [e for e in es if e["event"] == "start"][0]
    ev[tag] = {
        "run": name,
        "end": ends[0],
        "weights": starts[tag]["sampling"]["weights"],
        "evals": [e for e in es if e["event"] == "eval"],
    }
EVIDENCE.mkdir(parents=True, exist_ok=True)
(EVIDENCE / "evals.json").write_text(
    json.dumps(ev, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
)


def share(tag: str, source: str) -> float:
    w = ev[tag]["weights"]
    return w.get(source, 0.0) / sum(w.values())


def at(tag: str, step: int) -> dict:
    return [e for e in ev[tag]["evals"] if e["step"] == step][-1]


ar_total = sum(ev["arabic-only-800"]["weights"].values())  # Arabic train tokens in the files
assert abs(ar_total - 101_561_123) < 1

# ------------------------------------------------------------------ the runs, as a table
mix_rows = [
    "| run | steps | English share of train tokens | Arabic share | tokens seen | expected passes over the Arabic data | over the English data |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
english_total = ev["baseline-5000"]["weights"][EN]
for tag, label in [
    ("control-800", "control (`diag-lr-6e-4`)"),
    ("arabic-only-800", "`ablation-arabic-only`"),
    ("english-20-800", "`ablation-english-20`"),
    ("baseline-5000", "baseline (`local-17m`)"),
    ("english-20-5000", "`ablation-english-20-long`"),
]:
    seen = ev[tag]["end"]["tokens_seen"]
    se, sa = share(tag, EN), 1 - share(tag, EN)
    mix_rows.append(
        f"| {label} | {ev[tag]['end']['step']:,} | {se:.1%} | {sa:.1%} | {seen:,} | {seen * sa / ar_total:.2f} | {seen * se / english_total:.2f} |"
    )
passes_base = (
    ev["baseline-5000"]["end"]["tokens_seen"] * (1 - share("baseline-5000", EN)) / ar_total
)
passes_long = (
    ev["english-20-5000"]["end"]["tokens_seen"] * (1 - share("english-20-5000", EN)) / ar_total
)


# ------------------------------------------------------------------ 800-step arms
def compare_table(base_tag: str, arms: list[tuple[str, str]], step: int) -> tuple[list[str], dict]:
    cols = [SHORT[s] for s in AR]
    rows = [
        f"| run (step {step}) | "
        + " | ".join(cols)
        + " | Arabic mean | change vs control | English | change |",
        "|---|" + "---:|" * (len(cols) + 4),
    ]
    b = at(base_tag, step)
    info = {}
    entries = [(base_tag, "control")] + arms
    for tag, label in entries:
        e = at(tag, step)
        d_mean = ar_mean(e) - ar_mean(b)
        d_en = e["sources"][EN]["bits_per_char"] - b["sources"][EN]["bits_per_char"]
        cells = " | ".join(f"{e['sources'][s]['bits_per_char']:.3f}" for s in AR)
        dm = "" if tag == base_tag else f"{d_mean:+.3f}"
        de = "" if tag == base_tag else f"{d_en:+.3f}"
        rows.append(
            f"| {label} | {cells} | {ar_mean(e):.3f} | {dm} | {e['sources'][EN]['bits_per_char']:.3f} | {de} |"
        )
        info[tag] = {
            "d_mean": d_mean,
            "d_src": [
                e["sources"][s]["bits_per_char"] - b["sources"][s]["bits_per_char"] for s in AR
            ],
            "d_en": d_en,
        }
    return rows, info


t800, i800 = compare_table(
    "control-800", [("arabic-only-800", "Arabic only"), ("english-20-800", "English 20%")], 800
)
noise_mean = ar_mean(at("control-800-seed2", 800)) - ar_mean(at("control-800", 800))
noise_src = [
    at("control-800-seed2", 800)["sources"][s]["bits_per_char"]
    - at("control-800", 800)["sources"][s]["bits_per_char"]
    for s in AR
]
a800, e800 = i800["arabic-only-800"], i800["english-20-800"]
assert a800["d_mean"] <= -RULE_MEAN and e800["d_mean"] <= -RULE_MEAN

# ------------------------------------------------------------------ 5,000-step pair
cols = [SHORT[s] for s in AR]
t5000 = [
    "| run (step 5,000, validation) | " + " | ".join(cols) + " | Arabic mean | English |",
    "|---|" + "---:|" * (len(cols) + 2),
]
b5, l5 = at("baseline-5000", 5000), at("english-20-5000", 5000)
for e, label in [(b5, "baseline, English 39.3%"), (l5, "English 20.0%")]:
    t5000.append(
        f"| {label} | "
        + " | ".join(f"{e['sources'][s]['bits_per_char']:.3f}" for s in AR)
        + f" | {ar_mean(e):.4f} | {e['sources'][EN]['bits_per_char']:.4f} |"
    )
d5 = [l5["sources"][s]["bits_per_char"] - b5["sources"][s]["bits_per_char"] for s in AR]
d5_mean = ar_mean(l5) - ar_mean(b5)
d5_en = l5["sources"][EN]["bits_per_char"] - b5["sources"][EN]["bits_per_char"]
t5000.append(
    "| change | " + " | ".join(f"{d:+.3f}" for d in d5) + f" | {d5_mean:+.4f} | {d5_en:+.4f} |"
)
assert d5_mean <= -RULE_MEAN and all(d <= -RULE_ONE for d in d5) and d5_en > 0
assert abs(ar_mean(b5) - 2.0153) < 5e-4  # the value written into the config header

curve = [
    "| step | baseline Arabic mean | English 20% Arabic mean | change | baseline English | English 20% English | change |",
    "|---:|---:|---:|---:|---:|---:|---:|",
]
curve_d = {}
for st in range(500, 5001, 500):
    b, e = at("baseline-5000", st), at("english-20-5000", st)
    dm = ar_mean(e) - ar_mean(b)
    de = e["sources"][EN]["bits_per_char"] - b["sources"][EN]["bits_per_char"]
    curve_d[st] = (dm, de)
    curve.append(
        f"| {st:,} | {ar_mean(b):.3f} | {ar_mean(e):.3f} | {dm:+.3f} | {b['sources'][EN]['bits_per_char']:.3f} | {e['sources'][EN]['bits_per_char']:.3f} | {de:+.3f} |"
    )
first_pass = [st for st, (dm, _) in curve_d.items() if dm <= -RULE_MEAN]
assert all(dm <= -RULE_MEAN for dm, _ in curve_d.values())
peak_step = min(curve_d, key=lambda st: curve_d[st][0])
assert curve_d[peak_step][0] < d5_mean < 0  # the gain peaked and then settled somewhat lower

# the gap between held-out and the training probe, same tokens-per-source, so comparable across runs
gap = {}
for tag in ("baseline-5000", "english-20-5000"):
    e = at(tag, 5000)
    gap[tag] = {s: e["sources"][s]["loss"] - e["sources"][s]["train_loss"] for s in AR}
gap_rows = [
    "| run | " + " | ".join(cols) + " | mean |",
    "|---|" + "---:|" * (len(cols) + 1),
]
for tag, label in [("baseline-5000", "baseline"), ("english-20-5000", "English 20%")]:
    gap_rows.append(
        f"| {label} | "
        + " | ".join(f"{gap[tag][s]:+.3f}" for s in AR)
        + f" | {statistics.mean(gap[tag].values()):+.3f} |"
    )
gap_grew = [s for s in AR if gap["english-20-5000"][s] > gap["baseline-5000"][s]]
gap_inc = [gap["english-20-5000"][s] - gap["baseline-5000"][s] for s in AR]

# ------------------------------------------------------------------ full evaluation of both checkpoints
rep_b = json.loads((ROOT / "reports/eval/final-local-17m/report.json").read_text())
rep_l = json.loads((ROOT / "reports/eval/final-ablation-english-20-long/report.json").read_text())
rb = {(r["source"], r["split"]): r["bits_per_char"] for r in rep_b["rows"]}
rl = {(r["source"], r["split"]): r["bits_per_char"] for r in rep_l["rows"]}
assert rep_b["step"] == rep_l["step"] == 5000
full = [
    "| source | split | baseline bits/char | English 20% | change |",
    "|---|---|---:|---:|---:|",
]
keys = [(s, sp) for s in [EN, *AR] for sp in ("val", "test")] + [
    (s, "eval") for s in ("fw2_acm", "fw2_apc", "wiki_ary")
]
for s, sp in keys:
    full.append(
        f"| {SHORT[s]} | {sp} | {rb[(s, sp)]:.3f} | {rl[(s, sp)]:.3f} | {rl[(s, sp)] - rb[(s, sp)]:+.3f} |"
    )
fm = {
    sp: (statistics.mean(rb[(s, sp)] for s in AR), statistics.mean(rl[(s, sp)] for s in AR))
    for sp in ("val", "test")
}
fm["eval"] = (
    statistics.mean(rb[(s, "eval")] for s in ("fw2_acm", "fw2_apc", "wiki_ary")),
    statistics.mean(rl[(s, "eval")] for s in ("fw2_acm", "fw2_apc", "wiki_ary")),
)
for label, sp in [
    ("trained Arabic, val (mean)", "val"),
    ("trained Arabic, test (mean)", "test"),
    ("unseen dialects (mean)", "eval"),
]:
    full.append(
        f"| **{label}** | | **{fm[sp][0]:.3f}** | **{fm[sp][1]:.3f}** | **{fm[sp][1] - fm[sp][0]:+.3f}** |"
    )
test_d = [rl[(s, "test")] - rb[(s, "test")] for s in AR]
val_d = [rl[(s, "val")] - rb[(s, "val")] for s in AR]
n_val4 = sum(d <= -RULE_ONE for d in val_d)
n_test4 = sum(d <= -RULE_ONE for d in test_d)
below4 = [
    SHORT[s]
    for s, dv, dt in zip(AR, val_d, test_d, strict=True)
    if dv > -RULE_ONE or dt > -RULE_ONE
]
assert fm["val"][1] - fm["val"][0] <= -RULE_MEAN and fm["test"][1] - fm["test"][0] <= -RULE_MEAN
assert fm["eval"][1] - fm["eval"][0] > -RULE_MEAN
unseen_d = [rl[(s, "eval")] - rb[(s, "eval")] for s in ("fw2_acm", "fw2_apc", "wiki_ary")]
assert all(d < 0 for d in test_d) and all(d < 0 for d in unseen_d)

# ------------------------------------------------------------------ the entry
text = f"""## D-024 (2026-11-05): the English share, 39% to 20% of the tokens: Arabic gains {-d5_mean:.3f} bits/char at 5,000 steps, English loses {d5_en:.3f}

Code: none changed. Added: `configs/experiments/ablation-arabic-only.yaml`, `configs/experiments/ablation-english-20.yaml` (800 steps) and
`configs/experiments/ablation-english-20-long.yaml` (5,000 steps), each with the question and the rule of D-023 in its header, written
before the run. Evidence kept in git: `reports/eval/final-ablation-english-20-long/` (full evaluation of the 5,000-step
checkpoint) and `reports/eval/final-d024-mix-ablation/evals.json` (the evaluation events of the six runs below, copied from
their `metrics.jsonl`). Script: `scripts/archive/d024_entry.py` computed the tables and numbers of this entry from those runs and the
two full evaluations (prose by hand; sentences that depend on a comparison are guarded by an assert in the script). Not in git:
the runs themselves (`runs/`, ignored). The first attempt of the Arabic-only arm was killed when the session restarted, at step 370,
and rerun from step 0; its directory was kept aside and is not used.

**The question** (left open in D-023): does the English share, the largest source at 39.3% of the training tokens, cost the Arabic sources
anything? D-023's gradient evidence said the sources do not fight and share little; it could not say what removing English would do.
**The rule** (D-023, decision 3, fixed before the runs): a change counts only if the mean bits/char of the five Arabic sources on the same
split falls by at least {RULE_MEAN}, and a claim about one source needs at least {RULE_ONE}. English is reported and is not part of the rule.

Everything else is identical to `configs/local-17m.yaml`: the model, optimizer, batch (32,768 tokens), seed 1337 and the validation
windows. The Arabic sources keep their natural relative weights; only the English weight changes. Training batches are random windows drawn
with replacement, so "passes" below are expected coverage (tokens drawn from a source / its tokens in the files).

{chr(10).join(mix_rows)}

**800 steps** (the control is the learning-rate run `diag-lr-6e-4`, same settings, 50 warmup steps; bits/char, validation, step 800).
Seed noise measured on this control and its repeat (seed 2024): {noise_mean:+.3f} on the Arabic mean and {min(noise_src):+.3f} to {max(noise_src):+.3f} per source
(D-023 quoted 0.012, the mean over six sources with English; the Arabic mean used by the rule is {noise_mean:.3f}).

{chr(10).join(t800)}

Arabic only: mean {a800["d_mean"]:+.3f} ({verdict(a800["d_mean"], a800["d_src"])}); English {a800["d_en"]:+.3f}.
English 20%: mean {e800["d_mean"]:+.3f} ({verdict(e800["d_mean"], e800["d_src"])}); English {e800["d_en"]:+.3f}.
Both pass on the mean; the 20% arm passes only there, since no single source reaches {RULE_ONE} (its gain per source is {-max(e800["d_src"]):.3f} to {-min(e800["d_src"]):.3f}, against seed noise of up to {max(noise_src):.3f} per source).
The Arabic-only arm gains about {a800["d_mean"] / e800["d_mean"]:.1f} times as much as the 20% arm, at the cost of English losing {a800["d_en"]:.2f} bits/char (English was not in its mix at all).
Only 800 steps, one seed per arm, and the English share and the Arabic exposure change together.

**5,000 steps** (English 20% against the baseline of D-023, `local-17m`; the in-training validation evaluation, 20 batches per source):

{chr(10).join(t5000)}

By the rule: {verdict(d5_mean, d5)}. The margin per source is thin: the smallest gain is {min(-d for d in d5):.3f}, against a threshold of {RULE_ONE} and a per-source seed noise of up to {max(noise_src):.3f} measured at 800 steps; the mean gain is
{-d5_mean:.4f}, {-d5_mean / RULE_MEAN:.1f} times the threshold and {-d5_mean / abs(noise_mean):.1f} times the seed noise measured at 800 steps. English is {d5_en:+.4f}, not part of the rule.
Caveat on the baseline: it was stopped once at step 2,990 and resumed; the 20% run was not interrupted.

The gap is there at every checkpoint, not only the last:

{chr(10).join(curve)}

It reaches {RULE_MEAN} already at step {first_pass[0]:,} and stays beyond it at every checkpoint: {curve_d[500][0]:+.3f} at step 500, {curve_d[peak_step][0]:+.3f} at its largest (step {peak_step:,}) and {curve_d[5000][0]:+.3f} at 5,000, so the 800-step result carried over to the full length and settled a little lower as the loss flattened.
The English loss is {curve_d[500][1]:+.3f} at step 500, {curve_d[2500][1]:+.3f} at 2,500 and {curve_d[5000][1]:+.3f} at 5,000.

**Full evaluation** of the two final checkpoints (every held-out token; `reports/eval/final-local-17m` and `reports/eval/final-ablation-english-20-long`).
Its validation numbers differ slightly from the table above because it reads every validation token, not 20 batches; the comparison is between like and like.
The test split and the three dialects never seen in training are not part of the rule; they are the check that the validation result is not an accident of the validation windows.

{chr(10).join(full)}

The change on the Arabic mean is {fm["val"][1] - fm["val"][0]:+.3f} on validation and {fm["test"][1] - fm["test"][0]:+.3f} on test, both beyond {RULE_MEAN}. Per source, {n_val4} of {len(AR)} reach {RULE_ONE} on validation and {n_test4} of {len(AR)} on test; {", ".join(below4)} falls just short on at least one of them,
so a claim about every single source (the {RULE_ONE} rule) holds on the in-training validation table above and is thinner on the full evaluation. On the 3 dialects never seen in training the sign is the same for all three, but the mean change is {fm["eval"][1] - fm["eval"][0]:+.3f}, under {RULE_MEAN}, and wiki_ary moves by only {unseen_d[2]:+.3f}:
read it as no worse on unseen dialects, not as better.

**What the 20% run does not rule out.** At 20%, the Arabic data is drawn {passes_long:.2f} times in expectation over 5,000 steps, against {passes_base:.2f} for the baseline. Part of the
gain may be exposure to Arabic rather than relief from English (the same run cannot tell the two apart), and if repeating data hurt, it would show as a growing gap between the loss on held-out data and on the training data. The held-out loss minus the loss on a probe of
the training data (same tokens, so comparable across runs), in nats:

{chr(10).join(gap_rows)}

The gap is larger in the 20% run for {len(gap_grew)} of {len(AR)} Arabic sources ({", ".join(SHORT[s] for s in gap_grew) if gap_grew else "none"}) by {min(gap_inc):.3f} to {max(gap_inc):.3f} nats, so
the 20% run fits its training data a little more tightly relative to held-out data; held-out loss still fell, which is what the rule measures, and at 1.29 passes this is not yet overfitting by any large amount. Besides
this, every run here has one seed (noise {noise_mean:.3f} on the Arabic mean, measured once, at 800 steps), the baseline was interrupted once, and the English tokenization
and the `arz` train/validation gap of D-023 are still untested.

**Decisions.**

1. The English-20% mix meets the rule of D-023 at the full length (mean {-d5_mean:.3f} against {RULE_MEAN}; all five sources beyond {RULE_ONE} by a thin margin), and the mean gain holds on the test split. This is recorded
   as the result; the question "does English cost the Arabic sources anything" has the answer: at this model size and length, cutting English from 39% to 20% of the tokens
   improved the Arabic sources by {-d5_mean:.3f} bits/char, and English paid {d5_en:.3f}.
2. `configs/local-17m.yaml` is not changed, and no checkpoint is replaced. Whether to adopt the 20% mix (or another) as the default is a project decision, taken on
   how much English the model must keep, not an experimental one: the experiment gives the exchange rate, English losing about {d5_en / -d5_mean:.1f} times what the Arabic mean gains.
3. The rule of D-023 stands unchanged.

**Not decided, and not tested here.**

- Whether the mechanism is less English or more Arabic exposure. A run that gives the Arabic data the same {passes_long:.2f} passes while keeping English at 39% (about {ev["english-20-5000"]["end"]["tokens_seen"] * (1 - share("english-20-5000", EN)) / (1 - share("baseline-5000", EN)) / 1e6:.0f}M tokens, {ev["english-20-5000"]["end"]["tokens_seen"] * (1 - share("english-20-5000", EN)) / (1 - share("baseline-5000", EN)) / 32768:,.0f} steps)
  would separate the two; it is also the test of whether more steps alone help.
- The Arabic-only arm at 5,000 steps (not run; 800 steps only), other English shares (10%, 30%), and a second seed of the 5,000-step pair. Each result above
  is a single seed.
- How much English the project wants the model to keep (D-023's open point, now with a price), and whether English quality matters more than its bits/char
  number says; the English samples were already the weakest in D-023.
"""

out = Path(sys.argv[1])
out.write_text(text, encoding="utf-8")
print(f"wrote {out}: {len(text.splitlines())} lines")
