# ruff: noqa: E501
"""Builds the D-032 entry of docs/DECISIONS.md: stage 1 of the architecture ablation, judged by the rules
written in D-031 before the runs. Tables and numbers come from the run files and the full evaluations; the
prose is written here and each comparison sentence is guarded by an assert, so the script stops rather than
print a claim the numbers do not support.

Also writes the evaluation events of the runs to reports/eval/final-d032-ablation-stage1/evals.json.
Needs runs/{local-17m,modern-17m,ablation-{rmsnorm,rope,swiglu}-only} and the full evaluations
reports/eval/final-{local-17m,modern-17m,ablation-{rmsnorm,rope,swiglu}-only}.
Run from the repository root: uv run python scripts/d032_entry.py OUT.md"""

import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "reports/eval/final-d032-ablation-stage1"
AR = ["fw2_arb", "fw2_ars", "fw2_ary", "fw2_arz", "wiki_ar"]
EN = "fineweb_edu_10bt"
UNSEEN = ["fw2_acm", "fw2_apc", "wiki_ary"]
SHORT = {EN: "edu", "fw2_arb": "arb", "fw2_ars": "ars", "fw2_ary": "ary", "fw2_arz": "arz",
         "wiki_ar": "wiki_ar", "fw2_acm": "acm", "fw2_apc": "apc", "wiki_ary": "wiki_ary"}  # fmt: skip
RUNS = {"baseline": "local-17m", "rmsnorm": "ablation-rmsnorm-only", "rope": "ablation-rope-only",
        "swiglu": "ablation-swiglu-only", "modern": "modern-17m"}  # fmt: skip
REPORTS = {"baseline": "final-local-17m", "rmsnorm": "final-ablation-rmsnorm-only", "rope": "final-ablation-rope-only",
           "swiglu": "final-ablation-swiglu-only", "modern": "final-modern-17m"}  # fmt: skip
SWITCH = {"rmsnorm": ("norm", "rmsnorm"), "rope": ("position", "rope"), "swiglu": ("ffn", "swiglu")}
LABEL = {"rmsnorm": "RMSNorm", "rope": "rotary embeddings", "swiglu": "SwiGLU"}
# D-031, fixed before the runs
NEUTRAL, UNRESOLVED_LO, UNRESOLVED_HI, WIKI, SYSTEM = 0.01, 0.005, 0.015, 0.05, 0.10
TOTAL_EFFECT = -0.0660  # D-029: mean of the Arabic-mean effects at seeds 1337 and 2024


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
    assert v[0]["config"]["train"]["seed"] == 1337, k
for k in ("rmsnorm", "rope", "swiglu"):  # every setting but one switch
    defaults = {"norm": "layernorm", "position": "learned", "ffn": "gelu", "rope_theta": 10000.0}
    a, b = (json.loads(json.dumps(log[x][0]["config"])) for x in ("baseline", k))
    for c in (a, b):
        c["model"] = defaults | c["model"]
        c["name"] = ""
    key, value = SWITCH[k]
    assert b["model"][key] == value
    b["model"][key] = defaults[key]
    assert a == b, k
    assert not [e for e in log[k] if e["event"] in ("resume", "interrupted")], k
EVIDENCE.mkdir(parents=True, exist_ok=True)
(EVIDENCE / "evals.json").write_text(
    json.dumps({k: {"run": RUNS[k], "evals": ev[k]} for k in RUNS}, indent=1) + "\n",
    encoding="utf-8",
)


def at(k: str) -> dict:
    return [e for e in ev[k] if e["step"] == 5000][-1]


def mean_ar(e: dict) -> float:
    return statistics.mean(e["sources"][s]["bits_per_char"] for s in AR)


def classify(effect: float) -> str:
    size = abs(effect)
    if size < UNRESOLVED_LO:
        return "neutral"
    if size <= UNRESOLVED_HI:
        return "unresolved"
    return "helps" if effect < 0 else "hurts"


def run_stats(k: str) -> dict[str, float]:
    train = [e for e in log[k] if e["event"] == "train"]
    return {
        "tps": statistics.median(e["tokens_per_s"] for e in train),
        "mem": max(e.get("peak_memory_mib", 0) for e in train),
        "params": [e for e in log[k] if e["event"] == "start"][0]["parameters"],
    }


B = at("baseline")
stats = {k: run_stats(k) for k in RUNS}
eff: dict[str, dict[str, Any]] = {}
for k in ("rmsnorm", "rope", "swiglu", "modern"):
    E = at(k)
    eff[k] = {
        "ar": mean_ar(E) - mean_ar(B),
        "en": E["sources"][EN]["bits_per_char"] - B["sources"][EN]["bits_per_char"],
        "src": {s: E["sources"][s]["bits_per_char"] - B["sources"][s]["bits_per_char"] for s in AR},
        "speed": stats[k]["tps"] / stats["baseline"]["tps"] - 1,
        "mem": stats[k]["mem"] / stats["baseline"]["mem"] - 1,
    }
    eff[k]["class_ar"], eff[k]["class_en"] = classify(eff[k]["ar"]), classify(eff[k]["en"])
    eff[k]["faster"] = eff[k]["speed"] >= SYSTEM
    eff[k]["lighter"] = eff[k]["mem"] <= -SYSTEM

t1 = [
    "| run | Arabic mean | effect | class | English | effect | class | median tokens/s | vs baseline | peak memory MiB | vs baseline | parameters |",
    "|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---:|",
    f"| baseline (`local-17m`) | {mean_ar(B):.4f} | | | {B['sources'][EN]['bits_per_char']:.4f} | | | {stats['baseline']['tps']:,.0f} | | {stats['baseline']['mem']:,.0f} | | {stats['baseline']['params']:,} |",
]
for k in ("rmsnorm", "rope", "swiglu", "modern"):
    E = at(k)
    name = LABEL.get(k, "all three (`modern-17m`)")
    t1.append(
        f"| {name} | {mean_ar(E):.4f} | {eff[k]['ar']:+.4f} | {eff[k]['class_ar']} | {E['sources'][EN]['bits_per_char']:.4f} | {eff[k]['en']:+.4f} | {eff[k]['class_en']} "
        f"| {stats[k]['tps']:,.0f} | {eff[k]['speed']:+.0%} | {stats[k]['mem']:,.0f} | {eff[k]['mem']:+.0%} | {stats[k]['params']:,} |"
    )
t_src = [
    "| effect on bits/char | " + " | ".join(SHORT[s] for s in AR) + " |",
    "|---|" + "---:|" * 5,
]
for k in ("rmsnorm", "rope", "swiglu", "modern"):
    t_src.append(
        f"| {LABEL.get(k, 'all three')} | "
        + " | ".join(f"{eff[k]['src'][s]:+.3f}" for s in AR)
        + " |"
    )

# ---- additivity (D-031)
singles = sum(eff[k]["ar"] for k in ("rmsnorm", "rope", "swiglu"))
gap = singles - TOTAL_EFFECT
additive = abs(gap) < NEUTRAL
singles_en = sum(eff[k]["en"] for k in ("rmsnorm", "rope", "swiglu"))
gap_1337 = singles - eff["modern"]["ar"]

# ---- full evaluations
rep = {
    k: {
        (r["source"], r["split"]): r["bits_per_char"]
        for r in json.loads((ROOT / f"reports/eval/{REPORTS[k]}/report.json").read_text())["rows"]
    }
    for k in REPORTS
}
for k in REPORTS:
    assert len(rep[k]) >= 15, k
full = [
    "| change vs baseline | "
    + " | ".join(LABEL.get(k, "all three") for k in ("rmsnorm", "rope", "swiglu", "modern"))
    + " |",
    "|---|" + "---:|" * 4,
]
for s, sp in [(s, "test") for s in [*AR, EN]] + [(s, "eval") for s in UNSEEN]:
    full.append(
        f"| {SHORT[s]} {sp} | "
        + " | ".join(
            f"{rep[k][(s, sp)] - rep['baseline'][(s, sp)]:+.3f}"
            for k in ("rmsnorm", "rope", "swiglu", "modern")
        )
        + " |"
    )
full.append(
    "| unseen dialects (mean) | "
    + " | ".join(
        f"{statistics.mean(rep[k][(s, 'eval')] - rep['baseline'][(s, 'eval')] for s in UNSEEN):+.3f}"
        for k in ("rmsnorm", "rope", "swiglu", "modern")
    )
    + " |"
)
wiki = {
    k: rep[k][("wiki_ary", "eval")] - rep["baseline"][("wiki_ary", "eval")]
    for k in ("rmsnorm", "rope", "swiglu", "modern")
}
wiki_hurts = {k: wiki[k] >= WIKI for k in wiki}

# ---- keep / drop (D-031, rules 1 to 5)
decision: dict[str, str] = {}
for k in ("rmsnorm", "rope", "swiglu"):
    c = eff[k]["class_ar"]
    efficient = eff[k]["faster"] or eff[k]["lighter"]
    if wiki_hurts[k] and c == "helps":
        decision[k] = "trade-off (rule 4): helps Arabic, hurts wiki_ary; left to the project"
    elif wiki_hurts[k] and c == "neutral":
        decision[k] = "candidate to drop (rule 5)"
    elif c == "helps":
        decision[k] = "keep (rule 1)"
    elif c == "neutral":
        decision[k] = (
            "keep (rule 2: neutral but faster or lighter)"
            if efficient
            else "candidate to drop (rule 2: neutral, no speed or memory gain)"
        )
    elif c == "hurts":
        decision[k] = "candidate to drop (rule 3)"
    elif wiki_hurts[
        k
    ]:  # unresolved on Arabic and worse on wiki_ary: the rules answer differently by how it resolves
        decision[k] = (
            f"open: unresolved on Arabic and `wiki_ary` worse by {WIKI} or more; if the Arabic effect resolves to helps, rule 4 (a trade-off, left to the project), if to neutral, rule 5 (candidate to drop); the rules do not decide it until a second seed resolves the Arabic effect"
        )
    elif efficient:  # unresolved: kept whichever way it resolved (helps: rule 1; neutral: rule 2)
        decision[k] = (
            "keep: unresolved on Arabic, but kept under both ways it could resolve (helps: rule 1; neutral: rule 2, it is faster or lighter)"
        )
    else:
        decision[k] = "open: unresolved on Arabic, and the outcomes differ"
assert eff["rope"]["class_ar"] == "helps" and eff["rope"]["class_en"] == "helps"
assert (
    eff["swiglu"]["class_ar"] == "unresolved"
    and eff["swiglu"]["faster"]
    and eff["swiglu"]["lighter"]
)
assert eff["rmsnorm"]["class_ar"] == "neutral" and not (
    eff["rmsnorm"]["faster"] or eff["rmsnorm"]["lighter"]
)
assert additive
assert wiki_hurts == {"rmsnorm": False, "rope": False, "swiglu": True, "modern": True}, wiki_hurts
wiki_sum = wiki["rmsnorm"] + wiki["rope"] + wiki["swiglu"]
assert abs(wiki_sum - wiki["modern"]) < 0.02 and wiki["swiglu"] > 2 * wiki["rope"] > 0
assert decision["rope"].startswith("keep") and decision["rmsnorm"].startswith("candidate")
assert decision["swiglu"].startswith("open")

text = f"""## D-032 (2026-11-07): ablation stage 1: rotary embeddings carry the gain; SwiGLU brings the speed and most of the `wiki_ary` loss; RMSNorm shows no measurable effect

Code: none changed. Evidence kept in git: the full evaluations `reports/eval/final-ablation-{{rmsnorm,rope,swiglu}}-only/` and
`reports/eval/final-d032-ablation-stage1/evals.json` (the evaluation events of the five runs compared). Script: `scripts/d032_entry.py` computed
the tables, the classifications and the decisions of this entry by the rules of D-031 (prose by hand; comparison sentences guarded by asserts).
Not in git: the runs (`runs/`, ignored). The plan, the thresholds and the rules were committed before the first of these runs (D-031).

**What was run.** The three configs of stage 1 (`configs/ablation-*-only.yaml`), each the baseline `local-17m` with one switch on; the script checks that
the stored configurations differ from the baseline's in that switch only. All runs: seed 1337, 5,000 steps, 163,840,000 tokens, none interrupted, 0 skipped steps.
Effects are against the baseline, bits per character at step 5,000 (in-training validation), the thresholds of D-031: Arabic mean and English, helps at -{NEUTRAL} or lower, hurts at +{NEUTRAL} or higher,
sizes from {UNRESOLVED_LO} to {UNRESOLVED_HI} unresolved, below {UNRESOLVED_LO} neutral; speed and memory count at {SYSTEM:.0%}.

{chr(10).join(t1)}

Per source (the same effects):

{chr(10).join(t_src)}

**Results by D-031.**

- **Rotary embeddings: {eff["rope"]["class_ar"]}.** Arabic mean {eff["rope"]["ar"]:+.4f}, English {eff["rope"]["en"]:+.4f}, every Arabic source between {max(eff["rope"]["src"].values()):+.3f} and {min(eff["rope"]["src"].values()):+.3f}.
  Alone it gives {eff["rope"]["ar"] / eff["modern"]["ar"]:.0%} of the Arabic-mean effect of the whole package at this seed and {eff["rope"]["en"] / eff["modern"]["en"]:.0%} of the English one. It does not make training faster ({eff["rope"]["speed"]:+.0%}, inside the {SYSTEM:.0%} band)
  and uses about the same memory ({eff["rope"]["mem"]:+.0%}); it has {stats["baseline"]["params"] - stats["rope"]["params"]:,} fewer parameters.
- **SwiGLU: {eff["swiglu"]["class_ar"]} on Arabic** ({eff["swiglu"]["ar"]:+.4f}, between {UNRESOLVED_LO} and {UNRESOLVED_HI}), {eff["swiglu"]["class_en"]} on English ({eff["swiglu"]["en"]:+.4f}). All five Arabic sources move the same way (between {max(eff["swiglu"]["src"].values()):+.3f} and
  {min(eff["swiglu"]["src"].values()):+.3f}). It is **{eff["swiglu"]["speed"]:+.0%} faster and uses {eff["swiglu"]["mem"]:+.0%} peak memory**, with exactly the parameters of the baseline. So most of the speed of the modern package is SwiGLU's.
- **RMSNorm: {eff["rmsnorm"]["class_ar"]} on Arabic** ({eff["rmsnorm"]["ar"]:+.4f}, {UNRESOLVED_LO - abs(eff["rmsnorm"]["ar"]):.4f} below the unresolved band), {eff["rmsnorm"]["class_en"]} on English ({eff["rmsnorm"]["en"]:+.4f}, which is inside it).
  Speed {eff["rmsnorm"]["speed"]:+.0%} and memory {eff["rmsnorm"]["mem"]:+.0%}: no difference by the {SYSTEM:.0%} rule. All six sources move by -0.004 to -0.006, in the same direction, and the size is within about {abs(eff["rmsnorm"]["ar"]) / 0.0033:.1f} times the seed noise measured at this length (0.0033, D-029).
  It is classified neutral because the rule says so; it sits next to the boundary, so one more seed could move it.

**Additivity (decides stage 2).** The three single effects add to {singles:+.4f} on the Arabic mean; the package's total effect is {TOTAL_EFFECT:+.4f} (D-029), a gap of {gap:+.4f}, inside {NEUTRAL}: **the effects are taken as additive and stage 2 is not run.**
Against the package at seed 1337 alone ({eff["modern"]["ar"]:+.4f}) the gap is {gap_1337:+.4f}, also inside. The sum is a little larger than the package ({singles:+.4f} against {eff["modern"]["ar"]:+.4f}), i.e. the three together gain slightly less than their separate gains added up
(the parts overlap a little). On English the singles add to {singles_en:+.4f} against {eff["modern"]["en"]:+.4f}.

**The unseen dialects** (full evaluation, change in bits/char against the baseline, negative is lower; `wiki_ary` counts only from {WIKI}):

{chr(10).join(full)}

`wiki_ary`: RMSNorm {wiki["rmsnorm"]:+.3f}, rotary embeddings {wiki["rope"]:+.3f}, SwiGLU {wiki["swiglu"]:+.3f}, all three {wiki["modern"]:+.3f}.
**SwiGLU alone carries most of the `wiki_ary` loss, not the rotary embeddings.** SwiGLU's {wiki["swiglu"]:+.3f} is above the {WIKI} line, rotary embeddings' {wiki["rope"]:+.3f} is below it (no difference by the rule),
RMSNorm's {wiki["rmsnorm"]:+.3f} is nothing. The three singles add to {wiki_sum:+.3f} against {wiki["modern"]:+.3f} for the package, so the loss is additive too. This is **not** what D-029 guessed: it named the rotary embeddings as the first suspect,
"nothing here shows it", and the ablation shows it was the wrong suspect. SwiGLU does nothing for the other unseen dialects either (`acm` {rep["swiglu"][("fw2_acm", "eval")] - rep["baseline"][("fw2_acm", "eval")]:+.3f}, `apc` {rep["swiglu"][("fw2_apc", "eval")] - rep["baseline"][("fw2_apc", "eval")]:+.3f}): its small gain on the trained Arabic sources
does not carry over to unseen text, and on `wiki_ary` it reverses. The unseen sources are noisier (up to 0.032 between seeds, D-029), so {wiki["swiglu"]:.3f} is about {wiki["swiglu"] / 0.032:.1f} times that noise, one seed.

**Keep / drop by D-031 (rules 1 to 5).**

- Rotary embeddings: **{decision["rope"]}**. They help Arabic and English, and their `wiki_ary` change ({wiki["rope"]:+.3f}) is under the {WIKI} line, so rule 4 does not apply to them.
- SwiGLU: **{decision["swiglu"]}**. It is {eff["swiglu"]["speed"]:+.0%} faster and {eff["swiglu"]["mem"]:+.0%} in memory, which would count for it only under rule 2 (neutral Arabic effect, no `wiki_ary` loss); rule 5 comes first when `wiki_ary` is worse.
- RMSNorm: **{decision["rmsnorm"]}**. By the rule it is a candidate to drop: simpler code with no gain in quality, speed or memory at this size.

Nothing is removed from the code (D-027), and `configs/modern-17m.yaml` is not changed by this entry: dropping RMSNorm from the reference would need a run of the package without it (rotary embeddings and SwiGLU), which is a cell of stage 2 and has not been run.

**What this does not show.** One seed per cell. The attribution is on top of GPT-2: a component's effect on top of the others comes only from stage 2, which the additivity test says is not needed for the Arabic mean (and the `wiki_ary` sums agree), but which was not run.
The SwiGLU Arabic effect is unresolved, not zero, and the `wiki_ary` numbers sit on the noisy unseen sources. Rotary embeddings remove the position table, so their cell is 1.15% smaller; this works against them. Speeds are medians over whole runs on a GPU whose throughput
varies from run to run (D-030); the SwiGLU speed difference is far outside that variation but its exact size is not precise. Whether the effects hold at other sizes or longer runs is not known.

**Decisions.**

1. The gain of the modern architecture on the trained languages is attributed mainly to rotary embeddings; SwiGLU adds a small gain that D-031 leaves unresolved and most of the speed and memory saving; RMSNorm contributes nothing measurable.
2. The loss on the unseen Moroccan Wikipedia is attributed mainly to SwiGLU, not to rotary embeddings (this replaces the suspicion in D-029).
3. Rotary embeddings are kept (rule 1). RMSNorm is a candidate to drop, not dropped. **SwiGLU is open**: by the pre-registered rules its fate depends on how its Arabic effect resolves, so the second seed that D-031 reserves for unresolved effects is now needed for it; the run and how it will be judged are written below, before it is run.
4. Stage 2 is not run (the effects are additive by the rule of D-031).

**The follow-up, registered here before it is run.** `configs/ablation-swiglu-only-seed2024.yaml`: the baseline with SwiGLU on, seed 2024, compared with the baseline of the same seed (`local-17m-seed2024`, already run). Judged only by the rules of D-031, with these two additions fixed now:
the Arabic effect is classified on the mean of the two seeds' effects ({eff["swiglu"]["ar"]:+.4f} at seed 1337 and the new one), with the same bands; `wiki_ary` counts as worse only if it is {WIKI} or more at both seeds (if only at one, it is "no difference"). Then rules 1 to 5 are applied as written;
if the Arabic mean still falls in the unresolved band, SwiGLU stays open and nothing more is run for it without a new decision. No other threshold changes.

**Not decided, and not tested here.**

- Whether to drop RMSNorm from the reference: a run of rotary embeddings + SwiGLU (no RMSNorm) would show that dropping it costs nothing; not run.
- What SwiGLU does to `wiki_ary` on top of rotary embeddings and the reverse: the stage-2 cells would show interactions; not run.
- A second seed for the borderline RMSNorm effect; it changes no decision above.
"""
out = Path(sys.argv[1])
out.write_text(text, encoding="utf-8")
print(f"wrote {out}: {len(text.splitlines())} lines")
