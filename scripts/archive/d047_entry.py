# ruff: noqa: E501
"""Builds the tables of the D-048 entry of docs/DECISIONS.md: the clean-checkout rerun of the reference run
(`repro-reference-17m`) against the original (`ablation-rope-swiglu`), judged by the criterion registered in D-047
before the run. Prints the verdict; the prose of the entry is written by hand around the tables.

Also writes the evaluation events of the two runs to reports/eval/final-d047-repro/evals.json.
Needs runs/{ablation-rope-swiglu,repro-reference-17m} and the full evaluations final-<same names> in reports/eval.
Run from the repository root: uv run python scripts/archive/d047_entry.py OUT.md"""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # scripts/archive/ -> the repository root
EVIDENCE = ROOT / "reports/eval/final-d047-repro"
AR = ["fw2_arb", "fw2_ars", "fw2_ary", "fw2_arz", "wiki_ar"]
EN = "fineweb_edu_10bt"
FINAL, WIKI, OTHER, CURVE, TRAIN = 0.005, 0.02, 0.01, 0.015, 0.02  # D-047
RUNS = {"orig": "ablation-rope-swiglu", "repro": "repro-reference-17m"}


def events(name: str) -> list[dict]:
    path = ROOT / f"runs/{name}/metrics.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]


log = {k: events(v) for k, v in RUNS.items()}
start = {k: v[0] for k, v in log.items()}
end = {k: [e for e in v if e["event"] == "end"][0] for k, v in log.items()}
ev = {k: {e["step"]: e for e in v if e["event"] == "eval"} for k, v in log.items()}
train = {k: {e["step"]: e for e in v if e["event"] == "train"} for k, v in log.items()}

# ---- preconditions (D-047)
pre = {}
a, b = (json.loads(json.dumps(start[k]["config"])) for k in ("orig", "repro"))
for c in (a, b):
    c["name"] = c["runs_dir"] = ""
    c["model"].pop("n_kv_heads", None)  # added later (D-044); None means the same model
pre["config equal except the name and runs_dir"] = a == b
pre["token files equal by SHA-256"] = start["orig"]["data"] == start["repro"]["data"]
pre["parameters 16,913,280"] = start["repro"]["parameters"] == 16_913_280
env = start["repro"]["environment"]
pre["Python 3.13.12, torch 2.14.1+cu130"] = (env["python"], env["torch"]) == (
    "3.13.12",
    "2.14.1+cu130",
)
pre["worktree clean"] = start["repro"]["git"]["dirty"] is False and bool(
    start["repro"]["git"]["commit"]
)
pre["no skipped steps, no resume"] = (
    end["repro"]["skipped_steps"] == 0
    and end["repro"]["step"] == 5000
    and not [e for e in log["repro"] if e["event"] in ("resume", "interrupted")]
)

EVIDENCE.mkdir(parents=True, exist_ok=True)
(EVIDENCE / "evals.json").write_text(
    json.dumps({k: {"run": RUNS[k], "evals": list(ev[k].values())} for k in RUNS}, indent=1) + "\n",
    encoding="utf-8",
)


def mean_ar(e: dict) -> float:
    return statistics.mean(e["sources"][s]["bits_per_char"] for s in AR)


def bpc(e: dict, s: str) -> float:
    return e["sources"][s]["bits_per_char"]


rep = {
    k: {
        (r["source"], r["split"]): r["bits_per_char"]
        for r in json.loads((ROOT / f"reports/eval/final-{v}/report.json").read_text())["rows"]
    }
    for k, v in RUNS.items()
}

# ---- criterion 1: step 5,000
d_ar = mean_ar(ev["repro"][5000]) - mean_ar(ev["orig"][5000])
d_en = bpc(ev["repro"][5000], EN) - bpc(ev["orig"][5000], EN)
# ---- criterion 2: full evaluation
full = {key: rep["repro"][key] - rep["orig"][key] for key in rep["orig"]}
d_wiki = full[("wiki_ary", "eval")]
others = {k: v for k, v in full.items() if k != ("wiki_ary", "eval")}
worst_other = max(others, key=lambda k: abs(others[k]))
# ---- criterion 3: the curve
steps = sorted(ev["orig"])
assert steps == sorted(ev["repro"]) and len(steps) == 20
curve = {s: mean_ar(ev["repro"][s]) - mean_ar(ev["orig"][s]) for s in steps}
blocks = {
    b: statistics.mean(
        train["repro"][s]["loss"] - train["orig"][s]["loss"] for s in range(b - 240, b + 1, 10)
    )
    for b in range(250, 5001, 250)
}
worst_curve = max(curve, key=lambda s: abs(curve[s]))
worst_block = max(blocks, key=lambda s: abs(blocks[s]))

checks = {
    "1. Arabic mean at step 5,000": (abs(d_ar), FINAL),
    "1. English at step 5,000": (abs(d_en), FINAL),
    "2. `wiki_ary`, full evaluation": (abs(d_wiki), WIKI),
    "2. every other source and split, full evaluation (largest)": (abs(others[worst_other]), OTHER),
    "3. Arabic mean at every evaluation (largest)": (abs(curve[worst_curve]), CURVE),
    "3. training loss, 250-step means (largest)": (abs(blocks[worst_block]), TRAIN),
}


def verdict(name: str, value: float, limit: float) -> str:
    """D-047: criterion 1 is unresolved up to 0.015 (the D-031 band); criteria 2 and 3 when missed by
    less than their own limit again."""
    if value < limit:
        return "holds"
    unresolved_below = 0.015 if name.startswith("1.") else 2 * limit
    return "unresolved" if value < unresolved_below else "**fails**"


results = {name: verdict(name, *vl) for name, vl in checks.items()}
if not all(pre.values()):
    overall = "no comparison: a precondition fails"
elif all(r == "holds" for r in results.values()):
    overall = "reproduced within tolerance"
elif any(r == "**fails**" for r in results.values()):
    overall = "not reproduced"
else:
    overall = "unresolved"

out = ["| Precondition (D-047) | holds |", "|---|---|"]
out += [f"| {k} | {'yes' if v else '**no**'} |" for k, v in pre.items()]
out += [
    "",
    f"Original: commit `{start['orig']['git']['commit'][:7]}`, {start['orig']['environment']['python']}, torch {start['orig']['environment']['torch']}. "
    f"Rerun: commit `{start['repro']['git']['commit'][:7]}`, {env['python']}, torch {env['torch']}, numpy {env['numpy']}.",
    "",
    "| Criterion (D-047): rerun minus original, bits/char | |D| | limit | result |",
    "|---|---:|---:|---|",
]
detail = {
    "1. Arabic mean at step 5,000": d_ar,
    "1. English at step 5,000": d_en,
    "2. `wiki_ary`, full evaluation": d_wiki,
    "2. every other source and split, full evaluation (largest)": others[worst_other],
    "3. Arabic mean at every evaluation (largest)": curve[worst_curve],
    "3. training loss, 250-step means (largest)": blocks[worst_block],
}
where = {
    "2. every other source and split, full evaluation (largest)": f" ({worst_other[0]}, {worst_other[1]})",
    "3. Arabic mean at every evaluation (largest)": f" (step {worst_curve})",
    "3. training loss, 250-step means (largest)": f" (steps {worst_block - 249} to {worst_block})",
}
for name, (_, limit) in checks.items():
    out.append(f"| {name}{where.get(name, '')} | {detail[name]:+.4f} | {limit} | {results[name]} |")
out += [
    "",
    "| Step | " + " | ".join(str(s) for s in steps[::2]) + " |",
    "|---|" + "---:|" * len(steps[::2]),
]
out.append(
    "| Arabic mean, rerun minus original | "
    + " | ".join(f"{curve[s]:+.4f}" for s in steps[::2])
    + " |"
)

# reported, not judged
t10 = (train["orig"][10]["loss"], train["repro"][10]["loss"])
tps = {k: statistics.median(e["tokens_per_s"] for e in train[k].values()) for k in RUNS}
peak = {k: max(e.get("peak_memory_mib") or 0 for e in train[k].values()) for k in RUNS}
loss_max = max(abs(train["repro"][s]["loss"] - train["orig"][s]["loss"]) for s in train["orig"])
out += [
    "",
    f"Reported, not judged: training loss at step 10, original {t10[0]!r}, rerun {t10[1]!r} ({'equal' if t10[0] == t10[1] else 'not equal'}); "
    f"largest single logged training-loss difference {loss_max:.4f}; median tokens/s {tps['orig']:,.0f} and {tps['repro']:,.0f}; "
    f"peak memory {peak['orig']:.0f} and {peak['repro']:.0f} MiB.",
    "",
    f"**Verdict (D-047): {overall}.**",
]
Path(sys.argv[1]).write_text("\n".join(out) + "\n", encoding="utf-8")
print(json.dumps({"preconditions": pre, "results": results, "verdict": overall}, indent=1))
