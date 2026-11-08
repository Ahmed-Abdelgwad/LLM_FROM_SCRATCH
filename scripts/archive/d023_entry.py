# ruff: noqa: E501
"""Builds the D-023 entry of docs/DECISIONS.md from the run files and the diagnostic outputs.
Every table and every number in the text comes from a file (runs/, reports/); the prose around
them is written here. Needs runs/ as it was when D-023 was written (they are not in git).
Run from the repository root: uv run python scripts/archive/d023_entry.py OUT.md"""

import json
import math
import re
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # scripts/archive/ -> the repository root
DIAG = ROOT / "reports/diagnostics/final-d023"
SHORT = {
    "fineweb_edu_10bt": "edu", "fw2_arb": "arb", "fw2_ars": "ars", "fw2_ary": "ary",
    "fw2_arz": "arz", "wiki_ar": "wiki_ar", "fw2_acm": "acm", "fw2_apc": "apc", "wiki_ary": "wiki_ary",
}  # fmt: skip


def must(m: re.Match[str] | None) -> re.Match[str]:
    """A pattern the diagnostic output is known to contain; fail loudly if it does not."""
    assert m is not None
    return m


def events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# ------------------------------------------------------------------ baseline run
base = events(ROOT / "runs/local-17m/metrics.jsonl")
start = base[0]
cfg = start["config"]
train = [e for e in base if e["event"] == "train"]
evals = [e for e in base if e["event"] == "eval"]
marks = [
    (e["event"], e["step"], e["time"])
    for e in base
    if e["event"] in ("start", "interrupted", "resume", "end")
]
end = [e for e in base if e["event"] == "end"][0]
final = evals[-1]
names = list(final["sources"])
weights = start["sampling"]["weights"]
total_w = sum(weights.values())
share = {n: weights[n] / total_w for n in names}
tok_weighted = sum(final["sources"][n]["loss"] * share[n] for n in names)
ev2750 = [e for e in evals if e["step"] == 2750][0]
weighted_2750 = sum(ev2750["sources"][n]["loss"] * share[n] for n in names)
train_tail = statistics.mean(e["loss"] for e in train[-5:])
rate = statistics.median(e["tokens_per_s"] for e in train)
params = start["parameters"]
m = cfg["model"]
emb = m["vocab_size"] * m["d_model"]
pos = m["context_length"] * m["d_model"]
non_emb = params - emb - pos
over1 = [e for e in train if e["grad_norm"] > 1.0]
late_steps = [e for e in train if e["step"] > 200]
gmax_after = max(e["grad_norm"] for e in late_steps)
gmed = statistics.median(e["grad_norm"] for e in late_steps)
lr_at_stop = [e["lr"] for e in train if e["step"] == marks[1][1]][0]

curve = ["| step | tokens seen | mean val loss | mean bits/char |", "|---:|---:|---:|---:|"]
tokens_at = {e["step"]: e["tokens_seen"] for e in train}
for e in evals:
    if e["step"] % 500 == 0:
        curve.append(
            f"| {e['step']:,} | {tokens_at[e['step']]:,} | {e['val_loss']:.3f} | {e['val_bits_per_char']:.3f} |"
        )

per_source = [
    "| source | share of train tokens | val loss | val bits/char | train-probe loss | val − probe |",
    "|---|---:|---:|---:|---:|---:|",
]
for n in names:
    s = final["sources"][n]
    per_source.append(
        f"| {SHORT[n]} | {share[n]:.1%} | {s['loss']:.3f} | {s['bits_per_char']:.3f} | {s['train_loss']:.3f} | {s['loss'] - s['train_loss']:+.3f} |"
    )
at2000 = {e["step"]: e for e in evals}
gain = {
    n: at2000[2000]["sources"][n]["bits_per_char"] - at2000[2750]["sources"][n]["bits_per_char"]
    for n in names
}

gap_w = sum(
    share[n] * (final["sources"][n]["loss"] - final["sources"][n]["train_loss"]) for n in names
)
gap_ars_arz = sum(
    share[n] * (final["sources"][n]["loss"] - final["sources"][n]["train_loss"])
    for n in ("fw2_ars", "fw2_arz")
)

# ------------------------------------------------------------------ full evaluation vs pilot
new = json.loads((ROOT / "reports/eval/final-local-17m/report.json").read_text())
old = json.loads((ROOT / "reports/eval/final-pilot-17m/report.json").read_text())
oldrows = {(r["source"], r["split"]): r for r in old["rows"]}
ev_rows = [
    "| source | split | tokens | loss | perplexity | bits/char | pilot bits/char | change | unigram bits/char | model / unigram |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in new["rows"]:
    p = oldrows[(r["source"], r["split"])]
    ev_rows.append(
        f"| {SHORT[r['source']]} | {r['split']} | {r['tokens']:,} | {r['loss']:.3f} | {r['perplexity']:.0f} | {r['bits_per_char']:.3f} "
        f"| {p['bits_per_char']:.3f} | {r['bits_per_char'] - p['bits_per_char']:+.3f} | {r['unigram_bits_per_char']:.3f} "
        f"| {r['bits_per_char'] / r['unigram_bits_per_char']:.2f} |"
    )


def group_mean(report: dict, sources: list[str], split: str) -> float:
    rows = {(r["source"], r["split"]): r["bits_per_char"] for r in report["rows"]}
    return statistics.mean(rows[(s, split)] for s in sources)


trained_ar = ["fw2_arb", "fw2_ars", "fw2_ary", "fw2_arz", "wiki_ar"]
unseen = ["fw2_acm", "fw2_apc", "wiki_ary"]
new_tr, new_un = group_mean(new, trained_ar, "val"), group_mean(new, unseen, "eval")
old_tr, old_un = group_mean(old, trained_ar, "val"), group_mean(old, unseen, "eval")
changes = [
    r["bits_per_char"] - oldrows[(r["source"], r["split"])]["bits_per_char"]
    for r in new["rows"]
    if r["split"] != "eval"
]
rows_by = {(r["source"], r["split"]): r["bits_per_char"] for r in new["rows"]}
splitgaps = [
    abs(rows_by[(s, "val")] - rows_by[(s, "test")]) for s in ["fineweb_edu_10bt", *trained_ar]
]

# ------------------------------------------------------------------ learning-rate sweep and noise
sweep = [
    "| peak LR | val loss at 200 | 400 | 600 | 800 | mean bits/char at 800 | max grad norm after step 100 | skipped steps |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
sweep_final = {}
for tag in ["3e-4", "6e-4", "1.2e-3", "2.4e-3"]:
    ev = events(ROOT / f"runs/diag-lr-{tag}/metrics.jsonl")
    es = [e for e in ev if e["event"] == "eval"]
    tr = [e for e in ev if e["event"] == "train"]
    sweep_final[tag] = es[-1]
    sweep.append(
        f"| {tag} | "
        + " | ".join(f"{e['val_loss']:.3f}" for e in es)
        + f" | {es[-1]['val_bits_per_char']:.3f} "
        f"| {max(e['grad_norm'] for e in tr if e['step'] > 100):.2f} | {sum(e['event'] == 'skip' for e in ev)} |"
    )
ev2 = events(ROOT / "runs/diag-lr-6e-4-seed2/metrics.jsonl")
s2 = [e for e in ev2 if e["event"] == "eval"][-1]
s1 = sweep_final["6e-4"]
d_val, d_bpc = s2["val_loss"] - s1["val_loss"], s2["val_bits_per_char"] - s1["val_bits_per_char"]
d_src = [s2["sources"][n]["bits_per_char"] - s1["sources"][n]["bits_per_char"] for n in names]
lr_gap_min = sweep_final["1.2e-3"]["val_loss"] - s1["val_loss"]
lr_gap_max = sweep_final["3e-4"]["val_loss"] - s1["val_loss"]


# ------------------------------------------------------------------ diagnostics parsed from text files
def parse_diag(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    out: dict = {"step": int(must(re.search(r"step (\d+)", text)).group(1))}
    out["b_noise"] = int(
        must(re.search(r"B_noise = trace/\|G\|\^2 ~ ([\d,]+) tokens", text))
        .group(1)
        .replace(",", "")
    )
    out["ratio"] = float(must(re.search(r"batch / B_noise = ([\d.]+)", text)).group(1))
    out["gsmall"] = float(must(re.search(r"\|G_small\|\^2=([\d.]+)", text)).group(1))
    out["gbig"] = float(must(re.search(r"\|G_big\|\^2=([\d.]+)", text)).group(1))
    block = (
        text.split("diagonal = two independent batches of the same source)")[1]
        .strip()
        .splitlines()[1:7]
    )
    out["cos"] = np.array([[float(x) for x in line.split()[1:]] for line in block])
    pos = [
        line
        for line in text.splitlines()
        if line.split() and line.split()[0] in SHORT or line.startswith("fineweb")
    ][:6]
    out["bf16_gap"] = max(abs(float(row.split()[-2]) - float(row.split()[-1])) for row in pos)
    out["late_minus_mid"] = [
        float(row.split()[5]) - float(row.split()[6]) for row in pos
    ]  # 64-255 minus 256-511
    return out


d27, d50 = (
    parse_diag(DIAG / "step2750-checkpoint.txt"),
    parse_diag(DIAG / "step5000-checkpoint.txt"),
)
order = ["edu", "arb", "ars", "ary", "arz", "wiki_ar"]


def normalized(c: np.ndarray) -> np.ndarray:
    d = np.diag(c)
    n = c / np.sqrt(np.outer(d, d))
    return (n + n.T) / 2


def matrix_table(c: np.ndarray) -> list[str]:
    n = normalized(c)
    lines = ["| | " + " | ".join(order) + " |", "|---|" + "---:|" * 6]
    for i, a in enumerate(order):
        lines.append(
            f"| {a} | " + " | ".join("1" if i == j else f"{n[i, j]:.2f}" for j in range(6)) + " |"
        )
    return lines


def offdiag(c: np.ndarray, group_a: list[int], group_b: list[int]) -> tuple[float, float]:
    n = normalized(c)
    vals = [n[i, j] for i in group_a for j in group_b if i != j]
    return min(vals), max(vals)


def min_cross(c: np.ndarray) -> float:
    raw = c.copy()
    np.fill_diagonal(raw, np.nan)
    return float(np.nanmin(raw))


tokfiles = (DIAG / "token-files.txt").read_text(encoding="utf-8")
rep = re.findall(
    r"^(\w+)\s+mean ([\d.]+)\s+p90 [\d.]+\s+windows with >30% repeated: ([\d.]+)%",
    tokfiles,
    flags=re.M,
)
rep_mean = [float(r[1]) for r in rep]
rep_over = [float(r[2]) for r in rep]
dups = re.findall(
    r"^\w+\s+[\d,]+\s+[\d,]+\s+\d+\s+\d+\s+\d+\s+\d+\s+[\d.]+\s+([\d.]+)\s+[\d.]+$",
    tokfiles,
    flags=re.M,
)
shorts = re.findall(
    r"^\w+\s+[\d,]+\s+[\d,]+\s+\d+\s+\d+\s+\d+\s+\d+\s+([\d.]+)\s+[\d.]+\s+[\d.]+$",
    tokfiles,
    flags=re.M,
)
jsm = (
    must(re.search(r"Jensen-Shannon.*?\n(.*?)\n\nprobability", tokfiles, flags=re.S))
    .group(1)
    .splitlines()[1:]
)
js = np.array([[float(x) for x in row.split()[1:]] for row in jsm])
ids = must(
    re.search(
        r"ids used \(>1e-6\) by English only: (\d+), Arabic only: (\d+), both: (\d+), neither: (\d+)",
        tokfiles,
    )
).groups()
mass_en_only = float(
    must(re.search(r"Arabic almost never uses \(<1e-6\): English ([\d.]+)", tokfiles)).group(1)
)
tpc = [
    float(x)
    for x in re.findall(
        r"^\w+\s+[\d,]+\s+[\d,]+\s+\d+\s+\d+\s+\d+\s+\d+\s+[\d.]+\s+[\d.]+\s+([\d.]+)$",
        tokfiles,
        flags=re.M,
    )
]
overfit = (DIAG / "overfit-one-batch.txt").read_text().splitlines()
of_init, of_last = float(overfit[2].split()[2]), float(overfit[-1].split()[1])

# ------------------------------------------------------------------ scaling-law reference
E, A, B, a_, b_ = 1.69, 406.4, 410.7, 0.34, 0.28


def chinchilla(n: float, d: float) -> float:
    return E + A / n**a_ + B / d**b_


seen_mid = tokens_at[2750]
c_total, c_nonemb = chinchilla(params, seen_mid), chinchilla(non_emb, seen_mid)
c_end_total = chinchilla(params, end["tokens_seen"])
c_drop = c_total - c_end_total
macro_2750 = at2000[2750]["val_loss"]


def chinchilla_exact(n: float, d: float) -> float:
    return 1.6934 + 406.4 / n**0.3392 + 410.7 / d**0.2849


c_drop_exact = chinchilla_exact(params, seen_mid) - chinchilla_exact(params, end["tokens_seen"])

# ------------------------------------------------------------------ the entry
t_start, t_int, t_res, t_end = (m_[2] for m_ in marks)
text = f"""
## D-023 (2026-11-03): the 5,000-step baseline, and why the loss falls slowly: nothing broken was found

Code: none changed. Evidence kept in git: `reports/eval/final-local-17m` (full evaluation of the
final checkpoint) and `reports/diagnostics/final-d023/` (outputs of the diagnostic scripts).
Scripts: `scripts/diagnose_checkpoint.py`, `scripts/diagnose_tokens.py` and `scripts/overfit_one_batch.py`
made the files in `reports/diagnostics/final-d023/`; `scripts/archive/d023_entry.py` computed the tables and numbers
of this entry from the runs and those files (the prose around them is written by hand). Not in git:
the runs (`runs/local-17m`, `runs/diag-lr-*`, ignored), so the entry can be regenerated only on the machine
that has them. Rerunning `diagnose_checkpoint.py` on the same checkpoint gives identical output except
the last digits of the noise-scale lines, which change from run to run (GPU kernels are not bitwise
deterministic).

**The baseline run.** `configs/local-17m.yaml`, unchanged from the pilot except for length:
{params:,} parameters ({emb:,} in the token embedding, {pos:,} in the position embedding, {non_emb:,} elsewhere),
{cfg["train"]["micro_batch_size"]} x {cfg["train"]["grad_accum_steps"]} x {m["context_length"]} = {start["tokens_per_step"]:,} tokens per step,
natural mix, peak learning rate {cfg["optim"]["lr"]:g} decaying to {cfg["optim"]["min_lr"]:g}. It was stopped by SIGTERM at step {marks[1][1]:,} ({t_int[:16]}Z;
`last.pt` was written) and resumed from that checkpoint at {t_res[:16]}Z, then ran to step {end["step"]:,}.
{end["tokens_seen"]:,} tokens were seen ({end["tokens_seen"] / sum(weights.values()):.2f} epoch of the
{int(sum(weights.values())):,} train tokens), {end["skipped_steps"]} steps were skipped, the median throughput was {rate:,.0f} tokens/s.

{chr(10).join(curve)}

Final per-source state (step {end["step"]:,}); the probe is the loss on train windows, measured the same way as validation:

{chr(10).join(per_source)}

Mean validation loss {final["val_loss"]:.3f} nats, {final["val_bits_per_char"]:.3f} bits/char. Weighted by the
train mix the validation loss is {tok_weighted:.3f}, against {train_tail:.3f} for the training loss (mean of the last five logged
windows): the same weighting, and a difference of {tok_weighted - train_tail:.3f}, which is the generalization gap
in the usual sense; `ars` and `arz` (below) account for {gap_ars_arz / gap_w:.0%} of the train-mix weighted val-minus-probe gap of {gap_w:.3f}. The headline `val_loss` is an unweighted mean over six
sources, which is why it looks {final["val_loss"] - train_tail:.2f} higher than the training loss; the rest of that
difference ({final["val_loss"] - tok_weighted:.2f}) is the weighting, not a gap. The per-source
probe shows where the gap is: `ars` ({final["sources"]["fw2_ars"]["loss"] - final["sources"]["fw2_ars"]["train_loss"]:+.2f}, the concentrated
validation sites of D-021) and `arz` ({final["sources"]["fw2_arz"]["loss"] - final["sources"]["fw2_arz"]["train_loss"]:+.2f}, not investigated); the other four are within
{max(abs(final["sources"][n]["loss"] - final["sources"][n]["train_loss"]) for n in names if n not in ("fw2_ars", "fw2_arz")):.2f} nats.

**Full evaluation** of `best.pt` (step {new["step"]:,}), every non-overlapping 512-token window of val, test and the
evaluation-only dialects; `change` is against the pilot (D-022):

{chr(10).join(ev_rows)}

- Every split of a training source improved by {-max(changes):.2f} to {-min(changes):.2f} bits/char since the pilot (step 1,500), all of them by a similar amount.
- Mean of the five trained Arabic sources (val): {new_tr:.3f}; mean of the three dialects never seen: {new_un:.3f}. The ratio is
  {new_un / new_tr:.2f}; at the pilot it was {old_un / old_tr:.2f} ({old_un:.3f} / {old_tr:.3f}). The relative price of an unseen dialect did not move;
  what the model learned about the seen ones carried over in the same proportion.
- wiki_ary improved by {oldrows[("wiki_ary", "eval")]["bits_per_char"] - rows_by[("wiki_ary", "eval")]:.2f}, about twice the rest; it is the smallest of the three and the cause was not looked into.
- For one source, val and test differ by up to {max(splitgaps):.2f} bits/char (set against seed noise below): compare a split with the same
  split, never val with test.

**Why the loss falls slowly: what was checked and what it showed.** Each check is a measurement on the baseline or a short run
with one change.

- *Implementation.* The exact model, on CPU, memorizes four 128-token windows: loss {of_init:.2f} at the start (ln 16,384 = {math.log(16384):.2f}),
  {of_last:.4f} after 120 steps. Bf16 against fp32 on the validation windows differs by at most {d50["bf16_gap"]:.3f} nats on every source.
  The loop's own checks stand from D-020 (resume bit-exact on CPU, mutants caught). No defect found.
- *Gradient clipping.* The clip is {cfg["optim"]["grad_clip"]}. The log holds the norm of every 10th step: {len(over1)} of {len(train)} logged steps were above it, the last at step {over1[-1]["step"] if over1 else 0}
  (during warmup); after step 200 the median is {gmed:.2f} and the maximum {gmax_after:.2f}. Clipping acted only at the very start, and it does not slow the run.
- *Learning rate.* Four runs of 800 steps, identical except the peak learning rate (minimum 10% of it, 50 warmup steps, seed 1337).
  Command, for the rate LR (the minimum is LR/10): `scratchllm train configs/local-17m.yaml --set name=diag-lr-X --set optim.lr=LR
  --set optim.min_lr=LR/10 --set optim.warmup_steps=50 --set train.max_steps=800 --set train.eval_interval=200
  --set train.checkpoint_interval=800`; the repeat adds `--set train.seed=2024`
  and `--set name=diag-lr-6e-4-seed2`:

{chr(10).join(sweep)}

  {sweep_final["6e-4"]["val_loss"]:.3f} at 6e-4 is the best; 3e-4 is {lr_gap_max:.3f} worse and 1.2e-3 is {lr_gap_min:.3f} worse, with no instability at any rate.
  A higher rate was the hypothesis going in (6e-4 is a value for much wider models) and it is refuted at this horizon. A short sweep tends to favour
  higher rates than a long run does, so the best rate at 5,000 steps is unlikely to be higher than 6e-4.
- *Noise.* The 6e-4 run repeated with seed 2024 (different initialization and data order, same validation windows) gives {s2["val_loss"]:.3f} against {s1["val_loss"]:.3f}:
  {d_val:+.3f} nats, {d_bpc:+.3f} bits/char on the mean, {min(d_src):+.3f} to {max(d_src):+.3f} per source. One pair is a rough estimate of the noise; the
  learning-rate gaps above are {lr_gap_min / abs(d_val):.1f} to {lr_gap_max / abs(d_val):.1f} times it.
- *Batch size.* Gradient noise scale (McCandlish et al. 2018), 24 steps of 8 micro-batches, natural mix:
  B_noise = {d27["b_noise"]:,} tokens at step {d27["step"]:,} (batch/B_noise = {d27["ratio"]:.2f}) and {d50["b_noise"]:,} at step {d50["step"]:,} ({d50["ratio"]:.2f}).
  It grows as the loss falls, as the theory says, so the batch of {start["tokens_per_step"]:,} tokens is further below it as training advances.
  Below the noise scale every token is used with efficiency {1 / (1 + d50["ratio"]):.0%} at step {d50["step"]:,} ({1 / (1 + d27["ratio"]):.0%} at step {d27["step"]:,}) of the best possible
  (E/E_min = 1 + B/B_noise); halving the batch would save about {(1 - (1 + d50["ratio"] / 2) / (1 + d50["ratio"])):.0%} of the tokens at step {d50["step"]:,} and
  {(1 - (1 + d27["ratio"] / 2) / (1 + d27["ratio"])):.0%} at step {d27["step"]:,}, with the learning rate to be retuned. The estimate is 24 steps at one checkpoint.
  The batch is not a cause of the slow fall.
- *Data.* Train split, from the token files: exact duplicate documents (within a source) {", ".join(sorted(set(dups)))}% in every source; tokens in documents of fewer than 64 tokens {", ".join(sorted(set(shorts)))}%;
  the share of positions inside a 512-window whose previous 4-gram already occurred in it is {min(rep_mean):.3f} to {max(rep_mean):.3f} on average, and
  windows with more than 30% of them: {min(rep_over):.1f}% to {max(rep_over):.1f}% (wiki_ar, lists and tables).
- *Sources.* Cosine between the gradients of two 32,768-token batches, divided by what two batches of the same source reach (so 1 would mean as
  alike as the source is to itself). Step {d50["step"]:,}:

{chr(10).join(matrix_table(d50["cos"]))}

  Step {d27["step"]:,}:

{chr(10).join(matrix_table(d27["cos"]))}

  Raw cosines between different sources were never negative at either step (smallest {min_cross(d27["cos"]):.3f} and {min_cross(d50["cos"]):.3f}): no
  sign of one source working against another. Arabic sources with each other are {offdiag(d27["cos"], [1, 3, 4, 5], [1, 3, 4, 5])[0]:.2f} to {offdiag(d27["cos"], [1, 3, 4, 5], [1, 3, 4, 5])[1]:.2f} at step {d27["step"]:,} (arb, ary, arz,
  wiki_ar); English with Arabic {offdiag(d27["cos"], [0], [1, 2, 3, 4, 5])[0]:.2f} to {offdiag(d27["cos"], [0], [1, 2, 3, 4, 5])[1]:.2f}; ars is close only to arz (the other Arabic sources
  {offdiag(d27["cos"], [2], [1, 3, 5])[0]:.2f} to {offdiag(d27["cos"], [2], [1, 3, 5])[1]:.2f}). At step {d50["step"]:,} every value is smaller, because the
  gradient is mostly noise by then (the diagonal itself fell from {np.mean(np.diag(d27["cos"])):.2f} to {np.mean(np.diag(d50["cos"])):.2f}), so the later matrix says less. Token
  statistics agree: the Jensen-Shannon divergence between unigram distributions is {js[1:, 1:][np.triu_indices(5, 1)].min():.2f} to {js[1:, 1:][np.triu_indices(5, 1)].max():.2f} among the Arabic sources
  and {js[0, 1:].min():.2f} to {js[0, 1:].max():.2f} between English and any of them (maximum 0.69); {int(ids[0]):,} ids are used by English only, {int(ids[1]):,} by Arabic only, {int(ids[2]):,} by both.
  Under 'is it appropriate to combine them': nothing found against it, and nothing measured for it either, see the open points.
- *Where the loss goes, by position.* On the validation windows the loss drops from the middle of the window (64-255) to the end (256-511) by only
  {min(d50["late_minus_mid"]):.2f} to {max(d50["late_minus_mid"]):.2f} nats per source: the second half of the window adds little over the middle. The windows do not start at document starts, so
  this says how much context helps within a window, not how well long-range structure is understood.
- *Scale.* With the rounded Hoffmann et al. fit L = 1.69 + 406.4/N^0.34 + 410.7/D^0.28 (a different corpus and tokenizer, and the fit itself is disputed;
  read as an order of magnitude), N = {params / 1e6:.1f}M and D = {seen_mid / 1e6:.0f}M tokens (step 2,750) give {c_total:.2f} nats ({c_nonemb:.2f} counting only the {non_emb / 1e6:.1f}M non-embedding parameters);
  the train-mix weighted validation loss at that step was {weighted_2750:.2f}, below it. Going on to {end["tokens_seen"] / 1e6:.0f}M tokens the fit predicts a further fall of
  {c_drop:.2f} nats ({c_drop_exact:.2f} with the unrounded exponents of the replication paper); the weighted validation loss fell by {weighted_2750 - tok_weighted:.2f} ({macro_2750 - final["val_loss"]:.2f} for the unweighted mean), so the fit overstates the
  further fall. It is for another corpus and tokenizer, so this is a limit of the fit, not a symptom. The run was announced before it finished as
  "final mean validation loss between 4.4 and 4.6" and ended at {final["val_loss"]:.3f}: inside the range, in its upper half. The model is in the range expected for its size and its budget of
  {end["tokens_seen"] / params:.1f} tokens per parameter ({end["tokens_seen"] / non_emb:.1f} per non-embedding one; about 20 is the usual compute-optimal figure), not better or worse by a margin this analysis can resolve.

**Conclusion.** No implementation, optimizer, precision, batch or data fault was found. The loss falls as a 17M-parameter model on 164M tokens
is expected to fall. What makes it look worse than it is: the headline validation loss and the training loss are weighted differently; per-token
losses of different sources are not comparable (tokens per character range from {min(tpc):.3f} to {max(tpc):.3f}); and the first
stop, at step {marks[1][1]:,}, left the learning rate at {lr_at_stop:.1e}, in the middle of its decay, so the curve looked unfinished.

**Samples** (the same six prompts, settings and seeds as D-022; kept in `runs/local-17m/samples-*.txt`, not in git, for the same reason):
the first greedy sentence for the health-ministry prompt is now a plausible news sentence (a number of new deaths from a virus), where the
pilot's was not; greedy decoding still falls into a loop in all six prompts, after roughly 20 to 40 tokens. With sampling, Arabic keeps its
register and keeps drifting from the topic; web boilerplate appears ("تسجيل الدخول في وقت لاحق", a repeated "أبرزها", a chat-transcript speaker
label); the Egyptian prompts continue in a recognizable colloquial style; the English samples are the weakest, with invented or broken words
("jumpetal", "ented", "Self-Self-Self-") although English is the largest source. A possible reason is that the vocabulary was built for Arabic
and cuts English into more pieces (1.78 tokens per word against 1.32 for GPT-2, D-016); this was not tested. Six prompts and two seeds are an impression.

**Decisions.**

1. Hyperparameters, batch, clipping, precision and architecture of `local-17m` are kept; this run is the baseline every later change is compared with.
   Learning rate 6e-4 is not raised.
2. Bits per character is the number to compare runs and sources. Report it next to the unweighted mean loss and the train-mix weighted loss, and
   always on the same split.
3. Rule fixed before the next experiment, from the noise above: a change counts as an improvement only if the mean bits/char on the same split falls by
   at least 0.03 (about twice the {abs(d_bpc):.3f} measured) and, for a claim about one source, by at least 0.04.

**Not decided, and not tested here.**

- Whether the English share (the largest, {share["fineweb_edu_10bt"]:.0%}) costs the Arabic sources anything. The gradient evidence says the sources do not fight, and
  it says English shares little with Arabic; neither says what removing it would do. It needs the mix ablation (natural, Arabic only, English at 20%;
  800 steps each, judged by rule 3), and it depends on the project wanting an English-capable model.
- Whether several epochs on the same {int(sum(weights.values()) / 1e6)}M tokens, or more data, helps; the literature suggests repeated data is nearly as good as new for a few epochs (recalled, not checked in this session).
- The English tokenization hypothesis above; the `arz` train/validation gap; the single-seed basis of every run except the one repeated.
- README still shows the pilot results; it has not been updated.
"""

out = Path(sys.argv[1])
out.write_text(text, encoding="utf-8")
print(f"wrote {out}: {len(text.splitlines())} lines")
