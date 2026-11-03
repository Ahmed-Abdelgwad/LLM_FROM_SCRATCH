"""Data diagnostics on the token files (read-only, CPU).

Run from the repository root: uv run python scripts/diagnose_tokens.py
(used for D-023, output in reports/diagnostics/final-d023/token-files.txt)."""

import hashlib
import json
from pathlib import Path

import numpy as np

tdir = Path("data/processed/tokens-16k")
metas = {}
for p in sorted(tdir.glob("*.tokens.json")):
    m = json.loads(p.read_text())
    if m["role"] == "training":
        metas[m["source"]] = m
names = list(metas)
V = 16384
eot = metas[names[0]]["end_of_text_id"]
print("end_of_text id", eot)

print("\n[1] documents (train split)")
print(
    f"{'source':18s}{'tokens':>12s}{'docs':>9s}{'mean':>7s}{'p10':>6s}{'p50':>6s}{'p90':>7s}"
    f"{'<64tok%':>9s}{'dup docs%':>10s}{'tok/char':>9s}"
)
counts = {}
for n in names:
    info = metas[n]["splits"]["train"]
    a = np.memmap(tdir / info["file"], dtype="<u2", mode="r")
    ends = np.flatnonzero(a == eot)
    starts = np.concatenate([[0], ends[:-1] + 1])
    lens = ends - starts + 1
    short = lens[lens < 64].sum() / lens.sum()
    seen, dups = set(), 0
    for s, e in zip(starts, ends, strict=True):
        h = hashlib.blake2b(a[s:e].tobytes(), digest_size=8).digest()
        if h in seen:
            dups += 1
        seen.add(h)
    print(
        f"{n:18s}{info['tokens']:12,d}{len(lens):9,d}{lens.mean():7.0f}"
        f"{np.percentile(lens, 10):6.0f}{np.percentile(lens, 50):6.0f}"
        f"{np.percentile(lens, 90):7.0f}{100 * short:9.2f}{100 * dups / len(lens):10.2f}"
        f"{info['tokens'] / info['chars']:9.3f}"
    )
    counts[n] = np.bincount(np.asarray(a[:20_000_000]), minlength=V).astype(np.float64)

print("\n[2] unigram statistics (first 20M tokens of each train file)")
P = {n: c / c.sum() for n, c in counts.items()}
ent = {n: float(-(p[p > 0] * np.log(p[p > 0])).sum()) for n, p in P.items()}
print(f"{'source':18s}{'H1 nats':>9s}{'ids>1e-6':>10s}{'ids>1e-5':>10s}")
for n in names:
    print(f"{n:18s}{ent[n]:9.2f}{int((P[n] > 1e-6).sum()):10d}{int((P[n] > 1e-5).sum()):10d}")


def js(p, q):
    m = 0.5 * (p + q)

    def f(x):
        return float((x[x > 0] * np.log(x[x > 0] / m[x > 0])).sum())

    return 0.5 * f(p) + 0.5 * f(q)


print("\nJensen-Shannon divergence between unigram distributions (nats; max ln2=0.693)")
print(f"{'':18s}" + "".join(f"{n[-7:]:>9s}" for n in names))
for a in names:
    print(f"{a:18s}" + "".join(f"{js(P[a], P[b]):9.3f}" for b in names))

en, ar = P["fineweb_edu_10bt"], sum(P[n] for n in names if n != "fineweb_edu_10bt") / 5
mass_ar_only = float(ar[(en < 1e-6)].sum())
mass_en_only = float(en[(ar < 1e-6)].sum())
print(
    f"\nprobability mass on ids that English almost never uses (<1e-6): Arabic {mass_ar_only:.3f}"
)
print(f"probability mass on ids that Arabic almost never uses (<1e-6): English {mass_en_only:.3f}")
print(
    f"ids used (>1e-6) by English only: {int(((en > 1e-6) & (ar <= 1e-6)).sum())}, "
    f"Arabic only: {int(((ar > 1e-6) & (en <= 1e-6)).sum())}, "
    f"both: {int(((ar > 1e-6) & (en > 1e-6)).sum())}, "
    f"neither: {int(((ar <= 1e-6) & (en <= 1e-6)).sum())}"
)

print(
    "\n[3] repetition inside random 512-token windows "
    "(share of positions whose preceding 4-gram already occurred in the window)"
)
rng = np.random.default_rng(0)
for n in names:
    info = metas[n]["splits"]["train"]
    a = np.memmap(tdir / info["file"], dtype="<u2", mode="r")
    fr = []
    for _ in range(300):
        s = int(rng.integers(len(a) - 513))
        w = a[s : s + 512].tolist()
        seen, rep = set(), 0
        for i in range(4, 512):
            g = tuple(w[i - 4 : i])
            if g in seen:
                rep += 1
            seen.add(g)
        fr.append(rep / 508)
    fr = np.array(fr)
    print(
        f"{n:18s} mean {fr.mean():.3f}  p90 {np.percentile(fr, 90):.3f}  "
        f"windows with >30% repeated: {100 * (fr > 0.3).mean():.1f}%"
    )
