"""Diagnostics on a trained checkpoint: position-wise loss, bf16 vs fp32, gradient noise scale,
gradient agreement between sources. Read-only: writes nothing but stdout.

Run from the repository root: uv run python scripts/diagnose_checkpoint.py runs/NAME/best.pt
(needs a GPU; used for D-023, output in reports/diagnostics/final-d023/)."""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from scratchllm.training.batches import (
    evenly_spaced,
    load_source,
    training_source_names,
    validation_sets,
)
from scratchllm.training.checkpoint import load_checkpoint, model_from_state
from scratchllm.training.dataset import TokenWindows

ckpt = Path(sys.argv[1])
dev = "cuda"
state = load_checkpoint(ckpt, map_location=dev)
model = model_from_state(state, dev)
cfg = state["config"]
tdir = Path(cfg["data"]["tokens_dir"])
names = training_source_names(tdir)
srcs = [load_source(tdir, n) for n in names]
T = cfg["model"]["context_length"]
print(f"checkpoint {ckpt} step {state['step']}")

# ---------------------------------------------------------------- 1. loss by position, bf16 vs fp32
val = validation_sets(srcs, T)
buckets = [(0, 1), (1, 4), (4, 16), (16, 64), (64, 256), (256, 512)]
print("\n[1] val loss by position in the 512 window (nats/token), bf16 autocast")
print(
    f"{'source':18s}" + "".join(f"{f'{a}-{b - 1}':>9s}" for a, b in buckets) + "   all   fp32-all"
)
for s in srcs:
    ds = val[s.name]
    picks = evenly_spaced(len(ds), 160)
    per_pos = torch.zeros(T, device=dev)
    per_pos32 = 0.0
    for i in range(0, len(picks), 8):
        chunk = [ds[j] for j in picks[i : i + 8]]
        x = torch.stack([a for a, _ in chunk]).to(dev)
        y = torch.stack([b for _, b in chunk]).to(dev)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x, y)
            per_pos += F.cross_entropy(logits.float().transpose(1, 2), y, reduction="none").sum(0)
            logits32, _ = model(x, y)  # plain fp32
            per_pos32 += float(
                F.cross_entropy(logits32.transpose(1, 2), y, reduction="none").mean()
            ) * len(chunk)
    per_pos /= len(picks)
    row = "".join(f"{float(per_pos[a:b].mean()):9.3f}" for a, b in buckets)
    print(f"{s.name:18s}{row}  {float(per_pos.mean()):.3f}  {per_pos32 / len(picks):.3f}")

# ---------------------------------------------------------------- 2. gradient noise scale
rng = np.random.default_rng(0)
train_sets = {s.name: TokenWindows([s.train_path], T, stride=1) for s in srcs}
weights = np.array([s.train_tokens for s in srcs], dtype=float)
weights /= weights.sum()


def batch(source_idx: np.ndarray):
    pairs = []
    for k in source_idx:
        d = train_sets[names[k]]
        pairs.append(d[int(rng.integers(len(d)))])
    return torch.stack([a for a, _ in pairs]).to(dev), torch.stack([b for _, b in pairs]).to(dev)


params = [p for p in model.parameters() if p.requires_grad]


def grad_vector(x, y):
    model.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = model(x, y)
    loss.backward()
    return torch.cat([p.grad.reshape(-1).float() for p in params]).clone()


print("\n[2] gradient noise scale (McCandlish et al. 2018), natural mixture")
micro, accum, K = 8, 8, 24
b_small, b_big = micro * T, micro * accum * T
small_sq, big_sq = [], []
for _ in range(K):
    total = None
    for _ in range(accum):
        g = grad_vector(*batch(rng.choice(len(srcs), size=micro, p=weights)))
        small_sq.append(float(g @ g))
        total = g if total is None else total + g
    total /= accum
    big_sq.append(float(total @ total))
gs, gb = np.mean(small_sq), np.mean(big_sq)
g2 = (b_big * gb - b_small * gs) / (b_big - b_small)
sigma = (gs - gb) / (1 / b_small - 1 / b_big)
print(f"|G_small|^2={gs:.4f} (B={b_small})  |G_big|^2={gb:.4f} (B={b_big})")
print(
    f"true |G|^2 ~ {g2:.4f}   trace(Sigma) ~ {sigma:.1f}   "
    f"B_noise = trace/|G|^2 ~ {sigma / g2:,.0f} tokens"
)
print(f"current batch = {b_big:,} tokens -> batch / B_noise = {b_big / (sigma / g2):.2f}")

# ------------------------------------------------------------ 3. gradient agreement between sources
print(
    "\n[3] cosine between gradients of 32k-token batches "
    "(rows/cols = sources; diagonal = two independent batches of the same source)"
)
grads = {}
for k, n in enumerate(names):
    g1 = torch.zeros_like(grad_vector(*batch(np.full(micro, k))))
    g2_ = torch.zeros_like(g1)
    for _ in range(accum):
        g1 += grad_vector(*batch(np.full(micro, k))) / accum
    for _ in range(accum):
        g2_ += grad_vector(*batch(np.full(micro, k))) / accum
    grads[n] = (g1, g2_)


def cos(a, b):
    return float(a @ b / (a.norm() * b.norm()))


print(f"{'':18s}" + "".join(f"{n[-7:]:>9s}" for n in names))
for a in names:
    row = ""
    for b in names:
        row += f"{cos(grads[a][0], grads[b][1]):9.3f}"
    print(f"{a:18s}{row}")
