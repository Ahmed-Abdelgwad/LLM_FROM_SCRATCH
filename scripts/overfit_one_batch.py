"""Implementation check used for D-023: the exact 17.1M-parameter model must be able to memorize
four short windows (CPU, 120 steps). Run from the repository root:
uv run python scripts/overfit_one_batch.py
Output: reports/diagnostics/final-d023/overfit-one-batch.txt."""

import math
from pathlib import Path

import torch

from scratchllm.model.gpt import GPT, GPTConfig
from scratchllm.training.dataset import TokenWindows

torch.manual_seed(0)
torch.set_num_threads(3)
cfg = GPTConfig(
    vocab_size=16384, context_length=512, d_model=384, n_layers=6, n_heads=6, fast_attention=True
)
m = GPT(cfg)
ds = TokenWindows([Path("data/processed/tokens-16k/fw2_arb.train.bin")], 512, stride=1)
pairs = [ds[i * 100_003] for i in range(4)]
x = torch.stack([a[:128] for a, _ in pairs])
y = torch.stack([b[:128] for _, b in pairs])
opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.0, betas=(0.9, 0.95))
print("init loss", float(m(x, y)[1]), "ln(V) =", math.log(16384))
for i in range(1, 121):
    opt.zero_grad()
    loss = m(x, y)[1]
    loss.backward()
    opt.step()
    if i in (1, 10, 30, 60, 90, 120):
        print(i, round(float(loss), 4))
