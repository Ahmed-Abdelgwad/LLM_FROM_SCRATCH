"""LoRA: fine-tune small low-rank matrices beside frozen weights (Hu et al. 2021, arXiv:2106.09685).

A linear layer computes h = W0 x, with W0 of shape (out, in). LoRA freezes W0 and learns a change
of rank r, written as the product of two thin matrices (the paper's equation 3, with its scaling):

    h = W0 x + (alpha / r) * B A x          A: (r, in), B: (out, r), r much smaller than in, out

Only A and B are trained. For one 384 x 384 projection of this model at r = 8 that is
8 * (384 + 384) = 6,144 numbers instead of 147,456.

Choices, and where each comes from:

* B starts at zero, so B A = 0 and the adapted model is exactly the base model at step 0
  (paper, section 4.1). A must then be non-zero, or no gradient would ever reach B: the gradient
  of B is computed from A x. The paper says "random Gaussian"; Microsoft's reference code
  (`loralib`) uses Kaiming-uniform, the default of `nn.Linear`, and so does this file.
* alpha / r scales the change. The paper sets alpha to the first r tried and does not tune it,
  because with Adam alpha acts like a learning rate; the scale makes a change of r need no new
  learning rate.
* The attention of this model computes queries, keys and values with ONE fused map `attn.qkv`
  (out = 3 * d_model). The paper adapts W_q and W_v separately (its table 5 finds that pair the
  best use of a fixed budget). So a target can name a slice of a fused layer: "attn.q", "attn.k"
  and "attn.v" give that third of `attn.qkv` its own A and B, like `loralib.MergedLinear` with
  `enable_lora=[True, False, True]`. The slices leave the other thirds exactly unchanged. With
  grouped-query attention the key and value parts are narrower than the query part (`kv_dim`
  outputs each, not d_model), and the slices follow the attention's own sizes.
* After training, B A is added into W0 once (`merge_lora`, the paper's "no additional inference
  latency"): the merged model is a plain `GPT` with the base model's parameter names, and the
  ordinary checkpoint, evaluation and generation code reads it.
* Everything not adapted stays frozen, including the embedding matrix, which is also the output
  map (weight tying, `model/gpt.py`), the norms and the biases. The paper freezes the same.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

# what a target name means: the module it lives in and which third of `attn.qkv`, if any
QKV_SLICES = {"attn.q": 0, "attn.k": 1, "attn.v": 2}
LINEAR_TARGETS = ("attn.out_proj", "ff.up", "ff.gate", "ff.down")
PAPER_TARGETS = ("attn.q", "attn.v")


class LoRALinear(nn.Module):
    """A frozen `nn.Linear` plus one trainable low-rank change per adapted output slice.

    `slices` lists the (start, stop) output ranges that get their own A and B; None adapts the
    whole output. Rows outside every slice are exactly the base layer's.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        slices: Sequence[tuple[int, int]] | None = None,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be at least 1")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank, self.alpha = rank, alpha
        self.scale = alpha / rank
        self.slices = [(0, base.out_features)] if slices is None else sorted(slices)
        for (start, stop), nxt in zip(self.slices, [*self.slices[1:], None], strict=True):
            if not 0 <= start < stop <= base.out_features or (nxt is not None and nxt[0] < stop):
                raise ValueError(f"bad or overlapping output slice {(start, stop)}")
        device, dtype = base.weight.device, base.weight.dtype
        self.lora_a = nn.ParameterList()
        self.lora_b = nn.ParameterList()
        for start, stop in self.slices:
            a = nn.Parameter(torch.empty(rank, base.in_features, device=device, dtype=dtype))
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            self.lora_a.append(a)
            self.lora_b.append(
                nn.Parameter(torch.zeros(stop - start, rank, device=device, dtype=dtype))
            )
        self.drop = nn.Dropout(dropout)

    def delta_weight(self) -> torch.Tensor:
        """(alpha / r) B A for every slice, laid out like the base weight (zero elsewhere)."""
        delta = torch.zeros_like(self.base.weight)
        for (start, stop), a, b in zip(self.slices, self.lora_a, self.lora_b, strict=True):
            delta[start:stop] = self.scale * (b @ a)
        return delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        dropped = self.drop(x)
        parts = []
        position = 0
        for (start, stop), a, b in zip(self.slices, self.lora_a, self.lora_b, strict=True):
            if start > position:
                parts.append(out.new_zeros(*out.shape[:-1], start - position))
            parts.append(self.scale * (dropped @ a.T @ b.T))
            position = stop
        if position < out.shape[-1]:
            parts.append(out.new_zeros(*out.shape[:-1], out.shape[-1] - position))
        return out + torch.cat(parts, dim=-1)


def _qkv_bounds(model: nn.Module, name: str, module: nn.Linear) -> list[tuple[int, int]]:
    """Output ranges of the queries, keys and values in a fused `attn.qkv` map."""
    attention = model.get_submodule(name.rpartition(".")[0])
    q_dim = getattr(attention, "d_model", module.out_features // 3)
    kv_dim = getattr(attention, "kv_dim", q_dim)
    if q_dim + 2 * kv_dim != module.out_features:
        raise ValueError(f"{name}: {module.out_features} outputs, expected {q_dim + 2 * kv_dim}")
    return [(0, q_dim), (q_dim, q_dim + kv_dim), (q_dim + kv_dim, q_dim + 2 * kv_dim)]


def _resolve(model: nn.Module, targets: Sequence[str]) -> dict[str, list[tuple[int, int]] | None]:
    """Module name -> slices to adapt (None = all of it), for every layer the targets name."""
    unknown = [t for t in targets if t not in QKV_SLICES and t not in LINEAR_TARGETS]
    if unknown:
        raise ValueError(
            f"unknown LoRA targets {unknown}; choose from {[*QKV_SLICES, *LINEAR_TARGETS]}"
        )
    plan: dict[str, list[tuple[int, int]] | None] = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.endswith("attn.qkv"):
            wanted = [QKV_SLICES[t] for t in targets if t in QKV_SLICES]
            if wanted:
                bounds = _qkv_bounds(model, name, module)
                plan[name] = [bounds[i] for i in sorted(set(wanted))]
        elif name.endswith(tuple(t for t in targets if t in LINEAR_TARGETS)):
            plan[name] = None
    for target in targets:
        module_name = "attn.qkv" if target in QKV_SLICES else target
        if not any(name.endswith(module_name) for name in plan):
            raise ValueError(f"the model has no layer for LoRA target {target!r}")
    return plan


def add_lora(
    model: nn.Module,
    rank: int,
    alpha: float,
    targets: Sequence[str] = PAPER_TARGETS,
    dropout: float = 0.0,
) -> list[str]:
    """Freeze every parameter of `model`, then wrap the layers named by `targets` in
    `LoRALinear`. Returns the wrapped module names."""
    plan = _resolve(model, targets)
    for p in model.parameters():
        p.requires_grad_(False)
    for name, slices in plan.items():
        parent_name, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        base = getattr(parent, attribute)
        setattr(parent, attribute, LoRALinear(base, rank, alpha, dropout, slices))
    return sorted(plan)


@torch.no_grad()
def merge_lora(model: nn.Module) -> int:
    """Replace every `LoRALinear` by a plain `nn.Linear` holding W0 + its change. Returns how
    many were merged. Afterwards the model has the base model's parameter names again."""
    merged = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, LoRALinear):
            continue
        base = module.base
        plain = nn.Linear(
            base.in_features, base.out_features, bias=base.bias is not None,
            device=base.weight.device, dtype=base.weight.dtype,
        )  # fmt: skip
        plain.weight.copy_(base.weight + module.delta_weight())
        if base.bias is not None:
            plain.bias.copy_(base.bias)
        parent_name, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attribute, plain)
        merged += 1
    return merged


def lora_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Only the adapter matrices: what a LoRA checkpoint needs beyond the base model."""
    return {k: v for k, v in model.state_dict().items() if ".lora_a." in k or ".lora_b." in k}


def trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def lora_parameter_count(
    d_model: int,
    d_ff: int,
    n_layers: int,
    rank: int,
    targets: Sequence[str],
    kv_dim: int | None = None,
) -> int:
    """r * (in + out) per adapted map, summed: the formula the tests check the model against.
    `kv_dim` is the width of the keys and values under grouped-query attention (default d_model)."""
    kv = d_model if kv_dim is None else kv_dim
    shape = {
        "attn.q": (d_model, d_model), "attn.k": (d_model, kv), "attn.v": (d_model, kv),
        "attn.out_proj": (d_model, d_model), "ff.up": (d_model, d_ff), "ff.gate": (d_model, d_ff),
        "ff.down": (d_ff, d_model),
    }  # fmt: skip
    return n_layers * sum(rank * sum(shape[t]) for t in targets)
