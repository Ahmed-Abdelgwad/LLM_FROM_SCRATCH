"""Causal multi-head self-attention (Raschka, Build a Large Language Model From Scratch, ch. 3).

What attention does. Every position of the text looks at the positions before it (and itself),
decides how much each one matters for it, and takes a weighted average of what they carry. The
"decides how much" is learned: each position is turned into three vectors by three learned
linear maps,

    query  q = x Wq    "what am I looking for?"
    key    k = x Wk    "what do I contain, as seen from outside?"
    value  v = x Wv    "what do I hand over if I am chosen?"

and the weight of position j for position i is a softmax over the scores q_i . k_j. The output
at i is the weights-weighted sum of the values v_j.

    scores   = q k^T / sqrt(head_dim)         (T x T: how much each i likes each j)
    scores   = scores with j > i set to -inf   (the future is not allowed to be seen)
    weights  = softmax(scores, over j)         (each row sums to 1)
    output   = weights v

Why divide by sqrt(head_dim): a dot product of two vectors with independent unit-variance entries
has variance head_dim, so without the division the scores grow with the size of the head, the
softmax saturates (one weight near 1, the rest near 0) and its gradients vanish.

Why the mask: the model is trained to predict token i+1 from tokens 0..i. If position i could see
i+1 it would just copy it. The mask sets those scores to -inf, so their softmax weight is
exactly 0. `tests/test_attention.py` checks that changing the future never changes the past.

Multi-head: instead of one attention over vectors of size d_model, the vectors are cut into
`n_heads` slices of size head_dim = d_model / n_heads and each slice attends on its own, so
different heads can learn different relations (the previous word, the matching verb...). The
heads' outputs are put back side by side and mixed by one more linear map.

Two implementation choices (D-018): the three maps are one matrix of shape d_model x 3*d_model
(the same computation as three separate matrices, in one matrix product), and the attention is
written out with explicit matrix products. That version is the reference and the one that can
return the weights. With `fast=True`, `forward` uses `F.scaled_dot_product_attention` instead:
the same numbers (the tests compare them) without ever building the T x T tables in memory,
which on the RTX 3050 is about 4 times faster and uses about 3.5 times less memory per layer.

Positions. Attention alone does not know the order of the tokens. By default the model adds that
information to the inputs (a learned position embedding in GPT-2) and nothing happens here. With
`rope=True` the order is put into the queries and keys instead (rotary embeddings, Su et al. 2021,
D-027): see `rotary_cos_sin` and `apply_rotary`.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


def rotary_cos_sin(
    length: int, head_dim: int, theta: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin tables (length, head_dim) for rotary embeddings.

    Each head vector is cut in two halves; dimension i of the first half is paired with dimension
    i of the second, and the pair is rotated in its plane by the angle position * theta^(-2i/d).
    Low pairs turn fast (they tell neighbours apart), high pairs slowly (they track long range).
    """
    if head_dim % 2:
        raise ValueError(f"rotary embeddings need an even head size, got {head_dim}")
    inv_freq = theta ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    angles = torch.arange(length, dtype=torch.float32)[:, None] * inv_freq[None, :]
    angles = torch.cat(
        [angles, angles], dim=-1
    )  # (length, head_dim): the same angle for both halves
    return angles.cos(), angles.sin()


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate x (..., T, head_dim) pair by pair: (a, b) -> (a cos - b sin, b cos + a sin).

    The dot product of a rotated query at position i and a rotated key at position j depends on
    the two vectors and on i - j only, never on i and j separately: that is what makes the position
    relative. The rotation does not change a vector's length.
    """
    half = x.shape[-1] // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos.to(x.dtype) + rotated * sin.to(x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        context_length: int,
        dropout: float = 0.0,
        bias: bool = False,
        fast: bool = False,
        rope: bool = False,
        rope_theta: float = 10000.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.context_length = context_length
        self.fast = fast
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        # True where position j is in the future of position i (j > i): those scores are removed.
        # Not a parameter and not saved in the checkpoint: it is rebuilt from context_length.
        future = torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), 1)
        self.future: torch.Tensor
        self.register_buffer("future", future, persistent=False)
        self.rope = rope
        self.rope_cos: torch.Tensor
        self.rope_sin: torch.Tensor
        cos, sin = (
            rotary_cos_sin(context_length, self.head_dim, rope_theta) if rope else (None, None)
        )
        self.register_buffer("rope_cos", cos, persistent=False)  # tables, rebuilt from the config
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, T, d_model) -> (batch, T, d_model)."""
        if not self.fast:
            return self.forward_with_weights(x)[0]
        batch, length, width = self._check(x)
        q, k, v = self._qkv(x)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        out = out.transpose(1, 2).reshape(batch, length, width)
        return self.resid_dropout(self.out_proj(out))

    def _check(self, x: torch.Tensor) -> tuple[int, int, int]:
        batch, length, width = x.shape
        if length > self.context_length:
            raise ValueError(f"sequence of {length} tokens exceeds context {self.context_length}")
        if width != self.d_model:
            raise ValueError(f"expected last dimension {self.d_model}, got {width}")
        return batch, length, width

    def _qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = (self._heads(t) for t in self.qkv(x).split(self.d_model, dim=2))
        if self.rope:
            length = x.shape[1]
            cos, sin = self.rope_cos[:length], self.rope_sin[:length]
            q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        return q, k, v

    def _heads(self, t: torch.Tensor) -> torch.Tensor:
        """(batch, T, d_model) -> (batch, heads, T, head_dim)."""
        batch, length, _ = t.shape
        return t.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

    def forward_with_weights(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Also returns the attention weights (batch, heads, T, T), to look at or to test."""
        batch, length, width = self._check(x)
        q, k, v = self._qkv(x)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)  # (batch, H, T, T)
        scores = scores.masked_fill(self.future[:length, :length], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = self.attn_dropout(weights) @ v  # (batch, H, T, hd)
        out = out.transpose(1, 2).contiguous().view(batch, length, width)  # heads side by side
        out = self.resid_dropout(self.out_proj(out))
        return out, weights
