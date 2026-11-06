"""The small layers of a transformer block (Raschka, ch. 4): layer norm, GELU, feed-forward, and
the two "modern" replacements, RMS norm and the SwiGLU feed-forward (D-027).

They are written out, not taken from `torch.nn`, so that each line can be read; the tests compare
every one of them with the library version.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn


class LayerNorm(nn.Module):
    """Normalizes each token's vector to mean 0 and variance 1, then rescales it.

    Why: after many layers the numbers in a token's vector drift in size, which makes training
    unstable. Normalizing every token on its own (over its d_model numbers, not over the batch)
    keeps them in a fixed range, and the learned `weight` (scale) and `bias` (shift) let the
    model undo the normalization wherever that helps. The variance is the biased one
    (divide by d_model, not d_model - 1), as in PyTorch and GPT-2; `eps` avoids dividing by zero.
    The arithmetic is done in float32 even if the input is bfloat16, because a variance in
    bfloat16 has too few digits to be trusted.
    """

    def __init__(self, d_model: int, eps: float = 1e-5, bias: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        out = (x - mean) / torch.sqrt(var + self.eps) * self.weight
        if self.bias is not None:
            out = out + self.bias
        return out.to(dtype)


class RMSNorm(nn.Module):
    """Rescales each token's vector to root-mean-square 1, then multiplies by a learned scale.

    LayerNorm without the mean subtraction and without a shift: out = x / rms(x) * weight, with
    rms(x) = sqrt(mean(x^2) + eps). Zhang and Sennrich (2019) found the re-centering contributes
    little; dropping it saves work and one set of parameters. Used by Llama and most recent models.
    Float32 arithmetic, as in `LayerNorm`.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        out = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight
        return out.to(dtype)


class GELU(nn.Module):
    """x * Phi(x), where Phi is the probability that a standard normal is below x.

    A smooth cousin of ReLU: small negative inputs are not cut to exactly 0, so they still carry
    a gradient. This is the tanh approximation GPT-2 used:
    0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))).
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


class FeedForward(nn.Module):
    """d_model -> d_ff -> d_model, applied to every token separately.

    Attention moves information between tokens; this layer processes what each token has
    gathered. The wide middle (4x d_model in GPT-2) is where much of the model's capacity sits:
    it holds two thirds of the parameters of a block.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, bias: bool = False) -> None:
        super().__init__()
        self.up = nn.Linear(d_model, d_ff, bias=bias)
        self.act = GELU()
        self.down = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(self.act(self.up(x))))


def swiglu_hidden(d_model: int, multiple: int = 64) -> int:
    """Width of the SwiGLU middle layer that keeps the parameter count of a 4x GELU feed-forward.

    A GELU feed-forward has 2 * d_model * 4 d_model weights; SwiGLU has three matrices, so
    2/3 of the width, 8/3 d_model, gives the same total. Rounded to a multiple of `multiple`
    (the usual choice, for efficient matrix shapes); for d_model 384 that is exactly 1024.
    """
    return multiple * round(8 * d_model / 3 / multiple)


class SwiGLUFeedForward(nn.Module):
    """down( silu(gate(x)) * up(x) ): a gated feed-forward (Shazeer 2020, "GLU variants").

    One branch (`gate`, through SiLU = x * sigmoid(x)) decides how much of the other (`up`) passes,
    element by element. Same input and output size as `FeedForward`; the middle width is
    `swiglu_hidden(d_model)` by default so the parameter count stays the same. The output map is
    called `down` as in `FeedForward`, so initialization and tests find it by the same name.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, bias: bool = False) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=bias)
        self.up = nn.Linear(d_model, d_ff, bias=bias)
        self.down = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))
