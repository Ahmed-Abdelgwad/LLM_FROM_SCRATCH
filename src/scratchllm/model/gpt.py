"""The GPT model (Raschka, ch. 4; GPT-2 architecture).

The path of a batch of token ids (batch, T):

    ids -> token embedding + position embedding        (batch, T, d_model)
        -> n_layers x block                            (batch, T, d_model)
        -> final layer norm -> output map to the vocabulary   (batch, T, vocab_size) = logits

and a block is two residual steps, with the normalization BEFORE each sublayer (GPT-2's order,
"pre-norm", which trains more stably than normalizing after):

    x = x + attention(norm(x))      tokens look at earlier tokens
    x = x + feed_forward(norm(x))   each token processes what it gathered

The residual "x + ..." is a path along which the signal and the gradient pass unchanged, which is
what lets a deep stack be trained at all.

Options (D-027). The description above is the default, GPT-2. Three config switches replace its
parts with the ones recent models use: `norm="rmsnorm"` (RMS norm instead of layer norm),
`position="rope"` (rotary embeddings in the attention instead of a learned vector per position;
the model then has no position table) and `ffn="swiglu"` (a gated feed-forward instead of the
GELU one, at a width that keeps its parameter count). Each is off by default, so a config or a
checkpoint made before they existed builds the model it always was.

Positions. Attention does not know the order of tokens, so a learned vector per position
(0..context_length-1) is added to the token's embedding (GPT-2). `tests/test_gpt.py` shows the
effect in a one-layer model: without it the last token's prediction is the same for any
reordering of what came before. In a deeper model the causal mask alone already leaks some order
(each earlier token sees a different prefix), so the position vectors are the explicit signal, not
the only one.

Weight tying. The output map from d_model to the vocabulary is the token embedding matrix used
backwards (GPT-2 does this): it saves vocab_size x d_model parameters, a large share of a small
model, and gives the input and output sides one shared meaning of each token.

Initialization (GPT-2): weights ~ Normal(0, 0.02), biases 0, and the maps that write into the
residual stream (attention output, feed-forward output) are scaled by 1/sqrt(2 n_layers), so the
variance of the stream does not grow with depth. At initialization the model knows nothing, so
its loss should be close to ln(vocab_size), the loss of guessing uniformly; the tests check it.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from scratchllm.model.attention import CausalSelfAttention
from scratchllm.model.layers import (
    FeedForward,
    LayerNorm,
    RMSNorm,
    SwiGLUFeedForward,
    swiglu_hidden,
)


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int = 16384
    context_length: int = 512
    d_model: int = 384
    n_layers: int = 6
    n_heads: int = 6
    d_ff: int | None = None  # default 4 * d_model
    dropout: float = 0.0
    bias: bool = False
    tie_embeddings: bool = True
    fast_attention: bool = False
    init_std: float = 0.02
    # The "modern" options (D-027). The defaults are GPT-2, so an older checkpoint's config, which
    # lacks these keys, builds the same model it always was.
    norm: Literal["layernorm", "rmsnorm"] = "layernorm"
    position: Literal["learned", "rope"] = "learned"
    ffn: Literal["gelu", "swiglu"] = "gelu"
    rope_theta: float = 10000.0

    @property
    def hidden(self) -> int:
        """Width of the feed-forward's middle layer."""
        if self.d_ff is not None:
            return self.d_ff
        return swiglu_hidden(self.d_model) if self.ffn == "swiglu" else 4 * self.d_model

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_norm(cfg: GPTConfig) -> nn.Module:
    if cfg.norm == "rmsnorm":
        return RMSNorm(cfg.d_model)
    return LayerNorm(cfg.d_model, bias=cfg.bias)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.norm1 = make_norm(cfg)
        self.attn = CausalSelfAttention(
            cfg.d_model, cfg.n_heads, cfg.context_length, cfg.dropout, cfg.bias,
            cfg.fast_attention, rope=cfg.position == "rope", rope_theta=cfg.rope_theta,
        )  # fmt: skip
        self.norm2 = make_norm(cfg)
        ff_class = SwiGLUFeedForward if cfg.ffn == "swiglu" else FeedForward
        self.ff = ff_class(cfg.d_model, cfg.hidden, cfg.dropout, cfg.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ff(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = (
            nn.Embedding(cfg.context_length, cfg.d_model) if cfg.position == "learned" else None
        )  # with rotary embeddings the position lives in the attention, not in the input
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.final_norm = make_norm(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)
        # the maps that add into the residual stream get a smaller start, 1/sqrt(2 n_layers)
        for name, p in self.named_parameters():
            if name.endswith(("attn.out_proj.weight", "ff.down.weight")):
                nn.init.normal_(p, mean=0.0, std=cfg.init_std / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def num_parameters(self, include_embeddings: bool = True) -> int:
        """Parameters, counting tied weights once. Without embeddings: blocks and norms only."""
        total = sum(p.numel() for p in self.parameters())
        if include_embeddings:
            return total
        positions = self.pos_emb.weight.numel() if self.pos_emb is not None else 0
        return total - self.tok_emb.weight.numel() - positions

    def forward(
        self, ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """ids (batch, T) -> logits (batch, T, vocab); with `targets` (batch, T) also the mean
        cross-entropy of predicting each target from its left context."""
        _, length = ids.shape
        if length > self.cfg.context_length:
            raise ValueError(f"{length} tokens exceed the context of {self.cfg.context_length}")
        x = self.tok_emb(ids)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(length, device=ids.device))
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
