"""Typed, validated run configuration, loaded from one YAML file per run.

Everything that can change the result of a run is in here, so the file (plus the token files it
points to) is the whole recipe: its hash is stored in every checkpoint and a resumed run is
checked against it. Unknown keys are errors, not ignored, so a typo cannot silently fall back to
a default.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, model_validator

from scratchllm.model.gpt import GPTConfig


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelSection(_Strict):
    vocab_size: PositiveInt = 16384
    context_length: PositiveInt = 512
    d_model: PositiveInt = 384
    n_layers: PositiveInt = 6
    n_heads: PositiveInt = 6
    d_ff: PositiveInt | None = None
    dropout: float = Field(0.0, ge=0.0, lt=1.0)
    bias: bool = False
    tie_embeddings: bool = True
    fast_attention: bool = False
    init_std: float = Field(0.02, gt=0.0)
    norm: Literal["layernorm", "rmsnorm"] = "layernorm"
    position: Literal["learned", "rope"] = "learned"
    ffn: Literal["gelu", "swiglu"] = "gelu"
    rope_theta: float = Field(10000.0, gt=0.0)

    def to_gpt(self) -> GPTConfig:
        return GPTConfig(**self.model_dump())


class DataSection(_Strict):
    tokens_dir: Path
    # source -> sampling weight; None = every training source in tokens_dir, weighted by its
    # number of train tokens (the natural mix of D-017)
    train_sources: dict[str, float] | None = None
    # sources evaluated on their val split; None = the training sources
    val_sources: list[str] | None = None

    @model_validator(mode="after")
    def _positive_weights(self) -> DataSection:
        if self.train_sources is not None:
            if not self.train_sources:
                raise ValueError("train_sources cannot be empty")
            if any(w <= 0 for w in self.train_sources.values()):
                raise ValueError("train_sources weights must be positive")
        return self


class OptimSection(_Strict):
    lr: float = Field(gt=0.0)
    min_lr: float = Field(ge=0.0)
    warmup_steps: int = Field(ge=0)
    weight_decay: float = Field(0.1, ge=0.0)
    beta1: float = Field(0.9, gt=0.0, lt=1.0)
    beta2: float = Field(0.95, gt=0.0, lt=1.0)
    eps: float = Field(1e-8, gt=0.0)
    grad_clip: float = Field(1.0, ge=0.0)  # max global gradient norm; 0 turns clipping off

    @model_validator(mode="after")
    def _min_below_peak(self) -> OptimSection:
        if self.min_lr > self.lr:
            raise ValueError("min_lr cannot exceed lr")
        return self


class TrainSection(_Strict):
    seed: int = 1337
    max_steps: PositiveInt
    micro_batch_size: PositiveInt
    grad_accum_steps: PositiveInt = 1
    eval_interval: PositiveInt
    eval_batches: PositiveInt = 20  # windows per source = eval_batches * micro_batch_size
    # also score fixed windows of each training source's TRAIN tokens at every evaluation, so the
    # train and val curves of a source can be compared (costs one more evaluation pass)
    train_probe: bool = True
    checkpoint_interval: PositiveInt
    log_interval: PositiveInt = 10
    precision: Literal["auto", "bf16", "fp32"] = "auto"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    deterministic: bool = False
    max_consecutive_skips: PositiveInt = 10


class RunConfig(_Strict):
    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    runs_dir: Path = Path("runs")
    model: ModelSection
    data: DataSection
    optim: OptimSection
    train: TrainSection

    @model_validator(mode="after")
    def _consistent(self) -> RunConfig:
        if self.optim.warmup_steps > self.train.max_steps:
            raise ValueError("warmup_steps cannot exceed max_steps")
        if self.model.d_model % self.model.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        return self

    @property
    def run_dir(self) -> Path:
        return self.runs_dir / self.name

    @property
    def tokens_per_step(self) -> int:
        t = self.train
        return t.micro_batch_size * t.grad_accum_steps * self.model.context_length

    def as_dict(self) -> dict[str, Any]:
        return json.loads(self.model_dump_json())

    def fingerprint(self) -> str:
        blob = json.dumps(self.as_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()


def _set_dotted(tree: dict[str, Any], dotted: str, value: Any) -> None:
    *parents, leaf = dotted.split(".")
    node = tree
    for key in parents:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ValueError(f"cannot set {dotted}: {key} is not a section")
    node[leaf] = value


def load_config(path: Path, overrides: list[str] | None = None) -> RunConfig:
    """Read a YAML file; `overrides` are `section.key=value` strings (value parsed as YAML)."""
    tree = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override {item!r} must look like section.key=value")
        key, raw = item.split("=", 1)
        _set_dotted(tree, key.strip(), yaml.safe_load(raw))
    return RunConfig.model_validate(tree)
