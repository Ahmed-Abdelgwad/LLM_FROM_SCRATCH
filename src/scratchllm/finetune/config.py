"""Typed, validated configuration of a fine-tuning run (one YAML file per run, like `RunConfig`).

The model is not described here: it is the base checkpoint's, read from that file. What is here
is everything else that changes the result: the data, the method (full fine-tuning or LoRA and
its settings), the optimizer and the loop.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, PositiveInt, model_validator

from scratchllm.finetune.lora import LINEAR_TARGETS, QKV_SLICES
from scratchllm.training.config import OptimSection, _set_dotted, _Strict


class SFTData(_Strict):
    data_dir: Path = Path("data/processed/sft")
    # sampling weight of each language; equal by default (D-025: both languages matter equally)
    language_weights: dict[Literal["ar", "en"], float] = {"ar": 1.0, "en": 1.0}

    @model_validator(mode="after")
    def _positive(self) -> SFTData:
        if not self.language_weights or any(w <= 0 for w in self.language_weights.values()):
            raise ValueError("language_weights must be positive")
        return self


class SFTMethod(_Strict):
    kind: Literal["full", "lora"]
    rank: PositiveInt = 8
    alpha: float = Field(8.0, gt=0.0)
    targets: list[str] = ["attn.q", "attn.v"]
    dropout: float = Field(0.0, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _known_targets(self) -> SFTMethod:
        unknown = [t for t in self.targets if t not in QKV_SLICES and t not in LINEAR_TARGETS]
        if unknown or not self.targets:
            raise ValueError(f"bad LoRA targets {unknown or self.targets}")
        return self


class SFTTrain(_Strict):
    seed: int = 1337
    max_steps: PositiveInt
    micro_batch_size: PositiveInt
    grad_accum_steps: PositiveInt = 1
    eval_interval: PositiveInt
    checkpoint_interval: PositiveInt
    log_interval: PositiveInt = 10
    # windows per pretraining source scored at each evaluation, to watch forgetting as it happens
    forgetting_windows: int = Field(16, ge=0)
    # share of the rows of every step that are pretraining text instead of chat pairs (D-042,
    # `replay.py`); 0 = none, the runs of D-040
    replay_fraction: float = Field(0.0, ge=0.0, lt=1.0)
    precision: Literal["auto", "bf16", "fp32"] = "auto"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    max_consecutive_skips: PositiveInt = 10


class SFTConfig(_Strict):
    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    runs_dir: Path = Path("runs")
    base_checkpoint: Path
    tokenizer: Path = Path("artifacts/tokenizer/bpe-16k.json")
    data: SFTData = SFTData()
    method: SFTMethod
    optim: OptimSection
    train: SFTTrain

    @model_validator(mode="after")
    def _consistent(self) -> SFTConfig:
        if self.optim.warmup_steps > self.train.max_steps:
            raise ValueError("warmup_steps cannot exceed max_steps")
        return self

    @property
    def run_dir(self) -> Path:
        return self.runs_dir / self.name

    def as_dict(self) -> dict[str, Any]:
        return json.loads(self.model_dump_json())

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()


def load_sft_config(path: Path, overrides: list[str] | None = None) -> SFTConfig:
    tree = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override {item!r} must look like section.key=value")
        key, raw = item.split("=", 1)
        _set_dotted(tree, key.strip(), yaml.safe_load(raw))
    return SFTConfig.model_validate(tree)
