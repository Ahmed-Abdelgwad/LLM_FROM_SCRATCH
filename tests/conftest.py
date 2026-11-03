"""Shared fixtures for the training tests: tiny token files and a config factory."""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scratchllm.training.config import RunConfig

VOCAB = 64
PERIOD = 8


def write_source(
    directory: Path,
    name: str,
    base: int,
    train_tokens: int = 4000,
    val_tokens: int = 700,
    vocab: int = VOCAB,
) -> None:
    """A learnable stream (the ids base..base+7 in a repeating order) in the files `data tokenize`
    would write, with a matching description."""
    splits: dict[str, Any] = {}
    for split, n in (("train", train_tokens), ("val", val_tokens), ("test", 100)):
        ids = (base + np.arange(n) % PERIOD).astype("<u2")
        file = f"{name}.{split}.bin"
        ids.tofile(directory / file)
        splits[split] = {
            "file": file,
            "docs": n // 50,
            "chars": n * 3,
            "tokens": n,
            "sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
        }
    meta = {
        "source": name,
        "role": "training",
        "tokenizer_sha256": "t" * 64,
        "vocab_size": vocab,
        "end_of_text_id": vocab - 1,
        "dtype": "uint16",
        "splits": splits,
    }
    (directory / f"{name}.tokens.json").write_text(json.dumps(meta), encoding="utf-8")


@pytest.fixture()
def tokens_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "tokens"
    directory.mkdir()
    write_source(directory, "alpha", base=0)  # ids 0..7
    write_source(directory, "beta", base=32)  # ids 32..39
    return directory


@pytest.fixture()
def make_cfg(tmp_path: Path, tokens_dir: Path) -> Callable[..., RunConfig]:
    def factory(name: str = "run", **sections: dict[str, Any]) -> RunConfig:
        tree: dict[str, Any] = {
            "name": name,
            "runs_dir": str(tmp_path / "runs"),
            "model": {
                "vocab_size": VOCAB,
                "context_length": 16,
                "d_model": 32,
                "n_layers": 2,
                "n_heads": 4,
            },
            "data": {"tokens_dir": str(tokens_dir)},
            "optim": {"lr": 0.01, "min_lr": 0.001, "warmup_steps": 2, "grad_clip": 1.0},
            "train": {
                "seed": 7,
                "max_steps": 12,
                "micro_batch_size": 4,
                "grad_accum_steps": 2,
                "eval_interval": 4,
                "eval_batches": 2,
                "checkpoint_interval": 4,
                "log_interval": 1,
                "device": "cpu",
            },
        }
        for section, values in sections.items():
            tree[section] = (
                {**tree.get(section, {}), **values}
                if isinstance(tree.get(section), dict)
                else values
            )
        return RunConfig.model_validate(tree)

    return factory
