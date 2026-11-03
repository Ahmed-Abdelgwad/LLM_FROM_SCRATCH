"""Saving and loading training state so a run can stop and continue exactly.

A checkpoint holds everything that decides what the next step does: model weights, optimizer
state (Adam's moving averages), the step, the random-number generators (dropout), and the run's
bookkeeping. Data order needs no entry because it is a function of (seed, step).

Files are written to a temporary name and renamed, so a crash while saving leaves the previous
checkpoint intact and never a half-written one.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from scratchllm.model.gpt import GPT, GPTConfig

FORMAT_VERSION = 1


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        torch.save({**state, "format_version": FORMAT_VERSION}, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load with `weights_only=True`: a checkpoint can contain tensors and plain values only."""
    state: dict[str, Any] = torch.load(path, map_location=map_location, weights_only=True)
    if state.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported checkpoint format {state.get('format_version')}")
    return state


def capture_rng() -> dict[str, Any]:
    state: dict[str, Any] = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def model_from_state(state: dict[str, Any], device: str | torch.device = "cpu") -> GPT:
    """The model of a loaded checkpoint, in eval mode, built from the configuration inside it."""
    model = GPT(GPTConfig(**state["config"]["model"])).to(device)
    model.load_state_dict(state["model"])
    return model.eval()


def load_model(path: Path, device: str | torch.device = "cpu") -> GPT:
    return model_from_state(load_checkpoint(path, map_location=device), device)


def tokenizer_mismatch(config: dict[str, Any], tokenizer_path: Path) -> str | None:
    """A message if the tokenizer file is not the one the checkpoint's token files were made with.

    Checks the vocabulary size against the model and, when the token files are still where the
    run's configuration says, the tokenizer's sha256 against the one in their descriptions.
    Returns None when everything that can be checked matches.
    """
    import hashlib
    import json

    from scratchllm.tokenizer.tokenizer import Tokenizer

    vocab = Tokenizer.load(tokenizer_path).vocab_size
    if vocab != config["model"]["vocab_size"]:
        return f"the tokenizer has {vocab} ids, the model {config['model']['vocab_size']}"
    metas = sorted(Path(config["data"]["tokens_dir"]).glob("*.tokens.json"))
    if metas:
        wanted = json.loads(metas[0].read_text(encoding="utf-8"))["tokenizer_sha256"]
        if hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() != wanted:
            return "the tokenizer file differs from the one the token files were made with"
    return None
