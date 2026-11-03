"""The training loop (Raschka, ch. 5), with what a run needs to be trusted and to survive a stop.

One optimizer step is:

    for each of grad_accum_steps micro-batches:   forward, loss / grad_accum_steps, backward
    clip the global gradient norm                 (a rare huge gradient cannot wreck the weights)
    set the learning rate from the schedule, then AdamW step

Gradient accumulation adds the gradients of several small batches before stepping, which gives the
effect of a large batch on a GPU that can only hold a small one (the output scores at 16,384 ids
are what limit the batch size on the RTX 3050, D-019).

Reliability:
* A step whose loss or gradient norm is not finite is skipped (no weight change) and logged;
  `max_consecutive_skips` in a row aborts the run with `DivergedError`.
* `last.pt` is written atomically every `checkpoint_interval` steps and at the end; `best.pt`
  whenever the validation loss improves. `resume` continues from `last.pt`.
* Resuming reproduces the uninterrupted run exactly on CPU (tests compare the weights bit for
  bit): data order depends only on (seed, step), the optimizer state and the random generators
  are in the checkpoint. On GPU, kernels are not bitwise deterministic, so results agree to
  rounding unless `deterministic` is set (slower, and some operations are refused).
* Everything is logged as JSON lines in `<run_dir>/metrics.jsonl`.
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scratchllm.model.gpt import GPT
from scratchllm.training.batches import (
    SourceTokens,
    TrainSampler,
    bits_per_char,
    evaluate,
    load_source,
    training_source_names,
    validation_sets,
)
from scratchllm.training.checkpoint import (
    capture_rng,
    load_checkpoint,
    restore_rng,
    save_checkpoint,
)
from scratchllm.training.config import OptimSection, RunConfig
from scratchllm.training.logger import JsonlLogger
from scratchllm.training.schedule import lr_at

Progress = Callable[[str], None]


class TrainingError(RuntimeError):
    pass


class DivergedError(TrainingError):
    pass


@dataclass
class TrainResult:
    run_dir: Path
    final_step: int
    last_train_loss: float | None
    best_val_loss: float | None
    tokens_seen: int
    skipped_steps: int


def _ignore(_: str) -> None:
    pass


# ----------------------------------------------------------------------------- setup


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise TrainingError("device 'cuda' requested but CUDA is not available")
    return torch.device(choice)


@contextmanager
def stop_on_termination() -> Iterator[None]:
    """Turn SIGTERM and SIGHUP (kill, closed terminal) into KeyboardInterrupt while training.

    The loop already saves `last.pt` on KeyboardInterrupt, so a run that is asked to stop leaves a
    checkpoint at its last finished step instead of losing everything since the last interval.
    Only possible in the main thread; elsewhere it does nothing. SIGKILL cannot be caught.
    """
    names = [n for n in ("SIGTERM", "SIGHUP") if hasattr(signal, n)]
    previous: dict[int, Any] = {}

    def raise_interrupt(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt(f"signal {signum}")

    try:
        for name in names:
            number = getattr(signal, name)
            previous[number] = signal.signal(number, raise_interrupt)
    except ValueError:  # not the main thread
        previous.clear()
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def autocast_factory(
    device: torch.device, precision: str
) -> tuple[Callable[[], AbstractContextManager[Any]], str]:
    """bfloat16 autocast on GPU by default, plain float32 elsewhere."""
    want_bf16 = precision == "bf16" or (
        precision == "auto" and device.type == "cuda" and torch.cuda.is_bf16_supported()
    )
    if want_bf16:
        return (lambda: torch.autocast(device.type, dtype=torch.bfloat16)), "bf16"
    return (lambda: nullcontext()), "fp32"


def make_optimizer(model: torch.nn.Module, cfg: OptimSection, device: torch.device) -> Any:
    """AdamW; weight decay on matrices and embeddings only, not on norm scales or biases.

    Decay pulls weights toward zero, which regularizes the big matrices; applying it to a norm's
    scale (initialized at 1) or to a bias would only fight what they are for.
    """
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        fused=device.type == "cuda",
    )


def _git_state() -> dict[str, Any]:
    try:
        here = Path(__file__).resolve().parent
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, timeout=5
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=here, capture_output=True, text=True, timeout=5
        )
        if commit.returncode != 0:
            return {"commit": None}
        return {"commit": commit.stdout.strip(), "dirty": bool(status.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None}


def _environment(device: torch.device) -> dict[str, Any]:
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "device": str(device),
    }
    if device.type == "cuda":
        env["gpu"] = torch.cuda.get_device_name(device)
    return env


def prepare_data(
    cfg: RunConfig,
) -> tuple[list[SourceTokens], list[float], list[SourceTokens], dict[str, dict[str, str]]]:
    """Resolve the sources, check they fit the model, and fingerprint the files used."""
    tokens_dir = cfg.data.tokens_dir
    if cfg.data.train_sources is not None:
        names = list(cfg.data.train_sources)
    else:
        names = training_source_names(tokens_dir)
    if not names:
        raise TrainingError(f"no training source found in {tokens_dir}")
    train = [load_source(tokens_dir, n) for n in names]
    if cfg.data.train_sources is not None:
        weights = [cfg.data.train_sources[s.name] for s in train]
    else:
        weights = [float(s.train_tokens) for s in train]
    val_names = cfg.data.val_sources if cfg.data.val_sources is not None else names
    by_name = {s.name: s for s in train}
    val = [by_name.get(n) or load_source(tokens_dir, n) for n in val_names]
    for s in {*train, *val}:
        if s.vocab_size != cfg.model.vocab_size:
            raise TrainingError(
                f"{s.name}: token files use a vocabulary of {s.vocab_size} ids, "
                f"the model has {cfg.model.vocab_size}"
            )
    if len({s.tokenizer_sha256 for s in {*train, *val}}) != 1:
        raise TrainingError("the sources were tokenized with different tokenizers")
    fingerprint = {
        s.name: {"train": s.train_sha256, "val": s.val_sha256}
        for s in sorted({*train, *val}, key=lambda s: s.name)
    }
    return train, weights, val, fingerprint


# ----------------------------------------------------------------------------- the run


def train(
    cfg: RunConfig,
    *,
    resume: str | None = None,
    progress: Progress = _ignore,
) -> TrainResult:
    """Train `cfg` to `max_steps`. `resume` is a checkpoint path, or "auto" for run_dir/last.pt."""
    t = cfg.train
    device = resolve_device(t.device)
    autocast, precision = autocast_factory(device, t.precision)
    run_dir = cfg.run_dir

    # --- decide whether this is a fresh run or a continuation, before touching anything
    checkpoint_path: Path | None = None
    if resume == "auto":
        candidate = run_dir / "last.pt"
        checkpoint_path = candidate if candidate.exists() else None
    elif resume:
        checkpoint_path = Path(resume)
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
    if checkpoint_path is None and (run_dir / "metrics.jsonl").exists():
        raise TrainingError(
            f"{run_dir} already holds a run; continue it with resume='auto' or choose another name"
        )

    seed_everything(t.seed, t.deterministic)
    train_sources, weights, val_sources, fingerprint = prepare_data(cfg)
    sampling = {
        "seed": t.seed,
        "micro_batch_size": t.micro_batch_size,
        "grad_accum_steps": t.grad_accum_steps,
        "weights": {s.name: w for s, w in zip(train_sources, weights, strict=True)},
    }

    model = GPT(cfg.model.to_gpt()).to(device)
    optimizer = make_optimizer(model, cfg.optim, device)
    sampler = TrainSampler(
        train_sources, weights, cfg.model.context_length, t.micro_batch_size, t.seed
    )
    val_sets = validation_sets(val_sources, cfg.model.context_length)
    probe_sets = (
        validation_sets(train_sources, cfg.model.context_length, split="train")
        if t.train_probe
        else {}
    )
    val_meta = {s.name: s for s in val_sources}
    train_meta = {s.name: s for s in train_sources}

    step = 0
    best_val: float | None = None
    tokens_seen = 0
    skipped = 0
    changed: list[str] = []
    if checkpoint_path is not None:
        state = load_checkpoint(checkpoint_path, map_location=device)
        saved = state["config"]
        if saved["model"] != cfg.as_dict()["model"]:
            raise TrainingError("the checkpoint was made with a different model configuration")
        if state["data_fingerprint"] != fingerprint:
            raise TrainingError("the token files differ from those the checkpoint was made with")
        if state["sampling"] != sampling:
            raise TrainingError(
                "seed, batch sizes or source weights differ from the checkpoint; the data "
                "stream would not continue where it stopped"
            )
        changed = [k for k in ("optim", "train", "data") if saved[k] != cfg.as_dict()[k]]
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
        step, best_val = state["step"], state["best_val"]
        tokens_seen, skipped = state["tokens_seen"], state["skipped_steps"]

    logger = JsonlLogger(run_dir / "metrics.jsonl")
    (run_dir / "config.json").write_text(
        json.dumps(cfg.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.log(
        "resume" if checkpoint_path is not None else "start",
        step=step,
        config=cfg.as_dict(),
        config_sha256=cfg.fingerprint(),
        config_sections_changed=changed,
        parameters=model.num_parameters(),
        tokens_per_step=cfg.tokens_per_step,
        precision=precision,
        git=_git_state(),
        environment=_environment(device),
        data=fingerprint,
        sampling=sampling,
    )

    def snapshot(at_step: int) -> dict[str, Any]:
        return {
            "step": at_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": capture_rng(),
            "best_val": best_val,
            "tokens_seen": tokens_seen,
            "skipped_steps": skipped,
            "config": cfg.as_dict(),
            "config_sha256": cfg.fingerprint(),
            "data_fingerprint": fingerprint,
            "sampling": sampling,
        }

    params = [p for p in model.parameters() if p.requires_grad]
    max_norm = cfg.optim.grad_clip if cfg.optim.grad_clip > 0 else math.inf
    completed = step
    consecutive_skips = 0
    window_loss = 0.0
    window_steps = 0
    window_time = 0.0
    window_tokens = 0
    last_loss: float | None = None
    model.train()

    guard = ExitStack()
    guard.enter_context(stop_on_termination())
    try:
        for step_index in range(step, t.max_steps):
            step_started = time.perf_counter()
            lr = lr_at(
                step_index, cfg.optim.lr, cfg.optim.min_lr, cfg.optim.warmup_steps, t.max_steps
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            loss_sum = torch.zeros((), device=device)
            for micro in range(t.grad_accum_steps):
                x, y = sampler.batch(step_index, micro)
                with autocast():
                    _, loss = model(x.to(device), y.to(device))
                assert loss is not None
                (loss / t.grad_accum_steps).backward()
                loss_sum += loss.detach().float()
            mean_loss = float(loss_sum) / t.grad_accum_steps
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, max_norm))

            completed = step_index + 1
            if math.isfinite(mean_loss) and math.isfinite(grad_norm):
                optimizer.step()
                consecutive_skips = 0
                tokens_seen += cfg.tokens_per_step
                window_tokens += cfg.tokens_per_step
                window_loss += mean_loss
                window_steps += 1
                last_loss = mean_loss
            else:  # no weight change; the step still counts, so the schedule and data move on
                skipped += 1
                consecutive_skips += 1
                logger.log(
                    "skip", step=completed, loss=mean_loss, grad_norm=grad_norm, lr=lr,
                    consecutive=consecutive_skips,
                )  # fmt: skip
            optimizer.zero_grad(set_to_none=True)
            window_time += time.perf_counter() - step_started  # evaluation and saving excluded
            if consecutive_skips >= t.max_consecutive_skips:
                save_checkpoint(run_dir / "last.pt", snapshot(completed))
                logger.log("diverged", step=completed, consecutive_skips=consecutive_skips)
                raise DivergedError(
                    f"{consecutive_skips} consecutive non-finite steps at step {completed}"
                )

            if window_steps and (completed % t.log_interval == 0 or completed == t.max_steps):
                fields: dict[str, Any] = {
                    "step": completed,
                    "loss": window_loss / window_steps,
                    "lr": lr,
                    "grad_norm": grad_norm,
                    "tokens_seen": tokens_seen,
                    "tokens_per_s": window_tokens / window_time if window_time > 0 else None,
                    "skipped_steps": skipped,
                }
                if device.type == "cuda":
                    fields["peak_memory_mib"] = torch.cuda.max_memory_allocated() / 2**20
                logger.log("train", **fields)
                progress(
                    f"step {completed}/{t.max_steps}  loss {fields['loss']:.4f}  "
                    f"lr {lr:.2e}  grad {grad_norm:.2f}  "
                    f"{fields['tokens_per_s'] or 0:,.0f} tok/s"
                )
                window_loss, window_steps, window_tokens, window_time = 0.0, 0, 0, 0.0

            if completed % t.eval_interval == 0 or completed == t.max_steps:
                losses = evaluate(
                    model, val_sets, t.micro_batch_size, t.eval_batches * t.micro_batch_size,
                    device, autocast,
                )  # fmt: skip
                probe = (
                    evaluate(
                        model,
                        probe_sets,
                        t.micro_batch_size,
                        t.eval_batches * t.micro_batch_size,
                        device,
                        autocast,
                    )  # fmt: skip
                    if probe_sets
                    else {}
                )
                per_source = {}
                for n, v in losses.items():
                    entry = {
                        "loss": v,
                        "bits_per_char": bits_per_char(
                            v, val_meta[n].val_tokens, val_meta[n].val_chars
                        ),
                    }
                    if n in probe:
                        entry["train_loss"] = probe[n]
                        entry["train_bits_per_char"] = bits_per_char(
                            probe[n], train_meta[n].train_tokens, train_meta[n].train_chars
                        )
                    per_source[n] = entry
                macro = sum(losses.values()) / len(losses)
                macro_bpc = sum(s["bits_per_char"] for s in per_source.values()) / len(per_source)
                improved = math.isfinite(macro) and (best_val is None or macro < best_val)
                if improved:
                    best_val = macro
                logger.log(
                    "eval", step=completed, val_loss=macro, val_bits_per_char=macro_bpc,
                    best=improved, sources=per_source,
                )  # fmt: skip
                progress(
                    f"  eval step {completed}: val loss {macro:.4f}  "
                    f"{macro_bpc:.3f} bits/char{'  (best)' if improved else ''}"
                )
                if improved:
                    save_checkpoint(run_dir / "best.pt", snapshot(completed))

            if completed % t.checkpoint_interval == 0 or completed == t.max_steps:
                save_checkpoint(run_dir / "last.pt", snapshot(completed))
                logger.log("checkpoint", step=completed, path="last.pt")
        logger.log(
            "end", step=completed, best_val_loss=best_val, tokens_seen=tokens_seen,
            skipped_steps=skipped,
        )  # fmt: skip
    except KeyboardInterrupt:
        save_checkpoint(run_dir / "last.pt", snapshot(completed))
        logger.log("interrupted", step=completed)
        raise
    finally:
        logger.close()
        guard.close()

    return TrainResult(run_dir, completed, last_loss, best_val, tokens_seen, skipped)
