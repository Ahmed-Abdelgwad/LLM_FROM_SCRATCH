"""Supervised fine-tuning of a pretrained checkpoint, fully or with LoRA (D-039).

One optimizer step:

    take every micro-batch of the step and count N, the response tokens they hold
    for each micro-batch: summed cross-entropy over its response tokens, divided by N, backward
    clip, set the learning rate (warmup then cosine), AdamW step on the trainable parameters

The loss is L = -(1/N) * sum over response tokens of log p(token | everything before it)
(Shi et al. 2024, eq. 2): the prompt is read but never scored. Dividing by N for the whole step,
not by each micro-batch's own count, weights every response token equally; averaging per
micro-batch would give the tokens of a batch of short answers more weight than those of a batch
of long ones.

Evaluation, at every `eval_interval`:

* the response loss on the validation pairs, per variety (Modern Standard, Egyptian, English), in
  nats per token and in bits per character of the response (the end-of-text it must also predict
  counts as one character);
* forgetting: the loss on `forgetting_windows` fixed windows of each pretraining source's
  validation split, the same kind of windows pretraining was scored on;
* `best.pt` is kept by the selection score: the mean of the Arabic mean and English of the
  response bits per character (D-025: the two languages weigh the same).

Kept from the pretraining loop (`training/trainer.py`): the optimizer, schedule, autocast,
checkpoint files, JSON-lines log and the stop-at-a-step-boundary handling. With LoRA a checkpoint
holds only the adapter matrices; the base is read again from `base_checkpoint`, whose sha256 is
checked on resume. At the end the model is merged and written as `merged.pt`, an ordinary
checkpoint that `scratchllm eval` and `scratchllm generate` read.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

from scratchllm.data.tokens import END_OF_TEXT
from scratchllm.finetune.chat import IGNORE, EncodedExample, encode_example
from scratchllm.finetune.config import SFTConfig
from scratchllm.finetune.data import Pair, PairSampler, read_split
from scratchllm.finetune.lora import add_lora, lora_state, merge_lora, trainable_parameters
from scratchllm.finetune.replay import ReplaySampler
from scratchllm.model.gpt import GPT
from scratchllm.tokenizer.normalize import normalize
from scratchllm.tokenizer.tokenizer import Tokenizer
from scratchllm.training.batches import (
    bits_per_char,
    evaluate,
    load_source,
    training_source_names,
    validation_sets,
)
from scratchllm.training.checkpoint import (
    capture_rng,
    load_checkpoint,
    model_from_state,
    restore_rng,
    save_checkpoint,
    tokenizer_mismatch,
)
from scratchllm.training.logger import JsonlLogger
from scratchllm.training.schedule import lr_at
from scratchllm.training.trainer import (
    DivergedError,
    TrainingError,
    _environment,
    _git_state,
    autocast_factory,
    make_optimizer,
    resolve_device,
    seed_everything,
    stop_on_termination,
)

Progress = Callable[[str], None]


@dataclass(frozen=True)
class Encoded:
    pair: Pair
    example: EncodedExample
    chars: int  # characters of the normalized response, plus one for the end-of-text


@dataclass
class SFTResult:
    run_dir: Path
    final_step: int
    best_score: float | None
    merged: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def encode_pairs(
    pairs: Sequence[Pair], tokenizer: Tokenizer, context_length: int
) -> tuple[list[Encoded], dict[str, int]]:
    """Encode every pair; returns the ones that fit and, per variety, how many did not."""
    kept: list[Encoded] = []
    dropped: dict[str, int] = defaultdict(int)
    for p in pairs:
        ex = encode_example(tokenizer, p.prompt, p.response, context_length)
        if ex is None:
            dropped[p.variety] += 1
        else:
            kept.append(Encoded(p, ex, len(normalize(p.response)) + 1))
    return kept, dict(dropped)


def collate(examples: Sequence[EncodedExample], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad to the longest example: inputs with `pad_id` (never a target), targets with IGNORE."""
    width = max(len(e.inputs) for e in examples)
    x = torch.full((len(examples), width), pad_id, dtype=torch.long)
    y = torch.full((len(examples), width), IGNORE, dtype=torch.long)
    for row, e in enumerate(examples):
        x[row, : len(e.inputs)] = torch.tensor(e.inputs)
        y[row, : len(e.targets)] = torch.tensor(e.targets)
    return x, y


def summed_loss(model: GPT, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Sum of the cross-entropy over the target positions (IGNORE positions skipped)."""
    logits, _ = model(x)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1), ignore_index=IGNORE,
        reduction="sum",
    )  # fmt: skip


@torch.no_grad()
def response_loss(
    model: GPT,
    items: Sequence[Encoded],
    batch_size: int,
    pad_id: int,
    device: torch.device,
    autocast: Callable[[], AbstractContextManager[Any]],
) -> dict[str, dict[str, float]]:
    """Per variety: nats per response token and bits per response character, teacher-forced."""
    was_training = model.training
    model.eval()
    try:
        nats: dict[str, float] = defaultdict(float)
        tokens: dict[str, int] = defaultdict(int)
        chars: dict[str, int] = defaultdict(int)
        ordered = sorted(items, key=lambda e: len(e.example.inputs))  # less padding
        for start in range(0, len(ordered), batch_size):
            chunk = ordered[start : start + batch_size]
            x, y = collate([e.example for e in chunk], pad_id)
            x, y = x.to(device), y.to(device)
            with autocast():
                logits, _ = model(x)
            per_token = F.cross_entropy(
                logits.float().transpose(1, 2), y, ignore_index=IGNORE, reduction="none"
            ).sum(dim=1)
            for e, value in zip(chunk, per_token.tolist(), strict=True):
                v = e.pair.variety
                nats[v] += value
                tokens[v] += e.example.response_tokens
                chars[v] += e.chars
        return {
            v: {
                "loss": nats[v] / tokens[v],
                "bits_per_char": nats[v] / math.log(2) / chars[v],
                "pairs": sum(e.pair.variety == v for e in items),
            }
            for v in sorted(nats)
        }
    finally:
        model.train(was_training)


def selection_score(per_variety: dict[str, dict[str, float]]) -> float:
    """Mean of the Arabic mean and English response bits per character (D-025)."""
    arabic = [per_variety[v]["bits_per_char"] for v in ("msa", "egyptian") if v in per_variety]
    english = per_variety["english"]["bits_per_char"]
    return 0.5 * (sum(arabic) / len(arabic) + english)


def build_model(cfg: SFTConfig, device: torch.device) -> tuple[GPT, dict[str, Any]]:
    base_state = load_checkpoint(cfg.base_checkpoint, map_location=device)
    model = model_from_state(base_state, device)
    if cfg.method.kind == "lora":
        m = cfg.method
        add_lora(model, m.rank, m.alpha, m.targets, m.dropout)
    model.train()
    return model, base_state


def finetune(
    cfg: SFTConfig, *, resume: str | None = None, progress: Progress = lambda _: None
) -> SFTResult:
    """Fine-tune `cfg.base_checkpoint` to `max_steps`. `resume` = "auto" continues from last.pt."""
    t = cfg.train
    device = resolve_device(t.device)
    autocast, precision = autocast_factory(device, t.precision)
    run_dir = cfg.run_dir
    checkpoint_path: Path | None = None
    if resume == "auto" and (run_dir / "last.pt").exists():
        checkpoint_path = run_dir / "last.pt"
    if checkpoint_path is None and (run_dir / "metrics.jsonl").exists():
        raise TrainingError(f"{run_dir} already holds a run; continue it with resume='auto'")

    seed_everything(t.seed)
    model, base_state = build_model(cfg, device)
    base_sha = sha256_file(cfg.base_checkpoint)
    problem = tokenizer_mismatch(base_state["config"], cfg.tokenizer)
    if problem:
        raise TrainingError(problem)
    tokenizer = Tokenizer.load(cfg.tokenizer)
    pad_id = tokenizer.special_id(END_OF_TEXT)
    context = model.cfg.context_length

    train_items, train_dropped = encode_pairs(
        read_split(cfg.data.data_dir, "train"), tokenizer, context
    )
    val_items, val_dropped = encode_pairs(read_split(cfg.data.data_dir, "val"), tokenizer, context)
    by_language: dict[str, list[EncodedExample]] = defaultdict(list)
    for e in train_items:
        by_language[e.pair.language].append(e.example)
    weights_by_language = {str(k): v for k, v in cfg.data.language_weights.items()}
    sampler = PairSampler(by_language, weights_by_language, t.micro_batch_size, t.seed)
    manifest = json.loads((cfg.data.data_dir / "manifest.json").read_text(encoding="utf-8"))

    tokens_dir = Path(base_state["config"]["data"]["tokens_dir"])
    sources = [load_source(tokens_dir, n) for n in training_source_names(tokens_dir)]
    forget_sets = validation_sets(sources, context) if t.forgetting_windows else {}
    source_meta = {s.name: s for s in sources}
    replay = None
    if t.replay_fraction > 0:
        # the base model's own pretraining mix: its configured weights, or each source's train
        # tokens when none were configured (`training.trainer.prepare_data`)
        chosen = base_state["config"]["data"].get("train_sources")
        replay_sources = [s for s in sources if chosen is None or s.name in chosen]
        replay_weights = [
            float(s.train_tokens) if chosen is None else chosen[s.name] for s in replay_sources
        ]
        replay = ReplaySampler(
            replay_sources, replay_weights, t.replay_fraction, t.micro_batch_size,
            t.grad_accum_steps, t.seed,
        )  # fmt: skip

    optimizer = make_optimizer(model, cfg.optim, device)
    step, best = 0, None
    if checkpoint_path is not None:
        state = load_checkpoint(checkpoint_path, map_location=device)
        if state["config"] != cfg.as_dict() or state["base_sha256"] != base_sha:
            raise TrainingError("the checkpoint was made with another config or base model")
        missing, unexpected = model.load_state_dict(
            state["model"], strict=cfg.method.kind == "full"
        )
        if unexpected or (cfg.method.kind == "lora" and any("lora_" in k for k in missing)):
            raise TrainingError("the checkpoint's weights do not fit the model")
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
        step, best = state["step"], state["best_score"]

    logger = JsonlLogger(run_dir / "metrics.jsonl")
    (run_dir / "config.json").write_text(json.dumps(cfg.as_dict(), indent=2, sort_keys=True) + "\n")
    logger.log(
        "resume" if checkpoint_path else "start", step=step, config=cfg.as_dict(),
        config_sha256=cfg.fingerprint(), base_sha256=base_sha, precision=precision,
        parameters=model.num_parameters(), trainable_parameters=trainable_parameters(model),
        train_pairs=len(train_items), val_pairs=len(val_items),
        dropped_too_long={"train": train_dropped, "val": val_dropped},
        train_pairs_by_language={k: len(v) for k, v in by_language.items()},
        data_files=manifest["files"], git=_git_state(), environment=_environment(device),
    )  # fmt: skip

    def weights() -> dict[str, torch.Tensor]:
        return lora_state(model) if cfg.method.kind == "lora" else model.state_dict()

    def snapshot(at: int) -> dict[str, Any]:
        return {
            "step": at, "model": weights(), "optimizer": optimizer.state_dict(),
            "rng": capture_rng(), "best_score": best, "config": cfg.as_dict(),
            "base_sha256": base_sha,
        }  # fmt: skip

    params = [p for p in model.parameters() if p.requires_grad]
    max_norm = cfg.optim.grad_clip if cfg.optim.grad_clip > 0 else math.inf
    completed, skips, window, started = step, 0, [], time.perf_counter()
    # since the start or the resume: replayed rows, and their share of the scored tokens
    replay_rows = replay_tokens = scored_tokens = 0
    guard = ExitStack()
    stop = guard.enter_context(stop_on_termination())
    try:
        for index in range(step, t.max_steps):
            lr = lr_at(index, cfg.optim.lr, cfg.optim.min_lr, cfg.optim.warmup_steps, t.max_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            if replay is None:
                micro = [
                    collate(sampler.batch(index, m), pad_id) for m in range(t.grad_accum_steps)
                ]
            else:
                micro = []
                for m in range(t.grad_accum_steps):
                    examples, swapped = replay.mix(index, m, sampler.batch(index, m))
                    replay_rows += len(swapped)
                    replay_tokens += sum(len(examples[i].targets) for i in swapped)
                    micro.append(collate(examples, pad_id))
            n_tokens = sum(int((y != IGNORE).sum()) for _, y in micro)
            scored_tokens += n_tokens
            total = torch.zeros((), device=device)
            for x, y in micro:
                with autocast():
                    loss_sum = summed_loss(model, x.to(device), y.to(device))
                (loss_sum / n_tokens).backward()
                total += loss_sum.detach()
            mean_loss = float(total) / n_tokens
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, max_norm))
            if math.isfinite(mean_loss) and math.isfinite(grad_norm):
                optimizer.step()
                skips = 0
                window.append(mean_loss)
            else:
                skips += 1
                logger.log("skip", step=index + 1, loss=mean_loss, grad_norm=grad_norm)
            optimizer.zero_grad(set_to_none=True)
            completed = index + 1
            if skips >= t.max_consecutive_skips:
                save_checkpoint(run_dir / "last.pt", snapshot(completed))
                raise DivergedError(f"{skips} consecutive non-finite steps at step {completed}")

            if window and (completed % t.log_interval == 0 or completed == t.max_steps):
                elapsed = time.perf_counter() - started
                logger.log(
                    "train", step=completed, loss=sum(window) / len(window), lr=lr,
                    grad_norm=grad_norm, seconds=elapsed,
                    peak_memory_mib=torch.cuda.max_memory_allocated() / 2**20
                    if device.type == "cuda" else None,
                    **({"replay_rows": replay_rows, "replay_token_share": replay_tokens
                        / scored_tokens} if replay is not None else {}),
                )  # fmt: skip
                mean = sum(window) / len(window)
                progress(f"step {completed}/{t.max_steps}  loss {mean:.4f}  lr {lr:.2e}")
                window = []

            if completed % t.eval_interval == 0 or completed == t.max_steps:
                per_variety = response_loss(
                    model, val_items, t.micro_batch_size, pad_id, device, autocast
                )
                score = selection_score(per_variety)
                forgetting = {}
                if forget_sets:
                    losses = evaluate(model, forget_sets, 8, t.forgetting_windows, device, autocast)
                    forgetting = {
                        n: bits_per_char(v, source_meta[n].val_tokens, source_meta[n].val_chars)
                        for n, v in losses.items()
                    }
                improved = best is None or score < best
                if improved:
                    best = score
                logger.log(
                    "eval", step=completed, val=per_variety, score=score, best=improved,
                    forgetting_bits_per_char=forgetting,
                )  # fmt: skip
                progress(
                    f"  eval step {completed}: score {score:.4f}{'  (best)' if improved else ''}"
                )
                if improved:
                    save_checkpoint(run_dir / "best.pt", snapshot(completed))
            if completed % t.checkpoint_interval == 0 or completed == t.max_steps:
                save_checkpoint(run_dir / "last.pt", snapshot(completed))
            if stop.signal_number is not None and completed < t.max_steps:
                raise KeyboardInterrupt(f"signal {stop.signal_number}")
        logger.log("end", step=completed, best_score=best)
    except KeyboardInterrupt:
        save_checkpoint(run_dir / "last.pt", snapshot(completed))
        logger.log("interrupted", step=completed)
        raise
    finally:
        logger.close()
        guard.close()

    merged = export_merged(cfg, run_dir / "best.pt", run_dir / "merged.pt", device)
    return SFTResult(run_dir, completed, best, merged)


def export_merged(cfg: SFTConfig, checkpoint: Path, out: Path, device: torch.device) -> Path:
    """The fine-tuned model of `checkpoint` (adapters merged) as an ordinary model checkpoint."""
    state = load_checkpoint(checkpoint, map_location=device)
    model, base_state = build_model(cfg, device)
    model.load_state_dict(state["model"], strict=cfg.method.kind == "full")
    merge_lora(model)
    save_checkpoint(
        out,
        {
            "step": state["step"],
            "model": model.state_dict(),
            "config": {**base_state["config"], "sft": cfg.as_dict()},
            "base_sha256": state["base_sha256"],
        },
    )
    return out
