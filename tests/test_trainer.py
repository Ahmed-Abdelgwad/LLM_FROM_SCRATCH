import json
import math
from pathlib import Path

import pytest
import torch
import yaml
from typer.testing import CliRunner

from scratchllm.cli import app
from scratchllm.model.gpt import GPT
from scratchllm.training import trainer as trainer_module
from scratchllm.training.batches import TrainSampler
from scratchllm.training.checkpoint import load_checkpoint
from scratchllm.training.logger import read_events
from scratchllm.training.schedule import lr_at
from scratchllm.training.trainer import DivergedError, TrainingError, train


def events(run_dir: Path, kind: str) -> list[dict]:
    return [e for e in read_events(run_dir / "metrics.jsonl") if e["event"] == kind]


def weights(run_dir: Path, name: str = "last.pt") -> dict[str, torch.Tensor]:
    return load_checkpoint(run_dir / name)["model"]


def same_weights(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


class StopAt:
    """A progress callback that interrupts the run when a given step is reported."""

    def __init__(self, step: int, error: type[BaseException]) -> None:
        self.marker, self.error = f"step {step}/", error

    def __call__(self, message: str) -> None:
        if message.startswith(self.marker):
            raise self.error


# ---------- a normal run ----------


def test_a_run_learns_logs_validates_and_checkpoints(make_cfg):
    cfg = make_cfg(train={"max_steps": 40, "eval_interval": 10, "checkpoint_interval": 10})
    result = train(cfg)
    run = cfg.run_dir
    assert result.final_step == 40 and result.tokens_seen == 40 * cfg.tokens_per_step
    assert result.skipped_steps == 0

    log = read_events(run / "metrics.jsonl")
    assert log[0]["event"] == "start" and log[-1]["event"] == "end"
    assert log[0]["config_sha256"] == cfg.fingerprint() and log[0]["parameters"] > 0
    assert log[0]["environment"]["device"] == "cpu" and "git" in log[0]
    assert (
        (run / "config.json").exists() and (run / "last.pt").exists() and (run / "best.pt").exists()
    )

    train_events = events(run, "train")
    assert [e["step"] for e in train_events] == list(range(1, 41))
    assert train_events[-1]["loss"] < 0.5 * train_events[0]["loss"]  # it learns the cycle
    evals = events(run, "eval")
    assert [e["step"] for e in evals] == [10, 20, 30, 40]
    assert evals[-1]["val_loss"] < evals[0]["val_loss"]
    assert set(evals[0]["sources"]) == {"alpha", "beta"}
    assert [e["step"] for e in events(run, "checkpoint")] == [10, 20, 30, 40]
    assert load_checkpoint(run / "last.pt")["step"] == 40


def test_the_logged_learning_rate_follows_the_schedule(make_cfg):
    cfg = make_cfg()
    train(cfg)
    o, steps = cfg.optim, cfg.train.max_steps
    for e in events(cfg.run_dir, "train"):
        expected = lr_at(e["step"] - 1, o.lr, o.min_lr, o.warmup_steps, steps)
        assert e["lr"] == pytest.approx(expected)
    lrs = [e["lr"] for e in events(cfg.run_dir, "train")]
    assert lrs[0] < lrs[1] and lrs[-1] < lrs[3]  # warmup rises, decay falls
    # and it is the rate the optimizer really used, not only the one that was logged
    groups = load_checkpoint(cfg.run_dir / "last.pt")["optimizer"]["param_groups"]
    final_lr = lr_at(steps - 1, o.lr, o.min_lr, o.warmup_steps, steps)
    assert [g["lr"] for g in groups] == pytest.approx([final_lr, final_lr])


def test_validation_reports_bits_per_character_per_source(make_cfg):
    cfg = make_cfg()
    train(cfg)
    last = events(cfg.run_dir, "eval")[-1]
    for source in last["sources"].values():
        # every source has 700 val tokens and 2100 chars (see conftest)
        assert source["bits_per_char"] == pytest.approx(source["loss"] / math.log(2) * 700 / 2100)
    assert last["val_loss"] == pytest.approx(sum(s["loss"] for s in last["sources"].values()) / 2)


def test_best_checkpoint_is_the_lowest_validation_loss(make_cfg):
    cfg = make_cfg(train={"max_steps": 24})
    train(cfg)
    evals = events(cfg.run_dir, "eval")
    best = min(evals, key=lambda e: e["val_loss"])
    assert load_checkpoint(cfg.run_dir / "best.pt")["step"] == best["step"]
    assert [e["best"] for e in evals][0] is True


def test_best_checkpoint_is_not_replaced_by_a_worse_evaluation(make_cfg, monkeypatch):
    scripted = iter([1.0, 0.5, 0.8])  # the three evaluations of a 12-step run
    monkeypatch.setattr(
        trainer_module, "evaluate", lambda *a, **k: dict.fromkeys(("alpha", "beta"), next(scripted))
    )
    cfg = make_cfg(train={"train_probe": False})  # one evaluate() call per evaluation
    result = train(cfg)
    assert [e["best"] for e in events(cfg.run_dir, "eval")] == [True, True, False]
    assert load_checkpoint(cfg.run_dir / "best.pt")["step"] == 8
    assert load_checkpoint(cfg.run_dir / "last.pt")["step"] == 12
    assert result.best_val_loss == 0.5


# ---------- reproducibility ----------


def test_same_seed_same_weights_and_another_seed_other_weights(make_cfg):
    a, b = make_cfg("a"), make_cfg("b")
    c = make_cfg("c", train={"seed": 8})
    for cfg in (a, b, c):
        train(cfg)
    assert same_weights(weights(a.run_dir), weights(b.run_dir))
    assert not same_weights(weights(a.run_dir), weights(c.run_dir))


# ---------- stopping and resuming ----------


@pytest.mark.parametrize("dropout", [0.0, 0.2])
def test_an_interrupted_run_resumes_to_exactly_the_uninterrupted_result(make_cfg, dropout):
    model = {"dropout": dropout}
    reference = make_cfg("reference", model=model)
    train(reference)

    cut = make_cfg("cut", model=model)
    with pytest.raises(KeyboardInterrupt):
        train(cut, progress=StopAt(6, KeyboardInterrupt))
    assert load_checkpoint(cut.run_dir / "last.pt")["step"] == 6
    assert events(cut.run_dir, "interrupted")[0]["step"] == 6
    result = train(cut, resume="auto")

    assert result.final_step == 12
    assert same_weights(weights(reference.run_dir), weights(cut.run_dir))
    ref_steps = {e["step"]: e["loss"] for e in events(reference.run_dir, "train")}
    got_steps = {e["step"]: e["loss"] for e in events(cut.run_dir, "train")}
    assert got_steps == ref_steps  # the same loss at every step, before and after the cut
    ref_eval = [(e["step"], e["val_loss"]) for e in events(reference.run_dir, "eval")]
    assert [(e["step"], e["val_loss"]) for e in events(cut.run_dir, "eval")] == ref_eval
    assert [e["event"] for e in read_events(cut.run_dir / "metrics.jsonl")].count("resume") == 1


def test_a_crash_without_a_chance_to_save_resumes_from_the_last_checkpoint(make_cfg):
    reference = make_cfg("reference", model={"dropout": 0.1})
    train(reference)
    crash = make_cfg("crash", model={"dropout": 0.1})
    with pytest.raises(RuntimeError):
        train(crash, progress=StopAt(10, RuntimeError))  # checkpoints at 4 and 8; dies at 10
    assert load_checkpoint(crash.run_dir / "last.pt")["step"] == 8
    train(crash, resume="auto")
    assert same_weights(weights(reference.run_dir), weights(crash.run_dir))


def test_resuming_a_finished_run_changes_nothing(make_cfg):
    cfg = make_cfg()
    train(cfg)
    before = weights(cfg.run_dir)
    result = train(cfg, resume="auto")
    assert result.final_step == 12 and same_weights(before, weights(cfg.run_dir))


def test_a_run_can_be_extended_and_the_change_is_logged(make_cfg):
    first = make_cfg("ext", train={"max_steps": 8})
    train(first)
    longer = make_cfg("ext", train={"max_steps": 16}, optim={"min_lr": 0.0005})
    result = train(longer, resume="auto")
    assert result.final_step == 16
    resume_event = events(longer.run_dir, "resume")[0]
    assert resume_event["step"] == 8 and resume_event["config_sections_changed"] == [
        "optim",
        "train",
    ]


def test_resume_auto_without_a_checkpoint_starts_fresh(make_cfg):
    cfg = make_cfg()
    assert train(cfg, resume="auto").final_step == 12
    assert events(cfg.run_dir, "start") and not events(cfg.run_dir, "resume")


def test_an_existing_run_is_never_overwritten(make_cfg):
    cfg = make_cfg()
    train(cfg)
    with pytest.raises(TrainingError, match="already holds a run"):
        train(cfg)


def test_a_missing_checkpoint_path_is_an_error(make_cfg):
    with pytest.raises(FileNotFoundError):
        train(make_cfg(), resume="nowhere.pt")


def test_resuming_with_a_different_model_or_data_stream_or_files_is_refused(make_cfg, tokens_dir):
    base = make_cfg("base", train={"max_steps": 4})
    train(base)
    path = str(base.run_dir / "last.pt")
    with pytest.raises(TrainingError, match="model configuration"):
        train(make_cfg("m", train={"max_steps": 8}, model={"d_model": 64}), resume=path)
    with pytest.raises(TrainingError, match="seed"):
        train(make_cfg("s", train={"max_steps": 8, "seed": 99}), resume=path)
    with pytest.raises(TrainingError, match="seed, batch sizes"):
        train(make_cfg("g", train={"max_steps": 8, "grad_accum_steps": 3}), resume=path)
    with pytest.raises(TrainingError, match="seed, batch sizes"):
        weighted = make_cfg(
            "w", train={"max_steps": 8}, data={"train_sources": {"alpha": 1, "beta": 3}}
        )
        train(weighted, resume=path)
    meta_path = tokens_dir / "alpha.tokens.json"
    meta = json.loads(meta_path.read_text())
    meta["splits"]["train"]["sha256"] = "0" * 64
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(TrainingError, match="token files differ"):
        train(make_cfg("d", train={"max_steps": 8}), resume=path)


# ---------- data compatibility ----------


def test_a_vocabulary_that_does_not_match_the_token_files_is_refused(make_cfg):
    with pytest.raises(TrainingError, match="vocabulary"):
        train(make_cfg(model={"vocab_size": 128}))


def test_a_source_without_a_description_is_refused(make_cfg):
    with pytest.raises(FileNotFoundError):
        train(make_cfg(data={"train_sources": {"alpha": 1, "ghost": 1}}))


def test_val_sources_can_differ_from_train_sources(make_cfg):
    cfg = make_cfg(data={"train_sources": {"alpha": 1}, "val_sources": ["alpha", "beta"]})
    train(cfg)
    assert set(events(cfg.run_dir, "eval")[0]["sources"]) == {"alpha", "beta"}


def test_sources_with_different_tokenizers_are_refused(make_cfg, tokens_dir):
    meta_path = tokens_dir / "beta.tokens.json"
    meta = json.loads(meta_path.read_text())
    meta["tokenizer_sha256"] = "x" * 64
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(TrainingError, match="different tokenizers"):
        train(make_cfg())


# ---------- gradient accumulation, clipping, bad steps ----------


def test_accumulating_micro_batches_equals_one_big_batch(make_cfg, monkeypatch):
    fixed_x = torch.randint(0, 8, (8, 16), generator=torch.Generator().manual_seed(0))
    fixed_y = torch.roll(fixed_x, -1, dims=1)

    def fixed_batch(self: TrainSampler, step: int, micro: int):
        rows = slice(micro * self.micro_batch_size, (micro + 1) * self.micro_batch_size)
        return fixed_x[rows], fixed_y[rows]

    monkeypatch.setattr(TrainSampler, "batch", fixed_batch)
    kw = {"max_steps": 1, "eval_interval": 1, "checkpoint_interval": 1}
    no_warmup = {"warmup_steps": 0}
    small = make_cfg(
        "small", train={**kw, "micro_batch_size": 4, "grad_accum_steps": 2}, optim=no_warmup
    )
    big = make_cfg(
        "big", train={**kw, "micro_batch_size": 8, "grad_accum_steps": 1}, optim=no_warmup
    )
    train(small)
    train(big)
    a, b = weights(small.run_dir), weights(big.run_dir)
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=1e-6, rtol=1e-5)
    assert events(small.run_dir, "train")[0]["loss"] == pytest.approx(
        events(big.run_dir, "train")[0]["loss"], rel=1e-5
    )
    # Adam and clipping hide a wrong gradient scale in the weights; the raw norm does not
    assert events(small.run_dir, "train")[0]["grad_norm"] == pytest.approx(
        events(big.run_dir, "train")[0]["grad_norm"], rel=1e-4
    )


def test_gradients_are_clipped_to_the_configured_norm_and_the_raw_norm_is_logged(
    make_cfg, monkeypatch
):
    real = torch.nn.utils.clip_grad_norm_
    seen: list[tuple[float, float]] = []

    def spy(parameters, max_norm, *args, **kwargs):
        params = list(parameters)
        total = real(params, max_norm, *args, **kwargs)
        after = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in params if p.grad is not None))
        seen.append((max_norm, float(after)))
        return total

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    cfg = make_cfg(optim={"grad_clip": 0.01}, train={"max_steps": 3})
    train(cfg)
    assert len(seen) == 3
    for max_norm, after in seen:
        assert max_norm == 0.01 and after == pytest.approx(0.01, rel=1e-3)
    assert all(e["grad_norm"] > 0.01 for e in events(cfg.run_dir, "train"))  # raw, before clipping


def test_clipping_can_be_turned_off(make_cfg, monkeypatch):
    used: list[float] = []
    real = torch.nn.utils.clip_grad_norm_

    def spy(parameters, max_norm, *args, **kwargs):
        used.append(max_norm)
        return real(parameters, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    train(make_cfg(optim={"grad_clip": 0.0}, train={"max_steps": 2}))
    assert used == [math.inf, math.inf]


def poison_forward(monkeypatch, bad_calls: set[int]) -> None:
    """Make the loss NaN on the given forward calls (counted from 0)."""
    real, count = GPT.forward, {"n": 0}

    def forward(self, ids, targets=None):
        logits, loss = real(self, ids, targets)
        count["n"] += 1
        if loss is not None and (count["n"] - 1) in bad_calls:
            loss = loss * math.nan
        return logits, loss

    monkeypatch.setattr(GPT, "forward", forward)


def test_a_non_finite_step_is_skipped_logged_and_the_run_goes_on(make_cfg, monkeypatch):
    # two forward calls per step; calls 4 and 5 are step 3 (1-based)
    poison_forward(monkeypatch, {4, 5})
    cfg = make_cfg(train={"max_steps": 6, "eval_interval": 3, "checkpoint_interval": 3})
    result = train(cfg)
    assert result.skipped_steps == 1 and result.final_step == 6
    assert result.tokens_seen == 5 * cfg.tokens_per_step
    skip = events(cfg.run_dir, "skip")[0]
    assert skip["step"] == 3 and skip["loss"] == "nan"
    assert [e["step"] for e in events(cfg.run_dir, "train")] == [1, 2, 4, 5, 6]
    assert all(torch.isfinite(w).all() for w in weights(cfg.run_dir).values())
    assert [e["step"] for e in events(cfg.run_dir, "eval")] == [
        3,
        6,
    ]  # the skipped step still evaluates


def test_a_skipped_step_does_not_change_the_weights(make_cfg, monkeypatch):
    # with every step poisoned and a high limit, nothing is ever learned
    poison_forward(monkeypatch, set(range(100)))
    cfg = make_cfg(train={"max_steps": 4, "max_consecutive_skips": 99})
    torch.manual_seed(cfg.train.seed)
    initial = GPT(cfg.model.to_gpt()).state_dict()
    train(cfg)
    final = weights(cfg.run_dir)
    assert same_weights({k: initial[k] for k in final}, final)


def test_too_many_non_finite_steps_in_a_row_abort_the_run_with_a_checkpoint(make_cfg, monkeypatch):
    poison_forward(monkeypatch, set(range(4, 100)))  # from step 3 on
    cfg = make_cfg(train={"max_consecutive_skips": 3, "checkpoint_interval": 100})
    with pytest.raises(DivergedError, match="3 consecutive"):
        train(cfg)
    assert len(events(cfg.run_dir, "skip")) == 3
    assert events(cfg.run_dir, "diverged")[0]["step"] == 5
    assert load_checkpoint(cfg.run_dir / "last.pt")["step"] == 5


# ---------- the command line ----------


def test_cli_trains_and_resumes(make_cfg, tmp_path: Path):
    cfg = make_cfg("cli", train={"max_steps": 6})
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg.as_dict()), encoding="utf-8")
    runner = CliRunner()
    first = runner.invoke(app, ["train", str(path)])
    assert first.exit_code == 0, first.output
    assert "done: step 6" in first.output and "bits/char" in first.output
    again = runner.invoke(app, ["train", str(path)])
    assert again.exit_code == 1 and "already holds a run" in again.output
    longer = runner.invoke(
        app, ["train", str(path), "--resume", "auto", "--set", "train.max_steps=9"]
    )
    assert longer.exit_code == 0 and "done: step 9" in longer.output, longer.output


# ---------- being asked to stop ----------


def test_sigterm_saves_a_checkpoint_and_the_run_resumes_exactly(make_cfg):
    import os
    import signal

    reference = make_cfg("reference", model={"dropout": 0.1})
    train(reference)
    killed = make_cfg("killed", model={"dropout": 0.1})
    before = signal.getsignal(signal.SIGTERM)

    def terminate_at_6(message: str) -> None:
        if message.startswith("step 6/"):
            os.kill(os.getpid(), signal.SIGTERM)

    with pytest.raises(KeyboardInterrupt):
        train(killed, progress=terminate_at_6)
    assert signal.getsignal(signal.SIGTERM) is before  # the handler is put back
    assert load_checkpoint(killed.run_dir / "last.pt")["step"] == 6
    assert events(killed.run_dir, "interrupted")[0]["step"] == 6
    train(killed, resume="auto")
    assert same_weights(weights(reference.run_dir), weights(killed.run_dir))


def test_a_signal_inside_the_optimizer_update_still_leaves_an_exact_checkpoint(
    make_cfg, monkeypatch
):
    """D-026: the signal lands after the step was counted and before the weights changed. The run
    must finish that update and stop at the step boundary, so resuming gives the same weights."""
    import os
    import signal

    reference = make_cfg("reference")
    train(reference)
    killed = make_cfg("killed")
    real_make = trainer_module.make_optimizer

    def make_signalling_optimizer(model, cfg, device):
        optimizer = real_make(model, cfg, device)
        real_step, calls = optimizer.step, {"n": 0}

        def step(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 6:
                os.kill(os.getpid(), signal.SIGTERM)  # handled before the real update runs
            return real_step(*args, **kwargs)

        optimizer.step = step
        return optimizer

    monkeypatch.setattr(trainer_module, "make_optimizer", make_signalling_optimizer)
    with pytest.raises(KeyboardInterrupt):
        train(killed)
    monkeypatch.undo()
    assert load_checkpoint(killed.run_dir / "last.pt")["step"] == 6
    assert events(killed.run_dir, "interrupted")[0]["step"] == 6
    train(killed, resume="auto")
    assert same_weights(weights(reference.run_dir), weights(killed.run_dir))


def test_a_second_signal_stops_at_once(make_cfg):
    import os
    import signal

    cfg = make_cfg()

    def terminate_twice_at_6(message: str) -> None:
        if message.startswith("step 6/"):
            os.kill(os.getpid(), signal.SIGTERM)
            os.kill(os.getpid(), signal.SIGTERM)

    with pytest.raises(KeyboardInterrupt, match="second request"):
        train(cfg, progress=terminate_twice_at_6)
    assert load_checkpoint(cfg.run_dir / "last.pt")["step"] == 6


def test_the_signal_handlers_are_restored_after_a_normal_run(make_cfg):
    import signal

    names = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    before = [signal.getsignal(n) for n in names]
    train(make_cfg(train={"max_steps": 2}, optim={"warmup_steps": 0}))
    assert [signal.getsignal(n) for n in names] == before


# ---------- the train-side probe ----------


def test_each_evaluation_also_scores_fixed_windows_of_the_training_tokens(make_cfg, monkeypatch):
    calls: list[dict[str, int]] = []
    real = trainer_module.evaluate

    def spy(model, sets, *args, **kwargs):
        calls.append({name: len(ds) for name, ds in sets.items()})
        return real(model, sets, *args, **kwargs)

    monkeypatch.setattr(trainer_module, "evaluate", spy)
    cfg = make_cfg(train={"max_steps": 4})
    train(cfg)
    # per evaluation: first the 700-token val split (43 windows), then the 4000-token train split
    assert calls == [{"alpha": 43, "beta": 43}, {"alpha": 249, "beta": 249}]
    source = events(cfg.run_dir, "eval")[0]["sources"]["alpha"]
    assert set(source) == {"loss", "bits_per_char", "train_loss", "train_bits_per_char"}
    assert source["train_bits_per_char"] == pytest.approx(
        source["train_loss"] / math.log(2) * 4000 / 12000
    )


def test_the_train_probe_can_be_turned_off(make_cfg):
    cfg = make_cfg(train={"max_steps": 4, "train_probe": False})
    train(cfg)
    assert set(events(cfg.run_dir, "eval")[0]["sources"]["alpha"]) == {"loss", "bits_per_char"}


def test_the_probe_does_not_change_training(make_cfg):
    on, off = make_cfg("on"), make_cfg("off", train={"train_probe": False})
    train(on)
    train(off)
    assert same_weights(weights(on.run_dir), weights(off.run_dir))
