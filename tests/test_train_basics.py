import json
import math
from pathlib import Path

import pytest
import torch
import yaml
from pydantic import ValidationError

from scratchllm.model.gpt import GPT, GPTConfig
from scratchllm.training.checkpoint import (
    capture_rng,
    load_checkpoint,
    restore_rng,
    save_checkpoint,
)
from scratchllm.training.config import RunConfig, load_config
from scratchllm.training.logger import JsonlLogger, read_events
from scratchllm.training.schedule import lr_at
from scratchllm.training.trainer import autocast_factory, make_optimizer

# ---------- learning-rate schedule ----------


def test_warmup_is_linear_and_reaches_the_peak_on_its_last_step():
    lrs = [lr_at(s, 1.0, 0.1, warmup_steps=4, max_steps=14) for s in range(4)]
    assert lrs == pytest.approx([0.25, 0.5, 0.75, 1.0])


def test_cosine_decay_goes_from_the_peak_to_the_floor():
    kw = {"lr": 1.0, "min_lr": 0.1, "warmup_steps": 4, "max_steps": 14}
    assert lr_at(4, **kw) == pytest.approx(1.0)  # first decay step
    assert lr_at(9, **kw) == pytest.approx(0.55)  # halfway: the midpoint of peak and floor
    assert lr_at(14, **kw) == pytest.approx(0.1)
    assert lr_at(500, **kw) == pytest.approx(0.1)  # stays at the floor
    decay = [lr_at(s, **kw) for s in range(4, 15)]
    assert all(a >= b for a, b in zip(decay, decay[1:], strict=False))


def test_schedule_edge_cases():
    assert lr_at(0, 1.0, 0.0, warmup_steps=0, max_steps=10) == pytest.approx(1.0)
    assert lr_at(5, 1.0, 0.2, warmup_steps=5, max_steps=5) == pytest.approx(0.2)
    assert lr_at(0, 1.0, 1.0, warmup_steps=0, max_steps=10) == pytest.approx(1.0)


# ---------- configuration ----------


def test_a_yaml_config_loads_and_overrides_apply(tmp_path: Path, make_cfg):
    cfg = make_cfg()
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg.as_dict()), encoding="utf-8")
    loaded = load_config(path, ["train.max_steps=33", "optim.lr=0.005", "model.dropout=0.1"])
    assert loaded.train.max_steps == 33 and loaded.optim.lr == 0.005
    assert loaded.model.dropout == 0.1 and loaded.model.d_model == 32


def test_the_shipped_configs_are_valid():
    in_use = sorted(Path("configs").glob("*.yaml"))
    assert {p.stem for p in in_use} == {"smoke", "local-17m", "reference-17m"}
    experiments = sorted(Path("configs/experiments").glob("*.yaml"))
    assert experiments, "the configs of finished experiments live in configs/experiments/"
    for path in in_use + experiments:
        cfg = load_config(path)
        assert cfg.name == path.stem, f"{path.name}: the run name differs from the file name"
        assert cfg.model.to_gpt().vocab_size == 16384


def test_the_reference_architecture_is_rotary_swiglu_and_layernorm():
    cfg = load_config(Path("configs/reference-17m.yaml"))
    assert (cfg.model.norm, cfg.model.position, cfg.model.ffn) == ("layernorm", "rope", "swiglu")
    assert GPT(cfg.model.to_gpt()).num_parameters() == 16_913_280


def test_unknown_keys_and_bad_values_are_refused(make_cfg):
    tree = make_cfg().as_dict()
    with pytest.raises(ValidationError):
        RunConfig.model_validate({**tree, "optimiser": {}})  # typo of a section
    with pytest.raises(ValidationError):
        RunConfig.model_validate({**tree, "optim": {**tree["optim"], "learning_rate": 1}})
    for bad in (
        {"optim": {**tree["optim"], "min_lr": 1.0}},  # floor above the peak
        {"optim": {**tree["optim"], "warmup_steps": 99}},  # longer than the run
        {"model": {**tree["model"], "n_heads": 5}},  # 32 not divisible by 5
        {"train": {**tree["train"], "micro_batch_size": 0}},
        {"name": "has/slash"},
    ):
        with pytest.raises(ValidationError):
            RunConfig.model_validate({**tree, **bad})


def test_a_bad_override_is_refused(tmp_path: Path, make_cfg):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(make_cfg().as_dict()), encoding="utf-8")
    with pytest.raises(ValueError, match="section.key=value"):
        load_config(path, ["train.max_steps"])
    with pytest.raises(ValidationError):
        load_config(path, ["train.nonexistent=1"])


def test_the_fingerprint_changes_with_any_setting(make_cfg):
    a, b = make_cfg(), make_cfg()
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != make_cfg(optim={"lr": 0.02}).fingerprint()
    assert a.tokens_per_step == 4 * 2 * 16


# ---------- optimizer groups ----------


def test_weight_decay_applies_to_matrices_only():
    model = GPT(GPTConfig(vocab_size=64, context_length=16, d_model=32, n_layers=2, n_heads=4))
    cfg = RunConfig.model_validate(
        {
            "name": "x",
            "model": {
                "vocab_size": 64, "context_length": 16, "d_model": 32, "n_layers": 2, "n_heads": 4,
            },
            "data": {"tokens_dir": "."},
            "optim": {"lr": 0.01, "min_lr": 0.0, "warmup_steps": 0, "weight_decay": 0.1},
            "train": {"max_steps": 1, "micro_batch_size": 1, "eval_interval": 1,
                      "checkpoint_interval": 1},
        }
    )  # fmt: skip
    opt = make_optimizer(model, cfg.optim, torch.device("cpu"))
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.ndim >= 2 for p in decay["params"]) and all(p.ndim < 2 for p in no_decay["params"])
    assert len(decay["params"]) + len(no_decay["params"]) == len(list(model.parameters()))
    assert any(p is model.tok_emb.weight for p in decay["params"])  # tied weight counted once


def test_precision_choice():
    _, label = autocast_factory(torch.device("cpu"), "auto")
    assert label == "fp32"
    assert autocast_factory(torch.device("cpu"), "fp32")[1] == "fp32"
    assert autocast_factory(torch.device("cpu"), "bf16")[1] == "bf16"


# ---------- checkpoint files ----------


def test_a_checkpoint_round_trips_with_weights_only_loading(tmp_path: Path):
    path = tmp_path / "sub" / "c.pt"
    torch.manual_seed(1)
    state = {
        "step": 3,
        "w": torch.randn(2, 3),
        "nested": {"a": [1, 2.5, None]},
        "rng": capture_rng(),
    }
    save_checkpoint(path, state)
    back = load_checkpoint(path)
    assert (
        back["step"] == 3
        and torch.equal(back["w"], state["w"])
        and back["nested"] == state["nested"]
    )
    assert not list(path.parent.glob("*.tmp"))


def test_rng_state_is_restored_exactly():
    torch.manual_seed(5)
    saved = capture_rng()
    first = torch.rand(3)
    restore_rng(saved)
    assert torch.equal(torch.rand(3), first)


def test_a_failed_save_leaves_the_previous_checkpoint_intact(tmp_path: Path, monkeypatch):
    path = tmp_path / "c.pt"
    save_checkpoint(path, {"step": 1})

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", boom)
    with pytest.raises(OSError):
        save_checkpoint(path, {"step": 2})
    monkeypatch.undo()
    assert load_checkpoint(path)["step"] == 1


def test_a_checkpoint_of_another_format_is_refused(tmp_path: Path):
    path = tmp_path / "c.pt"
    torch.save({"step": 1, "format_version": 99}, path)
    with pytest.raises(ValueError, match="format"):
        load_checkpoint(path)


# ---------- the log ----------


def test_log_lines_are_json_with_event_and_time_and_survive_nan(tmp_path: Path):
    path = tmp_path / "m.jsonl"
    log = JsonlLogger(path)
    log.log("train", step=1, loss=0.5, where=Path("a/b"))
    log.log("skip", loss=math.nan, grad_norm=math.inf, nested={"x": -math.inf})
    log.close()
    first, second = read_events(path)
    assert first["event"] == "train" and first["loss"] == 0.5 and first["where"] == "a/b"
    assert "time" in first
    assert (second["loss"], second["grad_norm"], second["nested"]["x"]) == ("nan", "inf", "-inf")
    for line in path.read_text().splitlines():
        json.loads(line)  # strict JSON: no bare NaN


def test_a_second_logger_appends(tmp_path: Path):
    path = tmp_path / "m.jsonl"
    for event in ("a", "b"):
        log = JsonlLogger(path)
        log.log(event)
        log.close()
    assert [e["event"] for e in read_events(path)] == ["a", "b"]
