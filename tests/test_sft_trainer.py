"""The fine-tuning loop (finetune/trainer.py), on CPU with a tiny model, tokenizer and data."""

import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

from conftest import write_source
from scratchllm.data.tokens import END_OF_TEXT
from scratchllm.finetune.chat import encode_example
from scratchllm.finetune.config import SFTConfig, load_sft_config
from scratchllm.finetune.data import Pair, build
from scratchllm.finetune.lora import lora_parameter_count
from scratchllm.finetune.trainer import (
    collate,
    finetune,
    response_loss,
    summed_loss,
)
from scratchllm.model.gpt import GPT, GPTConfig
from scratchllm.tokenizer.tokenizer import Tokenizer
from scratchllm.training.checkpoint import load_checkpoint, load_model, save_checkpoint
from scratchllm.training.logger import read_events
from scratchllm.training.trainer import TrainingError

# adapters on every map, so that a few steps on a tiny model show learning clearly
TARGETS = ["attn.q", "attn.v", "attn.out_proj", "ff.gate", "ff.up", "ff.down"]

PAIRS = [
    ("ما عاصمة مصر", "القاهرة", "msa"),
    ("ما عاصمة تونس", "تونس", "msa"),
    ("عاصمة مصر ايه", "القاهرة يا عم", "egyptian"),
    ("capital of France", "Paris", "english"),
    ("capital of Italy", "Rome", "english"),
]


@pytest.fixture()
def setup(tmp_path: Path):
    texts = [f"{p} {r}" for p, r, _ in PAIRS] * 20 + ["<|user|> <|assistant|>"] * 5
    tok = Tokenizer.train(texts, 256 + 50 + 3, [END_OF_TEXT, "<email>", "<phone>"])
    tok_path = tmp_path / "tok.json"
    tok.save(tok_path)
    tokens = tmp_path / "tokens"
    tokens.mkdir()
    write_source(tokens, "alpha", base=0, vocab=tok.vocab_size)
    sha = hashlib.sha256(tok_path.read_bytes()).hexdigest()
    meta_path = tokens / "alpha.tokens.json"
    meta = json.loads(meta_path.read_text())
    meta["tokenizer_sha256"] = sha
    meta_path.write_text(json.dumps(meta))

    torch.manual_seed(0)
    gpt_cfg = GPTConfig(
        vocab_size=tok.vocab_size, context_length=64, d_model=32, n_layers=2, n_heads=4,
        position="rope", ffn="swiglu",
    )  # fmt: skip
    base = tmp_path / "base.pt"
    model = GPT(gpt_cfg)
    with torch.no_grad():
        # A random model's embeddings (std 0.02) are also its output map (tied): its logits cannot
        # exceed about 0.6, whatever the hidden state, so no adapter could make it confident.
        # A pretrained model's embeddings are far larger; this gives the stand-in the same room.
        model.tok_emb.weight.mul_(25)
    save_checkpoint(
        base,
        {
            "step": 1,
            "model": model.state_dict(),
            "config": {"model": gpt_cfg.to_dict(), "data": {"tokens_dir": str(tokens)}},
        },
    )
    data = tmp_path / "sft"
    data.mkdir()
    pool = [Pair(f"aya-{i}", "aya", v, p, r) for i, (p, r, v) in enumerate(PAIRS)]
    pairs, _ = build([], pool)
    for split in ("train", "val", "test"):  # every pair in every split: a memorization task
        lines = [json.dumps({**p.__dict__, "split": split}, ensure_ascii=False) for p in pairs]
        (data / f"{split}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (data / "manifest.json").write_text(json.dumps({"files": {}}))

    def make(name="run", kind="full", **train):
        tree = {
            "name": name,
            "runs_dir": str(tmp_path / "runs"),
            "base_checkpoint": str(base),
            "tokenizer": str(tok_path),
            "data": {"data_dir": str(data)},
            "method": {"kind": kind, "rank": 4, "alpha": 4, "targets": TARGETS},
            "optim": {"lr": 3e-3 if kind == "full" else 3e-2, "min_lr": 3e-4, "warmup_steps": 2},
            "train": {
                "seed": 3, "max_steps": 40, "micro_batch_size": 4, "grad_accum_steps": 2,
                "eval_interval": 10, "checkpoint_interval": 10, "forgetting_windows": 2,
                "device": "cpu", "precision": "fp32", **train,
            },
        }  # fmt: skip
        return SFTConfig.model_validate(tree)

    return make, tok, base


def evals(run: Path) -> list[dict]:
    return [e for e in read_events(run / "metrics.jsonl") if e["event"] == "eval"]


@pytest.mark.parametrize("kind", ["full", "lora"])
def test_a_run_learns_the_answers_and_writes_a_merged_ordinary_checkpoint(setup, kind):
    make, _, base = setup
    cfg = make(kind=kind)
    before = base.read_bytes()
    result = finetune(cfg)
    history = evals(cfg.run_dir)
    assert [e["step"] for e in history] == [10, 20, 30, 40]
    assert history[-1]["score"] < 0.9 * history[0]["score"]  # it learns
    assert set(history[0]["val"]) == {"msa", "egyptian", "english"}
    assert set(history[0]["forgetting_bits_per_char"]) == {"alpha"}
    merged = load_model(result.merged)  # the ordinary loader reads it
    assert isinstance(merged, GPT) and base.read_bytes() == before  # the base file is untouched
    start = read_events(cfg.run_dir / "metrics.jsonl")[0]
    if kind == "lora":
        hidden = load_model(base).cfg.hidden
        assert start["trainable_parameters"] == lora_parameter_count(32, hidden, 2, 4, TARGETS)
        base_weights = load_checkpoint(base)["model"]
        changed = {k for k, v in merged.state_dict().items() if not torch.equal(v, base_weights[k])}
        adapted = (
            "attn.qkv.weight",
            "attn.out_proj.weight",
            "ff.gate.weight",
            "ff.up.weight",
            "ff.down.weight",
        )
        assert changed and all(k.endswith(adapted) for k in changed)
        kept = load_checkpoint(cfg.run_dir / "last.pt")["model"]
        assert all(".lora_" in k for k in kept)  # adapters only
    else:
        assert start["trainable_parameters"] == start["parameters"]


def test_the_step_loss_weights_every_response_token_equally(setup):
    """Two micro-batches of different lengths give the same gradient as one batch of all."""
    _, tok, base = setup
    examples = [encode_example(tok, p, r, 64) for p, r, _ in PAIRS[:4]]
    pad = tok.special_id(END_OF_TEXT)

    def grads(groups):
        model = load_model(base).train()
        n = sum(e.response_tokens for g in groups for e in g)
        for g in groups:
            x, y = collate(g, pad)
            (summed_loss(model, x, y) / n).backward()
        return [p.grad.clone() for p in model.parameters()]

    split = grads([examples[:1], examples[1:]])
    whole = grads([examples])
    for a, b in zip(split, whole, strict=True):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)


def test_padding_is_never_a_target_and_response_loss_is_per_character(setup):
    _, tok, base = setup
    examples = [encode_example(tok, p, r, 64) for p, r, _ in PAIRS]
    x, y = collate(examples, tok.special_id(END_OF_TEXT))
    assert int((y != -100).sum()) == sum(e.response_tokens for e in examples)
    model = load_model(base)
    from scratchllm.finetune.trainer import Encoded

    items = [Encoded(Pair(str(i), "aya", v, p, r), e, len(r) + 1)
             for i, ((p, r, v), e) in enumerate(zip(PAIRS, examples, strict=True))]  # fmt: skip
    out = response_loss(model, items, 2, 0, torch.device("cpu"), lambda: torch.no_grad())
    eng = [i for i in items if i.pair.variety == "english"]
    with torch.no_grad():
        xs, ys = collate([i.example for i in eng], 0)
        nats = float(summed_loss(model, xs, ys))
    tokens = sum(i.example.response_tokens for i in eng)
    chars = sum(i.chars for i in eng)
    assert out["english"]["loss"] == pytest.approx(nats / tokens, rel=1e-5)
    assert out["english"]["bits_per_char"] == pytest.approx(nats / math.log(2) / chars, rel=1e-5)


@pytest.mark.parametrize("kind", ["full", "lora"])
def test_a_stopped_run_resumes_to_exactly_the_uninterrupted_result(setup, kind):
    make, _, _ = setup
    whole = finetune(make("whole", kind, max_steps=20))

    class Stop:
        def __call__(self, message: str) -> None:
            if message.startswith("step 10/"):
                raise KeyboardInterrupt

    cut = make("cut", kind, max_steps=20)
    with pytest.raises(KeyboardInterrupt):
        finetune(cut, progress=Stop())
    resumed = finetune(cut, resume="auto")
    a, b = load_checkpoint(whole.merged)["model"], load_checkpoint(resumed.merged)["model"]
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_a_run_is_not_overwritten_and_a_changed_config_cannot_resume(setup):
    make, _, _ = setup
    finetune(make("x", max_steps=10))
    with pytest.raises(TrainingError, match="already holds"):
        finetune(make("x", max_steps=10))
    with pytest.raises(TrainingError, match="another config"):
        finetune(make("x", max_steps=12), resume="auto")


def test_the_config_file_loads_with_overrides_and_refuses_bad_targets(tmp_path, setup):
    make, _, _ = setup
    path = tmp_path / "c.yaml"
    import yaml

    path.write_text(yaml.safe_dump(json.loads(make().model_dump_json())))
    cfg = load_sft_config(path, ["method.kind=lora", "train.max_steps=5"])
    assert cfg.method.kind == "lora" and cfg.train.max_steps == 5
    with pytest.raises(ValueError, match="bad LoRA targets"):
        load_sft_config(path, ["method.targets=[attn.query]"])


# ---------- the test-split evaluation ----------


def test_the_evaluation_reports_every_variety_and_improves_after_training(setup):
    from scratchllm.finetune.evaluate import evaluate_sft, loops, render_markdown

    make, tok, base = setup
    cfg = make("e", max_steps=40)
    result = finetune(cfg)
    data = cfg.data.data_dir

    def report(path):
        return evaluate_sft(
            load_model(path), tok, data, torch.device("cpu"), lambda: torch.no_grad(),
            generate_per_variety=3, max_new_tokens=8,
        )  # fmt: skip

    before, after = report(base), report(result.merged)
    for r in (before, after):
        assert set(r["generation"]) == {"msa", "egyptian", "english"}
        assert all(0 <= g["stopped_rate"] <= 1 for g in r["generation"].values())
        assert "| msa |" in render_markdown(r)
    assert (
        after["teacher_forced"]["english"]["bits_per_char"]
        < before["teacher_forced"]["english"]["bits_per_char"]
    )
    assert loops("a b c a b c a b c") and not loops("a b c d e f")


def test_the_shipped_sft_configs_load_and_differ_only_in_name_method_and_learning_rate():
    paths = sorted(Path("configs/sft").glob("*.yaml"))
    assert [p.stem for p in paths] == ["sft-full", "sft-lora"]
    full, lora = (load_sft_config(p).as_dict() for p in paths)
    assert full["name"] == "sft-full" and lora["name"] == "sft-lora"
    assert full["method"]["kind"] == "full" and lora["method"]["kind"] == "lora"
    for section in ("data", "train", "base_checkpoint", "tokenizer"):
        assert full[section] == lora[section]
    assert {k: v for k, v in full["optim"].items() if k not in ("lr", "min_lr")} == {
        k: v for k, v in lora["optim"].items() if k not in ("lr", "min_lr")
    }
