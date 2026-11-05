import json
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from conftest import PERIOD, VOCAB
from scratchllm.cli import app
from scratchllm.model.gpt import GPT, GPTConfig
from scratchllm.tokenizer.tokenizer import Tokenizer
from scratchllm.training.checkpoint import load_checkpoint, model_from_state, tokenizer_mismatch
from scratchllm.training.dataset import TokenWindows
from scratchllm.training.evaluation import (
    evaluate_checkpoint,
    full_loss,
    list_splits,
    token_counts,
    unigram_loss,
)
from scratchllm.training.trainer import train

CPU = torch.device("cpu")


def add_eval_only_source(directory: Path, name: str = "omega", tokens: int = 500) -> None:
    ids = (40 + np.arange(tokens) % PERIOD).astype("<u2")
    ids.tofile(directory / f"{name}.eval.bin")
    meta = {
        "source": name, "role": "evaluation-only", "tokenizer_sha256": "t" * 64,
        "vocab_size": VOCAB, "end_of_text_id": VOCAB - 1, "dtype": "uint16",
        "splits": {"eval": {"file": f"{name}.eval.bin", "docs": 5, "chars": tokens * 3,
                            "tokens": tokens, "sha256": "x"}},
    }  # fmt: skip
    (directory / f"{name}.tokens.json").write_text(json.dumps(meta), encoding="utf-8")


def tiny_model(seed: int = 0) -> GPT:
    torch.manual_seed(seed)
    return GPT(
        GPTConfig(vocab_size=VOCAB, context_length=16, d_model=32, n_layers=1, n_heads=4)
    ).eval()


# ---------- which splits ----------


def test_training_sources_give_val_and_test_and_evaluation_only_sources_give_eval(tokens_dir: Path):
    add_eval_only_source(tokens_dir)
    got = {(s.source, s.split): s.role for s in list_splits(tokens_dir)}
    assert got == {
        ("alpha", "val"): "training", ("alpha", "test"): "training",
        ("beta", "val"): "training", ("beta", "test"): "training",
        ("omega", "eval"): "evaluation-only",
    }  # fmt: skip
    only_val = {(s.source, s.split) for s in list_splits(tokens_dir, ["val"])}
    assert only_val == {("alpha", "val"), ("beta", "val"), ("omega", "eval")}


# ---------- loss ----------


def test_full_loss_is_the_mean_over_every_window(tokens_dir: Path):
    model = tiny_model()
    path = tokens_dir / "alpha.val.bin"
    got, windows = full_loss(model, path, 16, batch_size=5, device=CPU, autocast=nullcontext)
    dataset = TokenWindows([path], 16)
    assert windows == len(dataset) == (700 - 1) // 16
    with torch.no_grad():
        manual = [
            model(x[None], y[None])[1].item() for x, y in (dataset[i] for i in range(len(dataset)))
        ]
    assert got == pytest.approx(sum(manual) / len(manual), rel=1e-5)


def test_full_loss_does_not_depend_on_the_batch_size(tokens_dir: Path):
    model, path = tiny_model(), tokens_dir / "beta.test.bin"  # 100 tokens: 6 windows
    a = full_loss(model, path, 16, 1, CPU, nullcontext)[0]
    b = full_loss(model, path, 16, 4, CPU, nullcontext)[0]
    assert a == pytest.approx(b, rel=1e-5)


def test_a_file_shorter_than_one_window_is_refused(tmp_path: Path):
    np.arange(10, dtype="<u2").tofile(tmp_path / "s.bin")
    with pytest.raises(ValueError, match="fewer than"):
        full_loss(tiny_model(), tmp_path / "s.bin", 16, 4, CPU, nullcontext)


# ---------- baselines ----------


def test_the_unigram_baseline_is_the_smoothed_frequency_cross_entropy(tmp_path: Path):
    train_file = tmp_path / "train.bin"
    np.asarray([1] * 6 + [2] * 2, dtype="<u2").tofile(train_file)
    counts = token_counts([train_file], vocab_size=4)
    assert counts.tolist() == [0, 6, 2, 0]
    test_file = tmp_path / "t.bin"
    np.asarray([1, 2, 3], dtype="<u2").tofile(test_file)
    p = (counts + 1) / (8 + 4)  # add-one smoothing
    expected = -(math.log(p[1]) + math.log(p[2]) + math.log(p[3])) / 3
    assert unigram_loss(counts, test_file) == pytest.approx(expected)


def test_the_uniform_guess_costs_log_vocab_nats():
    # a model with all-equal scores predicts uniformly: loss = ln(vocab)
    model = tiny_model()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    ids = torch.randint(0, VOCAB, (2, 16))
    assert model(ids, ids)[1].item() == pytest.approx(math.log(VOCAB), rel=1e-5)


# ---------- a whole evaluation ----------


@pytest.fixture()
def checkpoint(make_cfg, tokens_dir: Path) -> Path:
    add_eval_only_source(tokens_dir)
    cfg = make_cfg("ev", train={"max_steps": 30, "eval_interval": 30, "checkpoint_interval": 30})
    train(cfg)
    return cfg.run_dir / "last.pt"


def run_cli(checkpoint: Path, tmp_path: Path, *extra: str):
    return CliRunner().invoke(
        app,
        [
            "eval",
            str(checkpoint),
            "--out-dir",
            str(tmp_path / "rep"),
            "--run-id",
            "t",
            "--device",
            "cpu",
            *extra,
        ],
    )


def test_cli_eval_reports_every_split_with_baselines(checkpoint: Path, tmp_path: Path):
    result = run_cli(checkpoint, tmp_path)
    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / "rep" / "t" / "report.json").read_text())
    assert report["step"] == 30 and report["parameters"] > 0
    rows = {(r["source"], r["split"]): r for r in report["rows"]}
    assert set(rows) == {
        ("alpha", "val"),
        ("alpha", "test"),
        ("beta", "val"),
        ("beta", "test"),
        ("omega", "eval"),
    }
    r = rows[("alpha", "val")]
    assert r["bits_per_char"] == pytest.approx(r["loss"] / math.log(2) * 700 / 2100)
    assert r["perplexity"] == pytest.approx(math.exp(r["loss"]))
    assert r["uniform_bits_per_char"] == pytest.approx(math.log2(VOCAB) * 700 / 2100)
    assert r["loss"] < 1.0  # it has learned the cycle (a uniform guess is ln 64 = 4.16)
    assert r["bits_per_char"] < r["unigram_bits_per_char"] < r["uniform_bits_per_char"]
    # a source the model never trained on (other token range) is far worse than its training sources
    assert rows[("omega", "eval")]["loss"] > r["loss"] + 2
    md = (tmp_path / "rep" / "t" / "report.md").read_text()
    assert "| omega | eval | evaluation-only |" in md and "model / unigram" in md


def test_the_same_language_unigram_counts_only_that_language(tokens_dir: Path):
    """D-026: the pooled unigram also counts the other language's tokens; the same-language one
    does not, so on a source of one language it is the stronger (lower) baseline."""
    add_eval_only_source(tokens_dir)
    specs = list_splits(tokens_dir, ["val"])
    by_source = {n: tokens_dir / f"{n}.train.bin" for n in ("alpha", "beta")}
    assert all(p.exists() for p in by_source.values())
    rows = evaluate_checkpoint(
        tiny_model(), specs, list(by_source.values()), 4, CPU, nullcontext,
        language_of={"alpha": "xx", "beta": "yy", "omega": "yy"}, train_paths_by_source=by_source,
    )  # fmt: skip
    got = {r["source"]: r for r in rows}
    alpha = got["alpha"]
    own = unigram_loss(token_counts([by_source["alpha"]], VOCAB), specs[0].path)
    assert alpha["unigram_same_language_loss"] == pytest.approx(own)
    assert alpha["unigram_same_language_sources"] == ["alpha"]
    assert alpha["unigram_same_language_loss"] < alpha["unigram_loss"]
    # an evaluation-only source is measured against the training sources of its language
    assert got["omega"]["unigram_same_language_sources"] == ["beta"]


def test_a_source_of_unknown_language_has_no_same_language_baseline(tokens_dir: Path):
    specs = list_splits(tokens_dir, ["val"])
    rows = evaluate_checkpoint(
        tiny_model(), specs, [tokens_dir / "alpha.train.bin"], 4, CPU, nullcontext
    )
    assert all(r["unigram_same_language_bits_per_char"] is None for r in rows)


def test_the_model_in_a_checkpoint_gives_the_same_predictions(checkpoint: Path):
    state = load_checkpoint(checkpoint)
    ids = torch.randint(0, 8, (2, 16))
    assert torch.equal(model_from_state(state)(ids)[0], model_from_state(state)(ids)[0])


def test_cli_eval_can_restrict_the_splits(checkpoint: Path, tmp_path: Path):
    assert run_cli(checkpoint, tmp_path, "--splits", "test").exit_code == 0
    rows = json.loads((tmp_path / "rep" / "t" / "report.json").read_text())["rows"]
    assert {(r["source"], r["split"]) for r in rows} == {
        ("alpha", "test"),
        ("beta", "test"),
        ("omega", "eval"),
    }


# ---------- the tokenizer guard ----------


def test_a_tokenizer_that_does_not_match_the_checkpoint_is_detected(
    checkpoint: Path, tmp_path: Path
):
    config = load_checkpoint(checkpoint)["config"]
    wrong_size = Tokenizer([], special_tokens=["<a>"])  # 257 ids, the model has 64
    wrong_size.save(tmp_path / "tok.json")
    assert "257 ids" in (tokenizer_mismatch(config, tmp_path / "tok.json") or "")
    config["data"]["tokens_dir"] = str(tmp_path / "missing")
    assert tokenizer_mismatch(config, tmp_path / "tok.json") is not None


def test_cli_generate_refuses_a_mismatched_tokenizer(checkpoint: Path, tmp_path: Path):
    Tokenizer([], special_tokens=["<|endoftext|>"]).save(tmp_path / "tok.json")
    result = CliRunner().invoke(
        app,
        ["generate", str(checkpoint), "--tokenizer", str(tmp_path / "tok.json"), "--device", "cpu"],
    )
    assert result.exit_code == 1 and "tokenizer" in result.output


def test_the_tokenizer_hash_is_compared_with_the_one_in_the_token_files(
    tokens_dir: Path, tmp_path: Path
):
    import hashlib

    tok = Tokenizer([], special_tokens=["<a>"])
    tok.save(tmp_path / "tok.json")
    config = {"model": {"vocab_size": tok.vocab_size}, "data": {"tokens_dir": str(tokens_dir)}}
    assert "differs" in (
        tokenizer_mismatch(config, tmp_path / "tok.json") or ""
    )  # metas say "ttt..."
    digest = hashlib.sha256((tmp_path / "tok.json").read_bytes()).hexdigest()
    for meta_path in tokens_dir.glob("*.tokens.json"):
        meta = json.loads(meta_path.read_text())
        meta["tokenizer_sha256"] = digest
        meta_path.write_text(json.dumps(meta))
    assert tokenizer_mismatch(config, tmp_path / "tok.json") is None
