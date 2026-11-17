"""The blind human rubric (finetune/rubric.py, D-041)."""

import csv
import dataclasses
import json

import pytest
import torch

from scratchllm.finetune.rubric import (
    FIELDS,
    make_sheet,
    read_scores,
    render_html,
    render_summary,
    score_sheet,
    summarize,
    write_sheet,
)
from scratchllm.training.checkpoint import load_model
from test_sft_trainer import setup  # noqa: F401  (the tiny model, tokenizer and data)

COUNTS = {"msa": 2, "english": 2}


def sheet(setup, tmp_path):  # noqa: F811
    make, tok, base = setup
    cfg = make()
    models = {"base": load_model(base), "other": load_model(base)}
    items, key = make_sheet(
        models, tok, cfg.data.data_dir, torch.device("cpu"), lambda: torch.no_grad(),
        counts=COUNTS, max_new_tokens=6,
    )  # fmt: skip
    write_sheet(items, key, {"note": "test"}, tmp_path / "sheet")
    return items, key, tmp_path / "sheet"


def fill(path, score):
    """Fill scores.csv: `score(model letter row) -> (overall, fails, incorrect)`."""
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            r["overall"], r["fails_instruction"], r["incorrect"] = score(r)
            w.writerow(r)


def test_the_sheet_is_blind_complete_and_keyed(setup, tmp_path):  # noqa: F811
    items, key, out = sheet(setup, tmp_path)
    assert len(items) == 4 and {i.variety for i in items} == {"msa", "english"}
    assert all(set(i.replies) == {"A", "B"} for i in items)
    assert {frozenset(k["letters"].values()) for k in key["prompts"].values()} == {
        frozenset({"base", "other"})
    }
    page = (out / "sheet.html").read_text(encoding="utf-8")
    assert "base" not in page and "other" not in page  # no model name anywhere on the page
    hostile = dataclasses.replace(items[0], replies={"A": "x</script><b>", "B": ""})
    block = render_html([hostile]).split('type="application/json">')[1].split("</script>")[0]
    assert json.loads(block)[0]["replies"]["A"] == "x</script><b>"  # cannot end the data early
    with (out / "scores.csv").open(encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert tuple(header[: len(FIELDS)]) == FIELDS
    assert json.loads((out / "key.json").read_text())["meta"] == {"note": "test"}


def test_scores_are_unblinded_to_the_right_model(setup, tmp_path):  # noqa: F811
    _, key, out = sheet(setup, tmp_path)
    letters = {n: k["letters"] for n, k in key["prompts"].items()}

    def score(r):  # "other" always scores 6, "base" 2; only "base" fails the instruction
        model = letters[r["prompt_no"]][r["reply"]]
        return ("6", "0", "0") if model == "other" else ("2", "1", "")

    fill(out / "scores.csv", score)
    s = score_sheet(out)
    assert s["per_model"]["other"]["all"]["overall"] == 6
    assert s["per_model"]["base"]["all"]["overall"] == 2
    assert s["per_model"]["base"]["all"]["fails_instruction_rate"] == 1
    assert s["per_model"]["base"]["all"]["incorrect_rate"] is None
    assert s["paired"]["base - other"] == {
        "overall_diff": -4, "ci95": [-4, -4], "wins": 0, "ties": 0, "losses": 4,
    }  # fmt: skip
    assert (out / "summary.md").exists()


def test_incomplete_or_out_of_range_scores_are_refused(setup, tmp_path):  # noqa: F811
    _, key, out = sheet(setup, tmp_path)
    with pytest.raises(ValueError, match="missing"):
        read_scores(out / "scores.csv")  # the blank template
    fill(out / "scores.csv", lambda r: ("8", "0", ""))
    with pytest.raises(ValueError, match="1-7"):
        read_scores(out / "scores.csv")
    rows = [{"prompt_no": "1", "reply": "A", "overall": 3, "fails_instruction": 0, "incorrect": 0}]
    with pytest.raises(ValueError, match="do not match"):
        summarize(rows, key)


def test_the_summary_names_the_denominator_of_each_rate():
    """An empty "incorrect" label is not counted: the rate is over the replies judged, and the
    table says how many those were (D-046: LoRA had 29 of 30 judged)."""

    def group(n, judged, rate):
        return {"n": n, "overall": 1.0, "overall_ci95": [1.0, 1.0], "fails_instruction_rate": 1.0,
                "incorrect_rate": rate, "incorrect_judged": judged}  # fmt: skip

    summary = {
        "prompts": 30,
        "per_model": {"m": {"all": group(30, 29, 1.0), "some": group(12, 11, 10 / 11),
                            "none": group(6, 0, None), "full": group(30, 30, 0.5)}},
        "paired": {},
    }  # fmt: skip
    text = render_summary(summary)
    assert "| 100% (30/30) | 100% (29/29 judged; 1 not judged) |" in text
    assert "| 91% (10/11 judged; 1 not judged) |" in text
    assert "| none judged (0/6) |" in text
    assert "| 50% (15/30 judged) |" in text
