import json
from collections import Counter
from pathlib import Path

import pytest
from typer.testing import CliRunner

from scratchllm.cli import app
from scratchllm.data.sampling import SampleSpec, manifest_path, text_hash, write_sample
from scratchllm.data.sources import get_source
from scratchllm.eda.checks_basic import LengthStats, PersonalData
from scratchllm.eda.checks_learner import NearDuplicates
from scratchllm.eda.runner import analyze_source, iter_train_docs, run_eda

SRC = get_source("fw2_arz")


def doc(i: int, text: str, split: str = "train") -> dict:
    return {
        "id": str(i),
        "source": SRC.name,
        "url": None,
        "split": split,
        "text_sha256": text_hash(text),
        "text": text,
    }


def feed(check, *texts: str) -> dict:
    for i, t in enumerate(texts):
        check.update(doc(i, t))
    return check.result()


# ---------- LengthStats ----------


def test_length_stats_on_known_strings():
    r = feed(LengthStats(short_chars=10), "a" * 5, "b" * 10, "c" * 100, "word " * 4)
    assert r["docs"] == 4 and r["chars_total"] == 5 + 10 + 100 + 20
    assert r["words_total"] == 1 + 1 + 1 + 4
    assert r["chars_max"] == 100
    assert r["share_shorter_than"] == {"10": 0.25}  # only the 5-char document is under 10


def test_length_stats_empty():
    assert LengthStats().result() == {"docs": 0}


# ---------- PersonalData ----------


@pytest.mark.parametrize(
    "text,emails,phones",
    [
        ("contact ahmed.k+test@example.co.uk now", 1, 0),
        ("اتصل على 01102106655 الآن", 0, 1),
        ("اتصل على ٠١١٠٢١٠٦٦٥٥ الآن", 0, 1),  # Arabic-Indic digits
        ("call +20 110 210 6655 today", 0, 1),
        ("السعر 1500 جنيه و 2019 سنة", 0, 0),  # short numbers are not phones
        ("نشر في 2019-05-12 وعدّل 12/05/2019", 0, 0),  # dates are not phones
        # false positives seen on real samples (wiki_ar / fw2_arz) with the first regex:
        ("بوردو (2002-2003) ويوفنتوس (1930-1935)", 0, 0),  # year ranges
        ("مطابخ 2018 2019 2019 جديدة", 0, 0),  # year lists separated by spaces
        ("الأحد 24 11 2019 وموعد", 0, 0),  # date written with spaces
        ("صورة 1014798595_n.jpg", 0, 0),  # image id inside a file name
        ("رمز المنطقة هو 100120413", 0, 0),  # a long code with no phone shape
        ("ISBN 978-3-16-148410-0", 0, 0),  # hyphens between short groups
        ("ISBN 978-0198150978", 0, 0),  # 10 digits after a hyphen
        ("ارتفاع 32.004000000000005 متر", 0, 0),  # long decimals
        ("الرابط mw0000750042 و mt0041966747", 0, 0),  # catalogue ids after Latin letters
        # real shapes seen on the same samples
        ("الخط الساخن /01225854916 يوميا", 0, 1),
        ("تليفون: +20 112 074 0067", 0, 1),
        ("روحاني0096176817976 سلام", 0, 1),  # international prefix, glued to Arabic letters
        ("اتصل 010-1234-5678 أو 010 1234 5678", 0, 2),
        ("call (555) 123-4567 or 555.123.4567", 0, 2),
        ("لا شيء هنا", 0, 0),
    ],
)
def test_personal_data_detection(text: str, emails: int, phones: int):
    r = feed(PersonalData(), text)
    assert (r["email_matches"], r["phone_like_matches"]) == (emails, phones)


def test_a_few_spam_pages_inflate_matches_but_not_the_document_rate():
    pd = PersonalData()
    spam = "افضل شيخ روحاني 0096176817976 " * 50  # one page repeating a number 50 times
    for i in range(1000):
        pd.update(doc(i, spam if i < 2 else "نص عادي"))
    r = pd.result()
    assert r["phone_like_matches"] == 100 and r["hits_per_1000_docs"] == 100.0  # inflated
    assert r["docs_with_pii"] == 2 and r["docs_with_pii_per_1000"] == 2.0  # the honest figure


def test_placeholder_emails_are_reported_apart_and_are_not_personal_data():
    r = feed(
        PersonalData(),
        "write to email@example.com or firstname.lastname@example.org",
        "write to real.person@gmail.com",
        "no contact here",
    )
    assert r["email_placeholder_matches"] == 2
    assert r["email_matches"] == 1 and r["docs_with_pii"] == 1


def test_personal_data_rate_is_per_thousand_documents():
    pd = PersonalData()
    for i in range(1000):
        pd.update(doc(i, "اتصل 01102106655" if i < 5 else "نص عادي"))
    assert pd.result()["hits_per_1000_docs"] == 5.0


# ---------- runner ----------


def write_fixture(tmp_path: Path) -> Path:
    docs = [doc(i, f"train doc {i} " + "word " * 20) for i in range(6)]
    docs += [doc(100, "validation doc " + "word " * 20, "val")]
    docs += [doc(200, "test doc " + "word " * 20, "test")]
    out = tmp_path / "s"
    write_sample(iter(docs), out, SRC, SampleSpec(n_docs=len(docs)), "rev0", Counter())
    return manifest_path(out, SRC.name)


def test_only_the_train_split_is_analysed(tmp_path: Path):
    m = write_fixture(tmp_path)
    assert len(list(iter_train_docs(m))) == 6
    result = analyze_source(m, [LengthStats()])
    assert result["train_docs_analysed"] == 6
    assert result["checks"]["length"]["docs"] == 6  # the val and test documents were not seen


class NotYetWritten:
    """A stand-in for a check that is still a contract (kept local so this test does not
    depend on which real checks happen to be implemented)."""

    name = "not_yet_written"

    def update(self, doc: dict) -> None:
        raise NotImplementedError("still to write")

    def result(self) -> dict:
        raise NotImplementedError("still to write")


def test_unimplemented_checks_are_reported_not_fatal(tmp_path: Path):
    result = analyze_source(write_fixture(tmp_path), [LengthStats(), NotYetWritten()])
    assert result["checks"]["length"]["status"] == "ok"
    assert result["checks"]["not_yet_written"]["status"] == "not_implemented"


def test_report_files_record_provenance(tmp_path: Path):
    m = write_fixture(tmp_path)
    out = run_eda([m], tmp_path / "reports", make_checks=lambda: [LengthStats()], run_id="r1")
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    src = report["sources"][0]
    assert report["run_id"] == "r1" and src["revision"] == "rev0" and len(src["data_sha256"]) == 64
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "## fw2_arz" in md and "### length" in md and "chars_median" in md


def cluster_examples(report_dir: Path) -> list[dict]:
    report = json.loads((report_dir / "report.json").read_text(encoding="utf-8"))
    return report["sources"][0]["checks"]["near_duplicates"]["largest_cluster_examples"]


def test_tracked_reports_carry_no_raw_text_but_local_reports_do(tmp_path: Path):
    base = " ".join(f"w{i}" for i in range(60))  # 60 distinct words, so 56 shared shingles
    docs = [doc(i, f"template page {i} " + base) for i in range(4)]
    out = tmp_path / "s"
    write_sample(iter(docs), out, SRC, SampleSpec(n_docs=len(docs)), "rev0", Counter())
    m = manifest_path(out, SRC.name)
    local = run_eda([m], tmp_path / "r", make_checks=lambda: [NearDuplicates()], run_id="local")
    tracked = run_eda([m], tmp_path / "r", make_checks=lambda: [NearDuplicates()], run_id="final-x")
    kept, dropped = cluster_examples(local), cluster_examples(tracked)
    assert kept and kept[0]["first_text"].startswith("template page")
    assert dropped and "first_text" not in dropped[0]
    assert dropped[0]["size"] == kept[0]["size"] and dropped[0]["ids"] == kept[0]["ids"]
    assert "template page" not in (tracked / "report.json").read_text(encoding="utf-8")
    assert "template page" not in (tracked / "report.md").read_text(encoding="utf-8")


def test_a_sample_that_drifted_from_its_manifest_is_refused(tmp_path: Path):
    m = write_fixture(tmp_path)
    data = m.parent / "fw2_arz.jsonl"
    data.write_text(data.read_text(encoding="utf-8").replace("train doc 0", "train doc X"), "utf-8")
    with pytest.raises(ValueError, match="sha256"):
        analyze_source(m, [LengthStats()])


def test_cli_exposes_the_analyze_command():
    result = CliRunner().invoke(app, ["data", "analyze", "--help"])
    assert result.exit_code == 0 and "report.json" in result.output


# ---------- progress, timing and the personal-data speed-ups ----------


def test_progress_lines_and_per_check_seconds(tmp_path: Path, monkeypatch):
    import scratchllm.eda.runner as runner

    monkeypatch.setattr(runner, "PROGRESS_EVERY", 2)  # the fixture has 6 train docs
    lines: list[str] = []
    result = analyze_source(write_fixture(tmp_path), [LengthStats()], progress=lines.append)
    assert lines[0].startswith("fw2_arz: verified")
    assert [ln for ln in lines if ln.endswith("documents")] == [
        "fw2_arz: 2 documents",
        "fw2_arz: 4 documents",
        "fw2_arz: 6 documents",
    ]
    assert "length " in lines[-1] and lines[-1].startswith("fw2_arz: done, 6 documents")
    assert result["checks"]["length"]["seconds"] >= 0


def _reference_personal_data(text: str) -> tuple[int, int]:
    """The straightforward version: always translate digits, always run both patterns."""
    from scratchllm.data import pii

    t = text.translate(pii.DIGITS)
    return len(pii.EMAIL.findall(t)), len(pii.PHONE.findall(pii.DATE.sub(" ", t)))


@pytest.mark.parametrize(
    "text",
    [
        "no digits and no at-sign",
        "mail a@b.co and 01102106655",
        "اتصل ٠١١٠٢١٠٦٦٥٥ أو on@x.org",  # Arabic-Indic digits and an email in one document
        "٠١١٠٢١٠٦٦٥٥ فقط",
        "۰۱۱۰۲۱۰۶۶۵۵ بالأرقام الفارسية",  # Extended Arabic-Indic digits
        "ahmed@ with nothing after it",
        "2002-2003 و 24 11 2019 و +20 112 074 0067",
        "",
    ],
)
def test_speedups_give_exactly_the_reference_counts(text: str):
    pd = PersonalData()
    pd.update(doc(0, text))
    r = pd.result()
    assert (r["email_matches"], r["phone_like_matches"]) == _reference_personal_data(text)


def test_cli_analyze_names_the_report_folder(tmp_path: Path):
    m = write_fixture(tmp_path)
    out = tmp_path / "reports"
    result = CliRunner().invoke(
        app, ["data", "analyze", str(m), "--out-dir", str(out), "--run-id", "final-demo"]
    )
    assert result.exit_code == 0, result.output
    assert (out / "final-demo" / "report.json").exists()
    assert "fw2_arz: done, 6 documents" in result.output  # progress lines reach the user
