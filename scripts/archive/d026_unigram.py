# ruff: noqa: E501
"""Same-language unigram baselines for the evaluation reports already in git (D-026).

The reports in reports/eval/final-* were made when the only unigram baseline pooled the train
splits of both languages. A unigram baseline depends only on the token files, not on the model,
so it can be added without re-scoring any checkpoint and without rewriting those reports. This
script computes, for every row of each report, the unigram loss with the training sources of the
row's own language only, and writes reports/eval/final-d026-unigram/{report.json,report.md}.

Run from the repository root: uv run python scripts/archive/d026_unigram.py
"""

import json
import statistics
from pathlib import Path

from scratchllm.data.sources import SOURCES
from scratchllm.training.batches import bits_per_char
from scratchllm.training.evaluation import token_counts, unigram_loss

ROOT = Path(__file__).resolve().parents[2]  # scripts/archive/ -> the repository root
REPORTS = ["final-pilot-17m", "final-local-17m", "final-ablation-english-20-long"]
OUT = ROOT / "reports/eval/final-d026-unigram"
TRAINED_AR = ["fw2_arb", "fw2_ars", "fw2_ary", "fw2_arz", "wiki_ar"]
UNSEEN = ["fw2_acm", "fw2_apc", "wiki_ary"]


def main() -> None:
    first = json.loads((ROOT / f"reports/eval/{REPORTS[0]}/report.json").read_text())
    tokens_dir = ROOT / first["tokens_dir"]
    metas = [
        json.loads(m.read_text(encoding="utf-8")) for m in sorted(tokens_dir.glob("*.tokens.json"))
    ]
    vocab = metas[0]["vocab_size"]
    train = {
        m["source"]: tokens_dir / m["splits"]["train"]["file"]
        for m in metas
        if m["role"] == "training"
    }
    langs = sorted({SOURCES[s].language for s in train})
    by_lang = {lang: sorted(s for s in train if SOURCES[s].language == lang) for lang in langs}
    counts = {
        lang: token_counts([train[s] for s in names], vocab) for lang, names in by_lang.items()
    }
    files = {
        (m["source"], split): tokens_dir / info["file"]
        for m in metas
        for split, info in m["splits"].items()
    }

    same: dict[tuple[str, str], float] = {}
    out: dict = {"tokens_dir": first["tokens_dir"], "languages": by_lang, "reports": {}}
    lines = [
        "# Same-language unigram baselines (D-026)",
        "",
        "Bits per character. *pooled unigram* is the baseline in the original reports (train splits of both",
        "languages); *same-language unigram* counts only the training sources of the row's language.",
        "",
    ]
    for name in REPORTS:
        report = json.loads((ROOT / f"reports/eval/{name}/report.json").read_text())
        assert report["tokens_dir"] == first["tokens_dir"]
        rows = []
        lines += [f"## {name} (step {report['step']})", "",
                  "| source | split | model | pooled unigram | model / pooled | same-language unigram | model / same-language |",
                  "|---|---|---:|---:|---:|---:|---:|"]  # fmt: skip
        for r in report["rows"]:
            key = (r["source"], r["split"])
            if key not in same:
                loss = unigram_loss(counts[SOURCES[r["source"]].language], files[key])
                same[key] = bits_per_char(loss, r["tokens"], r["chars"])
            s = same[key]
            rows.append({"source": r["source"], "split": r["split"], "bits_per_char": r["bits_per_char"],
                         "unigram_bits_per_char": r["unigram_bits_per_char"],
                         "unigram_same_language_bits_per_char": s})  # fmt: skip
            lines.append(
                f"| {r['source']} | {r['split']} | {r['bits_per_char']:.3f} | {r['unigram_bits_per_char']:.3f} "
                f"| {r['bits_per_char'] / r['unigram_bits_per_char']:.2f} | {s:.3f} | {r['bits_per_char'] / s:.2f} |"
            )
        get = {(r["source"], r["split"]): r for r in rows}
        summary = {}
        for label, keys in [("trained Arabic, val", [(s, "val") for s in TRAINED_AR]),
                            ("unseen dialects", [(s, "eval") for s in UNSEEN]),
                            ("English, val", [("fineweb_edu_10bt", "val")])]:  # fmt: skip
            model = statistics.mean(get[k]["bits_per_char"] for k in keys)
            pooled = statistics.mean(get[k]["unigram_bits_per_char"] for k in keys)
            own = statistics.mean(get[k]["unigram_same_language_bits_per_char"] for k in keys)
            summary[label] = {"model": model, "pooled_unigram": pooled, "same_language_unigram": own,
                              "model_over_pooled": model / pooled, "model_over_same_language": model / own}  # fmt: skip
        lines += ["", "| group (mean) | model | pooled unigram | model / pooled | same-language unigram | model / same-language |",
                  "|---|---:|---:|---:|---:|---:|"]  # fmt: skip
        for label, v in summary.items():
            lines.append(
                f"| {label} | {v['model']:.3f} | {v['pooled_unigram']:.3f} | {v['model_over_pooled']:.2f} "
                f"| {v['same_language_unigram']:.3f} | {v['model_over_same_language']:.2f} |"
            )
        lines.append("")
        out["reports"][name] = {"step": report["step"], "rows": rows, "summary": summary}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
