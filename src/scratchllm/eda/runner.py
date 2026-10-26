"""Run EDA checks over sample files and write a reproducible report.

Only the `train` split is analysed: val/test must not influence cleaning or normalization
decisions (that is what keeps the later evaluation honest). Each manifest is verified first, so
a report can never describe a file that drifted from what was recorded.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import scratchllm
from scratchllm.data.sampling import verify_manifest
from scratchllm.eda.base import Check
from scratchllm.eda.checks_basic import LengthStats, PersonalData
from scratchllm.eda.checks_learner import NearDuplicates, ScriptMix, SurfaceVariation


def default_checks() -> list[Check]:
    return [LengthStats(), PersonalData(), ScriptMix(), SurfaceVariation(), NearDuplicates()]


def iter_train_docs(manifest_path: Path) -> Iterator[dict[str, Any]]:
    manifest = verify_manifest(manifest_path)  # raises if the data file no longer matches
    with open(manifest_path.parent / manifest.file, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            if doc["split"] == "train":
                yield doc


def analyze_source(manifest_path: Path, checks: Sequence[Check]) -> dict[str, Any]:
    """Feed the source's train documents to every check and collect their results."""
    manifest = verify_manifest(manifest_path)
    active = list(checks)
    status: dict[str, dict[str, Any]] = {}
    n_docs = 0
    for doc in iter_train_docs(manifest_path):
        n_docs += 1
        for check in list(active):
            try:
                check.update(doc)
            except NotImplementedError as exc:
                active.remove(check)
                status[check.name] = {"status": "not_implemented", "detail": str(exc)}
    for check in active:
        try:
            status[check.name] = {"status": "ok", **check.result()}
        except NotImplementedError as exc:
            status[check.name] = {"status": "not_implemented", "detail": str(exc)}
    return {
        "source": manifest.source["name"],
        "revision": manifest.revision,
        "data_sha256": manifest.sha256,
        "seed": manifest.spec["seed"],
        "train_docs_analysed": n_docs,
        "checks": {c.name: status[c.name] for c in checks},
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# EDA report {report['run_id']}",
        "",
        f"scratchllm {report['scratchllm_version']}; train split only.",
        "",
    ]
    for src in report["sources"]:
        lines += [
            f"## {src['source']}",
            "",
            f"- dataset revision: `{src['revision']}`",
            f"- data sha256: `{src['data_sha256'][:16]}`, seed {src['seed']}",
            f"- train documents analysed: {src['train_docs_analysed']:,}",
            "",
        ]
        for name, res in src["checks"].items():
            lines.append(f"### {name}")
            if res["status"] != "ok":
                lines += [f"_{res['status']}_: {res.get('detail', '')}", ""]
                continue
            for key, value in res.items():
                if key != "status":
                    lines.append(f"- {key}: {json.dumps(value, ensure_ascii=False)}")
            lines.append("")
    return "\n".join(lines)


def run_eda(
    manifests: Sequence[Path],
    out_dir: Path,
    make_checks: Callable[[], Sequence[Check]] = default_checks,
    run_id: str | None = None,
) -> Path:
    """Analyse each manifest with fresh checks and write report.json + report.md; return the dir."""
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "run_id": run_id,
        "scratchllm_version": scratchllm.__version__,
        "sources": [analyze_source(m, make_checks()) for m in manifests],
    }
    target = out_dir / run_id
    target.mkdir(parents=True, exist_ok=True)
    (target / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (target / "report.md").write_text(render_markdown(report) + "\n", encoding="utf-8")
    return target
