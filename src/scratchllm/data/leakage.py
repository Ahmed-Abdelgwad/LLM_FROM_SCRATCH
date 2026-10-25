"""Exact-duplicate leakage between eval (val/test) and train, within and across sources.

Domain/id-based splitting cannot stop one text appearing under two domains, or in two
sources (e.g. Wikipedia text re-hosted on the web and also present in FineWeb-2). Detection
is by `text_sha256`, so it is exact only; near-duplicate leakage is measured with MinHash in
the EDA (question #9). Eval splits are never modified; leaking *train* docs are removed.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from scratchllm.data.sampling import verify_manifest

EVAL_SPLITS = ("val", "test")


def _docs(manifest_path: Path) -> Iterator[dict[str, Any]]:
    manifest = verify_manifest(manifest_path)  # refuse to analyse files that drifted
    with open(manifest_path.parent / manifest.file, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def collect_eval_hashes(manifests: list[Path]) -> set[str]:
    return {d["text_sha256"] for m in manifests for d in _docs(m) if d["split"] in EVAL_SPLITS}


def count_train_leaks(manifests: list[Path], eval_hashes: set[str]) -> dict[str, int]:
    """Train docs whose text also appears in an eval split, per source."""
    leaks: Counter[str] = Counter()
    for m in manifests:
        for d in _docs(m):
            if d["split"] == "train" and d["text_sha256"] in eval_hashes:
                leaks[d["source"]] += 1
    return dict(leaks)


def purge_train_leaks(
    manifests: list[Path], eval_hashes: set[str], out_dir: Path
) -> dict[str, dict[str, Any]]:
    """Write leak-free copies plus a receipt tying output hash to input hash. Inputs untouched."""
    out_dir.mkdir(parents=True, exist_ok=True)
    receipts: dict[str, dict[str, Any]] = {}
    for m in manifests:
        manifest = verify_manifest(m)
        final = out_dir / manifest.file
        tmp = final.with_suffix(".tmp")
        sha, kept, removed = hashlib.sha256(), 0, 0
        with open(tmp, "wb") as f:
            for d in _docs(m):
                if d["split"] == "train" and d["text_sha256"] in eval_hashes:
                    removed += 1
                    continue
                line = (json.dumps(d, ensure_ascii=False) + "\n").encode("utf-8")
                f.write(line)
                sha.update(line)
                kept += 1
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, final)
        receipt = {
            "input_manifest": str(m),
            "input_sha256": manifest.sha256,
            "output_file": final.name,
            "output_sha256": sha.hexdigest(),
            "kept": kept,
            "removed_train_leaks": removed,
        }
        (out_dir / f"{manifest.file}.purge.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
        receipts[manifest.source["name"]] = receipt
    return receipts
