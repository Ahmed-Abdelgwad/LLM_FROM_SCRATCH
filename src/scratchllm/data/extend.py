"""More training text for a source without touching its evaluation side (D-049).

The first sample of each source (`data/raw/eda`) fixed the validation and test sets, and every
reported number since is measured on them. Adding training text must not change them, so the
extension is built in two steps, each checked:

1. `build_extension`: from a second draw of the same source (same dataset revision, same split
   seed, so a site falls in the same split in both draws), keep only documents that
   - the split rule sends to train (a document of a validation or test site is dropped, never
     added to an evaluation set, so no site crosses the boundary),
   - are not in the first sample (any split), so the new text is new,
   - do not duplicate any evaluation document exactly (D-005), across all sources; every document
     of an evaluation-only source counts as evaluation text, whatever label the sampler gave it.
   The result is a sample file with its own manifest, which `data tokenize` reads as usual.
2. `compose_tokens`: a token directory whose train file per source is the first sample's train
   tokens followed by the extension's; every other file (val, test, eval) is a byte-identical
   copy, checked by sha256 against the base description.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from scratchllm.data.sampling import SampleSpec, verify_manifest, write_sample
from scratchllm.data.sources import Source
from scratchllm.data.tokens import verify_tokens
from scratchllm.eda.leakage import EVAL_ONLY_SOURCES

EVAL_SPLITS = ("val", "test")


def _docs(manifest_path: Path) -> Iterator[dict[str, Any]]:
    manifest = verify_manifest(manifest_path)
    with open(manifest_path.parent / manifest.file, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def evaluation_hashes(manifests: list[Path]) -> set[str]:
    """Text hashes of every evaluation document: val and test of the training sources, and all
    documents of the evaluation-only sources."""
    out: set[str] = set()
    for m in manifests:
        for d in _docs(m):
            if d["source"] in EVAL_ONLY_SOURCES or d["split"] in EVAL_SPLITS:
                out.add(d["text_sha256"])
    return out


def _spec(manifest: Any) -> SampleSpec:
    return SampleSpec(**manifest.spec)


def build_extension(
    base_manifest: Path, draw_manifest: Path, eval_hashes: set[str], out_dir: Path
) -> dict[str, Any]:
    """Write the new training documents of a second draw to `out_dir`; returns the counts."""
    base, draw = verify_manifest(base_manifest), verify_manifest(draw_manifest)
    name = base.source["name"]
    if name in EVAL_ONLY_SOURCES:
        raise ValueError(f"{name} is an evaluation-only source; it gets no training text")
    if draw.source != base.source:
        raise ValueError(f"the draw is of {draw.source['name']}, not {name}")
    if draw.revision != base.revision:
        raise ValueError(f"{name}: draw revision {draw.revision} differs from {base.revision}")
    b, d = _spec(base), _spec(draw)
    if d.effective_split_seed != b.effective_split_seed:
        raise ValueError(
            f"{name}: split seed {d.effective_split_seed} differs from the base's "
            f"{b.effective_split_seed}; sites would move between train and evaluation"
        )
    if (d.val_frac, d.test_frac) != (b.val_frac, b.test_frac):
        raise ValueError(f"{name}: split fractions differ from the base's")
    if d.seed == b.seed:
        raise ValueError(f"{name}: the draw seed equals the base's; the draw would repeat it")
    in_base = {doc["text_sha256"] for doc in _docs(base_manifest)}

    def reason(doc: dict[str, Any]) -> str:
        if doc["split"] != "train":
            return f"dropped_{doc['split']}_site"
        if doc["text_sha256"] in in_base:
            return "dropped_in_base_sample"
        if doc["text_sha256"] in eval_hashes:
            return "dropped_eval_duplicate"
        return "kept"

    # two passes over the draw: count first, so the manifest's n_docs is exactly what is written
    counts = Counter(reason(doc) for doc in _docs(draw_manifest))
    counts["drawn"] = draw.n_docs
    kept = (doc for doc in _docs(draw_manifest) if reason(doc) == "kept")
    stats: Counter[str] = Counter({f"draw_{k}": v for k, v in draw.stats.items()})
    spec = SampleSpec(**{**draw.spec, "n_docs": counts["kept"]})
    manifest = write_sample(kept, out_dir, Source(**draw.source), spec, draw.revision, stats)
    if manifest.n_docs != counts["kept"]:
        raise RuntimeError(f"{name}: wrote {manifest.n_docs} documents, counted {counts['kept']}")
    receipt = {
        "source": name,
        "base_manifest": str(base_manifest),
        "base_sha256": base.sha256,
        "draw_manifest": str(draw_manifest),
        "draw_sha256": draw.sha256,
        "output_sha256": manifest.sha256,
        **dict(counts),
    }
    (out_dir / f"{name}.extension.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 22):
            digest.update(chunk)
    return digest.hexdigest()


def compose_tokens(base_dir: Path, ext_dir: Path, out_dir: Path) -> dict[str, dict[str, Any]]:
    """Build `out_dir` from `base_dir` with each extended source's train file lengthened by the
    extension's train tokens. Returns the written descriptions, by source."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, dict[str, Any]] = {}
    for meta_path in sorted(base_dir.glob("*.tokens.json")):
        base = verify_tokens(meta_path)
        name = base["source"]
        ext_path = ext_dir / meta_path.name
        meta = json.loads(json.dumps(base))
        for split, info in base["splits"].items():
            if split == "train" and ext_path.exists():
                continue
            target = out_dir / info["file"]
            shutil.copyfile(base_dir / info["file"], target)
            if _sha256(target) != info["sha256"]:
                raise ValueError(f"{target.name}: copy differs from {base_dir / info['file']}")
        if ext_path.exists():
            ext = verify_tokens(ext_path)
            for key in ("tokenizer_sha256", "vocab_size", "end_of_text_id", "dtype"):
                if ext[key] != base[key]:
                    raise ValueError(f"{name}: extension {key} differs from the base's")
            if ext["cleaning_rules"] != base["cleaning_rules"]:
                raise ValueError(f"{name}: extension cleaning rules differ from the base's")
            stray = {s: i["docs"] for s, i in ext["splits"].items() if s != "train" and i["docs"]}
            if stray:
                raise ValueError(f"{name}: extension has evaluation documents {stray}")
            old, new = base["splits"]["train"], ext["splits"]["train"]
            target = out_dir / old["file"]
            with target.open("wb") as out:
                for part in (base_dir / old["file"], ext_dir / new["file"]):
                    with part.open("rb") as f:
                        shutil.copyfileobj(f, out, 1 << 22)
            meta["splits"]["train"] = {
                "file": old["file"],
                "docs": old["docs"] + new["docs"],
                "chars": old["chars"] + new["chars"],
                "tokens": old["tokens"] + new["tokens"],
                "sha256": _sha256(target),
            }
            meta["extension"] = {
                "base_train_sha256": old["sha256"],
                "extension_train_sha256": new["sha256"],
                "extension_input_manifest_sha256": ext["input_manifest_sha256"],
            }
        (out_dir / meta_path.name).write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        verify_tokens(out_dir / meta_path.name)
        written[name] = meta
    return written
