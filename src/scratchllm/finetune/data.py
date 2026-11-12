"""The fine-tuning data (D-038): which pairs are used, how they are cleaned and split.

Sources, pinned to a revision:

* Aya Dataset (Singh et al. 2024), human-written: Standard Arabic (`arb`), Egyptian (`arz`) and
  English (`eng`). Its Moroccan subset is not used: half of its responses are under 20 characters
  and some are misaligned with their prompt (D-038). Its other Arabic subsets are under 140 rows
  each and are left out as well.
* CIDAR (Alyafeai et al. 2024), 10,000 Modern Standard Arabic pairs reviewed by people; it takes
  the place of the Moroccan subset.

Rules, applied in this order and counted at each step:

1. A pair with an empty prompt or response is dropped.
2. Script: the response of an Arabic pair must be at least half Arabic letters, and that of an
   English pair at least half Latin letters (letters only are counted). This drops code, SQL and
   answers in the other language, which a model this small should not be asked to imitate.
3. Test set: the official Aya test split (`arb` and `eng`, 250 each), plus every pair whose
   prompt key falls in the first 5% of the hash range (see below), from every source.
4. Leakage: a training or validation pair whose prompt key equals that of an official test pair is
   dropped. The key (`prompt_key`) is looser than the project's normalization on purpose: it also
   ignores diacritics, punctuation, case and the alef/yeh/teh-marbuta variants, because "غزوه احد"
   and "غزوة أحد" are the same question to a reader (D-038 counts 87 overlaps under the project's
   normalization and 96 under this key).
5. Exact duplicates (same normalized prompt and response) are kept once.
6. Split by prompt key: sha256 of the key mapped to [0, 1); below 0.05 test, below 0.10
   validation, the rest training. All pairs that share a prompt (Aya's re-annotations, a CIDAR
   prompt also in Aya) land in the same split, so no prompt is seen in training and scored in
   validation or test.

Pairs that do not fit the 512-token context are dropped later, when they are encoded
(`chat.encode_example`), and counted there.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from scratchllm.tokenizer.normalize import normalize


@dataclass(frozen=True)
class RawFile:
    repo: str
    revision: str
    filename: str


AYA_REPO, AYA_REVISION = "CohereLabs/aya_dataset", "f9ea04583f02a8f86404ff6c58bf75fe637df8a2"
CIDAR_REPO, CIDAR_REVISION = "arbml/CIDAR", "bc2f9d7a9de34b534b126c32b8cce98e098138df"
RAW_FILES = {
    "aya_train": RawFile(AYA_REPO, AYA_REVISION, "data/train-00000-of-00001.parquet"),
    "aya_test": RawFile(AYA_REPO, AYA_REVISION, "data/test-00000-of-00001.parquet"),
    "cidar": RawFile(
        CIDAR_REPO, CIDAR_REVISION, "data/train-00000-of-00001-b2881e1b9f14c3b1.parquet"
    ),
}
AYA_VARIETIES = {"arb": "msa", "arz": "egyptian", "eng": "english"}
LANGUAGE = {"msa": "ar", "egyptian": "ar", "english": "en"}
TEST_FRACTION, VAL_FRACTION = 0.05, 0.05
MIN_SCRIPT_SHARE = 0.5

_ARABIC = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
_LATIN = re.compile(r"[A-Za-zÀ-ɏ]")
_DIACRITICS = re.compile(r"[ً-ْٰـ]")
_PUNCTUATION = re.compile(r"[^\w\s]")


@dataclass(frozen=True)
class Pair:
    id: str
    source: str  # aya | cidar
    variety: str  # msa | egyptian | english
    prompt: str
    response: str
    split: str = ""  # train | val | test, set by `build`

    @property
    def language(self) -> str:
        return LANGUAGE[self.variety]


def prompt_key(text: str) -> str:
    """What makes two prompts "the same question": the project's normalization, then no
    diacritics or tatweel, one alef, yeh for alef maqsura, heh for teh marbuta, no punctuation,
    lower case, single spaces."""
    t = _DIACRITICS.sub("", normalize(text)).lower()
    t = re.sub("[إأآٱ]", "ا", t).replace("ى", "ي").replace("ة", "ه")
    return " ".join(_PUNCTUATION.sub(" ", t).split())


def script_share(text: str, language: str) -> float:
    """Share of the letters of `text` written in the script of `language` (0 if no letters)."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    pattern = _ARABIC if language == "ar" else _LATIN
    return sum(bool(pattern.match(ch)) for ch in letters) / len(letters)


def hash_unit(key: str) -> float:
    """The key's place in [0, 1), from sha256: stable across machines, runs and Python versions."""
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") / 2**64


def bucket(key: str) -> str:
    u = hash_unit(key)
    return "test" if u < TEST_FRACTION else "val" if u < TEST_FRACTION + VAL_FRACTION else "train"


# ----------------------------------------------------------------------------- reading


def download(raw_dir: Path) -> dict[str, Path]:
    """Fetch the pinned files (once; later calls reuse them)."""
    from huggingface_hub import hf_hub_download

    paths = {}
    for name, f in RAW_FILES.items():
        paths[name] = Path(
            hf_hub_download(
                f.repo,
                f.filename,
                repo_type="dataset",
                revision=f.revision,
                local_dir=raw_dir / f.repo.replace("/", "__"),
            )  # fmt: skip
        )
    return paths


def local_paths(raw_dir: Path) -> dict[str, Path]:
    return {n: raw_dir / f.repo.replace("/", "__") / f.filename for n, f in RAW_FILES.items()}


def read_aya(path: Path, split_name: str) -> list[Pair]:
    rows = pq.read_table(path, columns=["inputs", "targets", "language_code"]).to_pylist()
    return [
        Pair(f"aya-{split_name}-{i}", "aya", AYA_VARIETIES[r["language_code"]], r["inputs"],
             r["targets"])
        for i, r in enumerate(rows)
        if r["language_code"] in AYA_VARIETIES
    ]  # fmt: skip


def read_cidar(path: Path) -> list[Pair]:
    rows = pq.read_table(path, columns=["index", "instruction", "output"]).to_pylist()
    return [
        Pair(f"cidar-{r['index']}", "cidar", "msa", r["instruction"], r["output"]) for r in rows
    ]


# ----------------------------------------------------------------------------- the rules


def _count(pairs: Iterable[Pair]) -> dict[str, int]:
    return dict(sorted(Counter(f"{p.source}/{p.variety}" for p in pairs).items()))


def build(official_test: Sequence[Pair], pool: Sequence[Pair]) -> tuple[list[Pair], dict[str, Any]]:
    """Apply the rules to the official test pairs and the rest. Returns every kept pair with its
    split, and the counts after each rule."""
    stages: dict[str, Any] = {
        "read": {"official_test": _count(official_test), "pool": _count(pool)}
    }

    def keep(pairs: Sequence[Pair]) -> tuple[list[Pair], list[Pair]]:
        nonempty = [p for p in pairs if p.prompt.strip() and p.response.strip()]
        right_script = [
            p for p in nonempty if script_share(p.response, p.language) >= MIN_SCRIPT_SHARE
        ]
        return nonempty, right_script

    test_nonempty, test_kept = keep(official_test)
    pool_nonempty, pool_kept = keep(pool)
    stages["1_nonempty"] = {"official_test": _count(test_nonempty), "pool": _count(pool_nonempty)}
    stages["2_script"] = {"official_test": _count(test_kept), "pool": _count(pool_kept)}

    test_keys = {prompt_key(p.prompt) for p in test_kept}
    leaked = [p for p in pool_kept if prompt_key(p.prompt) in test_keys]
    clean = [p for p in pool_kept if prompt_key(p.prompt) not in test_keys]
    stages["4_leakage_removed"] = _count(leaked)

    seen: set[tuple[str, str]] = set()
    unique: list[Pair] = []
    for p in [*test_kept, *clean]:
        identity = (normalize(p.prompt), normalize(p.response))
        if identity not in seen:
            seen.add(identity)
            unique.append(p)
    official_ids = {p.id for p in test_kept}
    stages["5_after_dedup"] = _count(unique)

    out = []
    for p in unique:
        split = "test" if p.id in official_ids else bucket(prompt_key(p.prompt))
        out.append(Pair(**{**asdict(p), "split": split}))
    by_split: dict[str, dict[str, int]] = defaultdict(dict)
    for split in ("train", "val", "test"):
        by_split[split] = _count(p for p in out if p.split == split)
    by_split["test_official"] = _count(p for p in out if p.id in official_ids)
    stages["6_split"] = dict(by_split)
    return out, stages


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(raw_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Read the pinned files, apply the rules, write `train/val/test.jsonl` and `manifest.json`."""
    paths = local_paths(raw_dir)
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing raw files (run with --download): {missing}")
    official = [p for p in read_aya(paths["aya_test"], "test") if p.variety in ("msa", "english")]
    pool = read_aya(paths["aya_train"], "train") + read_cidar(paths["cidar"])
    pairs, stages = build(official, pool)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for split in ("train", "val", "test"):
        path = out_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for p in pairs:
                if p.split == split:
                    f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
        files[split] = {"file": path.name, "sha256": sha256_file(path)}
    manifest = {
        "inputs": {n: {**asdict(f), "sha256": sha256_file(paths[n])} for n, f in RAW_FILES.items()},
        "rules": {
            "min_script_share": MIN_SCRIPT_SHARE,
            "test_fraction": TEST_FRACTION,
            "val_fraction": VAL_FRACTION,
            "aya_varieties": AYA_VARIETIES,
        },
        "stages": stages,
        "files": files,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def read_split(data_dir: Path, split: str) -> list[Pair]:
    lines = (data_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
    return [Pair(**json.loads(line)) for line in lines if line]


# ----------------------------------------------------------------------------- batches


class PairSampler:
    """Training batches as a pure function of (seed, step, micro-batch), like `TrainSampler`:
    each sequence first picks a language with the given weight, then a pair of that language
    uniformly. So a resumed run continues with exactly the batches it would have seen."""

    def __init__(
        self,
        examples_by_language: Mapping[str, Sequence[Any]],
        weights: Mapping[str, float],
        micro_batch_size: int,
        seed: int,
    ) -> None:
        self.languages = sorted(weights)
        missing = [lang for lang in self.languages if not examples_by_language.get(lang)]
        if missing:
            raise ValueError(f"no training examples for {missing}")
        total = sum(weights[lang] for lang in self.languages)
        self.probabilities = np.asarray([weights[lang] / total for lang in self.languages])
        self.pools = [list(examples_by_language[lang]) for lang in self.languages]
        self.micro_batch_size, self.seed = micro_batch_size, seed

    def batch(self, step: int, micro: int) -> list[Any]:
        rng = np.random.default_rng([self.seed, step, micro])
        which = rng.choice(len(self.pools), size=self.micro_batch_size, p=self.probabilities)
        return [self.pools[k][int(rng.integers(len(self.pools[k])))] for k in which]
