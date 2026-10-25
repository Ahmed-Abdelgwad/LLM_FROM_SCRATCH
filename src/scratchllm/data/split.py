"""Deterministic train/val/test assignment.

Assignment is a pure function of (seed, key), so it needs no stored state, gives the same
answer on every machine, and can be applied *before* any cleaning or analysis.

Grouping key:
- web sources: the URL's domain, so pages from one site cannot straddle the train/eval boundary.
  Caveat: a very large domain lands wholly in one split, so realised split sizes only
  approximate the requested fractions. The manifest records the realised counts.
- everything else (e.g. Wikipedia, where every URL shares one host): the source-qualified
  document id. The id must be non-empty; the sampler synthesizes one from content if missing.
"""

from __future__ import annotations

import hashlib
from urllib.parse import urlparse


def split_key(source_name: str, kind: str, doc_id: str, url: str | None) -> str:
    if not doc_id:
        raise ValueError("doc_id must be non-empty; synthesize one before splitting")
    if kind == "web" and url:
        host = urlparse(url).netloc.lower().removeprefix("www.")
        if host:
            return f"domain:{host}"
    return f"id:{source_name}:{doc_id}"


def assign_split(key: str, seed: int, val_frac: float, test_frac: float) -> str:
    if val_frac < 0 or test_frac < 0 or val_frac + test_frac >= 1:
        raise ValueError("require 0 <= val_frac, test_frac and val_frac + test_frac < 1")
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64  # uniform in [0, 1)
    if u < test_frac:
        return "test"
    if u < test_frac + val_frac:
        return "val"
    return "train"
