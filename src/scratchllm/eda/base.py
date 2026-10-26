"""The contract every EDA check follows.

A check is a small accumulator: the runner feeds it one document at a time (`update`) and asks
for the numbers at the end (`result`). This keeps memory flat, lets the same check run on a
50k-doc sample or a full corpus, and makes each check testable on a handful of strings.

`doc` is one record of a sample file: {"id", "source", "url", "split", "text_sha256", "text"}.
`result()` must return plain JSON values (numbers, strings, lists, dicts), because it goes
straight into the report.
"""

from __future__ import annotations

from typing import Any, Protocol


class Check(Protocol):
    name: str

    def update(self, doc: dict[str, Any]) -> None: ...

    def result(self) -> dict[str, Any]: ...
