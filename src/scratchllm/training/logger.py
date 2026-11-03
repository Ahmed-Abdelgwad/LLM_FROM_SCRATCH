"""Structured run log: one JSON object per line in `metrics.jsonl`.

Every line has `event` and `time` (UTC) plus the event's own fields, so a run can be analysed with
nothing but `json.loads` per line. Non-finite numbers are written as the strings "nan"/"inf"
(JSON has no such values), so a diverged run is still readable. Opened in append mode: a resumed
run continues the same file.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _clean(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file = path.open("a", encoding="utf-8")

    def log(self, event: str, **fields: Any) -> None:
        record = {"event": event, "time": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        record.update(_clean(fields))
        self._file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def read_events(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
