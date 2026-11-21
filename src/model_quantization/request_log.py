"""Append-only CSV log of every /ask call: who answered, when, how long, what was asked and returned.

The file opens in Excel (UTF-8 with BOM) and in pandas (`pd.read_csv(path)`).
Newlines inside prompts/answers are fine: every field is quoted.
"""

import csv
import threading
from pathlib import Path

from . import config

COLUMNS = [
    "request_id",     # same id for both rows of a model="both" request
    "model",          # "ollama" (original) or "nf4-dq" (our double-quantised model)
    "model_name",     # e.g. qwen3.5:4b or models/qwen3.5-4b-nf4-dq
    "start_time",     # local time, YYYY-MM-DD HH:MM:SS.mmm
    "end_time",
    "duration_sec",
    "duration_min",
    "prompt",         # what was asked
    "response",       # what the model returned
    "prompt_tokens",
    "new_tokens",
    "tokens_per_second",
    "status",         # ok / error
    "error",
]

_lock = threading.Lock()


def _path(path: str | Path | None) -> Path:
    return Path(path) if path else Path(config.LOG_CSV)


def append(row: dict, path: str | Path | None = None) -> None:
    p = _path(path)
    with _lock:
        p.parent.mkdir(parents=True, exist_ok=True)
        new = not p.exists() or p.stat().st_size == 0
        # BOM only when the file is created, so Excel detects UTF-8 (non-English text shows correctly).
        with p.open("a", newline="", encoding="utf-8-sig" if new else "utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, quoting=csv.QUOTE_ALL, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow({c: row.get(c, "") for c in COLUMNS})


def read(path: str | Path | None = None, limit: int | None = None) -> list[dict]:
    """Rows of the log, newest first."""
    p = _path(path)
    if not p.exists():
        return []
    with _lock, p.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    rows.reverse()
    return rows[:limit] if limit else rows
