"""JSONL logging helpers for finance poisoning experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TextIO


def open_log(path: Path | str) -> TextIO:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p.open("w", encoding="utf-8", buffering=1)


def log_row(f: TextIO, row: dict[str, Any]) -> None:
    f.write(json.dumps(row, default=str) + "\n")


def log_config(f: TextIO, config: dict[str, Any]) -> None:
    log_row(f, {"event": "config", **config})


def log_episode_row(f: TextIO, row: dict[str, Any]) -> None:
    log_row(f, {"event": "episode", **row})


def episode_to_log_dict(result) -> dict[str, Any]:
    d = result.to_log_dict()
    d["poison_in_top5"] = d.pop("poison_in_top5", d.get("poison_in_top5"))
    return d
