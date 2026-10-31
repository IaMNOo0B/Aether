"""Centralized application logging and redacted runtime diagnostics."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


def configure_logging(level: str, log_file: str | Path = "logs/aether.log") -> None:
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid logging level: {level}")
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(path, encoding="utf-8"),
        ],
        force=True,
    )


def write_diagnostics(path: str | Path, status: dict[str, Any]) -> Path:
    """Write a small JSON report; callers must pass only non-secret status fields."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    safe_status = {
        key: value for key, value in status.items()
        if not any(secret in key.lower() for secret in ("key", "secret", "token", "password"))
    }
    destination.write_text(
        json.dumps(safe_status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination
