"""Append-only hash-chained audit journal; integrity-evident, not attacker-proof."""
from __future__ import annotations
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import Any, Mapping
import json, time

GENESIS = "0" * 64


class AuditJournal:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    @staticmethod
    def _digest(record: Mapping[str, Any]) -> str:
        raw = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return sha256(raw.encode("utf-8")).hexdigest()

    def append(self, event_type: str, payload: Mapping[str, Any]) -> dict:
        if not event_type.strip():
            raise ValueError("Audit event type is required")
        with self._lock:
            check = self.verify()
            if not check["valid"]:
                raise RuntimeError("Audit journal integrity check failed")
            body = {"sequence": check["count"] + 1, "timestamp_epoch": time.time(),
                    "event_type": event_type, "payload": dict(payload),
                    "previous_hash": check["last_hash"]}
            entry = {**body, "entry_hash": self._digest(body)}
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
                handle.flush()
            return entry

    def verify(self) -> dict:
        previous, count = GENESIS, 0
        if not self.path.exists():
            return {"valid": True, "count": 0, "last_hash": GENESIS}
        with self.path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                try:
                    entry = json.loads(line)
                    digest = entry.pop("entry_hash")
                    if entry.get("sequence") != count + 1 or entry.get("previous_hash") != previous:
                        raise ValueError("sequence or previous hash mismatch")
                    if self._digest(entry) != digest:
                        raise ValueError("entry hash mismatch")
                    previous, count = digest, count + 1
                except (ValueError, KeyError, TypeError) as exc:
                    return {"valid": False, "count": count, "last_hash": previous,
                            "error": f"Invalid journal line {number}: {exc}"}
        return {"valid": True, "count": count, "last_hash": previous}
