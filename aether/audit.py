"""Append-only hash-chained audit journal with tamper-evident verification.

A local hash chain detects accidental edits and many offline modifications, but is
not immutable against an attacker who can rewrite the whole journal. For stronger
assurance, export signed checkpoints to independently controlled storage.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import Any, Mapping
import json
import time


GENESIS = "0" * 64


class AuditJournal:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    @staticmethod
    def _digest(record: Mapping[str, Any]) -> str:
        canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return sha256(canonical.encode("utf-8")).hexdigest()

    def append(self, event_type: str, payload: Mapping[str, Any]) -> dict:
        if not event_type.strip():
            raise ValueError("Audit event type is required")
        with self._lock:
            verification = self.verify()
            if not verification["valid"]:
                raise RuntimeError("Audit journal integrity check failed; refusing to append")
            previous = verification["last_hash"]
            body = {
                "sequence": verification["count"] + 1,
                "timestamp_epoch": time.time(),
                "event_type": event_type,
                "payload": dict(payload),
                "previous_hash": previous,
            }
            entry = {**body, "entry_hash": self._digest(body)}
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
                handle.flush()
            return entry

    def verify(self) -> dict:
        previous = GENESIS
        count = 0
        if not self.path.exists():
            return {"valid": True, "count": 0, "last_hash": GENESIS}
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    entry = json.loads(line)
                    digest = entry.pop("entry_hash")
                    if entry.get("sequence") != count + 1 or entry.get("previous_hash") != previous:
                        return {"valid": False, "count": count, "last_hash": previous,
                                "error": f"Broken chain at line {line_number}"}
                    if self._digest(entry) != digest:
                        return {"valid": False, "count": count, "last_hash": previous,
                                "error": f"Hash mismatch at line {line_number}"}
                    previous = digest
                    count += 1
                except (ValueError, KeyError, TypeError):
                    return {"valid": False, "count": count, "last_hash": previous,
                            "error": f"Malformed record at line {line_number}"}
        return {"valid": True, "count": count, "last_hash": previous}
