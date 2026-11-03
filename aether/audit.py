"""HMAC-signed, hash-chained local audit log.

The log is tamper-evident while the HMAC key remains secret. It is not WORM
storage: administrators with filesystem and key access can still rewrite it.
"""
from __future__ import annotations

from hashlib import sha256
import hmac
import json
import os
from pathlib import Path
from threading import RLock
from typing import Any
from datetime import datetime, timezone


class AuditIntegrityError(RuntimeError):
    pass


class SignedAuditLog:
    def __init__(self, path: str | Path, key: bytes):
        if not isinstance(key, bytes) or len(key) < 32:
            raise ValueError("Audit HMAC key must contain at least 32 bytes")
        self.path = Path(path)
        self.key = key
        self._lock = RLock()

    @staticmethod
    def _canonical(value: dict[str, Any]) -> bytes:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")

    def _read_entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        if self.path.stat().st_size > 50_000_000:
            raise AuditIntegrityError("Audit log exceeds the 50 MB verification limit")
        entries = []
        with self.path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if len(line.encode("utf-8")) > 1_000_000:
                    raise AuditIntegrityError(f"Audit entry {number} exceeds the size limit")
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AuditIntegrityError(f"Malformed audit entry at line {number}") from exc
                if not isinstance(item, dict) or set(item) != {"payload", "previous_signature", "signature"}:
                    raise AuditIntegrityError(f"Invalid audit envelope at line {number}")
                entries.append(item)
        return entries

    def verify(self) -> dict[str, Any]:
        with self._lock:
            entries = self._read_entries()
            previous = "0" * 64
            for number, entry in enumerate(entries, start=1):
                payload = entry["payload"]
                if not isinstance(payload, dict) or entry["previous_signature"] != previous:
                    raise AuditIntegrityError(f"Broken audit chain at line {number}")
                expected = hmac.new(
                    self.key,
                    self._canonical({"payload": payload, "previous_signature": previous}),
                    sha256,
                ).hexdigest()
                if not hmac.compare_digest(expected, str(entry["signature"])):
                    raise AuditIntegrityError(f"Audit signature mismatch at line {number}")
                previous = expected
            return {"valid": True, "entries": len(entries), "head": previous}

    def append(self, event_type: str, details: dict[str, Any]) -> dict[str, Any]:
        if not event_type.strip():
            raise ValueError("Audit event_type is required")
        if not isinstance(details, dict):
            raise ValueError("Audit details must be an object")
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type.strip(),
            "details": details,
        }
        with self._lock:
            current = self.verify()
            previous = current["head"]
            envelope = {"payload": payload, "previous_signature": previous}
            signature = hmac.new(self.key, self._canonical(envelope), sha256).hexdigest()
            entry = {**envelope, "signature": signature}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(entry, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return {"payload": payload, "signature": signature, "sequence": current["entries"] + 1}
