"""Low-cardinality structured observability with bounded, content-free telemetry."""
from __future__ import annotations
from collections import Counter, defaultdict
from contextlib import contextmanager
from math import isfinite
from pathlib import Path
from threading import RLock
from typing import Any, Iterator
import json, time


class TelemetryRegistry:
    def __init__(self, event_path: str | Path | None = None, *, max_events: int = 2000):
        if not 100 <= max_events <= 100_000:
            raise ValueError("max_events must be between 100 and 100000")
        self.event_path = Path(event_path) if event_path else None
        if self.event_path:
            self.event_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_events = max_events
        self._lock = RLock()
        self._counters: Counter[str] = Counter()
        self._gauges: dict[str, float] = {}
        self._durations: dict[str, list[float]] = defaultdict(list)
        self._events: list[dict[str, Any]] = []

    @staticmethod
    def _name(name: str) -> str:
        normalized = name.strip().casefold()
        if not normalized or len(normalized) > 100 or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789._-" for ch in normalized):
            raise ValueError("Metric names must be bounded lowercase identifiers")
        return normalized

    def increment(self, name: str, amount: int = 1) -> None:
        key = self._name(name)
        if isinstance(amount, bool) or not isinstance(amount, int) or not 1 <= amount <= 1_000_000:
            raise ValueError("Counter increment must be a positive bounded integer")
        with self._lock:
            self._counters[key] += amount

    def gauge(self, name: str, value: float) -> None:
        key = self._name(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(float(value)):
            raise ValueError("Gauge values must be finite numbers")
        with self._lock:
            self._gauges[key] = float(value)

    def observe(self, name: str, seconds: float) -> None:
        key = self._name(name)
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not isfinite(float(seconds)) or seconds < 0:
            raise ValueError("Duration observations must be finite and non-negative")
        with self._lock:
            values = self._durations[key]
            values.append(float(seconds))
            if len(values) > self.max_events:
                del values[:len(values) - self.max_events]

    @contextmanager
    def timer(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.observe(name, time.perf_counter() - started)

    def event(self, name: str, *, status: str, attributes: dict[str, Any] | None = None) -> dict:
        key = self._name(name)
        if not status.strip() or len(status) > 40:
            raise ValueError("Event status is required and bounded")
        safe: dict[str, Any] = {}
        for attr, value in (attributes or {}).items():
            if not isinstance(attr, str) or len(attr) > 50:
                raise ValueError("Telemetry attribute names must be short strings")
            if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
                safe[attr] = value
            elif isinstance(value, str):
                safe[attr] = value[:120]
            else:
                raise ValueError("Telemetry only accepts scalar attributes; content and objects are forbidden")
        record = {"timestamp_epoch": time.time(), "name": key, "status": status, "attributes": safe}
        with self._lock:
            self._events.append(record)
            if len(self._events) > self.max_events:
                del self._events[:len(self._events) - self.max_events]
            if self.event_path:
                with self.event_path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            durations = {
                name: {"count": len(values), "total_seconds": sum(values),
                       "mean_seconds": sum(values) / len(values),
                       "max_seconds": max(values)}
                for name, values in self._durations.items() if values
            }
            return {"counters": dict(self._counters), "gauges": dict(self._gauges),
                    "durations": durations, "recent_events": list(self._events[-100:])}
