"""Opt-in, rate-limited research discovery with explicit human-review status.

This is a source-refresh assistant, not an autonomous truth verifier. It stores
reports through the existing provenance-aware Researcher and never promotes
discovered material into validated knowledge.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import threading
from typing import Any

from aether.research import Researcher
from aether.storage import Store

logger = logging.getLogger(__name__)
MAX_TOPICS = 20
MAX_INTERVAL_HOURS = 24 * 30


class DiscoveryError(RuntimeError):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def load_topics(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file() or source.stat().st_size > 64_000:
        raise DiscoveryError("Discovery config must be a regular JSON file under 64 KB")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiscoveryError(f"Invalid discovery config: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("topics"), list):
        raise DiscoveryError("Config must contain a 'topics' array")
    if not 1 <= len(payload["topics"]) <= MAX_TOPICS:
        raise DiscoveryError(f"Configure between 1 and {MAX_TOPICS} topics")
    topics = []
    seen = set()
    for item in payload["topics"]:
        if not isinstance(item, dict):
            raise DiscoveryError("Each topic must be an object")
        query = item.get("query")
        interval = item.get("interval_hours", 24)
        if not isinstance(query, str) or not 3 <= len(query.strip()) <= 300:
            raise DiscoveryError("Each query must contain 3..300 characters")
        if isinstance(interval, bool) or not isinstance(interval, (int, float)) or not 1 <= interval <= MAX_INTERVAL_HOURS:
            raise DiscoveryError("interval_hours must be between 1 and 720")
        normalized = query.strip()
        if normalized.casefold() in seen:
            raise DiscoveryError("Duplicate discovery queries are not allowed")
        seen.add(normalized.casefold())
        topics.append({"query": normalized, "interval_hours": float(interval)})
    return topics


class ResearchDiscovery:
    def __init__(
        self, store: Store, config_path: str | Path, state_path: str | Path,
        *, researcher: Researcher | None = None, clock=_utcnow,
    ):
        self.store = store
        self.config_path = Path(config_path)
        self.state_path = Path(state_path)
        self.researcher = researcher or Researcher(store)
        self.clock = clock

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"last_run": {}, "pending_review": []}
        if self.state_path.stat().st_size > 256_000:
            raise DiscoveryError("Discovery state exceeds 256 KB")
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DiscoveryError(f"Discovery state is invalid: {exc}") from exc
        if not isinstance(state, dict) or not isinstance(state.get("last_run", {}), dict):
            raise DiscoveryError("Discovery state has an invalid schema")
        if not isinstance(state.get("pending_review", []), list):
            raise DiscoveryError("Discovery pending_review state must be a list")
        state.setdefault("last_run", {})
        state.setdefault("pending_review", [])
        return state

    def _save_state(self, state: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        os.replace(temporary, self.state_path)

    def run_once(self, *, force: bool = False) -> dict[str, Any]:
        topics = load_topics(self.config_path)
        state = self._load_state()
        now = self.clock()
        due, skipped = [], []
        for topic in topics:
            query = topic["query"]
            previous = state["last_run"].get(query)
            if not force and previous:
                try:
                    last = datetime.fromisoformat(previous)
                except (TypeError, ValueError) as exc:
                    raise DiscoveryError(f"Invalid last-run timestamp for {query!r}") from exc
                if now - last < timedelta(hours=topic["interval_hours"]):
                    skipped.append(query)
                    continue
            due.append(topic)

        reports = []
        for topic in due:
            query = topic["query"]
            try:
                sources = self.researcher.search(query, limit=8)
                report = self.researcher.create_report(query, sources)
                reports.append({"query": query, "report_id": report["id"],
                                "source_count": len(sources), "review_status": "pending"})
                state["last_run"][query] = now.isoformat()
                state["pending_review"].append({
                    "query": query, "report_id": report["id"],
                    "discovered_at": now.isoformat(), "status": "pending",
                })
            except Exception as exc:
                logger.exception("Research discovery failed for %s", query)
                reports.append({"query": query, "error": str(exc), "status": "failed"})
        state["pending_review"] = state["pending_review"][-500:]
        self._save_state(state)
        return {"checked_at": now.isoformat(), "reports": reports,
                "skipped_not_due": skipped, "review_required": True}

    def run_forever(self, stop_event: threading.Event, *, poll_seconds: int = 60) -> None:
        if not 10 <= poll_seconds <= 3600:
            raise ValueError("poll_seconds must be between 10 and 3600")
        while not stop_event.is_set():
            try:
                self.run_once()
            except Exception:
                logger.exception("Research discovery cycle failed")
            stop_event.wait(poll_seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Opt-in scheduled research source refresh.")
    parser.add_argument("--config", default="config/discovery.json")
    parser.add_argument("--state", default="data/discovery-state.json")
    parser.add_argument("--database", default="data/aether.sqlite3")
    parser.add_argument("--once", action="store_true", help="Run one due refresh cycle and exit")
    parser.add_argument("--force", action="store_true", help="Ignore per-topic intervals for this run")
    parser.add_argument("--watch", action="store_true", help="Poll continuously until Ctrl+C")
    args = parser.parse_args(argv)
    service = ResearchDiscovery(Store(args.database), args.config, args.state)
    if args.watch:
        stop = threading.Event()
        try:
            service.run_forever(stop)
        except KeyboardInterrupt:
            stop.set()
        return 0
    result = service.run_once(force=args.force)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
