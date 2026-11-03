"""Persistent research frontier with evidence-aware state transitions."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4
import json, time
from aether.audit import AuditJournal

REQUIRED = ("question","current_state","primary_sources","reproducible_evidence",
            "technical_requirements","feasibility","resource_cost","limitations",
            "aether_application","proposed_experiment")


class ResearchFrontier:
    def __init__(self, journal_path: str | Path):
        self.journal = AuditJournal(journal_path)

    def propose(self, record: Mapping[str, Any]) -> dict:
        missing = [key for key in REQUIRED if key not in record]
        if missing: raise ValueError("Missing research fields: " + ", ".join(missing))
        if not str(record["question"]).strip(): raise ValueError("Research question is required")
        sources = record["primary_sources"]
        if not isinstance(sources, list) or not all(
            isinstance(url, str) and url.startswith(("http://","https://")) for url in sources
        ): raise ValueError("primary_sources must be explicit HTTP(S) URLs")
        item = {"id": uuid4().hex, "status": "proposed", "created_at_epoch": time.time(),
                "record": dict(record), "evidence_status": "not_independently_validated"}
        self.journal.append("frontier.proposed", item)
        return item

    def transition(self, item_id: str, status: str, *, evidence: str,
                   reviewer: str, limitations: str,
                   result_observation: str | None = None) -> dict:
        allowed = {"investigating","experiment_designed","experiment_running","validated","rejected","deferred"}
        if status not in allowed or not evidence.strip() or not reviewer.strip() or not limitations.strip():
            raise ValueError("Transition requires status, evidence, reviewer and limitations")
        current = self.get(item_id)
        if current is None: raise KeyError(item_id)
        if current["status"] in {"validated","rejected"}: raise ValueError("Terminal research record cannot be reopened")
        if status == "validated" and not (result_observation and result_observation.strip()):
            raise ValueError("Validation requires a concrete observed result")
        self.journal.append("frontier.transition", {
            "id": item_id, "status": status, "evidence": evidence, "reviewer": reviewer,
            "limitations": limitations, "result_observation": result_observation,
            "updated_at_epoch": time.time(),
            "evidence_status": "reviewer_asserted_not_independently_verified"})
        return self.get(item_id) or {}

    def list(self) -> list[dict]:
        latest = {}
        for event in self._events():
            payload = event["payload"]
            if event["event_type"] == "frontier.proposed":
                latest[payload["id"]] = dict(payload)
            elif payload.get("id") in latest:
                latest[payload["id"]].update(payload)
        return sorted(latest.values(), key=lambda item: item.get("created_at_epoch", 0), reverse=True)

    def get(self, item_id: str) -> dict | None:
        return next((item for item in self.list() if item["id"] == item_id), None)

    def integrity(self) -> dict:
        return self.journal.verify()

    def _events(self) -> list[dict]:
        if not self.journal.path.exists(): return []
        with self.journal.path.open("r", encoding="utf-8") as handle:
            return [{"event_type": item["event_type"], "payload": item["payload"]}
                    for item in map(json.loads, handle)]
