"""Living frontier research backlog with provenance-rich, reviewable hypotheses."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4
import json
import time

from aether.audit import AuditJournal


REQUIRED_FIELDS = (
    "question", "current_state", "primary_sources", "reproducible_evidence",
    "technical_requirements", "feasibility", "resource_cost", "limitations",
    "aether_application", "proposed_experiment",
)


class ResearchFrontier:
    """Record research opportunities and their lifecycle; never treat proposals as results."""

    def __init__(self, journal_path: str | Path):
        self.journal = AuditJournal(journal_path)

    def propose(self, record: Mapping[str, Any]) -> dict[str, Any]:
        missing = [field for field in REQUIRED_FIELDS if field not in record]
        if missing:
            raise ValueError("Research frontier record missing: " + ", ".join(missing))
        if not str(record["question"]).strip():
            raise ValueError("Research question cannot be empty")
        sources = record["primary_sources"]
        if not isinstance(sources, list) or not all(
            isinstance(item, str) and item.startswith(("https://", "http://")) for item in sources
        ):
            raise ValueError("primary_sources must be a list of explicit HTTP(S) URLs")
        item = {
            "id": uuid4().hex,
            "status": "proposed",
            "created_at_epoch": time.time(),
            "record": dict(record),
            "evidence_status": "not_independently_validated",
        }
        self.journal.append("frontier.proposed", item)
        return item

    def transition(
        self, item_id: str, status: str, *, evidence: str,
        reviewer: str, limitations: str,
    ) -> dict[str, Any]:
        allowed = {"investigating", "experiment_designed", "experiment_running",
                   "validated", "rejected", "deferred"}
        if status not in allowed or not evidence.strip() or not reviewer.strip() or not limitations.strip():
            raise ValueError("Transition requires a valid status, evidence, reviewer and limitations")
        current = self.get(item_id)
        if current is None:
            raise KeyError(item_id)
        if current["status"] in {"validated", "rejected"}:
            raise ValueError("Terminal research records cannot be silently reopened")
        if status == "validated" and "result" not in evidence.casefold():
            raise ValueError("Validation transition must explicitly describe an observed result")
        update = {
            "id": item_id, "status": status, "evidence": evidence,
            "reviewer": reviewer, "limitations": limitations,
            "updated_at_epoch": time.time(),
            "evidence_status": "reviewer_asserted_not_independently_verified",
        }
        self.journal.append("frontier.transition", update)
        return self.get(item_id) or update

    def list(self) -> list[dict[str, Any]]:
        records = self._events()
        latest: dict[str, dict[str, Any]] = {}
        for event in records:
            payload = event["payload"]
            item_id = payload.get("id")
            if not item_id:
                continue
            if event["event_type"] == "frontier.proposed":
                latest[item_id] = dict(payload)
            elif event["event_type"] == "frontier.transition" and item_id in latest:
                latest[item_id].update(payload)
        return sorted(latest.values(), key=lambda item: item.get("created_at_epoch", 0), reverse=True)

    def get(self, item_id: str) -> dict[str, Any] | None:
        return next((item for item in self.list() if item["id"] == item_id), None)

    def integrity(self) -> dict[str, Any]:
        return self.journal.verify()

    def _events(self) -> list[dict[str, Any]]:
        if not self.journal.path.exists():
            return []
        events = []
        with self.journal.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                entry = json.loads(line)
                events.append({"event_type": entry["event_type"], "payload": entry["payload"]})
        return events
