"""Deterministic, bounded discovery of candidate goals from AETHER's own state.

Discovery proposes work only. It never schedules or executes a goal without an
explicit operator decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from aether.self_model import SelfModel
from aether.storage import Store


@dataclass(frozen=True)
class GoalCandidate:
    id: str
    objective: str
    source: str
    priority: float
    rationale: str


class GoalDiscovery:
    """Surface actionable gaps from persisted tasks and evidence-backed capability state."""

    def __init__(self, store: Store, self_model: SelfModel | None = None):
        self.store = store
        self.self_model = self_model

    @staticmethod
    def _candidate(objective: str, source: str, priority: float, rationale: str) -> GoalCandidate:
        key = sha256(f"{source}\0{objective}".encode("utf-8")).hexdigest()[:24]
        return GoalCandidate(key, objective, source, priority, rationale)

    def discover(self, limit: int = 20) -> list[dict[str, Any]]:
        if not 1 <= limit <= 100:
            raise ValueError("Goal discovery limit must be between 1 and 100")
        candidates: list[GoalCandidate] = []
        for status in ("failed", "paused"):
            for task in self.store.list_tasks(status=status, limit=limit):
                objective = f"Review and safely recover task {task['id']}: {task['objective']}"
                candidates.append(self._candidate(
                    objective, f"task:{status}", 0.90 if status == "failed" else 0.82,
                    f"Persisted task is {status}; inspect its receipts and remaining work before retrying.",
                ))

        if self.self_model is not None:
            for capability in self.self_model.snapshot().get("capabilities", []):
                state = capability.get("state")
                if state not in {"planned", "implemented_unverified", "degraded"}:
                    continue
                priority = {"degraded": 0.88, "implemented_unverified": 0.68, "planned": 0.48}[state]
                candidates.append(self._candidate(
                    f"Evaluate capability '{capability['name']}' with a reproducible acceptance test",
                    f"capability:{state}", priority,
                    f"Capability state is {state}; module presence alone is not verification.",
                ))

        candidates.sort(key=lambda item: (-item.priority, item.source, item.id))
        proposed = []
        for item in candidates[:limit]:
            self.store.save_goal_candidate(
                item.id, item.objective, item.source, item.priority, item.rationale,
                {"discovery_version": "goal-discovery-v1"},
            )
            proposed.append(item.id)
        selected = set(proposed)
        return [item for item in self.store.list_goal_candidates(status="proposed", limit=100)
                if item["id"] in selected]

    def approve(self, candidate_id: str) -> None:
        self.store.set_goal_candidate_status(candidate_id, "approved")

    def reject(self, candidate_id: str) -> None:
        self.store.set_goal_candidate_status(candidate_id, "rejected")

    def schedule(self, candidate_id: str, interval_seconds: int, max_runs: int = 1) -> str:
        candidate = next(
            (item for item in self.store.list_goal_candidates(limit=1000)
             if item["id"] == candidate_id), None
        )
        if candidate is None or candidate["status"] != "approved":
            raise PermissionError("A goal must be explicitly approved before scheduling")
        if not 5 <= interval_seconds <= 31_536_000 or max_runs < 1:
            raise ValueError("Schedule interval must be 5..31536000 seconds and max_runs >= 1")
        schedule_id = f"discovered-{candidate_id}"
        self.store.create_schedule(
            schedule_id, candidate["objective"], interval_seconds, max_runs=max_runs,
        )
        self.store.set_goal_candidate_status(candidate_id, "scheduled")
        return schedule_id
