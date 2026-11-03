"""Deterministic bounded goal discovery and approval-gated scheduling."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any
from uuid import uuid4
from aether.storage import Store


@dataclass(frozen=True)
class GoalCandidate:
    objective: str
    source: str
    priority: int
    expected_benefit: str
    risk: str
    resource_budget: dict[str, int]
    completion_criteria: tuple[str, ...]
    stopping_condition: str


class GoalManager:
    def __init__(self, store: Store):
        self.store = store

    def discover(self, *, max_candidates: int = 20) -> list[GoalCandidate]:
        if not 1 <= max_candidates <= 100:
            raise ValueError("max_candidates must be between 1 and 100")
        candidates, seen = [], set()
        def add(item: GoalCandidate) -> None:
            key = item.objective.casefold()
            if key not in seen and len(candidates) < max_candidates:
                seen.add(key)
                candidates.append(item)
        for task in self.store.list_tasks(limit=100):
            if task["status"] in {"queued", "paused", "failed"}:
                add(GoalCandidate(
                    f"Review and safely continue task {task['id']}: {task['objective']}",
                    f"task:{task['id']}", 1,
                    "Resolve existing unfinished work without replaying uncertain side effects.",
                    "medium" if task["status"] == "failed" else "low",
                    {"max_steps": 5, "max_runtime_seconds": 300, "max_model_calls": 5},
                    ("Inspect task state and receipts", "Do not replay unverified side effects"),
                    "Stop if the next action may have side effects or requires new approval.",
                ))
        for item in self.store.list_schedules(limit=100):
            if not item["enabled"] and (item["max_runs"] == 0 or item["run_count"] < item["max_runs"]):
                add(GoalCandidate(
                    f"Review paused scheduled objective: {item['objective']}",
                    f"schedule:{item['id']}", 2,
                    "Decide whether an existing read-only objective remains useful.", "low",
                    {"max_steps": 5, "max_runtime_seconds": 180, "max_model_calls": 3},
                    ("Review schedule", "Require explicit owner approval before reactivation"),
                    "Discovery never enables or executes a schedule.",
                ))
        return candidates

    def schedule(self, candidate: GoalCandidate, *, interval_seconds: int,
                 approved: bool = False, max_runs: int = 1) -> str:
        if not approved:
            raise PermissionError("Scheduling a discovered goal requires explicit approval")
        if candidate.risk != "low":
            raise PermissionError("Only low-risk goals may be scheduled")
        if interval_seconds < 60 or max_runs < 1:
            raise ValueError("Bounded schedules require interval >= 60 and max_runs >= 1")
        schedule_id = uuid4().hex
        self.store.create_schedule(schedule_id, candidate.objective,
                                   interval_seconds, max_runs=max_runs)
        return schedule_id

    @staticmethod
    def serialize(candidate: GoalCandidate) -> dict[str, Any]:
        return asdict(candidate)
