"""Deterministic bounded goal discovery and approval-gated scheduling."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable
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
    """Discover candidate work from persisted state; never self-authorize it."""

    def __init__(self, store: Store):
        self.store = store

    def discover(self, *, max_candidates: int = 20) -> list[GoalCandidate]:
        if not 1 <= max_candidates <= 100:
            raise ValueError("max_candidates must be between 1 and 100")
        candidates: list[GoalCandidate] = []
        seen: set[str] = set()

        def add(candidate: GoalCandidate) -> None:
            key = candidate.objective.casefold()
            if key not in seen and len(candidates) < max_candidates:
                seen.add(key)
                candidates.append(candidate)

        # Existing unfinished work has priority over speculative system goals.
        for task in self.store.list_tasks(limit=100):
            if task["status"] in {"queued", "paused", "failed"}:
                add(GoalCandidate(
                    objective=f"Review and safely continue task {task['id']}: {task['objective']}",
                    source=f"task:{task['id']}", priority=1,
                    expected_benefit="Resolve existing unfinished work without repeating uncertain side effects.",
                    risk="medium" if task["status"] == "failed" else "low",
                    resource_budget={"max_steps": 5, "max_runtime_seconds": 300, "max_model_calls": 5},
                    completion_criteria=("Task state and receipts inspected", "No unverified side effect replayed"),
                    stopping_condition="Stop if the next action may have side effects or requires new approval.",
                ))

        # Due/disabled schedules are surfaced, not silently re-enabled.
        for schedule in self.store.list_schedules(limit=100):
            if not schedule["enabled"] and (schedule["max_runs"] == 0 or schedule["run_count"] < schedule["max_runs"]):
                add(GoalCandidate(
                    objective=f"Review paused scheduled objective: {schedule['objective']}",
                    source=f"schedule:{schedule['id']}", priority=2,
                    expected_benefit="Decide whether a previously configured read-only objective remains useful.",
                    risk="low",
                    resource_budget={"max_steps": 5, "max_runtime_seconds": 180, "max_model_calls": 3},
                    completion_criteria=("Schedule reviewed", "Explicit owner approval recorded before reactivation"),
                    stopping_condition="Do not enable or execute the schedule during discovery.",
                ))
        return candidates

    def schedule(
        self, candidate: GoalCandidate, *, interval_seconds: int,
        approved: bool = False, max_runs: int = 1,
    ) -> str:
        if not approved:
            raise PermissionError("Scheduling a discovered goal requires explicit approval")
        if candidate.risk != "low":
            raise PermissionError("Only low-risk read-only goal candidates may be scheduled")
        if interval_seconds < 60 or max_runs < 1:
            raise ValueError("Bounded schedules require interval >= 60 seconds and max_runs >= 1")
        schedule_id = uuid4().hex
        self.store.create_schedule(
            schedule_id, candidate.objective, interval_seconds, max_runs=max_runs
        )
        return schedule_id

    @staticmethod
    def serialize(candidate: GoalCandidate) -> dict[str, Any]:
        return asdict(candidate)
