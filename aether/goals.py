"""Deterministic bounded goal discovery and approval-gated scheduling."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any
from uuid import uuid4
from aether.storage import Store
from aether.audit import AuditJournal
from pathlib import Path
import json
import time


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
    def __init__(self, store: Store, self_model: Any | None = None):
        self.store = store
        self.self_model = self_model

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
        if self.self_model is not None:
            snapshot = self.self_model.snapshot()
            for capability in snapshot.get("capabilities", []):
                if capability["state"] in {"planned", "implemented_unverified", "degraded", "unavailable"}:
                    state = capability["state"]
                    add(GoalCandidate(
                        f"Verify or repair capability {capability['id']}: {capability['name']}",
                        f"capability:{capability['id']}", 3,
                        "Replace capability status assumptions with recorded test evidence.",
                        "medium" if state in {"degraded", "unavailable"} else "low",
                        {"max_steps": 8, "max_runtime_seconds": 600, "max_model_calls": 6},
                        ("Inspect capability implementation", "Run targeted acceptance tests",
                         "Record evidence and limitations in the self-model"),
                        "Do not modify permissions, credentials, protected data or production code without separate approval.",
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



class HierarchicalGoalGraph:
    """Event-sourced goal tree with explicit lifecycle gates and bounded budgets."""

    MAX_DEPTH = 5
    MAX_CHILDREN = 12
    TERMINAL = {"completed", "cancelled"}

    TRANSITIONS = {
        "proposed": {"approved", "cancelled", "deferred"},
        "approved": {"running", "cancelled", "deferred"},
        "running": {"completed", "blocked", "failed", "cancelled"},
        "blocked": {"approved", "cancelled", "deferred"},
        "failed": {"approved", "cancelled"},
        "deferred": {"approved", "cancelled"},
        "completed": set(),
        "cancelled": set(),
    }

    def __init__(self, journal_path: str | Path):
        self.journal = AuditJournal(journal_path)

    @staticmethod
    def _validate_budget(budget: dict[str, Any]) -> dict[str, int]:
        limits = {"max_steps": 1000, "max_runtime_seconds": 86400,
                  "max_model_calls": 1000, "max_tokens": 10_000_000}
        result = {}
        for key, maximum in limits.items():
            value = budget.get(key, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > maximum:
                raise ValueError(f"{key} must be an integer between 0 and {maximum}")
            result[key] = value
        if not any(result.values()):
            raise ValueError("At least one explicit resource budget is required")
        return result

    def create(
        self, objective: str, *, parent_id: str | None = None,
        priority: int = 3, success_criteria: list[str] | None = None,
        constraints: list[str] | None = None,
        resource_budget: dict[str, Any] | None = None,
        owner: str = "user",
    ) -> dict[str, Any]:
        objective = " ".join(objective.split())
        if not objective or len(objective) > 2000:
            raise ValueError("Goal objective must contain 1..2000 characters")
        if isinstance(priority, bool) or not isinstance(priority, int) or not 1 <= priority <= 5:
            raise ValueError("Goal priority must be between 1 and 5")
        if not owner.strip():
            raise ValueError("Goal owner is required")
        criteria = [str(item).strip() for item in (success_criteria or []) if str(item).strip()]
        if not criteria:
            raise ValueError("At least one measurable success criterion is required")
        if parent_id:
            parent = self.get(parent_id)
            if parent is None:
                raise KeyError(f"Unknown parent goal: {parent_id}")
            if parent["status"] in self.TERMINAL:
                raise ValueError("Cannot add children to a terminal goal")
            if self._depth(parent_id) >= self.MAX_DEPTH:
                raise ValueError("Goal tree maximum depth exceeded")
        node = {
            "id": uuid4().hex, "parent_id": parent_id, "objective": objective,
            "priority": priority, "success_criteria": criteria,
            "constraints": [str(item).strip() for item in (constraints or []) if str(item).strip()],
            "resource_budget": self._validate_budget(resource_budget or {}),
            "owner": owner, "status": "proposed", "created_at_epoch": time.time(),
        }
        self.journal.append("goal.created", node)
        return node

    def decompose(self, parent_id: str, children: list[dict[str, Any]]) -> list[dict[str, Any]]:
        parent = self.get(parent_id)
        if parent is None:
            raise KeyError(parent_id)
        if parent["status"] in self.TERMINAL:
            raise ValueError("Cannot decompose a terminal goal")
        if not isinstance(children, list) or not 1 <= len(children) <= self.MAX_CHILDREN:
            raise ValueError(f"Decomposition requires 1..{self.MAX_CHILDREN} child goals")
        # Validate the complete batch before writing any child records.
        objectives = [str(item.get("objective", "")).strip().casefold() for item in children]
        if any(not item for item in objectives) or len(objectives) != len(set(objectives)):
            raise ValueError("Child objectives must be non-empty and unique")
        for item in children:
            self._validate_budget(item.get("resource_budget", {}))
            if not item.get("success_criteria"):
                raise ValueError("Every child goal requires measurable success criteria")
            if not 1 <= int(item.get("priority", 3)) <= 5:
                raise ValueError("Child priority must be between 1 and 5")
        return [
            self.create(
                item["objective"], parent_id=parent_id,
                priority=int(item.get("priority", 3)),
                success_criteria=item["success_criteria"],
                constraints=item.get("constraints", []),
                resource_budget=item["resource_budget"],
                owner=item.get("owner", parent["owner"]),
            )
            for item in children
        ]

    def transition(self, goal_id: str, status: str, *, actor: str, evidence: str) -> dict[str, Any]:
        current = self.get(goal_id)
        if current is None:
            raise KeyError(goal_id)
        if status not in self.TRANSITIONS.get(current["status"], set()):
            raise ValueError(f"Invalid goal transition: {current['status']} -> {status}")
        if not actor.strip() or not evidence.strip():
            raise ValueError("Every goal transition requires an actor and evidence/rationale")
        if status == "running" and current["status"] != "approved":
            raise PermissionError("Only an approved goal can start")
        if status == "completed":
            children = [item for item in self.list() if item["parent_id"] == goal_id]
            if any(item["status"] != "completed" for item in children):
                raise ValueError("A parent goal cannot complete before all child goals complete")
        event = {
            "id": goal_id, "status": status, "actor": actor, "evidence": evidence,
            "updated_at_epoch": time.time(),
        }
        self.journal.append("goal.transition", event)
        return self.get(goal_id) or {}

    def list(self) -> list[dict[str, Any]]:
        if not self.journal.verify()["valid"]:
            raise RuntimeError("Goal graph audit journal failed integrity verification")
        latest: dict[str, dict[str, Any]] = {}
        if not self.journal.path.exists():
            return []
        with self.journal.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)
                payload = event["payload"]
                if event["event_type"] == "goal.created":
                    latest[payload["id"]] = dict(payload)
                elif event["event_type"] == "goal.transition" and payload["id"] in latest:
                    latest[payload["id"]].update(payload)
        return sorted(latest.values(), key=lambda item: (item["priority"], item["created_at_epoch"]))

    def get(self, goal_id: str) -> dict[str, Any] | None:
        return next((item for item in self.list() if item["id"] == goal_id), None)

    def tree(self) -> list[dict[str, Any]]:
        nodes = {item["id"]: {**item, "children": []} for item in self.list()}
        roots = []
        for node in nodes.values():
            parent_id = node["parent_id"]
            if parent_id and parent_id in nodes:
                nodes[parent_id]["children"].append(node)
            else:
                roots.append(node)
        return roots

    def _depth(self, goal_id: str) -> int:
        depth, current = 0, self.get(goal_id)
        while current and current["parent_id"]:
            depth += 1
            current = self.get(current["parent_id"])
            if depth > self.MAX_DEPTH:
                raise RuntimeError("Goal graph contains a cycle or invalid depth")
        return depth
