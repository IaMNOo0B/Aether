"""Bounded goal discovery, evaluation, hierarchy and approval workflow.

Discovery creates proposals only. It never grants permission or executes a task.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4
import math

from aether.storage import Store


GOAL_TRANSITIONS = {
    "proposed": {"approved", "rejected", "cancelled"},
    "approved": {"queued", "blocked", "cancelled"},
    "queued": {"running", "paused", "failed", "cancelled", "blocked"},
    "running": {"paused", "completed", "failed", "cancelled", "blocked"},
    "paused": {"approved", "queued", "cancelled", "blocked"},
    "blocked": {"proposed", "approved", "cancelled"},
    "failed": {"proposed", "cancelled"},
    "completed": set(),
    "cancelled": set(),
    "rejected": set(),
}


class GoalManager:
    """Persist and rank goals from auditable local signals."""

    def __init__(
        self, store: Store, self_model: Any | None = None,
        max_discoveries: int = 20,
    ):
        self.store = store
        self.self_model = self_model
        self.max_discoveries = max(1, min(int(max_discoveries), 100))

    @staticmethod
    def _validate_text(value: str, field: str, maximum: int = 2000) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} is required")
        value = value.strip()
        if len(value) > maximum:
            raise ValueError(f"{field} exceeds {maximum} characters")
        return value

    @staticmethod
    def _validate_deadline(value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("deadline_at must be an ISO-8601 timestamp") from exc
        if parsed.tzinfo is None:
            raise ValueError("deadline_at must include a timezone")
        return parsed.isoformat()

    @staticmethod
    def priority_score(goal: dict[str, Any], now: datetime | None = None) -> float:
        """Transparent ranking heuristic; it is not a learned utility function."""
        current = now or datetime.now(timezone.utc)
        benefit = float(goal["priority"])
        complexity = int(goal["complexity"])
        risk = float(goal["risk_score"])
        urgency = 0.0
        deadline = goal.get("deadline_at")
        if deadline:
            due = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
            hours = (due - current).total_seconds() / 3600
            urgency = 0.20 if hours <= 0 else 0.15 * math.exp(-hours / 72)
        score = 0.65 * benefit + urgency + 0.20 * (1 - risk) - 0.04 * (complexity - 1)
        return round(max(0.0, min(1.0, score)), 4)

    def create(
        self, *, title: str, purpose: str, source: str, source_key: str,
        expected_benefit: str, priority: float = 0.5, complexity: int = 2,
        risk_score: float = 0.1, resource_budget: dict[str, Any] | None = None,
        risks: list[str] | None = None, deadline_at: str | None = None,
        dependencies: list[str] | None = None,
        completion_criteria: list[str] | None = None,
        stop_condition: str = "Stop when completion criteria are met, budget is exhausted, or safety conditions fail.",
        parent_goal_id: str | None = None,
    ) -> dict[str, Any]:
        title = self._validate_text(title, "title", 200)
        purpose = self._validate_text(purpose, "purpose")
        source = self._validate_text(source, "source", 100)
        source_key = self._validate_text(source_key, "source_key", 300)
        expected_benefit = self._validate_text(expected_benefit, "expected_benefit", 1000)
        stop_condition = self._validate_text(stop_condition, "stop_condition", 1000)
        if not math.isfinite(priority) or not 0 <= priority <= 1:
            raise ValueError("priority must be between 0 and 1")
        if not isinstance(complexity, int) or not 1 <= complexity <= 5:
            raise ValueError("complexity must be an integer from 1 to 5")
        if not math.isfinite(risk_score) or not 0 <= risk_score <= 1:
            raise ValueError("risk_score must be between 0 and 1")
        dependencies = list(dependencies or [])
        if len(set(dependencies)) != len(dependencies):
            raise ValueError("Goal dependencies cannot contain duplicates")
        for dependency in dependencies:
            if dependency == parent_goal_id:
                raise ValueError("A goal cannot depend on its own parent")
            if self.store.get_autonomy_goal(dependency) is None:
                raise KeyError(f"Unknown dependency goal: {dependency}")
        if parent_goal_id and self.store.get_autonomy_goal(parent_goal_id) is None:
            raise KeyError(f"Unknown parent goal: {parent_goal_id}")
        criteria = [self._validate_text(item, "completion criterion", 500) for item in (completion_criteria or [])]
        if not criteria:
            raise ValueError("At least one measurable completion criterion is required")
        budget = dict(resource_budget or {"max_steps": 5, "max_runtime_seconds": 300, "max_model_calls": 8})
        for key, value in budget.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"Resource budget {key} must be a positive number")
        goal = {
            "id": uuid4().hex, "parent_goal_id": parent_goal_id, "title": title,
            "purpose": purpose, "source": source, "source_key": source_key,
            "expected_benefit": expected_benefit, "priority": float(priority),
            "complexity": complexity, "risk_score": float(risk_score),
            "resource_budget": budget,
            "risks": [self._validate_text(item, "risk", 500) for item in (risks or [])],
            "deadline_at": self._validate_deadline(deadline_at),
            "dependencies": dependencies, "completion_criteria": criteria,
            "stop_condition": stop_condition, "status": "proposed",
        }
        self.store.create_autonomy_goal(goal)
        self._notify("goal_discovered", "New goal proposal", title, "info", goal_id=goal["id"])
        return self.store.get_autonomy_goal(goal["id"]) or goal

    def _notify(
        self, kind: str, title: str, message: str, severity: str = "info",
        task_id: str | None = None, goal_id: str | None = None,
    ) -> None:
        self.store.create_notification(uuid4().hex, kind, title, message, severity, task_id, goal_id)

    def discover(self) -> list[dict[str, Any]]:
        """Create bounded proposals from local, auditable evidence only."""
        candidates: list[dict[str, Any]] = []
        for task in self.store.list_tasks(status="failed", limit=self.max_discoveries):
            state = task.get("state", {})
            if state.get("kind") == "autonomy" and not state.get("goal_id"):
                candidates.append({
                    "title": f"Review failed task: {task['objective'][:120]}",
                    "purpose": f"Inspect task {task['id']} failure receipts and propose a safe corrective plan. Do not replay uncertain side effects.",
                    "source": "failed_task", "source_key": f"failed-task:{task['id']}",
                    "expected_benefit": "Resolve a recorded failure without repeating unverified actions.",
                    "priority": 0.65, "complexity": 2, "risk_score": 0.15,
                    "risks": ["The previous task may have performed a side effect before interruption."],
                    "dependencies": [], "completion_criteria": ["Failure cause is identified", "A safe next action is documented"],
                })
        if self.self_model is not None:
            try:
                snapshot = self.self_model.snapshot()
            except Exception:
                snapshot = {}
            for capability in snapshot.get("capabilities", []):
                if capability.get("state") in {"planned", "implemented_unverified", "degraded"}:
                    capability_id = capability.get("id", "unknown")
                    candidates.append({
                        "title": f"Validate capability: {capability.get('name', capability_id)[:130]}",
                        "purpose": f"Design and run a bounded validation for capability {capability_id}. Record actual evidence and limitations.",
                        "source": "capability_gap", "source_key": f"capability:{capability_id}:{capability.get('state')}",
                        "expected_benefit": "Replace assumed capability status with test-backed evidence.",
                        "priority": 0.55, "complexity": 2, "risk_score": 0.05,
                        "risks": ["A test pass may not establish broad real-world capability."],
                        "dependencies": [], "completion_criteria": ["A reproducible test is run", "Evidence and limitations are recorded"],
                    })
                    if len(candidates) >= self.max_discoveries:
                        break
        for candidate in candidates[:self.max_discoveries]:
            try:
                self.create(**candidate)
            except Exception as exc:
                # A duplicate source key is expected on subsequent discovery ticks.
                if "UNIQUE constraint failed" not in str(exc):
                    raise
        return self.store.list_autonomy_goals(status="proposed", limit=self.max_discoveries)

    def transition(self, goal_id: str, new_status: str, payload: dict[str, Any] | None = None, task_id: str | None = None) -> dict[str, Any]:
        goal = self.store.get_autonomy_goal(goal_id)
        if goal is None:
            raise KeyError(f"Unknown goal: {goal_id}")
        if new_status not in GOAL_TRANSITIONS.get(goal["status"], set()):
            raise ValueError(f"Invalid goal transition: {goal['status']} -> {new_status}")
        return self.store.transition_autonomy_goal(goal_id, goal["status"], new_status, payload, task_id)

    def approve(self, goal_id: str) -> dict[str, Any]:
        goal = self.store.get_autonomy_goal(goal_id)
        if goal is None:
            raise KeyError(f"Unknown goal: {goal_id}")
        if goal["status"] != "proposed":
            raise ValueError("Only proposed goals can be approved")
        result = self.transition(goal_id, "approved", {"approved_by": "operator"})
        self._notify("goal_approved", "Goal approved", goal["title"], "success", goal_id=goal_id)
        return result

    def reject(self, goal_id: str, reason: str = "Rejected by operator") -> dict[str, Any]:
        goal = self.store.get_autonomy_goal(goal_id)
        if goal is None:
            raise KeyError(f"Unknown goal: {goal_id}")
        result = self.transition(goal_id, "rejected", {"reason": reason[:500]})
        self._notify("goal_rejected", "Goal rejected", reason, "warning", goal_id=goal_id)
        return result

    def replan(self, goal_id: str, *, purpose: str, expected_benefit: str,
               priority: float, complexity: int, risk_score: float,
               resource_budget: dict[str, Any], risks: list[str], deadline_at: str | None,
               dependencies: list[str], completion_criteria: list[str],
               stop_condition: str) -> dict[str, Any]:
        goal = self.store.get_autonomy_goal(goal_id)
        if goal is None:
            raise KeyError(f"Unknown goal: {goal_id}")
        if goal["status"] not in {"proposed", "blocked", "failed", "paused"}:
            raise ValueError("Only proposed, blocked, failed or paused goals can be replanned")
        if goal["status"] in {"failed", "paused", "blocked"}:
            self.transition(goal_id, "proposed", {"reason": "Replanned by operator"})
        return self.store.update_autonomy_goal_plan(
            goal_id, "proposed", purpose=purpose, expected_benefit=expected_benefit,
            priority=priority, complexity=complexity, risk_score=risk_score,
            resource_budget=resource_budget, risks=risks, deadline_at=deadline_at,
            dependencies=dependencies, completion_criteria=completion_criteria,
            stop_condition=stop_condition,
        )

    def dependencies_satisfied(self, goal: dict[str, Any]) -> bool:
        for dependency_id in goal["dependencies"]:
            dependency = self.store.get_autonomy_goal(dependency_id)
            if dependency is None or dependency["status"] != "completed":
                return False
        if goal.get("parent_goal_id"):
            parent = self.store.get_autonomy_goal(goal["parent_goal_id"])
            if parent is None or parent["status"] not in {"approved", "queued", "running", "completed"}:
                return False
        return True

    def enqueue_approved(self, limit: int = 1) -> list[dict[str, Any]]:
        queued = []
        approved = self.store.list_autonomy_goals(status="approved", limit=max(1, min(limit, 10)) * 5)
        approved.sort(key=lambda goal: self.priority_score(goal), reverse=True)
        for goal in approved:
            if len(queued) >= max(1, min(limit, 10)):
                break
            if goal["risk_score"] > 0.25 or goal["complexity"] > 3:
                continue
            if not self.dependencies_satisfied(goal):
                if goal["status"] == "approved":
                    self.transition(goal["id"], "blocked", {"reason": "Dependencies are not complete"})
                continue
            task_id = uuid4().hex
            self.store.enqueue_approved_autonomy_goal(goal["id"], task_id)
            queued.append(self.store.get_autonomy_goal(goal["id"]))
        return queued

    def sync_task_result(self, goal_id: str, result: dict[str, Any]) -> dict[str, Any]:
        goal = self.store.get_autonomy_goal(goal_id)
        if goal is None:
            raise KeyError(f"Unknown goal: {goal_id}")
        status = result.get("status")
        if status == "completed":
            new_status = "completed"
            severity = "success"
        elif status == "paused":
            new_status = "paused"
            severity = "warning"
        elif status == "cancelled":
            new_status = "cancelled"
            severity = "warning"
        else:
            new_status = "failed"
            severity = "error"
        updated = self.transition(goal_id, new_status, {
            "task_id": result.get("task_id"), "task_status": status,
            "tool_results_verified": result.get("tool_results_verified", False),
            "objective_verified": result.get("verified", False),
        }, result.get("task_id"))
        self._notify("goal_"+new_status, "Goal "+new_status, goal["title"], severity, result.get("task_id"), goal_id)
        return updated
