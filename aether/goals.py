"""Goal discovery, evaluation, prioritization and authorization for bounded autonomy.

Discovered goals are proposals only. No discovery path grants execution permission.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
from uuid import uuid4

from aether.storage import Store
from aether.tasks import TaskManager
from aether.self_model import SelfModel


DEFAULT_BUDGET = {"max_runtime_seconds": 300, "max_tool_calls": 8, "max_model_calls": 1}


class GoalManager:
    def __init__(self, store: Store, project_root: str | Path = "."):
        self.store = store
        self.project_root = Path(project_root).resolve()
        self.tasks = TaskManager(store)

    @staticmethod
    def _validate(goal: dict[str, Any]) -> None:
        required = ("title", "purpose", "source", "source_key", "expected_benefit", "stop_condition")
        if any(not isinstance(goal.get(key), str) or not goal[key].strip() for key in required):
            raise ValueError("Goal title, purpose, source, source_key, benefit and stop condition are required")
        if goal.get("goal_level", "immediate") not in {
            "immediate", "project", "system", "research", "exploration"
        }:
            raise ValueError("Unsupported goal level")
        priority, complexity = goal.get("priority", 0.5), goal.get("complexity", 2)
        if isinstance(priority, bool) or not isinstance(priority, (int, float)) or not 0 <= priority <= 1:
            raise ValueError("Goal priority must be between 0 and 1")
        if isinstance(complexity, bool) or not isinstance(complexity, int) or not 1 <= complexity <= 5:
            raise ValueError("Goal complexity must be an integer from 1 to 5")
        budget = goal.get("resource_budget", DEFAULT_BUDGET)
        if not isinstance(budget, dict):
            raise ValueError("Goal resource_budget must be an object")
        for key, ceiling in (("max_runtime_seconds", 86400), ("max_tool_calls", 100), ("max_model_calls", 10)):
            value = budget.get(key, DEFAULT_BUDGET[key])
            if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value > ceiling:
                raise ValueError(f"Goal budget {key} must be between 1 and {ceiling}")
        for key in ("risks", "dependencies", "completion_criteria"):
            value = goal.get(key, [])
            if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
                raise ValueError(f"Goal {key} must be an array of non-empty strings")
        deadline = goal.get("deadline_at")
        if deadline is not None:
            if not isinstance(deadline, str) or not deadline.strip():
                raise ValueError("Goal deadline_at must be an ISO-8601 timestamp or null")
            try:
                parsed_deadline = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("Goal deadline_at must be a valid ISO-8601 timestamp") from exc
            if parsed_deadline.tzinfo is None:
                raise ValueError("Goal deadline_at must include a timezone")

    def propose(
        self, *, title: str, purpose: str, source: str, source_key: str,
        expected_benefit: str, priority: float = 0.5, complexity: int = 2,
        goal_level: str = "immediate",
        resource_budget: dict[str, int] | None = None, risks: list[str] | None = None,
        deadline_at: str | None = None, dependencies: list[str] | None = None,
        completion_criteria: list[str] | None = None,
        stop_condition: str = "Stop when completion criteria are met or any resource budget is exhausted.",
    ) -> dict[str, Any]:
        goal = {
            "id": uuid4().hex, "title": title.strip(), "purpose": purpose.strip(),
            "source": source.strip(), "source_key": source_key.strip(),
            "expected_benefit": expected_benefit.strip(), "priority": priority,
            "goal_level": goal_level,
            "complexity": complexity, "resource_budget": {**DEFAULT_BUDGET, **(resource_budget or {})},
            "risks": risks or [], "deadline_at": deadline_at,
            "dependencies": dependencies or [], "completion_criteria": completion_criteria or [],
            "stop_condition": stop_condition.strip(), "status": "proposed",
        }
        self._validate(goal)
        try:
            self.store.create_autonomy_goal(goal)
        except Exception:
            existing = next((item for item in self.store.list_autonomy_goals(limit=1000)
                             if item["source_key"] == goal["source_key"]), None)
            if existing is not None:
                return existing
            raise
        self.notify("goal_discovered", "New autonomy goal proposed",
                    f"{goal['title']} — awaiting explicit approval.")
        return self.store.get_autonomy_goal(goal["id"])

    def discover(self, limit: int = 10) -> list[dict[str, Any]]:
        """Convert failed autonomy tasks into review proposals; never auto-retry them."""
        discovered = []
        discovery_limit = max(1, min(limit, 50))
        for task in self.store.list_tasks(status="failed", limit=discovery_limit):
            if len(discovered) >= discovery_limit:
                break
            if task["state"].get("kind") != "autonomy":
                continue
            reason = (task["state"].get("recovery", {}).get("reason")
                      or task["state"].get("error") or "previous execution failed")
            goal = self.propose(
                title=f"Review failed task: {task['objective'][:90]}",
                purpose=(f"Inspect task {task['id']} and decide whether a safe retry or revised plan "
                         f"is appropriate. Original failure: {str(reason)[:400]}"),
                source="failed_task", source_key=f"failed-task:{task['id']}",
                expected_benefit="Recover useful work without blindly replaying uncertain side effects.",
                priority=0.65, complexity=2, resource_budget=dict(DEFAULT_BUDGET),
                risks=["The original task may have performed a side effect before interruption."],
                completion_criteria=[
                    "Inspect original task event history and receipts.",
                    "Decide whether retry is safe; do not repeat uncertain side effects.",
                    "Record a verified recovery recommendation.",
                ],
                stop_condition="Stop and request human review if prior side effects cannot be established.",
            )
            discovered.append(goal)
        # Acceptance reports are durable signals of known capability gaps and regressions.
        report_path = self.project_root / "reports" / "acceptance" / "latest.json"
        if len(discovered) < discovery_limit and report_path.is_file():
            try:
                if report_path.stat().st_size <= 1_000_000:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    generated_at = str(report.get("generated_at", "unknown"))
                    for item in report.get("results", []):
                        if len(discovered) >= discovery_limit:
                            break
                        status = item.get("acceptance_status")
                        if status not in {"FAIL", "PARTIAL"}:
                            continue
                        case_id = str(item.get("id", "unknown"))
                        name = str(item.get("name", "acceptance gap"))[:100]
                        limitation = "; ".join(str(value)[:300] for value in item.get("limitations", []))
                        goal = self.propose(
                            title=f"Review acceptance gap {case_id}: {name}",
                            purpose=(
                                f"Investigate OMEGA acceptance case {case_id} ({name}). "
                                f"Current status: {status}. Expected: {str(item.get('expected', 'not recorded'))[:500]}. "
                                f"Observed: {str(item.get('actual', 'not recorded'))[:500]}. "
                                f"Known limitations: {limitation or 'none recorded'}"
                            ),
                            source="acceptance_gap",
                            source_key=f"acceptance-gap:{generated_at}:{case_id}:{status}",
                            expected_benefit="Turn a measured acceptance gap into a bounded, reviewable engineering objective.",
                            priority=0.75 if status == "FAIL" else 0.5,
                            complexity=3, goal_level="system",
                            risks=["A partial acceptance result may represent a documented limitation rather than a defect."],
                            completion_criteria=[
                                "Reproduce the reported gap or verify the stated limitation.",
                                "Propose a root-cause correction or document why it is out of scope.",
                                "Re-run the relevant acceptance tests and preserve the result.",
                            ],
                            stop_condition="Stop if the report is stale, evidence cannot be reproduced, or the fix requires unsafe permissions.",
                        )
                        discovered.append(goal)
            except (OSError, ValueError, TypeError):
                pass
        # Unverified/degraded capabilities are knowledge gaps, not permission to mutate code.
        profile = SelfModel(self.store.path).snapshot()
        for capability in profile["capabilities"]:
            if len(discovered) >= discovery_limit:
                break
            if capability["state"] == "verified":
                continue
            goal = self.propose(
                title=f"Verify capability: {capability['name']}",
                purpose=(
                    f"Inspect the implementation and tests for capability {capability['id']}; "
                    "run safe validation and record concrete evidence. Do not modify production code."
                ),
                source="capability_gap",
                source_key=f"capability:{capability['id']}:{capability['state']}",
                expected_benefit="Replace assumed capability status with test-backed evidence.",
                priority=0.45 if capability["state"] == "planned" else 0.6,
                complexity=2, goal_level="system",
                risks=["A module may exist without being operational on this machine."],
                completion_criteria=[
                    "Inspect the actual implementation and relevant tests.",
                    "Run a reproducible, non-destructive validation.",
                    "Record the exact test result or leave the capability unverified.",
                ],
                stop_condition="Stop if required hardware, credentials, or user approval are unavailable.",
            )
            discovered.append(goal)
            if len(discovered) >= discovery_limit:
                break

        return discovered

    def approve(self, goal_id: str) -> dict[str, Any]:
        goal = self.store.get_autonomy_goal(goal_id)
        if goal is None:
            raise KeyError(f"Unknown goal: {goal_id}")
        if goal["status"] != "proposed":
            raise ValueError(f"Only proposed goals can be approved; found {goal['status']}")
        result = self.store.update_autonomy_goal(goal_id, "proposed", "approved")
        self.notify("goal_approved", "Autonomy goal approved", result["title"])
        return result

    def cancel(self, goal_id: str) -> dict[str, Any]:
        goal = self.store.get_autonomy_goal(goal_id)
        if goal is None:
            raise KeyError(f"Unknown goal: {goal_id}")
        if goal["status"] in {"queued", "running", "paused"} and goal.get("task_id"):
            task = self.store.get_task(goal["task_id"])
            if task and task["status"] in {"queued", "running", "paused"}:
                self.tasks.cancel(task["id"], "Parent autonomy goal cancelled")
                if task["status"] == "running":
                    return goal
        if goal["status"] not in {"proposed", "approved", "queued", "running", "paused"}:
            raise ValueError(f"Goal cannot be cancelled from status {goal['status']}")
        result = self.store.update_autonomy_goal(goal_id, goal["status"], "cancelled")
        self.notify("goal_cancelled", "Autonomy goal cancelled", result["title"], "warning",
                    result.get("task_id"))
        return result

    def prioritize(self, goals: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        candidates = goals if goals is not None else self.store.list_autonomy_goals()
        now = datetime.now(timezone.utc)
        def score(goal: dict[str, Any]) -> float:
            urgency = 0.0
            if goal.get("deadline_at"):
                try:
                    due = datetime.fromisoformat(goal["deadline_at"].replace("Z", "+00:00"))
                    seconds = (due - now).total_seconds()
                    urgency = 0.2 if seconds <= 0 else max(0.0, 0.2 * (1 - seconds / 604800))
                except ValueError:
                    pass
            benefit = min(0.1, len(goal.get("completion_criteria", [])) * 0.02)
            return goal["priority"] * 0.7 + urgency + benefit - 0.03 * (goal["complexity"] - 1)
        return sorted(candidates, key=score, reverse=True)

    def queue_approved(self, limit: int = 1) -> list[str]:
        queued = []
        for goal in self.prioritize(self.store.list_autonomy_goals(status="approved")):
            if len(queued) >= max(0, min(limit, 10)):
                break
            if any((dependency := self.store.get_autonomy_goal(item)) is None
                   or dependency["status"] != "completed" for item in goal["dependencies"]):
                continue
            task = self.tasks.create(goal["purpose"], {
                "kind": "autonomy", "goal_id": goal["id"], "goal_title": goal["title"],
                "receipts": [], "resource_budget": goal["resource_budget"],
                "completion_criteria": goal["completion_criteria"],
                "stop_condition": goal["stop_condition"],
            })
            try:
                self.store.update_autonomy_goal(goal["id"], "approved", "queued", task_id=task.id)
            except Exception:
                self.tasks.transition(task.id, "cancelled", {"reason": "Goal was claimed concurrently"})
                continue
            queued.append(task.id)
            self.notify("goal_queued", "Approved goal queued",
                        f"{goal['title']} is queued for bounded execution.", task_id=task.id)
        return queued

    def sync_task_status(self, task_id: str, status: str) -> None:
        task = self.store.get_task(task_id)
        if task is None or not task["state"].get("goal_id"):
            return
        goal = self.store.get_autonomy_goal(task["state"]["goal_id"])
        if goal is None:
            return
        target = {"running": "running", "paused": "paused", "completed": "completed",
                  "cancelled": "cancelled", "failed": "blocked"}.get(status)
        if target and goal["status"] != target:
            try:
                self.store.update_autonomy_goal(goal["id"], goal["status"], target, task_id=task_id)
            except ValueError:
                pass

    def notify(self, kind: str, title: str, message: str,
               severity: str = "info", task_id: str | None = None) -> None:
        self.store.create_notification(uuid4().hex, kind, title, message, severity, task_id)
