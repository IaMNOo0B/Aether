"""Opt-in bounded scheduler for approved, low-risk autonomy goals."""
from __future__ import annotations

import logging
from threading import Event
from typing import Any

from aether.config import SchedulerSettings
from aether.goals import GoalManager
from aether.storage import Store


class AutonomyScheduler:
    """Discovers proposals, queues approved low-risk goals and executes read-only tasks."""

    def __init__(
        self, store: Store, runner: Any, settings: SchedulerSettings,
        logger: logging.Logger | None = None, goal_manager: GoalManager | None = None,
    ):
        self.store = store
        self.runner = runner
        self.settings = settings
        self.logger = logger or logging.getLogger(__name__)
        self.goal_manager = goal_manager

    @staticmethod
    def _deny_side_effects(tool: str, purpose: str, args: dict[str, Any]) -> bool:
        # Background autonomy never authorizes writes, tests or external side effects.
        return False

    def _run_goal_tasks(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if self.goal_manager is None:
            return [], []
        self.goal_manager.reconcile()
        discovered = self.goal_manager.discover()
        self.goal_manager.enqueue_approved(limit=self.settings.max_tasks_per_tick)
        queued = self.store.list_autonomy_goals(status="queued", limit=100)
        results: list[dict[str, Any]] = []
        for goal in queued[:self.settings.max_tasks_per_tick]:
            task_id = goal.get("task_id")
            if not task_id:
                continue
            task = self.store.get_task(task_id)
            if task is None or task["status"] != "queued":
                continue
            self.goal_manager.transition(goal["id"], "running", {"task_id": task_id}, task_id)
            try:
                result = self.runner.run_task(task_id, self._deny_side_effects)
            except Exception as exc:
                self.logger.exception("Autonomous goal %s failed", goal["id"])
                current = self.store.get_task(task_id)
                result = {
                    "task_id": task_id,
                    "status": current["status"] if current else "failed",
                    "reason": str(exc)[:500],
                }
            try:
                self.goal_manager.sync_task_result(goal["id"], result)
            except Exception:
                self.logger.exception("Could not synchronize goal status for %s", goal["id"])
            results.append({"goal_id": goal["id"], **result})
        return discovered, results

    def run_once(self) -> dict[str, Any]:
        if self.store.emergency_stop_enabled():
            return {
                "status": "emergency_stop", "enqueued": [],
                "results": [], "discovered_goals": [], "goal_results": [],
            }
        if not self.settings.enabled:
            return {
                "status": "disabled", "enqueued": [],
                "results": [], "discovered_goals": [], "goal_results": [],
            }
        discovered, goal_results = self._run_goal_tasks()
        enqueued = self.store.enqueue_due_schedules(limit=self.settings.max_tasks_per_tick)
        queued_scheduled = [
            task for task in self.store.list_tasks(status="queued", limit=100)
            if task["state"].get("scheduled") is True
        ]
        results = []
        for task in queued_scheduled[:self.settings.max_tasks_per_tick]:
            try:
                result = self.runner.run_task(task["id"], self._deny_side_effects)
                results.append(result)
                self.logger.info(
                    "Scheduled AETHER task %s finished with status %s",
                    task["id"], result.get("status", "unknown"),
                )
            except Exception:
                self.logger.exception("Scheduled AETHER task %s failed", task["id"])
                results.append({"task_id": task["id"], "status": "failed"})
        return {
            "status": "completed", "enqueued": enqueued, "results": results,
            "discovered_goals": [goal["id"] for goal in discovered],
            "goal_results": goal_results,
        }

    def run_forever(self, stop_event: Event) -> None:
        self.logger.info(
            "AETHER scheduler started; poll interval=%s seconds",
            self.settings.poll_interval_seconds,
        )
        while not stop_event.wait(self.settings.poll_interval_seconds):
            try:
                self.run_once()
            except Exception:
                self.logger.exception("AETHER scheduler tick failed")
        self.logger.info("AETHER scheduler stopped")
