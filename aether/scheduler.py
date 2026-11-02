"""Opt-in bounded scheduler for approved goals and read-only scheduled objectives."""
from __future__ import annotations

import logging
from threading import Event
from typing import Any

from aether.config import SchedulerSettings
from aether.goals import GoalManager
from aether.storage import Store


class AutonomyScheduler:
    """Runs only explicitly approved goals and read-only schedules."""

    def __init__(
        self, store: Store, runner: Any, settings: SchedulerSettings,
        logger: logging.Logger | None = None,
    ):
        self.store = store
        self.runner = runner
        self.settings = settings
        self.goals = GoalManager(store)
        self.logger = logger or logging.getLogger(__name__)

    @staticmethod
    def _deny_side_effects(tool: str, purpose: str, args: dict[str, Any]) -> bool:
        return False

    def run_once(self) -> dict[str, Any]:
        if self.store.emergency_stop_enabled():
            return {"status": "emergency_stop", "discovered": [], "enqueued": [], "results": []}
        if not self.settings.enabled:
            return {"status": "disabled", "discovered": [], "enqueued": [], "results": []}

        discovered = self.goals.discover(limit=self.settings.max_tasks_per_tick)
        schedule_tasks = self.store.enqueue_due_schedules(
            limit=self.settings.max_tasks_per_tick
        )
        approved_goal_tasks = self.goals.queue_approved(
            limit=self.settings.max_tasks_per_tick
        )
        enqueued = schedule_tasks + approved_goal_tasks
        queued = [
            task for task in self.store.list_tasks(status="queued", limit=200)
            if task["state"].get("scheduled") is True or task["state"].get("goal_id")
        ]
        results = []
        for task in queued[:self.settings.max_tasks_per_tick]:
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
            "status": "completed",
            "discovered": [goal["id"] for goal in discovered],
            "enqueued": enqueued,
            "results": results,
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
