"""Opt-in bounded scheduler for pre-approved read-only autonomy goals."""
from __future__ import annotations

import logging
from threading import Event
from typing import Any

from aether.config import SchedulerSettings
from aether.storage import Store


class AutonomyScheduler:
    """Runs persisted scheduled goals without granting write or code-execution approval."""

    def __init__(
        self, store: Store, runner: Any, settings: SchedulerSettings,
        logger: logging.Logger | None = None,
    ):
        self.store = store
        self.runner = runner
        self.settings = settings
        self.logger = logger or logging.getLogger(__name__)

    @staticmethod
    def _deny_side_effects(tool: str, purpose: str, args: dict[str, Any]) -> bool:
        # Scheduled autonomy is intentionally read-only. It cannot approve writes or tests.
        return False

    def run_once(self) -> dict[str, Any]:
        if self.store.emergency_stop_enabled():
            return {"status": "emergency_stop", "enqueued": [], "results": []}
        if not self.settings.enabled:
            return {"status": "disabled", "enqueued": [], "results": []}
        enqueued = self.store.enqueue_due_schedules(
            limit=self.settings.max_tasks_per_tick
        )
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
        return {"status": "completed", "enqueued": enqueued, "results": results}

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
