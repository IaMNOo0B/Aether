"""Durable task lifecycle and checkpoint validation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from aether.storage import Store


ALLOWED_TRANSITIONS = {
    "queued": {"running", "cancelled"},
    "running": {"completed", "failed", "cancelled"},
    "failed": {"queued", "cancelled"},
    "completed": set(),
    "cancelled": set(),
}


@dataclass(frozen=True)
class Task:
    id: str
    objective: str
    status: str
    state: dict[str, Any]


class TaskManager:
    def __init__(self, store: Store):
        self.store = store

    def create(self, objective: str, state: dict[str, Any] | None = None) -> Task:
        text = objective.strip()
        if not text:
            raise ValueError("Task objective cannot be empty")
        task_id = uuid4().hex
        initial_state = state or {}
        self.store.save_task(task_id, text, "queued", initial_state)
        return Task(task_id, text, "queued", initial_state)

    def transition(
        self,
        task_id: str,
        new_status: str,
        state: dict[str, Any] | None = None,
    ) -> Task:
        current = self.store.get_task(task_id)
        if current is None:
            raise KeyError(f"Unknown task: {task_id}")
        allowed = ALLOWED_TRANSITIONS.get(current["status"])
        if allowed is None or new_status not in allowed:
            raise ValueError(
                f"Invalid task transition: {current['status']} -> {new_status}"
            )
        next_state = current["state"] if state is None else state
        self.store.save_task(task_id, current["objective"], new_status, next_state)
        return Task(task_id, current["objective"], new_status, next_state)
