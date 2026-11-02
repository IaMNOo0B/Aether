"""Durable task lifecycle, cooperative controls, checkpoints and recovery."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from aether.storage import Store


ALLOWED_TRANSITIONS = {
    "queued": {"running", "paused", "cancelled"},
    "running": {"completed", "failed", "cancelled", "paused"},
    "paused": {"queued", "cancelled"},
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
        self.store.create_task(task_id, text, initial_state)
        return Task(task_id, text, "queued", initial_state)

    def transition(self, task_id: str, new_status: str,
                   state: dict[str, Any] | None = None) -> Task:
        current = self.store.get_task(task_id)
        if current is None:
            raise KeyError(f"Unknown task: {task_id}")
        allowed = ALLOWED_TRANSITIONS.get(current["status"])
        if allowed is None or new_status not in allowed:
            raise ValueError(f"Invalid task transition: {current['status']} -> {new_status}")
        next_state = current["state"] if state is None else state
        updated = self.store.transition_task(task_id, current["status"], new_status, next_state)
        return Task(updated["id"], updated["objective"], updated["status"], updated["state"])

    def checkpoint(self, task_id: str, state: dict[str, Any],
                   event_data: dict[str, Any] | None = None) -> Task:
        updated = self.store.checkpoint_task(task_id, state, event_data)
        return Task(updated["id"], updated["objective"], updated["status"], updated["state"])

    def pause(self, task_id: str, reason: str = "Paused by user") -> Task:
        current = self.store.get_task(task_id)
        if current is None:
            raise KeyError(f"Unknown task: {task_id}")
        if current["status"] == "queued":
            state = {**current["state"], "pause_reason": reason[:500]}
            return self.transition(task_id, "paused", state)
        if current["status"] == "running":
            updated = self.store.request_task_control(task_id, "pause", reason)
            return Task(updated["id"], updated["objective"], updated["status"], updated["state"])
        if current["status"] == "paused":
            return Task(current["id"], current["objective"], current["status"], current["state"])
        raise ValueError(f"Cannot pause task in status {current['status']}")

    def resume(self, task_id: str) -> Task:
        current = self.store.get_task(task_id)
        if current is None:
            raise KeyError(f"Unknown task: {task_id}")
        if current["status"] != "paused":
            raise ValueError(f"Only paused tasks can be resumed; found {current['status']}")
        state = dict(current["state"])
        state.pop("pause_requested", None)
        state.pop("pause_reason", None)
        state.pop("cancel_requested", None)
        state.pop("cancel_reason", None)
        return self.transition(task_id, "queued", state)

    def cancel(self, task_id: str, reason: str = "Cancelled by user") -> Task:
        current = self.store.get_task(task_id)
        if current is None:
            raise KeyError(f"Unknown task: {task_id}")
        if current["status"] in {"queued", "paused"}:
            return self.transition(task_id, "cancelled", {**current["state"], "cancel_reason": reason[:500]})
        if current["status"] == "running":
            updated = self.store.request_task_control(task_id, "cancel", reason)
            return Task(updated["id"], updated["objective"], updated["status"], updated["state"])
        raise ValueError(f"Cannot cancel task in status {current['status']}")

    def events(self, task_id: str, limit: int = 500) -> list[dict[str, Any]]:
        if self.store.get_task(task_id) is None:
            raise KeyError(f"Unknown task: {task_id}")
        return self.store.task_events(task_id, limit)

    def recover_interrupted(self) -> list[str]:
        """Recover process-interrupted tasks without replaying uncertain side effects."""
        return self.store.recover_interrupted_tasks()
