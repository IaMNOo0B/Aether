"""Dependency-aware task plans with deterministic validation and progress persistence."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from aether.storage import Store
from aether.tasks import Task, TaskManager


@dataclass(frozen=True)
class PlanStep:
    id: str
    description: str
    depends_on: tuple[str, ...] = ()


class PlanGraph:
    def __init__(self, steps: Iterable[PlanStep]):
        self.steps = tuple(steps)
        self._by_id = {step.id: step for step in self.steps}
        self._validate()

    def _validate(self) -> None:
        if not self.steps:
            raise ValueError("A plan must contain at least one step")
        if len(self._by_id) != len(self.steps):
            raise ValueError("Plan step IDs must be unique")
        for step in self.steps:
            if not step.id.strip() or not step.description.strip():
                raise ValueError("Plan step IDs and descriptions cannot be empty")
            if step.id in step.depends_on:
                raise ValueError(f"Step cannot depend on itself: {step.id}")
            missing = set(step.depends_on) - self._by_id.keys()
            if missing:
                raise ValueError(f"Step {step.id} has unknown dependencies: {sorted(missing)}")

        remaining = {step.id: set(step.depends_on) for step in self.steps}
        resolved: set[str] = set()
        while remaining:
            ready = [step_id for step_id, deps in remaining.items() if deps <= resolved]
            if not ready:
                raise ValueError("Plan dependencies contain a cycle")
            for step_id in ready:
                resolved.add(step_id)
                del remaining[step_id]

    def topological_order(self) -> tuple[str, ...]:
        remaining = {step.id: set(step.depends_on) for step in self.steps}
        ordered: list[str] = []
        while remaining:
            ready = [step.id for step in self.steps
                     if step.id in remaining and remaining[step.id] <= set(ordered)]
            for step_id in ready:
                ordered.append(step_id)
                del remaining[step_id]
        return tuple(ordered)

    def ready_steps(self, completed: Iterable[str] = ()) -> tuple[str, ...]:
        completed_set = set(completed)
        unknown = completed_set - self._by_id.keys()
        if unknown:
            raise ValueError(f"Unknown completed steps: {sorted(unknown)}")
        return tuple(
            step.id for step in self.steps
            if step.id not in completed_set
            and set(step.depends_on) <= completed_set
        )

    def serialize(self) -> list[dict[str, object]]:
        return [
            {"id": step.id, "description": step.description,
             "depends_on": list(step.depends_on)}
            for step in self.steps
        ]


class PlanManager:
    """Persists a dependency graph inside a durable task checkpoint."""

    def __init__(self, store: Store):
        self.store = store
        self.tasks = TaskManager(store)

    def create(self, objective: str, steps: Iterable[PlanStep]) -> Task:
        graph = PlanGraph(steps)
        task = self.tasks.create(objective, {
            "plan": graph.serialize(),
            "completed_steps": [],
        })
        return task

    def graph_for(self, task_id: str) -> tuple[PlanGraph, set[str]]:
        task = self.store.get_task(task_id)
        if task is None:
            raise KeyError(f"Unknown task: {task_id}")
        state = task["state"]
        raw_steps = state.get("plan")
        if not isinstance(raw_steps, list):
            raise ValueError(f"Task {task_id} does not contain a plan")
        graph = PlanGraph(
            PlanStep(
                id=str(item["id"]),
                description=str(item["description"]),
                depends_on=tuple(item.get("depends_on", [])),
            )
            for item in raw_steps
        )
        completed = set(state.get("completed_steps", []))
        graph.ready_steps(completed)  # validates completed step IDs
        return graph, completed

    def ready_steps(self, task_id: str) -> tuple[str, ...]:
        graph, completed = self.graph_for(task_id)
        return graph.ready_steps(completed)

    def complete_step(self, task_id: str, step_id: str) -> tuple[str, ...]:
        task = self.store.get_task(task_id)
        if task is None:
            raise KeyError(f"Unknown task: {task_id}")
        graph, completed = self.graph_for(task_id)
        if step_id not in graph.ready_steps(completed):
            raise ValueError(f"Step is not ready to complete: {step_id}")
        completed.add(step_id)
        state = dict(task["state"])
        state["completed_steps"] = [
            step.id for step in graph.steps if step.id in completed
        ]
        self.store.save_task(task_id, task["objective"], task["status"], state)
        return graph.ready_steps(completed)
