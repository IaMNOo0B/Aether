import json

import pytest

from aether.autonomy import AutonomyRunner
from aether.config import AutonomySettings, LimitSettings, ResearchSettings, SchedulerSettings
from aether.goals import GoalManager
from aether.scheduler import AutonomyScheduler
from aether.storage import Store


class FakeProvider:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        return self.response


def _runner(tmp_path, store, response):
    return AutonomyRunner(
        FakeProvider(response), store,
        AutonomySettings(enabled=True, allow_read_only=True, allow_workspace_writes=True),
        LimitSettings(max_autonomy_steps=5), ResearchSettings(), tmp_path,
    )


def test_goal_discovery_creates_proposal_without_auto_approval_or_execution(tmp_path):
    store = Store(tmp_path / "goals.sqlite3")
    failed = store.create_task("failed-task-1", "Collect source evidence", {
        "kind": "autonomy", "error": "provider timeout", "receipts": [],
    })
    store.transition_task("failed-task-1", "queued", "running", failed and {"kind": "autonomy"})
    store.transition_task("failed-task-1", "running", "failed", {
        "kind": "autonomy", "error": "provider timeout", "receipts": [],
    })

    manager = GoalManager(store)
    first = manager.discover()
    second = manager.discover()

    assert len(first) == 1
    assert first[0]["status"] == "proposed"
    assert second[0]["id"] == first[0]["id"]
    assert store.get_task("failed-task-1")["status"] == "failed"
    assert manager.queue_approved() == []
    assert store.list_notifications(unread_only=True)


def test_goal_approval_queues_with_resource_budget_and_dependency_gate(tmp_path):
    store = Store(tmp_path / "goal-queue.sqlite3")
    manager = GoalManager(store)
    parent = manager.propose(
        title="Parent", purpose="Inspect public documentation", source="user",
        source_key="parent", expected_benefit="Reduce uncertainty", priority=0.9,
        completion_criteria=["A report exists"],
    )
    child = manager.propose(
        title="Child", purpose="Compare documented options", source="user",
        source_key="child", expected_benefit="Choose a method", priority=0.8,
        dependencies=[parent["id"]],
        resource_budget={"max_runtime_seconds": 90, "max_tool_calls": 2},
    )
    manager.approve(child["id"])
    assert manager.queue_approved() == []

    manager.approve(parent["id"])
    queued = manager.queue_approved(limit=2)
    assert len(queued) == 1
    task = store.get_task(queued[0])
    assert task["state"]["resource_budget"] == {
        "max_runtime_seconds": 300, "max_tool_calls": 8, "max_model_calls": 1
    }
    assert manager.prioritize()[0]["id"] == parent["id"]


def test_pause_resume_continues_from_checkpoint_without_replanning_or_repeating_step(tmp_path):
    store = Store(tmp_path / "pause-resume.sqlite3")
    plan = json.dumps({"steps": [
        {"purpose": "Create a note", "tool": "workspace.write",
         "args": {"path": "note.txt", "content": "once"}},
        {"purpose": "Read the note", "tool": "workspace.read", "args": {"path": "note.txt"}},
    ]})
    runner = _runner(tmp_path, store, plan)
    task = runner.enqueue("Create and verify a note")
    original_execute = runner.gateway.execute
    requested = {"done": False}

    def execute_and_request_pause(tool, args, approval=None):
        result = original_execute(tool, args, approval)
        if not requested["done"]:
            requested["done"] = True
            runner.tasks.pause(task.id, "Pause after first durable receipt")
        return result

    runner.gateway.execute = execute_and_request_pause
    first = runner.run_task(task.id, lambda *args: True)
    assert first["status"] == "paused"
    paused = store.get_task(task.id)
    assert paused["state"]["current_step"] == 1
    assert len(paused["state"]["receipts"]) == 1
    assert (tmp_path / "workspace" / "note.txt").read_text() == "once"

    runner.gateway.execute = original_execute
    runner.tasks.resume(task.id)
    second = runner.run_task(task.id, lambda *args: True)

    assert second["status"] == "completed"
    assert len(second["receipts"]) == 2
    assert runner.provider.calls == 1
    assert (tmp_path / "workspace" / "note.txt").read_text() == "once"
    assert store.get_task(task.id)["status"] == "completed"


def test_tool_call_budget_stops_task_and_records_notification(tmp_path):
    store = Store(tmp_path / "budget.sqlite3")
    runner = _runner(tmp_path, store, json.dumps({"steps": [
        {"purpose": "List workspace", "tool": "workspace.list", "args": {"path": "."}},
        {"purpose": "List workspace again", "tool": "workspace.list", "args": {"path": "."}},
    ]}))
    task = runner.tasks.create("Bounded task", {
        "kind": "autonomy", "receipts": [],
        "resource_budget": {"max_runtime_seconds": 30, "max_tool_calls": 1, "max_model_calls": 1},
    })
    result = runner.run_task(task.id)
    assert result["status"] == "cancelled"
    assert "configured tool-call budget" in result["reason"]
    assert len(result["receipts"]) == 0
    assert any(item["kind"] == "resource_budget_exhausted"
               for item in store.list_notifications(unread_only=True))


def test_running_task_cancel_is_cooperative_and_audited(tmp_path):
    store = Store(tmp_path / "cancel.sqlite3")
    runner = _runner(tmp_path, store, json.dumps({"steps": [
        {"purpose": "List workspace", "tool": "workspace.list", "args": {"path": "."}},
        {"purpose": "List again", "tool": "workspace.list", "args": {"path": "."}},
    ]}))
    task = runner.enqueue("Cancel after first step")
    original_execute = runner.gateway.execute
    requested = {"done": False}

    def execute_and_cancel(tool, args, approval=None):
        result = original_execute(tool, args, approval)
        if not requested["done"]:
            requested["done"] = True
            runner.tasks.cancel(task.id, "Test cancellation")
        return result

    runner.gateway.execute = execute_and_cancel
    result = runner.run_task(task.id)
    assert result["status"] == "cancelled"
    assert len(result["receipts"]) == 1
    assert runner.tasks.events(task.id)[-1]["to_status"] == "cancelled"


def test_scheduler_executes_only_approved_goal_and_emits_notifications(tmp_path):
    store = Store(tmp_path / "goal-scheduler.sqlite3")
    manager = GoalManager(store)
    goal = manager.propose(
        title="Read workspace", purpose="List workspace files", source="user",
        source_key="scheduler-goal", expected_benefit="Inventory workspace",
        completion_criteria=["Directory listing recorded"],
    )
    manager.approve(goal["id"])
    runner = _runner(tmp_path, store, json.dumps({"steps": [
        {"purpose": "List files", "tool": "workspace.list", "args": {"path": "."}}
    ]}))
    scheduler = AutonomyScheduler(
        store, runner, SchedulerSettings(enabled=True, max_tasks_per_tick=1)
    )

    result = scheduler.run_once()

    assert len(result["enqueued"]) == 1
    assert result["results"][0]["status"] == "completed"
    assert store.get_autonomy_goal(goal["id"])["status"] == "completed"
    assert any(item["kind"] == "task_completed"
               for item in store.list_notifications(unread_only=True))


def test_missing_dependencies_do_not_queue(tmp_path):
    store = Store(tmp_path / "cycle.sqlite3")
    manager = GoalManager(store)
    first = manager.propose(
        title="First", purpose="First objective", source="user", source_key="cycle-a",
        expected_benefit="Benefit", dependencies=["placeholder"],
    )
    second = manager.propose(
        title="Second", purpose="Second objective", source="user", source_key="cycle-b",
        expected_benefit="Benefit", dependencies=[first["id"]],
    )
    # Missing dependencies are never considered satisfied.
    manager.approve(first["id"])
    manager.approve(second["id"])
    assert manager.queue_approved(limit=10) == []


def test_replan_preserves_history_and_only_allows_read_only_completed_steps(tmp_path):
    store = Store(tmp_path / "replan.sqlite3")
    from aether.tasks import TaskManager
    manager = TaskManager(store)
    task = manager.create("Original research objective", {"kind": "autonomy"})
    manager.transition(task.id, "running")
    manager.checkpoint(task.id, {
        "kind": "autonomy",
        "plan": [{"tool": "workspace.read", "args": {"path": "notes.txt"}}],
        "receipts": [{"tool": "workspace.read", "result": {"content": "evidence"}}],
        "current_step": 1,
    })
    manager.transition(task.id, "paused")
    replanned = manager.replan(task.id, "Revised research objective")
    assert replanned.status == "queued"
    assert replanned.objective == "Revised research objective"
    assert replanned.state["plan"] == []
    assert replanned.state["receipts"] == []
    assert replanned.state["replan_history"][0]["receipts"][0]["tool"] == "workspace.read"
    assert manager.events(task.id)[-1]["event_type"] == "replanned"


def test_replan_refuses_to_repeat_possible_side_effects(tmp_path):
    from aether.tasks import TaskManager
    store = Store(tmp_path / "unsafe-replan.sqlite3")
    manager = TaskManager(store)
    task = manager.create("Task with side effect", {"kind": "autonomy"})
    manager.transition(task.id, "running")
    manager.checkpoint(task.id, {
        "kind": "autonomy",
        "plan": [{"tool": "workspace.write", "args": {"path": "a.txt"}}],
        "receipts": [{"tool": "workspace.write", "result": {"path": "a.txt"}}],
        "current_step": 1,
    })
    manager.transition(task.id, "paused")
    with pytest.raises(PermissionError, match="side effects"):
        manager.replan(task.id)


def test_capability_gap_discovery_creates_system_goal_proposal(tmp_path):
    from aether.self_model import SelfModel
    store = Store(tmp_path / "capability-goal.sqlite3")
    SelfModel(store.path).register_capability(
        "vision.ocr", "Vision OCR", "Read text from screenshots",
        state="implemented_unverified",
    )
    found = GoalManager(store).discover()
    proposal = next(item for item in found if item["source"] == "capability_gap")
    assert proposal["goal_level"] == "system"
    assert proposal["status"] == "proposed"
    assert "Do not modify production code" in proposal["purpose"]


def test_notification_can_be_marked_read(tmp_path):
    store = Store(tmp_path / "notifications.sqlite3")
    manager = GoalManager(store)
    goal = manager.propose(
        title="Notify", purpose="Review a task", source="user",
        source_key="notify-1", expected_benefit="Keep user informed",
    )
    unread = store.list_notifications(unread_only=True)
    assert unread
    store.mark_notification_read(unread[0]["id"])
    assert store.list_notifications(unread_only=True) == []
