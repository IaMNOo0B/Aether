import json

import pytest

from aether.autonomy import AutonomyRunner
from aether.config import AutonomySettings, LimitSettings, ResearchSettings, SchedulerSettings
from aether.goals import GoalManager
from aether.scheduler import AutonomyScheduler
from aether.storage import Store
from aether.tasks import TaskManager


class FakeProvider:
    def __init__(self, response):
        self.response = response

    def chat(self, messages):
        return self.response


def make_goal(manager, **overrides):
    values = {
        "title": "Inspect the project workspace",
        "purpose": "List the configured workspace and report its contents.",
        "source": "operator",
        "source_key": "operator:test-goal",
        "expected_benefit": "Identify current project artifacts.",
        "priority": 0.7,
        "complexity": 1,
        "risk_score": 0.05,
        "risks": ["Read-only inspection."],
        "completion_criteria": ["Workspace listing is returned and tool result is verified."],
        "stop_condition": "Stop after one verified listing.",
    }
    values.update(overrides)
    return manager.create(**values)


def test_goal_discovery_creates_proposals_but_never_approves_them(tmp_path):
    store = Store(tmp_path / "goals.sqlite3")
    tasks = TaskManager(store)
    failed = tasks.create("Investigate a broken read-only task", {"kind": "autonomy"})
    tasks.transition(failed.id, "running")
    tasks.transition(failed.id, "failed", {"kind": "autonomy", "error": "missing file"})

    manager = GoalManager(store)
    proposals = manager.discover()

    assert len(proposals) == 1
    assert proposals[0]["source"] == "failed_task"
    assert proposals[0]["status"] == "proposed"
    assert manager.discover()[0]["id"] == proposals[0]["id"]
    assert store.list_autonomy_goals(status="approved") == []


def test_goal_requires_measurable_criteria_and_valid_budgets(tmp_path):
    manager = GoalManager(Store(tmp_path / "goals.sqlite3"))
    with pytest.raises(ValueError, match="completion criterion"):
        make_goal(manager, completion_criteria=[])
    with pytest.raises(ValueError, match="positive number"):
        make_goal(manager, source_key="bad-budget", resource_budget={"max_steps": 0})
    with pytest.raises(ValueError, match="risk_score"):
        make_goal(manager, source_key="bad-risk", risk_score=1.5)


def test_goal_priority_is_transparent_and_bounded():
    goal = {"priority": 1.0, "complexity": 1, "risk_score": 0.0, "deadline_at": None}
    score = GoalManager.priority_score(goal)
    assert 0 <= score <= 1
    assert score == 0.85


def test_dependencies_block_auto_queue_until_parent_completes(tmp_path):
    store = Store(tmp_path / "goals.sqlite3")
    manager = GoalManager(store)
    parent = make_goal(manager, source_key="parent")
    child = make_goal(
        manager, source_key="child", title="Finish child work",
        dependencies=[parent["id"]], parent_goal_id=parent["id"],
    )
    manager.approve(child["id"])
    assert manager.enqueue_approved() == []
    assert store.get_autonomy_goal(child["id"])["status"] == "blocked"

    manager.approve(parent["id"])
    manager.transition(parent["id"], "queued")
    manager.transition(parent["id"], "running")
    manager.transition(parent["id"], "completed")
    assert manager.unblock_ready() == []
    manager.review_completion(parent["id"], True, "Parent acceptance criteria independently checked.")
    assert manager.unblock_ready() == [child["id"]]
    assert store.get_autonomy_goal(child["id"])["status"] == "approved"


def test_high_risk_goal_never_enters_unattended_queue(tmp_path):
    store = Store(tmp_path / "goals.sqlite3")
    manager = GoalManager(store)
    goal = make_goal(manager, risk_score=0.8)
    manager.approve(goal["id"])
    assert manager.enqueue_approved() == []
    assert store.get_autonomy_goal(goal["id"])["status"] == "approved"


def test_scheduler_runs_only_approved_low_risk_goals_and_records_receipts(tmp_path):
    store = Store(tmp_path / "goals.sqlite3")
    manager = GoalManager(store)
    goal = make_goal(manager)
    manager.approve(goal["id"])
    provider = FakeProvider(json.dumps({"steps": [
        {"id": "list", "depends_on": [], "purpose": "Inspect workspace",
         "tool": "workspace.list", "args": {"path": "."}}
    ]}))
    runner = AutonomyRunner(
        provider, store, AutonomySettings(enabled=True, allow_read_only=True),
        LimitSettings(), ResearchSettings(), tmp_path,
    )
    scheduler = AutonomyScheduler(
        store, runner, SchedulerSettings(enabled=True, max_tasks_per_tick=1),
        goal_manager=manager,
    )

    result = scheduler.run_once()

    assert result["goal_results"][0]["status"] == "completed"
    saved = store.get_autonomy_goal(goal["id"])
    assert saved["status"] == "completed"
    task = store.get_task(saved["task_id"])
    assert task["status"] == "completed"
    assert task["state"]["receipts"][0]["verification"]["passed"] is True
    assert store.list_notifications(unread_only=True)


def test_scheduler_discovers_but_does_not_auto_approve_goals(tmp_path):
    store = Store(tmp_path / "goals.sqlite3")
    manager = GoalManager(store)
    tasks = TaskManager(store)
    failed = tasks.create("Previous task", {"kind": "autonomy"})
    tasks.transition(failed.id, "running")
    tasks.transition(failed.id, "failed", {"kind": "autonomy"})

    runner = AutonomyRunner(
        FakeProvider('{"steps":[{"id":"x","depends_on":[],"tool":"workspace.list","args":{"path":"."}}]}'),
        store, AutonomySettings(enabled=True), LimitSettings(), ResearchSettings(), tmp_path,
    )
    scheduler = AutonomyScheduler(
        store, runner, SchedulerSettings(enabled=True), goal_manager=manager
    )
    result = scheduler.run_once()

    assert result["discovered_goals"]
    assert result["goal_results"] == []
    assert store.list_autonomy_goals(status="approved") == []


def test_task_cancellation_is_cooperative_and_stops_at_step_boundary(tmp_path):
    store = Store(tmp_path / "goals.sqlite3")
    runner = AutonomyRunner(
        FakeProvider(json.dumps({"steps": [
            {"id": "one", "depends_on": [], "tool": "workspace.list", "args": {"path": "."}},
            {"id": "two", "depends_on": ["one"], "tool": "workspace.list", "args": {"path": "."}},
        ]})),
        store, AutonomySettings(enabled=True), LimitSettings(), ResearchSettings(), tmp_path,
    )
    original = runner.gateway.execute
    calls = []

    def execute(tool, args, approval=None):
        calls.append(tool)
        result = original(tool, args, approval)
        if len(calls) == 1:
            running = next(task for task in store.list_tasks(status="running") if task["state"].get("kind") == "autonomy")
            store.request_task_cancel(running["id"], "test cancellation")
        return result

    runner.gateway.execute = execute
    result = runner.run("Run two read-only steps")

    assert result["status"] == "cancelled"
    assert len(calls) == 1
    assert store.get_task(result["task_id"])["status"] == "cancelled"


def test_goal_and_notification_events_are_persistent(tmp_path):
    database = tmp_path / "goals.sqlite3"
    store = Store(database)
    manager = GoalManager(store)
    goal = make_goal(manager)
    manager.approve(goal["id"])
    events = store.autonomy_goal_events(goal["id"])
    assert [event["to_status"] for event in events] == ["proposed", "approved"]

    reopened = Store(database)
    assert reopened.get_autonomy_goal(goal["id"])["status"] == "approved"
    notifications = reopened.list_notifications(unread_only=True)
    assert len(notifications) >= 2
    for notification in notifications:
        reopened.mark_notification_read(notification["id"])
    assert reopened.list_notifications(unread_only=True) == []

def test_scheduler_enforces_per_goal_step_budget(tmp_path):
    store = Store(tmp_path / "goals.sqlite3")
    manager = GoalManager(store)
    goal = make_goal(
        manager, resource_budget={"max_steps": 1, "max_runtime_seconds": 30, "max_model_calls": 2}
    )
    manager.approve(goal["id"])
    task_id = store.enqueue_approved_autonomy_goal(goal["id"], "budget-task")
    manager.transition(goal["id"], "running", {"task_id": task_id}, task_id)
    provider = FakeProvider(json.dumps({"steps": [
        {"id": "one", "depends_on": [], "tool": "workspace.list", "args": {"path": "."}},
        {"id": "two", "depends_on": ["one"], "tool": "workspace.list", "args": {"path": "."}},
    ]}))
    runner = AutonomyRunner(
        provider, store, AutonomySettings(enabled=True),
        LimitSettings(), ResearchSettings(), tmp_path,
    )

    result = runner.run_task(task_id)

    assert result["status"] == "cancelled"
    assert "budget allows 1" in result["reason"]
    saved = store.get_task(task_id)
    assert saved["status"] == "cancelled"
    assert saved["state"]["plan"]
    assert saved["state"]["resource_limit"]

def test_goal_reconciliation_marks_interrupted_running_task_failed(tmp_path):
    store = Store(tmp_path / "goals.sqlite3")
    manager = GoalManager(store)
    goal = make_goal(manager)
    manager.approve(goal["id"])
    task_id = store.enqueue_approved_autonomy_goal(goal["id"], "interrupted-goal-task")
    manager.transition(goal["id"], "running", {"task_id": task_id}, task_id)
    tasks = TaskManager(store)
    tasks.transition(task_id, "running", {"kind": "autonomy", "goal_id": goal["id"]})

    recovered = tasks.recover_interrupted()
    reconciled = manager.reconcile()

    assert recovered == [task_id]
    assert len(reconciled) == 1
    assert store.get_autonomy_goal(goal["id"])["status"] == "failed"

def test_goal_cannot_satisfy_dependencies_until_completion_is_reviewed(tmp_path):
    store = Store(tmp_path / "goals.sqlite3")
    manager = GoalManager(store)
    goal = make_goal(manager)
    manager.approve(goal["id"])
    manager.transition(goal["id"], "queued")
    manager.transition(goal["id"], "running")
    manager.transition(goal["id"], "completed")

    saved = store.get_autonomy_goal(goal["id"])
    assert saved["completion_review_status"] == "pending_review"
    assert not manager.dependencies_satisfied({
        "dependencies": [goal["id"]], "parent_goal_id": None
    })

    verified = manager.review_completion(
        goal["id"], True, "All stated completion criteria were manually checked."
    )
    assert verified["completion_review_status"] == "verified"
    assert manager.dependencies_satisfied({
        "dependencies": [goal["id"]], "parent_goal_id": None
    })


def test_rejected_goal_completion_returns_goal_to_failed_state(tmp_path):
    store = Store(tmp_path / "goals.sqlite3")
    manager = GoalManager(store)
    goal = make_goal(manager)
    manager.approve(goal["id"])
    manager.transition(goal["id"], "queued")
    manager.transition(goal["id"], "running")
    manager.transition(goal["id"], "completed")

    rejected = manager.review_completion(
        goal["id"], False, "The requested outcome was not achieved."
    )
    assert rejected["status"] == "failed"
    assert rejected["completion_review_status"] == "rejected"

def test_goal_resource_budgets_have_hard_upper_bounds(tmp_path):
    manager = GoalManager(Store(tmp_path / "goals.sqlite3"))
    with pytest.raises(ValueError, match="cannot exceed 3600"):
        make_goal(
            manager, source_key="too-long",
            resource_budget={"max_steps": 5, "max_runtime_seconds": 3601, "max_model_calls": 8},
        )
    with pytest.raises(ValueError, match="Unsupported goal resource budget fields"):
        make_goal(
            manager, source_key="unknown-budget",
            resource_budget={"max_steps": 5, "max_runtime_seconds": 30, "max_model_calls": 8, "unlimited": True},
        )

def test_goal_replanning_rejects_dependency_cycles(tmp_path):
    store = Store(tmp_path / "goals.sqlite3")
    manager = GoalManager(store)
    first = make_goal(manager, source_key="first")
    second = make_goal(
        manager, source_key="second", dependencies=[first["id"]],
        title="Dependent goal",
    )
    with pytest.raises(ValueError, match="cycle"):
        manager.replan(
            first["id"], purpose="Create a circular dependency",
            expected_benefit=first["expected_benefit"], priority=first["priority"],
            complexity=first["complexity"], risk_score=first["risk_score"],
            resource_budget=first["resource_budget"], risks=first["risks"],
            deadline_at=first["deadline_at"], dependencies=[second["id"]],
            completion_criteria=first["completion_criteria"],
            stop_condition=first["stop_condition"],
        )

def test_scheduler_uses_one_combined_tick_budget_for_goals_and_schedules(tmp_path):
    store = Store(tmp_path / "goals.sqlite3")
    manager = GoalManager(store)
    goal = make_goal(manager)
    manager.approve(goal["id"])
    store.create_schedule("one-schedule", "Run scheduled read-only report", 5, max_runs=1, now_epoch=0)
    provider = FakeProvider(json.dumps({"steps": [
        {"id": "list", "depends_on": [], "purpose": "List workspace",
         "tool": "workspace.list", "args": {"path": "."}}
    ]}))
    runner = AutonomyRunner(
        provider, store, AutonomySettings(enabled=True, allow_read_only=True),
        LimitSettings(), ResearchSettings(), tmp_path,
    )
    scheduler = AutonomyScheduler(
        store, runner, SchedulerSettings(enabled=True, max_tasks_per_tick=1),
        goal_manager=manager,
    )

    result = scheduler.run_once()

    assert len(result["goal_results"]) == 1
    assert result["results"] == []
    schedule = store.list_schedules()[0]
    assert schedule["run_count"] == 0
