import json

from aether.autonomy import AutonomyRunner
from aether.config import AutonomySettings, LimitSettings, ResearchSettings
from aether.storage import Store


class FakeProvider:
    def __init__(self, response):
        self.response = response

    def chat(self, messages):
        return self.response


def test_autonomy_executes_approved_workspace_tools_and_persists_receipts(tmp_path):
    response = json.dumps({"steps": [
        {"purpose": "Create a reversible note", "tool": "workspace.write",
         "args": {"path": "notes/result.txt", "content": "research complete"}},
        {"purpose": "Read back the note", "tool": "workspace.read",
         "args": {"path": "notes/result.txt"}},
    ]})
    store = Store(tmp_path / "aether.sqlite3")
    policy = AutonomySettings(
        enabled=True, allow_read_only=True, allow_workspace_writes=True,
        allow_code_execution=False, workspace="workspace",
    )
    runner = AutonomyRunner(
        FakeProvider(response), store, policy, LimitSettings(), ResearchSettings(), tmp_path
    )
    result = runner.run("Write a short note", lambda *args: True)
    assert result["status"] == "completed"
    assert len(result["receipts"]) == 2
    assert result["receipts"][1]["result"]["content"] == "research complete"
    assert (tmp_path / "workspace" / "notes" / "result.txt").read_text() == "research complete"
    assert store.get_task(result["task_id"])["status"] == "completed"


def test_autonomy_declined_write_is_cancelled_without_side_effect(tmp_path):
    response = json.dumps({"steps": [
        {"purpose": "Write a file", "tool": "workspace.write",
         "args": {"path": "blocked.txt", "content": "no"}},
    ]})
    store = Store(tmp_path / "aether.sqlite3")
    policy = AutonomySettings(enabled=True, allow_workspace_writes=True)
    runner = AutonomyRunner(
        FakeProvider(response), store, policy, LimitSettings(), ResearchSettings(), tmp_path
    )
    result = runner.run("Write a file", lambda *args: False)
    assert result["status"] == "cancelled"
    assert not (tmp_path / "workspace" / "blocked.txt").exists()


def test_autonomy_rejects_unknown_model_generated_tools(tmp_path):
    response = json.dumps({"steps": [
        {"purpose": "Run arbitrary command", "tool": "shell.exec", "args": {"command": "whoami"}},
    ]})
    store = Store(tmp_path / "aether.sqlite3")
    policy = AutonomySettings(enabled=True)
    runner = AutonomyRunner(
        FakeProvider(response), store, policy, LimitSettings(), ResearchSettings(), tmp_path
    )
    try:
        runner.run("Execute a command")
    except PermissionError:
        pass
    else:
        raise AssertionError("Unknown tool should be denied")


def test_autonomy_gateway_blocks_workspace_path_traversal(tmp_path):
    from aether.autonomy import ToolDenied

    store = Store(tmp_path / "aether.sqlite3")
    runner = AutonomyRunner(
        FakeProvider('{"steps":[{"tool":"workspace.list","args":{}}]}'),
        store, AutonomySettings(enabled=True), LimitSettings(), ResearchSettings(), tmp_path
    )
    try:
        runner.gateway.execute("workspace.read", {"path": "../../outside.txt"})
    except ToolDenied:
        pass
    else:
        raise AssertionError("Workspace traversal must be blocked")


def test_autonomy_gateway_obeys_research_feature_flag(tmp_path):
    from aether.autonomy import ToolDenied

    store = Store(tmp_path / "aether.sqlite3")
    runner = AutonomyRunner(
        FakeProvider('{"steps":[{"tool":"research.search","args":{"query":"test"}}]}'),
        store, AutonomySettings(enabled=True), LimitSettings(),
        ResearchSettings(enabled=False), tmp_path
    )
    try:
        runner.gateway.execute("research.search", {"query": "test"})
    except ToolDenied as exc:
        assert "Research is disabled" in str(exc)
    else:
        raise AssertionError("Research flag must be enforced inside the tool gateway")


def test_autonomy_never_overwrites_existing_workspace_files(tmp_path):
    from aether.autonomy import ToolDenied

    store = Store(tmp_path / "aether.sqlite3")
    runner = AutonomyRunner(
        FakeProvider('{"steps":[{"tool":"workspace.list","args":{}}]}'),
        store, AutonomySettings(enabled=True, allow_workspace_writes=True),
        LimitSettings(), ResearchSettings(), tmp_path
    )
    destination = tmp_path / "workspace" / "keep.txt"
    destination.write_text("preserve me", encoding="utf-8")
    try:
        runner.gateway.execute(
            "workspace.write", {"path": "keep.txt", "content": "overwrite"},
            lambda *args: True,
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("Existing workspace file must not be overwritten")
    assert destination.read_text(encoding="utf-8") == "preserve me"


def test_autonomy_sorts_dependency_graph_and_verifies_each_result(tmp_path):
    response = json.dumps({"steps": [
        {"id": "second", "depends_on": ["first"], "purpose": "List workspace after prerequisite",
         "tool": "workspace.list", "args": {"path": "."}},
        {"id": "first", "purpose": "List workspace first",
         "tool": "workspace.list", "args": {"path": "."}},
    ]})
    store = Store(tmp_path / "aether.sqlite3")
    runner = AutonomyRunner(
        FakeProvider(response), store, AutonomySettings(enabled=True),
        LimitSettings(), ResearchSettings(), tmp_path,
    )

    result = runner.run("Execute dependent workspace checks")

    assert result["status"] == "completed"
    assert result["tool_results_verified"] is True
    assert result["verified"] is False
    assert result["verification_scope"] == "tool_postconditions_only"
    assert result["completed_steps"] == ["first", "second"]
    assert [item["plan_step_id"] for item in result["receipts"]] == ["first", "second"]
    assert all(item["verification"]["passed"] for item in result["receipts"])


def test_autonomy_rejects_cyclic_plan_before_running_tools(tmp_path):
    response = json.dumps({"steps": [
        {"id": "first", "depends_on": ["second"], "tool": "workspace.list", "args": {}},
        {"id": "second", "depends_on": ["first"], "tool": "workspace.list", "args": {}},
    ]})
    store = Store(tmp_path / "aether.sqlite3")
    runner = AutonomyRunner(
        FakeProvider(response), store, AutonomySettings(enabled=True),
        LimitSettings(), ResearchSettings(), tmp_path,
    )

    try:
        runner.run("Reject a cyclic plan")
    except ValueError as exc:
        assert "cycle" in str(exc).casefold()
    else:
        raise AssertionError("Cyclic plans must be rejected")

    failed = store.list_tasks(status="failed", limit=5)
    assert len(failed) == 1
    assert failed[0]["state"]["receipts"] == []


def test_autonomy_does_not_complete_when_tool_postcondition_fails(tmp_path):
    response = json.dumps({"steps": [
        {"id": "read", "purpose": "Read a workspace file",
         "tool": "workspace.read", "args": {"path": "missing.txt"}},
    ]})
    store = Store(tmp_path / "aether.sqlite3")
    runner = AutonomyRunner(
        FakeProvider(response), store, AutonomySettings(enabled=True),
        LimitSettings(), ResearchSettings(), tmp_path,
    )
    runner.gateway.execute = lambda *args, **kwargs: {
        "path": "different.txt", "content": "unverified"
    }

    try:
        runner.run("Verify tool result")
    except RuntimeError as exc:
        assert "verification failed" in str(exc).casefold()
    else:
        raise AssertionError("An unverifiable tool result must not complete the task")

    failed = store.list_tasks(status="failed", limit=5)
    assert len(failed) == 1
    receipt = failed[0]["state"]["receipts"][0]
    assert receipt["verification"]["passed"] is False



def test_autonomy_pauses_at_step_boundary_and_resumes_saved_plan(tmp_path):
    response = json.dumps({"steps": [
        {"id": "one", "purpose": "First listing", "tool": "workspace.list", "args": {"path": "."}},
        {"id": "two", "depends_on": ["one"], "purpose": "Second listing",
         "tool": "workspace.list", "args": {"path": "."}},
    ]})
    store = Store(tmp_path / "pause.sqlite3")
    runner = AutonomyRunner(
        FakeProvider(response), store, AutonomySettings(enabled=True),
        LimitSettings(), ResearchSettings(), tmp_path,
    )
    task = runner.enqueue("Pause and resume a bounded plan")
    calls = []

    def execute(tool, args, approval=None):
        calls.append(tool)
        if len(calls) == 1:
            runner.tasks.request_pause(task.id)
        return {"path": args.get("path", "."), "entries": []}

    runner.gateway.execute = execute
    paused = runner.run_task(task.id)
    assert paused["status"] == "paused"
    assert store.get_task(task.id)["status"] == "paused"
    resumed = runner.resume_task(task.id)
    assert resumed["status"] == "completed"
    assert resumed["completed_steps"] == ["one", "two"]
    assert len(calls) == 2


def test_interrupted_task_resumes_only_when_next_step_is_safe(tmp_path):
    response = json.dumps({"steps": [
        {"id": "unused", "tool": "workspace.list", "args": {"path": "."}},
    ]})
    store = Store(tmp_path / "recovery.sqlite3")
    runner = AutonomyRunner(
        FakeProvider(response), store, AutonomySettings(enabled=True),
        LimitSettings(), ResearchSettings(), tmp_path,
    )
    task = runner.tasks.create("Recover a safe read-only task", {
        "kind": "autonomy",
        "plan": [{"id": "inspect", "depends_on": [], "purpose": "Inspect",
                  "tool": "workspace.list", "args": {"path": "."}}],
        "receipts": [], "completed_steps": [],
    })
    runner.tasks.transition(task.id, "running")
    assert store.recover_interrupted_tasks() == [task.id]
    runner.gateway.execute = lambda tool, args, approval=None: {
        "path": args.get("path", "."), "entries": []
    }
    result = runner.resume_task(task.id)
    assert result["status"] == "completed"
    assert result["resumed_from_interruption"] is True


def test_interrupted_task_refuses_to_replay_uncertain_workspace_write(tmp_path):
    store = Store(tmp_path / "unsafe-recovery.sqlite3")
    runner = AutonomyRunner(
        FakeProvider("{}"), store, AutonomySettings(enabled=True),
        LimitSettings(), ResearchSettings(), tmp_path,
    )
    task = runner.tasks.create("Do not replay an uncertain write", {
        "kind": "autonomy",
        "plan": [{"id": "write", "depends_on": [], "purpose": "Write",
                  "tool": "workspace.write",
                  "args": {"path": "result.txt", "content": "x"}}],
        "receipts": [], "completed_steps": [],
    })
    runner.tasks.transition(task.id, "running")
    store.recover_interrupted_tasks()
    try:
        runner.resume_task(task.id)
    except RuntimeError as exc:
        assert "side effects" in str(exc)
    else:
        raise AssertionError("Uncertain writes must never be replayed")



class SequenceProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def chat(self, messages):
        response = self.responses[self.calls]
        self.calls += 1
        return response


def test_autonomy_replans_once_after_safe_missing_workspace_read(tmp_path):
    first = json.dumps({"steps": [
        {"id": "missing", "purpose": "Read a missing note", "tool": "workspace.read",
         "args": {"path": "not-created.txt"}},
    ]})
    replacement = json.dumps({"steps": [
        {"id": "inspect", "purpose": "List available workspace files", "tool": "workspace.list",
         "args": {"path": "."}},
    ]})
    provider = SequenceProvider([first, replacement])
    store = Store(tmp_path / "replan.sqlite3")
    runner = AutonomyRunner(
        provider, store, AutonomySettings(enabled=True),
        LimitSettings(), ResearchSettings(), tmp_path,
    )

    result = runner.run("Find the available workspace notes")

    assert result["status"] == "completed"
    assert result["replan_history"][0]["attempt"] == 1
    assert result["receipts"][0]["tool"] == "workspace.list"
    assert provider.calls == 2


def test_autonomy_replan_rejects_repeating_a_completed_action(tmp_path):
    first = json.dumps({"steps": [
        {"id": "list", "purpose": "List workspace", "tool": "workspace.list",
         "args": {"path": "."}},
        {"id": "missing", "purpose": "Read missing note", "tool": "workspace.read",
         "args": {"path": "missing.txt"}},
    ]})
    replacement = json.dumps({"steps": [
        {"id": "repeat-list", "purpose": "List workspace again", "tool": "workspace.list",
         "args": {"path": "."}},
    ]})
    provider = SequenceProvider([first, replacement])
    store = Store(tmp_path / "replan-repeat.sqlite3")
    runner = AutonomyRunner(
        provider, store, AutonomySettings(enabled=True),
        LimitSettings(), ResearchSettings(), tmp_path,
    )

    try:
        runner.run("Inspect workspace and find missing note")
    except RuntimeError as exc:
        assert "repeat a completed action" in str(exc)
    else:
        raise AssertionError("Replanning must reject duplicate completed actions")
