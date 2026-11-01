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
