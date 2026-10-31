import pytest

from aether.planning import PlanGraph, PlanManager, PlanStep
from aether.storage import Store


def sample_steps():
    return [
        PlanStep("inspect", "Inspect the issue"),
        PlanStep("design", "Design a fix", ("inspect",)),
        PlanStep("test", "Test the fix", ("design",)),
        PlanStep("document", "Document the fix", ("inspect",)),
    ]


def test_plan_graph_orders_dependencies_and_exposes_ready_frontier():
    graph = PlanGraph(sample_steps())
    assert graph.topological_order() == ("inspect", "design", "test", "document")
    assert graph.ready_steps() == ("inspect",)
    assert graph.ready_steps({"inspect"}) == ("design", "document")


def test_plan_graph_rejects_cycles_and_missing_dependencies():
    with pytest.raises(ValueError, match="cycle"):
        PlanGraph([PlanStep("a", "A", ("b",)), PlanStep("b", "B", ("a",))])
    with pytest.raises(ValueError, match="unknown dependencies"):
        PlanGraph([PlanStep("a", "A", ("missing",))])


def test_plan_progress_survives_store_recreation(tmp_path):
    database = tmp_path / "plans.sqlite3"
    manager = PlanManager(Store(database))
    task = manager.create("Fix a reproducible defect", sample_steps())
    assert manager.ready_steps(task.id) == ("inspect",)
    assert manager.complete_step(task.id, "inspect") == ("design", "document")
    restarted = PlanManager(Store(database))
    assert restarted.ready_steps(task.id) == ("design", "document")
    with pytest.raises(ValueError, match="not ready"):
        restarted.complete_step(task.id, "test")
