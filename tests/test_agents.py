from aether.agents import MultiAgentCoordinator
from aether.storage import Store


class FakeProvider:
    def __init__(self):
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        return f"Deliverable {self.calls}"


def test_specialist_roles_run_and_persist_separate_deliverables(tmp_path):
    store = Store(tmp_path / "agents.sqlite3")
    provider = FakeProvider()
    coordinator = MultiAgentCoordinator(provider, store)
    outputs = coordinator.run("Research a software design")
    assert [item.role for item in outputs] == ["researcher", "planner", "critic"]
    assert provider.calls == 3
    assert len(store.list_agent_runs()) == 3
    assert outputs[-1].output == "Deliverable 3"


import pytest


def test_parallel_agents_require_independent_provider_instances_and_verifier(tmp_path):
    store = Store(tmp_path / "agents.sqlite3")
    provider = FakeProvider()
    coordinator = MultiAgentCoordinator(provider, store)
    shared = {role: provider for role, _ in coordinator.ROLES}
    with pytest.raises(ValueError, match="distinct provider instances"):
        coordinator.run_parallel("Evaluate architecture", shared, verifier=FakeProvider())
    independent = {role: FakeProvider() for role, _ in coordinator.ROLES}
    with pytest.raises(ValueError, match="separate independent verifier"):
        coordinator.run_parallel("Evaluate architecture", independent)


def test_parallel_agents_run_distinct_workers_and_persist_independent_review(tmp_path):
    store = Store(tmp_path / "agents.sqlite3")
    coordinator = MultiAgentCoordinator(FakeProvider(), store)
    providers = {role: FakeProvider() for role, _ in coordinator.ROLES}
    verifier = FakeProvider()
    outputs = coordinator.run_parallel(
        "Inspect an architecture", providers, verifier=verifier, max_workers=3
    )
    assert [item.role for item in outputs] == [
        "researcher", "planner", "critic", "independent_verifier"
    ]
    history = store.list_agent_runs(limit=10)
    assert len(history) == 4
    assert {item["role"] for item in history} == {
        "researcher", "planner", "critic", "independent_verifier"
    }
