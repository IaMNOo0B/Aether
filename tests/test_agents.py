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


def test_independent_specialists_run_in_parallel_and_receive_separate_verifier(tmp_path):
    store = Store(tmp_path / "parallel-agents.sqlite3")

    class NamedProvider:
        def __init__(self, output):
            self.output = output
            self.calls = 0

        def chat(self, messages):
            self.calls += 1
            return self.output

    researcher = NamedProvider("Evidence with source references")
    engineer = NamedProvider("Implementation plan with test criteria")
    verifier = NamedProvider("Verified: none; unresolved: run the tests")
    coordinator = MultiAgentCoordinator(researcher, store)

    result = coordinator.run_independent(
        "Build a tested feature",
        {"researcher": researcher, "engineer": engineer},
        verifier,
        max_workers=2,
    )

    assert [item.role for item in result] == [
        "engineer", "researcher", "independent_verifier"
    ]
    assert researcher.calls == engineer.calls == verifier.calls == 1
    assert len(store.list_agent_runs()) == 3
    assert store.list_agent_runs(limit=1)[0]["role"] == "independent_verifier"


def test_independent_agent_mode_rejects_shared_provider_instances(tmp_path):
    store = Store(tmp_path / "shared-provider.sqlite3")
    provider = FakeProvider()
    coordinator = MultiAgentCoordinator(provider, store)
    import pytest
    with pytest.raises(ValueError, match="distinct provider instances"):
        coordinator.run_independent(
            "Objective", {"researcher": provider, "planner": FakeProvider()},
            provider, max_workers=2,
        )
