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
