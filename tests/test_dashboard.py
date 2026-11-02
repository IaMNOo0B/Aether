from aether.dashboard import build_dashboard, render_dashboard


class FakeStore:
    def list_decisions(self, resolved_only=False, limit=10000):
        return [
            {"outcome": 1, "metadata": {"producer_type": "world_model"}},
            {"outcome": None, "metadata": {"producer_type": "assistant_model"}},
        ]

    def list_tasks(self, limit=100):
        return [{"status": "completed"}, {"status": "failed"}]

    def list_research_reports(self, limit=100):
        return [{"id": "r1", "query": "test"}]


class FakeSelfModel:
    def snapshot(self):
        return {"capabilities": [
            {"capability_id": "a", "state": "implemented_unverified"},
            {"capability_id": "b", "state": "verified"},
        ]}


class FakeResources:
    def snapshot(self):
        return {"cpu_percent": 12.0, "ram_percent": 40.0}


class FakeEngine:
    self_model = FakeSelfModel()
    resources = FakeResources()

    def status(self):
        return {
            "version": "0.3.0", "provider": "ollama", "model": "test",
            "model_enabled": True, "model_reachable": False,
            "autonomy_enabled": False, "emergency_stop": False,
        }


def test_dashboard_aggregates_read_only_runtime_and_evidence_counts(tmp_path):
    snapshot = build_dashboard(FakeEngine(), FakeStore(), tmp_path / "missing.json")

    assert snapshot["capabilities"]["total"] == 2
    assert snapshot["capabilities"]["states"] == {
        "implemented_unverified": 1, "verified": 1
    }
    assert snapshot["decisions"]["resolved"] == 1
    assert snapshot["decisions"]["unresolved"] == 1
    assert snapshot["research"]["stored_reports_visible"] == 1
    assert snapshot["tasks"]["by_status"] == {"completed": 1, "failed": 1}
    assert snapshot["acceptance"] is None


def test_dashboard_renderer_does_not_claim_unverified_capabilities_are_verified(tmp_path):
    snapshot = build_dashboard(FakeEngine(), FakeStore(), tmp_path / "missing.json")
    rendered = render_dashboard(snapshot)

    assert "implemented_unverified" in rendered
    assert "not proof of verification" in rendered
