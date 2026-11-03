import json
from urllib.request import urlopen
from urllib.error import HTTPError

import pytest

from aether.dashboard import AetherDashboard
from aether.storage import Store


class FakeMonitor:
    def snapshot(self, cpu_sample_seconds=0.1):
        return {
            "collected_at": "2026-11-03T00:00:00+00:00",
            "cpu_percent": 12.0,
            "ram": {"used_percent": 40.0},
            "gpu": None,
        }


def test_dashboard_serves_read_only_status_on_loopback(tmp_path):
    store = Store(tmp_path / "dashboard.sqlite3")
    store.create_task("task-1", "Inspect project", {"kind": "autonomy"})
    dashboard = AetherDashboard(
        store, model_provider="ollama", model_id="phi3:mini",
        host="127.0.0.1", port=0, monitor=FakeMonitor(),
    )
    host, port = dashboard.start()
    try:
        with urlopen(f"http://{host}:{port}/api/status", timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
            assert response.headers["Cache-Control"] == "no-store"
        assert payload["model"]["id"] == "phi3:mini"
        assert payload["tasks"][0]["id"] == "task-1"
        assert payload["resources"]["cpu_percent"] == 12.0
        with urlopen(f"http://{host}:{port}/", timeout=3) as response:
            assert b"Operational Dashboard" in response.read()
        with pytest.raises(HTTPError) as error:
            from urllib.request import Request
            urlopen(Request(f"http://{host}:{port}/api/status", data=b"{}", method="POST"), timeout=3)
        assert error.value.code == 405
    finally:
        dashboard.stop()


def test_dashboard_rejects_non_loopback_binding(tmp_path):
    with pytest.raises(ValueError, match="loopback"):
        AetherDashboard(Store(tmp_path / "dashboard.sqlite3"), host="0.0.0.0")
