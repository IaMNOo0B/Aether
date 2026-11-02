import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from aether.discovery import DiscoveryError, ResearchDiscovery, load_topics


class FakeResearcher:
    def __init__(self):
        self.searches = []

    def search(self, query, limit=8):
        self.searches.append((query, limit))
        return [{"title": "source"}]

    def create_report(self, query, sources):
        return {"id": f"report-{len(self.searches)}"}


class FakeStore:
    pass


def test_discovery_persists_report_as_pending_review(tmp_path):
    config = tmp_path / "topics.json"
    config.write_text(json.dumps({"topics": [{"query": "calibration methods", "interval_hours": 24}]}))
    researcher = FakeResearcher()
    clock = lambda: datetime(2026, 11, 2, tzinfo=timezone.utc)
    service = ResearchDiscovery(FakeStore(), config, tmp_path / "state.json",
                                researcher=researcher, clock=clock)
    result = service.run_once()
    assert result["reports"][0]["review_status"] == "pending"
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["pending_review"][0]["status"] == "pending"


def test_discovery_respects_refresh_interval(tmp_path):
    config = tmp_path / "topics.json"
    config.write_text(json.dumps({"topics": [{"query": "calibration methods", "interval_hours": 24}]}))
    state = tmp_path / "state.json"
    state.write_text(json.dumps({
        "last_run": {"calibration methods": "2026-11-01T12:00:00+00:00"},
        "pending_review": [],
    }))
    researcher = FakeResearcher()
    clock = lambda: datetime(2026, 11, 2, 0, 0, tzinfo=timezone.utc)
    service = ResearchDiscovery(FakeStore(), config, state, researcher=researcher, clock=clock)
    result = service.run_once()
    assert result["skipped_not_due"] == ["calibration methods"]
    assert researcher.searches == []


def test_discovery_rejects_duplicate_queries(tmp_path):
    config = tmp_path / "topics.json"
    config.write_text(json.dumps({"topics": [
        {"query": "calibration methods", "interval_hours": 24},
        {"query": "Calibration Methods", "interval_hours": 48},
    ]}))
    with pytest.raises(DiscoveryError, match="Duplicate"):
        load_topics(config)


def test_discovery_rejects_too_frequent_refresh(tmp_path):
    config = tmp_path / "topics.json"
    config.write_text(json.dumps({"topics": [{"query": "calibration methods", "interval_hours": 0.5}]}))
    with pytest.raises(DiscoveryError, match="interval_hours"):
        load_topics(config)
