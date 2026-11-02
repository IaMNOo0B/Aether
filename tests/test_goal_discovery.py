import pytest

from aether.goal_discovery import GoalDiscovery
from aether.storage import Store


def test_external_events_become_reviewable_proposals_not_executable_tasks(tmp_path):
    database = tmp_path / "aether.sqlite3"
    discovery = GoalDiscovery(database, tmp_path)
    event_id = discovery.record_external_event(
        "provider_degraded", "The configured model endpoint became unavailable",
        "health-check-17",
    )

    proposals = discovery.discover()

    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal["source_type"] == "external_event"
    assert proposal["source_ref"] == event_id
    assert proposal["status"] == "proposed"
    assert Store(database).list_tasks() == []


def test_discovery_deduplicates_events_and_requires_reasoned_approval(tmp_path):
    discovery = GoalDiscovery(tmp_path / "aether.sqlite3", tmp_path)
    discovery.record_external_event("research", "A new research opportunity was found")
    first = discovery.discover()
    assert len(first) == 1
    assert discovery.discover() == []

    with pytest.raises(ValueError, match="reason is required"):
        discovery.approve(first[0]["id"], "")

    approved = discovery.approve(first[0]["id"], "Reviewed source and accepted follow-up")
    assert approved["status"] == "approved"
    assert [event["event_type"] for event in discovery.events(first[0]["id"])] == [
        "proposed", "approved"
    ]
    with pytest.raises(ValueError, match="Only proposed"):
        discovery.reject(first[0]["id"], "Too late")


def test_failed_tasks_create_diagnosis_proposals_never_retry_proposals(tmp_path):
    database = tmp_path / "aether.sqlite3"
    store = Store(database)
    task = store.create_task("failed-1", "Update the project", {"kind": "autonomy"})
    store.transition_task(task_id="failed-1", expected_status="queued",
                          new_status="running", state={"kind": "autonomy"})
    store.transition_task(task_id="failed-1", expected_status="running",
                          new_status="failed", state={"kind": "autonomy"})

    discovery = GoalDiscovery(database, tmp_path)
    proposals = discovery.discover()

    assert len(proposals) == 1
    assert proposals[0]["source_type"] == "failed_task"
    assert "do not repeat" in proposals[0]["objective"]
    assert store.get_task("failed-1")["status"] == "failed"


def test_research_reports_can_seed_followup_review_proposals(tmp_path):
    database = tmp_path / "aether.sqlite3"
    store = Store(database)
    store.save_research_report("report-1", "GPU isolation", "Review Windows sandboxing",
                               [{"url": "https://example.org"}])
    proposals = GoalDiscovery(database, tmp_path).discover()
    assert any(item["source_type"] == "research_report" for item in proposals)


def test_external_event_input_is_bounded_and_requires_content(tmp_path):
    discovery = GoalDiscovery(tmp_path / "aether.sqlite3", tmp_path)
    with pytest.raises(ValueError):
        discovery.record_external_event(" ", "no event")
    event_id = discovery.record_external_event("x" * 500, "y" * 5000)
    assert len(event_id) == 32
    item = discovery.discover()[0]
    assert len(item["title"]) <= 300
    assert len(item["objective"]) <= 1000
