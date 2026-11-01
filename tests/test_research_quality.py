from unittest.mock import patch
from urllib.error import HTTPError
from io import BytesIO

import pytest

from aether.research import ResearchSource, ResearchError
from aether.research_quality import ResearchWorkflow, validate_public_url
from aether.storage import Store


def source(**changes):
    values = dict(
        title="A paper", url="https://example.org/paper",
        publisher="Example", published="2025-01-01",
        evidence_type="preprint", summary="A" * 800,
    )
    values.update(changes)
    return ResearchSource(**values)


def test_public_url_rejects_local_and_non_http_targets():
    with pytest.raises(ValueError):
        validate_public_url("file:///etc/passwd")
    with pytest.raises(ValueError):
        validate_public_url("http://127.0.0.1/admin")
    with pytest.raises(ValueError):
        validate_public_url("http://localhost/")


def test_public_url_rejects_private_dns_resolution(monkeypatch):
    monkeypatch.setattr(
        "aether.research_quality.socket.getaddrinfo",
        lambda *args: [(None, None, None, None, ("10.0.0.8", 0))],
    )
    with pytest.raises(ValueError, match="public IP"):
        validate_public_url("https://internal.example")


def test_quality_score_explains_caveats_and_is_not_truth_probability():
    result = ResearchWorkflow.assess_source(source())
    assert result.score > 0
    assert result.band in {"low-triage", "medium-triage", "high-triage"}
    assert any("not a probability" in item for item in result.caveats)
    assert any("Preprints" in item for item in result.caveats)


def test_review_protocol_persists_all_sources_as_pending(tmp_path):
    store = Store(tmp_path / "research.sqlite3")
    workflow = ResearchWorkflow(store)
    review = workflow.create_review_protocol("agent reliability", [source()])
    saved = store.get_research_report(review["id"])
    assert saved is not None
    assert saved["sources"][0]["screening_status"] == "awaiting_human_screen"
    assert "no source is silently excluded" in saved["content"]


def test_citation_verification_only_reports_reachability(tmp_path, monkeypatch):
    store = Store(tmp_path / "research.sqlite3")
    store.save_research_report("r1", "q", "report", [source().to_dict()])
    workflow = ResearchWorkflow(store)
    monkeypatch.setattr("aether.research_quality.validate_public_url", lambda url: url)

    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): return False

    monkeypatch.setattr(workflow._opener, "open", lambda *args, **kwargs: Response())
    result = workflow.verify_citations("r1")
    assert result["results"][0]["status"] == "reachable_only"
    assert result["results"][0]["citation_support"] == "not_checked"


def test_experiment_requires_preregistered_protocol_and_valid_hypothesis(tmp_path):
    store = Store(tmp_path / "research.sqlite3")
    with pytest.raises(ValueError, match="objective, method"):
        store.create_research_experiment("e1", "Test", None, {"objective": "x"})
    with pytest.raises(ValueError, match="existing hypothesis"):
        store.create_research_experiment(
            "e2", "Test", "missing",
            {"objective": "x", "method": "y", "success_criteria": "z"},
        )


def test_experiment_lifecycle_is_audited_and_terminal(tmp_path):
    from uuid import uuid4

    store = Store(tmp_path / "research.sqlite3")
    graph = __import__("aether.knowledge", fromlist=["KnowledgeGraph"]).KnowledgeGraph(store)
    hypothesis = graph.add_claim(
        "Model A", "outperforms", "Model B", claim_type="hypothesis"
    )
    experiment = store.create_research_experiment(
        "exp-1", "Benchmark A vs B", hypothesis["id"],
        {
            "objective": "Compare latency on a fixed workload",
            "method": "Run 30 repetitions per model with identical settings",
            "success_criteria": "A has lower median latency without lower task accuracy",
            "metrics": ["median_latency_ms", "task_accuracy"],
        },
    )
    assert experiment["status"] == "planned"
    started = store.transition_research_experiment("exp-1", "running")
    assert started["status"] == "running"
    with pytest.raises(ValueError, match="requires a non-empty result"):
        store.transition_research_experiment("exp-1", "completed", {"evidence": "log"})
    completed = store.transition_research_experiment(
        "exp-1", "completed",
        {"result": "A was 8% faster; accuracy unchanged", "evidence": "benchmark-run-2026-11-01"},
    )
    assert completed["status"] == "completed"
    assert completed["outcome"]["evidence"] == "benchmark-run-2026-11-01"
    with pytest.raises(ValueError, match="Invalid experiment transition"):
        store.transition_research_experiment("exp-1", "running")
    events = store.list_research_experiment_events("exp-1")
    assert [event["event_type"] for event in events] == ["created", "running", "completed"]
