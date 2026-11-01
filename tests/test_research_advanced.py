import pytest

from aether.research import ResearchSource, _SearchResultParser
from aether.research_advanced import AdvancedResearch
from aether.storage import Store
import json


def sample_source(url="https://example.org/article", summary="The study found that the treatment reduced measured response time in the tested group."):
    return ResearchSource(
        title="A controlled study", url=url, publisher="Example Journal",
        published="2025-02-01", evidence_type="peer-reviewed-journal",
        summary=summary,
    )


def test_general_web_search_parser_extracts_result_and_ignores_invalid_links():
    parser = _SearchResultParser()
    parser.feed("""
      <div class="result">
        <h2><a class="result__a" href="https://example.org/paper">A paper title</a></h2>
        <a class="result__snippet">A useful abstract snippet.</a>
      </div>
      <div class="result"><a class="result__a" href="javascript:alert(1)">Bad</a></div>
    """)
    assert parser.results[0]["title"] == "A paper title"
    assert parser.results[0]["url"] == "https://example.org/paper"


def test_systematic_review_requires_explicit_protocol_and_decisions(tmp_path):
    store = Store(tmp_path / "research.sqlite3")
    workflow = AdvancedResearch(store)
    review = workflow.create_systematic_review(
        "Does intervention X improve outcome Y?",
        ["Human randomized studies", "Reports outcome Y"],
        ["Animal studies", "Editorials"],
        {"databases": ["Crossref", "DuckDuckGo"], "queries": ["intervention X outcome Y"], "searched_on": "2026-11-01"},
    )
    workflow.add_review_sources(review["id"], [sample_source()])
    with pytest.raises(ValueError, match="Every retrieved source"):
        workflow.finalize_review(review["id"])
    workflow.screen_source(review["id"], "https://example.org/article", "include", "Population and outcome match criteria.")
    finalized = workflow.finalize_review(review["id"])
    assert finalized["status"] == "screening_complete_synthesis_pending"
    assert finalized["final_summary"]["included"] == 1
    with pytest.raises(ValueError, match="already finalized"):
        workflow.screen_source(review["id"], "https://example.org/article", "exclude", "Changed mind after finalization.")


def test_citation_check_requires_source_membership_and_exact_passage(tmp_path):
    store = Store(tmp_path / "research.sqlite3")
    source = sample_source()
    store.save_research_report("report-1", "query", "report", [{
        **source.to_dict(),
        "full_text": "Introduction. The study found that the treatment reduced measured response time in the tested group. Limitations follow.",
    }])
    workflow = AdvancedResearch(store)
    result = workflow.check_citation_support(
        "The treatment reduced response time.",
        "report-1", source.url,
        "The study found that the treatment reduced measured response time in the tested group.",
    )
    assert result["assessment"] == "candidate_support_requires_human_adjudication"
    assert result["normalized_character_offset"] >= 0
    with pytest.raises(ValueError, match="must belong"):
        workflow.check_citation_support(
            "A sufficiently long claim statement.", "report-1",
            "https://other.example/paper",
            "This is a sufficiently long quoted passage to be checked.",
        )
    adjudicated = workflow.adjudicate_citation(
        result["id"], "supports", "The passage directly states the outcome and population described in the claim."
    )
    assert adjudicated["human_decision"] == "supports"
    with pytest.raises(ValueError, match="already been adjudicated"):
        workflow.adjudicate_citation(result["id"], "insufficient", "Second review attempt should not overwrite the first.")


def test_csv_analysis_is_deterministic_and_saved_with_hash(tmp_path):
    store = Store(tmp_path / "research.sqlite3")
    store.create_research_experiment(
        "exp-analysis", "Latency test", None,
        {"objective": "Compare latency", "method": "Measure samples", "success_criteria": "Lower mean"},
    )
    store.transition_research_experiment("exp-analysis", "running")
    workflow = AdvancedResearch(store)
    dataset = b"latency_ms,accuracy\n10,0.9\n20,0.95\n30,0.92\n"
    first = workflow.analyze_csv("exp-analysis", dataset, ["latency_ms", "accuracy"])
    second = workflow.analyze_csv("exp-analysis", dataset, ["latency_ms", "accuracy"])
    assert first["reproducibility_fingerprint"] == second["reproducibility_fingerprint"]
    assert first["results"]["latency_ms"]["mean"] == 20
    assert first["input_sha256"] == second["input_sha256"]
    assert store.get_research_report(first["report_id"]) is not None
    events = store.list_research_experiment_events("exp-analysis")
    assert sum(event["event_type"] == "analysis_recorded" for event in events) == 2


def test_csv_analysis_rejects_non_numeric_and_non_finite_data(tmp_path):
    store = Store(tmp_path / "research.sqlite3")
    store.create_research_experiment(
        "exp-invalid", "Invalid data", None,
        {"objective": "Check", "method": "Read", "success_criteria": "No invalid values"},
    )
    store.transition_research_experiment("exp-invalid", "running")
    workflow = AdvancedResearch(store)
    with pytest.raises(ValueError, match="Non-numeric"):
        workflow.analyze_csv("exp-invalid", b"value\nabc\n", ["value"])
    with pytest.raises(ValueError, match="Non-finite"):
        workflow.analyze_csv("exp-invalid", b"value\nNaN\n", ["value"])
