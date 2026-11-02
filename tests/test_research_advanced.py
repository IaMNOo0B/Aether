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
    workflow.screen_source(review["id"], "https://example.org/article", "reviewer-a", "include", "Population and outcome match criteria.")
    with pytest.raises(ValueError, match="Every retrieved source"):
        workflow.finalize_review(review["id"])
    workflow.screen_source(review["id"], "https://example.org/article", "reviewer-b", "include", "The same population and outcome criteria are met.")
    finalized = workflow.finalize_review(review["id"])
    assert finalized["status"] == "screening_complete_synthesis_pending"
    assert finalized["final_summary"]["included"] == 1
    with pytest.raises(ValueError, match="already finalized"):
        workflow.screen_source(review["id"], "https://example.org/article", "reviewer-c", "exclude", "Changed mind after finalization.")


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
    assert result["source_text_kind"] == "captured_full_text"
    assert result["normalized_character_offset"] >= 0
    with pytest.raises(ValueError, match="must belong"):
        workflow.check_citation_support(
            "A sufficiently long claim statement.", "report-1",
            "https://other.example/paper",
            "This is a sufficiently long quoted passage to be checked.",
        )
    first_review = workflow.adjudicate_citation(
        result["id"], "reviewer-a", "supports",
        "The passage directly states the outcome and population described in the claim."
    )
    assert first_review["review_status"] == "awaiting_second_reviewer"
    assert first_review["human_decision"] is None
    adjudicated = workflow.adjudicate_citation(
        result["id"], "reviewer-b", "supports",
        "Independent reading confirms the passage states the described outcome."
    )
    assert adjudicated["human_decision"] == "supports"
    assert adjudicated["review_status"] == "two_reviewer_agreement"
    with pytest.raises(ValueError, match="already adjudicated"):
        workflow.adjudicate_citation(result["id"], "reviewer-a", "insufficient", "This duplicate reviewer cannot overwrite the decision.")


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


def test_systematic_review_disagreement_requires_distinct_adjudicator(tmp_path):
    store = Store(tmp_path / "research.sqlite3")
    workflow = AdvancedResearch(store)
    review = workflow.create_systematic_review(
        "Does intervention X improve outcome Y?",
        ["Human studies"], ["Animal studies"],
        {"databases": ["Crossref"], "queries": ["intervention X"], "searched_on": "2026-11-01"},
    )
    workflow.add_review_sources(review["id"], [sample_source()])
    workflow.screen_source(review["id"], "https://example.org/article", "reviewer-a", "include", "The title and abstract meet the inclusion criteria.")
    workflow.screen_source(review["id"], "https://example.org/article", "reviewer-b", "exclude", "The population does not match the inclusion criteria.")
    review_state = store.get_systematic_review(review["id"])
    assert review_state["sources"][0]["decision"] == "uncertain"
    with pytest.raises(ValueError, match="distinct from screening reviewers"):
        workflow.resolve_review_conflict(review["id"], "https://example.org/article", "reviewer-a", "include", "This is not a separate adjudicator.")
    workflow.resolve_review_conflict(review["id"], "https://example.org/article", "adjudicator-c", "include", "Full text confirms the prespecified target population.")
    finalized = workflow.finalize_review(review["id"])
    assert finalized["final_summary"]["included"] == 1


def test_citation_disagreement_is_not_silently_resolved(tmp_path):
    store = Store(tmp_path / "research.sqlite3")
    source = sample_source()
    store.save_research_report("report-2", "query", "report", [{**source.to_dict(), "full_text": source.summary}])
    workflow = AdvancedResearch(store)
    evidence = workflow.check_citation_support(
        "The treatment reduced response time in the studied group.",
        "report-2", source.url, source.summary,
    )
    workflow.adjudicate_citation(evidence["id"], "reviewer-a", "supports", "The text states a reduction in response time.")
    result = workflow.adjudicate_citation(evidence["id"], "reviewer-b", "contradicts", "The context qualifies the result and does not support the broader claim.")
    assert result["human_decision"] is None
    assert result["review_status"] == "reviewer_disagreement_requires_resolution"
    resolved = workflow.resolve_citation_conflict(
        evidence["id"], "adjudicator-c", "insufficient",
        "The evidence does not establish the full scope of the original claim."
    )
    assert resolved["human_decision"] == "insufficient"
    assert resolved["review_status"] == "resolved_by_third_reviewer"


def test_grouped_experiment_analysis_is_reproducible(tmp_path):
    store = Store(tmp_path / "research.sqlite3")
    store.create_research_experiment(
        "exp-groups", "Control vs treatment", None,
        {"objective": "Compare groups", "method": "Collect scores", "success_criteria": "Treatment mean higher"},
    )
    store.transition_research_experiment("exp-groups", "running")
    workflow = AdvancedResearch(store)
    data = b"group,score\ncontrol,10\ncontrol,11\ncontrol,12\ntreatment,20\ntreatment,21\ntreatment,22\n"
    first = workflow.analyze_csv("exp-groups", data, ["score"], "group", ["treatment", "control"])
    second = workflow.analyze_csv("exp-groups", data, ["score"], "group", ["treatment", "control"])
    result = first["group_comparison"]["score"]
    assert result["left_group"] == "treatment"
    assert result["mean_difference_left_minus_right"] == 10
    assert 0 <= result["two_sided_p_value"] <= 1
    assert first["reproducibility_fingerprint"] == second["reproducibility_fingerprint"]


def test_grouped_analysis_rejects_unrequested_third_group(tmp_path):
    store = Store(tmp_path / "research.sqlite3")
    store.create_research_experiment(
        "exp-three", "Three groups", None,
        {"objective": "Compare", "method": "Sample", "success_criteria": "No third group"},
    )
    store.transition_research_experiment("exp-three", "running")
    with pytest.raises(ValueError, match="outside the requested two-group"):
        AdvancedResearch(store).analyze_csv(
            "exp-three",
            b"group,score\na,1\na,2\nb,3\nb,4\nc,5\nc,6\n",
            ["score"], "group", ["a", "b"],
        )


def _risk_payload():
    domains = {}
    for name in ["randomization", "deviations", "missing_data", "outcome_measurement", "selective_reporting"]:
        domains[name] = {"rating": "low", "rationale": "The study report provides enough detail to assess this domain."}
    return {"domains": domains}


def _extraction(effect):
    return {
        "study_design": "randomized controlled trial",
        "population": "Adults with condition X",
        "sample_size": 120,
        "outcome": "response time",
        "findings": "Intervention group showed reduced response time.",
        "effect_measure": "mean_difference_ms",
        "effect_estimate": effect,
        "standard_error": 2.0,
    }


def test_full_systematic_review_requires_two_reviewers_and_runs_meta_analysis(tmp_path):
    store = Store(tmp_path / "research.sqlite3")
    workflow = AdvancedResearch(store)
    review = workflow.create_systematic_review(
        "Does intervention X reduce response time?",
        ["Adult human studies"], ["Animal studies"],
        {"databases": ["Crossref", "DuckDuckGo"], "queries": ["intervention X response time"], "searched_on": "2026-11-01"},
    )
    sources = [
        sample_source("https://example.org/study-a"),
        sample_source("https://example.org/study-b"),
    ]
    workflow.add_review_sources(review["id"], sources)
    for source in sources:
        key = source.url
        workflow.screen_source(review["id"], key, "screen-a", "include", "The abstract meets the prespecified population and outcome criteria.")
        workflow.screen_source(review["id"], key, "screen-b", "include", "The abstract independently meets the population and outcome criteria.")
    workflow.finalize_review(review["id"])
    for index, source in enumerate(sources):
        key = source.url
        full_text = {"decision": "include", "rationale": "Full text confirms eligible population and prespecified outcome.", "text_locator": "Results, paragraph 2"}
        risk = _risk_payload()
        extraction = _extraction(float(index + 1) * 3.0)
        for stage, payload in [
            ("full_text", full_text),
            ("risk_of_bias", risk),
            ("extraction", extraction),
        ]:
            first = workflow.record_review_stage(review["id"], key, stage, "review-a", payload)
            assert first["status"] == "awaiting_second_reviewer"
            second = workflow.record_review_stage(review["id"], key, stage, "review-b", payload)
            assert second["status"] == "agreed"
    synthesized = workflow.synthesize_review(review["id"])
    summary = synthesized["final_summary"]
    assert synthesized["status"] == "synthesized"
    assert summary["meta_analysis"]["studies"] == 2
    assert summary["meta_analysis"]["model"] == "DerSimonian-Laird random effects"
    assert len(summary["included_studies"]) == 2


def test_systematic_review_cannot_synthesize_with_unresolved_full_text_stage(tmp_path):
    store = Store(tmp_path / "research.sqlite3")
    workflow = AdvancedResearch(store)
    review = workflow.create_systematic_review(
        "Does intervention X improve outcome Y?",
        ["Human studies"], ["Animal studies"],
        {"databases": ["Crossref"], "queries": ["intervention X"], "searched_on": "2026-11-01"},
    )
    source = sample_source()
    workflow.add_review_sources(review["id"], [source])
    workflow.screen_source(review["id"], source.url, "screen-a", "include", "The abstract matches the inclusion criteria.")
    workflow.screen_source(review["id"], source.url, "screen-b", "include", "The abstract matches the inclusion criteria independently.")
    workflow.finalize_review(review["id"])
    workflow.record_review_stage(
        review["id"], source.url, "full_text", "review-a",
        {"decision": "include", "rationale": "Full text seems to match the eligibility requirements.", "text_locator": "Methods section"},
    )
    with pytest.raises(ValueError, match="Full-text eligibility"):
        workflow.synthesize_review(review["id"])


def test_general_web_search_returns_structured_sources_without_fetching_results(tmp_path, monkeypatch):
    from aether.research import Researcher

    researcher = Researcher(Store(tmp_path / "web.sqlite3"))
    html = b"""
      <div class="result">
        <h2><a class="result__a" href="https://example.org/research">Research result</a></h2>
        <a class="result__snippet">An untrusted search snippet.</a>
      </div>
    """
    monkeypatch.setattr(researcher, "_get", lambda url: html)
    sources = researcher.search_web("research topic", 3)
    assert len(sources) == 1
    assert sources[0].url == "https://example.org/research"
    assert sources[0].evidence_type == "general-web-search-result"
    assert sources[0].full_text == ""


def test_snippet_only_quote_cannot_be_adjudicated_as_support(tmp_path):
    store = Store(tmp_path / "research.sqlite3")
    source = sample_source()
    store.save_research_report("snippet-report", "query", "report", [source.to_dict()])
    workflow = AdvancedResearch(store)
    evidence = workflow.check_citation_support(
        "The treatment reduced response time in the studied group.",
        "snippet-report", source.url, source.summary,
    )
    assert evidence["source_text_kind"] == "abstract_or_search_snippet"
    with pytest.raises(ValueError, match="captured full text"):
        workflow.adjudicate_citation(
            evidence["id"], "reviewer-a", "supports",
            "The source summary appears to support the supplied claim wording."
        )


def test_systematic_review_deduplicates_sources_but_preserves_report_provenance(tmp_path):
    store = Store(tmp_path / "research.sqlite3")
    workflow = AdvancedResearch(store)
    review = workflow.create_systematic_review(
        "Does intervention X improve outcome Y?",
        ["Human studies"], ["Animal studies"],
        {"databases": ["Crossref"], "queries": ["intervention X"], "searched_on": "2026-11-01"},
    )
    source = sample_source()
    store.save_research_report("search-a", "query a", "report a", [source.to_dict()])
    store.save_research_report("search-b", "query b", "report b", [source.to_dict()])
    workflow.add_review_sources(review["id"], [source], "search-a")
    workflow.add_review_sources(review["id"], [source], "search-b")
    saved = store.get_systematic_review(review["id"])
    assert len(saved["sources"]) == 1
    assert saved["sources"][0]["source"]["search_report_ids"] == ["search-a", "search-b"]


def test_systematic_review_can_synthesize_zero_eligible_studies(tmp_path):
    store = Store(tmp_path / "research.sqlite3")
    workflow = AdvancedResearch(store)
    review = workflow.create_systematic_review(
        "Does intervention X improve outcome Y?",
        ["Human studies"], ["Animal studies"],
        {"databases": ["Crossref"], "queries": ["intervention X"], "searched_on": "2026-11-01"},
    )
    source = sample_source()
    workflow.add_review_sources(review["id"], [source])
    workflow.screen_source(review["id"], source.url, "review-a", "exclude", "The study uses an ineligible animal population.")
    workflow.screen_source(review["id"], source.url, "review-b", "exclude", "The population does not meet the human-study criterion.")
    workflow.finalize_review(review["id"])
    result = workflow.synthesize_review(review["id"])
    assert result["status"] == "synthesized"
    assert result["final_summary"]["screening_flow"]["included_in_synthesis"] == 0
    assert result["final_summary"]["meta_analysis_status"] == "not_performed_fewer_than_two_eligible_studies"


def test_qualitative_systematic_review_synthesizes_without_forcing_meta_analysis(tmp_path):
    store = Store(tmp_path / "research.sqlite3")
    workflow = AdvancedResearch(store)
    review = workflow.create_systematic_review(
        "How do participants describe intervention X?",
        ["Qualitative human studies"], ["Animal studies"],
        {"databases": ["Crossref"], "queries": ["intervention X experience"], "searched_on": "2026-11-01"},
    )
    source = sample_source()
    workflow.add_review_sources(review["id"], [source])
    workflow.screen_source(review["id"], source.url, "screen-a", "include", "Abstract meets the qualitative population criteria.")
    workflow.screen_source(review["id"], source.url, "screen-b", "include", "Independent screening confirms population eligibility.")
    workflow.finalize_review(review["id"])
    full = {"decision": "include", "rationale": "Full text contains eligible qualitative participant interviews.", "text_locator": "Findings section"}
    risk = _risk_payload()
    qualitative = {
        "study_design": "qualitative interviews",
        "population": "Adults with condition X",
        "sample_size": 18,
        "outcome": "participant experience",
        "findings": "Participants described improved access and persistent usability barriers.",
    }
    for reviewer in ["review-a", "review-b"]:
        workflow.record_review_stage(review["id"], source.url, "full_text", reviewer, full)
        workflow.record_review_stage(review["id"], source.url, "risk_of_bias", reviewer, risk)
        workflow.record_review_stage(review["id"], source.url, "extraction", reviewer, qualitative)
    result = workflow.synthesize_review(review["id"])
    assert result["final_summary"]["meta_analysis"] is None
    assert result["final_summary"]["meta_analysis_status"] == "not_performed_fewer_than_two_eligible_studies"


def test_combined_research_search_interleaves_web_and_scholarly_sources(tmp_path, monkeypatch):
    from aether.research import Researcher

    researcher = Researcher(Store(tmp_path / "combined.sqlite3"), max_results=4)
    providers = {
        "search_web": [sample_source("https://web.example/a"), sample_source("https://web.example/b")],
        "search_crossref": [sample_source("https://doi.org/10.1/a"), sample_source("https://doi.org/10.1/b")],
        "search_arxiv": [sample_source("https://arxiv.org/a"), sample_source("https://arxiv.org/b")],
        "search_wikipedia": [sample_source("https://wikipedia.org/a")],
    }
    for name, results in providers.items():
        monkeypatch.setattr(researcher, name, lambda *args, _results=results: _results)
    results = researcher.search("agent evidence", limit=4)
    assert [item.url for item in results] == [
        "https://web.example/a",
        "https://doi.org/10.1/a",
        "https://arxiv.org/a",
        "https://wikipedia.org/a",
    ]

def test_quantitative_extraction_validates_exact_source_quote_and_numeric_values(tmp_path):
    store = Store(tmp_path / "research.sqlite3")
    workflow = AdvancedResearch(store)
    review = workflow.create_systematic_review(
        "Does intervention X reduce response time?",
        ["Adult studies"], ["Animal studies"],
        {"databases": ["Synthetic"], "queries": ["intervention X"], "searched_on": "2026-11-02"},
    )
    source = ResearchSource(
        title="Evidence source", url="https://example.org/effect",
        publisher="Example Journal", published="2026", evidence_type="peer-reviewed-study",
        summary="Effect summary.",
        full_text="Results table: Mean difference was -2.5 ms with standard error 0.8.",
    )
    workflow.add_review_sources(review["id"], [source])
    extraction = {
        "study_design": "randomized controlled trial",
        "population": "Adults with condition X",
        "sample_size": 80,
        "outcome": "response time",
        "findings": "Intervention reduced response time.",
        "effect_measure": "mean_difference_ms",
        "effect_estimate": -2.5,
        "standard_error": 0.8,
        "effect_evidence_quote": "Mean difference was -2.5 ms with standard error 0.8.",
        "effect_text_locator": "Results table 2",
    }
    invalid = dict(extraction, effect_evidence_quote="Mean difference was -9.9 ms with standard error 0.8.")
    with pytest.raises(ValueError, match="match captured full text"):
        workflow.record_review_stage(review["id"], source.url, "extraction", "reviewer-a", invalid)
    result = workflow.record_review_stage(review["id"], source.url, "extraction", "reviewer-a", extraction)
    assert result["status"] == "awaiting_second_reviewer"
    assert result["reviewers"][0]["payload"]["effect_evidence_status"] == "exact_quote_and_numeric_values_found"


def test_full_text_not_retrieved_is_counted_in_prisma_flow(tmp_path):
    from aether.research_completion import ResearchCompletion

    store = Store(tmp_path / "research.sqlite3")
    workflow = AdvancedResearch(store)
    review = workflow.create_systematic_review(
        "Does intervention X improve outcome Y?",
        ["Human studies"], ["Animal studies"],
        {"databases": ["Crossref"], "queries": ["intervention X"], "searched_on": "2026-11-02"},
    )
    source = sample_source()
    workflow.add_review_sources(review["id"], [source])
    workflow.screen_source(review["id"], source.url, "screen-a", "include", "The abstract meets the eligibility criteria.")
    workflow.screen_source(review["id"], source.url, "screen-b", "include", "Independent screening confirms eligibility.")
    workflow.finalize_review(review["id"])
    unavailable = {"decision": "not_retrieved", "rationale": "The full text could not be obtained after reasonable attempts."}
    workflow.record_review_stage(review["id"], source.url, "full_text", "review-a", unavailable)
    workflow.record_review_stage(review["id"], source.url, "full_text", "review-b", unavailable)
    result = workflow.synthesize_review(review["id"])
    assert len(result["final_summary"]["full_text_not_retrieved"]) == 1
    flow = ResearchCompletion(store).prisma_flow(review["id"])
    assert flow["eligibility"]["reports_not_retrieved"] == 1
    assert flow["eligibility"]["full_text_assessed"] == 0

