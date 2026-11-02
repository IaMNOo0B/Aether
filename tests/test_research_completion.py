from __future__ import annotations

import pytest

from aether.research_completion import ResearchCompletion
from aether.storage import Store


def test_reviewer_password_authentication_and_duplicate_identity(tmp_path):
    workflow = ResearchCompletion(Store(tmp_path / "aether.sqlite3"))
    workflow.register_reviewer("reviewer-1", "correct horse battery")
    assert workflow.authenticate_reviewer("reviewer-1", "correct horse battery")
    assert not workflow.authenticate_reviewer("reviewer-1", "wrong password")
    assert not workflow.authenticate_reviewer("unknown", "wrong password")
    with pytest.raises(ValueError, match="already exists"):
        workflow.register_reviewer("reviewer-1", "another strong password")


def test_design_specific_risk_of_bias_requires_exact_domains(tmp_path):
    workflow = ResearchCompletion(Store(tmp_path / "aether.sqlite3"))
    workflow.register_reviewer("reviewer-1", "correct horse battery")
    workflow.register_reviewer("reviewer-2", "another correct battery")
    from aether.research import ResearchSource
    from aether.research_advanced import AdvancedResearch
    advanced = AdvancedResearch(workflow.store)
    review = advanced.create_systematic_review(
        "Does intervention X improve outcome Y?", ["Human studies"], ["Animal studies"],
        {"databases": ["Crossref"], "queries": ["intervention X"], "searched_on": "2026-11-02"},
    )
    source = ResearchSource(title="Study", url="https://example.org/study", publisher="Journal", published="2026", evidence_type="article", summary="Study summary.")
    advanced.add_review_sources(review["id"], [source])
    domains = {
        name: {"rating": "low", "rationale": "The paper provides sufficient detail for this domain."}
        for name in (
            "randomization_process", "deviations_from_intended_interventions",
            "missing_outcome_data", "outcome_measurement", "selection_of_reported_result",
        )
    }
    result = workflow.record_risk_of_bias(
        review["id"], "https://example.org/study", "randomized",
        "reviewer-1", "correct horse battery", domains,
    )
    assert result["instrument"] == "AETHER-structured-randomized-v1"
    assert result["status"] == "awaiting_second_reviewer"
    assert "not a validated" in result["payload"]["limitation"]
    second = workflow.record_risk_of_bias(
        review["id"], "https://example.org/study", "randomized",
        "reviewer-2", "another correct battery", domains,
    )
    assert second["status"] == "agreed"
    with pytest.raises(PermissionError):
        workflow.record_risk_of_bias(
            review["id"], "https://example.org/study", "randomized",
            "reviewer-1", "bad password", domains,
        )


def test_grade_certainty_is_explicit_and_deterministic(tmp_path):
    workflow = ResearchCompletion(Store(tmp_path / "aether.sqlite3"))
    judgments = {
        name: {"judgment": "none", "rationale": "No downgrade is justified by this evidence."}
        for name in ("risk_of_bias", "inconsistency", "indirectness", "imprecision", "publication_bias")
    }
    judgments["imprecision"] = {"judgment": "serious", "rationale": "The confidence interval includes materially different effects."}
    result = workflow.grade_certainty("review-1", "mortality", "high", judgments, "The evidence base is limited and requires cautious interpretation.")
    assert result["certainty"] == "moderate"
    assert result["total_downgrades"] == 1
    assert workflow.store.get_research_report(result["report_id"]) is not None


def test_prisma_style_flow_report_requires_existing_review(tmp_path):
    workflow = ResearchCompletion(Store(tmp_path / "aether.sqlite3"))
    with pytest.raises(KeyError):
        workflow.prisma_flow("missing")


def test_egger_test_refuses_small_study_count_and_runs_at_ten(tmp_path):
    workflow = ResearchCompletion(Store(tmp_path / "aether.sqlite3"))
    assert workflow.egger_publication_bias([
        {"effect_estimate": float(i), "standard_error": 1.0 + i / 10}
        for i in range(9)
    ])["status"] == "not_tested_fewer_than_10_studies"
    result = workflow.egger_publication_bias([
        {"effect_estimate": float(i) / 10, "standard_error": 0.5 + i / 20}
        for i in range(10)
    ])
    assert result["status"] == "tested"
    assert 0 <= result["two_sided_p_approximation"] <= 1


def test_raw_observations_are_validated_and_saved_to_experiment_history(tmp_path):
    store = Store(tmp_path / "aether.sqlite3")
    store.create_research_experiment(
        "exp-raw", "Raw data", None,
        {"objective": "Compare groups", "method": "Measure", "success_criteria": "Treatment lower"},
    )
    store.transition_research_experiment("exp-raw", "running")
    workflow = ResearchCompletion(store)
    data = b"group,value\ncontrol,10\ncontrol,12\ncontrol,14\ntreatment,7\ntreatment,8\ntreatment,9\n"
    result = workflow.ingest_raw_observations("exp-raw", data, "group", "value", "treatment", "control")
    assert result["mean_difference_treatment_minus_control"] == -4
    assert result["standard_error_welch"] > 0
    saved_event = store.list_research_experiment_events("exp-raw")[-1]
    assert saved_event["event_type"] == "raw_observations_ingested"
    import base64
    assert base64.b64decode(saved_event["payload"]["raw_dataset_base64"]) == data
    with pytest.raises(ValueError, match="Unexpected group"):
        workflow.ingest_raw_observations(
            "exp-raw", b"group,value\ncontrol,1\nother,2\n", "group", "value", "treatment", "control"
        )


def test_registered_reviewer_authentication_is_enforced_for_all_review_decisions(tmp_path):
    store = Store(tmp_path / "aether.sqlite3")
    completion = ResearchCompletion(store)
    completion.register_reviewer("screen-a", "correct horse battery")
    completion.register_reviewer("screen-b", "another correct battery")
    workflow = __import__("aether.research_advanced", fromlist=["AdvancedResearch"]).AdvancedResearch(store)
    review = workflow.create_systematic_review(
        "Does intervention X improve outcome Y?",
        ["Human studies"], ["Animal studies"],
        {"databases": ["Crossref"], "queries": ["intervention X"], "searched_on": "2026-11-02"},
    )
    from aether.research import ResearchSource
    source = ResearchSource(
        title="Study", url="https://example.org/study", publisher="Journal",
        published="2026", evidence_type="article", summary="Study summary.",
    )
    workflow.add_review_sources(review["id"], [source])
    with pytest.raises(PermissionError, match="Authenticated reviewer"):
        workflow.screen_source(review["id"], source.url, "screen-a", "include", "The study meets the criteria.")
    workflow.screen_source(
        review["id"], source.url, "screen-a", "include",
        "The study meets the criteria.", password="correct horse battery",
    )
    workflow.screen_source(
        review["id"], source.url, "screen-b", "include",
        "Independent screening confirms the study meets criteria.",
        password="another correct battery",
    )
    assert store.get_systematic_review(review["id"])["sources"][0]["decision"] == "include"


def test_authenticated_stage_and_citation_actions_reject_wrong_credentials(tmp_path):
    store = Store(tmp_path / "aether.sqlite3")
    completion = ResearchCompletion(store)
    completion.register_reviewer("reviewer-a", "correct horse battery")
    workflow = __import__("aether.research_advanced", fromlist=["AdvancedResearch"]).AdvancedResearch(store)
    with pytest.raises(PermissionError, match="Authenticated reviewer"):
        workflow.record_review_stage(
            "missing-review", "https://example.org/study", "full_text", "reviewer-a",
            {"decision": "exclude", "rationale": "This study is outside scope."},
        )
    with pytest.raises(PermissionError, match="Authenticated reviewer"):
        workflow.adjudicate_citation(
            "missing-evidence", "reviewer-a", "insufficient", "The evidence is not sufficient.",
            password="wrong password",
        )

def test_prisma_2020_checklist_requires_all_items_and_persists_completed_report(tmp_path):
    from aether.research_advanced import AdvancedResearch
    from aether.research_completion import PRISMA_2020_CHECKLIST

    store = Store(tmp_path / "aether.sqlite3")
    workflow = ResearchCompletion(store)
    review = AdvancedResearch(store).create_systematic_review(
        "Does intervention X improve outcome Y?", ["Human studies"], ["Animal studies"],
        {"databases": ["Crossref"], "queries": ["intervention X"], "searched_on": "2026-11-02"},
    )
    responses = {
        item_id: {"status": "reported", "location": f"Section {item_id}",
                  "rationale": "The corresponding manuscript section reports this information."}
        for item_id, _, _ in PRISMA_2020_CHECKLIST
    }
    assert len(responses) == 27
    with pytest.raises(ValueError, match="all 27 items"):
        workflow.record_prisma_checklist(review["id"], {"1": responses["1"]})
    result = workflow.record_prisma_checklist(review["id"], responses)
    assert result["checklist_completed"] is True
    assert result["compliance_claim"] is False
    assert result["reported_items"] == 27
    assert store.get_research_report(result["report_id"]) is not None



def test_certainty_grading_requires_authenticated_reviewer_after_registration(tmp_path):
    workflow = ResearchCompletion(Store(tmp_path / "aether.sqlite3"))
    workflow.register_reviewer("grade-reviewer", "correct horse battery")
    judgments = {
        name: {"judgment": "none", "rationale": "The available evidence does not justify a downgrade."}
        for name in ("risk_of_bias", "inconsistency", "indirectness", "imprecision", "publication_bias")
    }
    with pytest.raises(PermissionError, match="Authenticated reviewer"):
        workflow.grade_certainty(
            "review-1", "mortality", "high", judgments,
            "The evidence is consistent but requires careful interpretation.",
        )
    result = workflow.grade_certainty(
        "review-1", "mortality", "high", judgments,
        "The evidence is consistent but requires careful interpretation.",
        reviewer_id="grade-reviewer", password="correct horse battery",
    )
    assert result["reviewer_id"] == "grade-reviewer"
