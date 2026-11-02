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
    domains = {
        name: {"rating": "low", "rationale": "The paper provides sufficient detail for this domain."}
        for name in (
            "randomization_process", "deviations_from_intended_interventions",
            "missing_outcome_data", "outcome_measurement", "selection_of_reported_result",
        )
    }
    result = workflow.record_risk_of_bias(
        "review-1", "https://example.org/study", "randomized",
        "reviewer-1", "correct horse battery", domains,
    )
    assert result["instrument"] == "AETHER-structured-randomized-v1"
    assert "not a validated" in result["limitation"]
    with pytest.raises(PermissionError):
        workflow.record_risk_of_bias(
            "review-1", "https://example.org/study", "randomized",
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
    assert store.list_research_experiment_events("exp-raw")[-1]["event_type"] == "raw_observations_ingested"
    with pytest.raises(ValueError, match="Unexpected group"):
        workflow.ingest_raw_observations(
            "exp-raw", b"group,value\ncontrol,1\nother,2\n", "group", "value", "treatment", "control"
        )
