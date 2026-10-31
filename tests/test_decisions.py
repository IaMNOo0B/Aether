import pytest

from aether.decisions import DecisionManager, wilson_lower_bound
from aether.storage import Store


def test_wilson_lower_bound_is_conservative_and_validates_counts():
    assert 0.90 < wilson_lower_bound(40, 40) < 1.0
    assert wilson_lower_bound(0, 40) < 0.1
    with pytest.raises(ValueError):
        wilson_lower_bound(41, 40)


def test_decision_log_and_resolution_preserve_outcome_provenance(tmp_path):
    manager = DecisionManager(Store(tmp_path / "decisions.sqlite3"))
    decision_id = manager.log_prediction(
        "Will the controlled test pass?", 0.8, "software-tests"
    )
    unresolved = manager.store.list_decisions(resolved_only=False)
    assert unresolved[0]["id"] == decision_id
    assert unresolved[0]["metadata"]["domain"] == "software-tests"
    assert unresolved[0]["outcome"] is None

    manager.resolve(decision_id, True)
    resolved = manager.store.list_decisions(resolved_only=True)
    assert resolved[0]["outcome"] == 1
    with pytest.raises(ValueError, match="already resolved"):
        manager.resolve(decision_id, False)


def test_decision_gate_abstains_without_sufficient_domain_evidence(tmp_path):
    manager = DecisionManager(Store(tmp_path / "decisions.sqlite3"))
    assessment = manager.assess(0.99, "medical-triage")
    assert assessment.action == "abstain"
    assert assessment.sample_count == 0


def test_decision_gate_accepts_only_when_observed_lower_bound_is_high(tmp_path):
    manager = DecisionManager(Store(tmp_path / "decisions.sqlite3"))
    for index in range(40):
        decision_id = manager.log_prediction(
            f"Software test prediction {index}", 0.95, "software-tests"
        )
        manager.resolve(decision_id, True)

    assessment = manager.assess(
        0.95, "software-tests", min_samples=30, min_accuracy=0.90
    )
    assert assessment.action == "accept"
    assert assessment.sample_count == 40
    assert assessment.wilson_lower_bound >= 0.90


def test_decision_gate_abstains_when_historical_accuracy_is_poor(tmp_path):
    manager = DecisionManager(Store(tmp_path / "decisions.sqlite3"))
    for index in range(40):
        decision_id = manager.log_prediction(
            f"Bad prediction {index}", 0.95, "unstable-domain"
        )
        manager.resolve(decision_id, False)

    assessment = manager.assess(
        0.95, "unstable-domain", min_samples=30, min_accuracy=0.90
    )
    assert assessment.action == "abstain"
    assert assessment.wilson_lower_bound < 0.90


def test_decision_gate_never_borrows_samples_from_another_domain(tmp_path):
    manager = DecisionManager(Store(tmp_path / "decisions.sqlite3"))
    for index in range(40):
        decision_id = manager.log_prediction(
            f"Passed {index}", 0.95, "software-tests"
        )
        manager.resolve(decision_id, True)

    assessment = manager.assess(0.95, "finance", min_samples=30)
    assert assessment.action == "abstain"
    assert assessment.sample_count == 0


def test_decision_metrics_require_resolved_outcomes(tmp_path):
    manager = DecisionManager(Store(tmp_path / "decisions.sqlite3"))
    decision_id = manager.log_prediction("Will it pass?", 0.7, "software-tests")
    with pytest.raises(ValueError, match="At least one resolved decision"):
        manager.metrics("software-tests")
    manager.resolve(decision_id, True)
    assert manager.metrics("software-tests").count == 1
