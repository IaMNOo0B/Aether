import pytest

from aether.calibration import evaluate_binary_decisions
from aether.storage import Store


def test_binary_calibration_metrics_are_computed_from_resolved_outcomes():
    report = evaluate_binary_decisions([
        {"probability": 0.9, "outcome": True},
        {"probability": 0.8, "outcome": False},
        {"probability": 0.2, "outcome": False},
        {"probability": 0.1, "outcome": False},
    ], bin_count=10, risk_limit=0.05)

    assert report.count == 4
    assert report.brier_score == pytest.approx(0.175)
    assert report.expected_calibration_error == pytest.approx(0.3)
    assert 0 <= report.aurc <= 1
    assert report.coverage_at_risk == pytest.approx(0.5)


def test_calibration_rejects_invalid_probabilities_and_empty_data():
    with pytest.raises(ValueError):
        evaluate_binary_decisions([])
    with pytest.raises(ValueError):
        evaluate_binary_decisions([{"probability": 1.2, "outcome": True}])


def test_decision_outcomes_persist_and_cannot_be_resolved_twice(tmp_path):
    store = Store(tmp_path / "decisions.sqlite3")
    store.record_decision("d-1", "Will the test pass?", 0.8, True)
    assert store.list_decisions() == []
    store.resolve_decision("d-1", True)
    records = store.list_decisions()
    assert len(records) == 1
    assert records[0]["probability"] == 0.8
    assert records[0]["outcome"] == 1
    with pytest.raises(ValueError, match="already resolved"):
        store.resolve_decision("d-1", False)
    with pytest.raises(KeyError):
        store.resolve_decision("missing", True)
