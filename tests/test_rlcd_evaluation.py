import pytest

from aether.rlcd_evaluation import HistogramCalibrator, evaluate_held_out


def sample_records(count=100):
    return [
        {
            "id": f"decision-{index:03d}",
            "probability": 0.9 if index % 2 == 0 else 0.6,
            "outcome": index % 3 != 0,
        }
        for index in range(count)
    ]


def test_held_out_benchmark_is_deterministic_and_reports_disjoint_counts():
    first = evaluate_held_out(sample_records(), seed=17)
    second = evaluate_held_out(sample_records(), seed=17)

    assert first == second
    assert first.sample_count == 100
    assert first.train_count + first.held_out_count == 100
    assert first.held_out_count > 0
    assert len(first.dataset_sha256) == 64
    assert first.baseline["count"] == first.held_out_count
    assert first.calibrated["count"] == first.held_out_count
    assert "not OpenJev-RLCD training" in first.interpretation


def test_held_out_benchmark_changes_split_when_seed_changes():
    first = evaluate_held_out(sample_records(), seed=1)
    second = evaluate_held_out(sample_records(), seed=2)
    assert first.dataset_sha256 == second.dataset_sha256
    assert first.seed != second.seed
    assert first.baseline != second.baseline or first.calibrated != second.calibrated


def test_held_out_benchmark_rejects_small_or_invalid_datasets():
    with pytest.raises(ValueError, match="At least 10"):
        evaluate_held_out(sample_records(9))
    with pytest.raises(ValueError, match="resolved binary"):
        evaluate_held_out([{"probability": 0.5, "outcome": None}] * 10)
    with pytest.raises(ValueError, match="between 0.5 and 0.9"):
        evaluate_held_out(sample_records(), train_fraction=0.4)


def test_histogram_calibrator_uses_smoothed_training_frequency_only():
    calibrator = HistogramCalibrator(2).fit([
        type("Row", (), {"probability": 0.1, "outcome": 1})(),
        type("Row", (), {"probability": 0.2, "outcome": 1})(),
        type("Row", (), {"probability": 0.3, "outcome": 0})(),
    ])
    assert calibrator.predict(0.1) == pytest.approx(3 / 5)
    assert calibrator.predict(0.9) == 0.9
