import pytest

from aether.calibration_benchmark import CalibrationBenchmark, CalibrationCase


def cases():
    return [
        CalibrationCase("tr1", "train", "general", 0.8, True, "dataset-v1"),
        CalibrationCase("tr2", "train", "general", 0.7, True, "dataset-v1"),
        CalibrationCase("tr3", "train", "general", 0.2, False, "dataset-v1"),
        CalibrationCase("va1", "validation", "general", 0.6, True, "dataset-v1"),
        CalibrationCase("va2", "validation", "general", 0.4, False, "dataset-v1"),
        CalibrationCase("te1", "test", "general", 0.9, True, "dataset-v1"),
        CalibrationCase("te2", "test", "general", 0.8, True, "dataset-v1"),
        CalibrationCase("te3", "test", "general", 0.2, False, "dataset-v1"),
        CalibrationCase("te4", "test", "general", 0.1, False, "dataset-v1"),
    ]


def test_calibration_benchmark_uses_test_split_and_train_only_baseline():
    report = CalibrationBenchmark().run(
        cases(), model_id="test-model@1", dataset_id="fixture-v1", seed=42
    )
    assert report["split_counts"] == {"train": 3, "validation": 2, "test": 4}
    assert report["train_base_rate"] == pytest.approx(2 / 3)
    assert report["test_metrics"]["count"] == 4
    assert report["brier_improvement_vs_train_base_rate"] > 0
    assert len(report["input_sha256"]) == 64
    assert "do not establish universal calibration" in report["interpretation"]


def test_calibration_benchmark_requires_all_disjoint_splits_and_unique_ids():
    benchmark = CalibrationBenchmark()
    with pytest.raises(ValueError, match="train, validation and test"):
        benchmark.run(cases()[:5], model_id="m", dataset_id="d", seed=1)
    duplicate = cases()
    duplicate[-1] = CalibrationCase("tr1", "test", "general", 0.2, False, "dataset-v1")
    with pytest.raises(ValueError, match="unique"):
        benchmark.run(duplicate, model_id="m", dataset_id="d", seed=1)


def test_calibration_case_rejects_nonfinite_or_untyped_outcomes():
    with pytest.raises(ValueError, match="finite"):
        CalibrationCase("x", "test", "general", float("nan"), True, "source")
    with pytest.raises(ValueError, match="boolean"):
        CalibrationCase("x", "test", "general", 0.5, 1, "source")
