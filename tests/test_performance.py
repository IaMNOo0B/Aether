import pytest

from aether.performance import PerformanceBenchmark, PerformanceCase


def test_performance_benchmark_compares_output_and_records_repeatable_metrics():
    cases = [PerformanceCase("trim", "  Aether  "), PerformanceCase("lower", "JARVIS")]
    report = PerformanceBenchmark().compare(
        "baseline@1", "candidate@1", cases,
        lambda model, value: value.strip().lower(),
        repetitions=2, warmups=1,
    )
    assert report["case_count"] == 2
    assert report["baseline"]["samples"] == 4
    assert report["candidate"]["samples"] == 4
    assert report["equivalent_outputs"] is True
    assert report["output_regressions"] == []
    assert len(report["input_sha256"]) == 64


def test_performance_benchmark_flags_candidate_output_regression():
    report = PerformanceBenchmark().compare(
        "baseline", "candidate", [PerformanceCase("case-1", "AETHER")],
        lambda model, value: value if model == "baseline" else value.lower(),
        repetitions=1, warmups=0,
    )
    assert report["equivalent_outputs"] is False
    assert report["output_regressions"] == [
        {"case_id": "case-1", "reason": "output_changed"}
    ]


def test_performance_benchmark_rejects_invalid_cases_and_limits():
    benchmark = PerformanceBenchmark()
    with pytest.raises(ValueError, match="repetitions"):
        benchmark.compare("base", "candidate", [PerformanceCase("x", 1)],
                          lambda model, value: value, repetitions=101)
    with pytest.raises(ValueError, match="unique"):
        benchmark.compare(
            "base", "candidate",
            [PerformanceCase("x", 1), PerformanceCase("x", 2)],
            lambda model, value: value,
        )
