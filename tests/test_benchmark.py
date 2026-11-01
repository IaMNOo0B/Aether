import json
from pathlib import Path

from aether.benchmark import run_phase1_benchmark


def test_phase1_reference_benchmark(tmp_path):
    dataset = Path(__file__).resolve().parents[1] / "benchmarks" / "phase1_cases.json"
    cases = json.loads(dataset.read_text(encoding="utf-8"))
    report = run_phase1_benchmark(cases, tmp_path)
    assert report["metrics"]["total"] == 5
    assert report["metrics"]["passed"] == 5
    assert report["metrics"]["completion_rate"] == 1.0
