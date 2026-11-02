from pathlib import Path
from types import SimpleNamespace
import subprocess

from aether.acceptance import ACCEPTANCE_CASES, run_acceptance_suite


def test_acceptance_suite_records_expected_actual_and_partial_scope(tmp_path):
    calls = []

    def fake_runner(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="3 passed in 0.2s", stderr="")

    report = run_acceptance_suite(tmp_path, runner=fake_runner)

    assert len(report["results"]) == 12
    assert len(calls) == 12
    assert report["counts"] == {
        "PASS": 4, "PARTIAL": 8, "FAIL": 0, "BLOCKED": 0
    }
    assert report["complete"] is False
    for result in report["results"]:
        assert result["expected"]
        assert result["actual"] == "3 passed in 0.2s"
        assert result["test_execution"] == "PASS"
        assert result["acceptance_status"] in {"PASS", "PARTIAL"}


def test_acceptance_suite_marks_failed_group_as_fail(tmp_path):
    def fake_runner(command, **kwargs):
        return SimpleNamespace(returncode=1, stdout="1 failed", stderr="")

    report = run_acceptance_suite(tmp_path, runner=fake_runner)

    assert report["counts"]["FAIL"] == 12
    assert report["complete"] is False


def test_acceptance_suite_marks_missing_pytest_as_blocked(tmp_path):
    def fake_runner(command, **kwargs):
        return SimpleNamespace(
            returncode=1, stdout="", stderr="No module named pytest"
        )

    report = run_acceptance_suite(tmp_path, runner=fake_runner)

    assert report["counts"]["BLOCKED"] == 12


def test_acceptance_suite_handles_timeout_and_writes_json(tmp_path):
    def fake_runner(command, **kwargs):
        raise subprocess.TimeoutExpired(command, timeout=kwargs["timeout"])

    output = tmp_path / "reports" / "acceptance.json"
    report = run_acceptance_suite(
        tmp_path, timeout_seconds=1, output_path=output, runner=fake_runner
    )

    assert report["counts"]["FAIL"] == 12
    assert output.exists()
    assert '"complete": false' in output.read_text(encoding="utf-8")


def test_all_12_acceptance_cases_have_explicit_test_nodes_and_scope():
    assert [case["id"] for case in ACCEPTANCE_CASES] == list("ABCDEFGHIJKL")
    assert all(case["nodes"] for case in ACCEPTANCE_CASES)
    assert all(case["expected"] and case["limitations"] is not None for case in ACCEPTANCE_CASES)
