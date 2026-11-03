"""Reproducible held-out skill evaluation with explicit baseline comparison."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Callable, Iterable, Any
import json


@dataclass(frozen=True)
class HeldOutCase:
    case_id: str
    input: str
    expected_output: str
    category: str = "transfer"
    split: str = "held_out"


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def run_skill_benchmark(
    cases: Iterable[HeldOutCase],
    skill_executor: Callable[[str], str],
    baseline_executor: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Execute a supplied workflow against a caller-owned held-out test set.

    The harness does not invent a workflow executor. The caller must provide a
    real implementation adapter; exceptions are recorded as failures.
    """
    selected = list(cases)
    if not selected or len(selected) > 500:
        raise ValueError("Benchmark requires 1..500 held-out cases")
    identifiers = [case.case_id for case in selected]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Held-out case IDs must be unique")
    for case in selected:
        if (not case.case_id.strip() or not case.input.strip()
                or not case.expected_output.strip() or case.split != "held_out"
                or not case.category.strip()):
            raise ValueError("Every benchmark case must be complete and explicitly held_out")

    serialized = json.dumps(
        [case.__dict__ for case in selected], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    outcomes = []
    skill_passes = 0
    baseline_passes = 0
    for case in selected:
        try:
            actual = skill_executor(case.input)
            passed = isinstance(actual, str) and _normalize(actual) == _normalize(case.expected_output)
            error = None
        except Exception as exc:
            passed, error = False, type(exc).__name__
        skill_passes += int(passed)
        row = {
            "case_id": case.case_id, "category": case.category,
            "skill_passed": passed, "error": error,
        }
        if baseline_executor is not None:
            try:
                baseline = baseline_executor(case.input)
                baseline_passed = (
                    isinstance(baseline, str)
                    and _normalize(baseline) == _normalize(case.expected_output)
                )
            except Exception:
                baseline_passed = False
            baseline_passes += int(baseline_passed)
            row["baseline_passed"] = baseline_passed
        outcomes.append(row)

    total = len(selected)
    by_category = {}
    for category in sorted({case.category for case in selected}):
        rows = [row for row in outcomes if row["category"] == category]
        by_category[category] = {
            "count": len(rows),
            "skill_accuracy": sum(row["skill_passed"] for row in rows) / len(rows),
            **({
                "baseline_accuracy": sum(row["baseline_passed"] for row in rows) / len(rows)
            } if baseline_executor is not None else {}),
        }
    report = {
        "benchmark": "AETHER-held-out-skill-v1",
        "dataset_sha256": sha256(serialized).hexdigest(),
        "cases": total,
        "skill_accuracy": skill_passes / total,
        "outcomes": outcomes,
        "by_category": by_category,
        "evaluation_split": "held_out",
        "baseline_accuracy": baseline_passes / total if baseline_executor is not None else None,
        "accuracy_delta": (
            (skill_passes - baseline_passes) / total
            if baseline_executor is not None else None
        ),
        "limitations": [
            "Exact normalized output matching is task-specific and does not measure semantic quality.",
            "The harness does not establish causal learning; the caller supplies the skill executor and held-out cases.",
            "A single run does not establish statistical significance or long-term retention.",
        ],
    }
    return report
