"""Held-out, transfer and retention evaluation for learned procedures.

Execution is delegated to an explicitly supplied trusted executor. This module
never evals generated procedure text or executes model-produced Python.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable
from uuid import uuid4

from aether.storage import Store


SPLITS = {"held_out", "transfer", "retention"}


@dataclass(frozen=True)
class SkillEvaluationCase:
    case_id: str
    split: str
    task: Any
    expected: Any

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError("Evaluation case_id is required")
        if self.split not in SPLITS:
            raise ValueError(f"Evaluation split must be one of {sorted(SPLITS)}")


@dataclass(frozen=True)
class SkillCaseResult:
    case_id: str
    split: str
    passed: bool
    error: str | None


@dataclass(frozen=True)
class SkillEvaluationReport:
    evaluation_id: str
    skill_id: str
    total_cases: int
    passed_cases: int
    accuracy: float
    split_accuracy: dict[str, float]
    validated: bool
    results: tuple[SkillCaseResult, ...]
    limitations: tuple[str, ...]


class SkillBenchmarkRunner:
    """Evaluate a draft skill through a trusted, caller-supplied execution adapter."""

    def __init__(self, store: Store):
        self.store = store

    def run(
        self,
        skill_id: str,
        cases: Iterable[SkillEvaluationCase],
        executor: Callable[[str, Any], Any],
        scorer: Callable[[Any, Any], bool],
        *,
        minimum_accuracy: float = 0.9,
        minimum_split_accuracy: float = 0.8,
    ) -> SkillEvaluationReport:
        if not 0 < minimum_accuracy <= 1 or not 0 < minimum_split_accuracy <= 1:
            raise ValueError("Accuracy thresholds must be in (0, 1]")
        skill = self.store.get_skill(skill_id)
        if skill is None:
            raise KeyError(f"Unknown skill: {skill_id}")
        if skill["status"] == "validated":
            raise ValueError("A validated skill must be forked into a new draft before re-evaluation")
        items = list(cases)
        if not items:
            raise ValueError("At least one evaluation case is required")
        ids = [item.case_id for item in items]
        if len(ids) != len(set(ids)):
            raise ValueError("Evaluation case IDs must be unique")
        missing = SPLITS - {item.split for item in items}
        if missing:
            raise ValueError(f"Evaluation requires held_out, transfer and retention cases; missing {sorted(missing)}")

        results: list[SkillCaseResult] = []
        split_totals = {split: 0 for split in SPLITS}
        split_passed = {split: 0 for split in SPLITS}
        for case in items:
            split_totals[case.split] += 1
            try:
                actual = executor(skill["procedure"], case.task)
                passed = bool(scorer(actual, case.expected))
                error = None
            except Exception as exc:
                passed = False
                error = f"{type(exc).__name__}: {str(exc)[:300]}"
            split_passed[case.split] += int(passed)
            results.append(SkillCaseResult(case.case_id, case.split, passed, error))

        total_passed = sum(split_passed.values())
        accuracy = total_passed / len(items)
        split_accuracy = {
            split: split_passed[split] / split_totals[split]
            for split in sorted(SPLITS)
        }
        validated = (
            accuracy >= minimum_accuracy
            and all(value >= minimum_split_accuracy for value in split_accuracy.values())
        )
        evaluation_id = uuid4().hex
        report = SkillEvaluationReport(
            evaluation_id=evaluation_id,
            skill_id=skill_id,
            total_cases=len(items),
            passed_cases=total_passed,
            accuracy=round(accuracy, 6),
            split_accuracy={key: round(value, 6) for key, value in split_accuracy.items()},
            validated=validated,
            results=tuple(results),
            limitations=(
                "The executor is supplied by the caller and defines the actual execution semantics.",
                "A passing fixture suite does not establish broad generalization outside the tested distributions.",
                "This evaluates procedural execution; it does not update model weights or demonstrate meta-learning.",
            ),
        )
        evidence = {
            "evaluation_id": evaluation_id,
            "report": asdict(report),
            "executor": getattr(executor, "__qualname__", type(executor).__name__),
        }
        self.store.evaluate_skill(
            skill_id, validated,
            __import__("json").dumps(evidence, sort_keys=True),
        )
        return report
