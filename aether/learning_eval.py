"""Held-out, transfer and retention evaluation for learned procedures and adapters.

Executors are injected by the caller so this harness never executes untrusted
skill text itself. Results are deterministic records, not model-training claims.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from statistics import mean
from typing import Callable, Iterable, Sequence
import json
import platform
import time


@dataclass(frozen=True)
class LearningCase:
    case_id: str
    task: str
    expected: str
    split: str = "held_out"
    domain: str = "same_domain"

    def validate(self) -> None:
        if not self.case_id.strip() or not self.task.strip() or not self.expected.strip():
            raise ValueError("Learning cases require an ID, task and expected result")
        if self.split not in {"train", "validation", "held_out", "transfer", "retention"}:
            raise ValueError("Unknown learning evaluation split")


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    split: str
    domain: str
    passed: bool
    output_sha256: str
    elapsed_seconds: float


def evaluate_executor(
    cases: Iterable[LearningCase],
    executor: Callable[[str], str],
    *,
    system_version: str,
    repetitions: int = 1,
) -> dict:
    rows = list(cases)
    if not rows or not system_version.strip() or not 1 <= repetitions <= 20:
        raise ValueError("Cases, system_version and a repetition count from 1 to 20 are required")
    ids = [case.case_id for case in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation case IDs must be unique")
    results: list[dict] = []
    for case in rows:
        case.validate()
        for repetition in range(repetitions):
            started = time.perf_counter()
            output = executor(case.task)
            elapsed = time.perf_counter() - started
            if not isinstance(output, str):
                raise TypeError("Learning executor must return text")
            normalized = " ".join(output.casefold().split())
            expected = " ".join(case.expected.casefold().split())
            results.append(asdict(CaseResult(
                case_id=case.case_id, split=case.split, domain=case.domain,
                passed=expected in normalized,
                output_sha256=sha256(output.encode("utf-8")).hexdigest(),
                elapsed_seconds=elapsed,
            )) | {"repetition": repetition + 1})
    grouped: dict[str, dict] = {}
    for split in ("train", "validation", "held_out", "transfer", "retention"):
        members = [item for item in results if item["split"] == split]
        if members:
            grouped[split] = {
                "cases": len({item["case_id"] for item in members}),
                "trials": len(members),
                "pass_rate": sum(item["passed"] for item in members) / len(members),
                "mean_latency_seconds": mean(item["elapsed_seconds"] for item in members),
            }
    return {
        "schema": "aether-learning-evaluation-v1",
        "system_version": system_version,
        "python_version": platform.python_version(),
        "created_at_epoch": time.time(),
        "metrics": grouped,
        "results": results,
        "limitations": [
            "Substring matching is a simple task-specific acceptance rule, not semantic grading.",
            "Retention is measured only when the caller supplies retention cases after a defined delay.",
            "No model-weight update or meta-learning is implied by this evaluation.",
        ],
    }


def compare_learning_runs(before: dict, after: dict, *, target_split: str = "held_out") -> dict:
    left = before.get("metrics", {}).get(target_split)
    right = after.get("metrics", {}).get(target_split)
    if not left or not right:
        raise ValueError(f"Both runs require {target_split!r} metrics")
    delta = float(right["pass_rate"]) - float(left["pass_rate"])
    return {
        "split": target_split,
        "before_pass_rate": left["pass_rate"],
        "after_pass_rate": right["pass_rate"],
        "absolute_delta": delta,
        "improved": delta > 0,
        "regressed": delta < 0,
        "interpretation": "Observed difference on the supplied cases only; repeat with multiple seeds and independent cases before generalizing.",
    }


def save_evaluation(report: dict, path: str) -> str:
    encoded = json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8")
    from pathlib import Path
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(encoded)
    return sha256(encoded).hexdigest()
