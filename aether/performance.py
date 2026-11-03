"""Repeatable baseline/candidate performance and output-regression benchmark."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import statistics
import time
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class PerformanceCase:
    case_id: str
    input: Any


class PerformanceBenchmark:
    def __init__(self, clock: Callable[[], float] = time.perf_counter):
        self.clock = clock

    @staticmethod
    def _digest(value: Any) -> str:
        encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        return sha256(encoded).hexdigest()

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        ordered = sorted(values)
        index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
        return ordered[index]

    def _measure(
        self, model_ref: str, cases: list[PerformanceCase],
        executor: Callable[[str, Any], Any], repetitions: int, warmups: int,
    ) -> dict[str, Any]:
        for _ in range(warmups):
            for case in cases:
                executor(model_ref, case.input)
        durations: list[float] = []
        output_hashes: dict[str, list[str]] = {case.case_id: [] for case in cases}
        for _ in range(repetitions):
            for case in cases:
                started = self.clock()
                output = executor(model_ref, case.input)
                elapsed = self.clock() - started
                if not math.isfinite(elapsed) or elapsed < 0:
                    raise ValueError("Benchmark clock returned an invalid duration")
                durations.append(elapsed)
                output_hashes[case.case_id].append(self._digest(output))
        return {
            "samples": len(durations),
            "median_seconds": statistics.median(durations),
            "p95_seconds": self._percentile(durations, 0.95),
            "mean_seconds": statistics.fmean(durations),
            "output_hashes": output_hashes,
        }

    def compare(
        self,
        baseline_ref: str,
        candidate_ref: str,
        cases: Iterable[PerformanceCase],
        executor: Callable[[str, Any], Any],
        *,
        repetitions: int = 5,
        warmups: int = 1,
    ) -> dict[str, Any]:
        if not baseline_ref.strip() or not candidate_ref.strip() or baseline_ref == candidate_ref:
            raise ValueError("Distinct baseline and candidate model references are required")
        if not 1 <= repetitions <= 100 or not 0 <= warmups <= 20:
            raise ValueError("Benchmark repetitions/warmups exceed the bounded limits")
        items = list(cases)
        if not items or any(not item.case_id.strip() for item in items):
            raise ValueError("At least one named benchmark case is required")
        ids = [item.case_id for item in items]
        if len(ids) != len(set(ids)):
            raise ValueError("Benchmark case IDs must be unique")
        baseline = self._measure(baseline_ref, items, executor, repetitions, warmups)
        candidate = self._measure(candidate_ref, items, executor, repetitions, warmups)
        regressions = []
        for case_id in ids:
            baseline_hashes = baseline["output_hashes"][case_id]
            candidate_hashes = candidate["output_hashes"][case_id]
            if len(set(baseline_hashes)) != 1 or len(set(candidate_hashes)) != 1:
                regressions.append({"case_id": case_id, "reason": "nondeterministic_output"})
            elif baseline_hashes[0] != candidate_hashes[0]:
                regressions.append({"case_id": case_id, "reason": "output_changed"})
        speedup = (
            baseline["median_seconds"] / candidate["median_seconds"]
            if candidate["median_seconds"] > 0 else None
        )
        return {
            "benchmark": "baseline-candidate-performance-v1",
            "baseline_ref": baseline_ref,
            "candidate_ref": candidate_ref,
            "case_count": len(items),
            "repetitions": repetitions,
            "warmups": warmups,
            "baseline": {key: value for key, value in baseline.items() if key != "output_hashes"},
            "candidate": {key: value for key, value in candidate.items() if key != "output_hashes"},
            "speedup": speedup,
            "output_regressions": regressions,
            "equivalent_outputs": not regressions,
            "input_sha256": self._digest([asdict(item) for item in items]),
            "limitations": [
                "Results depend on executor, machine load, model cache and hardware state.",
                "Equivalent hashes check exact serialized outputs, not semantic quality.",
                "Use repeated runs on the target Windows machine before making performance claims.",
            ],
        }
