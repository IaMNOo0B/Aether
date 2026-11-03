"""Typed, held-out evaluation for decision-probability producers.

This is an evaluation harness, not RLCD training. It requires externally observed
outcomes and a producer implementation supplied by the integration.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol, Iterable, Any
import json
import math

from aether.calibration import evaluate_binary_decisions


@dataclass(frozen=True)
class DecisionCase:
    case_id: str
    question: str
    domain: str
    outcome: bool
    split: str = "held_out"


@dataclass(frozen=True)
class DecisionPrediction:
    probability_true: float
    source: str


class DecisionProducer(Protocol):
    def predict(self, question: str, domain: str) -> DecisionPrediction: ...


def _run_producer(
    cases: list[DecisionCase], producer: DecisionProducer,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records, predictions = [], []
    for case in cases:
        prediction = producer.predict(case.question, case.domain)
        if not isinstance(prediction, DecisionPrediction):
            raise TypeError("Decision producer must return DecisionPrediction")
        if (not math.isfinite(prediction.probability_true)
                or not 0 <= prediction.probability_true <= 1
                or not prediction.source.strip()):
            raise ValueError("Decision prediction requires a finite probability and source")
        records.append({"probability": prediction.probability_true, "outcome": case.outcome})
        predictions.append({
            "case_id": case.case_id, "domain": case.domain,
            "probability_true": prediction.probability_true,
            "source": prediction.source,
            "outcome": case.outcome,
        })
    return records, predictions


def evaluate_held_out_decision_producer(
    cases: Iterable[DecisionCase],
    producer: DecisionProducer,
    baseline: DecisionProducer | None = None,
) -> dict[str, Any]:
    selected = list(cases)
    if not selected or len(selected) > 10000:
        raise ValueError("Decision benchmark requires 1..10000 held-out cases")
    if len({case.case_id for case in selected}) != len(selected):
        raise ValueError("Decision case IDs must be unique")
    if any(
        not case.case_id.strip() or not case.question.strip()
        or not case.domain.strip() or case.split != "held_out"
        for case in selected
    ):
        raise ValueError("Every decision case must be complete and explicitly held_out")
    dataset = json.dumps(
        [case.__dict__ for case in selected], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    records, predictions = _run_producer(selected, producer)
    metrics = evaluate_binary_decisions(records)
    result: dict[str, Any] = {
        "benchmark": "AETHER-RLCD-held-out-v1",
        "dataset_sha256": sha256(dataset).hexdigest(),
        "split": "held_out",
        "count": len(selected),
        "producer": {
            "brier_score": metrics.brier_score,
            "log_loss": metrics.log_loss,
            "expected_calibration_error": metrics.expected_calibration_error,
            "aurc": metrics.aurc,
            "coverage_at_risk": metrics.coverage_at_risk,
        },
        "predictions": predictions,
        "baseline": None,
        "limitations": [
            "This evaluates probability outputs against supplied observed outcomes; it does not train or reproduce RLCD.",
            "Results depend on the representativeness and independence of the held-out cases.",
            "Calibration metrics do not by themselves establish decision utility or causal improvement.",
        ],
    }
    if baseline is not None:
        baseline_records, baseline_predictions = _run_producer(selected, baseline)
        baseline_metrics = evaluate_binary_decisions(baseline_records)
        result["baseline"] = {
            "brier_score": baseline_metrics.brier_score,
            "log_loss": baseline_metrics.log_loss,
            "expected_calibration_error": baseline_metrics.expected_calibration_error,
            "aurc": baseline_metrics.aurc,
            "coverage_at_risk": baseline_metrics.coverage_at_risk,
            "predictions": baseline_predictions,
        }
        result["delta_brier_score"] = (
            metrics.brier_score - baseline_metrics.brier_score
        )
        result["delta_log_loss"] = metrics.log_loss - baseline_metrics.log_loss
    return result
