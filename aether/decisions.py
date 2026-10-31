"""Outcome-grounded decision ledger and conservative calibration-aware abstention.

This is a runtime decision gate, not RLCD model training. It only trusts observed
outcomes from comparable decisions in the same domain and confidence band.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any
from uuid import uuid4

from aether.calibration import CalibrationReport, evaluate_binary_decisions
from aether.storage import Store


@dataclass(frozen=True)
class DecisionAssessment:
    domain: str
    probability_true: float
    predicted_label: bool
    confidence: float
    sample_count: int
    empirical_accuracy: float | None
    wilson_lower_bound: float | None
    action: str
    reason: str


def wilson_lower_bound(successes: int, trials: int, z: float = 1.96) -> float:
    if trials < 1 or successes < 0 or successes > trials:
        raise ValueError("Wilson interval requires 0 <= successes <= trials and trials > 0")
    rate = successes / trials
    z2 = z * z
    denominator = 1 + z2 / trials
    center = rate + z2 / (2 * trials)
    margin = z * sqrt((rate * (1 - rate) + z2 / (4 * trials)) / trials)
    return (center - margin) / denominator


class DecisionManager:
    """Record typed binary predictions and abstain until empirical evidence is sufficient."""

    def __init__(self, store: Store):
        self.store = store

    def log_prediction(
        self, question: str, probability_true: float, domain: str,
        probability_source: str = "external-calibrated-scorer",
    ) -> str:
        text = question.strip()
        category = domain.strip().casefold()
        if not text or not category:
            raise ValueError("Decision question and domain are required")
        if (isinstance(probability_true, bool)
                or not isinstance(probability_true, (int, float))
                or not 0 <= probability_true <= 1):
            raise ValueError("probability_true must be between zero and one")
        if not probability_source.strip():
            raise ValueError("probability_source is required for provenance")
        decision_id = uuid4().hex
        self.store.record_decision(
            decision_id, text, float(probability_true),
            predicted_label=probability_true >= 0.5,
            metadata={
                "domain": category,
                "probability_source": probability_source,
                "decision_schema": "binary-v1",
            },
        )
        return decision_id

    def resolve(self, decision_id: str, outcome: bool) -> None:
        self.store.resolve_decision(decision_id, outcome)

    def assess(
        self, probability_true: float, domain: str, *,
        min_samples: int = 30, min_accuracy: float = 0.90,
    ) -> DecisionAssessment:
        category = domain.strip().casefold()
        if not category:
            raise ValueError("Decision domain is required")
        if isinstance(probability_true, bool) or not isinstance(probability_true, (int, float)):
            raise ValueError("probability_true must be a number")
        if not 0 <= probability_true <= 1:
            raise ValueError("probability_true must be between zero and one")
        if min_samples < 5:
            raise ValueError("min_samples must be at least five")
        if not 0.5 < min_accuracy <= 1:
            raise ValueError("min_accuracy must be greater than 0.5 and at most 1")

        predicted = probability_true >= 0.5
        confidence = probability_true if predicted else 1 - probability_true
        band = min(9, int(confidence * 10))
        comparable = [
            item for item in self.store.list_decisions(resolved_only=True, limit=10000)
            if item["metadata"].get("domain") == category
            and min(9, int(max(
                item["probability"], 1 - item["probability"]
            ) * 10)) == band
        ]
        successes = sum(
            int(bool(item["outcome"]) == bool(item["predicted_label"]))
            for item in comparable
        )
        count = len(comparable)
        accuracy = successes / count if count else None
        lower = wilson_lower_bound(successes, count) if count else None

        if count < min_samples:
            action = "abstain"
            reason = f"Only {count}/{min_samples} resolved examples in this domain/confidence band"
        elif lower is None or lower < min_accuracy:
            action = "abstain"
            reason = (
                f"Empirical evidence is insufficient: Wilson lower bound "
                f"{lower:.3f} is below required {min_accuracy:.3f}"
            )
        else:
            action = "accept"
            reason = (
                f"Observed accuracy {accuracy:.3f}; Wilson lower bound "
                f"{lower:.3f} meets the {min_accuracy:.3f} threshold"
            )
        return DecisionAssessment(
            domain=category, probability_true=float(probability_true),
            predicted_label=predicted, confidence=float(confidence),
            sample_count=count, empirical_accuracy=accuracy,
            wilson_lower_bound=lower, action=action, reason=reason,
        )

    def metrics(self, domain: str | None = None) -> CalibrationReport:
        records = self.store.list_decisions(resolved_only=True, limit=10000)
        if domain is not None:
            category = domain.strip().casefold()
            records = [
                item for item in records if item["metadata"].get("domain") == category
            ]
        return evaluate_binary_decisions(records)
