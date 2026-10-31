"""Calibration metrics for typed binary decisions with observed outcomes."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_probability: float
    observed_frequency: float


@dataclass(frozen=True)
class CalibrationReport:
    count: int
    brier_score: float
    log_loss: float
    expected_calibration_error: float
    aurc: float
    coverage_at_risk: float
    bins: tuple[CalibrationBin, ...]


def evaluate_binary_decisions(
    records: Iterable[Mapping[str, object]],
    bin_count: int = 10,
    risk_limit: float = 0.05,
) -> CalibrationReport:
    """Evaluate logged probabilities against resolved binary outcomes.

    Each record must contain probability (P(outcome=True)) and outcome (bool/0/1).
    Confidence is ranked separately from probability for selective prediction.
    """
    if bin_count < 1:
        raise ValueError("bin_count must be positive")
    if not 0 <= risk_limit <= 1:
        raise ValueError("risk_limit must be between zero and one")

    rows: list[tuple[float, int]] = []
    for record in records:
        probability = record.get("probability")
        outcome = record.get("outcome")
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise ValueError("probability must be a number between zero and one")
        if not math.isfinite(float(probability)) or not 0 <= probability <= 1:
            raise ValueError("probability must be a finite number between zero and one")
        if outcome not in (False, True, 0, 1):
            raise ValueError("outcome must be boolean or binary")
        rows.append((float(probability), int(bool(outcome))))

    if not rows:
        raise ValueError("At least one resolved decision is required")

    count = len(rows)
    brier = sum((probability - outcome) ** 2 for probability, outcome in rows) / count
    epsilon = 1e-15
    log_loss = -sum(
        outcome * math.log(max(probability, epsilon))
        + (1 - outcome) * math.log(max(1 - probability, epsilon))
        for probability, outcome in rows
    ) / count

    bins: list[CalibrationBin] = []
    ece = 0.0
    for index in range(bin_count):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        members = [
            (probability, outcome) for probability, outcome in rows
            if lower <= probability < upper
            or (index == bin_count - 1 and probability == 1.0)
        ]
        if not members:
            continue
        mean_probability = sum(item[0] for item in members) / len(members)
        observed = sum(item[1] for item in members) / len(members)
        ece += len(members) / count * abs(mean_probability - observed)
        bins.append(CalibrationBin(lower, upper, len(members), mean_probability, observed))

    ranked = sorted(
        rows,
        key=lambda item: max(item[0], 1.0 - item[0]),
        reverse=True,
    )
    cumulative_errors = 0
    previous_coverage = 0.0
    previous_risk = 0.0
    aurc = 0.0
    coverage_at_risk = 0.0
    for index, (probability, outcome) in enumerate(ranked, start=1):
        predicted = int(probability >= 0.5)
        cumulative_errors += int(predicted != outcome)
        coverage = index / count
        risk = cumulative_errors / index
        aurc += (coverage - previous_coverage) * (risk + previous_risk) / 2
        if risk <= risk_limit:
            coverage_at_risk = coverage
        previous_coverage = coverage
        previous_risk = risk

    return CalibrationReport(
        count=count,
        brier_score=brier,
        log_loss=log_loss,
        expected_calibration_error=ece,
        aurc=aurc,
        coverage_at_risk=coverage_at_risk,
        bins=tuple(bins),
    )
