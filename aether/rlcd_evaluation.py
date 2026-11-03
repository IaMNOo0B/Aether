"""Held-out calibration benchmark for AETHER's RLCD investigation.

This module evaluates a simple post-hoc histogram calibrator on a deterministic
train/held-out split. It is an evaluation baseline, not RLCD training and not a
claim that model weights or reasoning policies have been updated.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

from aether.calibration import evaluate_binary_decisions
from aether.storage import Store


@dataclass(frozen=True)
class DecisionExample:
    probability: float
    outcome: int
    decision_id: str = ""

    @classmethod
    def from_mapping(cls, item: Mapping[str, Any]) -> "DecisionExample":
        probability = item.get("probability")
        outcome = item.get("outcome")
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            raise ValueError("Each decision requires a numeric probability")
        probability = float(probability)
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Decision probability must be finite and between zero and one")
        if outcome not in (0, 1, False, True):
            raise ValueError("Each decision requires a resolved binary outcome")
        return cls(probability, int(bool(outcome)), str(item.get("id", "")))


@dataclass(frozen=True)
class HeldOutCalibrationReport:
    sample_count: int
    train_count: int
    held_out_count: int
    train_fraction: float
    seed: int
    bin_count: int
    dataset_sha256: str
    baseline: dict[str, Any]
    calibrated: dict[str, Any]
    calibrator_bins: tuple[dict[str, Any], ...]
    interpretation: str = (
        "The calibrated result is a post-hoc histogram baseline evaluated on held-out "
        "records. It is not OpenJev-RLCD training, does not update model weights, and "
        "does not establish generalization beyond this dataset."
    )


class HistogramCalibrator:
    """Small deterministic calibration baseline with Laplace smoothing."""

    def __init__(self, bin_count: int = 10):
        if not 2 <= bin_count <= 100:
            raise ValueError("bin_count must be between 2 and 100")
        self.bin_count = bin_count
        self._counts = [0] * bin_count
        self._positives = [0] * bin_count
        self._fitted = False

    def _index(self, probability: float) -> int:
        return min(self.bin_count - 1, int(probability * self.bin_count))

    def fit(self, rows: Iterable[DecisionExample]) -> "HistogramCalibrator":
        for row in rows:
            index = self._index(row.probability)
            self._counts[index] += 1
            self._positives[index] += row.outcome
        if not any(self._counts):
            raise ValueError("At least one training example is required")
        self._fitted = True
        return self

    def predict(self, probability: float) -> float:
        if not self._fitted:
            raise RuntimeError("Calibrator must be fitted before prediction")
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("probability must be finite and between zero and one")
        index = self._index(probability)
        count = self._counts[index]
        if count == 0:
            return probability
        # Beta(1, 1) smoothing prevents zero/one estimates from tiny bins.
        return (self._positives[index] + 1) / (count + 2)

    def summary(self) -> tuple[dict[str, Any], ...]:
        result = []
        for index, count in enumerate(self._counts):
            result.append({
                "lower": index / self.bin_count,
                "upper": (index + 1) / self.bin_count,
                "count": count,
                "positive_count": self._positives[index],
                "calibrated_probability": (
                    (self._positives[index] + 1) / (count + 2) if count else None
                ),
            })
        return tuple(result)


def _metric_dict(report: Any) -> dict[str, Any]:
    return {
        "count": report.count,
        "brier_score": report.brier_score,
        "log_loss": report.log_loss,
        "expected_calibration_error": report.expected_calibration_error,
        "aurc": report.aurc,
        "coverage_at_5_percent_risk": report.coverage_at_risk,
    }


def evaluate_held_out(
    records: Iterable[Mapping[str, Any]],
    *,
    train_fraction: float = 0.7,
    seed: int = 42,
    bin_count: int = 10,
) -> HeldOutCalibrationReport:
    """Fit on one deterministic split and report metrics only on unseen rows."""
    if isinstance(train_fraction, bool) or not isinstance(train_fraction, (int, float)) or not 0.5 <= train_fraction <= 0.9:
        raise ValueError("train_fraction must be between 0.5 and 0.9")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    rows = [DecisionExample.from_mapping(item) for item in records]
    if len(rows) < 10:
        raise ValueError("At least 10 resolved decisions are required for a held-out benchmark")
    if len({row.decision_id for row in rows if row.decision_id}) != len(
        [row for row in rows if row.decision_id]
    ):
        raise ValueError("Duplicate decision IDs are not allowed in a benchmark")

    canonical = [
        {"id": row.decision_id, "probability": row.probability, "outcome": row.outcome}
        for row in sorted(rows, key=lambda item: (item.decision_id, item.probability, item.outcome))
    ]
    digest = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    split_at = int(len(indices) * train_fraction)
    split_at = max(1, min(len(indices) - 1, split_at))
    if split_at < 5 or len(indices) - split_at < 5:
        raise ValueError("The split must contain at least five training and five held-out decisions")
    train = [rows[index] for index in indices[:split_at]]
    held_out = [rows[index] for index in indices[split_at:]]

    calibrator = HistogramCalibrator(bin_count).fit(train)
    baseline_metrics = evaluate_binary_decisions([
        {"probability": row.probability, "outcome": row.outcome} for row in held_out
    ])
    calibrated_metrics = evaluate_binary_decisions([
        {"probability": calibrator.predict(row.probability), "outcome": row.outcome}
        for row in held_out
    ])
    return HeldOutCalibrationReport(
        sample_count=len(rows),
        train_count=len(train),
        held_out_count=len(held_out),
        train_fraction=train_fraction,
        seed=seed,
        bin_count=bin_count,
        dataset_sha256=digest,
        baseline=_metric_dict(baseline_metrics),
        calibrated=_metric_dict(calibrated_metrics),
        calibrator_bins=calibrator.summary(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a deterministic held-out calibration benchmark on resolved AETHER decisions."
    )
    parser.add_argument("--database", default="data/aether.sqlite3")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    args = parser.parse_args(argv)

    records = Store(args.database).list_decisions(resolved_only=True, limit=10000)
    domains = {item.get("metadata", {}).get("domain") for item in records}
    domains.discard(None)
    if args.domain:
        domain = args.domain.strip().casefold()
        records = [item for item in records if item.get("metadata", {}).get("domain") == domain]
    elif len(domains) > 1:
        raise ValueError(
            "Resolved decisions span multiple domains; specify --domain to avoid mixing incomparable tasks"
        )
    report = evaluate_held_out(
        records, train_fraction=args.train_fraction, seed=args.seed, bin_count=args.bins
    )
    payload = json.dumps(asdict(report), indent=2)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
