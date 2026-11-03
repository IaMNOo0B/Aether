"""Reproducible held-out calibration benchmark with a train-only baseline.

This evaluates predictions against observed labels. It does not train an RLCD
policy or treat model-reported confidence as calibrated probability.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from typing import Iterable

from aether.calibration import evaluate_binary_decisions


SPLITS = {"train", "validation", "test"}


@dataclass(frozen=True)
class CalibrationCase:
    case_id: str
    split: str
    domain: str
    probability: float
    outcome: bool
    source: str

    def __post_init__(self) -> None:
        if not self.case_id.strip() or not self.domain.strip() or not self.source.strip():
            raise ValueError("Calibration case ID, domain and source are required")
        if self.split not in SPLITS:
            raise ValueError(f"split must be one of {sorted(SPLITS)}")
        if isinstance(self.probability, bool) or not isinstance(self.probability, (int, float)):
            raise ValueError("probability must be numeric")
        if not math.isfinite(float(self.probability)) or not 0 <= self.probability <= 1:
            raise ValueError("probability must be finite and between zero and one")
        if not isinstance(self.outcome, bool):
            raise ValueError("outcome must be a boolean observed label")


class CalibrationBenchmark:
    def run(
        self,
        cases: Iterable[CalibrationCase],
        *,
        model_id: str,
        dataset_id: str,
        seed: int,
        bin_count: int = 10,
    ) -> dict:
        if not model_id.strip() or not dataset_id.strip():
            raise ValueError("model_id and dataset_id are required for reproducibility")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        items = list(cases)
        if not items:
            raise ValueError("At least one calibration case is required")
        ids = [item.case_id for item in items]
        if len(ids) != len(set(ids)):
            raise ValueError("Calibration case IDs must be unique")
        split_counts = {
            split: sum(item.split == split for item in items)
            for split in ("train", "validation", "test")
        }
        if (set(item.split for item in items) != SPLITS
                or split_counts["train"] < 3
                or split_counts["validation"] < 2
                or split_counts["test"] < 4):
            raise ValueError(
                "Benchmark requires train, validation and test splits with at least "
                "3, 2 and 4 cases respectively"
            )
        train = [item for item in items if item.split == "train"]
        test = [item for item in items if item.split == "test"]
        train_rate = sum(item.outcome for item in train) / len(train)
        records = [{"probability": item.probability, "outcome": item.outcome} for item in test]
        baseline = [{"probability": train_rate, "outcome": item.outcome} for item in test]
        model_metrics = evaluate_binary_decisions(records, bin_count=bin_count)
        baseline_metrics = evaluate_binary_decisions(baseline, bin_count=bin_count)
        canonical = json.dumps(
            [asdict(item) for item in items], sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        data_hash = sha256(canonical).hexdigest()
        return {
            "benchmark": "held-out-calibration-v1",
            "model_id": model_id,
            "dataset_id": dataset_id,
            "seed": seed,
            "input_sha256": data_hash,
            "split_counts": split_counts,
            "train_base_rate": train_rate,
            "test_metrics": asdict(model_metrics),
            "train_base_rate_baseline": asdict(baseline_metrics),
            "brier_improvement_vs_train_base_rate": (
                baseline_metrics.brier_score - model_metrics.brier_score
            ),
            "interpretation": (
                "Only test-split outcomes are scored. The constant baseline probability is "
                "estimated from training labels only. Validation labels are reserved and not "
                "used by this evaluator. Results apply only to this dataset, split and model "
                "version; they do not establish universal calibration or reproduce RLCD training."
            ),
        }
