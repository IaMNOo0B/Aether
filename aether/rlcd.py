"""Reproducibility contract for optional external RLCD training.

This module validates run manifests; it does not implement the OpenJev training
algorithm and cannot make an API-only model differentiable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
import json
import re
import tempfile


_HASH = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class RLCDRunManifest:
    method: str
    method_version: str
    source_commit: str
    base_model: str
    trained_model_sha256: str
    training_dataset_sha256: str
    validation_dataset_sha256: str
    test_dataset_sha256: str
    seeds: tuple[int, ...]
    hardware: str
    baseline_metrics: dict[str, float]
    held_out_metrics: dict[str, float]
    training_config: dict[str, Any]
    status: str = "completed"

    def validate(self) -> None:
        required = {
            "method": self.method,
            "method_version": self.method_version,
            "source_commit": self.source_commit,
            "base_model": self.base_model,
            "hardware": self.hardware,
        }
        if any(not isinstance(value, str) or not value.strip() for value in required.values()):
            raise ValueError("RLCD manifest is missing required run identity fields")
        if self.method != "OpenJev-RLCD":
            raise ValueError("Unsupported RLCD methodology; do not substitute generic RL")
        for name in (
            "trained_model_sha256", "training_dataset_sha256",
            "validation_dataset_sha256", "test_dataset_sha256",
        ):
            if not _HASH.fullmatch(getattr(self, name)):
                raise ValueError(f"{name} must be a 64-character SHA-256 digest")
        if len({self.training_dataset_sha256, self.validation_dataset_sha256, self.test_dataset_sha256}) != 3:
            raise ValueError("Training, validation and test datasets must have distinct hashes")
        if not self.seeds or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in self.seeds):
            raise ValueError("At least one non-negative integer training seed is required")
        if self.status != "completed":
            raise ValueError("Only completed RLCD runs can be registered as reproducible")
        for label, metrics in (("baseline_metrics", self.baseline_metrics), ("held_out_metrics", self.held_out_metrics)):
            if not metrics or any(
                not isinstance(key, str) or isinstance(value, bool)
                or not isinstance(value, (int, float))
                for key, value in metrics.items()
            ):
                raise ValueError(f"{label} must contain numeric evaluation metrics")
        if not isinstance(self.training_config, dict) or not self.training_config:
            raise ValueError("training_config must record the actual training parameters")

    def fingerprint(self) -> str:
        self.validate()
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return sha256(payload.encode("utf-8")).hexdigest()


def save_manifest(manifest: RLCDRunManifest, destination: str | Path) -> dict[str, str]:
    """Validate and atomically save a completed run manifest."""
    fingerprint = manifest.fingerprint()
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**asdict(manifest), "manifest_sha256": fingerprint}
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=path.name + ".",
        suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        import os
        os.fsync(handle.fileno())
    temporary.replace(path)
    return {"path": str(path), "manifest_sha256": fingerprint}
