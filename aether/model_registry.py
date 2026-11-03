"""Evidence-aware model catalog and capability/privacy-constrained model selection."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from threading import RLock
from typing import Any, Iterable
import json, os


PRIVACY_LEVELS = {"local", "confidential", "public"}


@dataclass
class ModelProfile:
    model_id: str
    provider_name: str
    capabilities: list[str]
    context_window: int
    max_output_tokens: int
    privacy: str
    local: bool
    max_vram_mb: int = 0
    mean_latency_ms: float | None = None
    cost_per_1k_tokens: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    enabled: bool = True
    revision: str | None = None

    @property
    def sample_count(self) -> int:
        return self.success_count + self.failure_count

    @property
    def success_rate(self) -> float | None:
        return self.success_count / self.sample_count if self.sample_count else None

    def validate(self) -> None:
        if not self.model_id.strip() or not self.provider_name.strip():
            raise ValueError("Model and provider identifiers are required")
        if self.privacy not in PRIVACY_LEVELS:
            raise ValueError("Privacy must be local, confidential or public")
        if self.local != (self.privacy == "local"):
            raise ValueError("Local models must have local privacy; remote models cannot claim local privacy")
        if self.context_window < 1 or self.max_output_tokens < 1 or self.max_output_tokens > self.context_window:
            raise ValueError("Model context/output token limits are invalid")
        if self.max_vram_mb < 0 or self.cost_per_1k_tokens < 0:
            raise ValueError("VRAM and cost limits cannot be negative")
        if not isfinite(self.cost_per_1k_tokens):
            raise ValueError("Cost must be finite")
        if self.mean_latency_ms is not None and (
            not isfinite(self.mean_latency_ms) or self.mean_latency_ms < 0
        ):
            raise ValueError("Latency must be finite and non-negative")
        normalized = sorted({str(item).strip().casefold() for item in self.capabilities if str(item).strip()})
        if not normalized:
            raise ValueError("At least one model capability is required")
        self.capabilities = normalized


class ModelCatalog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._profiles = self._load()

    def _load(self) -> dict[str, ModelProfile]:
        if not self.path.exists():
            return {}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema") != "aether-model-catalog-v1":
            raise ValueError("Unsupported model catalog schema")
        profiles = {}
        for raw in payload.get("profiles", []):
            profile = ModelProfile(**raw)
            profile.validate()
            key = f"{profile.provider_name}:{profile.model_id}"
            if key in profiles:
                raise ValueError(f"Duplicate model profile: {key}")
            profiles[key] = profile
        return profiles

    def _save(self) -> None:
        payload = {"schema": "aether-model-catalog-v1",
                   "profiles": [asdict(item) for item in self._profiles.values()]}
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)

    def register(self, profile: ModelProfile) -> str:
        profile.validate()
        key = f"{profile.provider_name}:{profile.model_id}"
        with self._lock:
            self._profiles[key] = profile
            self._save()
        return key

    def list(self, *, enabled_only: bool = False) -> list[dict[str, Any]]:
        with self._lock:
            profiles = [asdict(item) | {"sample_count": item.sample_count,
                         "success_rate": item.success_rate}
                        for item in self._profiles.values()
                        if item.enabled or not enabled_only]
        return sorted(profiles, key=lambda item: (item["provider_name"], item["model_id"]))

    def record_outcome(self, provider_name: str, model_id: str, *,
                       success: bool, latency_ms: float) -> dict[str, Any]:
        if (not isinstance(success, bool) or isinstance(latency_ms, bool)
                or not isinstance(latency_ms, (int, float))
                or not isfinite(latency_ms) or latency_ms < 0):
            raise ValueError("Outcome and latency are invalid")
        key = f"{provider_name}:{model_id}"
        with self._lock:
            profile = self._profiles.get(key)
            if profile is None:
                raise KeyError(key)
            old_count = profile.sample_count
            profile.success_count += int(success)
            profile.failure_count += int(not success)
            profile.mean_latency_ms = (
                latency_ms if profile.mean_latency_ms is None
                else (profile.mean_latency_ms * old_count + latency_ms) / (old_count + 1)
            )
            self._save()
            return asdict(profile) | {"sample_count": profile.sample_count,
                                      "success_rate": profile.success_rate}

    def select(
        self, required_capabilities: Iterable[str], *,
        privacy: str = "confidential", min_context_window: int = 1,
        max_latency_ms: float | None = None, max_cost_per_1k: float | None = None,
        max_vram_mb: int | None = None, min_samples: int = 20,
    ) -> dict[str, Any]:
        required = {str(item).strip().casefold() for item in required_capabilities if str(item).strip()}
        if privacy not in PRIVACY_LEVELS:
            raise ValueError("Unknown privacy requirement")
        if min_context_window < 1 or min_samples < 0:
            raise ValueError("Context and sample requirements are invalid")
        rejected: list[dict[str, str]] = []
        eligible = []
        for profile in self._profiles.values():
            key = f"{profile.provider_name}:{profile.model_id}"
            reason = None
            if not profile.enabled: reason = "disabled"
            elif not required.issubset(set(profile.capabilities)): reason = "missing_capability"
            elif privacy == "local" and not profile.local: reason = "remote_model_for_local_only_task"
            elif privacy == "confidential" and profile.privacy == "public": reason = "public_privacy_profile"
            elif profile.context_window < min_context_window: reason = "context_window_too_small"
            elif max_latency_ms is not None and (profile.mean_latency_ms is None or profile.mean_latency_ms > max_latency_ms): reason = "latency_limit"
            elif max_cost_per_1k is not None and profile.cost_per_1k_tokens > max_cost_per_1k: reason = "cost_limit"
            elif max_vram_mb is not None and profile.max_vram_mb > max_vram_mb: reason = "vram_limit"
            if reason:
                rejected.append({"profile": key, "reason": reason})
                continue
            rate = profile.success_rate
            evidence = "measured" if profile.sample_count >= min_samples else "insufficient_observations"
            reliability = rate if evidence == "measured" and rate is not None else 0.5
            latency = profile.mean_latency_ms if profile.mean_latency_ms is not None else 10000.0
            score = reliability * 100 - latency / 1000 - profile.cost_per_1k_tokens
            eligible.append((score, profile, evidence))
        if not eligible:
            return {"selected": None, "reason": "No model satisfies the requested capability/privacy/resource constraints.",
                    "rejected": rejected}
        eligible.sort(key=lambda item: (item[0], item[1].sample_count), reverse=True)
        score, chosen, evidence = eligible[0]
        return {"selected": asdict(chosen), "score": score, "evidence_status": evidence,
                "sample_count": chosen.sample_count,
                "warning": "Unmeasured profiles use a neutral ranking prior; model self-reported confidence is never used.",
                "rejected": rejected}
