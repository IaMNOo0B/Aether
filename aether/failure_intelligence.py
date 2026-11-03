"""Rule-based failure classification and safe recovery guidance.

The analyzer is deliberately conservative: it recommends recovery but never retries
an operation or replays side effects automatically.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4
import time

from aether.audit import AuditJournal


@dataclass(frozen=True)
class FailureReport:
    failure_id: str
    category: str
    severity: str
    operation: str
    message: str
    idempotent: bool
    retryable: bool
    automatic_retry_allowed: bool
    recovery_action: str
    human_intervention_required: bool
    classification_basis: str


class FailureAnalyzer:
    RULES = (
        ("permission", ("permissionerror", "not authorized", "approval required", "access denied"),
         "high", "Do not retry. Request the correct owner approval or permission."),
        ("authentication", ("unauthorized", "invalid api key", "authentication", "401"),
         "high", "Check the configured credential source; never print or copy the secret into logs."),
        ("rate_limit", ("429", "rate limit", "too many requests"),
         "medium", "Wait for the provider's retry window, then retry only if the operation is idempotent."),
        ("timeout", ("timeout", "timed out", "deadline exceeded"),
         "medium", "Check whether the operation completed before retrying; reconcile side effects first."),
        ("resource", ("out of memory", "cuda oom", "resource limit", "memoryerror", "disk full"),
         "high", "Reduce batch size, context or concurrency; verify available CPU, RAM, VRAM and disk."),
        ("dependency", ("modulenotfounderror", "no module named", "executable not found", "not installed"),
         "medium", "Verify the optional dependency or executable version; do not install packages silently."),
        ("provider_unavailable", ("connection refused", "connection reset", "service unavailable", "502", "503", "504"),
         "medium", "Check provider health and use an explicitly configured fallback if safe."),
        ("validation", ("valueerror", "invalid", "schema", "malformed", "validation"),
         "low", "Correct the input or schema; repeating the same request will not fix it."),
        ("storage_integrity", ("database is locked", "database disk image is malformed", "integrity check"),
         "critical", "Stop writes, preserve the database and verify backups before any recovery attempt."),
    )

    def __init__(self, journal_path: str | Path):
        self.journal = AuditJournal(journal_path)

    def analyze(
        self, error: BaseException | str, *, operation: str,
        idempotent: bool = False, side_effect_state: str = "unknown",
    ) -> FailureReport:
        message = str(error)
        error_type = type(error).__name__.casefold() if isinstance(error, BaseException) else "external"
        lowered = (error_type + " " + message).casefold()
        category, severity, recovery, basis = (
            "unknown", "high",
            "Capture reproduction steps, inspect logs and receipts, and isolate the smallest failing component.",
            "No deterministic rule matched; no retry is recommended.",
        )
        for name, patterns, level, advice in self.RULES:
            matched = [pattern for pattern in patterns if pattern in lowered]
            if matched:
                category, severity, recovery = name, level, advice
                basis = "Matched rule tokens: " + ", ".join(matched)
                break
        if category == "timeout":
            retryable = idempotent and side_effect_state in {"none", "not_started", "confirmed_not_applied"}
        elif category in {"rate_limit", "provider_unavailable"}:
            retryable = idempotent and side_effect_state in {"none", "not_started", "confirmed_not_applied"}
        else:
            retryable = False
        human_required = (
            category in {"permission", "authentication", "storage_integrity", "unknown"}
            or side_effect_state == "unknown"
        )
        return FailureReport(
            uuid4().hex, category, severity, operation[:120], message[:2000],
            bool(idempotent), retryable, False, recovery, human_required, basis,
        )

    def record(self, report: FailureReport) -> dict[str, Any]:
        return self.journal.append("failure.classified", asdict(report))

    def classify_and_record(
        self, error: BaseException | str, *, operation: str,
        idempotent: bool = False, side_effect_state: str = "unknown",
    ) -> FailureReport:
        report = self.analyze(error, operation=operation, idempotent=idempotent,
                              side_effect_state=side_effect_state)
        self.record(report)
        return report

    def trends(self) -> dict[str, Any]:
        if not self.journal.verify()["valid"]:
            raise RuntimeError("Failure journal integrity verification failed")
        counts: Counter[str] = Counter()
        if self.journal.path.exists():
            import json
            with self.journal.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    entry = json.loads(line)
                    if entry["event_type"] == "failure.classified":
                        counts[entry["payload"]["category"]] += 1
        return {"total": sum(counts.values()), "by_category": dict(sorted(counts.items())),
                "automatic_retries": 0,
                "note": "Classification trends are deterministic rule counts, not predictive failure probabilities."}
