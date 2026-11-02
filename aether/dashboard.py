"""Unified runtime, research, capability and decision dashboard."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aether.config import load_settings
from aether.engine import AetherEngine
from aether.storage import Store


def build_dashboard(engine: Any, store: Any, acceptance_path: str | Path = "reports/acceptance/latest.json") -> dict[str, Any]:
    """Build a read-only snapshot; never triggers research or autonomous actions."""
    status = engine.status()
    identity = engine.self_model.snapshot()
    capabilities = identity.get("capabilities", [])
    decisions = store.list_decisions(resolved_only=False, limit=10000)
    tasks = store.list_tasks(limit=100)
    reports = store.list_research_reports(limit=100)
    capability_states: dict[str, int] = {}
    producer_counts: dict[str, int] = {}
    for item in capabilities:
        state = str(item.get("state", "unknown"))
        capability_states[state] = capability_states.get(state, 0) + 1
    for item in decisions:
        producer = item.get("metadata", {}).get("producer_type", "legacy_or_operator")
        producer_counts[producer] = producer_counts.get(producer, 0) + 1

    acceptance_file = Path(acceptance_path)
    acceptance = None
    if acceptance_file.is_file():
        try:
            acceptance = json.loads(acceptance_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            acceptance = {"status": "unreadable"}

    return {
        "runtime": status,
        "resources": engine.resources.snapshot(),
        "capabilities": {
            "total": len(capabilities),
            "states": capability_states,
            "unverified_ids": [
                item.get("capability_id", item.get("id", "unknown"))
                for item in capabilities
                if item.get("state") in {"implemented_unverified", "partial", "planned", "degraded", "unavailable"}
            ],
        },
        "research": {
            "stored_reports_visible": len(reports),
            "latest_reports": reports[:5],
        },
        "decisions": {
            "total": len(decisions),
            "resolved": sum(item.get("outcome") is not None for item in decisions),
            "unresolved": sum(item.get("outcome") is None for item in decisions),
            "by_producer": producer_counts,
        },
        "tasks": {
            "visible": len(tasks),
            "by_status": {
                state: sum(item.get("status") == state for item in tasks)
                for state in sorted({str(item.get("status", "unknown")) for item in tasks})
            },
        },
        "acceptance": acceptance,
    }


def render_dashboard(snapshot: dict[str, Any]) -> str:
    runtime = snapshot["runtime"]
    resources = snapshot["resources"]
    decisions = snapshot["decisions"]
    capabilities = snapshot["capabilities"]
    research = snapshot["research"]
    tasks = snapshot["tasks"]
    lines = [
        f"AETHER {runtime.get('version', 'unknown')} — SYSTEM DASHBOARD",
        "=" * 58,
        f"Model: {runtime.get('provider')} / {runtime.get('model')} | enabled={runtime.get('model_enabled')} | reachable={runtime.get('model_reachable')}",
        f"Autonomy: enabled={runtime.get('autonomy_enabled')} | emergency_stop={runtime.get('emergency_stop')}",
        f"Capabilities: {capabilities['total']} total | states={capabilities['states']}",
        f"Research reports: {research['stored_reports_visible']} visible",
        f"Decisions: {decisions['resolved']} resolved / {decisions['unresolved']} unresolved | producers={decisions['by_producer']}",
        f"Tasks: {tasks['visible']} visible | statuses={tasks['by_status']}",
        f"Resources: {json.dumps(resources, sort_keys=True)}",
    ]
    acceptance = snapshot.get("acceptance")
    if isinstance(acceptance, dict) and isinstance(acceptance.get("counts"), dict):
        lines.append(f"OMEGA acceptance: {acceptance['counts']} | complete={acceptance.get('complete')}")
    else:
        lines.append("OMEGA acceptance: no readable saved report")
    lines.append("")
    lines.append("Note: capability registration is not proof of verification; inspect /self-model and /acceptance.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Display a read-only AETHER system dashboard.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument("--acceptance", default="reports/acceptance/latest.json")
    args = parser.parse_args(argv)
    settings = load_settings()
    store = Store(settings.storage.database)
    engine = AetherEngine(settings, store)
    snapshot = build_dashboard(engine, store, args.acceptance)
    print(json.dumps(snapshot, indent=2) if args.json else render_dashboard(snapshot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
