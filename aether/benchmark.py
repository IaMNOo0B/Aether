"""Deterministic end-to-end benchmark for AETHER's bounded autonomy executor.

The fixture provider supplies reference plans. This measures orchestration, tool
execution and postcondition verification; it does not measure model planning quality.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any

from aether.autonomy import AutonomyRunner
from aether.config import AutonomySettings, LimitSettings, ResearchSettings
from aether.storage import Store


class _FixtureProvider:
    def __init__(self, plan: dict[str, Any]):
        self.plan = plan

    def chat(self, messages: list[dict[str, str]]) -> str:
        return json.dumps(self.plan)


def run_phase1_benchmark(
    cases: list[dict[str, Any]], workspace_root: str | Path
) -> dict[str, Any]:
    """Run independent fixed-plan scenarios and report objective-level checks."""
    root = Path(workspace_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for case in cases:
        case_id = str(case["id"]).strip()
        if not case_id or case_id in seen_ids:
            raise ValueError("Benchmark case IDs must be non-empty and unique")
        seen_ids.add(case_id)
        project_root = (root / case_id).resolve()
        if root not in project_root.parents:
            raise ValueError(f"Benchmark case ID escapes benchmark root: {case_id}")
        workspace = project_root / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        for relative, content in case.get("seed_files", {}).items():
            target = (workspace / relative).resolve()
            if workspace not in target.parents:
                raise ValueError(f"Benchmark seed path escapes workspace: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(content), encoding="utf-8")

        runner = AutonomyRunner(
            _FixtureProvider({"steps": case["steps"]}),
            Store(project_root / "benchmark.sqlite3"),
            AutonomySettings(
                enabled=True, allow_read_only=True, allow_workspace_writes=True,
                allow_code_execution=False, workspace="workspace",
            ),
            LimitSettings(), ResearchSettings(), project_root,
        )
        result = runner.run(
            str(case["objective"]),
            approval=lambda tool, purpose, args: True,
        )
        actual_tools = [receipt["tool"] for receipt in result.get("receipts", [])]
        expected_tools = list(case["expected_tools"])
        complete = (
            result.get("status") == "completed"
            and result.get("tool_results_verified") is True
            and actual_tools == expected_tools
            and len(result.get("completed_steps", [])) == len(case["steps"])
        )
        results.append({
            "id": case_id,
            "passed": complete,
            "status": result.get("status"),
            "expected_tools": expected_tools,
            "actual_tools": actual_tools,
            "tool_results_verified": result.get("tool_results_verified", False),
        })

    total = len(results)
    passed = sum(1 for item in results if item["passed"])
    return {
        "benchmark": "phase1_bounded_autonomy",
        "scope": "fixed reference plans; executor and postcondition checks only",
        "cases": results,
        "metrics": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "completion_rate": passed / total if total else 0.0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run AETHER Phase 1 autonomy benchmark")
    parser.add_argument(
        "--cases", type=Path,
        default=Path(__file__).resolve().parent.parent / "benchmarks" / "phase1_cases.json",
    )
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="aether-phase1-benchmark-") as directory:
        report = run_phase1_benchmark(cases, directory)
    print(json.dumps(report, indent=2))
    return 0 if report["metrics"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
