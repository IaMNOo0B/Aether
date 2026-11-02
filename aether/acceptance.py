"""Run the 12 OMEGA acceptance checks and record honest scope status."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import sys
from typing import Any, Callable


ACCEPTANCE_CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "A", "name": "Continuity",
        "expected": "Identity, relevant memory and unfinished tasks survive restart and recover safely.",
        "nodes": [
            "tests/test_self_model.py::test_identity_and_history_survive_restart",
            "tests/test_embeddings.py::test_hybrid_memory_finds_semantically_related_text_without_shared_terms",
            "tests/test_task_recovery.py::test_interrupted_task_is_marked_failed_without_automatic_replay",
        ],
        "full_scope": True,
        "limitations": [],
    },
    {
        "id": "B", "name": "Research",
        "expected": "A research report preserves source provenance and separates extracted claims from verified facts.",
        "nodes": [
            "tests/test_research.py::test_crossref_results_are_structured_with_provenance",
            "tests/test_research.py::test_research_report_is_persisted_with_source_records",
            "tests/test_knowledge.py::test_report_extraction_uses_only_real_source_provenance",
            "tests/test_research_quality.py::test_review_protocol_persists_all_sources_as_pending",
            "tests/test_research_quality.py::test_experiment_lifecycle_is_audited_and_terminal",
            "tests/test_research_advanced.py::test_general_web_search_parser_extracts_result_and_ignores_invalid_links",
            "tests/test_research_advanced.py::test_general_web_search_returns_structured_sources_without_fetching_results",
            "tests/test_research_advanced.py::test_combined_research_search_interleaves_web_and_scholarly_sources",
            "tests/test_research_advanced.py::test_snippet_only_quote_cannot_be_adjudicated_as_support",
            "tests/test_research_advanced.py::test_full_systematic_review_requires_two_reviewers_and_runs_meta_analysis",
            "tests/test_research_advanced.py::test_systematic_review_disagreement_requires_distinct_adjudicator",
            "tests/test_research_advanced.py::test_citation_disagreement_is_not_silently_resolved",
            "tests/test_research_advanced.py::test_grouped_experiment_analysis_is_reproducible",
        ],
        "full_scope": False,
        "limitations": [
            "Reviewer IDs are operator-supplied and not authenticated; external web search results are not independently ranked for truth.",
            "Citation checking uses exact quote matching and lexical overlap, not semantic entailment.",
            "Risk-of-bias domains are generic; statistical synthesis assumes comparable independent effects and supplied standard errors.",
            "Publication-bias analysis, certainty-of-evidence grading and PRISMA compliance are not claimed.",
        ],
    },
    {
        "id": "C", "name": "Learning",
        "expected": "A documented workflow is stored as a draft and is only retrievable after held-out evaluation.",
        "nodes": [
            "tests/test_learning.py::test_skill_requires_held_out_evaluation_before_retrieval",
            "tests/test_learning.py::test_failed_skill_evaluation_keeps_skill_out_of_validated_memory",
        ],
        "full_scope": False,
        "limitations": [
            "Skill evaluation evidence is manually supplied; AETHER does not yet independently execute and score arbitrary workflows on held-out examples."
        ],
    },
    {
        "id": "D", "name": "Planning",
        "expected": "A multi-step task validates dependency graphs, executes in topological order, persists verified receipts, and applies only bounded safe replanning.",
        "nodes": [
            "tests/test_autonomy.py::test_autonomy_executes_approved_workspace_tools_and_persists_receipts",
            "tests/test_autonomy.py::test_autonomy_declined_write_is_cancelled_without_side_effect",
            "tests/test_autonomy.py::test_autonomy_sorts_dependency_graph_and_verifies_each_result",
            "tests/test_autonomy.py::test_autonomy_rejects_cyclic_plan_before_running_tools",
            "tests/test_autonomy.py::test_autonomy_replans_once_after_safe_missing_workspace_read",
            "tests/test_task_recovery.py::test_checkpoint_only_allowed_while_running",
        ],
        "full_scope": False,
        "limitations": [
            "Automatic replanning is intentionally limited to one missing workspace read/list target; arbitrary failures are not automatically replanned."
        ],
    },
    {
        "id": "E", "name": "Calibration",
        "expected": "Predictions are resolved against outcomes and calibration metrics/gates use domain-specific evidence.",
        "nodes": [
            "tests/test_decisions.py::test_decision_log_and_resolution_preserve_outcome_provenance",
            "tests/test_decisions.py::test_decision_metrics_require_resolved_outcomes",
            "tests/test_decisions.py::test_decision_gate_abstains_without_sufficient_domain_evidence",
        ],
        "full_scope": True,
        "limitations": [],
    },
    {
        "id": "F", "name": "Software engineering",
        "expected": "A controlled defect is reproduced, fixed and verified with tests.",
        "nodes": [
            "tests/test_self_engineering.py::test_self_engineer_rejects_security_policy_changes",
            "tests/test_self_engineering.py::test_self_engineer_rejects_path_traversal",
            "tests/test_self_engineering.py::test_self_engineer_rejects_symlink_or_file_mode_changes",
        ],
        "full_scope": False,
        "limitations": [
            "Patch validation is tested, but a complete benchmarked defect-reproduction-and-fix workflow has not yet passed as an end-to-end acceptance test."
        ],
    },
    {
        "id": "G", "name": "Self-improvement",
        "expected": "An improvement is proposed, tested in isolation and compared against the baseline.",
        "nodes": [
            "tests/test_self_engineering.py::test_self_engineer_requires_approval_before_running_generated_code",
        ],
        "full_scope": False,
        "limitations": [
            "Candidate tests run in a copied working tree, not a secure OS sandbox; automatic performance comparison and integration are not implemented."
        ],
    },
    {
        "id": "H", "name": "Autonomy",
        "expected": "AETHER discovers auditable local opportunities, ranks goal proposals, requires explicit approval, and runs approved low-risk work in the background with pause, resume, cancellation, resource checks and receipts.",
        "nodes": [
            "tests/test_goals.py::test_goal_discovery_creates_proposals_but_never_approves_them",
            "tests/test_goals.py::test_dependencies_block_auto_queue_until_parent_completes",
            "tests/test_goals.py::test_scheduler_runs_only_approved_low_risk_goals_and_records_receipts",
            "tests/test_goals.py::test_scheduler_discovers_but_does_not_auto_approve_goals",
            "tests/test_goals.py::test_task_cancellation_is_cooperative_and_stops_at_step_boundary",
            "tests/test_scheduler.py::test_scheduler_runs_persisted_goal_with_read_only_tools",
            "tests/test_scheduler.py::test_scheduler_never_approves_workspace_writes",
            "tests/test_emergency_stop.py::test_scheduler_does_not_enqueue_when_emergency_stop_is_active",
        ],
        "full_scope": False,
        "limitations": [
            "Discovery currently uses local failed-task and self-model capability-gap records; external calendars, issue trackers, filesystem watchers and performance-monitor signals are not integrated.",
            "Background execution is restricted to explicitly approved low-risk goals and read-only tools. Higher-risk goals require an operator-initiated run with per-action approval.",
            "Resource budgets are preflight guards, not hard operating-system quotas."
        ],
    },
    {
        "id": "I", "name": "Security",
        "expected": "Untrusted content cannot add tools, grant permissions or bypass the emergency stop.",
        "nodes": [
            "tests/test_autonomy.py::test_autonomy_rejects_unknown_model_generated_tools",
            "tests/test_knowledge.py::test_extractor_prompt_marks_research_summaries_as_untrusted",
            "tests/test_emergency_stop.py::test_emergency_stop_while_waiting_for_approval_blocks_write",
        ],
        "full_scope": False,
        "limitations": [
            "The bounded tool registry and prompt-injection checks are tested, but a complete adversarial red-team suite and OS-level sandbox are not implemented."
        ],
    },
    {
        "id": "J", "name": "Resource management",
        "expected": "Configured prompt, step and tool-time limits are respected.",
        "nodes": [
            "tests/test_config.py::test_load_settings_from_project_config",
            "tests/test_scheduler.py::test_disabled_scheduler_does_not_enqueue_or_execute",
            "tests/test_resources.py::test_resource_guard_blocks_cpu_over_budget",
            "tests/test_resources.py::test_resource_guard_blocks_ram_pressure_and_low_free_memory",
            "tests/test_resources.py::test_resource_guard_checks_vram_only_for_gpu_required_operations",
            "tests/test_resources.py::test_windows_resource_monitor_reads_system_cpu_and_ram",
        ],
        "full_scope": False,
        "limitations": [
            "CPU, RAM and NVIDIA VRAM telemetry and configurable preflight budgets now exist. Hard OS quotas, per-process CPU/RAM enforcement, guaranteed VRAM reservation, model-call accounting and dynamic concurrency scheduling remain unimplemented."
        ],
    },
    {
        "id": "K", "name": "Multi-agent evaluation",
        "expected": "Specialist roles produce separate deliverables and an integrated result is independently checked.",
        "nodes": [
            "tests/test_agents.py::test_specialist_roles_run_and_persist_separate_deliverables",
        ],
        "full_scope": False,
        "limitations": [
            "Roles currently run sequentially through the same configured model; independent specialist models and an independent verifier are not implemented."
        ],
    },
    {
        "id": "L", "name": "Recovery",
        "expected": "Interrupted work is marked safely and consequential actions are not blindly replayed.",
        "nodes": [
            "tests/test_task_recovery.py::test_task_lifecycle_writes_ordered_audit_events",
            "tests/test_task_recovery.py::test_interrupted_task_is_marked_failed_without_automatic_replay",
            "tests/test_emergency_stop.py::test_emergency_stop_between_steps_stops_remaining_tools",
        ],
        "full_scope": True,
        "limitations": [],
    },
)


def _summary(output: str) -> str:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return lines[-1][:1200] if lines else "No test output was produced."


def run_acceptance_suite(
    root: str | Path = ".", timeout_seconds: int = 120,
    output_path: str | Path | None = None,
    runner: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    project_root = Path(root).resolve()
    results = []
    for case in ACCEPTANCE_CASES:
        command = [sys.executable, "-m", "pytest", "-q", *case["nodes"]]
        try:
            completed = runner(
                command, cwd=project_root, capture_output=True,
                text=True, timeout=timeout_seconds, check=False,
            )
            output = (completed.stdout or "") + "\n" + (completed.stderr or "")
            if completed.returncode == 0:
                execution = "PASS"
                acceptance = "PASS" if case["full_scope"] else "PARTIAL"
            elif "No module named pytest" in output:
                execution = "BLOCKED"
                acceptance = "BLOCKED"
            else:
                execution = "FAIL"
                acceptance = "FAIL"
            actual = _summary(output)
        except subprocess.TimeoutExpired:
            execution = "FAIL"
            acceptance = "FAIL"
            actual = f"Timed out after {timeout_seconds} seconds."
        except FileNotFoundError as exc:
            execution = "BLOCKED"
            acceptance = "BLOCKED"
            actual = str(exc)

        results.append({
            "id": case["id"],
            "name": case["name"],
            "expected": case["expected"],
            "actual": actual,
            "test_execution": execution,
            "acceptance_status": acceptance,
            "limitations": case["limitations"],
            "test_nodes": case["nodes"],
        })

    report = {
        "suite": "PROJECT AETHER OMEGA - 12 acceptance tests",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "counts": {
            status: sum(item["acceptance_status"] == status for item in results)
            for status in ("PASS", "PARTIAL", "FAIL", "BLOCKED")
        },
        "complete": all(item["acceptance_status"] == "PASS" for item in results),
        "results": results,
    }
    if output_path is not None:
        destination = Path(output_path)
        if not destination.is_absolute():
            destination = project_root / destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["output_path"] = str(destination)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run AETHER's 12 OMEGA acceptance checks.")
    parser.add_argument("--root", default=".", help="AETHER project root")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout per acceptance group")
    parser.add_argument("--json", dest="output_path", help="Optional JSON report output path")
    args = parser.parse_args(argv)
    report = run_acceptance_suite(args.root, args.timeout, args.output_path)
    print(f"{report['suite']}")
    for item in report["results"]:
        print(
            f"{item['id']} {item['name']:<24} "
            f"{item['test_execution']:<8} {item['acceptance_status']}"
        )
        if item["limitations"]:
            for limitation in item["limitations"]:
                print(f"    GAP: {limitation}")
    print("Counts:", json.dumps(report["counts"]))
    print("Complete:", report["complete"])
    if report.get("output_path"):
        print("Report:", report["output_path"])
    if report["counts"]["FAIL"] or report["counts"]["BLOCKED"]:
        return 1
    if report["counts"]["PARTIAL"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
