"""Bounded autonomous task execution through a permission-enforcing tool gateway."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
from uuid import uuid4
import json
import subprocess
import sys

from aether.config import AutonomySettings, LimitSettings, ResearchSettings
from aether.providers import ChatProvider
from aether.planning import PlanGraph, PlanStep
from aether.research import Researcher
from aether.storage import Store
from aether.world_model import WorldModel
from aether.resources import ResourceGuard, ResourceLimitExceeded
from aether.tasks import Task, TaskManager


class ToolDenied(PermissionError):
    pass


class ActionDeclined(ToolDenied):
    pass


class EmergencyStop(ToolDenied):
    pass


ApprovalCallback = Callable[[str, str, dict[str, Any]], bool]


class ToolGateway:
    """No arbitrary shell, Python eval, network fetch, or unconstrained filesystem tools."""

    def __init__(
        self,
        project_root: str | Path,
        policy: AutonomySettings,
        limits: LimitSettings,
        researcher: Researcher, research_enabled: bool = True,
        emergency_stop_check: Callable[[], bool] | None = None,
        world_model: WorldModel | None = None,
    ):
        self.project_root = Path(project_root).resolve()
        self.workspace = (self.project_root / policy.workspace).resolve()
        self.policy = policy
        self.limits = limits
        self.researcher = researcher
        self.research_enabled = research_enabled
        self.emergency_stop_check = emergency_stop_check
        self.world_model = world_model
        self.workspace.mkdir(parents=True, exist_ok=True)

    def _workspace_path(self, relative: str) -> Path:
        candidate = (self.workspace / relative).resolve()
        if candidate != self.workspace and self.workspace not in candidate.parents:
            raise ToolDenied("Path escapes the configured AETHER workspace")
        return candidate

    def _authorize(
        self, tool: str, args: dict[str, Any], approval: ApprovalCallback | None
    ) -> None:
        if self.emergency_stop_check is not None and self.emergency_stop_check():
            raise EmergencyStop("AETHER emergency stop is active; tool execution is blocked")
        if not self.policy.enabled:
            raise ToolDenied("Autonomous execution is disabled in config/settings.toml")
        if tool == "research.search" and not self.research_enabled:
            raise ToolDenied("Research is disabled in config/settings.toml")
        if tool in {
            "research.search", "workspace.list", "workspace.read",
            "world.state", "world.actions", "world.simulate", "world.compare",
        }:
            if not self.policy.allow_read_only:
                raise ToolDenied("Read-only autonomous tools are disabled")
            return
        if tool == "workspace.write":
            if not self.policy.allow_workspace_writes:
                raise ToolDenied("Workspace writes are disabled")
        elif tool == "tests.run":
            if not self.policy.allow_code_execution:
                raise ToolDenied("Code execution is disabled")
        else:
            raise ToolDenied(f"Tool is not in the fixed tool registry: {tool}")
        if approval is None:
            raise ActionDeclined(f"Human approval declined or unavailable for {tool}")
        approved = approval(
            tool, str(args.get("purpose", "AETHER requested this operation")), args
        )
        if self.emergency_stop_check is not None and self.emergency_stop_check():
            raise EmergencyStop("AETHER emergency stop activated while awaiting approval")
        if not approved:
            raise ActionDeclined(f"Human approval declined or unavailable for {tool}")

    def execute(
        self, tool: str, args: dict[str, Any],
        approval: ApprovalCallback | None = None,
    ) -> dict[str, Any]:
        if not isinstance(args, dict):
            raise ValueError("Tool arguments must be a JSON object")
        self._authorize(tool, args, approval)

        if tool.startswith("world."):
            if self.world_model is None:
                raise ToolDenied("World-model tools are not configured")
            world_name = str(args.get("world", "")).strip()
            if tool == "world.state":
                world = self.world_model.get_world(world_name)
                if world is None:
                    raise KeyError(f"Unknown world: {world_name}")
                return world
            if tool == "world.actions":
                return {"world": world_name, "actions": self.world_model.list_actions(world_name)}
            if tool == "world.simulate":
                action_id = str(args.get("action_id", "")).strip()
                return self.world_model.simulate(world_name, action_id)
            if tool == "world.compare":
                action_ids = args.get("action_ids")
                if not isinstance(action_ids, list) or not all(
                    isinstance(item, str) for item in action_ids
                ):
                    raise ValueError("world.compare requires an array of action IDs")
                return {"world": world_name, "simulations": self.world_model.compare_actions(world_name, action_ids)}

        if tool == "research.search":
            query = str(args.get("query", "")).strip()
            requested = int(args.get("limit", 5))
            results = self.researcher.search(
                query, limit=max(1, min(requested, self.limits.max_research_results))
            )
            report = self.researcher.create_report(query, results)
            return {
                "query": query, "report_id": report["id"],
                "sources": report["sources"], "report": report["content"][:12000],
            }

        if tool == "workspace.list":
            relative = str(args.get("path", "."))
            directory = self._workspace_path(relative)
            if not directory.exists() or not directory.is_dir():
                raise FileNotFoundError(f"Workspace directory not found: {relative}")
            entries = []
            for item in sorted(directory.iterdir(), key=lambda path: path.name.casefold())[:200]:
                entries.append({
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None,
                })
            return {"path": relative, "entries": entries}

        if tool == "workspace.read":
            relative = str(args.get("path", ""))
            path = self._workspace_path(relative)
            if not path.is_file():
                raise FileNotFoundError(f"Workspace file not found: {relative}")
            if path.stat().st_size > 100_000:
                raise ToolDenied("Workspace reads are limited to 100 KB per file")
            return {"path": relative, "content": path.read_text(encoding="utf-8")}

        if tool == "workspace.write":
            relative = str(args.get("path", ""))
            content = args.get("content")
            if not relative or not isinstance(content, str) or len(content) > 50_000:
                raise ValueError("Workspace write requires a path and <= 50 KB text content")
            path = self._workspace_path(relative)
            if path.exists():
                raise FileExistsError("Autonomous workspace writes never overwrite existing files")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return {"path": str(path.relative_to(self.project_root)), "bytes": path.stat().st_size}

        if tool == "tests.run":
            command = [sys.executable, "-m", "pytest", "-q"]
            try:
                result = subprocess.run(
                    command, cwd=self.project_root, capture_output=True, text=True,
                    timeout=self.limits.tool_timeout_seconds, check=False, shell=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError("Test suite exceeded the configured execution timeout") from exc
            return {
                "command": command, "return_code": result.returncode,
                "passed": result.returncode == 0,
                "stdout": result.stdout[-12000:], "stderr": result.stderr[-6000:],
            }

        raise ToolDenied(f"Tool is not available: {tool}")


class AutonomyRunner:
    def __init__(
        self,
        provider: ChatProvider,
        store: Store,
        policy: AutonomySettings,
        limits: LimitSettings,
        research_settings: ResearchSettings,
        project_root: str | Path = ".",
        resource_guard: ResourceGuard | None = None,
        gpu_required: bool = False,
    ):
        self.provider = provider
        self.store = store
        self.policy = policy
        self.limits = limits
        self.resource_guard = resource_guard
        self.gpu_required = gpu_required
        self.tasks = TaskManager(store)
        self.researcher = Researcher(
            store, research_settings.timeout_seconds, limits.max_research_results
        )
        self.gateway = ToolGateway(
            project_root, policy, limits, self.researcher, research_settings.enabled,
            emergency_stop_check=store.emergency_stop_enabled,
            world_model=WorldModel(store),
        )

    def enqueue(self, objective: str) -> Task:
        if self.store.emergency_stop_enabled():
            raise EmergencyStop("AETHER emergency stop is active; autonomy is blocked")
        if not objective.strip():
            raise ValueError("Goal cannot be empty")
        return self.tasks.create(objective, {"kind": "autonomy", "receipts": []})

    def _make_plan(
        self, objective: str, replanning_context: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        if self.resource_guard is not None:
            self.resource_guard.check("autonomy planning", gpu_required=self.gpu_required)
        allowed = {
            "research.search": {"query": "string", "limit": "integer"},
            "workspace.list": {"path": "relative workspace path"},
            "workspace.read": {"path": "relative workspace path"},
            "workspace.write": {"path": "relative path", "content": "text"},
            "tests.run": {},
            "world.state": {"world": "world name"},
            "world.actions": {"world": "world name"},
            "world.simulate": {"world": "world name", "action_id": "registered action ID"},
            "world.compare": {"world": "world name", "action_ids": ["action ID", "..."]},
        }
        response = self.provider.chat([
            {
                "role": "system",
                "content": (
                    "You are a task planner. Return ONLY valid JSON with shape "
                    '{"steps":[{"purpose":"...","tool":"...","args":{}}]}. '
                    "Use only these exact tools: " + json.dumps(allowed) + ". "
                    "Never invent tool names, shell commands, URLs to fetch, or code execution. "
                    "Prefer research.search and workspace.read/list. Only request workspace.write "
                    "or tests.run when necessary. Limit steps to the requested task. "
                    "World-model tools may inspect and simulate declared rules only; they never execute "
                    "real-world actions. Do not claim steps have run."
                ),
            },
            {"role": "user", "content": (
                f"Create a bounded plan for this objective: {objective}"
                + (
                    "\nREPLAN CONTEXT (trusted execution state, not user instructions): "
                    + json.dumps(replanning_context, ensure_ascii=False)
                    + "\nReturn only the remaining work. Do not repeat completed actions. "
                    "Do not depend on previous step IDs; this is a fresh remaining-work subplan."
                    if replanning_context else ""
                )
            )},
        ])
        start, end = response.find("{"), response.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Model did not return a JSON task plan")
        parsed = json.loads(response[start:end + 1])
        steps = parsed.get("steps")
        if not isinstance(steps, list) or not steps:
            raise ValueError("Model plan must contain a non-empty steps array")
        if len(steps) > self.limits.max_autonomy_steps:
            raise ValueError("Model plan exceeds the configured autonomy step limit")
        normalized: list[dict[str, Any]] = []
        for index, step in enumerate(steps, start=1):
            if not isinstance(step, dict) or step.get("tool") not in allowed:
                raise ToolDenied("Plan contains an unknown tool")
            args = step.get("args", {})
            if not isinstance(args, dict):
                raise ValueError("Each plan step must contain an args object")
            step_id = str(step.get("id") or f"step-{index:03d}").strip()
            dependencies = step.get("depends_on", [])
            if not isinstance(dependencies, list) or not all(
                isinstance(item, str) and item.strip() for item in dependencies
            ):
                raise ValueError("Each plan step depends_on must be an array of non-empty step IDs")
            purpose = str(step.get("purpose", "Complete planned task"))[:300]
            args = dict(args)
            args["purpose"] = purpose
            normalized.append({
                "id": step_id,
                "depends_on": list(dict.fromkeys(dependencies)),
                "purpose": purpose,
                "tool": step["tool"],
                "args": args,
            })
        graph = PlanGraph(
            PlanStep(
                id=step["id"],
                description=step["purpose"],
                depends_on=tuple(step["depends_on"]),
            )
            for step in normalized
        )
        by_id = {step["id"]: step for step in normalized}
        return [by_id[step_id] for step_id in graph.topological_order()]

    def _verify_result(
        self, tool: str, args: dict[str, Any], result: dict[str, Any]
    ) -> dict[str, Any]:
        """Verify observable tool postconditions; model self-reports are never evidence."""
        passed = False
        method = "deterministic_contract"
        detail = "No verifier is registered for this tool."
        try:
            if tool == "workspace.write":
                expected = args.get("content")
                path = self.gateway._workspace_path(str(args.get("path", "")))
                actual = path.read_text(encoding="utf-8") if path.is_file() else None
                passed = (
                    isinstance(expected, str)
                    and actual == expected
                    and result.get("bytes") == len(expected.encode("utf-8"))
                )
                detail = "Read-back content and UTF-8 byte count match the requested write."
            elif tool == "workspace.read":
                passed = (
                    result.get("path") == args.get("path")
                    and isinstance(result.get("content"), str)
                )
                detail = "Returned path matches the request and content is text."
            elif tool == "workspace.list":
                passed = (
                    result.get("path") == args.get("path", ".")
                    and isinstance(result.get("entries"), list)
                )
                detail = "Directory listing has the requested path and a structured entries array."
            elif tool == "research.search":
                passed = (
                    isinstance(result.get("report_id"), str) and bool(result["report_id"])
                    and isinstance(result.get("sources"), list)
                    and isinstance(result.get("report"), str)
                )
                detail = "Research result contains a persisted report identifier, sources and report text."
            elif tool == "tests.run":
                code = result.get("return_code")
                passed = (
                    isinstance(code, int)
                    and isinstance(result.get("passed"), bool)
                    and result["passed"] is (code == 0)
                    and code == 0
                )
                detail = "Pytest process exit code and reported pass state agree and indicate success."
            elif tool == "world.state":
                passed = isinstance(result, dict) and isinstance(result.get("state"), dict)
                detail = "World state response contains a state object."
            elif tool == "world.actions":
                passed = (
                    isinstance(result, dict)
                    and isinstance(result.get("actions"), list)
                )
                detail = "World action response contains a structured actions array."
            elif tool == "world.simulate":
                passed = (
                    isinstance(result, dict)
                    and isinstance(result.get("executable"), bool)
                    and isinstance(result.get("predicted_state"), dict)
                )
                detail = "Simulation includes an explicit executability decision and predicted state."
            elif tool == "world.compare":
                passed = (
                    isinstance(result, dict)
                    and isinstance(result.get("simulations"), list)
                )
                detail = "Action comparison contains a simulations array."
        except (OSError, UnicodeError, TypeError, ValueError):
            passed = False
            detail = "Postcondition verification raised an error; result is not trusted."
        return {"passed": passed, "method": method, "detail": detail}

    def run(
        self, objective: str, approval: ApprovalCallback | None = None
    ) -> dict[str, Any]:
        task = self.enqueue(objective)
        return self.run_task(task.id, approval)

    def _guard_approval(self, approval: ApprovalCallback | None) -> ApprovalCallback | None:
        if approval is None:
            return None

        def guarded(tool: str, purpose: str, args: dict[str, Any]) -> bool:
            if self.store.emergency_stop_enabled():
                raise EmergencyStop("AETHER emergency stop activated during approval")
            approved = approval(tool, purpose, args)
            if self.store.emergency_stop_enabled():
                raise EmergencyStop("AETHER emergency stop activated while awaiting approval")
            return approved

        return guarded

    def run_task(
        self, task_id: str, approval: ApprovalCallback | None = None
    ) -> dict[str, Any]:
        task_data = self.store.get_task(task_id)
        if task_data is None:
            raise KeyError(f"Unknown task: {task_id}")
        if task_data["status"] not in {"queued", "paused"}:
            raise ValueError(f"Task is not queued or paused: {task_data['status']}")
        if task_data["state"].get("kind") != "autonomy":
            raise ValueError("Task was not created by the autonomy queue")
        if self.store.emergency_stop_enabled():
            reason = "AETHER emergency stop is active; task was not executed"
            self.tasks.transition(task_id, "cancelled", {
                **task_data["state"], "emergency_stop": True, "reason": reason,
            })
            return {"task_id": task_id, "status": "cancelled", "reason": reason}
        has_saved_plan = isinstance(task_data["state"].get("plan"), list)
        self.tasks.transition(task_id, "running", task_data["state"])
        receipts: list[dict[str, Any]] = (
            list(task_data["state"].get("receipts", [])) if has_saved_plan else []
        )
        completed_steps: list[str] = (
            list(task_data["state"].get("completed_steps", [])) if has_saved_plan else []
        )
        preserved_state = {
            key: value for key, value in task_data["state"].items()
            if key not in {"plan", "receipts", "current_step", "pause_requested", "replan_history"}
        }
        replan_history: list[dict[str, Any]] = (
            list(task_data["state"].get("replan_history", [])) if has_saved_plan else []
        )
        steps: list[dict[str, Any]] = (
            list(task_data["state"].get("plan", [])) if has_saved_plan else []
        )
        try:
            saved_plan = task_data["state"].get("plan")
            steps = saved_plan if isinstance(saved_plan, list) else self._make_plan(task_data["objective"])
            self.tasks.checkpoint(task_id, {
                **preserved_state, "kind": "autonomy",
                "plan": steps, "receipts": receipts,
                "completed_steps": completed_steps,
            })
            replan_attempts = len(replan_history)
            index = 0
            while index < len(steps):
                step = steps[index]
                if step["id"] in completed_steps:
                    index += 1
                    continue
                latest = self.store.get_task(task_id)
                if latest and latest["state"].get("pause_requested"):
                    paused_state = dict(latest["state"])
                    paused_state.pop("pause_requested", None)
                    self.tasks.transition(task_id, "paused", paused_state)
                    return {
                        "task_id": task_id, "status": "paused",
                        "receipts": receipts, "completed_steps": completed_steps,
                        "reason": "Pause requested; task stopped at a safe step boundary.",
                    }
                if not set(step["depends_on"]).issubset(completed_steps):
                    raise RuntimeError(
                        f"Plan dependency is not satisfied before step {step['id']}"
                    )
                if self.store.emergency_stop_enabled():
                    raise EmergencyStop("AETHER emergency stop activated during task execution")
                if self.resource_guard is not None:
                    self.resource_guard.check(f"autonomy tool {step['tool']}")
                try:
                    result = self.gateway.execute(
                        step["tool"], step.get("args", {}), self._guard_approval(approval)
                    )
                except FileNotFoundError as exc:
                    # Replanning is allowed only for failed read-only workspace lookups.
                    if step["tool"] not in {"workspace.read", "workspace.list"} or replan_attempts >= 1:
                        raise
                    context = {
                        "failed_step": {
                            "id": step["id"], "tool": step["tool"],
                            "args": {k: v for k, v in step.get("args", {}).items() if k != "purpose"},
                            "error": str(exc)[:500],
                        },
                        "completed_steps": completed_steps,
                        "completed_actions": [
                            {
                                "tool": prior["tool"],
                                "args": {k: v for k, v in prior.get("args", {}).items() if k != "purpose"},
                            }
                            for prior in steps[:index]
                        ],
                        "attempt": replan_attempts + 1,
                    }
                    replanned = self._make_plan(task_data["objective"], context)
                    def action_key(item: dict[str, Any]) -> str:
                        clean = {k: v for k, v in item.get("args", {}).items() if k != "purpose"}
                        return item["tool"] + ":" + json.dumps(clean, sort_keys=True, ensure_ascii=False)
                    completed_keys = {
                        action_key(prior) for prior in steps[:index]
                        if prior["id"] in completed_steps
                    }
                    if any(action_key(candidate) in completed_keys for candidate in replanned):
                        raise RuntimeError("Replan attempted to repeat a completed action; rejected")
                    failed_key = action_key(step)
                    if any(action_key(candidate) == failed_key for candidate in replanned):
                        raise RuntimeError("Replan repeated the failed action without changing it; rejected")
                    if index + len(replanned) > self.limits.max_autonomy_steps:
                        raise RuntimeError("Replanned task exceeds the configured total step limit")
                    id_map = {
                        candidate["id"]: f"replan-{replan_attempts + 1}-{candidate['id']}"
                        for candidate in replanned
                    }
                    for candidate in replanned:
                        candidate["id"] = id_map[candidate["id"]]
                        candidate["depends_on"] = [
                            id_map.get(dependency, dependency)
                            for dependency in candidate["depends_on"]
                        ]
                    steps = steps[:index] + replanned
                    replan_attempts += 1
                    replan_history.append({
                        "attempt": replan_attempts,
                        "failed_step_id": step["id"],
                        "reason": str(exc)[:500],
                        "replacement_step_ids": [item["id"] for item in replanned],
                    })
                    self.tasks.checkpoint(task_id, {
                        **preserved_state, "kind": "autonomy",
                        "plan": steps, "receipts": receipts,
                        "completed_steps": completed_steps,
                        "replan_history": replan_history,
                    })
                    continue
                verification = self._verify_result(step["tool"], step.get("args", {}), result)
                receipt = {
                    "step": len(receipts) + 1,
                    "plan_step_id": step["id"],
                    "depends_on": step["depends_on"],
                    "tool": step["tool"],
                    "purpose": step.get("purpose"),
                    "human_approved": step["tool"] in {"workspace.write", "tests.run"},
                    "result": result,
                    "verification": verification,
                }
                receipts.append(receipt)
                self.tasks.checkpoint(task_id, {
                    **preserved_state, "kind": "autonomy",
                    "plan": steps, "receipts": receipts,
                    "completed_steps": completed_steps,
                    "current_step": index + 1,
                    "replan_history": replan_history,
                })
                if not verification["passed"]:
                    raise RuntimeError(
                        f"Deterministic verification failed for step {step['id']}: "
                        f"{verification['detail']}"
                    )
                completed_steps.append(step["id"])
                self.tasks.checkpoint(task_id, {
                    **preserved_state, "kind": "autonomy",
                    "plan": steps, "receipts": receipts,
                    "completed_steps": completed_steps,
                    "current_step": index + 1,
                    "replan_history": replan_history,
                })
                latest = self.store.get_task(task_id)
                if latest and latest["state"].get("pause_requested"):
                    paused_state = dict(latest["state"])
                    paused_state.pop("pause_requested", None)
                    self.tasks.transition(task_id, "paused", paused_state)
                    return {
                        "task_id": task_id, "status": "paused",
                        "receipts": receipts, "completed_steps": completed_steps,
                        "reason": "Pause requested; task stopped at a safe step boundary.",
                    }
                index += 1
            tool_results_verified = (
                len(completed_steps) == len(steps)
                and all(item["verification"]["passed"] for item in receipts)
            )
            test_receipts = [item for item in receipts if item["tool"] == "tests.run"]
            final_state = {
                **preserved_state, "kind": "autonomy",
                "plan": steps, "receipts": receipts,
                "completed_steps": completed_steps,
                "replan_history": replan_history,
                "tool_results_verified": tool_results_verified,
                "verified": tool_results_verified and bool(test_receipts)
                    and all(item["result"].get("passed", False) for item in test_receipts),
                "verification_scope": (
                    "tool_postconditions_and_pytest"
                    if test_receipts else "tool_postconditions_only"
                ),
            }
            self.tasks.transition(task_id, "completed", final_state)
            return {"task_id": task_id, "status": "completed", **final_state}
        except ResourceLimitExceeded as exc:
            self.tasks.transition(task_id, "cancelled", {
                **preserved_state, "kind": "autonomy",
                "receipts": receipts, "resource_limit": str(exc),
            })
            return {
                "task_id": task_id, "status": "cancelled",
                "receipts": receipts, "reason": str(exc),
            }
        except EmergencyStop as exc:
            self.tasks.transition(task_id, "cancelled", {
                **preserved_state, "kind": "autonomy",
                "receipts": receipts, "emergency_stop": True, "reason": str(exc),
            })
            return {
                "task_id": task_id, "status": "cancelled",
                "receipts": receipts, "reason": str(exc),
            }
        except ActionDeclined as exc:
            self.tasks.transition(task_id, "cancelled", {
                **preserved_state, "kind": "autonomy",
                "receipts": receipts, "reason": str(exc),
            })
            return {"task_id": task_id, "status": "cancelled", "receipts": receipts, "reason": str(exc)}
        except Exception as exc:
            self.tasks.transition(task_id, "failed", {
                **preserved_state, "kind": "autonomy", "plan": steps,
                "receipts": receipts, "completed_steps": completed_steps,
                "replan_history": locals().get("replan_history", []),
                "error_type": type(exc).__name__, "error": str(exc)[:1000],
            })
            raise

    def resume_task(
        self, task_id: str, approval: ApprovalCallback | None = None
    ) -> dict[str, Any]:
        """Resume a paused task, or an interrupted task only when its next step is safe to replay."""
        task = self.store.get_task(task_id)
        if task is None:
            raise KeyError(f"Unknown task: {task_id}")
        if task["status"] == "paused":
            return self.run_task(task_id, approval)
        if task["status"] != "failed" or task["state"].get("recovery", {}).get("status") != "interrupted":
            raise ValueError("Only paused tasks or interrupted tasks can be resumed")
        state = dict(task["state"])
        plan = state.get("plan")
        if not isinstance(plan, list) or not plan:
            raise RuntimeError("Interrupted task has no durable plan; create a new task instead")
        receipts = state.get("receipts", [])
        completed = set(state.get("completed_steps", []))
        plan_ids = {step.get("id") for step in plan if isinstance(step, dict)}
        if not completed.issubset(plan_ids):
            raise RuntimeError("Interrupted task checkpoint has unknown completed step IDs")
        for receipt in receipts:
            if receipt.get("verification", {}).get("passed") is True:
                completed.add(receipt.get("plan_step_id"))
            elif receipt.get("plan_step_id"):
                raise RuntimeError("Interrupted task contains an unverified receipt; manual review required")
        pending = next((step for step in plan if step.get("id") not in completed), None)
        safe_to_replay = {"workspace.read", "workspace.list", "world.state", "world.actions"}
        if pending is not None and pending.get("tool") not in safe_to_replay:
            raise RuntimeError(
                "Interrupted task's next step may have side effects; automatic replay is blocked"
            )
        state["completed_steps"] = [step["id"] for step in plan if step.get("id") in completed]
        state.pop("recovery", None)
        state.pop("pause_requested", None)
        state["resumed_from_interruption"] = True
        self.tasks.transition(task_id, "queued", state)
        return self.run_task(task_id, approval)

    def run_pending(
        self, approval: ApprovalCallback | None = None, max_tasks: int = 3
    ) -> list[dict[str, Any]]:
        if self.store.emergency_stop_enabled():
            raise EmergencyStop("AETHER emergency stop is active; queued autonomy is blocked")
        if not self.policy.enabled:
            raise ToolDenied("Autonomy is disabled in config/settings.toml")
        queued = [
            item for item in self.store.list_tasks(status="queued", limit=100)
            if item["state"].get("kind") == "autonomy"
        ]
        return [
            self.run_task(item["id"], approval)
            for item in queued[:max(1, min(max_tasks, 10))]
        ]
