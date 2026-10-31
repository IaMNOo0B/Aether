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
from aether.research import Researcher
from aether.storage import Store
from aether.world_model import WorldModel
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
    ):
        self.provider = provider
        self.store = store
        self.policy = policy
        self.limits = limits
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

    def _make_plan(self, objective: str) -> list[dict[str, Any]]:
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
            {"role": "user", "content": f"Create a bounded plan for this objective: {objective}"},
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
        for step in steps:
            if not isinstance(step, dict) or step.get("tool") not in allowed:
                raise ToolDenied("Plan contains an unknown tool")
            if not isinstance(step.get("args", {}), dict):
                raise ValueError("Each plan step must contain an args object")
            step["args"]["purpose"] = str(step.get("purpose", "Complete planned task"))[:300]
        return steps

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
        if task_data["status"] != "queued":
            raise ValueError(f"Task is not queued: {task_data['status']}")
        if task_data["state"].get("kind") != "autonomy":
            raise ValueError("Task was not created by the autonomy queue")
        if self.store.emergency_stop_enabled():
            reason = "AETHER emergency stop is active; task was not executed"
            self.tasks.transition(task_id, "cancelled", {
                **task_data["state"], "emergency_stop": True, "reason": reason,
            })
            return {"task_id": task_id, "status": "cancelled", "reason": reason}
        self.tasks.transition(task_id, "running")
        receipts: list[dict[str, Any]] = []
        preserved_state = {
            key: value for key, value in task_data["state"].items()
            if key not in {"plan", "receipts", "current_step"}
        }
        try:
            steps = self._make_plan(task_data["objective"])
            self.tasks.checkpoint(task_id, {
                **preserved_state, "kind": "autonomy",
                "plan": steps, "receipts": receipts,
            })
            for index, step in enumerate(steps, start=1):
                if self.store.emergency_stop_enabled():
                    raise EmergencyStop("AETHER emergency stop activated during task execution")
                result = self.gateway.execute(
                    step["tool"], step.get("args", {}), self._guard_approval(approval)
                )
                receipts.append({
                    "step": index, "tool": step["tool"], "purpose": step.get("purpose"),
                    "human_approved": step["tool"] in {"workspace.write", "tests.run"},
                    "result": result,
                })
                self.tasks.checkpoint(task_id, {
                    **preserved_state, "kind": "autonomy",
                    "plan": steps, "receipts": receipts,
                    "current_step": index,
                })
                if step["tool"] == "tests.run" and not result.get("passed", False):
                    raise RuntimeError("Independent test suite failed; autonomy task rejected")
            test_receipts = [item for item in receipts if item["tool"] == "tests.run"]
            final_state = {
                **preserved_state, "kind": "autonomy",
                "plan": steps, "receipts": receipts,
                "verified": bool(test_receipts) and all(
                    item["result"].get("passed", False) for item in test_receipts
                ),
            }
            self.tasks.transition(task_id, "completed", final_state)
            return {"task_id": task_id, "status": "completed", **final_state}
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
                **preserved_state, "kind": "autonomy", "receipts": receipts,
                "error_type": type(exc).__name__, "error": str(exc)[:1000],
            })
            raise

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
