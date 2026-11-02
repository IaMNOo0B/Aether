# Phase 06 — Bounded Autonomous Operation

This phase adds a persistent goal lifecycle around AETHER's existing permission-enforced tool gateway. It is deliberately opt-in.

## Enablement

In `config/settings.toml`, set:

```toml
[models]
enabled = true

[autonomy]
enabled = true
allow_read_only = true
allow_workspace_writes = false
allow_code_execution = false

[scheduler]
enabled = true
poll_interval_seconds = 30
max_tasks_per_tick = 1
```

The scheduler requires both `models.enabled` and `autonomy.enabled`. It runs in the AETHER process and stops when that process exits. Do not enable unattended autonomy until the selected model and workspace configuration have been checked.

## Goal lifecycle

Goals are persisted in SQLite with:
- Level: `immediate`, `project`, `system`, `research`, or `exploration`.
- Purpose, expected benefit, priority, complexity, risks, optional deadline and dependencies.
- Completion criteria and a human-readable stop condition.
- Resource budget: maximum active runtime, model calls and tool calls.
- Status and an append-only lifecycle event history.

Goal discovery currently proposes recovery work from failed autonomy tasks, verification work for unverified/degraded capability records, and engineering objectives from persisted OMEGA acceptance gaps. It also supports explicit user-entered goals and recurring schedules. A proposal is not authorization. A human must approve it before the scheduler can queue it.

## Console commands

- `/goal add TITLE | PURPOSE | EXPECTED BENEFIT | PRIORITY(optional 0..1) | LEVEL(optional)`
- `/goal discover` — inspect known failure/capability-gap signals and create proposals.
- `/goal list` — show goals ordered by priority.
- `/goal approve GOAL_ID` — explicitly authorize queueing.
- `/goal queue` — queue approved goals and run them interactively.
- `/goal cancel GOAL_ID`
- `/goal events GOAL_ID` — inspect the goal audit history.
- `/schedule add INTERVAL_SECONDS | OBJECTIVE | MAX_RUNS(optional; 0 means unlimited)`
- `/schedule list`, `/schedule pause ID`, `/schedule resume ID`, `/schedule tick`
- `/task pause TASK_ID`, `/task resume TASK_ID`, `/task cancel TASK_ID`
- `/task replan TASK_ID | NEW OBJECTIVE` — allowed only while paused and only when recorded completed steps are read-only.
- `/task-events TASK_ID` — inspect task events and receipts.
- `/notifications`, `/notification-read NOTIFICATION_ID`
- `/emergency-stop` and `/resume-autonomy`

## Execution and safety rules

- Background scheduling only uses read-only tools. It cannot approve workspace writes or test execution.
- Interactive workspace writes and test execution still require feature flags and per-action human approval.
- The fixed tool registry remains in force; no arbitrary shell, unrestricted filesystem, or arbitrary network fetch tool is added.
- Pause and cancel are cooperative: an already-running tool call is allowed to return, then AETHER checkpoints its receipt and stops before the next step.
- Resume continues from the saved plan and completed-step index; it does not regenerate or replay completed steps.
- Replanning is refused if any completed receipt may have caused a side effect.
- A crash marks a running task interrupted for manual review. AETHER never blindly replays an uncertain action.
- Emergency stop blocks scheduler and tool execution.
- Completion criteria and the human-readable stop condition are persisted as goal metadata. A task is marked `completed` when its planned steps return successfully; this does not mean natural-language completion criteria were independently proven. The `verified` flag is true only when an included test-run step passes.
- Machine-enforced stop conditions are emergency stop, approval denial, cancellation, and configured resource/tool/model-call budgets.

## Resource budgets

Per-goal budgets default to 300 seconds of active runtime, 8 tool calls and 1 model call. They are bounded by the global maximum plan-step setting. CPU, RAM and NVIDIA GPU/VRAM telemetry are checked before planning and tool execution when available.

These are cooperative pre-action checks, not hard real-time OS quotas. They do not reserve VRAM or impose per-process CPU/RAM limits. The scheduler is bounded by its poll interval and maximum tasks per tick.

## Goal discovery limitations

Discovery is intentionally conservative. It currently covers persisted failed autonomy tasks, unverified/degraded capability records, and failed/partial cases in the latest saved OMEGA acceptance report, plus explicit user-entered goals and recurring schedules. It does not yet infer every project dependency, external-system event or research opportunity. Discovered goals never self-approve.
