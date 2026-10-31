"""Persistent SQLite storage for AETHER's operational and learning records."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import math
import sqlite3
import time
from uuid import uuid4
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, database_path: str | Path):
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at);
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'unverified',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    objective TEXT NOT NULL,
                    status TEXT NOT NULL,
                    state_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_tasks_status_updated ON tasks(status, updated_at);
                CREATE TABLE IF NOT EXISTS task_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    from_status TEXT,
                    to_status TEXT,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES tasks(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_task_events_task_sequence
                    ON task_events(task_id, sequence);
                CREATE TABLE IF NOT EXISTS scheduled_goals (
                    id TEXT PRIMARY KEY,
                    objective TEXT NOT NULL,
                    interval_seconds INTEGER NOT NULL CHECK(interval_seconds >= 5),
                    next_run_at REAL NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0, 1)),
                    run_count INTEGER NOT NULL DEFAULT 0,
                    max_runs INTEGER NOT NULL DEFAULT 0 CHECK(max_runs >= 0),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_scheduled_goals_due
                    ON scheduled_goals(enabled, next_run_at);
                CREATE TABLE IF NOT EXISTS system_controls (
                    name TEXT PRIMARY KEY,
                    enabled INTEGER NOT NULL CHECK(enabled IN (0, 1)),
                    reason TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS system_control_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    control_name TEXT NOT NULL,
                    enabled INTEGER NOT NULL CHECK(enabled IN (0, 1)),
                    reason TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS decisions (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    probability REAL NOT NULL CHECK(probability >= 0 AND probability <= 1),
                    predicted_label INTEGER NOT NULL CHECK(predicted_label IN (0, 1)),
                    outcome INTEGER CHECK(outcome IN (0, 1) OR outcome IS NULL),
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    resolved_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_decisions_resolved ON decisions(resolved_at);
                CREATE TABLE IF NOT EXISTS research_reports (
                    id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    content TEXT NOT NULL,
                    sources_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS skills (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    procedure TEXT NOT NULL,
                    source TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'draft',
                    evaluation_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS agent_runs (
                    id TEXT PRIMARY KEY,
                    objective TEXT NOT NULL,
                    role TEXT NOT NULL,
                    output TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_agent_runs_created ON agent_runs(created_at);
            """)
            db.execute(
                """INSERT OR IGNORE INTO system_controls(name, enabled, reason, updated_at)
                   VALUES ('emergency_stop', 0, '', ?)""",
                (utc_now(),),
            )

    def add_message(self, role: str, content: str) -> int:
        if role not in {"user", "assistant", "system"}:
            raise ValueError(f"Unsupported message role: {role}")
        with self._connect() as db:
            cursor = db.execute(
                "INSERT INTO messages(role, content, created_at) VALUES (?, ?, ?)",
                (role, content, utc_now()),
            )
            return int(cursor.lastrowid)

    def recent_messages(self, limit: int = 12) -> list[dict[str, str]]:
        if limit < 1:
            return []
        with self._connect() as db:
            rows = db.execute(
                "SELECT role, content FROM messages ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def add_memory(
        self, content: str, source: str = "user", status: str = "unverified",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        with self._connect() as db:
            cursor = db.execute(
                """INSERT INTO memories(content, source, status, metadata_json, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (content, source, status, json.dumps(metadata or {}), utc_now()),
            )
            return int(cursor.lastrowid)


    def get_memory(self, memory_id: int) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT id, content, source, status, metadata_json, created_at FROM memories WHERE id=?",
                (memory_id,),
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["metadata"] = json.loads(item.pop("metadata_json"))
        return item

    def list_memories(self, limit: int = 5000) -> list[dict[str, Any]]:
        if limit < 1:
            return []
        with self._connect() as db:
            rows = db.execute(
                """SELECT id, content, source, status, metadata_json, created_at
                   FROM memories ORDER BY id DESC LIMIT ?""",
                (min(limit, 20000),),
            ).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json"))
            items.append(item)
        return items

    def update_memory(
        self, memory_id: int, *, status: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as db:
            row = db.execute(
                "SELECT status, metadata_json FROM memories WHERE id=?", (memory_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown memory: {memory_id}")
            next_status = row["status"] if status is None else status
            next_metadata = (
                json.loads(row["metadata_json"]) if metadata is None else metadata
            )
            db.execute(
                "UPDATE memories SET status=?, metadata_json=? WHERE id=?",
                (next_status, json.dumps(next_metadata), memory_id),
            )


    def supersede_memory(
        self, memory_id: int, content: str, reason: str, *,
        source: str, status: str, metadata: dict[str, Any],
    ) -> int:
        """Atomically preserve an old memory and append its corrected successor."""
        text = content.strip()
        explanation = reason.strip()
        if not text or not explanation:
            raise ValueError("Replacement content and correction reason are required")
        now = utc_now()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT status, metadata_json FROM memories WHERE id=?", (memory_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown memory: {memory_id}")
            if row["status"] == "superseded":
                raise ValueError("Memory has already been superseded")
            successor_metadata = dict(metadata)
            successor_metadata["supersedes_memory_id"] = memory_id
            cursor = db.execute(
                """INSERT INTO memories(content, source, status, metadata_json, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (text, source, status, json.dumps(successor_metadata), now),
            )
            successor_id = int(cursor.lastrowid)
            previous_metadata = json.loads(row["metadata_json"])
            previous_metadata["superseded_by_id"] = successor_id
            previous_metadata["superseded_by_reason"] = explanation
            db.execute(
                "UPDATE memories SET status='superseded', metadata_json=? WHERE id=?",
                (json.dumps(previous_metadata), memory_id),
            )
        return successor_id

    def search_memories(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        terms = [term.strip() for term in query.split() if len(term.strip()) > 2]
        if not terms or limit < 1:
            return []
        clauses = " OR ".join("content LIKE ?" for _ in terms)
        parameters = [f"%{term}%" for term in terms]
        with self._connect() as db:
            rows = db.execute(
                f"""SELECT id, content, source, status, created_at
                    FROM memories WHERE {clauses} ORDER BY id DESC LIMIT ?""",
                (*parameters, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_task(
        self, task_id: str, objective: str, status: str,
        state: dict[str, Any] | None = None,
    ) -> None:
        with self._connect() as db:
            db.execute(
                """INSERT INTO tasks(id, objective, status, state_json, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET objective=excluded.objective,
                     status=excluded.status, state_json=excluded.state_json,
                     updated_at=excluded.updated_at""",
                (task_id, objective, status, json.dumps(state or {}), utc_now()),
            )

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT id, objective, status, state_json, updated_at FROM tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["state"] = json.loads(result.pop("state_json"))
        return result

    def list_tasks(self, status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        if limit < 1:
            return []
        with self._connect() as db:
            if status is None:
                rows = db.execute(
                    "SELECT id FROM tasks ORDER BY updated_at DESC LIMIT ?", (limit,)
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT id FROM tasks WHERE status = ? ORDER BY updated_at DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
        return [task for row in rows if (task := self.get_task(row["id"])) is not None]


    def create_task(
        self, task_id: str, objective: str, state: dict[str, Any] | None = None,
    ) -> None:
        """Create a task and its first event in one transaction."""
        now = utc_now()
        payload = state or {}
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "INSERT INTO tasks(id, objective, status, state_json, updated_at) VALUES (?, ?, 'queued', ?, ?)",
                (task_id, objective, json.dumps(payload), now),
            )
            db.execute(
                """INSERT INTO task_events(task_id, event_type, from_status, to_status,
                   payload_json, created_at) VALUES (?, 'created', NULL, 'queued', ?, ?)""",
                (task_id, json.dumps(payload), now),
            )

    def transition_task(
        self, task_id: str, expected_status: str, new_status: str,
        state: dict[str, Any],
    ) -> dict[str, Any]:
        """Atomically compare-and-transition a task and append its audit event."""
        now = utc_now()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT objective, status FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown task: {task_id}")
            if row["status"] != expected_status:
                raise ValueError(
                    f"Task status changed concurrently: expected {expected_status}, found {row['status']}"
                )
            db.execute(
                "UPDATE tasks SET status=?, state_json=?, updated_at=? WHERE id=?",
                (new_status, json.dumps(state), now, task_id),
            )
            db.execute(
                """INSERT INTO task_events(task_id, event_type, from_status, to_status,
                   payload_json, created_at) VALUES (?, 'transition', ?, ?, ?, ?)""",
                (task_id, expected_status, new_status, json.dumps(state), now),
            )
        return self.get_task(task_id)  # type: ignore[return-value]

    def checkpoint_task(
        self, task_id: str, state: dict[str, Any], event_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Persist a running task checkpoint and audit event atomically."""
        now = utc_now()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status FROM tasks WHERE id=?", (task_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown task: {task_id}")
            if row["status"] != "running":
                raise ValueError(f"Only running tasks can be checkpointed, found {row['status']}")
            db.execute(
                "UPDATE tasks SET state_json=?, updated_at=? WHERE id=?",
                (json.dumps(state), now, task_id),
            )
            db.execute(
                """INSERT INTO task_events(task_id, event_type, from_status, to_status,
                   payload_json, created_at) VALUES (?, 'checkpoint', 'running', 'running', ?, ?)""",
                (task_id, json.dumps(event_data or state), now),
            )
        return self.get_task(task_id)  # type: ignore[return-value]

    def task_events(self, task_id: str, limit: int = 500) -> list[dict[str, Any]]:
        if limit < 1:
            return []
        with self._connect() as db:
            rows = db.execute(
                """SELECT sequence, event_type, from_status, to_status, payload_json, created_at
                   FROM task_events WHERE task_id=? ORDER BY sequence LIMIT ?""",
                (task_id, limit),
            ).fetchall()
        return [
            {"sequence": row["sequence"], "event_type": row["event_type"],
             "from_status": row["from_status"], "to_status": row["to_status"],
             "payload": json.loads(row["payload_json"]), "created_at": row["created_at"]}
            for row in rows
        ]

    def recover_interrupted_tasks(self) -> list[str]:
        """Mark stale running tasks interrupted; never replay side effects automatically."""
        recovered: list[str] = []
        now = utc_now()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute(
                "SELECT id, state_json FROM tasks WHERE status='running' ORDER BY updated_at"
            ).fetchall()
            for row in rows:
                state = json.loads(row["state_json"])
                state["recovery"] = {
                    "status": "interrupted",
                    "detected_at": now,
                    "policy": "manual_review_required",
                    "reason": "Process stopped while task was running; side effects were not replayed.",
                }
                db.execute(
                    "UPDATE tasks SET status='failed', state_json=?, updated_at=? WHERE id=? AND status='running'",
                    (json.dumps(state), now, row["id"]),
                )
                db.execute(
                    """INSERT INTO task_events(task_id, event_type, from_status, to_status,
                       payload_json, created_at) VALUES (?, 'interrupted_recovery', 'running', 'failed', ?, ?)""",
                    (row["id"], json.dumps(state["recovery"]), now),
                )
                recovered.append(row["id"])
        return recovered



    def emergency_stop_enabled(self) -> bool:
        with self._connect() as db:
            row = db.execute(
                "SELECT enabled FROM system_controls WHERE name='emergency_stop'"
            ).fetchone()
        return bool(row["enabled"]) if row else False

    def set_emergency_stop(self, enabled: bool, reason: str = "User activated emergency stop") -> None:
        now = utc_now()
        explanation = reason.strip()[:500]
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                """INSERT INTO system_controls(name, enabled, reason, updated_at)
                   VALUES ('emergency_stop', ?, ?, ?)
                   ON CONFLICT(name) DO UPDATE SET enabled=excluded.enabled,
                     reason=excluded.reason, updated_at=excluded.updated_at""",
                (int(bool(enabled)), explanation, now),
            )
            db.execute(
                """INSERT INTO system_control_events(control_name, enabled, reason, created_at)
                   VALUES ('emergency_stop', ?, ?, ?)""",
                (int(bool(enabled)), explanation, now),
            )

    def system_control_events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                """SELECT sequence, control_name, enabled, reason, created_at
                   FROM system_control_events ORDER BY sequence DESC LIMIT ?""",
                (max(0, min(limit, 1000)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def create_schedule(
        self, schedule_id: str, objective: str, interval_seconds: int,
        max_runs: int = 0, now_epoch: float | None = None,
    ) -> None:
        if not objective.strip() or interval_seconds < 5 or max_runs < 0:
            raise ValueError("Schedule objective, interval >= 5 and max_runs >= 0 are required")
        now = utc_now()
        current = time.time() if now_epoch is None else float(now_epoch)
        with self._connect() as db:
            db.execute(
                """INSERT INTO scheduled_goals
                   (id, objective, interval_seconds, next_run_at, enabled,
                    run_count, max_runs, created_at, updated_at)
                   VALUES (?, ?, ?, ?, 1, 0, ?, ?, ?)""",
                (schedule_id, objective.strip(), interval_seconds,
                 current + interval_seconds, max_runs, now, now),
            )

    def list_schedules(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                """SELECT id, objective, interval_seconds, next_run_at, enabled,
                          run_count, max_runs, created_at, updated_at
                   FROM scheduled_goals ORDER BY next_run_at LIMIT ?""",
                (max(0, min(limit, 1000)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def set_schedule_enabled(self, schedule_id: str, enabled: bool) -> None:
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE scheduled_goals SET enabled=?, updated_at=? WHERE id=?",
                (int(bool(enabled)), utc_now(), schedule_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown schedule: {schedule_id}")

    def enqueue_due_schedules(
        self, now_epoch: float | None = None, limit: int = 1,
    ) -> list[str]:
        """Atomically advance due schedules and enqueue one durable task per due goal."""
        current = __import__("time").time() if now_epoch is None else float(now_epoch)
        now = utc_now()
        created: list[str] = []
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute(
                """SELECT * FROM scheduled_goals
                   WHERE enabled=1 AND next_run_at <= ?
                     AND (max_runs=0 OR run_count < max_runs)
                   ORDER BY next_run_at LIMIT ?""",
                (current, max(0, min(limit, 10))),
            ).fetchall()
            for row in rows:
                task_id = uuid4().hex
                state = {
                    "kind": "autonomy",
                    "scheduled": True,
                    "scheduled_goal_id": row["id"],
                    "receipts": [],
                    "execution_policy": "read_only_only",
                }
                db.execute(
                    """INSERT INTO tasks(id, objective, status, state_json, updated_at)
                       VALUES (?, ?, 'queued', ?, ?)""",
                    (task_id, row["objective"], json.dumps(state), now),
                )
                db.execute(
                    """INSERT INTO task_events(task_id, event_type, from_status, to_status,
                       payload_json, created_at)
                       VALUES (?, 'scheduled_task_created', NULL, 'queued', ?, ?)""",
                    (task_id, json.dumps({"schedule_id": row["id"]}), now),
                )
                next_count = row["run_count"] + 1
                enabled = 0 if row["max_runs"] and next_count >= row["max_runs"] else 1
                db.execute(
                    """UPDATE scheduled_goals SET next_run_at=?, enabled=?,
                       run_count=?, updated_at=? WHERE id=?""",
                    (current + row["interval_seconds"], enabled, next_count, now, row["id"]),
                )
                created.append(task_id)
        return created

    def record_decision(
        self, decision_id: str, question: str, probability: float,
        predicted_label: bool | int, metadata: dict[str, Any] | None = None,
    ) -> None:
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Decision probability must be finite and between zero and one")
        if predicted_label not in (False, True, 0, 1):
            raise ValueError("Predicted label must be boolean or binary")
        if not question.strip() or not decision_id.strip():
            raise ValueError("Decision ID and question cannot be empty")
        with self._connect() as db:
            db.execute(
                """INSERT INTO decisions(
                    id, question, probability, predicted_label, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)""",
                (decision_id, question, probability, int(bool(predicted_label)),
                 json.dumps(metadata or {}), utc_now()),
            )

    def resolve_decision(self, decision_id: str, outcome: bool | int) -> None:
        if outcome not in (False, True, 0, 1):
            raise ValueError("Decision outcome must be boolean or binary")
        with self._connect() as db:
            cursor = db.execute(
                """UPDATE decisions SET outcome = ?, resolved_at = ?
                   WHERE id = ? AND outcome IS NULL""",
                (int(bool(outcome)), utc_now(), decision_id),
            )
            if cursor.rowcount == 1:
                return
            exists = db.execute(
                "SELECT outcome FROM decisions WHERE id = ?", (decision_id,)
            ).fetchone()
        if exists is None:
            raise KeyError(f"Unknown decision: {decision_id}")
        raise ValueError(f"Decision already resolved: {decision_id}")

    def list_decisions(self, resolved_only: bool = True, limit: int = 1000) -> list[dict[str, Any]]:
        if limit < 1:
            return []
        condition = "WHERE outcome IS NOT NULL" if resolved_only else ""
        with self._connect() as db:
            rows = db.execute(
                f"""SELECT id, question, probability, predicted_label, outcome,
                           metadata_json, created_at, resolved_at
                    FROM decisions {condition} ORDER BY created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json"))
            results.append(item)
        return results

    def save_research_report(
        self, report_id: str, query: str, content: str, sources: list[dict[str, Any]]
    ) -> None:
        with self._connect() as db:
            db.execute(
                """INSERT INTO research_reports(id, query, content, sources_json, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (report_id, query, content, json.dumps(sources), utc_now()),
            )

    def get_research_report(self, report_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM research_reports WHERE id = ?", (report_id,)
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["sources"] = json.loads(item.pop("sources_json"))
        return item

    def list_research_reports(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT id, query, created_at FROM research_reports ORDER BY created_at DESC LIMIT ?",
                (max(0, limit),),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_skill(
        self, skill_id: str, title: str, procedure: str, source: str,
        status: str = "draft", evaluation: dict[str, Any] | None = None,
    ) -> None:
        now = utc_now()
        with self._connect() as db:
            db.execute(
                """INSERT INTO skills(id, title, procedure, source, status,
                    evaluation_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (skill_id, title, procedure, source, status,
                 json.dumps(evaluation or {}), now, now),
            )

    def get_skill(self, skill_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM skills WHERE id = ?", (skill_id,)).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["evaluation"] = json.loads(item.pop("evaluation_json"))
        return item

    def list_skills(self, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as db:
            if status is None:
                rows = db.execute(
                    "SELECT id FROM skills ORDER BY updated_at DESC LIMIT ?", (limit,)
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT id FROM skills WHERE status = ? ORDER BY updated_at DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
        return [skill for row in rows if (skill := self.get_skill(row["id"])) is not None]

    def evaluate_skill(self, skill_id: str, passed: bool, evidence: str) -> None:
        if not evidence.strip():
            raise ValueError("Skill evaluation requires evidence")
        skill = self.get_skill(skill_id)
        if skill is None:
            raise KeyError(f"Unknown skill: {skill_id}")
        evaluation = dict(skill["evaluation"])
        history = evaluation.setdefault("runs", [])
        history.append({"passed": bool(passed), "evidence": evidence, "at": utc_now()})
        status = "validated" if passed else "draft"
        with self._connect() as db:
            db.execute(
                """UPDATE skills SET status = ?, evaluation_json = ?, updated_at = ?
                   WHERE id = ?""",
                (status, json.dumps(evaluation), utc_now(), skill_id),
            )

    def save_agent_run(
        self, run_id: str, objective: str, role: str, output: str, status: str = "completed"
    ) -> None:
        with self._connect() as db:
            db.execute(
                """INSERT INTO agent_runs(id, objective, role, output, status, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (run_id, objective, role, output, status, utc_now()),
            )

    def list_agent_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                """SELECT id, objective, role, output, status, created_at
                   FROM agent_runs ORDER BY created_at DESC LIMIT ?""",
                (max(0, limit),),
            ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        """Reserved for interface symmetry; SQLite connections are short-lived."""
