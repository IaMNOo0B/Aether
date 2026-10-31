"""Persistent SQLite storage for AETHER's operational and learning records."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import math
import sqlite3
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
