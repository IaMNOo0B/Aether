"""Persistent SQLite storage for conversations, memories, and task checkpoints."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
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
                CREATE INDEX IF NOT EXISTS idx_messages_created
                    ON messages(created_at);
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
                CREATE INDEX IF NOT EXISTS idx_tasks_status_updated
                    ON tasks(status, updated_at);
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
                "SELECT role, content FROM messages ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def add_memory(
        self,
        content: str,
        source: str = "user",
        status: str = "unverified",
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
                    FROM memories WHERE {clauses}
                    ORDER BY id DESC LIMIT ?""",
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
                   ON CONFLICT(id) DO UPDATE SET
                     objective=excluded.objective,
                     status=excluded.status,
                     state_json=excluded.state_json,
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

    def close(self) -> None:
        """Reserved for interface symmetry; SQLite connections are short-lived."""
