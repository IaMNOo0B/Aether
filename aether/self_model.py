"""Persistent, versioned operational self-model and evidence-backed capabilities.

This is an operational record, not a claim of consciousness or general intelligence.
A capability is never marked verified merely because its module exists.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
import json
import sqlite3


CAPABILITY_STATES = {"planned", "implemented_unverified", "verified", "degraded", "unavailable"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SelfModel:
    def __init__(self, database_path: str | Path, name: str = "AETHER", version: str = "0.3.0"):
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.version = version
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS self_model (
                    id INTEGER PRIMARY KEY CHECK(id = 1),
                    identity_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    objectives_json TEXT NOT NULL DEFAULT '[]',
                    limitations_json TEXT NOT NULL DEFAULT '[]',
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS self_model_history (
                    revision INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS capabilities (
                    capability_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    state TEXT NOT NULL,
                    description TEXT NOT NULL,
                    evidence_json TEXT NOT NULL DEFAULT '[]',
                    limitations_json TEXT NOT NULL DEFAULT '[]',
                    updated_at TEXT NOT NULL
                );
            """)
            row = db.execute("SELECT id FROM self_model WHERE id=1").fetchone()
            if row is None:
                snapshot = {
                    "identity_id": uuid4().hex,
                    "name": self.name,
                    "version": self.version,
                    "objectives": [],
                    "limitations": ["Capability status requires explicit test evidence."],
                }
                db.execute(
                    """INSERT INTO self_model
                       (id, identity_id, name, version, objectives_json, limitations_json, updated_at)
                       VALUES (1, ?, ?, ?, ?, ?, ?)""",
                    (snapshot["identity_id"], snapshot["name"], snapshot["version"],
                     json.dumps(snapshot["objectives"]), json.dumps(snapshot["limitations"]), _now()),
                )
                self._record_snapshot(db, snapshot)

    @staticmethod
    def _record_snapshot(db: sqlite3.Connection, snapshot: dict[str, Any]) -> None:
        db.execute(
            "INSERT INTO self_model_history(snapshot_json, created_at) VALUES (?, ?)",
            (json.dumps(snapshot, sort_keys=True), _now()),
        )

    def snapshot(self) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM self_model WHERE id=1").fetchone()
            capabilities = db.execute(
                "SELECT capability_id, name, state, description, evidence_json, limitations_json, updated_at "
                "FROM capabilities ORDER BY name"
            ).fetchall()
        if row is None:
            raise RuntimeError("Self-model was not initialized")
        return {
            "identity_id": row["identity_id"],
            "name": row["name"],
            "version": row["version"],
            "objectives": json.loads(row["objectives_json"]),
            "limitations": json.loads(row["limitations_json"]),
            "updated_at": row["updated_at"],
            "capabilities": [
                {
                    "id": item["capability_id"],
                    "name": item["name"],
                    "state": item["state"],
                    "description": item["description"],
                    "evidence": json.loads(item["evidence_json"]),
                    "limitations": json.loads(item["limitations_json"]),
                    "updated_at": item["updated_at"],
                }
                for item in capabilities
            ],
        }

    def update_profile(
        self, *, version: str | None = None, objectives: list[str] | None = None,
        limitations: list[str] | None = None,
    ) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM self_model WHERE id=1").fetchone()
            next_version = version if version is not None else row["version"]
            next_objectives = objectives if objectives is not None else json.loads(row["objectives_json"])
            next_limitations = limitations if limitations is not None else json.loads(row["limitations_json"])
            if not isinstance(next_version, str) or not next_version.strip():
                raise ValueError("Version cannot be empty")
            if not all(isinstance(x, str) and x.strip() for x in next_objectives + next_limitations):
                raise ValueError("Objectives and limitations must be non-empty strings")
            db.execute(
                """UPDATE self_model SET version=?, objectives_json=?, limitations_json=?, updated_at=?
                   WHERE id=1""",
                (next_version, json.dumps(next_objectives), json.dumps(next_limitations), _now()),
            )
            snapshot = {
                "identity_id": row["identity_id"], "name": row["name"], "version": next_version,
                "objectives": next_objectives, "limitations": next_limitations,
            }
            self._record_snapshot(db, snapshot)
        return self.snapshot()

    def register_capability(
        self, capability_id: str, name: str, description: str,
        state: str = "planned", limitations: list[str] | None = None,
    ) -> dict[str, Any]:
        if state not in CAPABILITY_STATES:
            raise ValueError(f"Unsupported capability state: {state}")
        if not all(isinstance(x, str) and x.strip() for x in (capability_id, name, description)):
            raise ValueError("Capability ID, name, and description are required")
        with self._connect() as db:
            db.execute(
                """INSERT INTO capabilities
                   (capability_id, name, state, description, evidence_json, limitations_json, updated_at)
                   VALUES (?, ?, ?, ?, '[]', ?, ?)
                   ON CONFLICT(capability_id) DO UPDATE SET name=excluded.name,
                     description=excluded.description, updated_at=excluded.updated_at""",
                (capability_id, name, state, description, json.dumps(limitations or []), _now()),
            )
        return self.get_capability(capability_id)

    def record_capability_result(
        self, capability_id: str, state: str, evidence: str,
        limitations: list[str] | None = None,
    ) -> dict[str, Any]:
        if state not in CAPABILITY_STATES:
            raise ValueError(f"Unsupported capability state: {state}")
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError("A capability result requires concrete evidence")
        with self._connect() as db:
            row = db.execute(
                "SELECT evidence_json FROM capabilities WHERE capability_id=?",
                (capability_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown capability: {capability_id}")
            history = json.loads(row["evidence_json"])
            history.append({"state": state, "evidence": evidence, "at": _now()})
            db.execute(
                """UPDATE capabilities SET state=?, evidence_json=?, limitations_json=?, updated_at=?
                   WHERE capability_id=?""",
                (state, json.dumps(history), json.dumps(limitations or []), _now(), capability_id),
            )
        return self.get_capability(capability_id)

    def get_capability(self, capability_id: str) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM capabilities WHERE capability_id=?", (capability_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown capability: {capability_id}")
        return {
            "id": row["capability_id"], "name": row["name"], "state": row["state"],
            "description": row["description"], "evidence": json.loads(row["evidence_json"]),
            "limitations": json.loads(row["limitations_json"]), "updated_at": row["updated_at"],
        }

    def history(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT revision, snapshot_json, created_at FROM self_model_history ORDER BY revision DESC LIMIT ?",
                (max(0, limit),),
            ).fetchall()
        return [
            {"revision": row["revision"], "snapshot": json.loads(row["snapshot_json"]),
             "created_at": row["created_at"]}
            for row in reversed(rows)
        ]
