"""Persistent SQLite storage for AETHER's operational and learning records."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import os
from contextlib import closing
import tempfile
import math
import sqlite3
import time
from uuid import uuid4
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _ClosingConnection(sqlite3.Connection):
    """Connection context manager that also releases OS file handles on exit."""

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


class Store:
    def __init__(self, database_path: str | Path):
        self.path = Path(database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path, timeout=15, factory=_ClosingConnection
        )
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
                CREATE TABLE IF NOT EXISTS memory_embeddings (
                    memory_id INTEGER NOT NULL,
                    model TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    dimensions INTEGER NOT NULL CHECK(dimensions > 0),
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(memory_id, model),
                    FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_memory_embeddings_model
                    ON memory_embeddings(model, memory_id);
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
                CREATE TABLE IF NOT EXISTS knowledge_entities (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    normalized_name TEXT NOT NULL UNIQUE,
                    entity_type TEXT NOT NULL DEFAULT 'concept',
                    description TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_knowledge_entities_name
                    ON knowledge_entities(normalized_name);
                CREATE TABLE IF NOT EXISTS knowledge_claims (
                    id TEXT PRIMARY KEY,
                    subject_id TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object_value TEXT NOT NULL,
                    object_entity_id TEXT,
                    claim_type TEXT NOT NULL CHECK(claim_type IN ('fact', 'hypothesis', 'observation', 'prediction')),
                    confidence REAL CHECK(confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
                    status TEXT NOT NULL DEFAULT 'unverified'
                        CHECK(status IN ('unverified', 'verified', 'disputed', 'retracted')),
                    valid_from TEXT,
                    valid_to TEXT,
                    source_url TEXT NOT NULL DEFAULT '',
                    source_title TEXT NOT NULL DEFAULT '',
                    evidence_class TEXT NOT NULL DEFAULT 'unspecified',
                    source_report_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(subject_id) REFERENCES knowledge_entities(id),
                    FOREIGN KEY(object_entity_id) REFERENCES knowledge_entities(id)
                );
                CREATE INDEX IF NOT EXISTS idx_knowledge_claims_subject
                    ON knowledge_claims(subject_id, predicate, status);
                CREATE INDEX IF NOT EXISTS idx_knowledge_claims_report
                    ON knowledge_claims(source_report_id);
                CREATE TABLE IF NOT EXISTS knowledge_claim_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    claim_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    from_status TEXT,
                    to_status TEXT,
                    evidence TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(claim_id) REFERENCES knowledge_claims(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_knowledge_claim_events_claim
                    ON knowledge_claim_events(claim_id, sequence);
                CREATE TABLE IF NOT EXISTS research_experiments (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    hypothesis_claim_id TEXT,
                    protocol_json TEXT NOT NULL,
                    outcome_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'planned'
                        CHECK(status IN ('planned', 'running', 'completed', 'cancelled')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    FOREIGN KEY(hypothesis_claim_id) REFERENCES knowledge_claims(id)
                );
                CREATE INDEX IF NOT EXISTS idx_research_experiments_status
                    ON research_experiments(status, updated_at);
                CREATE TABLE IF NOT EXISTS research_experiment_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(experiment_id) REFERENCES research_experiments(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_research_experiment_events
                    ON research_experiment_events(experiment_id, sequence);
                CREATE TABLE IF NOT EXISTS systematic_reviews (
                    id TEXT PRIMARY KEY,
                    protocol_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'screening'
                        CHECK(status IN ('screening', 'screening_complete_synthesis_pending')),
                    final_summary_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS systematic_review_sources (
                    review_id TEXT NOT NULL,
                    source_key TEXT NOT NULL,
                    source_json TEXT NOT NULL,
                    decision TEXT NOT NULL DEFAULT 'pending'
                        CHECK(decision IN ('pending', 'include', 'exclude', 'uncertain')),
                    reviewer_decisions_json TEXT NOT NULL DEFAULT '[]',
                    reason TEXT NOT NULL DEFAULT '',
                    screened_at TEXT,
                    PRIMARY KEY(review_id, source_key),
                    FOREIGN KEY(review_id) REFERENCES systematic_reviews(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS citation_evidence (
                    id TEXT PRIMARY KEY,
                    evidence_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS worlds (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    normalized_name TEXT NOT NULL UNIQUE,
                    description TEXT NOT NULL DEFAULT '',
                    state_json TEXT NOT NULL,
                    constraints_json TEXT NOT NULL DEFAULT '{}',
                    version INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS world_state_history (
                    world_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    state_json TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(world_id, version),
                    FOREIGN KEY(world_id) REFERENCES worlds(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS world_actions (
                    id TEXT PRIMARY KEY,
                    world_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    normalized_name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    preconditions_json TEXT NOT NULL,
                    effects_json TEXT NOT NULL,
                    failure_effects_json TEXT NOT NULL DEFAULT '{}',
                    resource_costs_json TEXT NOT NULL DEFAULT '{}',
                    failure_probability REAL NOT NULL DEFAULT 0
                        CHECK(failure_probability >= 0 AND failure_probability <= 1),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(world_id, normalized_name),
                    FOREIGN KEY(world_id) REFERENCES worlds(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_world_actions_world
                    ON world_actions(world_id, normalized_name);
                CREATE TABLE IF NOT EXISTS world_simulations (
                    id TEXT PRIMARY KEY,
                    world_id TEXT NOT NULL,
                    action_id TEXT NOT NULL,
                    world_version INTEGER NOT NULL,
                    initial_state_json TEXT NOT NULL,
                    predicted_state_json TEXT NOT NULL,
                    failure_state_json TEXT NOT NULL,
                    preconditions_met INTEGER NOT NULL CHECK(preconditions_met IN (0, 1)),
                    constraints_met INTEGER NOT NULL CHECK(constraints_met IN (0, 1)),
                    failure_probability REAL NOT NULL,
                    resource_costs_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(world_id) REFERENCES worlds(id),
                    FOREIGN KEY(action_id) REFERENCES world_actions(id)
                );
                CREATE INDEX IF NOT EXISTS idx_world_simulations_world
                    ON world_simulations(world_id, created_at);
                CREATE TABLE IF NOT EXISTS world_observations (
                    id TEXT PRIMARY KEY,
                    simulation_id TEXT NOT NULL,
                    actual_state_json TEXT NOT NULL,
                    prediction_error_json TEXT NOT NULL,
                    applied_to_world INTEGER NOT NULL CHECK(applied_to_world IN (0, 1)),
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(simulation_id) REFERENCES world_simulations(id)
                );
                CREATE TABLE IF NOT EXISTS world_action_outcomes (
                    id TEXT PRIMARY KEY,
                    action_id TEXT NOT NULL,
                    simulation_id TEXT NOT NULL,
                    outcome_class TEXT NOT NULL CHECK(outcome_class IN ('success', 'failure', 'ambiguous')),
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(action_id) REFERENCES world_actions(id),
                    FOREIGN KEY(simulation_id) REFERENCES world_simulations(id)
                );
                CREATE INDEX IF NOT EXISTS idx_world_action_outcomes_action
                    ON world_action_outcomes(action_id, outcome_class);
            """)
            db.execute(
                """INSERT OR IGNORE INTO system_controls(name, enabled, reason, updated_at)
                   VALUES ('emergency_stop', 0, '', ?)""",
                (utc_now(),),
            )


    def upsert_knowledge_entity(
        self, entity_id: str, name: str, normalized_name: str,
        entity_type: str, description: str = "",
    ) -> str:
        now = utc_now()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT id FROM knowledge_entities WHERE normalized_name=?",
                (normalized_name,),
            ).fetchone()
            if row:
                db.execute(
                    """UPDATE knowledge_entities SET name=?, entity_type=?,
                       description=CASE WHEN ? != '' THEN ? ELSE description END,
                       updated_at=? WHERE id=?""",
                    (name, entity_type, description, description, now, row["id"]),
                )
                return str(row["id"])
            db.execute(
                """INSERT INTO knowledge_entities
                   (id, name, normalized_name, entity_type, description, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (entity_id, name, normalized_name, entity_type, description, now, now),
            )
        return entity_id

    def get_knowledge_entity(self, entity_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM knowledge_entities WHERE id=?", (entity_id,)
            ).fetchone()
        return dict(row) if row else None

    def find_knowledge_entity(self, normalized_name: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM knowledge_entities WHERE normalized_name=?",
                (normalized_name,),
            ).fetchone()
        return dict(row) if row else None

    def search_knowledge_entities(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        text = query.strip().casefold()
        if not text or limit < 1:
            return []
        with self._connect() as db:
            rows = db.execute(
                """SELECT * FROM knowledge_entities
                   WHERE normalized_name LIKE ? OR description LIKE ?
                   ORDER BY name LIMIT ?""",
                (f"%{text}%", f"%{text}%", min(limit, 100)),
            ).fetchall()
        return [dict(row) for row in rows]

    def insert_knowledge_claim(self, claim: dict[str, Any]) -> None:
        now = utc_now()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                """INSERT INTO knowledge_claims
                   (id, subject_id, predicate, object_value, object_entity_id, claim_type,
                    confidence, status, valid_from, valid_to, source_url, source_title,
                    evidence_class, source_report_id, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    claim["id"], claim["subject_id"], claim["predicate"],
                    claim["object_value"], claim.get("object_entity_id"),
                    claim["claim_type"], claim.get("confidence"), claim.get("status", "unverified"),
                    claim.get("valid_from"), claim.get("valid_to"),
                    claim.get("source_url", ""), claim.get("source_title", ""),
                    claim.get("evidence_class", "unspecified"), claim.get("source_report_id"),
                    now, now,
                ),
            )
            db.execute(
                """INSERT INTO knowledge_claim_events
                   (claim_id, event_type, from_status, to_status, evidence, created_at)
                   VALUES (?, 'created', NULL, 'unverified', ?, ?)""",
                (claim["id"], claim.get("creation_note", "Claim captured; not yet verified."), now),
            )

    def get_knowledge_claim(self, claim_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                """SELECT c.*, s.name AS subject_name, o.name AS object_entity_name
                   FROM knowledge_claims c
                   JOIN knowledge_entities s ON s.id=c.subject_id
                   LEFT JOIN knowledge_entities o ON o.id=c.object_entity_id
                   WHERE c.id=?""",
                (claim_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_knowledge_claims(
        self, subject_id: str | None = None, predicate: str | None = None,
        source_report_id: str | None = None, limit: int = 500,
    ) -> list[dict[str, Any]]:
        clauses = []
        parameters: list[Any] = []
        if subject_id is not None:
            clauses.append("c.subject_id=?")
            parameters.append(subject_id)
        if predicate is not None:
            clauses.append("c.predicate=?")
            parameters.append(predicate)
        if source_report_id is not None:
            clauses.append("c.source_report_id=?")
            parameters.append(source_report_id)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        parameters.append(max(0, min(limit, 5000)))
        with self._connect() as db:
            rows = db.execute(
                """SELECT c.*, s.name AS subject_name, o.name AS object_entity_name
                   FROM knowledge_claims c
                   JOIN knowledge_entities s ON s.id=c.subject_id
                   LEFT JOIN knowledge_entities o ON o.id=c.object_entity_id"""
                + where + " ORDER BY c.created_at DESC LIMIT ?",
                tuple(parameters),
            ).fetchall()
        return [dict(row) for row in rows]

    def update_knowledge_claim_status(
        self, claim_id: str, status: str, evidence: str,
    ) -> None:
        now = utc_now()
        if status not in {"verified", "disputed", "retracted"}:
            raise ValueError("Claim status must be verified, disputed or retracted")
        if not evidence.strip():
            raise ValueError("Claim review requires evidence")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT status FROM knowledge_claims WHERE id=?", (claim_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown knowledge claim: {claim_id}")
            db.execute(
                "UPDATE knowledge_claims SET status=?, updated_at=? WHERE id=?",
                (status, now, claim_id),
            )
            db.execute(
                """INSERT INTO knowledge_claim_events
                   (claim_id, event_type, from_status, to_status, evidence, created_at)
                   VALUES (?, 'reviewed', ?, ?, ?, ?)""",
                (claim_id, row["status"], status, evidence, now),
            )

    def list_knowledge_claim_events(self, claim_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                """SELECT sequence, event_type, from_status, to_status, evidence, created_at
                   FROM knowledge_claim_events WHERE claim_id=? ORDER BY sequence""",
                (claim_id,),
            ).fetchall()
        return [dict(row) for row in rows]


    def create_world(
        self, world_id: str, name: str, normalized_name: str,
        description: str, state: dict[str, Any], constraints: dict[str, Any],
    ) -> None:
        now = utc_now()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                """INSERT INTO worlds
                   (id, name, normalized_name, description, state_json,
                    constraints_json, version, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)""",
                (world_id, name, normalized_name, description, json.dumps(state),
                 json.dumps(constraints), now, now),
            )
            db.execute(
                """INSERT INTO world_state_history(world_id, version, state_json, source, created_at)
                   VALUES (?, 1, ?, 'initial_state', ?)""",
                (world_id, json.dumps(state), now),
            )

    def get_world(self, world_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM worlds WHERE id=?", (world_id,)).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["state"] = json.loads(item.pop("state_json"))
        item["constraints"] = json.loads(item.pop("constraints_json"))
        return item

    def find_world(self, normalized_name: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT id FROM worlds WHERE normalized_name=?", (normalized_name,)
            ).fetchone()
        return self.get_world(row["id"]) if row else None

    def list_worlds(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT id FROM worlds ORDER BY name LIMIT ?", (max(0, min(limit, 500)),)
            ).fetchall()
        return [world for row in rows if (world := self.get_world(row["id"])) is not None]

    def update_world_state(
        self, world_id: str, state: dict[str, Any], source: str,
    ) -> int:
        now = utc_now()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT version FROM worlds WHERE id=?", (world_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown world: {world_id}")
            version = int(row["version"]) + 1
            db.execute(
                "UPDATE worlds SET state_json=?, version=?, updated_at=? WHERE id=?",
                (json.dumps(state), version, now, world_id),
            )
            db.execute(
                """INSERT INTO world_state_history(world_id, version, state_json, source, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (world_id, version, json.dumps(state), source, now),
            )
        return version

    def list_world_state_history(self, world_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                """SELECT version, state_json, source, created_at
                   FROM world_state_history WHERE world_id=? ORDER BY version""",
                (world_id,),
            ).fetchall()
        return [
            {"version": row["version"], "state": json.loads(row["state_json"]),
             "source": row["source"], "created_at": row["created_at"]}
            for row in rows
        ]

    def insert_world_action(self, action: dict[str, Any]) -> None:
        now = utc_now()
        with self._connect() as db:
            db.execute(
                """INSERT INTO world_actions
                   (id, world_id, name, normalized_name, description, preconditions_json,
                    effects_json, failure_effects_json, resource_costs_json,
                    failure_probability, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    action["id"], action["world_id"], action["name"], action["normalized_name"],
                    action.get("description", ""), json.dumps(action["preconditions"]),
                    json.dumps(action["effects"]), json.dumps(action.get("failure_effects", {})),
                    json.dumps(action.get("resource_costs", {})),
                    action.get("failure_probability", 0.0), now, now,
                ),
            )

    def get_world_action(self, action_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM world_actions WHERE id=?", (action_id,)).fetchone()
        if row is None:
            return None
        item = dict(row)
        for key in ("preconditions_json", "effects_json", "failure_effects_json", "resource_costs_json"):
            item[key.removesuffix("_json")] = json.loads(item.pop(key))
        return item

    def list_world_actions(self, world_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT id FROM world_actions WHERE world_id=? ORDER BY name", (world_id,)
            ).fetchall()
        return [action for row in rows if (action := self.get_world_action(row["id"])) is not None]

    def insert_world_simulation(self, simulation: dict[str, Any]) -> None:
        with self._connect() as db:
            db.execute(
                """INSERT INTO world_simulations
                   (id, world_id, action_id, world_version, initial_state_json,
                    predicted_state_json, failure_state_json, preconditions_met,
                    constraints_met, failure_probability, resource_costs_json, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    simulation["id"], simulation["world_id"], simulation["action_id"],
                    simulation["world_version"], json.dumps(simulation["initial_state"]),
                    json.dumps(simulation["predicted_state"]), json.dumps(simulation["failure_state"]),
                    int(simulation["preconditions_met"]), int(simulation["constraints_met"]),
                    simulation["failure_probability"], json.dumps(simulation["resource_costs"]),
                    utc_now(),
                ),
            )

    def get_world_simulation(self, simulation_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM world_simulations WHERE id=?", (simulation_id,)
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        for key in ("initial_state_json", "predicted_state_json", "failure_state_json", "resource_costs_json"):
            item[key.removesuffix("_json")] = json.loads(item.pop(key))
        item["preconditions_met"] = bool(item["preconditions_met"])
        item["constraints_met"] = bool(item["constraints_met"])
        return item

    def insert_world_observation(
        self, observation: dict[str, Any], action_id: str, outcome_class: str,
    ) -> None:
        now = utc_now()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                """INSERT INTO world_observations
                   (id, simulation_id, actual_state_json, prediction_error_json,
                    applied_to_world, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    observation["id"], observation["simulation_id"],
                    json.dumps(observation["actual_state"]),
                    json.dumps(observation["prediction_error"]),
                    int(observation["applied_to_world"]), now,
                ),
            )
            db.execute(
                """INSERT INTO world_action_outcomes
                   (id, action_id, simulation_id, outcome_class, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (uuid4().hex, action_id, observation["simulation_id"], outcome_class, now),
            )

    def world_action_outcome_counts(self, action_id: str) -> dict[str, int]:
        with self._connect() as db:
            rows = db.execute(
                """SELECT outcome_class, COUNT(*) AS count FROM world_action_outcomes
                   WHERE action_id=? GROUP BY outcome_class""",
                (action_id,),
            ).fetchall()
        counts = {"success": 0, "failure": 0, "ambiguous": 0}
        counts.update({row["outcome_class"]: int(row["count"]) for row in rows})
        return counts


    def apply_world_observation(
        self, observation: dict[str, Any], world_id: str,
        expected_version: int, actual_state: dict[str, Any],
        action_id: str, outcome_class: str,
    ) -> int:
        """Atomically append an observation and update only the exact simulated world version."""
        now = utc_now()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT version FROM worlds WHERE id=?", (world_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown world: {world_id}")
            if int(row["version"]) != expected_version:
                raise ValueError(
                    "World changed after this simulation; observation was not applied to the newer state"
                )
            db.execute(
                """INSERT INTO world_observations
                   (id, simulation_id, actual_state_json, prediction_error_json,
                    applied_to_world, created_at)
                   VALUES (?, ?, ?, ?, 1, ?)""",
                (
                    observation["id"], observation["simulation_id"],
                    json.dumps(observation["actual_state"]),
                    json.dumps(observation["prediction_error"]), now,
                ),
            )
            db.execute(
                """INSERT INTO world_action_outcomes
                   (id, action_id, simulation_id, outcome_class, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (uuid4().hex, action_id, observation["simulation_id"], outcome_class, now),
            )
            next_version = expected_version + 1
            db.execute(
                "UPDATE worlds SET state_json=?, version=?, updated_at=? WHERE id=?",
                (json.dumps(actual_state), next_version, now, world_id),
            )
            db.execute(
                """INSERT INTO world_state_history(world_id, version, state_json, source, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    world_id, next_version, json.dumps(actual_state),
                    f"observed_after_simulation:{observation['simulation_id']}", now,
                ),
            )
        return next_version

    def list_world_observations(self, simulation_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                """SELECT id, actual_state_json, prediction_error_json,
                          applied_to_world, created_at
                   FROM world_observations WHERE simulation_id=? ORDER BY created_at""",
                (simulation_id,),
            ).fetchall()
        return [
            {"id": row["id"], "actual_state": json.loads(row["actual_state_json"]),
             "prediction_error": json.loads(row["prediction_error_json"]),
             "applied_to_world": bool(row["applied_to_world"]), "created_at": row["created_at"]}
            for row in rows
        ]

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

    def backup_to(self, destination: str | Path) -> Path:
        """Create a consistent SQLite snapshot and atomically publish it."""
        target = Path(destination).expanduser()
        if target.resolve() == self.path.resolve():
            raise ValueError("Backup destination cannot be the live database")
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
        )
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            with closing(sqlite3.connect(self.path)) as source, closing(sqlite3.connect(temporary)) as backup:
                source.backup(backup)
                result = backup.execute("PRAGMA integrity_check").fetchone()
                if not result or result[0] != "ok":
                    raise RuntimeError("SQLite backup failed its integrity check")
            os.replace(temporary, target)
            return target
        finally:
            temporary.unlink(missing_ok=True)

    def restore_from(self, source_path: str | Path) -> None:
        """Validate a backup, stage it beside the live DB, then atomically replace it.

        Stop other AETHER processes before restoring. The method never accepts the
        live database itself as its source and leaves the current DB untouched if
        validation or staging fails.
        """
        source = Path(source_path).expanduser()
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.resolve() == self.path.resolve():
            raise ValueError("Restore source cannot be the live database")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.restore.", suffix=".tmp", dir=self.path.parent
        )
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            with closing(sqlite3.connect(source)) as candidate:
                check = candidate.execute("PRAGMA integrity_check").fetchone()
                if not check or check[0] != "ok":
                    raise ValueError("Restore source failed SQLite integrity_check")
                tables = {
                    row[0] for row in candidate.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                required = {"messages", "memories", "tasks", "skills"}
                if not required.issubset(tables):
                    raise ValueError("Restore source is not a compatible AETHER database")
                with closing(sqlite3.connect(temporary)) as staged:
                    candidate.backup(staged)
                    staged_check = staged.execute("PRAGMA integrity_check").fetchone()
                    if not staged_check or staged_check[0] != "ok":
                        raise ValueError("Staged restore failed SQLite integrity_check")
            live = self._connect()
            try:
                checkpoint = live.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
                if checkpoint and checkpoint[0] != 0:
                    raise RuntimeError("Could not checkpoint the live database; stop other AETHER processes")
            finally:
                live.close()
            os.replace(temporary, self.path)
            Path(str(self.path) + "-wal").unlink(missing_ok=True)
            Path(str(self.path) + "-shm").unlink(missing_ok=True)
        finally:
            temporary.unlink(missing_ok=True)

    def list_memory_versions(self, memory_id: int) -> list[dict[str, Any]]:
        """Return a validated, ordered correction chain containing memory_id."""
        records = {item["id"]: item for item in self.list_memories(20000)}
        if memory_id not in records:
            raise KeyError(f"Unknown memory: {memory_id}")
        current = records[memory_id]
        backward_seen: set[int] = set()
        while current["metadata"].get("supersedes_memory_id") is not None:
            if current["id"] in backward_seen:
                raise RuntimeError("Memory correction history contains a cycle")
            backward_seen.add(current["id"])
            parent_id = current["metadata"]["supersedes_memory_id"]
            parent = records.get(parent_id)
            if parent is None or parent["metadata"].get("superseded_by_id") != current["id"]:
                raise RuntimeError("Memory correction history has a broken predecessor link")
            current = parent
        lineage = []
        visited: set[int] = set()
        while True:
            if current["id"] in visited:
                raise RuntimeError("Memory correction history contains a cycle")
            visited.add(current["id"])
            lineage.append(current)
            child_id = current["metadata"].get("superseded_by_id")
            if child_id is None:
                break
            child = records.get(child_id)
            if child is None or child["metadata"].get("supersedes_memory_id") != current["id"]:
                raise RuntimeError("Memory correction history has a broken successor link")
            current = child
        return lineage

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



    def save_memory_embedding(
        self, memory_id: int, model: str, vector: list[float],
    ) -> None:
        if not model.strip() or not vector:
            raise ValueError("Embedding model and vector are required")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in vector
        ):
            raise ValueError("Embedding vectors must contain finite numeric values")
        with self._connect() as db:
            db.execute(
                """INSERT INTO memory_embeddings(memory_id, model, vector_json, dimensions, created_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(memory_id, model) DO UPDATE SET
                     vector_json=excluded.vector_json,
                     dimensions=excluded.dimensions,
                     created_at=excluded.created_at""",
                (memory_id, model, json.dumps(vector), len(vector), utc_now()),
            )

    def get_memory_embedding(self, memory_id: int, model: str) -> list[float] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT vector_json FROM memory_embeddings WHERE memory_id=? AND model=?",
                (memory_id, model),
            ).fetchone()
        return json.loads(row["vector_json"]) if row else None

    def list_memory_embeddings(self, model: str, limit: int = 20000) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                """SELECT e.memory_id, e.vector_json, m.content, m.source, m.status,
                          m.metadata_json, m.created_at
                   FROM memory_embeddings e
                   JOIN memories m ON m.id=e.memory_id
                   WHERE e.model=? AND m.status NOT IN ('superseded', 'rejected', 'deleted')
                   ORDER BY e.memory_id DESC LIMIT ?""",
                (model, max(0, min(limit, 20000))),
            ).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item["vector"] = json.loads(item.pop("vector_json"))
            item["metadata"] = json.loads(item.pop("metadata_json"))
            results.append(item)
        return results

    def list_memories_without_embedding(
        self, model: str, limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                """SELECT m.id, m.content, m.source, m.status, m.metadata_json, m.created_at
                   FROM memories m LEFT JOIN memory_embeddings e
                     ON e.memory_id=m.id AND e.model=?
                   WHERE e.memory_id IS NULL
                     AND m.status NOT IN ('superseded', 'rejected', 'deleted')
                   ORDER BY m.id LIMIT ?""",
                (model, max(0, min(limit, 1000))),
            ).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json"))
            results.append(item)
        return results

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
            if row["status"] in {"rejected", "deleted"}:
                raise ValueError("Rejected or deleted memories cannot be corrected")
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
            row = db.execute("SELECT status, state_json FROM tasks WHERE id=?", (task_id,)).fetchone()
            if row is None:
                raise KeyError(f"Unknown task: {task_id}")
            if row["status"] != "running":
                raise ValueError(f"Only running tasks can be checkpointed, found {row['status']}")
            payload = dict(state)
            existing = json.loads(row["state_json"])
            if existing.get("pause_requested"):
                payload["pause_requested"] = True
            db.execute(
                "UPDATE tasks SET state_json=?, updated_at=? WHERE id=?",
                (json.dumps(payload), now, task_id),
            )
            db.execute(
                """INSERT INTO task_events(task_id, event_type, from_status, to_status,
                   payload_json, created_at) VALUES (?, 'checkpoint', 'running', 'running', ?, ?)""",
                (task_id, json.dumps(event_data or state), now),
            )
        return self.get_task(task_id)  # type: ignore[return-value]

    def request_task_pause(self, task_id: str) -> dict[str, Any]:
        """Persist a cooperative pause request without interrupting an active tool."""
        now = utc_now()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT status, state_json FROM tasks WHERE id=?", (task_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown task: {task_id}")
            if row["status"] != "running":
                raise ValueError(f"Only running tasks can be paused, found {row['status']}")
            state = json.loads(row["state_json"])
            state["pause_requested"] = True
            db.execute(
                "UPDATE tasks SET state_json=?, updated_at=? WHERE id=?",
                (json.dumps(state), now, task_id),
            )
            db.execute(
                """INSERT INTO task_events(task_id, event_type, from_status, to_status,
                   payload_json, created_at) VALUES (?, 'pause_requested', 'running', 'running', ?, ?)""",
                (task_id, json.dumps({"requested": True}), now),
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


    def create_research_experiment(
        self, experiment_id: str, title: str, hypothesis_claim_id: str | None,
        protocol: dict[str, Any],
    ) -> dict[str, Any]:
        clean_title = title.strip()
        if not clean_title or len(clean_title) > 300:
            raise ValueError("Experiment title must contain 1..300 characters")
        if not isinstance(protocol, dict) or not protocol:
            raise ValueError("Experiment protocol must be a non-empty JSON object")
        required = {"objective", "method", "success_criteria"}
        if not required.issubset(protocol) or any(
            not isinstance(protocol.get(key), str) or not protocol[key].strip()
            for key in required
        ):
            raise ValueError("Protocol requires non-empty objective, method and success_criteria")
        if hypothesis_claim_id:
            claim = self.get_knowledge_claim(hypothesis_claim_id)
            if claim is None or claim.get("claim_type") != "hypothesis":
                raise ValueError("hypothesis_claim_id must identify an existing hypothesis claim")
        now = utc_now()
        with self._connect() as db:
            db.execute(
                """INSERT INTO research_experiments
                   (id, title, hypothesis_claim_id, protocol_json, outcome_json,
                    status, created_at, updated_at)
                   VALUES (?, ?, ?, ?, '{}', 'planned', ?, ?)""",
                (experiment_id, clean_title, hypothesis_claim_id,
                 json.dumps(protocol), now, now),
            )
            db.execute(
                """INSERT INTO research_experiment_events
                   (experiment_id, event_type, payload_json, created_at)
                   VALUES (?, 'created', ?, ?)""",
                (experiment_id, json.dumps({"protocol_locked": True}), now),
            )
        return self.get_research_experiment(experiment_id)

    def get_research_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM research_experiments WHERE id=?", (experiment_id,)
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["protocol"] = json.loads(item.pop("protocol_json"))
        item["outcome"] = json.loads(item.pop("outcome_json"))
        return item

    def list_research_experiments(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT id FROM research_experiments ORDER BY updated_at DESC LIMIT ?",
                (max(0, min(limit, 500)),),
            ).fetchall()
        return [
            item for row in rows
            if (item := self.get_research_experiment(row["id"])) is not None
        ]

    def transition_research_experiment(
        self, experiment_id: str, target_status: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        allowed = {
            "planned": {"running", "cancelled"},
            "running": {"completed", "cancelled"},
            "completed": set(),
            "cancelled": set(),
        }
        now = utc_now()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT status FROM research_experiments WHERE id=?", (experiment_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown research experiment: {experiment_id}")
            current = str(row["status"])
            if target_status not in allowed.get(current, set()):
                raise ValueError(
                    f"Invalid experiment transition: {current} -> {target_status}"
                )
            details = payload or {}
            if target_status == "completed":
                if not details or not isinstance(details.get("result"), str) or not details["result"].strip():
                    raise ValueError("Completing an experiment requires a non-empty result")
                if not isinstance(details.get("evidence"), str) or not details["evidence"].strip():
                    raise ValueError("Completing an experiment requires evidence")
                db.execute(
                    """UPDATE research_experiments SET status=?, outcome_json=?,
                       completed_at=?, updated_at=? WHERE id=?""",
                    (target_status, json.dumps(details), now, now, experiment_id),
                )
            elif target_status == "running":
                db.execute(
                    """UPDATE research_experiments SET status=?, started_at=?,
                       updated_at=? WHERE id=?""",
                    (target_status, now, now, experiment_id),
                )
            else:
                db.execute(
                    "UPDATE research_experiments SET status=?, updated_at=? WHERE id=?",
                    (target_status, now, experiment_id),
                )
            db.execute(
                """INSERT INTO research_experiment_events
                   (experiment_id, event_type, payload_json, created_at)
                   VALUES (?, ?, ?, ?)""",
                (experiment_id, target_status, json.dumps(details), now),
            )
        return self.get_research_experiment(experiment_id)

    def list_research_experiment_events(self, experiment_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                """SELECT sequence, experiment_id, event_type, payload_json, created_at
                   FROM research_experiment_events
                   WHERE experiment_id=? ORDER BY sequence""",
                (experiment_id,),
            ).fetchall()
        return [
            {**dict(row), "payload": json.loads(row["payload_json"])}
            for row in rows
        ]


    def create_systematic_review(self, review_id: str, protocol: dict[str, Any]) -> None:
        now = utc_now()
        with self._connect() as db:
            db.execute(
                """INSERT INTO systematic_reviews
                   (id, protocol_json, status, final_summary_json, created_at, updated_at)
                   VALUES (?, ?, 'screening', '{}', ?, ?)""",
                (review_id, json.dumps(protocol), now, now),
            )

    def get_systematic_review(self, review_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM systematic_reviews WHERE id=?", (review_id,)
            ).fetchone()
            if row is None:
                return None
            sources = db.execute(
                """SELECT source_key, source_json, decision, reviewer_decisions_json, reason, screened_at
                   FROM systematic_review_sources WHERE review_id=? ORDER BY source_key""",
                (review_id,),
            ).fetchall()
        item = dict(row)
        item["protocol"] = json.loads(item.pop("protocol_json"))
        item["final_summary"] = json.loads(item.pop("final_summary_json"))
        item["sources"] = [
            {
                **dict(source),
                "source": json.loads(source["source_json"]),
                "reviewer_decisions": json.loads(source["reviewer_decisions_json"]),
            }
            for source in sources
        ]
        for source in item["sources"]:
            source.pop("source_json", None)
            source.pop("reviewer_decisions_json", None)
        return item

    def add_systematic_review_source(
        self, review_id: str, source_key: str, source: dict[str, Any],
    ) -> None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            review = db.execute(
                "SELECT status FROM systematic_reviews WHERE id=?", (review_id,)
            ).fetchone()
            if review is None:
                raise KeyError(f"Unknown systematic review: {review_id}")
            if review["status"] != "screening":
                raise ValueError("Cannot add sources after screening is finalized")
            db.execute(
                """INSERT OR IGNORE INTO systematic_review_sources
                   (review_id, source_key, source_json, decision, reason)
                   VALUES (?, ?, ?, 'pending', '')""",
                (review_id, source_key, json.dumps(source)),
            )
            db.execute(
                "UPDATE systematic_reviews SET updated_at=? WHERE id=?",
                (utc_now(), review_id),
            )


    def screen_systematic_review_source(
        self, review_id: str, source_key: str, reviewer_id: str,
        decision: str, reason: str,
    ) -> None:
        reviewer = reviewer_id.strip()
        if not reviewer or len(reviewer) > 120:
            raise ValueError("Reviewer ID must contain 1..120 characters")
        if decision not in {"include", "exclude"}:
            raise ValueError("Reviewer decision must be include or exclude")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            review = db.execute(
                "SELECT status FROM systematic_reviews WHERE id=?", (review_id,)
            ).fetchone()
            if review is None:
                raise KeyError(f"Unknown systematic review: {review_id}")
            if review["status"] != "screening":
                raise ValueError("Cannot change screening decisions after finalization")
            row = db.execute(
                """SELECT reviewer_decisions_json FROM systematic_review_sources
                   WHERE review_id=? AND source_key=?""",
                (review_id, source_key),
            ).fetchone()
            if row is None:
                raise KeyError("Source is not part of this systematic review")
            decisions = json.loads(row["reviewer_decisions_json"])
            if any(item["reviewer_id"] == reviewer for item in decisions):
                raise ValueError("This reviewer has already screened this source")
            decisions.append({
                "reviewer_id": reviewer, "decision": decision,
                "reason": reason, "screened_at": utc_now(),
            })
            distinct = {item["decision"] for item in decisions}
            consensus = "pending" if len(decisions) < 2 else (
                next(iter(distinct)) if len(distinct) == 1 else "uncertain"
            )
            db.execute(
                """UPDATE systematic_review_sources
                   SET reviewer_decisions_json=?, decision=?, reason=?, screened_at=?
                   WHERE review_id=? AND source_key=?""",
                (json.dumps(decisions), consensus, reason, utc_now(), review_id, source_key),
            )
            db.execute(
                "UPDATE systematic_reviews SET updated_at=? WHERE id=?",
                (utc_now(), review_id),
            )

    def resolve_systematic_review_conflict(
        self, review_id: str, source_key: str, adjudicator_id: str,
        decision: str, rationale: str,
    ) -> None:
        adjudicator = adjudicator_id.strip()
        if not adjudicator or len(adjudicator) > 120:
            raise ValueError("Adjudicator ID must contain 1..120 characters")
        if decision not in {"include", "exclude"} or len(rationale.strip()) < 12:
            raise ValueError("Resolution requires include/exclude and a rationale of at least 12 characters")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                """SELECT r.status, s.reviewer_decisions_json, s.decision
                   FROM systematic_reviews r JOIN systematic_review_sources s
                   ON s.review_id=r.id WHERE r.id=? AND s.source_key=?""",
                (review_id, source_key),
            ).fetchone()
            if row is None:
                raise KeyError("Review source not found")
            if row["status"] != "screening" or row["decision"] != "uncertain":
                raise ValueError("Only unresolved reviewer disagreements can be adjudicated")
            decisions = json.loads(row["reviewer_decisions_json"])
            if any(item["reviewer_id"] == adjudicator for item in decisions):
                raise ValueError("Adjudicator must be distinct from screening reviewers")
            decisions.append({
                "reviewer_id": adjudicator, "decision": decision,
                "reason": rationale.strip(), "role": "adjudicator",
                "screened_at": utc_now(),
            })
            db.execute(
                """UPDATE systematic_review_sources
                   SET reviewer_decisions_json=?, decision=?, reason=?, screened_at=?
                   WHERE review_id=? AND source_key=?""",
                (json.dumps(decisions), decision, rationale.strip(), utc_now(), review_id, source_key),
            )

    def finalize_systematic_review(self, review_id: str, summary: dict[str, Any]) -> None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT status FROM systematic_reviews WHERE id=?", (review_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown systematic review: {review_id}")
            if row["status"] != "screening":
                raise ValueError("Systematic review screening is already finalized")
            pending = db.execute(
                """SELECT COUNT(*) AS n FROM systematic_review_sources
                   WHERE review_id=? AND decision IN ('pending', 'uncertain')""",
                (review_id,),
            ).fetchone()["n"]
            if pending:
                raise ValueError("All sources must have include/exclude decisions")
            db.execute(
                """UPDATE systematic_reviews SET status='screening_complete_synthesis_pending',
                   final_summary_json=?, updated_at=? WHERE id=?""",
                (json.dumps(summary), utc_now(), review_id),
            )

    def save_citation_evidence(self, evidence: dict[str, Any]) -> None:
        now = utc_now()
        with self._connect() as db:
            db.execute(
                """INSERT INTO citation_evidence(id, evidence_json, created_at, updated_at)
                   VALUES (?, ?, ?, ?)""",
                (evidence["id"], json.dumps(evidence), now, now),
            )

    def get_citation_evidence(self, evidence_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT evidence_json FROM citation_evidence WHERE id=?", (evidence_id,)
            ).fetchone()
        return json.loads(row["evidence_json"]) if row else None

    def adjudicate_citation_evidence(
        self, evidence_id: str, decision: str, rationale: str,
    ) -> None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT evidence_json FROM citation_evidence WHERE id=?", (evidence_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown citation evidence: {evidence_id}")
            evidence = json.loads(row["evidence_json"])
            if evidence.get("human_decision") is not None:
                raise ValueError("Citation evidence has already been adjudicated")
            evidence["human_decision"] = decision
            evidence["human_rationale"] = rationale
            evidence["human_adjudicated_at"] = utc_now()
            db.execute(
                "UPDATE citation_evidence SET evidence_json=?, updated_at=? WHERE id=?",
                (json.dumps(evidence), utc_now(), evidence_id),
            )

    def append_research_experiment_event(
        self, experiment_id: str, event_type: str, payload: dict[str, Any],
    ) -> None:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            exists = db.execute(
                "SELECT 1 FROM research_experiments WHERE id=?", (experiment_id,)
            ).fetchone()
            if exists is None:
                raise KeyError(f"Unknown research experiment: {experiment_id}")
            db.execute(
                """INSERT INTO research_experiment_events
                   (experiment_id, event_type, payload_json, created_at)
                   VALUES (?, ?, ?, ?)""",
                (experiment_id, event_type, json.dumps(payload), utc_now()),
            )

    def close(self) -> None:
        """Reserved for interface symmetry; SQLite connections are short-lived."""
