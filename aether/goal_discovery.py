"""Evidence-bounded discovery of autonomy goal proposals.

Discovery is deliberately separate from execution: this module can create and
rank proposals, but it cannot approve, enqueue, or execute them.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
from typing import Any
from uuid import uuid4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean(value: str, limit: int = 1000) -> str:
    return " ".join(str(value).split())[:limit]


class GoalDiscovery:
    """Turn observable project/runtime events into reviewable goal proposals."""

    def __init__(
        self, database_path: str | Path, project_root: str | Path = ".",
        max_proposals: int = 20,
    ):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.project_root = Path(project_root).resolve()
        self.max_proposals = max(1, min(int(max_proposals), 100))
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.database_path, timeout=15)
        db.row_factory = sqlite3.Row
        return db

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS autonomy_discovery_events (
                    id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    source_ref TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'pending'
                        CHECK(status IN ('pending', 'processed', 'dismissed')),
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_autonomy_events_status
                    ON autonomy_discovery_events(status, created_at);
                CREATE TABLE IF NOT EXISTS autonomy_goal_proposals (
                    id TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL UNIQUE,
                    title TEXT NOT NULL,
                    objective TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_ref TEXT NOT NULL,
                    priority INTEGER NOT NULL CHECK(priority BETWEEN 1 AND 100),
                    rationale TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'proposed'
                        CHECK(status IN ('proposed', 'approved', 'rejected')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_autonomy_proposals_status
                    ON autonomy_goal_proposals(status, priority DESC, created_at);
                CREATE TABLE IF NOT EXISTS autonomy_proposal_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    proposal_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    detail TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(proposal_id) REFERENCES autonomy_goal_proposals(id)
                );
            """)

    def record_external_event(
        self, event_type: str, summary: str, source_ref: str = "",
    ) -> str:
        """Persist an externally observed event as untrusted data; never execute it."""
        kind, text, source = _clean(event_type, 80), _clean(summary, 700), _clean(source_ref, 500)
        if not kind or not text:
            raise ValueError("External event type and summary are required")
        event_id = uuid4().hex
        with self._connect() as db:
            db.execute(
                """INSERT INTO autonomy_discovery_events
                   (id, event_type, summary, source_ref, status, created_at)
                   VALUES (?, ?, ?, ?, 'pending', ?)""",
                (event_id, kind, text, source, _now()),
            )
        return event_id

    def _candidates(self) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        with self._connect() as db:
            events = db.execute(
                """SELECT id, event_type, summary, source_ref
                   FROM autonomy_discovery_events WHERE status='pending'
                   ORDER BY created_at LIMIT 50"""
            ).fetchall()
            for event in events:
                candidates.append({
                    "source_type": "external_event", "source_ref": event["id"],
                    "title": f"Review event: {event['event_type']}",
                    "objective": f"Review and triage this reported event: {event['summary']}",
                    "priority": 55,
                    "rationale": "An external event was explicitly submitted; its contents are untrusted and require review.",
                    "_event_id": event["id"],
                })

            # Read-only observation of recent failed tasks. No retry is proposed
            # automatically because the original task may have had side effects.
            task_table = db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='tasks'"
            ).fetchone()
            if task_table:
                rows = db.execute(
                    """SELECT id, objective, state_json FROM tasks
                       WHERE status='failed' ORDER BY updated_at DESC LIMIT 10"""
                ).fetchall()
                for row in rows:
                    state = json.loads(row["state_json"] or "{}")
                    recovery = state.get("recovery", {})
                    candidates.append({
                        "source_type": "failed_task", "source_ref": row["id"],
                        "title": "Review failed task",
                        "objective": f"Diagnose the failure of task {row['id']} ({_clean(row['objective'], 300)}); do not repeat its actions.",
                        "priority": 65,
                        "rationale": _clean(recovery.get("reason") or "A prior task failed and requires diagnosis before any retry."),
                    })

            report_table = db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='research_reports'"
            ).fetchone()
            if report_table:
                rows = db.execute(
                    """SELECT id, query FROM research_reports
                       ORDER BY created_at DESC LIMIT 5"""
                ).fetchall()
                for row in rows:
                    candidates.append({
                        "source_type": "research_report", "source_ref": row["id"],
                        "title": "Review research report",
                        "objective": f"Review saved research report {row['id']} for unresolved evidence, source quality and actionable follow-up: {_clean(row['query'], 240)}",
                        "priority": 35,
                        "rationale": "A saved report may contain unresolved evidence; report contents are not treated as verified facts.",
                    })

        # Git status is metadata-only: file contents are never read or executed.
        try:
            result = subprocess.run(
                ["git", "status", "--short", "--untracked-files=no"],
                cwd=self.project_root, capture_output=True, text=True,
                timeout=5, check=False, shell=False,
            )
            if result.returncode == 0:
                changed = [
                    _clean(line[3:], 240) for line in result.stdout.splitlines()
                    if len(line) >= 4 and line[:2].strip()
                ][:20]
                if changed:
                    candidates.append({
                        "source_type": "project_change", "source_ref": "git-status",
                        "title": "Review project changes",
                        "objective": "Review these uncommitted project changes and identify tests or documentation needed: " + ", ".join(changed),
                        "priority": 45,
                        "rationale": "Git reported changed tracked paths. Only path metadata was inspected; no file was read or executed.",
                    })
        except (OSError, subprocess.TimeoutExpired):
            pass
        return candidates

    def discover(self) -> list[dict[str, Any]]:
        """Create deduplicated proposals. Does not approve or enqueue anything."""
        created: list[str] = []
        candidates = self._candidates()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for item in candidates[:self.max_proposals]:
                fingerprint = hashlib.sha256(
                    (item["source_type"] + "\0" + item["source_ref"] + "\0" + item["objective"]).encode("utf-8")
                ).hexdigest()
                proposal_id = uuid4().hex
                now = _now()
                cursor = db.execute(
                    """INSERT OR IGNORE INTO autonomy_goal_proposals
                       (id, fingerprint, title, objective, source_type, source_ref,
                        priority, rationale, status, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'proposed', ?, ?)""",
                    (proposal_id, fingerprint, item["title"], item["objective"],
                     item["source_type"], item["source_ref"], item["priority"],
                     item["rationale"], now, now),
                )
                if cursor.rowcount:
                    created.append(proposal_id)
                    db.execute(
                        """INSERT INTO autonomy_proposal_events
                           (proposal_id, event_type, detail, created_at)
                           VALUES (?, 'proposed', ?, ?)""",
                        (proposal_id, item["rationale"], now),
                    )
                if item.get("_event_id"):
                    db.execute(
                        "UPDATE autonomy_discovery_events SET status='processed' WHERE id=?",
                        (item["_event_id"],),
                    )
        return [item for item in self.list_proposals() if item["id"] in set(created)]

    def list_proposals(
        self, status: str | None = None, limit: int = 50,
    ) -> list[dict[str, Any]]:
        if status is not None and status not in {"proposed", "approved", "rejected"}:
            raise ValueError("Unknown proposal status")
        cap = max(1, min(int(limit), 200))
        with self._connect() as db:
            if status:
                rows = db.execute(
                    """SELECT * FROM autonomy_goal_proposals WHERE status=?
                       ORDER BY priority DESC, created_at LIMIT ?""", (status, cap)
                ).fetchall()
            else:
                rows = db.execute(
                    """SELECT * FROM autonomy_goal_proposals
                       ORDER BY priority DESC, created_at LIMIT ?""", (cap,)
                ).fetchall()
        return [dict(row) for row in rows]

    def _transition(self, proposal_id: str, status: str, detail: str) -> dict[str, Any]:
        if status not in {"approved", "rejected"}:
            raise ValueError("Proposal transition must be approved or rejected")
        explanation = _clean(detail, 500)
        if not explanation:
            raise ValueError("An approval/rejection reason is required")
        now = _now()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT status FROM autonomy_goal_proposals WHERE id=?", (proposal_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown goal proposal: {proposal_id}")
            if row["status"] != "proposed":
                raise ValueError(f"Only proposed goals can be reviewed; found {row['status']}")
            db.execute(
                "UPDATE autonomy_goal_proposals SET status=?, updated_at=? WHERE id=?",
                (status, now, proposal_id),
            )
            db.execute(
                """INSERT INTO autonomy_proposal_events
                   (proposal_id, event_type, detail, created_at)
                   VALUES (?, ?, ?, ?)""", (proposal_id, status, explanation, now)
            )
        return next(item for item in self.list_proposals(limit=200) if item["id"] == proposal_id)

    def approve(self, proposal_id: str, reason: str) -> dict[str, Any]:
        """Record human approval; caller must explicitly enqueue via AutonomyRunner."""
        return self._transition(proposal_id, "approved", reason)

    def reject(self, proposal_id: str, reason: str) -> dict[str, Any]:
        return self._transition(proposal_id, "rejected", reason)

    def events(self, proposal_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                """SELECT sequence, event_type, detail, created_at
                   FROM autonomy_proposal_events WHERE proposal_id=? ORDER BY sequence""",
                (proposal_id,),
            ).fetchall()
        return [dict(row) for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover bounded AETHER goal proposals")
    parser.add_argument("--db", default="data/aether.sqlite3")
    parser.add_argument("--root", default=".")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    discovery = GoalDiscovery(args.db, args.root)
    proposals = discovery.list_proposals() if args.list else discovery.discover()
    print(json.dumps({"count": len(proposals), "proposals": proposals}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
