"""Completion tools for auditable systematic-review reporting and experiment evidence.

These are structured operator tools, not a claim of PRISMA certification or validated
clinical risk-of-bias instruments. Reviewer authentication is local to one AETHER DB.
"""
from __future__ import annotations

from hashlib import pbkdf2_hmac, sha256
from math import sqrt
from pathlib import Path
from statistics import mean, stdev
from typing import Any
from uuid import uuid4
import csv
import io
import json
import secrets
import sqlite3

from aether.storage import Store


RO_B_DOMAINS = {
    "randomized": (
        "randomization_process", "deviations_from_intended_interventions",
        "missing_outcome_data", "outcome_measurement", "selection_of_reported_result",
    ),
    "nonrandomized": (
        "confounding", "participant_selection", "intervention_classification",
        "deviations_from_intended_interventions", "missing_data",
        "outcome_measurement", "selection_of_reported_result",
    ),
    "qualitative": (
        "researcher_reflexivity", "sampling_appropriateness", "data_collection",
        "analysis_rigor", "participant_voice", "ethical_considerations",
    ),
    "cross_sectional": (
        "sampling_frame", "sample_selection", "measurement_validity",
        "confounding", "statistical_analysis",
    ),
}
GRADE_DOMAINS = (
    "risk_of_bias", "inconsistency", "indirectness", "imprecision", "publication_bias",
)
GRADE_LEVELS = ("high", "moderate", "low", "very_low")
GRADE_DOWNGRADES = {"none": 0, "serious": 1, "very_serious": 2}


class ResearchCompletion:
    def __init__(self, store: Store):
        self.store = store
        self.identity_path = store.path.with_name(store.path.stem + ".reviewers.sqlite3")
        self._init_identity_db()

    def _init_identity_db(self) -> None:
        self.identity_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.identity_path) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS reviewers (
                reviewer_id TEXT PRIMARY KEY, salt BLOB NOT NULL,
                password_hash BLOB NOT NULL, created_at TEXT NOT NULL
            )""")

    def register_reviewer(self, reviewer_id: str, password: str) -> dict[str, str]:
        reviewer_id = reviewer_id.strip()
        if not reviewer_id or len(reviewer_id) > 120 or len(password) < 12:
            raise ValueError("Reviewer ID must be 1..120 chars and password at least 12 chars")
        salt = secrets.token_bytes(16)
        digest = pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 310_000)
        from datetime import datetime, timezone
        with sqlite3.connect(self.identity_path) as db:
            try:
                db.execute(
                    "INSERT INTO reviewers VALUES (?, ?, ?, ?)",
                    (reviewer_id, salt, digest, datetime.now(timezone.utc).isoformat()),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("Reviewer ID already exists") from exc
        return {"reviewer_id": reviewer_id, "status": "registered"}

    def authentication_enabled(self) -> bool:
        with sqlite3.connect(self.identity_path) as db:
            count = db.execute("SELECT COUNT(*) FROM reviewers").fetchone()[0]
        return count > 0

    def authenticate_reviewer(self, reviewer_id: str, password: str) -> bool:
        with sqlite3.connect(self.identity_path) as db:
            row = db.execute(
                "SELECT salt, password_hash FROM reviewers WHERE reviewer_id=?",
                (reviewer_id.strip(),),
            ).fetchone()
        if row is None:
            # Keep unknown-user and wrong-password paths computationally similar.
            pbkdf2_hmac("sha256", password.encode("utf-8"), b"AETHER-unknown-user", 310_000)
            return False
        candidate = pbkdf2_hmac("sha256", password.encode("utf-8"), row[0], 310_000)
        import hmac
        return hmac.compare_digest(candidate, row[1])

    def _require_identity(self, reviewer_id: str, password: str) -> None:
        if not self.authenticate_reviewer(reviewer_id, password):
            raise PermissionError("Reviewer authentication failed")

    def record_risk_of_bias(
        self, review_id: str, source_key: str, design: str, reviewer_id: str,
        password: str, ratings: dict[str, dict[str, str]],
    ) -> dict[str, Any]:
        self._require_identity(reviewer_id, password)
        domains = RO_B_DOMAINS.get(design)
        if domains is None:
            raise ValueError(f"Unsupported study design: {design}")
        if set(ratings) != set(domains):
            raise ValueError(f"{design} assessment requires exactly these domains: {', '.join(domains)}")
        valid = {"low", "some_concerns", "high", "not_applicable"}
        for domain, item in ratings.items():
            if item.get("rating") not in valid or len(item.get("rationale", "").strip()) < 12:
                raise ValueError(f"{domain} requires a valid rating and rationale")
        payload = {
            "instrument": f"AETHER-structured-{design}-v1",
            "design": design,
            "domains": ratings,
            "limitation": "Structured domain checklist; not a validated RoB 2, ROBINS-I, CASP or JBI instrument.",
        }
        # Keep design-specific assessments in the same dual-reviewed evidence pipeline.
        from aether.research_advanced import AdvancedResearch
        consensus = AdvancedResearch(self.store).record_review_stage(
            review_id, source_key, "risk_of_bias", reviewer_id, payload, password=password,
        )
        return {"instrument": payload["instrument"], "design": design, **consensus}

    def grade_certainty(
        self, review_id: str, outcome: str, starting_level: str,
        domain_judgments: dict[str, dict[str, str]], rationale: str,
    ) -> dict[str, Any]:
        if starting_level not in GRADE_LEVELS:
            raise ValueError("Starting level must be high, moderate, low or very_low")
        if set(domain_judgments) != set(GRADE_DOMAINS):
            raise ValueError("GRADE assessment requires all five certainty domains")
        if len(rationale.strip()) < 20 or len(outcome.strip()) < 3:
            raise ValueError("Outcome and an explicit overall rationale are required")
        downgrades = 0
        for domain, item in domain_judgments.items():
            if item.get("judgment") not in GRADE_DOWNGRADES:
                raise ValueError(f"{domain} judgment must be none, serious or very_serious")
            if len(item.get("rationale", "").strip()) < 12:
                raise ValueError(f"{domain} requires a rationale")
            downgrades += GRADE_DOWNGRADES[item["judgment"]]
        level = GRADE_LEVELS[max(0, GRADE_LEVELS.index(starting_level) + downgrades)]
        artifact = {
            "review_id": review_id, "outcome": outcome, "framework": "GRADE-inspired structured assessment",
            "starting_level": starting_level, "domain_judgments": domain_judgments,
            "total_downgrades": downgrades, "certainty": level, "rationale": rationale.strip(),
            "limitations": "Operator judgment is required. This is not an official GRADEpro assessment and does not replace guideline-specific methods.",
        }
        report_id = uuid4().hex
        self.store.save_research_report(report_id, f"certainty:{review_id}:{outcome}",
            json.dumps(artifact, indent=2, sort_keys=True),
            [{"title": outcome, "url": "", "evidence_type": "certainty-of-evidence-assessment",
              "summary": rationale, "artifact": artifact}])
        return {"report_id": report_id, **artifact}

    def prisma_flow(self, review_id: str) -> dict[str, Any]:
        review = self.store.get_systematic_review(review_id)
        if review is None:
            raise KeyError(f"Unknown systematic review: {review_id}")
        sources = review.get("sources", [])
        summary = review.get("final_summary") or {}
        flow = summary.get("screening_flow", {})
        retrieved = len(sources)
        included_title = sum(x.get("decision") == "include" for x in sources)
        excluded_title = sum(x.get("decision") == "exclude" for x in sources)
        full_excluded = len(summary.get("full_text_excluded", []))
        synthesis = len(summary.get("included_studies", []))
        report = {
            "review_id": review_id,
            "reporting_framework": "PRISMA-style flow counts (not PRISMA compliance)",
            "identification": {"records_from_searches": retrieved, "duplicate_records_removed": max(0, int(summary.get("duplicates_removed", 0))), "other_records": 0},
            "screening": {"records_screened": retrieved, "records_excluded": excluded_title},
            "eligibility": {"reports_sought": included_title, "reports_not_retrieved": int(summary.get("full_text_not_retrieved", 0)), "full_text_assessed": max(0, included_title - int(summary.get("full_text_not_retrieved", 0))), "full_text_excluded": full_excluded},
            "included": {"studies_in_review": len(summary.get("included_studies", [])), "studies_in_quantitative_synthesis": int((summary.get("meta_analysis") or {}).get("studies", 0))},
            "source_counts": {"title_abstract_included": included_title, "title_abstract_excluded": excluded_title, "included_in_synthesis": synthesis},
            "limitations": ["Counts reflect records imported into AETHER, not necessarily all records returned by external databases.", "No deduplication across independent databases can be inferred beyond stored review sources.", "This is a PRISMA-style count summary, not a completed PRISMA checklist or compliance claim."],
        }
        return report

    @staticmethod
    def egger_publication_bias(effects: list[dict[str, float]]) -> dict[str, Any]:
        if len(effects) < 10:
            return {"status": "not_tested_fewer_than_10_studies", "studies": len(effects),
                    "limitation": "Egger regression is generally underpowered with fewer than ten studies."}
        rows = []
        for item in effects:
            estimate, se = float(item["effect_estimate"]), float(item["standard_error"])
            if not (se > 0 and all(map(lambda x: abs(x) < float("inf"), (estimate, se)))):
                raise ValueError("Effects must be finite and standard errors positive")
            rows.append((1.0 / se, estimate / se))
        xs, ys = [r[0] for r in rows], [r[1] for r in rows]
        xbar, ybar = mean(xs), mean(ys)
        sxx = sum((x-xbar)**2 for x in xs)
        if sxx == 0:
            raise ValueError("Egger regression requires variation in study precision")
        slope = sum((x-xbar)*(y-ybar) for x,y in zip(xs,ys)) / sxx
        intercept = ybar - slope*xbar
        residual = sum((y-(intercept+slope*x))**2 for x,y in zip(xs,ys))
        se_intercept = sqrt((residual / (len(xs)-2)) * (1/len(xs) + xbar*xbar/sxx))
        statistic = intercept / se_intercept if se_intercept else float("inf")
        # Normal approximation, deliberately labeled as approximate.
        import math
        p = math.erfc(abs(statistic) / sqrt(2)) if se_intercept else 0.0
        return {"status": "tested", "test": "Egger regression (normal-approximation p-value)",
                "studies": len(rows), "intercept": intercept, "standard_error": se_intercept,
                "z_approximation": statistic, "two_sided_p_approximation": p,
                "interpretation": "Asymmetry is not proof of publication bias; small-study effects and heterogeneity can also cause asymmetry.",
                "limitations": ["Low power remains possible.", "Requires independent comparable effects.", "No trim-and-fill correction or selection model is performed."]}

    def publication_bias_for_review(self, review_id: str) -> dict[str, Any]:
        review = self.store.get_systematic_review(review_id)
        if review is None:
            raise KeyError(f"Unknown systematic review: {review_id}")
        summary = review.get("final_summary") or {}
        studies = summary.get("included_studies", [])
        effects = [x["extraction"] for x in studies if {"effect_estimate", "standard_error"} <= x.get("extraction", {}).keys()]
        result = self.egger_publication_bias(effects)
        result["review_id"] = review_id
        result["effect_measure"] = studies[0]["extraction"].get("effect_measure") if studies else None
        return result

    def ingest_raw_observations(
        self, experiment_id: str, csv_bytes: bytes, group_column: str,
        outcome_column: str, treatment_label: str, control_label: str,
    ) -> dict[str, Any]:
        experiment = self.store.get_research_experiment(experiment_id)
        if experiment is None:
            raise KeyError(f"Unknown experiment: {experiment_id}")
        if experiment["status"] not in {"running", "completed"}:
            raise ValueError("Experiment must be running or completed")
        if len(csv_bytes) > 5_000_000:
            raise ValueError("Raw observation CSV is limited to 5 MB")
        try:
            reader = csv.DictReader(io.StringIO(csv_bytes.decode("utf-8-sig")))
            if not reader.fieldnames or group_column not in reader.fieldnames or outcome_column not in reader.fieldnames:
                raise ValueError("CSV must include the requested group and outcome columns")
            groups: dict[str, list[float]] = {treatment_label: [], control_label: []}
            for line, row in enumerate(reader, 2):
                label = (row.get(group_column) or "").strip()
                if label not in groups:
                    raise ValueError(f"Unexpected group {label!r} at row {line}")
                try:
                    value = float(row[outcome_column])
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"Invalid numeric outcome at row {line}") from exc
                if not __import__("math").isfinite(value):
                    raise ValueError(f"Non-finite outcome at row {line}")
                groups[label].append(value)
        except UnicodeDecodeError as exc:
            raise ValueError("Raw observation CSV must be UTF-8") from exc
        treatment, control = groups[treatment_label], groups[control_label]
        if len(treatment) < 2 or len(control) < 2:
            raise ValueError("At least two observations per group are required")
        difference = mean(treatment) - mean(control)
        standard_error = sqrt(stdev(treatment)**2/len(treatment) + stdev(control)**2/len(control))
        dataset_hash = sha256(csv_bytes).hexdigest()
        artifact = {
            "experiment_id": experiment_id, "input_sha256": dataset_hash,
            "group_column": group_column, "outcome_column": outcome_column,
            "treatment_n": len(treatment), "control_n": len(control),
            "treatment_mean": mean(treatment), "control_mean": mean(control),
            "mean_difference_treatment_minus_control": difference,
            "standard_error_welch": standard_error,
            "effect_measure": "raw_mean_difference",
            "limitations": "Raw rows are summarized using an independent-groups Welch standard error. Pairing, clustering, repeated measures, missingness mechanisms and causal identification are not inferred.",
        }
        event = {"type": "raw_observations_ingested", **artifact}
        self.store.append_research_experiment_event(experiment_id, "raw_observations_ingested", event)
        return artifact
