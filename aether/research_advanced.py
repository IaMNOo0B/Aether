"""Advanced research workflows with auditable screening, evidence checks and reproducible analysis.

The deterministic citation checker finds exact quoted passages and lexical alignment candidates.
It does not infer semantic entailment or independently establish truth; human adjudication is explicit.
"""
from __future__ import annotations

from collections import Counter
from datetime import date
from hashlib import sha256
from io import StringIO
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any
from urllib.parse import urlsplit
import csv
import json
import math
import platform
import re
import sys
from uuid import uuid4

from aether.research import ResearchError, ResearchSource
from aether.storage import Store


ANALYSIS_VERSION = "1.0"


def _tokens(value: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", value.casefold())
        if len(token) > 2
    }


class AdvancedResearch:
    def __init__(self, store: Store):
        self.store = store

    def create_systematic_review(
        self, question: str, inclusion_criteria: list[str],
        exclusion_criteria: list[str], search_strategy: dict[str, Any],
    ) -> dict[str, Any]:
        question = question.strip()
        if len(question) < 8:
            raise ValueError("Review question must contain at least 8 characters")
        if not inclusion_criteria or not all(isinstance(x, str) and x.strip() for x in inclusion_criteria):
            raise ValueError("At least one explicit inclusion criterion is required")
        if not exclusion_criteria or not all(isinstance(x, str) and x.strip() for x in exclusion_criteria):
            raise ValueError("At least one explicit exclusion criterion is required")
        required = {"databases", "queries", "searched_on"}
        if not isinstance(search_strategy, dict) or not required.issubset(search_strategy):
            raise ValueError("Search strategy requires databases, queries and searched_on")
        if not search_strategy["databases"] or not search_strategy["queries"]:
            raise ValueError("Search strategy must name at least one database and query")
        try:
            date.fromisoformat(str(search_strategy["searched_on"])[:10])
        except ValueError as exc:
            raise ValueError("searched_on must be an ISO date") from exc
        review_id = uuid4().hex
        protocol = {
            "question": question,
            "inclusion_criteria": [x.strip() for x in inclusion_criteria],
            "exclusion_criteria": [x.strip() for x in exclusion_criteria],
            "search_strategy": search_strategy,
            "status": "screening",
            "created_on": date.today().isoformat(),
        }
        self.store.create_systematic_review(review_id, protocol)
        return self.store.get_systematic_review(review_id)

    def add_review_sources(self, review_id: str, sources: list[ResearchSource]) -> dict[str, Any]:
        review = self.store.get_systematic_review(review_id)
        if review is None:
            raise KeyError(f"Unknown systematic review: {review_id}")
        added = 0
        for source in sources:
            self.store.add_systematic_review_source(
                review_id, source.url or source.title, source.to_dict()
            )
            added += 1
        return {"review_id": review_id, "submitted": len(sources), "added_or_already_present": added}

    def screen_source(
        self, review_id: str, source_key: str, decision: str, reason: str,
    ) -> dict[str, Any]:
        if decision not in {"include", "exclude", "uncertain"}:
            raise ValueError("Screening decision must be include, exclude or uncertain")
        if len(reason.strip()) < 8:
            raise ValueError("A screening reason of at least 8 characters is required")
        self.store.screen_systematic_review_source(review_id, source_key, decision, reason.strip())
        return self.store.get_systematic_review(review_id)

    def finalize_review(self, review_id: str) -> dict[str, Any]:
        review = self.store.get_systematic_review(review_id)
        if review is None:
            raise KeyError(f"Unknown systematic review: {review_id}")
        sources = review["sources"]
        if not sources:
            raise ValueError("Cannot finalize a review with no retrieved sources")
        if any(item["decision"] in {"pending", "uncertain"} for item in sources):
            raise ValueError("Every retrieved source must have an include/exclude decision before finalization")
        included = [item for item in sources if item["decision"] == "include"]
        if not included:
            raise ValueError("Review cannot be finalized with zero included sources")
        summary = {
            "review_id": review_id,
            "question": review["protocol"]["question"],
            "retrieved": len(sources),
            "included": len(included),
            "excluded": len(sources) - len(included),
            "included_sources": [item["source"] for item in included],
            "status": "screening_complete_synthesis_pending",
            "limitations": [
                "Screening decisions are single-reviewer unless independent reviewers are separately recorded.",
                "Full-text eligibility, risk-of-bias assessment and evidence synthesis are not inferred from title/abstract screening.",
                "No PRISMA compliance or meta-analysis is claimed.",
            ],
        }
        self.store.finalize_systematic_review(review_id, summary)
        return self.store.get_systematic_review(review_id)

    def check_citation_support(
        self, claim_text: str, report_id: str, source_url: str, quoted_passage: str,
    ) -> dict[str, Any]:
        claim_text = claim_text.strip()
        quote = quoted_passage.strip()
        if len(claim_text) < 12 or len(quote) < 20:
            raise ValueError("Claim must be at least 12 characters and quoted passage at least 20 characters")
        report = self.store.get_research_report(report_id)
        if report is None:
            raise KeyError(f"Unknown research report: {report_id}")
        source = next((item for item in report["sources"] if item.get("url") == source_url), None)
        if source is None:
            raise ValueError("Source URL must belong to the specified research report")
        source_text = str(source.get("full_text") or source.get("summary") or "")
        normalized_source = " ".join(source_text.casefold().split())
        normalized_quote = " ".join(quote.casefold().split())
        position = normalized_source.find(normalized_quote)
        claim_terms = _tokens(claim_text)
        quote_terms = _tokens(quote)
        overlap = len(claim_terms & quote_terms) / max(1, len(claim_terms))
        exact_match = position >= 0
        if not exact_match:
            assessment = "quote_not_found"
        elif overlap >= 0.6:
            assessment = "candidate_support_requires_human_adjudication"
        else:
            assessment = "passage_found_but_weak_lexical_alignment"
        result = {
            "id": uuid4().hex,
            "claim": claim_text,
            "report_id": report_id,
            "source_url": source_url,
            "quoted_passage": quote,
            "normalized_character_offset": position if exact_match else None,
            "lexical_claim_coverage": round(overlap, 4),
            "assessment": assessment,
            "human_decision": None,
            "limitations": "Exact quote matching and lexical overlap are deterministic screening aids, not semantic entailment. A passage can mention the same terms while contradicting or qualifying a claim.",
        }
        self.store.save_citation_evidence(result)
        return result

    def adjudicate_citation(
        self, evidence_id: str, decision: str, rationale: str,
    ) -> dict[str, Any]:
        if decision not in {"supports", "contradicts", "insufficient"}:
            raise ValueError("Decision must be supports, contradicts or insufficient")
        if len(rationale.strip()) < 12:
            raise ValueError("Human adjudication requires a rationale of at least 12 characters")
        self.store.adjudicate_citation_evidence(evidence_id, decision, rationale.strip())
        return self.store.get_citation_evidence(evidence_id)

    def analyze_csv(
        self, experiment_id: str, csv_bytes: bytes, numeric_columns: list[str],
    ) -> dict[str, Any]:
        experiment = self.store.get_research_experiment(experiment_id)
        if experiment is None:
            raise KeyError(f"Unknown experiment: {experiment_id}")
        if experiment["status"] not in {"running", "completed"}:
            raise ValueError("Experiment must be running or completed before analysis")
        if len(csv_bytes) > 5_000_000:
            raise ValueError("CSV input is limited to 5 MB")
        try:
            text = csv_bytes.decode("utf-8-sig")
            reader = csv.DictReader(StringIO(text))
            if not reader.fieldnames:
                raise ValueError("CSV must contain a header row")
            requested = list(dict.fromkeys(numeric_columns))
            if not requested or any(column not in reader.fieldnames for column in requested):
                raise ValueError("Every requested numeric column must exist in the CSV header")
            rows = list(reader)
        except (UnicodeDecodeError, csv.Error) as exc:
            raise ValueError(f"Invalid UTF-8 CSV: {exc}") from exc
        if not rows:
            raise ValueError("CSV must contain at least one data row")
        results: dict[str, Any] = {}
        for column in requested:
            values = []
            for row_number, row in enumerate(rows, start=2):
                raw = (row.get(column) or "").strip()
                if not raw:
                    continue
                try:
                    value = float(raw)
                except ValueError as exc:
                    raise ValueError(f"Non-numeric value in {column!r} at CSV row {row_number}") from exc
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite value in {column!r} at CSV row {row_number}")
                values.append(value)
            if not values:
                raise ValueError(f"No numeric observations found for {column!r}")
            results[column] = {
                "n": len(values),
                "missing": len(rows) - len(values),
                "mean": mean(values),
                "median": median(values),
                "population_stddev": pstdev(values),
                "min": min(values),
                "max": max(values),
            }
        dataset_hash = sha256(csv_bytes).hexdigest()
        reproducibility = {
            "analysis_version": ANALYSIS_VERSION,
            "python_version": platform.python_version(),
            "algorithm": "stdlib-descriptive-statistics-v1",
            "input_sha256": dataset_hash,
            "columns": requested,
            "row_count": len(rows),
            "results": results,
        }
        fingerprint = sha256(
            json.dumps(reproducibility, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        artifact = {
            "experiment_id": experiment_id,
            "reproducibility_fingerprint": fingerprint,
            **reproducibility,
            "interpretation": "Descriptive statistics only. No causal inference, significance test, or hypothesis confirmation is performed.",
        }
        report_id = uuid4().hex
        self.store.save_research_report(
            report_id,
            f"experiment-analysis:{experiment_id}",
            json.dumps(artifact, indent=2, sort_keys=True),
            [{
                "title": "Input CSV",
                "url": "",
                "publisher": "local operator-supplied dataset",
                "published": date.today().isoformat(),
                "evidence_type": "experiment-analysis-artifact",
                "summary": f"sha256:{dataset_hash}",
                "artifact": artifact,
            }],
        )
        self.store.append_research_experiment_event(
            experiment_id, "analysis_recorded",
            {"report_id": report_id, "input_sha256": dataset_hash, "fingerprint": fingerprint},
        )
        return {"report_id": report_id, **artifact}

