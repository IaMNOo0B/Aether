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


ANALYSIS_VERSION = "1.1"


def _tokens(value: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", value.casefold())
        if len(token) > 2
    }



def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Numerical Recipes continued fraction for the regularized beta function."""
    max_iterations, epsilon, floor = 200, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c_value = 1.0
    d_value = 1.0 - qab * x / qap
    if abs(d_value) < floor:
        d_value = floor
    d_value = 1.0 / d_value
    result = d_value
    for iteration in range(1, max_iterations + 1):
        m2 = 2 * iteration
        aa = iteration * (b - iteration) * x / ((qam + m2) * (a + m2))
        d_value = 1.0 + aa * d_value
        if abs(d_value) < floor:
            d_value = floor
        c_value = 1.0 + aa / c_value
        if abs(c_value) < floor:
            c_value = floor
        d_value = 1.0 / d_value
        result *= d_value * c_value
        aa = -(a + iteration) * (qab + iteration) * x / ((a + m2) * (qap + m2))
        d_value = 1.0 + aa * d_value
        if abs(d_value) < floor:
            d_value = floor
        c_value = 1.0 + aa / c_value
        if abs(c_value) < floor:
            c_value = floor
        d_value = 1.0 / d_value
        delta = d_value * c_value
        result *= delta
        if abs(delta - 1.0) < epsilon:
            return result
    raise ArithmeticError("Incomplete beta continued fraction did not converge")


def _regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    factor = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return factor * _beta_continued_fraction(a, b, x) / a
    return 1.0 - factor * _beta_continued_fraction(b, a, 1.0 - x) / b


def _welch_test(left: list[float], right: list[float]) -> dict[str, float]:
    if len(left) < 2 or len(right) < 2:
        raise ValueError("Welch comparison requires at least two observations per group")
    left_var, right_var = pstdev(left) ** 2 * len(left) / (len(left) - 1), pstdev(right) ** 2 * len(right) / (len(right) - 1)
    left_term, right_term = left_var / len(left), right_var / len(right)
    standard_error = math.sqrt(left_term + right_term)
    if standard_error == 0:
        raise ValueError("Welch comparison is undefined when both groups have zero variance")
    statistic = (mean(left) - mean(right)) / standard_error
    degrees = (left_term + right_term) ** 2 / (
        left_term ** 2 / (len(left) - 1) + right_term ** 2 / (len(right) - 1)
    )
    p_value = _regularized_beta(
        degrees / (degrees + statistic * statistic), degrees / 2.0, 0.5
    )
    pooled_variance = (
        (len(left) - 1) * left_var + (len(right) - 1) * right_var
    ) / (len(left) + len(right) - 2)
    effect_size = (
        (mean(left) - mean(right)) / math.sqrt(pooled_variance)
        if pooled_variance > 0 else 0.0
    )
    return {
        "mean_difference_left_minus_right": mean(left) - mean(right),
        "welch_t": statistic,
        "welch_degrees_of_freedom": degrees,
        "two_sided_p_value": p_value,
        "cohens_d_pooled_sd": effect_size,
        "mean_difference_95pct_normal_approximation": [
            mean(left) - mean(right) - 1.96 * standard_error,
            mean(left) - mean(right) + 1.96 * standard_error,
        ],
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

    def add_review_sources(
        self, review_id: str, sources: list[ResearchSource],
        search_report_id: str | None = None,
    ) -> dict[str, Any]:
        review = self.store.get_systematic_review(review_id)
        if review is None:
            raise KeyError(f"Unknown systematic review: {review_id}")
        search_report = (
            self.store.get_research_report(search_report_id)
            if search_report_id is not None else None
        )
        if search_report_id is not None and search_report is None:
            raise ValueError("search_report_id must identify an existing research report")
        if search_report is not None:
            valid_sources = {
                (item.get("url", ""), item.get("title", ""))
                for item in search_report["sources"]
            }
            if any((source.url, source.title) not in valid_sources for source in sources):
                raise ValueError("Every imported source must belong to the linked research report")
        added = 0
        for source in sources:
            source_record = source.to_dict()
            source_record["search_report_id"] = search_report_id
            self.store.add_systematic_review_source(
                review_id, source.url or source.title, source_record
            )
            added += 1
        return {"review_id": review_id, "submitted": len(sources), "added_or_already_present": added}

    def screen_source(
        self, review_id: str, source_key: str, reviewer_id: str,
        decision: str, reason: str,
    ) -> dict[str, Any]:
        if decision not in {"include", "exclude"}:
            raise ValueError("Screening decision must be include or exclude")
        if len(reason.strip()) < 8:
            raise ValueError("A screening reason of at least 8 characters is required")
        self.store.screen_systematic_review_source(review_id, source_key, reviewer_id, decision, reason.strip())
        return self.store.get_systematic_review(review_id)

    def resolve_review_conflict(self, review_id: str, source_key: str, adjudicator_id: str, decision: str, rationale: str) -> dict[str, Any]:
        self.store.resolve_systematic_review_conflict(
            review_id, source_key, adjudicator_id, decision, rationale
        )
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
        summary = {
            "review_id": review_id,
            "question": review["protocol"]["question"],
            "retrieved": len(sources),
            "included": len(included),
            "excluded": len(sources) - len(included),
            "included_sources": [item["source"] for item in included],
            "status": "screening_complete_synthesis_pending",
            "limitations": [
                "Title/abstract screening requires two distinct operator-supplied reviewer IDs; those IDs are not authenticated identities.",
                "Full-text eligibility, risk-of-bias assessment, extraction and synthesis require separate explicit records.",
                "No PRISMA compliance or meta-analysis is claimed.",
            ],
        }
        self.store.finalize_systematic_review(review_id, summary)
        return self.store.get_systematic_review(review_id)


    def _stage_consensus(self, review_id: str, source_key: str, stage: str) -> dict[str, Any]:
        entries = self.store.list_systematic_review_assessments(review_id, source_key)
        relevant = [item for item in entries if item["stage"] == stage]
        adjudicator = next((item for item in reversed(relevant) if item["role"] == "adjudicator"), None)
        reviewers = [item for item in relevant if item["role"] == "reviewer"]
        if adjudicator:
            return {"status": "adjudicated", "payload": adjudicator["payload"], "reviewers": reviewers}
        if len(reviewers) < 2:
            return {"status": "awaiting_second_reviewer", "payload": None, "reviewers": reviewers}
        left, right = reviewers[-2]["payload"], reviewers[-1]["payload"]
        if stage == "full_text":
            agrees = left.get("decision") == right.get("decision")
        else:
            agrees = json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)
        return {
            "status": "agreed" if agrees else "disagreement",
            "payload": left if agrees else None,
            "reviewers": reviewers,
        }

    @staticmethod
    def _validate_extraction_payload(payload: dict[str, Any]) -> None:
        required = {"study_design", "population", "sample_size", "outcome", "findings"}
        if not required.issubset(payload):
            raise ValueError("Extraction requires study_design, population, sample_size, outcome and findings")
        if isinstance(payload["sample_size"], bool) or int(payload["sample_size"]) < 1:
            raise ValueError("sample_size must be a positive integer")
        for key in ("study_design", "population", "outcome", "findings"):
            if not str(payload[key]).strip():
                raise ValueError(f"{key} must not be empty")
        quantitative = {"effect_measure", "effect_estimate", "standard_error"}
        supplied = quantitative & set(payload)
        if supplied and supplied != quantitative:
            raise ValueError("Quantitative extraction must supply effect_measure, effect_estimate and standard_error together")
        if supplied:
            estimate, standard_error = float(payload["effect_estimate"]), float(payload["standard_error"])
            if not str(payload["effect_measure"]).strip():
                raise ValueError("effect_measure must not be empty")
            if not math.isfinite(estimate) or not math.isfinite(standard_error) or standard_error <= 0:
                raise ValueError("effect_estimate must be finite and standard_error must be finite and positive")

    def record_review_stage(
        self, review_id: str, source_key: str, stage: str,
        reviewer_id: str, payload: dict[str, Any],
    ) -> dict[str, Any]:
        review = self.store.get_systematic_review(review_id)
        if review is None:
            raise KeyError(f"Unknown systematic review: {review_id}")
        if stage == "full_text":
            if payload.get("decision") not in {"include", "exclude"}:
                raise ValueError("Full-text assessment requires include or exclude")
            if len(str(payload.get("rationale", "")).strip()) < 12:
                raise ValueError("Full-text eligibility requires a detailed rationale")
            if payload["decision"] == "include" and not str(payload.get("text_locator", "")).strip():
                raise ValueError("Included full text requires a page, section or paragraph locator")
        elif stage == "risk_of_bias":
            domains = payload.get("domains")
            required = {"randomization", "deviations", "missing_data", "outcome_measurement", "selective_reporting"}
            if not isinstance(domains, dict) or not required.issubset(domains):
                raise ValueError("Risk-of-bias assessment requires five prespecified domains")
            valid = {"low", "some_concerns", "high", "not_applicable"}
            for name in required:
                item = domains[name]
                if not isinstance(item, dict) or item.get("rating") not in valid or len(str(item.get("rationale", "")).strip()) < 12:
                    raise ValueError(f"Risk-of-bias domain {name} requires a valid rating and rationale")
        elif stage == "extraction":
            self._validate_extraction_payload(payload)
        else:
            raise ValueError("Stage must be full_text, risk_of_bias or extraction")
        self.store.record_systematic_review_assessment(
            review_id, source_key, stage, reviewer_id, payload, "reviewer"
        )
        return self._stage_consensus(review_id, source_key, stage)

    def resolve_review_stage(
        self, review_id: str, source_key: str, stage: str,
        adjudicator_id: str, payload: dict[str, Any],
    ) -> dict[str, Any]:
        current = self._stage_consensus(review_id, source_key, stage)
        if current["status"] != "disagreement":
            raise ValueError("Stage has no unresolved two-reviewer disagreement")
        # Reuse the same stage validators while keeping the adjudicator distinct.
        if stage == "full_text":
            if payload.get("decision") not in {"include", "exclude"} or len(str(payload.get("rationale", "")).strip()) < 12:
                raise ValueError("Adjudication requires a decision and detailed rationale")
            if payload["decision"] == "include" and not str(payload.get("text_locator", "")).strip():
                raise ValueError("Included full text requires a text locator")
        elif stage == "risk_of_bias":
            self._validate_risk_domains(payload)
        elif stage == "extraction":
            self._validate_extraction_payload(payload)
        reviewers = {item["reviewer_id"] for item in current["reviewers"]}
        if adjudicator_id in reviewers:
            raise ValueError("Adjudicator must be distinct from both reviewers")
        self.store.record_systematic_review_assessment(
            review_id, source_key, stage, adjudicator_id, payload, "adjudicator"
        )
        return self._stage_consensus(review_id, source_key, stage)

    @staticmethod
    def _validate_risk_domains(payload: dict[str, Any]) -> None:
        domains = payload.get("domains")
        required = {"randomization", "deviations", "missing_data", "outcome_measurement", "selective_reporting"}
        valid = {"low", "some_concerns", "high", "not_applicable"}
        if not isinstance(domains, dict) or not required.issubset(domains):
            raise ValueError("Risk-of-bias assessment requires five prespecified domains")
        for name in required:
            item = domains[name]
            if not isinstance(item, dict) or item.get("rating") not in valid or len(str(item.get("rationale", "")).strip()) < 12:
                raise ValueError(f"Risk-of-bias domain {name} requires a valid rating and rationale")

    def synthesize_review(self, review_id: str) -> dict[str, Any]:
        review = self.store.get_systematic_review(review_id)
        if review is None:
            raise KeyError(f"Unknown systematic review: {review_id}")
        if review["status"] != "screening_complete_synthesis_pending":
            raise ValueError("Complete title/abstract screening before synthesis")
        included = []
        full_text_excluded = []
        for source in review["sources"]:
            if source["decision"] != "include":
                continue
            key = source["source_key"]
            full = self._stage_consensus(review_id, key, "full_text")
            if full["status"] not in {"agreed", "adjudicated"}:
                raise ValueError(f"Full-text eligibility for {key} is unresolved")
            if full["payload"]["decision"] == "exclude":
                full_text_excluded.append({"source": source["source"], "assessment": full["payload"]})
                continue
            risk = self._stage_consensus(review_id, key, "risk_of_bias")
            extraction = self._stage_consensus(review_id, key, "extraction")
            if risk["status"] not in {"agreed", "adjudicated"}:
                raise ValueError(f"Risk-of-bias assessment for {key} is unresolved")
            if extraction["status"] not in {"agreed", "adjudicated"}:
                raise ValueError(f"Data extraction for {key} is unresolved")
            included.append({
                "source": source["source"],
                "full_text_eligibility": full["payload"],
                "risk_of_bias": risk["payload"],
                "extraction": extraction["payload"],
            })
        quantitative_complete = all(
            {"effect_measure", "effect_estimate", "standard_error"}.issubset(item["extraction"])
            for item in included
        )
        effect_measures = {
            item["extraction"].get("effect_measure")
            for item in included if "effect_measure" in item["extraction"]
        }
        meta_analysis = None
        if len(included) >= 2 and quantitative_complete and len(effect_measures) == 1:
            rows = [item["extraction"] for item in included]
            estimates = [float(item["effect_estimate"]) for item in rows]
            variances = [float(item["standard_error"]) ** 2 for item in rows]
            weights = [1.0 / value for value in variances]
            fixed = sum(w * x for w, x in zip(weights, estimates)) / sum(weights)
            q = sum(w * (x - fixed) ** 2 for w, x in zip(weights, estimates))
            df = len(rows) - 1
            c_value = sum(weights) - sum(w * w for w in weights) / sum(weights)
            tau_squared = max(0.0, (q - df) / c_value) if c_value > 0 else 0.0
            random_weights = [1.0 / (value + tau_squared) for value in variances]
            pooled = sum(w * x for w, x in zip(random_weights, estimates)) / sum(random_weights)
            pooled_se = math.sqrt(1.0 / sum(random_weights))
            meta_analysis = {
                "model": "DerSimonian-Laird random effects",
                "effect_measure": next(iter(effect_measures)),
                "studies": len(rows),
                "pooled_effect": pooled,
                "standard_error": pooled_se,
                "confidence_interval_95pct_normal_approximation": [pooled - 1.96 * pooled_se, pooled + 1.96 * pooled_se],
                "cochran_q": q,
                "degrees_of_freedom": df,
                "tau_squared": tau_squared,
                "i_squared_percent": max(0.0, (q - df) / q * 100.0) if q > 0 else 0.0,
                "assumption": "Supplied effect estimates and standard errors are comparable and independent; this is not checked automatically.",
            }
        if len(included) < 2:
            meta_analysis_status = "not_performed_fewer_than_two_eligible_studies"
        elif not quantitative_complete:
            meta_analysis_status = "not_performed_incomplete_quantitative_effect_data"
        elif len(effect_measures) != 1:
            meta_analysis_status = "not_performed_incompatible_effect_measure_labels"
        else:
            meta_analysis_status = "performed"
        synthesis = {
            "review_id": review_id,
            "question": review["protocol"]["question"],
            "status": "synthesized",
            "screening_flow": {
                "records_retrieved": len(review["sources"]),
                "title_abstract_excluded": sum(item["decision"] == "exclude" for item in review["sources"]),
                "title_abstract_included": sum(item["decision"] == "include" for item in review["sources"]),
                "full_text_excluded": len(full_text_excluded),
                "included_in_synthesis": len(included),
            },
            "full_text_excluded": full_text_excluded,
            "included_studies": included,
            "meta_analysis": meta_analysis,
            "meta_analysis_status": meta_analysis_status,
            "narrative_synthesis": (
                "No studies met full-text eligibility; no quantitative synthesis was possible."
                if not included else
                "Structured evidence table generated from independently reviewed extraction records. Interpret results alongside risk-of-bias ratings and protocol limitations."
            ),
            "limitations": [
                "Reviewer IDs are operator-supplied labels, not authenticated identities.",
                "Risk-of-bias domains are a generic structured template, not a validated tool for every study design.",
                "Meta-analysis is performed only when at least two included studies share an effect-measure label and provide effect estimates with standard errors.",
                "The normal-approximation confidence interval and DerSimonian-Laird model may be inappropriate for some outcomes or small study counts.",
                "No publication-bias test, certainty-of-evidence grading or PRISMA compliance is claimed.",
            ],
        }
        self.store.complete_systematic_review_synthesis(review_id, synthesis)
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
        source_text_kind = "captured_full_text" if source.get("full_text") else "abstract_or_search_snippet"
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
        elif overlap >= 0.6 and source_text_kind == "captured_full_text":
            assessment = "candidate_support_requires_human_adjudication"
        elif overlap >= 0.6:
            assessment = "quote_found_in_abstract_or_snippet_full_text_required"
        else:
            assessment = "passage_found_but_weak_lexical_alignment"
        result = {
            "id": uuid4().hex,
            "claim": claim_text,
            "report_id": report_id,
            "source_url": source_url,
            "quoted_passage": quote,
            "source_text_kind": source_text_kind,
            "normalized_character_offset": position if exact_match else None,
            "lexical_claim_coverage": round(overlap, 4),
            "assessment": assessment,
            "human_decision": None,
            "review_status": "awaiting_second_reviewer",
            "reviewer_decisions": [],
            "limitations": "Exact quote matching and lexical overlap are deterministic screening aids, not semantic entailment. A passage can mention the same terms while contradicting or qualifying a claim.",
        }
        self.store.save_citation_evidence(result)
        return result

    def adjudicate_citation(
        self, evidence_id: str, reviewer_id: str, decision: str, rationale: str,
    ) -> dict[str, Any]:
        if decision not in {"supports", "contradicts", "insufficient"}:
            raise ValueError("Decision must be supports, contradicts or insufficient")
        if len(rationale.strip()) < 12:
            raise ValueError("Human adjudication requires a rationale of at least 12 characters")
        evidence = self.store.get_citation_evidence(evidence_id)
        if evidence is None:
            raise KeyError(f"Unknown citation evidence: {evidence_id}")
        if decision in {"supports", "contradicts"} and (
            evidence.get("normalized_character_offset") is None
            or evidence.get("source_text_kind") != "captured_full_text"
        ):
            raise ValueError("Support/contradiction decisions require an exact passage in captured full text; otherwise use insufficient")
        self.store.adjudicate_citation_evidence(evidence_id, reviewer_id, decision, rationale.strip())
        return self.store.get_citation_evidence(evidence_id)

    def resolve_citation_conflict(
        self, evidence_id: str, adjudicator_id: str, decision: str, rationale: str,
    ) -> dict[str, Any]:
        if decision not in {"supports", "contradicts", "insufficient"}:
            raise ValueError("Resolution must be supports, contradicts or insufficient")
        if len(rationale.strip()) < 12:
            raise ValueError("Resolution requires a rationale of at least 12 characters")
        self.store.resolve_citation_evidence(
            evidence_id, adjudicator_id, decision, rationale.strip()
        )
        return self.store.get_citation_evidence(evidence_id)

    def analyze_csv(
        self, experiment_id: str, csv_bytes: bytes, numeric_columns: list[str],
        group_column: str | None = None, comparison_groups: list[str] | None = None,
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
        comparison = None
        if group_column is not None:
            if group_column not in reader.fieldnames:
                raise ValueError("Group column does not exist in the CSV header")
            groups = comparison_groups or sorted({
                (row.get(group_column) or "").strip() for row in rows
                if (row.get(group_column) or "").strip()
            })
            if len(groups) != 2 or groups[0] == groups[1]:
                raise ValueError("Grouped comparison requires exactly two distinct group labels")
            if any(not (row.get(group_column) or "").strip() for row in rows):
                raise ValueError("Grouped comparison does not allow missing group labels")
            observed_groups = {
                (row.get(group_column) or "").strip() for row in rows
            }
            if observed_groups != set(groups):
                raise ValueError("CSV contains group labels outside the requested two-group comparison")
            comparison = {}
            for column in requested:
                grouped = {label: [] for label in groups}
                for row_number, row in enumerate(rows, start=2):
                    label = (row.get(group_column) or "").strip()
                    raw = (row.get(column) or "").strip()
                    if label not in grouped or not raw:
                        continue
                    try:
                        value = float(raw)
                    except ValueError as exc:
                        raise ValueError(f"Non-numeric value in {column!r} at CSV row {row_number}") from exc
                    if not math.isfinite(value):
                        raise ValueError(f"Non-finite value in {column!r} at CSV row {row_number}")
                    grouped[label].append(value)
                comparison[column] = {
                    "group_column": group_column,
                    "left_group": groups[0],
                    "right_group": groups[1],
                    "left_n": len(grouped[groups[0]]),
                    "right_n": len(grouped[groups[1]]),
                    **_welch_test(grouped[groups[0]], grouped[groups[1]]),
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
            "group_comparison": comparison,
        }
        fingerprint = sha256(
            json.dumps(reproducibility, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        artifact = {
            "experiment_id": experiment_id,
            "reproducibility_fingerprint": fingerprint,
            **reproducibility,
            "interpretation": "Descriptive statistics and optional Welch two-sample comparison. Statistical assumptions, sampling design and confounding are not validated; no causal inference or hypothesis confirmation is performed.",
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

