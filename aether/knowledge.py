"""Provenance-aware knowledge graph with temporal claims and conflict surfacing."""
from __future__ import annotations

from datetime import date, datetime
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4
import json
import re
import unicodedata

from aether.storage import Store


CLAIM_TYPES = {"fact", "hypothesis", "observation", "prediction"}


def normalize_name(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _validate_date(value: str | None, label: str) -> str | None:
    if value is None or value == "":
        return None
    try:
        date.fromisoformat(value[:10])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an ISO date or datetime") from exc
    return value


class KnowledgeGraph:
    """Store claims as unverified by default; verification always needs review evidence."""

    def __init__(self, store: Store, provider: Any | None = None):
        self.store = store
        self.provider = provider

    def add_entity(
        self, name: str, entity_type: str = "concept", description: str = "",
    ) -> dict[str, Any]:
        clean_name = " ".join(name.split())
        clean_type = entity_type.strip().casefold()
        if not clean_name or len(clean_name) > 250:
            raise ValueError("Entity name must contain 1..250 characters")
        if not clean_type or len(clean_type) > 80:
            raise ValueError("Entity type must contain 1..80 characters")
        entity_id = uuid4().hex
        normalized = normalize_name(clean_name)
        saved_id = self.store.upsert_knowledge_entity(
            entity_id, clean_name, normalized, clean_type, description.strip()[:2000]
        )
        entity = self.store.get_knowledge_entity(saved_id)
        if entity is None:
            raise RuntimeError("Knowledge entity could not be read after saving")
        return entity

    def add_claim(
        self, subject: str, predicate: str, object_value: str, *,
        claim_type: str = "fact", source_url: str = "",
        source_title: str = "", evidence_class: str = "user-provided",
        confidence: float | None = None, valid_from: str | None = None,
        valid_to: str | None = None, source_report_id: str | None = None,
        object_is_entity: bool = False,
    ) -> dict[str, Any]:
        clean_predicate = " ".join(predicate.split()).casefold()
        clean_object = " ".join(object_value.split())
        if not clean_predicate or len(clean_predicate) > 150:
            raise ValueError("Predicate must contain 1..150 characters")
        if not clean_object or len(clean_object) > 1000:
            raise ValueError("Object value must contain 1..1000 characters")
        if claim_type not in CLAIM_TYPES:
            raise ValueError(f"Claim type must be one of: {', '.join(sorted(CLAIM_TYPES))}")
        if confidence is not None and (
            isinstance(confidence, bool) or not isinstance(confidence, (int, float))
            or not 0 <= confidence <= 1
        ):
            raise ValueError("Confidence must be between zero and one")
        parsed = urlparse(source_url) if source_url else None
        if parsed and (parsed.scheme not in {"http", "https"} or not parsed.netloc):
            raise ValueError("Source URL must be an absolute HTTP or HTTPS URL")
        start = _validate_date(valid_from, "valid_from")
        end = _validate_date(valid_to, "valid_to")
        if start and end and end[:10] < start[:10]:
            raise ValueError("valid_to cannot be earlier than valid_from")

        subject_entity = self.add_entity(subject)
        object_entity = self.add_entity(clean_object) if object_is_entity else None
        claim_id = uuid4().hex
        self.store.insert_knowledge_claim({
            "id": claim_id,
            "subject_id": subject_entity["id"],
            "predicate": clean_predicate,
            "object_value": clean_object,
            "object_entity_id": object_entity["id"] if object_entity else None,
            "claim_type": claim_type,
            "confidence": confidence,
            "status": "unverified",
            "valid_from": start,
            "valid_to": end,
            "source_url": source_url,
            "source_title": source_title[:500],
            "evidence_class": evidence_class[:120],
            "source_report_id": source_report_id,
            "creation_note": "Captured with provenance; awaiting verification.",
        })
        claim = self.store.get_knowledge_claim(claim_id)
        if claim is None:
            raise RuntimeError("Knowledge claim could not be read after saving")
        return claim

    def review_claim(self, claim_id: str, status: str, evidence: str) -> dict[str, Any]:
        self.store.update_knowledge_claim_status(claim_id, status, evidence)
        claim = self.store.get_knowledge_claim(claim_id)
        if claim is None:
            raise RuntimeError("Reviewed claim disappeared")
        return claim

    @staticmethod
    def _intervals_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
        left_start = left.get("valid_from") or "0001-01-01"
        left_end = left.get("valid_to") or "9999-12-31"
        right_start = right.get("valid_from") or "0001-01-01"
        right_end = right.get("valid_to") or "9999-12-31"
        return left_start[:10] <= right_end[:10] and right_start[:10] <= left_end[:10]

    def find_conflicts(
        self, subject: str, predicate: str | None = None,
    ) -> list[dict[str, Any]]:
        entity = self.store.find_knowledge_entity(normalize_name(subject))
        if entity is None:
            return []
        claims = self.store.list_knowledge_claims(subject_id=entity["id"], limit=5000)
        active = [
            claim for claim in claims
            if claim["status"] != "retracted"
            and (predicate is None or claim["predicate"] == predicate.casefold())
        ]
        conflicts = []
        for index, left in enumerate(active):
            for right in active[index + 1:]:
                if left["claim_type"] != right["claim_type"]:
                    continue
                if normalize_name(left["object_value"]) == normalize_name(right["object_value"]):
                    continue
                if self._intervals_overlap(left, right):
                    conflicts.append({
                        "subject": entity["name"],
                        "predicate": left["predicate"],
                        "claim_type": left["claim_type"],
                        "left": left,
                        "right": right,
                        "resolution": "unresolved; human/source review required",
                    })
        return conflicts

    def entity_view(self, name: str) -> dict[str, Any] | None:
        entity = self.store.find_knowledge_entity(normalize_name(name))
        if entity is None:
            return None
        claims = self.store.list_knowledge_claims(limit=5000)
        outgoing = [claim for claim in claims if claim["subject_id"] == entity["id"]]
        incoming = [claim for claim in claims if claim.get("object_entity_id") == entity["id"]]
        return {
            "entity": entity,
            "outgoing_claims": outgoing,
            "incoming_claims": incoming,
            "conflicts": self.find_conflicts(entity["name"]),
        }

    def search_entities(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        return self.store.search_knowledge_entities(query, limit)

    def extract_from_report(self, report_id: str, max_claims: int = 20) -> list[dict[str, Any]]:
        if self.provider is None:
            raise RuntimeError("Knowledge extraction requires an enabled model provider")
        report = self.store.get_research_report(report_id)
        if report is None:
            raise KeyError(f"Unknown research report: {report_id}")
        sources = report["sources"]
        if not sources:
            return []
        selected = sources[:max(1, min(max_claims, 20))]
        source_context = [
            {
                "source_index": index,
                "title": source["title"],
                "url": source["url"],
                "evidence_class": source["evidence_type"],
                "summary": source["summary"][:1400],
            }
            for index, source in enumerate(selected)
        ]
        response = self.provider.chat([
            {
                "role": "system",
                "content": (
                    "Extract atomic, source-grounded knowledge claims from the supplied research "
                    "summaries. Return ONLY JSON: {\"claims\":[{\"subject\":\"...\","
                    "\"predicate\":\"...\",\"object\":\"...\","
                    "\"claim_type\":\"fact|hypothesis|observation|prediction\","
                    "\"source_index\":0}]}. Only use the supplied source_index values. "
                    "Do not invent facts, URLs, dates, confidence scores or citations. "
                    "If a claim is speculative, classify it as hypothesis. Omit unsupported claims."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"research_question": report["query"], "sources": source_context}),
            },
        ])
        try:
            payload = json.loads(response)
            proposed = payload["claims"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError("Knowledge extractor returned invalid JSON") from exc
        if not isinstance(proposed, list):
            raise ValueError("Knowledge extractor claims must be a JSON array")

        stored = []
        for item in proposed[:max_claims]:
            if not isinstance(item, dict):
                continue
            source_index = item.get("source_index")
            if isinstance(source_index, bool) or not isinstance(source_index, int):
                continue
            if not 0 <= source_index < len(selected):
                continue
            source = selected[source_index]
            try:
                stored.append(self.add_claim(
                    str(item.get("subject", "")),
                    str(item.get("predicate", "")),
                    str(item.get("object", "")),
                    claim_type=str(item.get("claim_type", "fact")),
                    source_url=source["url"],
                    source_title=source["title"],
                    evidence_class=source["evidence_type"],
                    source_report_id=report_id,
                ))
            except ValueError:
                continue
        return stored
