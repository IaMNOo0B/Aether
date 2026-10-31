"""Structured, provenance-aware lexical memory retrieval.

This is a transparent local retrieval baseline, not semantic embedding search.
"""
from __future__ import annotations

from collections import Counter
from math import log
import re
from typing import Any

from aether.storage import Store


MEMORY_KINDS = {"working", "episodic", "semantic", "procedural", "prospective"}
STOPWORDS = {
    "the", "and", "for", "that", "with", "this", "from", "have", "your",
    "you", "are", "was", "were", "will", "would", "could", "should", "into",
    "about", "what", "when", "where", "which", "while", "then", "than",
    "them", "they", "their", "there", "here", "how", "why", "who", "does",
    "did", "not", "but", "can", "all", "any", "has", "had", "its", "our",
}


def _tokens(text: str) -> list[str]:
    return [
        token for token in re.findall(r"[a-z0-9_]{2,}", text.casefold())
        if token not in STOPWORDS
    ]


class MemoryManager:
    """Manage typed memories, rank retrieval and preserve correction lineage."""

    def __init__(self, store: Store, corpus_limit: int = 5000):
        self.store = store
        self.corpus_limit = max(100, min(corpus_limit, 20000))

    def remember(
        self, content: str, *, source: str = "user", kind: str = "semantic",
        status: str = "unverified", confidence: float | None = None,
        tags: list[str] | None = None, metadata: dict[str, Any] | None = None,
    ) -> int:
        text = content.strip()
        if not text:
            raise ValueError("Memory content cannot be empty")
        if kind not in MEMORY_KINDS:
            raise ValueError(f"Unsupported memory kind: {kind}")
        if confidence is not None and (
            isinstance(confidence, bool) or not isinstance(confidence, (int, float))
            or not 0 <= confidence <= 1
        ):
            raise ValueError("Memory confidence must be between zero and one")
        normalized_tags = []
        for tag in tags or []:
            value = str(tag).strip().casefold()
            if value and value not in normalized_tags:
                normalized_tags.append(value)
        record = dict(metadata or {})
        record.update({
            "kind": kind,
            "confidence": confidence,
            "tags": normalized_tags,
            "schema_version": 1,
        })
        return self.store.add_memory(text, source=source, status=status, metadata=record)

    def search(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        terms = _tokens(query)
        if not terms or limit < 1:
            return []
        records = [
            item for item in self.store.list_memories(self.corpus_limit)
            if item["status"] not in {"superseded", "rejected", "deleted"}
        ]
        if not records:
            return []
        documents = []
        document_frequency: Counter[str] = Counter()
        for item in records:
            meta = item["metadata"]
            searchable = " ".join([
                item["content"],
                " ".join(str(tag) for tag in meta.get("tags", [])),
                str(meta.get("kind", "")),
            ])
            tokens = _tokens(searchable)
            counts = Counter(tokens)
            documents.append((item, counts, len(tokens)))
            document_frequency.update(set(counts))
        average_length = sum(length for _, _, length in documents) / len(documents) or 1.0
        query_terms = set(terms)
        ranked = []
        for item, counts, length in documents:
            score = 0.0
            for term in query_terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                idf = log(1 + (len(documents) - document_frequency[term] + 0.5)
                          / (document_frequency[term] + 0.5))
                denominator = frequency + 1.5 * (1 - 0.75 + 0.75 * length / average_length)
                score += idf * frequency * 2.5 / denominator
            if score <= 0:
                continue
            meta = item["metadata"]
            matched_tags = query_terms.intersection(
                str(tag).casefold() for tag in meta.get("tags", [])
            )
            score += 0.5 * len(matched_tags)
            ranked.append((score, item["id"], item))
        ranked.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
        results = []
        for score, _, item in ranked[:limit]:
            results.append({
                **item,
                "relevance": round(score, 6),
                "kind": item["metadata"].get("kind", "semantic"),
                "confidence": item["metadata"].get("confidence"),
                "tags": item["metadata"].get("tags", []),
            })
        return results

    def correct(self, memory_id: int, corrected_content: str, reason: str) -> int:
        """Supersede a memory without deleting history or rewriting its provenance."""
        previous = self.store.get_memory(memory_id)
        if previous is None:
            raise KeyError(f"Unknown memory: {memory_id}")
        correction = corrected_content.strip()
        explanation = reason.strip()
        if not correction or not explanation:
            raise ValueError("A correction requires replacement content and a reason")
        old_metadata = dict(previous["metadata"])
        old_metadata["superseded_by_reason"] = explanation
        self.store.update_memory(memory_id, status="superseded", metadata=old_metadata)
        return self.remember(
            correction,
            source="user-correction",
            kind=old_metadata.get("kind", "semantic"),
            status="user-corrected",
            tags=old_metadata.get("tags", []),
            metadata={
                "supersedes_memory_id": memory_id,
                "correction_reason": explanation,
            },
        )
