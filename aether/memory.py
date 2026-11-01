"""Typed local memory with BM25 lexical and optional embedding retrieval."""
from __future__ import annotations

from collections import Counter
from math import log, sqrt
import re
from typing import Any

from aether.embeddings import EmbeddingProvider, EmbeddingUnavailable
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


def _cosine(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return None
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


class MemoryManager:
    """Manage typed memories, ranked retrieval, optional embeddings and corrections."""

    def __init__(
        self, store: Store, corpus_limit: int = 5000,
        embedding_provider: EmbeddingProvider | None = None,
        semantic_weight: float = 0.45,
    ):
        if not 0 <= semantic_weight <= 1:
            raise ValueError("semantic_weight must be between zero and one")
        self.store = store
        self.corpus_limit = max(100, min(corpus_limit, 20000))
        self.embedding_provider = embedding_provider
        self.semantic_weight = semantic_weight

    def _index_one(self, memory_id: int, content: str) -> bool:
        if self.embedding_provider is None:
            return False
        vector = self.embedding_provider.embed(content)
        self.store.save_memory_embedding(
            memory_id, self.embedding_provider.model, vector
        )
        return True

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
        memory_id = self.store.add_memory(
            text, source=source, status=status, metadata=record
        )
        if self.embedding_provider is not None:
            try:
                self._index_one(memory_id, text)
            except EmbeddingUnavailable:
                # Preserve the memory; lexical retrieval remains available.
                pass
        return memory_id

    def index_pending(self, limit: int = 100) -> dict[str, Any]:
        if self.embedding_provider is None:
            raise ValueError("Semantic memory is disabled; configure an embedding provider first")
        if not 1 <= limit <= 1000:
            raise ValueError("Embedding index limit must be between 1 and 1000")
        pending = self.store.list_memories_without_embedding(
            self.embedding_provider.model, limit
        )
        indexed = 0
        for item in pending:
            try:
                self._index_one(item["id"], item["content"])
                indexed += 1
            except EmbeddingUnavailable as exc:
                return {
                    "indexed": indexed, "remaining": len(pending) - indexed,
                    "error": str(exc),
                }
        remaining = len(self.store.list_memories_without_embedding(
            self.embedding_provider.model, 1000
        ))
        return {"indexed": indexed, "remaining": remaining, "error": None}

    def search(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip() or limit < 1:
            return []
        terms = _tokens(query)
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
        lexical_scores: dict[int, float] = {}
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
            matched_tags = query_terms.intersection(
                str(tag).casefold() for tag in item["metadata"].get("tags", [])
            )
            lexical_scores[item["id"]] = score + 0.5 * len(matched_tags)
        max_lexical = max(lexical_scores.values(), default=0.0)

        query_vector = None
        embedding_map: dict[int, list[float]] = {}
        if self.embedding_provider is not None and self.semantic_weight > 0:
            try:
                query_vector = self.embedding_provider.embed(query)
                embedding_map = {
                    item["memory_id"]: item["vector"]
                    for item in self.store.list_memory_embeddings(
                        self.embedding_provider.model, self.corpus_limit
                    )
                }
            except EmbeddingUnavailable:
                # An unavailable semantic backend must not reduce lexical recall.
                query_vector = None
                embedding_map = {}

        # A query with no lexical tokens is valid when semantic retrieval is available.
        if not terms and query_vector is None:
            return []

        ranked = []
        for item, _, _ in documents:
            lexical = lexical_scores[item["id"]]
            lexical_normalized = lexical / max_lexical if max_lexical else 0.0
            semantic = None
            if query_vector is not None and item["id"] in embedding_map:
                semantic = _cosine(query_vector, embedding_map[item["id"]])
            if query_vector is None:
                relevance = lexical
            elif semantic is None:
                # Do not penalize records that have not yet been indexed or have
                # incompatible vector dimensions; retain their full lexical score.
                relevance = lexical_normalized
            else:
                relevance = (
                    lexical_normalized * (1 - self.semantic_weight)
                    + max(0.0, semantic) * self.semantic_weight
                )
            if relevance <= 0:
                continue
            ranked.append((relevance, item["id"], lexical, semantic, item))
        ranked.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
        return [
            {
                **item,
                "relevance": round(relevance, 6),
                "lexical_score": round(lexical, 6),
                "semantic_similarity": (
                    round(semantic, 6) if semantic is not None else None
                ),
                "kind": item["metadata"].get("kind", "semantic"),
                "confidence": item["metadata"].get("confidence"),
                "tags": item["metadata"].get("tags", []),
            }
            for relevance, _, lexical, semantic, item in ranked[:limit]
        ]

    def versions(self, memory_id: int) -> list[dict[str, Any]]:
        """Return the correction lineage from the original memory to its latest version."""
        return self.store.list_memory_versions(memory_id)

    def find_contradictions(self, memory_id: int | None = None) -> list[dict[str, Any]]:
        """Find explicit, user-supplied fact slots that disagree; never infer truth from prose.

        Memories may carry metadata['facts'] as [{subject, predicate, object}].
        Only exact subject/predicate matches with different normalized objects are flagged.
        """
        records = self.store.list_memories(self.corpus_limit)
        active = [item for item in records
                  if item["status"] not in {"superseded", "rejected", "deleted"}]
        if memory_id is not None:
            target = self.store.get_memory(memory_id)
            if target is None:
                raise KeyError(f"Unknown memory: {memory_id}")
            if target["status"] in {"superseded", "rejected", "deleted"}:
                return []
            active = [item for item in active if item["id"] == memory_id or item["id"] != memory_id]
        facts: dict[tuple[str, str], list[tuple[dict[str, Any], str]]] = {}
        for item in active:
            for fact in item["metadata"].get("facts", []):
                if not isinstance(fact, dict):
                    continue
                subject = " ".join(str(fact.get("subject", "")).casefold().split())
                predicate = " ".join(str(fact.get("predicate", "")).casefold().split())
                value = " ".join(str(fact.get("object", "")).casefold().split())
                if subject and predicate and value:
                    facts.setdefault((subject, predicate), []).append((item, value))
        conflicts = []
        for (subject, predicate), entries in facts.items():
            for index, (left, left_value) in enumerate(entries):
                for right, right_value in entries[index + 1:]:
                    if left_value == right_value:
                        continue
                    if memory_id is not None and memory_id not in {left["id"], right["id"]}:
                        continue
                    conflicts.append({
                        "subject": subject, "predicate": predicate,
                        "left_memory_id": left["id"], "left_value": left_value,
                        "right_memory_id": right["id"], "right_value": right_value,
                        "resolution": "unresolved; explicit review required",
                    })
        return conflicts

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
        metadata = {
            "kind": old_metadata.get("kind", "semantic"),
            "confidence": None,
            "tags": old_metadata.get("tags", []),
            "schema_version": 1,
            "correction_reason": explanation,
        }
        new_id = self.store.supersede_memory(
            memory_id, correction, explanation,
            source="user-correction", status="user-corrected",
            metadata=metadata,
        )
        if self.embedding_provider is not None:
            try:
                self._index_one(new_id, correction)
            except EmbeddingUnavailable:
                pass
        return new_id
