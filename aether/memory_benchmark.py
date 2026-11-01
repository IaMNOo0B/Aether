"""Deterministic retrieval and procedural-skill retention benchmark.

This evaluates retrieval behavior on a fixed, inspectable corpus. It does not
measure whether a model can execute a procedure correctly or generalize to
unseen real-world tasks.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any

from aether.learning import SkillLearner
from aether.memory import MemoryManager
from aether.storage import Store


CASES_PATH = Path(__file__).resolve().parent.parent / "benchmarks" / "memory_cases.json"


def run_benchmark(cases_path: str | Path = CASES_PATH) -> dict[str, Any]:
    cases = json.loads(Path(cases_path).read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="aether-memory-benchmark-") as directory:
        store = Store(Path(directory) / "benchmark.sqlite3")
        memory = MemoryManager(store)
        for item in cases["memories"]:
            memory.remember(
                item["content"], source="benchmark", kind=item.get("kind", "semantic"),
                tags=item.get("tags", []), status="benchmark",
            )

        learner = SkillLearner(store)
        skill_ids: dict[str, str] = {}
        for item in cases["skills"]:
            skill = learner.learn_from_text(
                item["title"], "fixed benchmark fixture", item["procedure"]
            )
            if item.get("validated", False):
                learner.evaluate(
                    skill["id"], True,
                    "Passed the fixture's held-out benchmark scenario.",
                )
            skill_ids[item["key"]] = skill["id"]

        memory_ranks = []
        for query in cases["memory_queries"]:
            results = memory.search(query["query"], limit=cases.get("top_k", 5))
            expected = query["expected_content"]
            rank = next(
                (index for index, result in enumerate(results, start=1)
                 if result["content"] == expected),
                None,
            )
            memory_ranks.append(rank)

        skill_ranks = []
        for query in cases["skill_queries"]:
            results = learner.retrieve(query["query"], limit=cases.get("top_k", 5))
            expected_id = skill_ids[query["expected_skill"]]
            rank = next(
                (index for index, result in enumerate(results, start=1)
                 if result["id"] == expected_id),
                None,
            )
            skill_ranks.append(rank)

        draft_ids = {
            skill_ids[item["key"]] for item in cases["skills"]
            if not item.get("validated", False)
        }
        draft_results = learner.retrieve(
            cases.get("draft_exclusion_query", "unreviewed draft workflow"), limit=20
        )
        draft_leakage = sum(item["id"] in draft_ids for item in draft_results)

    def metrics(ranks: list[int | None]) -> dict[str, float]:
        total = len(ranks) or 1
        return {
            "recall_at_k": round(sum(rank is not None for rank in ranks) / total, 4),
            "mean_reciprocal_rank": round(
                sum((1 / rank) if rank is not None else 0 for rank in ranks) / total, 4
            ),
        }

    result = {
        "memory_retrieval": metrics(memory_ranks),
        "skill_retrieval_retention": metrics(skill_ranks),
        "draft_skill_leakage_count": draft_leakage,
        "memory_query_ranks": memory_ranks,
        "skill_query_ranks": skill_ranks,
        "limitations": [
            "Fixed lexical benchmark corpus; no held-out natural-language paraphrase generation.",
            "Skill retention means validated-skill retrievability, not successful procedure execution.",
            "No embedding model quality, long-term forgetting curve, or model-weight adaptation is measured.",
        ],
    }
    result["passed"] = (
        result["memory_retrieval"]["recall_at_k"] == 1.0
        and result["skill_retrieval_retention"]["recall_at_k"] == 1.0
        and draft_leakage == 0
    )
    return result


def main() -> int:
    result = run_benchmark()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
