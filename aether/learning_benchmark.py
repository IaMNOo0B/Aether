"""Deterministic evaluation for procedural-skill retrieval.

This measures retrieval behavior only. It does not claim model-weight learning,
task execution competence, or generalization beyond the supplied fixture.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any

from aether.learning import SkillLearner
from aether.storage import Store


CASES: tuple[dict[str, Any], ...] = (
    {
        "title": "Windows application deployment",
        "source": "deployment fixture",
        "material": "Build package, verify signature, install application, smoke-test launch.",
        "query": "package install application launch",
        "relevant": "windows application deployment",
    },
    {
        "title": "SQLite database backup",
        "source": "database fixture",
        "material": "Use online backup, verify integrity, restore to a staging database.",
        "query": "sqlite backup restore integrity",
        "relevant": "sqlite database backup",
    },
    {
        "title": "Video subtitle extraction",
        "source": "media fixture",
        "material": "Inspect media metadata, extract audio, transcribe, align subtitle timestamps.",
        "query": "extract audio transcribe subtitle timestamps",
        "relevant": "video subtitle extraction",
    },
)


def run_learning_benchmark(root: str | Path | None = None, k: int = 3) -> dict[str, Any]:
    """Run a reproducible synthetic retrieval benchmark in an isolated SQLite DB."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if root is None:
        temporary = tempfile.TemporaryDirectory(prefix="aether-learning-bench-")
        db_path = Path(temporary.name) / "benchmark.sqlite3"
    else:
        temporary = None
        directory = Path(root)
        directory.mkdir(parents=True, exist_ok=True)
        db_path = directory / "learning-benchmark.sqlite3"

    try:
        store = Store(db_path)
        learner = SkillLearner(store)
        skill_ids: dict[str, str] = {}
        for case in CASES:
            skill = learner.learn_from_text(case["title"], case["source"], case["material"])
            skill_ids[case["relevant"]] = skill["id"]
            learner.evaluate(
                skill["id"], True,
                "Synthetic benchmark fixture: stored workflow matches its declared scenario.",
            )

        draft = learner.learn_from_text(
            "Unvalidated credential rotation",
            "draft-only fixture",
            "Rotate credentials and verify access.",
        )
        results = []
        true_positives = 0
        for case in CASES:
            retrieved = learner.retrieve(case["query"], limit=k)
            retrieved_ids = [item["id"] for item in retrieved]
            expected_id = skill_ids[case["relevant"]]
            hit = expected_id in retrieved_ids
            true_positives += int(hit)
            results.append({
                "query": case["query"],
                "expected_skill": case["title"],
                "retrieved_titles": [item["title"] for item in retrieved],
                "hit": hit,
            })

        draft_excluded = all(
            item["id"] != draft["id"]
            for case in CASES
            for item in learner.retrieve(case["query"], limit=100)
        )
        total = len(CASES)
        return {
            "benchmark": "aether-procedural-retrieval-v1",
            "scope": "synthetic lexical retrieval; not execution, model training, or real-world transfer",
            "k": k,
            "queries": total,
            "hits": true_positives,
            "recall_at_k": true_positives / total if total else 0.0,
            "draft_skill_excluded": draft_excluded,
            "complete": true_positives == total and draft_excluded,
            "results": results,
        }
    finally:
        if temporary is not None:
            temporary.cleanup()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run AETHER procedural retrieval benchmark.")
    parser.add_argument("--root", help="Optional directory for a persistent benchmark database.")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args(argv)
    report = run_learning_benchmark(args.root, args.k)
    print(json.dumps(report, indent=2))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
