from aether.learning_benchmark import run_learning_benchmark


def test_learning_benchmark_is_deterministic_and_excludes_drafts():
    first = run_learning_benchmark()
    second = run_learning_benchmark()
    assert first["benchmark"] == "aether-procedural-retrieval-v1"
    assert first["recall_at_k"] == 1.0
    assert first["draft_skill_excluded"] is True
    assert first["complete"] is True
    assert first["results"] == second["results"]


def test_learning_benchmark_rejects_invalid_k():
    import pytest

    with pytest.raises(ValueError, match="k must be at least 1"):
        run_learning_benchmark(k=0)
