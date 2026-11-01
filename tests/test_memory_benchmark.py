from aether.memory_benchmark import run_benchmark


def test_memory_and_skill_retrieval_benchmark_passes_fixed_cases():
    result = run_benchmark()
    assert result["passed"] is True
    assert result["memory_retrieval"]["recall_at_k"] == 1.0
    assert result["skill_retrieval_retention"]["recall_at_k"] == 1.0
    assert result["draft_skill_leakage_count"] == 0
