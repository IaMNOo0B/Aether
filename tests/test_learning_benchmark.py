import pytest

from aether.learning import SkillLearner
from aether.learning_benchmark import HeldOutCase, run_skill_benchmark
from aether.storage import Store


def test_held_out_benchmark_compares_skill_to_baseline_and_hashes_dataset():
    cases = [
        HeldOutCase("a", "2+2", "4", category="in_domain"),
        HeldOutCase("b", "3+5", "8", category="transfer"),
    ]
    report = run_skill_benchmark(
        cases,
        lambda value: {"2+2": "4", "3+5": "8"}[value],
        lambda value: "wrong",
    )
    assert report["skill_accuracy"] == 1.0
    assert report["baseline_accuracy"] == 0.0
    assert report["accuracy_delta"] == 1.0
    assert len(report["dataset_sha256"]) == 64
    assert report["by_category"]["transfer"]["skill_accuracy"] == 1.0


def test_skill_is_validated_only_after_actual_held_out_executor_passes(tmp_path):
    learner = SkillLearner(Store(tmp_path / "skills.sqlite3"))
    skill = learner.learn_from_text("Arithmetic", "manual", "Add two integers and return the sum.")
    cases = [
        HeldOutCase("one", "2,2", "4"),
        HeldOutCase("two", "3,5", "8"),
    ]
    result = learner.benchmark(
        skill["id"], cases,
        executor=lambda procedure, value: str(sum(int(x) for x in value.split(","))),
        baseline_executor=lambda value: "0",
    )
    assert result["validated"] is True
    assert learner.store.get_skill(skill["id"])["status"] == "validated"


def test_benchmark_rejects_training_split_and_duplicate_ids():
    with pytest.raises(ValueError, match="held_out"):
        run_skill_benchmark(
            [HeldOutCase("train", "input", "output", split="train")],
            lambda value: value,
        )
    with pytest.raises(ValueError, match="unique"):
        run_skill_benchmark(
            [HeldOutCase("same", "a", "a"), HeldOutCase("same", "b", "b")],
            lambda value: value,
        )
