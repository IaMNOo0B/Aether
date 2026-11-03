from aether.learning import SkillLearner
from aether.storage import Store


class FakeProvider:
    def chat(self, messages):
        return "1. Check prerequisites.\n2. Perform the operation.\n3. Verify the result."


def test_skill_requires_held_out_evaluation_before_retrieval(tmp_path):
    store = Store(tmp_path / "learning.sqlite3")
    learner = SkillLearner(store, FakeProvider())
    skill = learner.learn_from_text("Deploy app", "deployment guide", "Source steps...")
    assert skill["status"] == "draft"
    assert learner.retrieve("deploy app") == []
    learner.evaluate(skill["id"], True, "Passed a new deployment scenario with expected output")
    assert learner.retrieve("deploy app")[0]["id"] == skill["id"]


def test_failed_skill_evaluation_keeps_skill_out_of_validated_memory(tmp_path):
    store = Store(tmp_path / "learning.sqlite3")
    learner = SkillLearner(store)
    skill = learner.learn_from_text("Backup", "manual", "Copy files and verify checksum")
    learner.evaluate(skill["id"], False, "Checksum mismatch")
    assert store.get_skill(skill["id"])["status"] == "draft"
    assert learner.retrieve("backup") == []

from aether.learning import SkillLearner
from aether.learning_benchmarks import SkillBenchmarkRunner, SkillEvaluationCase
from aether.storage import Store


def test_skill_benchmark_requires_held_out_transfer_and_retention(tmp_path):
    store = Store(tmp_path / "learning.sqlite3")
    skill = SkillLearner(store).learn_from_text(
        "Normalize text", "operator guide", "Strip spaces and lowercase the text."
    )
    runner = SkillBenchmarkRunner(store)
    cases = [
        SkillEvaluationCase("h1", "held_out", "  HELLO ", "hello"),
        SkillEvaluationCase("t1", "transfer", "  WORLD ", "world"),
        SkillEvaluationCase("r1", "retention", " AETHER ", "aether"),
    ]

    def executor(procedure, task):
        assert "lowercase" in procedure
        return task.strip().lower()

    report = runner.run(skill["id"], cases, executor, lambda actual, expected: actual == expected)
    assert report.validated is True
    assert report.accuracy == 1.0
    assert set(report.split_accuracy) == {"held_out", "transfer", "retention"}
    assert store.get_skill(skill["id"])["status"] == "validated"


def test_skill_benchmark_does_not_validate_when_transfer_fails(tmp_path):
    store = Store(tmp_path / "learning.sqlite3")
    skill = SkillLearner(store).learn_from_text("Sort values", "guide", "Sort ascending.")
    runner = SkillBenchmarkRunner(store)
    cases = [
        SkillEvaluationCase("h1", "held_out", [2, 1], [1, 2]),
        SkillEvaluationCase("t1", "transfer", [3, 1], [1, 3]),
        SkillEvaluationCase("r1", "retention", [4, 2], [2, 4]),
    ]
    report = runner.run(
        skill["id"], cases, lambda procedure, task: list(task),
        lambda actual, expected: actual == expected,
    )
    assert report.validated is False
    assert report.split_accuracy["transfer"] == 0.0
    assert store.get_skill(skill["id"])["status"] == "draft"


def test_skill_benchmark_rejects_missing_splits_and_duplicate_ids(tmp_path):
    store = Store(tmp_path / "learning.sqlite3")
    skill = SkillLearner(store).learn_from_text("Workflow", "guide", "Step one.")
    runner = SkillBenchmarkRunner(store)
    executor = lambda procedure, task: task
    scorer = lambda actual, expected: actual == expected
    with pytest.raises(ValueError, match="requires held_out"):
        runner.run(
            skill["id"], [SkillEvaluationCase("one", "held_out", 1, 1)],
            executor, scorer,
        )
    duplicates = [
        SkillEvaluationCase("same", split, 1, 1)
        for split in ("held_out", "transfer", "retention")
    ]
    duplicates[1] = SkillEvaluationCase("same", "transfer", 1, 1)
    with pytest.raises(ValueError, match="unique"):
        runner.run(skill["id"], duplicates, executor, scorer)
