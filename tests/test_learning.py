import json

import pytest

from aether.learning import SkillLearner
from aether.storage import Store


class FakeProvider:
    def chat(self, messages):
        if "Execute the supplied workflow procedure" in messages[0]["content"]:
            payload = json.loads(messages[-1]["content"])
            return json.dumps({"output": f"done:{payload['held_out_input']}"})
        return "1. Check prerequisites. 2. Perform the operation. 3. Verify the result."


def test_skill_requires_three_distinct_held_out_examples_before_retrieval(tmp_path):
    store = Store(tmp_path / "learning.sqlite3")
    learner = SkillLearner(store, FakeProvider())
    skill = learner.learn_from_text("Deploy app", "deployment guide", "Source steps...")
    assert skill["status"] == "draft"
    assert learner.retrieve("deploy app") == []

    learner.evaluate(skill["id"], True, "Manual reviewer says this looks correct")
    assert learner.retrieve("deploy app") == []

    first = learner.evaluate_held_out(skill["id"], "case one", "done:case one")
    assert first["status"] == "draft"
    duplicate = learner.evaluate_held_out(skill["id"], "case one", "done:case one")
    assert duplicate["status"] == "draft"
    second = learner.evaluate_held_out(skill["id"], "case two", "done:case two")
    assert second["status"] == "draft"
    third = learner.evaluate_held_out(skill["id"], "case three", "done:case three")

    assert third["status"] == "validated"
    assert learner.retrieve("deploy app")[0]["id"] == skill["id"]


def test_failed_held_out_example_prevents_or_revokes_validation(tmp_path):
    learner = SkillLearner(Store(tmp_path / "learning.sqlite3"), FakeProvider())
    skill = learner.learn_from_text("Format report", "style guide", "Apply style rules")
    for value in ("one", "two", "three"):
        learner.evaluate_held_out(skill["id"], value, f"done:{value}")
    assert learner.retrieve("format report")

    failed = learner.evaluate_held_out(skill["id"], "four", "wrong expected result")
    assert failed["last_held_out_result"]["passed"] is False
    assert failed["status"] == "draft"
    assert learner.retrieve("format report") == []


def test_manual_evaluation_never_validates_a_skill(tmp_path):
    learner = SkillLearner(Store(tmp_path / "learning.sqlite3"))
    skill = learner.learn_from_text("Backup", "manual", "Copy files and verify checksum")
    learner.evaluate(skill["id"], True, "Manual reviewer marked this as passed")
    assert learner.store.get_skill(skill["id"])["status"] == "draft"
    assert learner.retrieve("backup") == []


def test_held_out_evaluation_requires_provider_and_expected_output(tmp_path):
    learner = SkillLearner(Store(tmp_path / "learning.sqlite3"))
    skill = learner.learn_from_text("Backup", "manual", "Copy files")
    with pytest.raises(ValueError, match="requires an enabled model provider"):
        learner.evaluate_held_out(skill["id"], "input", "output")


def test_held_out_evaluation_rejects_empty_and_oversized_cases(tmp_path):
    learner = SkillLearner(Store(tmp_path / "learning.sqlite3"), FakeProvider())
    skill = learner.learn_from_text("Backup", "manual", "Copy files")
    with pytest.raises(ValueError, match="required"):
        learner.evaluate_held_out(skill["id"], "", "output")
    with pytest.raises(ValueError, match="4000 characters"):
        learner.evaluate_held_out(skill["id"], "x" * 4001, "output")


def test_failed_manual_skill_evaluation_keeps_skill_out_of_validated_memory(tmp_path):
    store = Store(tmp_path / "learning.sqlite3")
    learner = SkillLearner(store)
    skill = learner.learn_from_text("Backup", "manual", "Copy files and verify checksum")
    learner.evaluate(skill["id"], False, "Checksum mismatch")
    assert store.get_skill(skill["id"])["status"] == "draft"
    assert learner.retrieve("backup") == []
