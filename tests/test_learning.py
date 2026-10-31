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
