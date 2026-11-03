import json

import pytest

from aether.continual_learning import ContinualLearningManager


class FakeTrainer:
    trainer_id = "fixture-lora-v1"

    def train(self, base_model, training_data, output_directory, config):
        artifact = output_directory / "adapter.bin"
        artifact.write_bytes(b"adapter weights")
        return output_directory


def test_external_adapter_training_requires_approval_and_heldout_forgetting_guard(tmp_path):
    training = tmp_path / "train.jsonl"
    new_eval = tmp_path / "new.jsonl"
    old_eval = tmp_path / "old.jsonl"
    training.write_text('{"prompt":"train"}\n', encoding="utf-8")
    new_eval.write_text('{"prompt":"new-heldout"}\n', encoding="utf-8")
    old_eval.write_text('{"prompt":"retained-heldout"}\n', encoding="utf-8")
    manager = ContinualLearningManager(tmp_path / "artifacts")

    with pytest.raises(PermissionError, match="explicit approval"):
        manager.train_adapter(
            base_model="base-model",
            training_data=training,
            evaluation_suites={"new_task": new_eval, "retained_task": old_eval},
            trainer=FakeTrainer(), evaluator=lambda model, suite: 0.9,
            config={"rank": 8}, seed=7,
        )

    def evaluator(model, suite):
        if "new" in suite.name:
            return 0.92
        return 0.91 if model == "base-model" else 0.89

    result = manager.train_adapter(
        base_model="base-model",
        training_data=training,
        evaluation_suites={"new_task": new_eval, "retained_task": old_eval},
        trainer=FakeTrainer(), evaluator=evaluator,
        config={"rank": 8}, seed=7, approved=True,
    )
    assert result.status == "accepted"
    assert result.new_task_score == 0.92
    assert result.maximum_forgetting == pytest.approx(0.02)
    saved = json.loads((tmp_path / "artifacts" / result.run_id / "manifest.json").read_text())
    assert saved["adapter_sha256"] == result.adapter_sha256


def test_external_adapter_is_rejected_for_catastrophic_forgetting(tmp_path):
    training = tmp_path / "train.jsonl"
    new_eval = tmp_path / "new.jsonl"
    old_eval = tmp_path / "old.jsonl"
    for path in (training, new_eval, old_eval):
        path.write_text(path.name, encoding="utf-8")
    result = ContinualLearningManager(tmp_path / "artifacts").train_adapter(
        base_model="base",
        training_data=training,
        evaluation_suites={"new_task": new_eval, "old_task": old_eval},
        trainer=FakeTrainer(),
        evaluator=lambda model, suite: (
            0.95 if suite.name == "new.jsonl" else
            0.95 if model == "base" else 0.70
        ),
        config={"method": "qlora"}, seed=1, approved=True,
        maximum_forgetting=0.05,
    )
    assert result.status == "rejected_regression"
    assert result.maximum_forgetting == pytest.approx(0.25)


def test_external_adapter_rejects_training_data_leakage(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("same data", encoding="utf-8")
    with pytest.raises(ValueError, match="must not be reused"):
        ContinualLearningManager(tmp_path / "artifacts").train_adapter(
            base_model="base", training_data=dataset,
            evaluation_suites={"new_task": dataset, "old_task": dataset},
            trainer=FakeTrainer(), evaluator=lambda model, suite: 1.0,
            config={"rank": 4}, seed=1, approved=True,
        )
