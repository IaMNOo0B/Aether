import json
import wave

import pytest

from aether.audit import AuditJournal
from aether.goals import GoalManager
from aether.learning_eval import LearningCase, compare_learning_runs, evaluate_executor
from aether.perception import AudioPerception, PerceptionSuite
from aether.rlcd_training import (
    RLCDConfigurationError, RLCDRunConfig, RLCDTrainingAdapter, prepare_dataset,
)
from aether.sandbox import DockerSandbox


def test_rlcd_dataset_splits_are_reproducible_disjoint_and_hashed(tmp_path):
    records = [{"prompt": f"question {i}", "label": str(i % 2), "rationale": f"reason {i}"}
               for i in range(120)]
    first = prepare_dataset(records, tmp_path / "one.json", seed=12)
    second = prepare_dataset(records, tmp_path / "two.json", seed=12)
    left = json.loads((tmp_path / "one.json").read_text())
    right = json.loads((tmp_path / "two.json").read_text())
    assert first["sha256"] == second["sha256"]
    assert left["split_counts"] == right["split_counts"]
    splits = left["splits"]
    hashes = [{row["_aether_example_hash"] for row in splits[name]}
              for name in ("train", "validation", "test")]
    assert not hashes[0] & hashes[1]
    assert not hashes[0] & hashes[2]
    assert not hashes[1] & hashes[2]
    assert all(hashes)


def test_rlcd_training_requires_external_compute_approval_and_unchanged_dataset(tmp_path):
    records = [{"prompt": f"question {i}", "label": str(i % 2)} for i in range(120)]
    prepare_dataset(records, tmp_path / "data" / "split.json")
    adapter = RLCDTrainingAdapter(tmp_path)
    config = RLCDRunConfig("model/example", "data/split.json", "runs/output")
    adapter.create_manifest(config, "runs/manifest.json")
    with pytest.raises(PermissionError, match="external GPU"):
        adapter.run("runs/manifest.json", ["python", "runner.py"])
    (tmp_path / "data" / "split.json").write_text("{}")
    with pytest.raises(RLCDConfigurationError, match="Dataset changed"):
        adapter.run("runs/manifest.json", ["python", "runner.py"],
                    approved=True, external_compute_confirmed=True)


def test_learning_eval_reports_held_out_transfer_and_before_after_delta():
    cases = [
        LearningCase("h1", "do alpha", "alpha", "held_out", "same"),
        LearningCase("t1", "do beta", "beta", "transfer", "new"),
        LearningCase("r1", "do gamma", "gamma", "retention", "same"),
    ]
    report = evaluate_executor(cases, lambda task: task.split()[-1],
                               system_version="v2", repetitions=2)
    assert report["metrics"]["held_out"]["pass_rate"] == 1
    assert report["metrics"]["transfer"]["cases"] == 1
    assert report["metrics"]["retention"]["trials"] == 2
    before = {"metrics": {"held_out": {"pass_rate": 0.5}}}
    after = {"metrics": {"held_out": {"pass_rate": 1.0}}}
    assert compare_learning_runs(before, after)["improved"] is True


def test_hash_chained_audit_detects_record_tampering(tmp_path):
    path = tmp_path / "audit.jsonl"
    journal = AuditJournal(path)
    journal.append("task.started", {"task_id": "abc"})
    journal.append("task.finished", {"status": "completed"})
    assert journal.verify()["valid"] is True
    lines = path.read_text().splitlines()
    entry = json.loads(lines[0])
    entry["payload"]["task_id"] = "changed"
    lines[0] = json.dumps(entry)
    path.write_text("\n".join(lines) + "\n")
    assert journal.verify()["valid"] is False
    with pytest.raises(RuntimeError, match="integrity"):
        journal.append("new", {})


def test_docker_sandbox_fails_closed_without_approval():
    sandbox = DockerSandbox(".")
    with pytest.raises(PermissionError, match="explicit approval"):
        sandbox.run_tests(".", approved=False)


def test_audio_metadata_and_perception_capabilities_are_explicit(tmp_path):
    path = tmp_path / "sample.wav"
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x00\x00" * 16000)
    result = AudioPerception().inspect_wav(path)
    assert result.data["duration_seconds"] == 1
    assert PerceptionSuite().capabilities()["audio"]["configured_model"] is False


def test_goal_discovery_does_not_self_authorize_scheduling(tmp_path):
    from aether.storage import Store

    store = Store(tmp_path / "goals.sqlite3")
    store.create_task("unfinished-1", "Review pending architecture task")
    manager = GoalManager(store)
    candidates = manager.discover()
    assert candidates
    with pytest.raises(PermissionError, match="explicit approval"):
        manager.schedule(candidates[0], interval_seconds=120)
\n
def test_decision_ledger_records_typed_producer_model_and_split(tmp_path):
    from aether.decisions import DecisionManager
    from aether.storage import Store

    manager = DecisionManager(Store(tmp_path / "decision-provenance.sqlite3"))
    decision_id = manager.log_prediction(
        "Will held-out task succeed?", 0.72, "workflow-transfer",
        probability_source="calibrated_readout",
        producer_id="rlcd-readout-v1", model_id="local-model@sha256:abc",
        evaluation_split="held_out",
    )
    item = manager.store.list_decisions(resolved_only=False)[0]
    assert item["id"] == decision_id
    assert item["metadata"]["producer_id"] == "rlcd-readout-v1"
    assert item["metadata"]["model_id"] == "local-model@sha256:abc"
    assert item["metadata"]["evaluation_split"] == "held_out"
