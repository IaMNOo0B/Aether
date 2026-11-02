import pytest

from aether.decisions import DecisionManager
from aether.storage import Store
from aether.world_model import WorldModel


def test_typed_decision_requires_supported_producer_and_source_reference(tmp_path):
    manager = DecisionManager(Store(tmp_path / "decisions.sqlite3"))
    with pytest.raises(ValueError, match="Unsupported decision producer"):
        manager.log_typed_prediction(
            "Will the test pass?", 0.8, "software-tests",
            producer_type="unknown", source_ref="run-123",
        )
    with pytest.raises(ValueError, match="source_ref"):
        manager.log_typed_prediction(
            "Will the test pass?", 0.8, "software-tests",
            producer_type="tool_verification", source_ref=" ",
        )


def test_typed_decision_persists_producer_and_evidence_provenance(tmp_path):
    store = Store(tmp_path / "decisions.sqlite3")
    manager = DecisionManager(store)
    decision_id = manager.log_typed_prediction(
        "Will this experiment succeed?", 0.7, "experiments",
        producer_type="experiment_prediction",
        source_ref="experiment-42",
        model_id="qwen-local",
        evidence_hash="a" * 64,
    )

    record = store.list_decisions(resolved_only=False)[0]
    assert record["id"] == decision_id
    assert record["metadata"]["decision_schema"] == "typed-binary-v2"
    assert record["metadata"]["producer_type"] == "experiment_prediction"
    assert record["metadata"]["source_ref"] == "experiment-42"
    assert record["metadata"]["model_id"] == "qwen-local"
    assert record["metadata"]["evidence_hash"] == "a" * 64


def test_world_model_simulations_become_resolved_calibration_examples(tmp_path):
    store = Store(tmp_path / "world.sqlite3")
    decisions = DecisionManager(store)
    world = WorldModel(store, decisions)
    world.create_world("Pump", {"running": False})
    action = world.register_action(
        "Pump", "Start pump",
        preconditions={"running": False},
        effects={"running": True},
        failure_effects={"running": False},
        failure_probability=0.2,
    )

    failed = world.simulate("Pump", action["id"])
    pending = store.list_decisions(resolved_only=False)
    assert len(pending) == 1
    assert pending[0]["probability"] == pytest.approx(0.8)
    assert pending[0]["metadata"]["producer_type"] == "world_model"
    assert pending[0]["metadata"]["source_ref"] == failed["id"]
    assert pending[0]["outcome"] is None

    world.record_observation(failed["id"], {"running": False})
    resolved = store.list_decisions(resolved_only=True)
    assert len(resolved) == 1
    assert resolved[0]["outcome"] == 0

    succeeded = world.simulate("Pump", action["id"])
    world.record_observation(succeeded["id"], {"running": True})
    resolved = store.list_decisions(resolved_only=True)
    assert len(resolved) == 2
    assert resolved[0]["outcome"] == 1 or resolved[1]["outcome"] == 1


def test_ambiguous_world_observation_does_not_become_binary_training_label(tmp_path):
    store = Store(tmp_path / "world.sqlite3")
    world = WorldModel(store, DecisionManager(store))
    world.create_world("Robot", {"position": "A"})
    action = world.register_action(
        "Robot", "Move", preconditions={}, effects={"position": "B"},
        failure_effects={"position": "A"}, failure_probability=0.1,
    )
    simulation = world.simulate("Robot", action["id"])
    world.record_observation(simulation["id"], {"position": "C"})

    assert store.list_decisions(resolved_only=True) == []
    pending = store.list_decisions(resolved_only=False)
    assert len(pending) == 1
    assert pending[0]["outcome"] is None



def test_typed_decision_rejects_invalid_evidence_hash_and_nonfinite_probability(tmp_path):
    manager = DecisionManager(Store(tmp_path / "decisions.sqlite3"))
    with pytest.raises(ValueError, match="SHA-256"):
        manager.log_typed_prediction(
            "Will it pass?", 0.8, "software-tests",
            producer_type="tool_verification", source_ref="run-1",
            evidence_hash="not-a-hash",
        )
    with pytest.raises(ValueError, match="between zero and one"):
        manager.log_typed_prediction(
            "Will it pass?", float("nan"), "software-tests",
            producer_type="tool_verification", source_ref="run-1",
        )


def test_non_executable_world_simulation_is_not_logged_as_a_decision(tmp_path):
    store = Store(tmp_path / "world.sqlite3")
    world = WorldModel(store, DecisionManager(store))
    world.create_world("Locked door", {"locked": True})
    action = world.register_action(
        "Locked door", "Open door", preconditions={"locked": False},
        effects={"locked": False}, failure_probability=0.1,
    )
    simulation = world.simulate("Locked door", action["id"])
    assert simulation["executable"] is False
    assert store.list_decisions(resolved_only=False) == []
