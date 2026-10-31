import pytest

from aether.storage import Store
from aether.world_model import WorldModel


def test_world_state_and_action_simulation_are_persistent(tmp_path):
    model = WorldModel(Store(tmp_path / "world.sqlite3"))
    world = model.create_world(
        "Battery system",
        {"battery": 50, "temperature": 25, "online": True},
        constraints={"battery": {"min": 0, "max": 100}},
    )
    action = model.register_action(
        "Battery system", "Charge",
        preconditions={"online": True},
        effects={"battery": 80},
        failure_effects={"battery": 45},
        resource_costs={"energy": 4.5, "time_minutes": 20},
        failure_probability=0.1,
    )

    simulation = model.simulate("battery system", action["id"])
    assert simulation["world_version"] == world["version"] == 1
    assert simulation["executable"] is True
    assert simulation["predicted_state"]["battery"] == 80
    assert simulation["failure_state"]["battery"] == 45
    assert simulation["expected_resource_cost"] == 24.5
    assert simulation["failure_probability"] == 0.1

    reloaded = WorldModel(Store(tmp_path / "world.sqlite3"))
    assert reloaded.get_world("BATTERY SYSTEM")["state"]["battery"] == 50
    assert reloaded.store.get_world_simulation(simulation["id"])["predicted_state"]["battery"] == 80


def test_world_preconditions_and_constraints_block_invalid_actions(tmp_path):
    model = WorldModel(Store(tmp_path / "world.sqlite3"))
    model.create_world(
        "Door", {"locked": True, "open": False},
        constraints={"open": {"allowed": [False, True]}},
    )
    action = model.register_action(
        "Door", "Open door",
        preconditions={"locked": False},
        effects={"open": True},
    )
    simulation = model.simulate("Door", action["id"])
    assert simulation["preconditions_met"] is False
    assert simulation["executable"] is False
    assert simulation["predicted_state"]["open"] is False

    invalid = model.register_action(
        "Door", "Impossible state",
        preconditions={"locked": True},
        effects={"open": "yes"},
    )
    blocked = model.simulate("Door", invalid["id"])
    assert blocked["constraints_met"] is False
    assert blocked["executable"] is False


def test_compare_actions_sorts_by_feasibility_risk_and_resource_cost(tmp_path):
    model = WorldModel(Store(tmp_path / "world.sqlite3"))
    model.create_world("Robot", {"battery": 70, "location": "A"})
    slow = model.register_action(
        "Robot", "Long route", preconditions={"location": "A"},
        effects={"location": "B"}, resource_costs={"energy": 50},
        failure_probability=0.2,
    )
    efficient = model.register_action(
        "Robot", "Short route", preconditions={"location": "A"},
        effects={"location": "C"}, resource_costs={"energy": 10},
        failure_probability=0.05,
    )
    blocked = model.register_action(
        "Robot", "Wrong start", preconditions={"location": "Z"},
        effects={"location": "D"}, resource_costs={"energy": 1},
        failure_probability=0.0,
    )

    results = model.compare_actions(
        "Robot", [slow["id"], blocked["id"], efficient["id"]]
    )
    assert results[0]["action_id"] == efficient["id"]
    assert results[-1]["action_id"] == blocked["id"]


def test_observation_records_prediction_error_without_mutating_world_by_default(tmp_path):
    model = WorldModel(Store(tmp_path / "world.sqlite3"))
    model.create_world("Thermostat", {"temperature": 20})
    action = model.register_action(
        "Thermostat", "Heat", preconditions={},
        effects={"temperature": 22},
    )
    simulation = model.simulate("Thermostat", action["id"])

    observation = model.record_observation(
        simulation["id"], {"temperature": 21}
    )

    assert observation["prediction_error"]["temperature"] == {
        "predicted": 22, "actual": 21
    }
    assert observation["applied_to_world"] is False
    assert model.get_world("Thermostat")["state"]["temperature"] == 20


def test_observation_can_update_world_with_versioned_history(tmp_path):
    model = WorldModel(Store(tmp_path / "world.sqlite3"))
    model.create_world("Room", {"temperature": 18})
    action = model.register_action(
        "Room", "Heat", preconditions={},
        effects={"temperature": 22},
    )
    simulation = model.simulate("Room", action["id"])

    observation = model.record_observation(
        simulation["id"], {"temperature": 21}, apply_to_world=True
    )

    world = model.get_world("Room")
    assert observation["applied_to_world"] is True
    assert world["state"]["temperature"] == 21
    assert world["version"] == 2
    assert [entry["version"] for entry in model.history("Room")] == [1, 2]


def test_stale_simulation_cannot_overwrite_a_newer_world_version(tmp_path):
    model = WorldModel(Store(tmp_path / "world.sqlite3"))
    model.create_world("Grid", {"position": "A"})
    action = model.register_action(
        "Grid", "Move", preconditions={"position": "A"},
        effects={"position": "B"},
    )
    first = model.simulate("Grid", action["id"])
    second = model.simulate("Grid", action["id"])

    model.record_observation(first["id"], {"position": "B"}, apply_to_world=True)
    with pytest.raises(ValueError, match="World changed after this simulation"):
        model.record_observation(second["id"], {"position": "C"}, apply_to_world=True)

    assert model.get_world("Grid")["state"]["position"] == "B"
    assert len(model.store.list_world_observations(second["id"])) == 0


def test_world_model_validates_costs_probability_and_constraints(tmp_path):
    model = WorldModel(Store(tmp_path / "world.sqlite3"))
    with pytest.raises(ValueError, match="Initial world state violates"):
        model.create_world("Bad", {"battery": -1}, constraints={"battery": {"min": 0}})
    model.create_world("Good", {"battery": 10})
    with pytest.raises(ValueError, match="non-negative"):
        model.register_action(
            "Good", "Bad cost", preconditions={}, effects={},
            resource_costs={"energy": -1},
        )
    with pytest.raises(ValueError, match="failure_probability"):
        model.register_action(
            "Good", "Bad probability", preconditions={}, effects={},
            failure_probability=1.5,
        )


def test_world_actions_cannot_be_reused_across_worlds(tmp_path):
    model = WorldModel(Store(tmp_path / "world.sqlite3"))
    model.create_world("World A", {"x": 1})
    model.create_world("World B", {"x": 2})
    action = model.register_action(
        "World A", "Change", preconditions={}, effects={"x": 3}
    )
    with pytest.raises(KeyError, match="does not belong"):
        model.simulate("World B", action["id"])

def test_world_model_updates_empirical_action_risk_from_observed_outcomes(tmp_path):
    model = WorldModel(Store(tmp_path / "world.sqlite3"))
    model.create_world("Pump", {"running": False})
    action = model.register_action(
        "Pump", "Start pump",
        preconditions={"running": False},
        effects={"running": True},
        failure_probability=0.2,
    )

    failed_simulation = model.simulate("Pump", action["id"])
    model.record_observation(
        failed_simulation["id"], {"running": False}
    )
    successful_simulation = model.simulate("Pump", action["id"])
    model.record_observation(
        successful_simulation["id"], {"running": True}
    )

    updated = model.simulate("Pump", action["id"])
    assert updated["observed_successes"] == 1
    assert updated["observed_failures"] == 1
    assert updated["observed_failure_rate"] == 0.5
    assert updated["risk_score"] == 0.35
