"""Transparent symbolic world-state model and deterministic action simulation.

This module models declared preconditions/effects. It is not a learned dynamics model;
failure probabilities and resource costs are supplied by the operator and are not calibrated.
"""
from __future__ import annotations

from math import isfinite
from typing import Any
from uuid import uuid4
import json

from aether.storage import Store


def _json_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object with string keys")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain JSON-compatible finite values") from exc
    return value


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


class WorldModel:
    """Versioned symbolic world states, action effects, constraints and observed outcomes."""

    def __init__(self, store: Store):
        self.store = store

    @staticmethod
    def _validate_constraints(constraints: dict[str, Any]) -> None:
        _json_object(constraints, "constraints")
        for key, rule in constraints.items():
            if not isinstance(rule, dict):
                raise ValueError(f"Constraint for {key} must be an object")
            for bound in ("min", "max"):
                if bound in rule and (
                    isinstance(rule[bound], bool)
                    or not isinstance(rule[bound], (int, float))
                    or not isfinite(float(rule[bound]))
                ):
                    raise ValueError(f"Constraint {key}.{bound} must be a finite number")
            if "min" in rule and "max" in rule and rule["min"] > rule["max"]:
                raise ValueError(f"Constraint {key} min cannot exceed max")
            if "allowed" in rule and not isinstance(rule["allowed"], list):
                raise ValueError(f"Constraint {key}.allowed must be an array")

    def create_world(
        self, name: str, initial_state: dict[str, Any], *,
        constraints: dict[str, Any] | None = None, description: str = "",
    ) -> dict[str, Any]:
        clean_name = " ".join(name.split())
        if not clean_name or len(clean_name) > 150:
            raise ValueError("World name must contain 1..150 characters")
        if self.get_world(clean_name) is not None:
            raise ValueError(f"World already exists: {clean_name}")
        state = _json_object(initial_state, "initial_state")
        rules = _json_object(constraints or {}, "constraints")
        self._validate_constraints(rules)
        if not self._satisfies_constraints(state, rules):
            raise ValueError("Initial world state violates its declared constraints")
        world_id = uuid4().hex
        self.store.create_world(
            world_id, clean_name, _normalize(clean_name), description.strip()[:1000],
            state, rules,
        )
        world = self.store.get_world(world_id)
        if world is None:
            raise RuntimeError("World state could not be read after creation")
        return world

    def get_world(self, name: str) -> dict[str, Any] | None:
        return self.store.find_world(_normalize(name))

    def list_worlds(self) -> list[dict[str, Any]]:
        return self.store.list_worlds()

    def register_action(
        self, world_name: str, name: str, *,
        preconditions: dict[str, Any], effects: dict[str, Any],
        failure_effects: dict[str, Any] | None = None,
        resource_costs: dict[str, float] | None = None,
        failure_probability: float = 0.0,
        description: str = "",
    ) -> dict[str, Any]:
        world = self.get_world(world_name)
        if world is None:
            raise KeyError(f"Unknown world: {world_name}")
        clean_name = " ".join(name.split())
        if not clean_name or len(clean_name) > 150:
            raise ValueError("Action name must contain 1..150 characters")
        if any(item["normalized_name"] == _normalize(clean_name)
               for item in self.store.list_world_actions(world["id"])):
            raise ValueError(f"Action already exists in world: {clean_name}")
        pre = _json_object(preconditions, "preconditions")
        effect = _json_object(effects, "effects")
        failure = _json_object(failure_effects or {}, "failure_effects")
        costs = _json_object(resource_costs or {}, "resource_costs")
        for resource, cost in costs.items():
            if (isinstance(cost, bool) or not isinstance(cost, (int, float))
                    or not isfinite(float(cost)) or cost < 0):
                raise ValueError(f"Resource cost {resource} must be a finite non-negative number")
        if (isinstance(failure_probability, bool)
                or not isinstance(failure_probability, (int, float))
                or not isfinite(float(failure_probability))
                or not 0 <= failure_probability <= 1):
            raise ValueError("failure_probability must be between zero and one")
        action = {
            "id": uuid4().hex, "world_id": world["id"],
            "name": clean_name, "normalized_name": _normalize(clean_name),
            "description": description.strip()[:1000],
            "preconditions": pre, "effects": effect, "failure_effects": failure,
            "resource_costs": costs, "failure_probability": float(failure_probability),
        }
        self.store.insert_world_action(action)
        saved = self.store.get_world_action(action["id"])
        if saved is None:
            raise RuntimeError("World action could not be read after registration")
        return saved

    def list_actions(self, world_name: str) -> list[dict[str, Any]]:
        world = self.get_world(world_name)
        if world is None:
            raise KeyError(f"Unknown world: {world_name}")
        return self.store.list_world_actions(world["id"])

    @staticmethod
    def _satisfies_constraints(
        state: dict[str, Any], constraints: dict[str, Any],
    ) -> bool:
        for key, rule in constraints.items():
            if key not in state:
                return False
            value = state[key]
            if "min" in rule:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or value < rule["min"]:
                    return False
            if "max" in rule:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or value > rule["max"]:
                    return False
            if "allowed" in rule and value not in rule["allowed"]:
                return False
        return True

    def simulate(self, world_name: str, action_id: str) -> dict[str, Any]:
        world = self.get_world(world_name)
        if world is None:
            raise KeyError(f"Unknown world: {world_name}")
        action = self.store.get_world_action(action_id)
        if action is None or action["world_id"] != world["id"]:
            raise KeyError(f"Action {action_id} does not belong to world {world_name}")

        initial = dict(world["state"])
        preconditions_met = all(
            key in initial and initial[key] == expected
            for key, expected in action["preconditions"].items()
        )
        predicted = dict(initial)
        failure_state = dict(initial)
        if preconditions_met:
            predicted.update(action["effects"])
            failure_state.update(action["failure_effects"])
        constraints_met = self._satisfies_constraints(predicted, world["constraints"])
        simulation = {
            "id": uuid4().hex,
            "world_id": world["id"],
            "action_id": action_id,
            "world_version": world["version"],
            "initial_state": initial,
            "predicted_state": predicted,
            "failure_state": failure_state,
            "preconditions_met": preconditions_met,
            "constraints_met": constraints_met,
            "executable": preconditions_met and constraints_met,
            "failure_probability": action["failure_probability"],
            "resource_costs": action["resource_costs"],
            "expected_resource_cost": sum(action["resource_costs"].values()),
        }
        self.store.insert_world_simulation(simulation)
        return simulation

    def compare_actions(self, world_name: str, action_ids: list[str]) -> list[dict[str, Any]]:
        if not action_ids:
            raise ValueError("At least one action ID is required")
        results = [self.simulate(world_name, action_id) for action_id in action_ids]
        return sorted(
            results,
            key=lambda item: (
                not item["executable"],
                item["failure_probability"],
                item["expected_resource_cost"],
                item["id"],
            ),
        )

    def record_observation(
        self, simulation_id: str, actual_state: dict[str, Any], *,
        apply_to_world: bool = False,
    ) -> dict[str, Any]:
        actual = _json_object(actual_state, "actual_state")
        simulation = self.store.get_world_simulation(simulation_id)
        if simulation is None:
            raise KeyError(f"Unknown world simulation: {simulation_id}")
        predicted = simulation["predicted_state"]
        differences = {}
        for key in sorted(set(predicted) | set(actual)):
            if predicted.get(key) != actual.get(key) or (key not in predicted) != (key not in actual):
                differences[key] = {
                    "predicted": predicted.get(key, "<missing>"),
                    "actual": actual.get(key, "<missing>"),
                }
        observation = {
            "id": uuid4().hex,
            "simulation_id": simulation_id,
            "actual_state": actual,
            "prediction_error": differences,
            "applied_to_world": apply_to_world,
        }
        if apply_to_world:
            self.store.apply_world_observation(
                observation, simulation["world_id"],
                simulation["world_version"], actual,
            )
        else:
            self.store.insert_world_observation(observation)
        return observation

    def history(self, world_name: str) -> list[dict[str, Any]]:
        world = self.get_world(world_name)
        if world is None:
            raise KeyError(f"Unknown world: {world_name}")
        return self.store.list_world_state_history(world["id"])
