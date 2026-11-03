"""Transparent scenario sampling over declared symbolic world-model probabilities."""
from __future__ import annotations

from collections import Counter
import random
from typing import Any


def simulate_scenarios(world_model: Any, world_name: str, action_id: str, *,
                       trials: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Sample declared success/failure effects; simulations are not real-world observations."""
    if isinstance(trials, bool) or not isinstance(trials, int) or not 1 <= trials <= 100_000:
        raise ValueError("trials must be an integer between 1 and 100000")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    base = world_model.simulate(world_name, action_id)
    if not base["preconditions_met"]:
        return {
            "status": "blocked_preconditions", "trials": 0, "requested_trials": trials,
            "simulation": base,
            "limitation": "No scenario was sampled because action preconditions are not met.",
        }
    rng = random.Random(seed)
    probability = float(base["failure_probability"])
    counts: Counter[str] = Counter()
    state_counts: Counter[str] = Counter()
    violations = 0
    for _ in range(trials):
        failed = rng.random() < probability
        state = base["failure_state"] if failed else base["predicted_state"]
        outcome = "failure" if failed else "success"
        counts[outcome] += 1
        if not world_model._satisfies_constraints(state, world_model.get_world(world_name)["constraints"]):
            violations += 1
            outcome = "constraint_violation"
        state_key = __import__("json").dumps(state, sort_keys=True, separators=(",", ":"))
        state_counts[state_key] += 1
    return {
        "status": "sampled", "world": world_name, "action_id": action_id,
        "trials": trials, "seed": seed,
        "declared_failure_probability": probability,
        "sampled_failures": counts["failure"], "sampled_successes": counts["success"],
        "constraint_violations": violations,
        "observed_failure_count": base["observed_failures"],
        "observed_success_count": base["observed_successes"],
        "state_frequencies": [
            {"state": __import__("json").loads(key), "count": count, "frequency": count / trials}
            for key, count in state_counts.most_common()
        ],
        "limitation": "Monte Carlo frequencies reflect the operator-declared failure probability and effects. They are not learned predictions, calibrated probabilities, or actual-world evidence.",
    }
