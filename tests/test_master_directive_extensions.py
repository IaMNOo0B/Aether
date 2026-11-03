import json
import wave

import pytest

from aether.audit import AuditJournal
from aether.goals import GoalManager
from aether.perception import AudioPerception, PerceptionSuite
from aether.rlcd_training import RLCDConfigurationError, RLCDExperimentConfig, RLCDRunConfig, RLCDTrainingAdapter, prepare_dataset
from aether.sandbox import DockerSandbox


def test_rlcd_dataset_is_prompt_deduplicated_stratified_and_reproducible(tmp_path):
    rows = [{"prompt": f"case {i}", "label": str(i % 2), "rationale": f"why {i}"}
            for i in range(120)]
    one = prepare_dataset(rows, tmp_path / "one.json", seed=9)
    two = prepare_dataset(rows, tmp_path / "two.json", seed=9)
    left, right = json.loads((tmp_path / "one.json").read_text()), json.loads((tmp_path / "two.json").read_text())
    assert one["sha256"] == two["sha256"]
    assert left["split_counts"] == right["split_counts"]
    groups = [{item["_aether_example_hash"] for item in left["splits"][key]}
              for key in ("train", "validation", "test")]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    assert all(groups)


def test_rlcd_dataset_rejects_same_prompt_with_conflicting_labels(tmp_path):
    with pytest.raises(RLCDConfigurationError, match="Conflicting labels"):
        prepare_dataset([{"prompt": "same", "label": "A"},
                         {"prompt": "same", "label": "B"}], tmp_path / "split.json")


def test_rlcd_adapter_requires_external_gpu_and_immutable_dataset(tmp_path):
    rows = [{"prompt": f"case {i}", "label": str(i % 2)} for i in range(120)]
    prepare_dataset(rows, tmp_path / "data" / "split.json")
    adapter = RLCDTrainingAdapter(tmp_path)
    adapter.create_manifest(RLCDRunConfig("model/id", "data/split.json", "runs/output"),
                             "runs/manifest.json")
    with pytest.raises(PermissionError, match="external GPU"):
        adapter.run("runs/manifest.json", ["python", "runner.py"])
    (tmp_path / "data" / "split.json").write_text("{}")
    with pytest.raises(RLCDConfigurationError, match="Dataset changed"):
        adapter.run("runs/manifest.json", ["python", "runner.py"],
                    approved=True, external_compute_confirmed=True)


def test_audit_journal_detects_tampering_and_refuses_append(tmp_path):
    path = tmp_path / "audit.jsonl"
    journal = AuditJournal(path)
    journal.append("task.started", {"id": "one"})
    assert journal.verify()["valid"]
    item = json.loads(path.read_text().splitlines()[0])
    item["payload"]["id"] = "tampered"
    path.write_text(json.dumps(item) + "\n")
    assert not journal.verify()["valid"]
    with pytest.raises(RuntimeError, match="integrity"):
        journal.append("task.finished", {})


def test_docker_sandbox_requires_explicit_approval():
    with pytest.raises(PermissionError, match="explicit approval"):
        DockerSandbox(".").run_tests(".", approved=False)


def test_audio_inspection_and_perception_capabilities_are_truthful(tmp_path):
    path = tmp_path / "test.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000)
    result = AudioPerception().inspect_wav(path)
    assert result.data["duration_seconds"] == 1
    assert PerceptionSuite().capabilities()["audio"]["configured_model"] is False


def test_goal_discovery_requires_approval_before_scheduling(tmp_path):
    from aether.storage import Store
    store = Store(tmp_path / "goals.sqlite3")
    store.create_task("pending-task", "Review unfinished design")
    manager = GoalManager(store)
    candidates = manager.discover()
    assert candidates
    with pytest.raises(PermissionError, match="explicit approval"):
        manager.schedule(candidates[0], interval_seconds=120)


def test_frontier_requires_observed_result_before_validation(tmp_path):
    from aether.frontier import ResearchFrontier
    frontier = ResearchFrontier(tmp_path / "frontier.jsonl")
    item = frontier.propose({
        "question": "Can a smaller router reduce latency?",
        "current_state": "One model route.",
        "primary_sources": ["https://arxiv.org/abs/0000.00000"],
        "reproducible_evidence": "No benchmark yet.",
        "technical_requirements": ["Held-out tasks"],
        "feasibility": "Needs measurement.",
        "resource_cost": "Benchmark time.",
        "limitations": "Synthetic examples may not generalize.",
        "aether_application": "Model routing.",
        "proposed_experiment": "Compare routes on a held-out task set.",
    })
    with pytest.raises(ValueError, match="observed result"):
        frontier.transition(item["id"], "validated", evidence="Seems good",
                            reviewer="reviewer", limitations="small sample")
    updated = frontier.transition(item["id"], "investigating", evidence="Reviewed sources",
                                   reviewer="reviewer", limitations="No replication")
    assert updated["status"] == "investigating"
    assert frontier.integrity()["valid"]


def test_decision_ledger_tracks_producer_and_excludes_held_out_from_operational_gate(tmp_path):
    from aether.decisions import DecisionManager
    from aether.storage import Store
    manager = DecisionManager(Store(tmp_path / "decisions.sqlite3"))
    for index in range(40):
        decision_id = manager.log_prediction(
            f"held-out case {index}", 0.95, "routing",
            probability_source="calibrated_readout", producer_id="rlcd-v1",
            model_id="model@revision", evaluation_split="held_out",
        )
        manager.resolve(decision_id, True)
    item = manager.store.list_decisions(resolved_only=False)[0]
    assert item["metadata"]["producer_id"] == "rlcd-v1"
    assert item["metadata"]["evaluation_split"] == "held_out"
    assert manager.assess(0.95, "routing").sample_count == 0
    assert manager.metrics("routing", evaluation_split="held_out").count == 40


def test_symbolic_logic_truth_table_and_linear_solver():
    from aether.reasoning import evaluate_proposition, solve_linear_system, truth_table

    assert evaluate_proposition("A and not B", {"A": True, "B": False}) is True
    table = truth_table("A or not A")
    assert table["tautology"] is True
    solved = solve_linear_system([[2, 1], [1, -1]], [5, 1])
    assert solved["solution"] == pytest.approx([2.0, 1.0])
    assert solved["max_absolute_residual"] < 1e-10


def test_hypothesis_ranking_requires_explicit_evidence_likelihoods():
    from aether.reasoning import rank_hypotheses

    result = rank_hypotheses([
        {"id": "H1", "prior_probability": 0.5,
         "evidence": [{"id": "E1", "log_bayes_factor": 1.0}]},
        {"id": "H2", "prior_probability": 0.5,
         "evidence": [{"id": "E2", "log_bayes_factor": -1.0}]},
    ])
    assert result["ranked"][0]["id"] == "H1"
    assert "not empirically calibrated" in result["warning"]


def test_world_model_scenarios_are_reproducible_and_not_observations(tmp_path):
    from aether.storage import Store
    from aether.world_model import WorldModel
    from aether.world_scenarios import simulate_scenarios

    world = WorldModel(Store(tmp_path / "world.sqlite3"))
    world.create_world("factory", {"machine": "idle", "units": 0},
                       constraints={"units": {"min": 0, "max": 100}})
    action = world.register_action(
        "factory", "run", preconditions={"machine": "idle"},
        effects={"machine": "running", "units": 10},
        failure_effects={"machine": "idle", "units": 0},
        failure_probability=0.2,
    )
    first = simulate_scenarios(world, "factory", action["id"], trials=500, seed=4)
    second = simulate_scenarios(world, "factory", action["id"], trials=500, seed=4)
    assert first["sampled_failures"] == second["sampled_failures"]
    assert first["state_frequencies"] == second["state_frequencies"]
    assert "not learned predictions" in first["limitation"]


def test_rlcd_small_but_valid_dataset_keeps_each_label_in_every_split(tmp_path):
    rows = [{"prompt": f"short {i}", "label": str(i % 2)} for i in range(10)]
    report = prepare_dataset(rows, tmp_path / "small.json", seed=3)
    payload = json.loads((tmp_path / "small.json").read_text())
    assert all(payload["splits"][split] for split in ("train", "validation", "test"))
    for split in ("train", "validation", "test"):
        assert {row["label"] for row in payload["splits"][split]} == {"0", "1"}


def test_exact_rlcd_two_stage_plan_matches_published_fork_structure(tmp_path):
    adapter = RLCDTrainingAdapter(tmp_path)
    plan = adapter.create_two_stage_plan(
        RLCDExperimentConfig(task="mmlupro", seeds=(17, 23, 29)),
        "plans/rlcd.json",
    )
    assert plan["upstream_commit"] == adapter.UPSTREAM_COMMIT
    assert plan["model_revision"] == "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    assert plan["dataset"]["revision"] == "b189ec765aa7ed75c8acfea42df31fdae71f97be"
    assert len(plan["runs"]) == 3
    assert sum(1 + len(item["stage2"]) for item in plan["runs"]) == 15
    first = plan["runs"][0]
    stage1 = first["stage1"]["argv"]
    assert "--no-score-function" in stage1
    assert "--diversity-weight" in stage1 and stage1[stage1.index("--diversity-weight") + 1] == "0"
    assert stage1[stage1.index("--kl-beta") + 1] == "0.04"
    assert [item["arm"] for item in first["stage2"]] == [
        "fork_rl", "fork_ro", "fork_anchor", "fork_grpo"
    ]
    rlcd = first["stage2"][0]["argv"]
    assert rlcd[rlcd.index("--seed") + 1] == "1017"
    assert rlcd[rlcd.index("--sf-coef") + 1] == "0.3"
    assert rlcd[rlcd.index("--init-checkpoint") + 1] == first["stage1"]["checkpoint"]


def test_rlcd_two_stage_plan_detects_post_creation_modification(tmp_path):
    adapter = RLCDTrainingAdapter(tmp_path)
    adapter.create_two_stage_plan(RLCDExperimentConfig(seeds=(17,)), "plans/rlcd.json")
    path = tmp_path / "plans" / "rlcd.json"
    payload = json.loads(path.read_text())
    payload["runs"][0]["stage2"][0]["argv"].append("--no-score-function")
    path.write_text(json.dumps(payload))
    with pytest.raises(RLCDConfigurationError, match="modified"):
        adapter.run_two_stage_plan(
            "plans/rlcd.json", "external/openjev-rlcd",
            approved=True, external_compute_confirmed=True,
        )


def test_hierarchical_goal_graph_requires_approval_and_child_completion(tmp_path):
    from aether.goals import HierarchicalGoalGraph

    graph = HierarchicalGoalGraph(tmp_path / "goal-graph.jsonl")
    parent = graph.create(
        "Complete a reliable evaluation subsystem",
        success_criteria=["All acceptance tests pass"],
        resource_budget={"max_steps": 20, "max_runtime_seconds": 3600, "max_model_calls": 20},
    )
    children = graph.decompose(parent["id"], [
        {"objective": "Build benchmark cases", "success_criteria": ["Cases are versioned"],
         "resource_budget": {"max_steps": 5}},
        {"objective": "Build result reporting", "success_criteria": ["Metrics are reproducible"],
         "resource_budget": {"max_steps": 5}},
    ])
    with pytest.raises(ValueError, match="Invalid goal transition"):
        graph.transition(parent["id"], "running", actor="user", evidence="start")
    graph.transition(parent["id"], "approved", actor="user", evidence="Owner approved bounded implementation")
    graph.transition(parent["id"], "running", actor="user", evidence="Execution started")
    with pytest.raises(ValueError, match="child goals"):
        graph.transition(parent["id"], "completed", actor="user", evidence="Parent work complete")
    for child in children:
        graph.transition(child["id"], "approved", actor="user", evidence="Child approved")
        graph.transition(child["id"], "running", actor="agent", evidence="Child started")
        graph.transition(child["id"], "completed", actor="agent", evidence="Acceptance evidence recorded")
    graph.transition(parent["id"], "completed", actor="user", evidence="All child acceptance evidence reviewed")
    assert graph.get(parent["id"])["status"] == "completed"
    assert len(graph.tree()[0]["children"]) == 2


def test_goal_discovery_includes_unverified_self_model_capabilities(tmp_path):
    from aether.goals import GoalManager
    from aether.self_model import SelfModel
    from aether.storage import Store

    store = Store(tmp_path / "goals-selfmodel.sqlite3")
    model = SelfModel(store.path)
    model.register_capability("test.capability", "Test capability", "Requires verification",
                              state="implemented_unverified")
    candidates = GoalManager(store, model).discover()
    assert any(item.source == "capability:test.capability" for item in candidates)


def test_failure_intelligence_never_auto_retries_and_checks_side_effect_state(tmp_path):
    from aether.failure_intelligence import FailureAnalyzer

    analyzer = FailureAnalyzer(tmp_path / "failures.jsonl")
    safe = analyzer.analyze(
        "request timed out", operation="read_status",
        idempotent=True, side_effect_state="not_started",
    )
    unsafe = analyzer.analyze(
        "request timed out", operation="send_payment",
        idempotent=False, side_effect_state="unknown",
    )
    auth = analyzer.analyze("401 invalid API key", operation="provider_call")
    assert safe.retryable is True
    assert safe.automatic_retry_allowed is False
    assert unsafe.retryable is False and unsafe.human_intervention_required
    assert auth.category == "authentication"
    analyzer.record(safe)
    analyzer.record(unsafe)
    assert analyzer.trends()["total"] == 2
    assert analyzer.trends()["automatic_retries"] == 0
