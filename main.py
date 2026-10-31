"""AETHER Windows-first command interface."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import Event, Thread
from uuid import uuid4

from aether.acceptance import run_acceptance_suite
from aether.agents import MultiAgentCoordinator
from aether.autonomy import AutonomyRunner
from aether.calibration import evaluate_binary_decisions
from aether.config import load_settings
from aether.desktop import DesktopController
from aether.diagnostics import configure_logging, write_diagnostics
from aether.engine import AetherEngine
from aether.learning import SkillLearner
from aether.research import ResearchError, Researcher
from aether.scheduler import AutonomyScheduler
from aether.self_engineering import SelfEngineer
from aether.storage import Store
from aether.voice import VoiceInterface, VoiceUnavailable


def ask_approval(tool: str, purpose: str, args: dict) -> bool:
    print("\nAPPROVAL REQUIRED")
    print(f"Action: {tool}")
    print(f"Purpose: {purpose}")
    if args:
        print(f"Arguments: { {key: value for key, value in args.items() if key != 'purpose'} }")
    answer = input("Approve this one action? Type YES: ").strip()
    return answer == "YES"


def main() -> int:
    try:
        settings = load_settings()
        configure_logging(settings.app.log_level)
        store = Store(settings.storage.database)
        engine = AetherEngine(settings, store)
        recovered_tasks = engine.tasks.recover_interrupted()
    except Exception:
        logging.exception("AETHER could not initialize")
        print("AETHER could not start. Check logs/aether.log and config/settings.toml.")
        return 1

    scheduler = None
    scheduler_stop = Event()
    scheduler_thread = None
    if settings.scheduler.enabled:
        scheduled_runner = AutonomyRunner(
            engine.provider(), store, settings.autonomy, settings.limits,
            settings.research, Path.cwd(),
        )
        scheduler = AutonomyScheduler(store, scheduled_runner, settings.scheduler)
        scheduler_thread = Thread(
            target=scheduler.run_forever, args=(scheduler_stop,),
            name="AETHER-Scheduler", daemon=True,
        )
        scheduler_thread.start()

    status = engine.status()
    print(f"{status['name']} v{status['version']} | Windows-first local AI")
    print(f"Model enabled: {status['model_enabled']} | Reachable: {status['model_reachable']}")
    if recovered_tasks:
        print(f"Recovered {len(recovered_tasks)} interrupted task(s). They are marked failed for manual review; no actions were replayed.")
    print("Commands: /status /self-model /tasks /task-events ID /emergency-stop /resume-autonomy /research /agents /autonomy /learn /skills /desktop /voice")
    print("          /remember /memory-search /memory-index /correct-memory /decision-log /decision-resolve /decision-assess /decision-metrics")
    print("          /schedule add|list|pause|resume /knowledge /world /acceptance /self-improve /self-improve-test /reports /calibration /quit\n")

    while True:
        try:
            user_text = input("YOU > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAETHER session closed.")
            break
        if not user_text:
            continue
        lower = user_text.casefold()

        if lower in {"/quit", "/exit"}:
            print("AETHER session closed.")
            break

        try:
            if lower == "/emergency-stop":
                store.set_emergency_stop(True, "Activated from the AETHER console")
                print("AETHER > EMERGENCY STOP ACTIVE. Scheduled and autonomous execution is blocked.")
                continue

            if lower == "/resume-autonomy":
                if not store.emergency_stop_enabled():
                    print("AETHER > Emergency stop is not active.")
                    continue
                if input("This re-enables autonomous execution. Type RESUME to confirm: ").strip() != "RESUME":
                    print("AETHER > Resume cancelled.")
                    continue
                store.set_emergency_stop(False, "User explicitly resumed autonomy")
                print("AETHER > Emergency stop cleared. Autonomy remains subject to its configured permissions.")
                continue

            if lower == "/acceptance":
                report = run_acceptance_suite(
                    Path.cwd(), output_path="reports/acceptance/latest.json"
                )
                print(json.dumps(report["counts"], indent=2))
                print(f"AETHER > Acceptance complete: {report['complete']}")
                print(f"AETHER > Report: {report.get('output_path', 'not saved')}")
                continue

            if lower == "/status":
                for key, value in engine.status().items():
                    print(f"{key}: {value}")
                continue

            if lower == "/self-model":
                print(json.dumps(engine.self_model.snapshot(), indent=2))
                print("AETHER > Capability states require concrete evidence; module presence is not verification.")
                continue

            if lower == "/tasks":
                tasks = store.list_tasks(limit=15)
                if not tasks:
                    print("AETHER > No recorded tasks.")
                for task in tasks:
                    print(f"{task['id']} | {task['status']} | {task['objective']}")
                continue

            if lower.startswith("/task-events "):
                task_id = user_text[len("/task-events "):].strip()
                events = engine.tasks.events(task_id)
                if not events:
                    print("AETHER > No audit events recorded for this task.")
                for event in events:
                    print(
                        f"#{event['sequence']} {event['created_at']} | "
                        f"{event['event_type']} | {event['from_status']} -> {event['to_status']}"
                    )
                    if event["payload"]:
                        print(json.dumps(event["payload"], indent=2))
                continue

            if lower == "/diagnostics":
                destination = write_diagnostics("logs/diagnostics.json", engine.status())
                print(f"AETHER > Diagnostics saved to {destination}.")
                continue

            if lower == "/calibration":
                decisions = store.list_decisions(resolved_only=True)
                if not decisions:
                    print("AETHER > No resolved decision outcomes have been logged yet.")
                    continue
                report = evaluate_binary_decisions(decisions)
                print(f"Resolved decisions: {report.count}")
                print(f"Brier: {report.brier_score:.4f} | Log loss: {report.log_loss:.4f}")
                print(f"ECE: {report.expected_calibration_error:.4f} | AURC: {report.aurc:.4f}")
                print(f"Coverage at 5% risk: {report.coverage_at_risk:.1%}")
                continue

            if lower.startswith("/decision-log "):
                parts = user_text[len("/decision-log "):].split("|", 2)
                if len(parts) != 3:
                    print("Usage: /decision-log question | probability_true | domain")
                    continue
                decision_id = engine.decisions.log_prediction(
                    parts[0].strip(), float(parts[1].strip()), parts[2].strip()
                )
                print(f"AETHER > Decision logged: {decision_id}. Resolve it when the real outcome is known.")
                continue

            if lower.startswith("/decision-resolve "):
                parts = user_text[len("/decision-resolve "):].split("|", 1)
                if len(parts) != 2 or parts[1].strip().casefold() not in {"true", "false"}:
                    print("Usage: /decision-resolve decision-id | true/false")
                    continue
                engine.decisions.resolve(
                    parts[0].strip(), parts[1].strip().casefold() == "true"
                )
                print("AETHER > Outcome recorded.")
                continue

            if lower.startswith("/decision-assess "):
                parts = user_text[len("/decision-assess "):].split("|", 1)
                if len(parts) != 2:
                    print("Usage: /decision-assess probability_true | domain")
                    continue
                assessment = engine.decisions.assess(
                    float(parts[0].strip()), parts[1].strip()
                )
                print(f"AETHER > Policy: {assessment.action.upper()}")
                print(f"Confidence: {assessment.confidence:.3f} | Samples: {assessment.sample_count}")
                if assessment.empirical_accuracy is not None:
                    print(
                        f"Observed accuracy: {assessment.empirical_accuracy:.3f} | "
                        f"Wilson lower bound: {assessment.wilson_lower_bound:.3f}"
                    )
                print(assessment.reason)
                continue

            if lower.startswith("/decision-metrics"):
                domain = user_text[len("/decision-metrics"):].strip() or None
                report = engine.decisions.metrics(domain)
                print(f"Resolved decisions: {report.count}")
                print(
                    f"Brier: {report.brier_score:.4f} | Log loss: {report.log_loss:.4f} | "
                    f"ECE: {report.expected_calibration_error:.4f} | AURC: {report.aurc:.4f}"
                )
                continue

            if lower.startswith("/schedule "):
                parts = user_text[len("/schedule "):].split(maxsplit=1)
                action = parts[0].casefold()
                argument = parts[1] if len(parts) == 2 else ""
                if action == "add":
                    fields = argument.split("|", 2)
                    if len(fields) not in {2, 3}:
                        print("Usage: /schedule add interval_seconds | objective | max_runs(optional; 0=unlimited)")
                        continue
                    interval = int(fields[0].strip())
                    objective = fields[1].strip()
                    max_runs = int(fields[2].strip()) if len(fields) == 3 else 0
                    schedule_id = uuid4().hex
                    store.create_schedule(schedule_id, objective, interval, max_runs)
                    print(f"AETHER > Read-only scheduled goal created: {schedule_id}")
                    continue
                if action == "list":
                    schedules = store.list_schedules()
                    if not schedules:
                        print("AETHER > No scheduled goals.")
                    for item in schedules:
                        print(
                            f"{item['id']} | {'enabled' if item['enabled'] else 'paused'} | "
                            f"every {item['interval_seconds']}s | runs {item['run_count']}/"
                            f"{item['max_runs'] or 'unlimited'} | {item['objective']}"
                        )
                    continue
                if action in {"pause", "resume"} and argument.strip():
                    store.set_schedule_enabled(argument.strip(), action == "resume")
                    print(f"AETHER > Schedule {argument.strip()} {action}d.")
                    continue
                if action == "tick":
                    if scheduler is None:
                        print("AETHER > Scheduler is disabled. Enable [scheduler], [autonomy] and [models] in config/settings.toml.")
                    else:
                        print(json.dumps(scheduler.run_once(), indent=2))
                    continue
                print("Usage: /schedule add|list|pause ID|resume ID|tick")
                continue

            if lower.startswith("/world "):
                parts = user_text[len("/world "):].split(maxsplit=1)
                action = parts[0].casefold()
                argument = parts[1] if len(parts) == 2 else ""
                if action == "create":
                    fields = argument.split("|", 2)
                    if len(fields) not in {2, 3}:
                        print("Usage: /world create name | initial_state_json | constraints_json(optional)")
                        continue
                    state = json.loads(fields[1].strip())
                    constraints = json.loads(fields[2].strip()) if len(fields) == 3 else {}
                    world = engine.world.create_world(
                        fields[0].strip(), state, constraints=constraints
                    )
                    print(f"AETHER > World created: {world['name']} | version {world['version']}")
                    continue
                if action == "list":
                    worlds = engine.world.list_worlds()
                    if not worlds:
                        print("AETHER > No world models.")
                    for world in worlds:
                        print(f"{world['id']} | {world['name']} | version {world['version']}")
                    continue
                if action == "state" and argument:
                    world = engine.world.get_world(argument)
                    if world is None:
                        print("AETHER > World not found.")
                    else:
                        print(json.dumps(world, indent=2))
                    continue
                if action == "action":
                    fields = argument.split("|", 6)
                    if len(fields) < 4:
                        print("Usage: /world action world | name | preconditions_json | effects_json | costs_json(optional) | failure_probability(optional) | failure_effects_json(optional)")
                        continue
                    costs = json.loads(fields[4].strip()) if len(fields) >= 5 and fields[4].strip() else {}
                    failure_probability = float(fields[5].strip()) if len(fields) >= 6 and fields[5].strip() else 0.0
                    failure_effects = json.loads(fields[6].strip()) if len(fields) >= 7 and fields[6].strip() else {}
                    registered = engine.world.register_action(
                        fields[0].strip(), fields[1].strip(),
                        preconditions=json.loads(fields[2].strip()),
                        effects=json.loads(fields[3].strip()),
                        resource_costs=costs,
                        failure_probability=failure_probability,
                        failure_effects=failure_effects,
                    )
                    print(f"AETHER > Symbolic action registered: {registered['id']}")
                    continue
                if action == "actions" and argument:
                    print(json.dumps(engine.world.list_actions(argument), indent=2))
                    continue
                if action == "simulate":
                    fields = argument.split("|", 1)
                    if len(fields) != 2:
                        print("Usage: /world simulate world_name | action_id")
                        continue
                    result = engine.world.simulate(fields[0].strip(), fields[1].strip())
                    print(json.dumps(result, indent=2))
                    continue
                if action == "compare":
                    fields = argument.split("|", 1)
                    if len(fields) != 2:
                        print("Usage: /world compare world_name | action_id, action_id")
                        continue
                    action_ids = [item.strip() for item in fields[1].split(",") if item.strip()]
                    print(json.dumps(engine.world.compare_actions(fields[0].strip(), action_ids), indent=2))
                    continue
                if action == "observe":
                    fields = argument.split("|", 2)
                    if len(fields) not in {2, 3}:
                        print("Usage: /world observe simulation_id | actual_state_json | apply/no(optional)")
                        continue
                    apply_state = len(fields) == 3 and fields[2].strip().casefold() == "apply"
                    observation = engine.world.record_observation(
                        fields[0].strip(), json.loads(fields[1].strip()),
                        apply_to_world=apply_state,
                    )
                    print(json.dumps(observation, indent=2))
                    continue
                if action == "history" and argument:
                    print(json.dumps(engine.world.history(argument), indent=2))
                    continue
                print("Usage: /world create|list|state|action|actions|simulate|compare|observe|history")
                continue

            if lower.startswith("/knowledge "):
                parts = user_text[len("/knowledge "):].split(maxsplit=1)
                action = parts[0].casefold()
                argument = parts[1] if len(parts) == 2 else ""
                if action == "extract" and argument:
                    claims = engine.knowledge.extract_from_report(argument)
                    print(f"AETHER > Extracted {len(claims)} unverified claim(s) from report {argument}.")
                    for claim in claims:
                        print(
                            f"{claim['id']} | {claim['claim_type']} | "
                            f"{claim['subject_name']} --{claim['predicate']}--> {claim['object_value']}"
                        )
                    continue
                if action == "search" and argument:
                    entities = engine.knowledge.search_entities(argument)
                    if not entities:
                        print("AETHER > No matching knowledge entities.")
                    for entity in entities:
                        print(f"{entity['id']} | {entity['entity_type']} | {entity['name']}")
                    continue
                if action == "show" and argument:
                    view = engine.knowledge.entity_view(argument)
                    if view is None:
                        print("AETHER > Entity not found.")
                    else:
                        print(json.dumps(view, indent=2))
                    continue
                if action == "conflicts" and argument:
                    conflicts = engine.knowledge.find_conflicts(argument)
                    if not conflicts:
                        print("AETHER > No overlapping contradictory claims found.")
                    else:
                        print(json.dumps(conflicts, indent=2))
                    continue
                if action == "claim":
                    fields = argument.split("|", 4)
                    if len(fields) != 5:
                        print("Usage: /knowledge claim subject | predicate | object | fact/hypothesis/observation/prediction | source_url")
                        continue
                    claim = engine.knowledge.add_claim(
                        fields[0].strip(), fields[1].strip(), fields[2].strip(),
                        claim_type=fields[3].strip().casefold(),
                        source_url=fields[4].strip(),
                    )
                    print(f"AETHER > Unverified claim saved: {claim['id']}")
                    continue
                if action == "relation":
                    fields = argument.split("|", 3)
                    if len(fields) != 4:
                        print("Usage: /knowledge relation subject | predicate | object_entity | source_url")
                        continue
                    claim = engine.knowledge.add_claim(
                        fields[0].strip(), fields[1].strip(), fields[2].strip(),
                        source_url=fields[3].strip(), object_is_entity=True,
                    )
                    print(f"AETHER > Unverified relationship saved: {claim['id']}")
                    continue
                if action == "review":
                    fields = argument.split("|", 2)
                    if len(fields) != 3:
                        print("Usage: /knowledge review claim-id | verified/disputed/retracted | evidence")
                        continue
                    claim = engine.knowledge.review_claim(
                        fields[0].strip(), fields[1].strip().casefold(), fields[2].strip()
                    )
                    print(f"AETHER > Claim {claim['id']} status: {claim['status']}")
                    continue
                if action == "history" and argument:
                    events = store.list_knowledge_claim_events(argument)
                    print(json.dumps(events, indent=2))
                    continue
                print("Usage: /knowledge extract REPORT_ID | search QUERY | show ENTITY | conflicts ENTITY | claim ... | review ... | relation subject | predicate | object_entity | source_url | history CLAIM_ID")
                continue

            if lower.startswith("/research "):
                if not settings.research.enabled:
                    print("AETHER > Research is disabled in config/settings.toml.")
                    continue
                query = user_text[len("/research "):].strip()
                researcher = Researcher(
                    store, settings.research.timeout_seconds,
                    settings.limits.max_research_results,
                )
                sources = researcher.search(query)
                report = researcher.create_report(query, sources)
                print(f"AETHER > Research report saved. ID: {report['id']}")
                print(report["content"])
                continue

            if lower == "/reports":
                reports = store.list_research_reports()
                if not reports:
                    print("AETHER > No saved research reports.")
                for report in reports:
                    print(f"{report['id']} | {report['query']} | {report['created_at']}")
                continue

            if lower.startswith("/agents "):
                coordinator = MultiAgentCoordinator(engine.provider(), store)
                objective = user_text[len("/agents "):].strip()
                deliverables = coordinator.run(objective)
                for item in deliverables:
                    print(f"\n--- {item.role.upper()} ---\n{item.output}")
                print("\nAETHER > Specialist outputs saved. They are not independent proof; verify claims.")
                continue

            if lower.startswith("/autonomy "):
                if not settings.autonomy.enabled:
                    print("AETHER > Autonomy is disabled. Set [autonomy] enabled = true in config/settings.toml.")
                    continue
                objective = user_text[len("/autonomy "):].strip()
                runner = AutonomyRunner(
                    engine.provider(), store, settings.autonomy, settings.limits,
                    settings.research, Path.cwd(),
                )
                if objective.casefold() == "queue":
                    results = runner.run_pending(ask_approval, max_tasks=3)
                    for result in results:
                        print(f"AETHER > {result['task_id']} | {result['status']}")
                else:
                    result = runner.run(objective, ask_approval)
                    print(f"AETHER > Task {result['task_id']} | {result['status']}")
                    for receipt in result.get("receipts", []):
                        print(f"- {receipt['tool']}: {str(receipt['result'])[:1000]}")
                continue

            if lower.startswith("/learn "):
                parts = user_text[len("/learn "):].split("|", 2)
                if len(parts) != 3:
                    print("Usage: /learn title | source reference | workflow text")
                    continue
                learner = SkillLearner(
                    store, engine.provider() if settings.models.enabled else None
                )
                skill = learner.learn_from_text(parts[0].strip(), parts[1].strip(), parts[2].strip())
                print(f"AETHER > Skill draft saved: {skill['title']} | ID: {skill['id']}")
                print("AETHER > It will not be used as a validated skill until you test it.")
                continue

            if lower == "/skills":
                skills = store.list_skills()
                if not skills:
                    print("AETHER > No learned skills.")
                for skill in skills:
                    print(f"{skill['id']} | {skill['status']} | {skill['title']}")
                continue

            if lower.startswith("/skill-eval "):
                parts = user_text[len("/skill-eval "):].split("|", 2)
                if len(parts) != 3 or parts[1].strip().casefold() not in {"pass", "fail"}:
                    print("Usage: /skill-eval skill-id | pass/fail | held-out test evidence")
                    continue
                skill = SkillLearner(store).evaluate(
                    parts[0].strip(), parts[1].strip().casefold() == "pass", parts[2].strip()
                )
                print(f"AETHER > {skill['title']} status: {skill['status']}")
                continue

            if lower.startswith("/desktop "):
                if store.emergency_stop_enabled():
                    print("AETHER > Emergency stop is active; desktop automation is blocked.")
                    continue
                parts = user_text.split(maxsplit=3)
                if len(parts) < 3:
                    print("Usage: /desktop open notepad.exe | click X Y | type TEXT | key enter")
                    continue
                controller = DesktopController(
                    settings.desktop.enabled, settings.desktop.allowed_apps,
                    settings.desktop.allow_input,
                )
                action = parts[1].casefold()
                if action == "open":
                    result = controller.open_app(" ".join(parts[2:]))
                    print(f"AETHER > Started {result['application']} (PID {result['pid']}).")
                elif action == "click" and len(parts) == 4:
                    coordinates = parts[2:]
                    x, y = int(coordinates[0]), int(coordinates[1])
                    approved = input(f"Move mouse and click screen coordinate ({x}, {y})? Type APPROVE: ").strip() == "APPROVE"
                    result = controller.click(x, y, approved=approved)
                    print(f"AETHER > {result}")
                elif action == "type" and len(parts) >= 3:
                    approved = input("Type text into the currently focused window? Type APPROVE: ").strip() == "APPROVE"
                    result = controller.type_text(" ".join(parts[2:]), approved=approved)
                    print(f"AETHER > {result}")
                elif action == "key" and len(parts) == 3:
                    approved = input(f"Press {parts[2]} in the focused window? Type APPROVE: ").strip() == "APPROVE"
                    result = controller.press_key(parts[2], approved=approved)
                    print(f"AETHER > {result}")
                elif action == "screenshot" and len(parts) == 3:
                    approved = input("Capture the current screen to the specified file? Type APPROVE: ").strip() == "APPROVE"
                    result = controller.screenshot(parts[2], approved=approved)
                    print(f"AETHER > Screenshot saved to {result}")
                else:
                    print("Usage: /desktop open APP | click X Y | type TEXT | key enter | screenshot PATH")
                continue

            if lower == "/voice":
                voice = VoiceInterface(
                    settings.voice.enabled, settings.voice.vosk_model_path,
                    settings.voice.wake_word, settings.voice.speak_responses,
                )
                print("AETHER > Listening once. Say the wake word followed by your command.")
                command = voice.listen_once()
                if not command:
                    print("AETHER > No matching voice command was detected.")
                    continue
                print(f"YOU (voice) > {command}")
                answer = engine.respond(command)
                print(f"AETHER > {answer}")
                voice.speak(answer)
                continue

            if lower.startswith("/self-improve "):
                engineer = SelfEngineer(Path.cwd(), engine.provider(), settings.limits.tool_timeout_seconds * 5)
                proposal = engineer.propose(user_text[len("/self-improve "):].strip())
                print(f"AETHER > Patch proposal saved: {proposal['id']}")
                print(f"Review file: {proposal['patch']}")
                print("AETHER > Nothing has been applied to the production project.")
                continue

            if lower.startswith("/self-improve-test "):
                if store.emergency_stop_enabled():
                    print("AETHER > Emergency stop is active; code evaluation is blocked.")
                    continue
                experiment_id = user_text[len("/self-improve-test "):].strip()
                if input(
                    "Generated code will execute in a copied project folder, not a secure OS sandbox. "
                    "Type TEST to approve baseline and patched tests: "
                ).strip() != "TEST":
                    print("AETHER > Evaluation cancelled.")
                    continue
                engineer = SelfEngineer(Path.cwd(), engine.registry.get(settings.models.provider), settings.limits.tool_timeout_seconds * 5)
                result = engineer.evaluate(experiment_id, approved=True)
                print(f"AETHER > Candidate status: {result['status']}")
                print(f"Integrated: {result.get('integrated', False)}")
                print(result.get("test_output", result.get("reason", "")))
                continue

            if lower.startswith("/remember "):
                memory = user_text[len("/remember "):].strip()
                if memory:
                    memory_id = engine.memory.remember(
                        memory, source="user", kind="semantic", status="user-provided"
                    )
                    print(f"AETHER > Saved memory #{memory_id}.")
                else:
                    print("AETHER > Usage: /remember information to save")
                continue

            if lower == "/memory-index":
                result = engine.memory.index_pending(settings.memory.index_batch_size)
                print(json.dumps(result, indent=2))
                continue

            if lower.startswith("/memory-search "):
                query = user_text[len("/memory-search "):].strip()
                matches = engine.memory.search(query, limit=10)
                if not matches:
                    print("AETHER > No relevant memories found.")
                for item in matches:
                    print(
                        f"#{item['id']} [{item['kind']}; {item['status']}; "
                        f"relevance={item['relevance']:.3f}; lexical={item['lexical_score']:.3f}; "
                        f"semantic={item['semantic_similarity']}] {item['content']}"
                    )
                continue

            if lower.startswith("/correct-memory "):
                parts = user_text[len("/correct-memory "):].split("|", 2)
                if len(parts) != 3:
                    print("Usage: /correct-memory memory-id | corrected content | reason")
                    continue
                new_id = engine.memory.correct(
                    int(parts[0].strip()), parts[1].strip(), parts[2].strip()
                )
                print(
                    f"AETHER > Correction saved as memory #{new_id}. "
                    "The previous record is preserved as superseded."
                )
                continue

            print(f"AETHER > {engine.respond(user_text)}\n")
        except (ValueError, PermissionError, FileNotFoundError, ResearchError, VoiceUnavailable) as exc:
            print(f"AETHER > {exc}")
        except Exception:
            logging.exception("Command failed")
            print("AETHER > The command failed. Check logs/aether.log for details.")

    if scheduler_thread is not None:
        scheduler_stop.set()
        scheduler_thread.join(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
