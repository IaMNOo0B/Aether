"""AETHER Windows-first command interface."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from aether.agents import MultiAgentCoordinator
from aether.autonomy import AutonomyRunner
from aether.calibration import evaluate_binary_decisions
from aether.config import load_settings
from aether.desktop import DesktopController
from aether.diagnostics import configure_logging, write_diagnostics
from aether.engine import AetherEngine
from aether.learning import SkillLearner
from aether.research import ResearchError, Researcher
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

    status = engine.status()
    print(f"{status['name']} v{status['version']} | Windows-first local AI")
    print(f"Model enabled: {status['model_enabled']} | Reachable: {status['model_reachable']}")
    if recovered_tasks:
        print(f"Recovered {len(recovered_tasks)} interrupted task(s). They are marked failed for manual review; no actions were replayed.")
    print("Commands: /status /self-model /tasks /task-events ID /research /agents /autonomy /learn /skills /desktop /voice")
    print("          /self-improve /self-improve-test /tasks /reports /calibration /quit\n")

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
                    store.add_memory(memory, source="user", status="user-provided")
                    print("AETHER > Saved to local memory.")
                else:
                    print("AETHER > Usage: /remember information to save")
                continue

            print(f"AETHER > {engine.respond(user_text)}\n")
        except (ValueError, PermissionError, FileNotFoundError, ResearchError, VoiceUnavailable) as exc:
            print(f"AETHER > {exc}")
        except Exception:
            logging.exception("Command failed")
            print("AETHER > The command failed. Check logs/aether.log for details.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
