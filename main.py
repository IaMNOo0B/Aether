"""AETHER command-line entry point."""
from __future__ import annotations

import logging

from aether.config import load_settings
from aether.diagnostics import configure_logging, write_diagnostics
from aether.engine import AetherEngine
from aether.storage import Store


def main() -> int:
    try:
        settings = load_settings()
        configure_logging(settings.app.log_level)
        store = Store(settings.storage.database)
        engine = AetherEngine(settings, store)
    except Exception:
        logging.exception("AETHER could not initialize")
        print("AETHER could not start. Check logs/aether.log and config/settings.toml.")
        return 1

    status = engine.status()
    print(f"{status['name']} v{status['version']} | Windows-ready foundation")
    print(f"Model enabled: {status['model_enabled']} | Reachable: {status['model_reachable']}")
    print("Type /status, /tasks, /diagnostics, /remember <fact>, or /quit.\n")

    while True:
        try:
            user_text = input("YOU > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAETHER session closed.")
            break

        if user_text.lower() in {"/quit", "/exit"}:
            print("AETHER session closed.")
            break
        if user_text.lower() == "/status":
            for key, value in engine.status().items():
                print(f"{key}: {value}")
            continue
        if user_text.lower() == "/tasks":
            tasks = store.list_tasks(limit=10)
            if not tasks:
                print("AETHER > No recorded tasks.")
            for task in tasks:
                print(f"{task['id'][:8]} | {task['status']} | {task['objective']}")
            continue
        if user_text.lower() == "/diagnostics":
            destination = write_diagnostics("logs/diagnostics.json", engine.status())
            print(f"AETHER > Diagnostics saved to {destination}.")
            continue
        if user_text.startswith("/remember "):
            memory = user_text[len("/remember "):].strip()
            if memory:
                store.add_memory(memory, source="user", status="user-provided")
                print("AETHER > Saved to local memory.")
            else:
                print("AETHER > Usage: /remember information to save")
            continue

        try:
            print(f"AETHER > {engine.respond(user_text)}\n")
        except Exception:
            logging.exception("Message handling failed")
            print("AETHER > The request failed. Check logs/aether.log for details.\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
