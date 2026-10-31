"""AETHER command-line entry point."""
from __future__ import annotations

import logging
from pathlib import Path

from aether.config import load_settings
from aether.engine import AetherEngine
from aether.storage import Store


def main() -> int:
    try:
        settings = load_settings()
        logging.basicConfig(
            level=getattr(logging, settings.app.log_level.upper(), logging.INFO),
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(Path("logs/aether.log"), encoding="utf-8"),
            ],
        )
        store = Store(settings.storage.database)
        engine = AetherEngine(settings, store)
    except Exception:
        logging.exception("AETHER could not initialize")
        print("AETHER could not start. Check logs/aether.log and config/settings.toml.")
        return 1

    status = engine.status()
    print(f"{status['name']} v{status['version']} | Windows-ready foundation")
    print(f"Model enabled: {status['model_enabled']} | Reachable: {status['model_reachable']}")
    print("Type /status, /quit, or enter a message.\n")

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
        if user_text.startswith("/remember "):
            memory = user_text[len("/remember "):].strip()
            if memory:
                store.add_memory(memory, source="user", status="user-provided")
                print("AETHER > Saved to local memory.")
            else:
                print("AETHER > Usage: /remember information to save")
            continue

        print(f"AETHER > {engine.respond(user_text)}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
