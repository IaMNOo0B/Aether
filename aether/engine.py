"""Conversation engine with persistent history and explicit model availability."""
from __future__ import annotations

from aether.config import Settings
from aether.ollama import ModelUnavailable, OllamaClient
from aether.storage import Store


SYSTEM_PROMPT = (
    "You are AETHER, a practical AI assistant. Be accurate and direct. "
    "Never claim a tool ran, a file changed, or a task completed unless verified. "
    "Treat retrieved memories as context, not unquestionable truth. "
    "Ask before destructive or consequential actions."
)


class AetherEngine:
    def __init__(self, settings: Settings, store: Store):
        self.settings = settings
        self.store = store
        self.client = OllamaClient(
            settings.models.base_url,
            settings.models.model,
            settings.models.timeout_seconds,
        )

    def status(self) -> dict[str, object]:
        available = self.client.is_available() if self.settings.models.enabled else False
        return {
            "name": self.settings.app.name,
            "version": "0.1.0",
            "provider": self.settings.models.provider,
            "model_enabled": self.settings.models.enabled,
            "model_reachable": available,
            "database": str(self.store.path),
        }

    def respond(self, user_text: str) -> str:
        text = user_text.strip()
        if not text:
            return "Please enter a message."

        self.store.add_message("user", text)
        memories = self.store.search_memories(text)
        memory_context = ""
        if memories:
            memory_context = "\n\nRelevant stored memories (unverified unless marked otherwise):\n"
            memory_context += "\n".join(
                f"- [{item['source']}; {item['status']}] {item['content']}"
                for item in memories
            )

        if not self.settings.models.enabled:
            answer = (
                "AETHER is running, but its language model is disabled. "
                "Enable models.enabled in config/settings.toml and make sure Ollama "
                f"is running with the configured model ({self.settings.models.model})."
            )
        else:
            history = self.store.recent_messages(self.settings.limits.max_history_messages)
            messages = [{"role": "system", "content": SYSTEM_PROMPT + memory_context}]
            messages.extend(history)
            try:
                answer = self.client.chat(messages)
            except ModelUnavailable as exc:
                answer = f"I couldn't reach the configured model. No answer was generated. Details: {exc}"

        self.store.add_message("assistant", answer)
        return answer
