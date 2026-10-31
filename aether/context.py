"""Deterministic character-budgeting for model context construction."""
from __future__ import annotations


def fit_messages(
    system_message: str,
    history: list[dict[str, str]],
    max_characters: int,
) -> list[dict[str, str]]:
    """Keep system instructions and newest history within a conservative hard budget.

    Oldest history is discarded first. The system instruction is never discarded.
    """
    system_cost = len(system_message) + len("system") + 16
    if max_characters < system_cost:
        raise ValueError("Prompt budget is smaller than the system instruction")

    result = [{"role": "system", "content": system_message}]
    remaining = max_characters - system_cost
    selected: list[dict[str, str]] = []

    for message in reversed(history):
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant", "system"} or not isinstance(content, str):
            continue
        cost = len(content) + len(role) + 16
        if cost <= remaining:
            selected.append({"role": role, "content": content})
            remaining -= cost
            continue
        if not selected and remaining > len(role) + 16:
            available = remaining - len(role) - 16
            selected.append({"role": role, "content": content[-available:]})
        break

    result.extend(reversed(selected))
    return result
