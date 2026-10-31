import pytest

from aether.context import fit_messages


def test_context_keeps_system_and_newest_history_within_budget():
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "old" * 20},
        {"role": "assistant", "content": "older answer"},
        {"role": "user", "content": "latest"},
    ]
    fitted = fit_messages("rules", messages[1:], 80)
    assert fitted[0] == {"role": "system", "content": "rules"}
    assert fitted[-1]["content"] == "latest"
    assert sum(len(item["content"]) + len(item["role"]) + 16 for item in fitted) <= 80


def test_context_rejects_budget_smaller_than_system_prompt():
    with pytest.raises(ValueError):
        fit_messages("system rules", [], 3)
