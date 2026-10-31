import json

from aether.diagnostics import write_diagnostics


def test_diagnostics_redacts_secret_named_fields(tmp_path):
    path = write_diagnostics(
        tmp_path / "diagnostics.json",
        {"provider": "ollama", "api_key": "do-not-write", "model": "phi3"},
    )
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved == {"model": "phi3", "provider": "ollama"}
