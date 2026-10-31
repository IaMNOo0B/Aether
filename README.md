# AETHER

A Windows-first, local-first AI orchestration platform.

## Initial engineering target

- Windows 10/11
- Python 3.12+
- RTX 2060 6GB / Ryzen 5 3600 / 16GB RAM
- Optional Ollama local inference
- Optional cloud model providers (disabled until explicitly configured)

## Status

The repository is being initialized from an empty baseline. The first milestone focuses on a reliable application shell, persistent SQLite state, model-provider interfaces, honest capability reporting, and Windows launch/setup scripts.

Advanced capabilities such as continual learning, RLCD, autonomous self-engineering, multimodal perception, and multi-agent execution are research milestones—not claims about features that already exist.

## Quick start

1. Install Python 3.12 or newer.
2. Double-click `INSTALL.bat`.
3. Double-click `START_AETHER.bat`.

The initial application runs without a model. Configure Ollama separately to enable local LLM responses.

## Safety and data

- User data is stored under `data/`.
- Configuration is stored under `config/`.
- Secrets must never be committed to Git.
- Consequential or destructive actions require explicit approval.
- Self-modification is not enabled in the initial milestone.
