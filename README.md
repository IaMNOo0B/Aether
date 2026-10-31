# AETHER

A Windows-first, local-first AI orchestration platform.

## Target machine

- Windows 10/11
- Python 3.12
- NVIDIA RTX 2060 6GB
- Ryzen 5 3600
- 16GB RAM
- Ollama-compatible quantized local inference

The runtime foundation uses Python's standard library. It does not download models, install background services, or require API keys during setup.

## Windows setup

1. Install Python 3.12 from https://www.python.org/downloads/windows/ and enable the Python Launcher.
2. Download this repository using **Code → Download ZIP**, then extract it; or clone it with Git.
3. Double-click `INSTALL.bat`.
4. Double-click `START_AETHER.bat`.

AETHER stores its SQLite database in `data/aether.sqlite3` and logs in `logs/aether.log`. Do not delete `data/` if you want to preserve its memory and conversation history.

## Enable local Ollama chat

1. Install Ollama for Windows from https://ollama.com/download/windows.
2. In Command Prompt or PowerShell, run:

   ```powershell
   ollama pull phi3:mini-4k-q4_0
   ```

3. Open `config/settings.toml` and change `enabled = false` to `enabled = true` under `[models]`.
4. Ensure Ollama is running, then launch AETHER.

The configured model must already exist in Ollama. AETHER deliberately does not download it automatically. The default is Phi-3 Mini 4K Q4_0. Its small size is suitable for constrained local hardware, but its 4K context and smaller model capacity limit long-context work and complex reasoning. Keep prompts and conversation history compact.

## Commands

- `/status` — show runtime and model connectivity.
- `/tasks` — show the latest recorded task receipts.
- `/diagnostics` — write a redacted runtime diagnostic report to `logs/diagnostics.json`.
- `/remember <fact>` — save a user-provided memory locally.
- `/quit` — exit without deleting data.

## Current implementation status

Implemented: Windows setup/launch scripts, validated TOML configuration, SQLite conversation and memory storage, a provider protocol and registry (Ollama only), deterministic context budgeting, durable task lifecycle records, a dependency-aware planning core, centralized logging, diagnostic reports, and automated tests.

Not implemented yet: provider retries/fallback, task recovery after crashes, vector/hybrid memory, browser/desktop tools, cloud providers, LLM-driven plan generation and execution, RLCD training, model training, continual learning, self-modifying code, multimodal systems, and multi-agent orchestration. The RLCD method investigation and hardware feasibility note are documented in `docs/research/RLCD_FEASIBILITY.md`. These are planned research and engineering milestones, not existing capabilities.

## Development tests

Install the development extra and run:

```powershell
python -m pip install -e ".[dev]"
python -m pytest
```

## Safety and data

- Secrets must never be committed to Git.
- External content is untrusted input.
- Consequential or destructive actions require explicit approval.
- Self-modification and unrestricted autonomy are intentionally absent from the foundation.
