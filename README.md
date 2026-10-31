# AETHER

A Windows-first, local-first AI orchestration platform.

## Target machine

- Windows 10/11
- Python 3.12+
- NVIDIA RTX 2060 6 GB
- Ryzen 5 3600
- 16 GB RAM
- Ollama local inference

The core runtime uses Python's standard library. Optional desktop and voice features require extra packages and are not installed automatically.

## Windows setup

1. Install Python 3.12+ from https://www.python.org/downloads/windows/ and enable the Python Launcher.
2. Download this repository using **Code → Download ZIP**, extract it, or clone it with Git.
3. Double-click `INSTALL.bat`.
4. Double-click `START_AETHER.bat`.

AETHER stores history, memory, research reports, skills, and task receipts in `data/aether.sqlite3`. Do not delete `data/` if you want to preserve them. Generated self-engineering candidates are stored under `.aether/experiments/`; the approved workspace is `workspace/`.

## Enable the local model

1. Install Ollama for Windows: https://ollama.com/download/windows
2. Run:

   ```powershell
   ollama pull phi3:mini-4k-q4_0
   ```

3. In `config/settings.toml`, set `enabled = true` under `[models]`.
4. Ensure Ollama is running, then launch AETHER.

AETHER never downloads models automatically. Phi-3 Mini's 4K context and model capacity limit long research synthesis and complex multi-step reasoning. Keep the local prompt small; the architecture is designed to allow a stronger provider to be added later.

## Implemented capabilities

### Research

- `/research <question>` searches Crossref scholarly records, arXiv preprints, and Wikipedia summaries.
- Research reports retain source URLs, publisher, date, evidence class, and limitations in SQLite.
- `/reports` lists saved reports.

This is a bounded source adapter, not a general-purpose web crawler or a systematic literature review. Search ranking is not evidence quality.

### Bounded autonomous execution

Enable `[autonomy] enabled = true` in `config/settings.toml` to use `/autonomy <objective>` and `/autonomy queue`.

The model can plan only against a fixed tool registry:
- Public research search.
- List/read files inside the configured workspace.
- Write text files inside the workspace, only when enabled and individually approved.
- Run the repository's pytest suite, only when enabled and individually approved.

There is no arbitrary shell, unrestricted filesystem, arbitrary URL fetch, or Python eval tool. The task plan, progress, and action receipts are persisted. The default configuration disables autonomy and workspace writes.

### Desktop control

Install the optional package:

```powershell
.venv\Scripts\python -m pip install -e ".[desktop]"
```

Set `[desktop] enabled = true`. The default application allowlist contains Notepad and Calculator. Use:
- `/desktop open notepad.exe`
- `/desktop click X Y`
- `/desktop type TEXT`
- `/desktop key enter`

To enable mouse/keyboard input, also set `allow_input = true`. Each click or typing action requires an explicit `APPROVE` confirmation. Only a small single-key allowlist is supported. Dangerous shell/admin applications are blocked even if accidentally added to the allowlist.

### Offline voice

Install the optional packages:

```powershell
.venv\Scripts\python -m pip install -e ".[voice]"
```

Download a Vosk offline speech model from https://alphacephei.com/vosk/models, extract it to `models/vosk`, and keep `[voice] enabled = true` in the configuration. Then run `/voice`, say the configured wake word (default: "Jarvis") followed by a command, and AETHER will speak its reply using Windows speech output.

Voice is push-to-talk, not an always-listening background microphone. The model files are not included in this repository.

### Procedural learning

- `/learn title | source | reference material` asks the model to extract a reusable workflow, or stores a clearly marked draft if the model is disabled.
- `/skills` lists stored workflows.
- `/skill-eval skill-id | pass/fail | held-out test evidence` records an evaluation.
- Only skills marked validated are injected into relevant chat context.

This is knowledge and procedural learning. It does **not** update Phi-3's model weights, and a saved note is not proof of learning.

### Multi-agent coordination

- `/agents <objective>` runs researcher, planner, and critic roles.
- Each deliverable is persisted and can be inspected through the agent-run records.

To fit one 6 GB GPU, these roles run sequentially through the same configured model. They are not three independent models, and agreement is not proof. The critic is an additional review pass, not an independent ground-truth oracle.

### Controlled self-engineering

- `/self-improve <objective>` inspects project Python files and proposes a unified patch.
- `/self-improve-test <experiment-id>` asks for explicit approval, runs baseline tests, applies the patch to a copied candidate tree, then compiles and tests that candidate.
- Candidate patches and receipts are stored in `.aether/experiments/`.

A candidate is never automatically merged or copied into production. The copied worktree is **not** a secure OS sandbox: tests execute code with the current user's permissions. Review the patch before approving evaluation. Protected policy, startup, and safety-control files cannot be patched through this workflow.

## Other commands

- `/status` — runtime and feature flags.
- `/tasks` — recent task records.
- `/diagnostics` — redacted report in `logs/diagnostics.json`.
- `/calibration` — evaluate explicitly logged binary decision outcomes.
- `/remember <fact>` — save a user-provided memory.
- `/quit` — exit without deleting data.

## Development tests

```powershell
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest
```

## Honest capability boundaries

Implemented in this milestone: source-based research, bounded model-planned tool execution, permission checks, persistent task receipts, allowlisted desktop interaction, optional offline push-to-talk voice, draft-to-validated procedural skills, sequential specialist roles, and isolated patch proposal/evaluation.

Not implemented: always-on background autonomy, general web browsing, unrestricted computer use, independent specialist models, true continual model-weight training, LoRA fine-tuning, meta-learning, a secure OS sandbox, automatic production code integration, full multimodal video understanding, or full RLCD training. The RTX 2060 6 GB target is not suitable for reproducing the referenced 48 GB RLCD training setup.

## Safety and data

- External web pages and model-generated plans are untrusted input.
- Never commit API keys, tokens, private documents, or model files.
- Destructive and consequential actions are not exposed as autonomous tools.
- Preserve `data/` and review candidate patches before use.
