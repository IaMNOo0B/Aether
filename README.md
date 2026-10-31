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

AETHER never downloads models automatically. Phi-3 Mini's 4K context and model capacity limit long research synthesis and complex multi-step reasoning. AETHER also supports an OpenAI-compatible endpoint (cloud or local) and ordered fallbacks; it does not silently download or substitute models.

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


### Model providers and failover

The default provider is Ollama. To use an OpenAI-compatible endpoint such as LM Studio, vLLM or a compatible cloud API, edit `[models]`:

```toml
provider = "openai_compatible"
enabled = true
base_url = "http://127.0.0.1:1234/v1"
model = "your-loaded-model"
api_key_env = ""
```

For a service requiring a key, set `api_key_env = "AETHER_API_KEY"` and provide that environment variable to the AETHER process. Keys are not written to TOML. Optional `[[models.fallbacks]]` entries define ordered fallback endpoints; AETHER tries the primary first and falls back only after a provider error.

### Memory and continuity

- Memories are typed and stored locally with source/status metadata.
- Retrieval uses a local BM25-style lexical ranker; it is **not** embedding-based semantic search.
- `/memory-search <query>` shows ranked matches.
- `/correct-memory id | corrected content | reason` preserves the old record and appends a correction.
- Task state changes and checkpoints have an append-only event history. `/task-events <full-task-id>` inspects it.
- On startup, tasks left running by a crash are marked interrupted/failed for manual review. AETHER does not blindly replay uncertain side effects.

### Decisions and calibration

- `/decision-log question | probability_true | domain` records a typed prediction.
- `/decision-resolve id | true/false` records the observed outcome.
- `/decision-metrics [domain]` computes outcome-based calibration metrics.
- `/decision-assess probability_true | domain` uses historical outcomes in the same domain/confidence band and a Wilson lower confidence bound to accept or abstain.

This is a conservative runtime decision gate, **not RLCD training**. Manually supplied probabilities are labelled as user-supplied; they are not treated as calibrated model outputs.

### Emergency stop

- `/emergency-stop` immediately records a persistent stop flag in SQLite.
- It blocks new autonomy tasks, cancels queued tasks before execution, prevents further scheduled tasks, and blocks desktop automation and self-engineering test execution.
- The runner rechecks the flag between tool calls and around approval prompts.
- `/resume-autonomy` requires the literal confirmation `RESUME`.
- Every state change is recorded in the system-control event history.

This cannot undo an external action already in progress. A long-running subprocess or individual OS operation may finish before the next stop check; the stop prevents subsequent actions rather than forcibly terminating the Windows process.

### Scheduled read-only autonomy

Schedules can be created with `/schedule add interval_seconds | objective | max_runs(optional)`, inspected with `/schedule list`, paused/resumed with `/schedule pause ID` and `/schedule resume ID`.

To run them in the background, set all of the following:
- `[models] enabled = true`
- `[autonomy] enabled = true`
- `[scheduler] enabled = true`

The scheduler is disabled by default. It only processes scheduled tasks through read-only tools. It never approves workspace writes or test execution. A model-generated plan that requests a side effect is cancelled rather than approved.

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
- `/desktop screenshot workspace/screen.png`

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

A candidate is never automatically merged or copied into production. Git for Windows must be installed for patch application. The copied worktree is **not** a secure OS sandbox: tests execute code with the current user's permissions. Review the patch before approving evaluation. Protected policy, startup, and safety-control files cannot be patched through this workflow.

## Other commands

- `/status` — runtime and feature flags, including emergency-stop state.
- `/emergency-stop` / `/resume-autonomy` — stop and explicitly resume autonomous execution.
- `/tasks` — recent task records.
- `/diagnostics` — redacted report in `logs/diagnostics.json`.
- `/calibration` — evaluate explicitly logged binary decision outcomes.
- `/self-model` — inspect identity and capability evidence states.
- `/remember <fact>` — save a user-provided memory.
- `/quit` — exit without deleting data.

## Development tests

```powershell
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest
```

## Honest capability boundaries

Implemented in this milestone: source-based research, bounded model-planned tool execution, permission checks, persistent task receipts, allowlisted desktop interaction, optional offline push-to-talk voice, draft-to-validated procedural skills, sequential specialist roles, and isolated patch proposal/evaluation.

Not implemented: autonomous goal discovery, unrestricted/background project autonomy, general web browsing, semantic embedding memory, knowledge graph extraction, independent parallel specialist models, true continual model-weight training, LoRA fine-tuning, meta-learning, a secure OS sandbox, automatic production code integration, full multimodal vision/video understanding, the external RLCD training adapter, or full RLCD training. The RTX 2060 6 GB target is not suitable for reproducing the referenced 48 GB RLCD training setup.

## Safety and data

- External web pages and model-generated plans are untrusted input.
- Never commit API keys, tokens, private documents, or model files.
- Destructive and consequential actions are not exposed as autonomous tools.
- Preserve `data/` and review candidate patches before use.
