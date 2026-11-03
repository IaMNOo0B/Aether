# Master Directive implementation traceability

This document maps the 1,992-line PROJECT AETHER — OMEGA infinity master directive to the current repository. It is an implementation ledger, not a declaration of general intelligence. A checked source module is not considered operational until relevant tests and target-machine validation pass.

## Milestones 0–10

| Milestone | Current implementation | Evidence / boundary |
|---|---|---|
| 0 — Frontier investigation | Partial research and hardware feasibility documents | RLCD method review and feasibility are documented. A complete current survey of every frontier topic and Windows hardware validation remain outstanding. |
| 1 — Cognitive foundation | Core orchestrator, provider registry/failover, tasks, context, SQLite, diagnostics, configuration | Windows CI covers software tests. Clean install and launch on the user's PC remain unverified. |
| 2 — Memory and continuity | Persistent memory, hybrid retrieval, provenance, correction lineage, self-model, task checkpoints | Existing tests cover persistence and recovery. Long-term retention curves and user-machine restart validation remain outstanding. |
| 3 — Reasoning and planning | Dependency DAG, bounded tool gateway, receipts, pause/resume, recovery, symbolic world-state simulation | Deterministic planning tests exist. Formal theorem proving, broad causal inference and generalization to unseen tasks are not claimed. |
| 4 — Research intelligence | Source discovery, ingestion, systematic review, citation review, experiment analysis, provenance and knowledge graph | Phase 4's bounded workflow is implemented in the stacked Phase 4 branch. Search ranking is not truth; citation quote matching is not semantic entailment. |
| 5 — RLCD | Outcome-grounded decision ledger, calibration metrics, held-out calibration benchmark, prompt-deduplicated label-stratified dataset preparation, external training manifest/runner adapter | RLCD training has not been executed. The adapter requires an explicitly provisioned external GPU; the published recipe's 48 GB GPU environment is outside the RTX 2060 6 GB machine. |
| 6 — Autonomous operation | Permission-gated autonomy, persisted scheduler, bounded goal discovery and approval-gated scheduling | Scheduled work is read-only and opt-in. Project autonomy, independent goal generation loops and unrestricted background execution are not enabled. |
| 7 — Self-engineering | Read-only inspection, patch proposals, protected paths, candidate worktree, Docker no-network sandbox path, rollback, hash-chained audit journal | Docker and a locally prebuilt test image are required for OS-isolated evaluation. Hash chaining is tamper-evident, not immutable against a privileged attacker. Production integration remains human-reviewed. |
| 8 — Advanced learning | Draft procedural skills and held-out/transfer/retention benchmark runner | Skill validation requires all three evaluation splits. No model-weight update, LoRA training, catastrophic-forgetting experiment or meta-learning result is claimed. |
| 9 — Multimodal and collective intelligence | Independent parallel specialist workers with a separate verifier; optional OCR, Vosk transcription, FFmpeg frame extraction and vision-provider interface | Parallel workers require four distinct configured provider instances. OCR is not visual understanding; frame extraction is not video reasoning; audio transcription is not event classification. Local camera, GPU and microphone validation remains outstanding. |
| 10 — Frontier expansion | Persistent research frontier with proposal, investigation, experiment and outcome lifecycle | Research opportunities are recorded and reviewed; no unbounded autonomous internet research or automatic integration is enabled. |

## Part-level implementation map

| Master directive part | Repository mapping | Status |
|---|---|---|
| 0 — Fundamental mission | aether/engine.py, aether/self_model.py, capability records | Partial; architecture is not AGI. |
| 1 — Engineering philosophy | tests, acceptance suite, provenance and explicit limitations | Implemented as engineering controls, not every target capability. |
| 2 — Cognitive architecture | engine.py, tasks.py, providers.py, decisions.py, acceptance.py | Partial; cognitive routing remains bounded. |
| 3 — General reasoning | planning.py, world_model.py, decisions.py, research synthesis | Partial; formal logic, theorem proving and broad causal reasoning are not complete. |
| 4 — Persistent identity | self_model.py, SQLite history, task recovery | Partial; identity is operational metadata, not subjective consciousness. |
| 5 — Memory | memory.py, embeddings.py, knowledge.py, storage.py | Core memory implemented; scheduled consolidation and long-horizon retention need more validation. |
| 6 — Open-ended autonomy | autonomy.py, scheduler.py, goals.py, permission controls | Bounded and opt-in; no unlimited autonomous initiative. |
| 7 — RLCD and decision intelligence | decisions.py, calibration.py, calibration_benchmark.py, rlcd_training.py | Evaluation infrastructure implemented; training reproduction not executed. |
| 8 — Continual learning | learning.py, learning_benchmarks.py | Procedural learning implemented; weight adaptation and meta-learning outstanding. |
| 9 — Recursive self-improvement | self_engineering.py, sandbox.py, audit.py | Candidate-only workflow; secure execution depends on Docker configuration; no automatic integration. |
| 10 — Scientific discovery | research.py, research_advanced.py, research_completion.py, frontier.py | Bounded research workflows implemented; automatic discovery and replication are not claimed. |
| 11 — World models | world_model.py | Symbolic state transitions only; no learned predictive world model. |
| 12 — Multi-agent intelligence | agents.py | Sequential roles and independent parallel providers; independent reviewer output is not ground truth. |
| 13 — Multimodal intelligence | perception.py, voice.py | Optional OCR/transcription/frame extraction; full multimodal reasoning remains incomplete. |
| 14 — Software engineering | self_engineering.py, planning.py, tests and diagnostics | Candidate code inspection/patching/testing; broad autonomous engineering remains bounded. |
| 15 — Computational optimization | resources.py, context.py, provider routing | Basic budgets and resource checks; model/GPU scheduling optimizations need measurements. |
| 16 — Model research | providers.py, ollama.py, model endpoint configuration | Replaceable text providers; no automatic model discovery or model training. |
| 17 — Knowledge architecture | knowledge.py, research_advanced.py, storage.py | Provenance-aware entities and temporal claims; cross-domain inference remains experimental. |
| 18 — Environment interaction | desktop.py, autonomy.py | Allowlisted and approval-gated actions; no unrestricted desktop or browser control. |
| 19 — Cybersecurity | autonomy.py, self_engineering.py, sandbox.py, secret environment handling | Defensive boundaries exist; Docker isolation must be provisioned and independently reviewed. |
| 20 — Resource management | resources.py, config.py, task limits | Core limits exist; per-agent and per-experiment cost accounting needs extension. |
| 21 — Permission architecture | emergency stop, tool allowlists, explicit approval callbacks | Bounded workflows enforce permissions; autonomous tasks cannot change their own safeguards. |
| 22 — Evaluation | acceptance.py, calibration_benchmark.py, learning_benchmarks.py, test suite | Multiple deterministic tests; broad held-out benchmark coverage still needs real datasets. |
| 23 — Failure intelligence | task receipts, diagnostics, provider fallback, recovery | Basic classification/recovery; predictive failure modeling not implemented. |
| 24 — Future research frontier | frontier.py, research/ | Persistent manual research ledger; continuous evidence discovery not automated. |
| 25 — Hardware strategy | resources.py, Ollama provider, Windows scripts | Configured for the RTX 2060/16 GB machine; actual local validation is still required. |
| 26 — Observability | logs, diagnostics, status and resource commands | Operational logs/status exist; a live graphical dashboard is not implemented. |
| 27 — Project architecture | modular aether package and tests | Existing modular layout retained; no empty placeholder modules created. |
| 28 — Development milestones | this traceability ledger plus docs/ROADMAP.md | Both records are maintained separately; the master directive remains the source of truth for this implementation. |
| 29 — Software quality | pytest, CI, configuration validation, backups and docs | CI is the software gate; local Windows installation and hardware tests remain required. |
| 30 — Acceptance tests | acceptance.py, tests/ | Some deterministic acceptance tests exist; all 12 master tests need end-to-end evidence before global completion. |
| 31 — Engineering rules | branch-based work, CI, source review, backup and explicit limitations | Applied to repository work; user-machine operations require local validation. |
| 32 — Final design principle | replaceable interfaces, frontier ledger and candidate workflow | Architectural direction only; continual capability growth must be measured, not asserted. |

## External prerequisites and honest blockers

- RLCD training: suitable external GPU, verified training implementation, dataset/license review, baseline, checkpoint identity and held-out evaluation.
- OS-isolated candidate execution: Docker daemon plus locally built aether-python-test:3.12 image. Build with: docker build -t aether-python-test:3.12 -f sandbox/Dockerfile sandbox
- Multimodal models: user-approved local model files or separately configured providers; no model downloads occur automatically.
- Parallel independent workers: four separate configured provider endpoints/instances (researcher, planner, critic, verifier).
- Windows validation: run the installer, startup, model inference, GPU/VRAM monitoring, audio devices and desktop-control checks on the target PC.
- External identity, paid APIs, public publishing and irreversible actions require user-controlled credentials/approval and are not silently enabled.

No milestone is marked complete solely because a module or interface exists. Completion requires tests, independent evidence and target-environment validation appropriate to that capability.
