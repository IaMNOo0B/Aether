# AETHER Engineering Roadmap

A capability is marked complete only when its implementation and tests exist. Optional hardware integrations still require a clean local test.

## Stage 0 — Repository and environment baseline

- [x] Windows-first Python 3.12 foundation and double-click setup/launch scripts.
- [x] Standard-library configuration, SQLite persistence, and Ollama adapter.
- [x] Windows CI workflow.
- [ ] Clean install and launch test on the user's Windows PC.

## Stage 1 — Cognitive foundation

- [x] Provider protocol and explicit registry (Ollama only).
- [x] Durable task lifecycle states and task receipts.
- [x] Character-budgeted context construction.
- [x] Structured logging and redacted diagnostics.
- [x] Runtime configuration validation.
- [ ] Provider retry/backoff/fallback policy.
- [x] Conservative task recovery after process interruption (interrupted work is marked failed for manual review; uncertain actions are never replayed).
- [ ] Configuration schema migrations and model capability metadata.

## Stage 2 — Reliable memory

- [x] Persistent conversations and user memories.
- [x] Basic lexical memory retrieval.
- [x] Persistent research reports, skills, and specialist-agent records.
- [x] Only validated procedural skills are retrieved into conversation context.
- [ ] Hybrid keyword/vector retrieval.
- [ ] Memory correction, contradiction tracking, version history, backup/restore.
- [ ] Retrieval and skill-retention benchmarks.

## Stage 3 — Planning and verification

- [x] Deterministic dependency graph, cycle detection, and persisted plan progress.
- [x] Model-generated autonomy plans are normalized into validated DAGs and executed in topological order.
- [x] Model-generated bounded tool plans with a fixed tool registry.
- [x] Persisted per-step action receipts.
- [x] Test failures cause autonomous tasks to fail rather than report success.
- [ ] Full pause/resume and crash recovery.
- [x] Deterministic postcondition verification for every fixed autonomous tool; task receipts distinguish tool-result verification from objective verification.
- [ ] Held-out multi-step task benchmark.

## Stage 4 — Research and knowledge

- [x] Public source adapters for Crossref, arXiv, and Wikipedia.
- [x] Provenance-aware reports with source URL, publisher, date, evidence class, and limitations.
- [x] Persistent research report history.
- [ ] Full-text ingestion, broader web search, source-quality scoring, and citation verification.
- [ ] Entity graph, hypothesis/experiment store, and systematic review workflow.

Research scope is intentionally bounded; index ranking is not evidence quality and this is not a systematic literature review.

## Stage 5 — RLCD investigation

- [x] Identify OpenJev-RLCD and review its two-stage method.
- [x] Document GPU, benchmark, and licensing constraints in [RLCD feasibility](research/RLCD_FEASIBILITY.md).
- [x] Persistent binary decision ledger and Brier/log-loss/ECE/AURC metrics.
- [ ] Held-out calibration benchmarks and typed decision-producer integration.
- [ ] Training reproduction on suitable GPU resources.

AETHER is not RLCD-trained. Full reproduction is deferred because the target RTX 2060 has 6 GB VRAM, while the paper reports 48 GB GPU runs.

## Stage 6 — Bounded autonomy

- [x] Permission-enforced fixed tool gateway.
- [x] Read-only research and workspace tools.
- [x] Workspace writes require feature flag and per-action human approval.
- [x] Test execution requires feature flag and per-action human approval.
- [x] Workspace path containment, file-size caps, step limits, and execution timeout.
- [x] Persisted plans, progress, and action receipts.
- [ ] Background scheduler, bounded goal discovery, pause/resume, and crash recovery.
- [ ] OS-level sandbox for arbitrary code (not provided by the current copy-based candidate workflow).

No arbitrary shell, unrestricted filesystem, or unrestricted URL-fetch tool is exposed.

## Stage 7 — Controlled self-engineering

- [x] Read-only Python source inspection.
- [x] Unified patch proposal saved as a reviewable experiment.
- [x] Protected safety/configuration/startup paths and patch path/type validation.
- [x] Explicit approval before running baseline and candidate tests.
- [x] Candidate copy, compilation, test result recording, and rollback of candidate worktree.
- [ ] OS-level isolation, security scanning, performance benchmarking, and signed immutable audit history.
- [ ] Human-reviewed integration/merge workflow.

The candidate copy is not an OS security sandbox. Generated tests execute with the current user's permissions. Production integration is never automatic.

## Stage 8 — Continual learning

- [x] Extract procedural workflows from supplied reference material using the configured model.
- [x] Store drafts separately from validated skills.
- [x] Require held-out evaluation evidence before retrieval into assistant context.
- [ ] Automated benchmark harness for transfer, retention, and regression.
- [ ] Continual model-weight adaptation/LoRA and catastrophic-forgetting evaluation.
- [ ] Meta-learning experiments.

Current learning is knowledge/procedure-level. No model weights are updated.

## Stage 9 — Desktop, voice, and collective intelligence

- [x] Optional allowlisted Windows application launch.
- [x] Optional mouse/keyboard input with explicit per-action approval and small key allowlist.
- [x] Optional offline Vosk push-to-talk recognition and local speech output.
- [x] Three persisted specialist roles: researcher, planner, critic.
- [ ] Independent specialist models, parallel execution, and stronger evidence-based arbitration.
- [ ] Vision/OCR/screenshot understanding, audio event understanding, and video processing.
- [ ] Local Windows hardware and microphone validation.

Specialist roles currently share one local model and run sequentially to fit constrained VRAM. Voice is push-to-talk, not an always-on wake-word service.

## Engineering rules

1. Preserve existing user data.
2. Prefer root-cause fixes over symptom patches.
3. Never report an unexecuted test as passing.
4. Separate retrieval, procedural learning, policy adaptation, and model-weight updates.
5. Do not claim consciousness or general intelligence from architecture labels.
6. Keep Windows 10/11 as the primary platform.


## Phase 1 implementation log — orchestration and verification

- [x] Integrate the existing `PlanGraph` into model-generated autonomous plans.
- [x] Reject duplicate IDs, missing dependencies, self-dependencies and cycles before executing tools.
- [x] Execute dependency-ready steps in deterministic topological order.
- [x] Persist plan-step IDs, dependencies, receipts and completed-step IDs at checkpoints.
- [x] Verify observable postconditions for workspace reads/writes/listings, research reports, test runs and symbolic world-model tools.
- [x] Fail the task when a tool result cannot be independently verified.
- [x] Separate `tool_results_verified` from the stronger `verified` flag, which additionally requires a successful test-run receipt.
- [ ] Automatic replanning after a failed step (not enabled; failure remains fail-closed).
- [ ] Explicit pause/resume and safe recovery of resumable, idempotent steps.
- [ ] End-to-end goal-level benchmark with held-out tasks.
