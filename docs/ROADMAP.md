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
- [x] Conservative task recovery after process interruption; only read-only pending steps may be resumed automatically, while uncertain side effects require manual review.
- [ ] Configuration schema migrations and model capability metadata.

## Stage 2 — Reliable memory

- [x] Persistent conversations and user memories.
- [x] BM25-style lexical retrieval with optional local embedding/cosine hybrid ranking.
- [x] Persistent research reports, skills, and specialist-agent records.
- [x] Only validated procedural skills are retrieved into conversation context.
- [x] Hybrid retrieval handles semantic-only queries and does not penalize unindexed memories on lexical matches.
- [x] Memory corrections preserve immutable prior content and expose validated correction lineage.
- [x] Explicit fact-slot contradiction candidates can be inspected without the system pretending to resolve truth automatically.
- [x] SQLite online backup and integrity-checked staged restore, with explicit operator confirmation in the console.
- [x] Deterministic retrieval and validated-skill retrievability benchmark (run with python -m aether.memory_benchmark).

Benchmark boundary: fixed lexical fixtures measure retrieval and draft-skill exclusion. They do not establish semantic-model quality, long-term forgetting curves, procedural execution success, or generalization to unseen real-world tasks. Contradictions are detected only when structured metadata facts are explicitly supplied; AETHER does not infer arbitrary factual contradictions from free-form prose.

## Stage 3 — Planning and verification

- [x] Deterministic dependency graph, cycle detection, and persisted plan progress.
- [x] Model-generated autonomy plans are normalized into validated DAGs and executed in topological order.
- [x] Model-generated bounded tool plans with a fixed tool registry.
- [x] Persisted per-step action receipts.
- [x] Test failures cause autonomous tasks to fail rather than report success.
- [x] Cooperative pause requests at safe step boundaries and resume from persisted plans/checkpoints.
- [x] Conservative crash recovery: resume only when the next unreceipted step is on a read-only replay allowlist; uncertain side effects require manual review.
- [x] Bounded automatic replanning after a missing workspace read/list target (one retry; completed actions cannot be repeated).
- [x] Deterministic postcondition verification for every fixed autonomous tool; task receipts distinguish tool-result verification from objective verification.
- [x] Five-scenario deterministic end-to-end executor benchmark (reference plans; does not measure LLM planning quality).

## Stage 4 — Research and knowledge

- [x] Public source adapters for Crossref, arXiv, Wikipedia and general web search through a bounded DuckDuckGo HTML adapter; results are treated as untrusted leads.
- [x] Provenance-aware research reports with source URL, publisher, date, evidence class, limitations and persistent report history.
- [x] Bounded public HTML/plain-text full-text ingestion with localhost/private-network rejection, redirect refusal, content-type checks, timeout and size cap.
- [x] Transparent source-triage scoring with factor breakdown; scores are not truth probabilities.
- [x] Citation evidence workflow with report membership, captured-full-text exact quote matching, lexical screening, two authenticated reviewer decisions and distinct adjudication.
- [x] Systematic-review protocol with databases, explicit queries/date, inclusion/exclusion criteria, source deduplication and provenance preservation.
- [x] Dual authenticated title/abstract screening, full-text eligibility, data extraction, risk-of-bias assessment and adjudication.
- [x] Design-specific structured risk-of-bias checklists for randomized, nonrandomized, qualitative and cross-sectional studies; the workflow explicitly does not claim these custom checklists are validated RoB 2/ROBINS-I/CASP/JBI instruments.
- [x] PRISMA 2020 27-item operator-completed reporting checklist with per-item locations/rationales, plus PRISMA-style study-flow counts. Checklist completion is not automatic compliance certification.
- [x] GRADE-inspired five-domain certainty assessment with explicit judgments, rationale and deterministic downgrade levels; not an official GRADEpro assessment.
- [x] Egger regression publication-asymmetry screening for at least ten compatible extracted effects; mixed effect measures are rejected, and no trim-and-fill/selection model is claimed.
- [x] Evidence-table synthesis and DerSimonian-Laird random-effects meta-analysis for compatible supplied effects and standard errors. Quantitative extraction can verify exact source quotes and numeric estimate/SE tokens when captured full text is available; unverified extractions are explicitly labeled.
- [x] Reproducible CSV experiment analysis with SHA-256, Python/algorithm version, descriptive statistics, Welch two-sample tests, pooled effect size and confidence intervals.
- [x] Raw independent-group CSV observation ingestion with row validation, SHA-256 provenance, persisted source CSV bytes in the local experiment event history, group summaries and Welch standard error.
- [x] Preregistered experiment records, typed hypothesis claims, explicit lifecycle transitions and append-only event history.
- [x] Deterministic synthetic end-to-end Phase 4 acceptance fixture covers dual-review authentication, systematic-review synthesis/meta-analysis, PRISMA flow/checklist, GRADE-inspired grading, publication-bias eligibility, citation adjudication and raw experiment ingestion.
- [x] Local reviewer accounts use salted PBKDF2-HMAC-SHA256 hashes. After the first account is registered, every screening, full-text, risk-of-bias, extraction, adjudication and citation-review decision requires the matching account password.
- [ ] Optional adoption of externally validated/licensed risk-of-bias instruments and independent methodological validation. AETHER's included domain checklists are transparent structured aids, not substitutes for those validated instruments.
- [ ] External identity-provider integration and cross-machine reviewer federation; local AETHER accounts are enforced for this repository workflow.

Phase 4's implemented software workflow is complete within the documented bounded scope. The remaining unchecked items require external validated instruments/methodological validation or external identity infrastructure; they are not silently represented as complete. Search rankings and source scores are not truth measures; exact quote matching is not semantic entailment. Meta-analysis and Egger screening depend on comparable independent effects and supplied standard errors. Raw observation ingestion currently supports independent groups only.

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
- [ ] Background scheduler and bounded goal discovery.
- [x] Cooperative pause/resume and conservative crash recovery for explicitly queued autonomy tasks.
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
- [x] One bounded replan after a failed read-only workspace lookup; completed actions are not repeated.
- [x] Cooperative pause/resume at step boundaries with persisted plan and receipts.
- [x] Conservative recovery after process interruption; only read-only pending steps may be replayed automatically.
- [x] Five-scenario end-to-end executor benchmark in `benchmarks/phase1_cases.json` (fixed reference plans; planner intelligence is outside its scope).


### Phase 1 operation notes

- Request a cooperative pause with `TaskManager.request_pause(task_id)`; the active tool is allowed to finish, then execution pauses at the next checkpoint boundary.
- Resume paused work with `AutonomyRunner.resume_task(task_id)`.
- Interrupted tasks are resumed only when their next unreceipted operation is one of `workspace.read`, `workspace.list`, `world.state`, or `world.actions`. Writes, tests, research persistence, and simulations are never automatically replayed.
- Automatic replanning is intentionally narrow: one retry after a missing workspace read/list target. It does not retry writes, tests, network research, or arbitrary failures.
- Run the deterministic executor benchmark with `python -m aether.benchmark`. It validates reference plans and tool postconditions; it is not a held-out evaluation of the LLM's reasoning ability.
