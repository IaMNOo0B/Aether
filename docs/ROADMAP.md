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

- [x] Public source adapters for Crossref, arXiv, and Wikipedia.
- [x] Provenance-aware reports with source URL, publisher, date, evidence class, and limitations.
- [x] Persistent research report history.
- [x] Operator-supplied public HTML/plain-text full-text ingestion with private-network/localhost rejection, redirect refusal, content-type checks, timeout and 2 MB cap.
- [x] Transparent source-triage scoring with explicit factor breakdown and caveats; scores are not truth probabilities.
- [x] Citation URL reachability checks tied to persisted report sources; claim-support verification remains a human review task.
- [x] Persisted evidence-review inventory and protocol checklist; sources remain pending human screening and are never silently excluded.
- [x] General public web search through a bounded DuckDuckGo HTML adapter, with untrusted result metadata and no automatic arbitrary result-page fetching.
- [x] Systematic review protocol with databases, explicit queries/date, inclusion/exclusion criteria, source deduplication, dual title/abstract screening and third-reviewer conflict resolution.
- [x] Full-text eligibility screening with locators, dual-review agreement and explicit adjudication.
- [x] Generic five-domain risk-of-bias assessment and dual-reviewed data extraction with adjudication.\n- [x] Design-specific structured domain checklists for randomized, nonrandomized, qualitative and cross-sectional studies; these are not validated RoB 2/ROBINS-I/CASP/JBI instruments.\n- [x] GRADE-inspired five-domain certainty assessment with explicit operator judgments and downgrade rationale.\n- [x] PRISMA-style screening-flow count report with explicit limits; no PRISMA compliance certification.\n- [x] Egger regression publication-asymmetry screening for at least ten extracted studies; no trim-and-fill or selection model.
- [x] Evidence-table synthesis and DerSimonian-Laird random-effects meta-analysis when at least two compatible effect estimates and standard errors are supplied.
- [x] Exact-quote citation passage matching, lexical alignment screening, two-reviewer adjudication and third-reviewer disagreement resolution.
- [x] Reproducible CSV experiment analysis with input SHA-256, Python/algorithm version, descriptive statistics, Welch two-sample tests, pooled effect size and confidence intervals.
- [x] Analysis artifacts persist as research reports and append to experiment event history.
- [x] Preregistered experiment records linked to typed hypothesis claims, with required objective/method/success criteria, explicit start/complete/cancel transitions, evidence-backed outcomes and append-only lifecycle events.
- [ ] Held-out end-to-end systematic-review acceptance benchmark, publication-bias analysis, certainty-of-evidence grading, validated design-specific risk-of-bias tools and authenticated reviewer identities.
- [ ] Raw observation ingestion from experiment runs and automatic validation of supplied effect estimates/standard errors.

Stage 4 remains partial until the held-out end-to-end acceptance benchmark passes and authenticated reviewer identities are enforced across all review stages. Search rankings and source scores are not truth measures; exact-quote matching is not semantic entailment. The design-specific checklists are not validated instruments. Meta-analysis and Egger screening assume comparable independent effects and supplied standard errors; raw ingestion currently supports independent groups only.

## Stage 5 — RLCD investigation and frontier interface

- [x] Identify OpenJev-RLCD and review its two-stage method.
- [x] Document GPU, benchmark, and licensing constraints in [RLCD feasibility](research/RLCD_FEASIBILITY.md).
- [x] Persistent binary decision ledger and Brier/log-loss/ECE/AURC metrics.
- [x] Typed decision-producer API with producer type, source reference, optional model ID and evidence hash.
- [x] Symbolic world-model integration: declared success probabilities are logged at simulation time and only unambiguous observed outcomes are resolved.
- [x] Deterministic held-out calibration benchmark with fixed seed, dataset SHA-256, disjoint train/test records and Brier/log-loss/ECE/AURC reporting.
- [x] Read-only runtime dashboard covering provider status, resource telemetry, capability evidence states, research inventory, decisions, tasks and saved acceptance results.
- [x] Bounded local image intake and vision-provider routing for PNG/JPEG/GIF/WebP, with size/signature checks and explicit consent before remote upload.
- [ ] Held-out benchmark on a sufficiently large, independently curated real-world dataset.
- [ ] Continuous research/capability discovery with scheduled source refresh, independent evidence review and reproducible validation; the current dashboard is an inventory, not an autonomous discovery agent.
- [ ] Video understanding, OCR validation, and audio-event perception beyond the existing offline speech-to-text path.
- [ ] RLCD training reproduction on suitable GPU resources.

The implementation adds a post-hoc histogram calibration baseline for evaluation only. AETHER is not RLCD-trained and no model weights are updated. Full reproduction is deferred because the target RTX 2060 has 6 GB VRAM, while the paper reports 48 GB GPU runs.

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
