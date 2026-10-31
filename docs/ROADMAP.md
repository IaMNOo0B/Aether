# AETHER Engineering Roadmap

This roadmap translates the PROJECT AETHER — OMEGA directive into testable delivery stages. A folder or interface is not evidence that a capability works.

## Stage 0 — Repository and environment baseline

- [x] Initialize an empty repository with a Windows-first Python 3.12 foundation.
- [x] Add double-click setup and launch scripts.
- [x] Add standard-library runtime configuration and SQLite persistence.
- [x] Add an Ollama adapter that does not download models automatically.
- [x] Add a Windows CI test workflow.
- [x] Confirm green GitHub Actions runs on the repository's Windows test workflow.
- [ ] Perform a clean Windows 10/11 install and launch test on the user's PC.

## Stage 1 — Cognitive foundation

- [x] Provider protocol and explicit provider registry (Ollama adapter registered).
- [x] Structured task lifecycle with durable queued/running/completed/failed/cancelled states.
- [x] Deterministic character-budgeted context construction and oldest-history trimming.
- [x] Centralized structured application logging and JSON diagnostics.
- [x] Runtime configuration validation for supported provider, URL, limits, and log level.
- [ ] Provider health policy with bounded retries, backoff, and explicit fallback.
- [ ] Configuration schema versioning and migration support.
- [ ] Provider/model capability metadata and model selection policy.
- [ ] End-to-end task recovery after process interruption.

## Stage 2 — Reliable memory

- [x] Persistent conversation history.
- [x] Persistent user memory records with source and verification status.
- [x] Basic lexical retrieval.
- [ ] Hybrid keyword/vector retrieval with an offline-capable embedding option.
- [ ] Memory correction, contradiction tracking, and version history.
- [ ] Explicit memory export, backup, restore, and retention controls.
- [ ] Retrieval relevance benchmark.

## Stage 3 — Planning and verification

- [x] Deterministic dependency graph, cycle detection, ready-step frontier, and persisted plan progress.\n- [ ] LLM-assisted task decomposition with user-reviewable plans.
- [ ] Plan/action/observation/verification state machine.
- [ ] Independent validators for tool results.
- [ ] Failure taxonomy and resumable task recovery.
- [ ] Held-out multi-step task benchmark.

## Stage 4 — Research and knowledge

- [ ] Source ingestion with provenance.
- [ ] Citation-aware research reports.
- [ ] Entity and relationship store.
- [ ] Hypothesis and experiment records.
- [ ] Reproducibility metadata and evidence grading.

## Stage 5 — RLCD investigation

- [x] Identify the public method and primary paper: OpenJev-RLCD, arXiv:2609.38850.
- [x] Review its two-stage calibrate-then-reinforce recipe, proper scoring rules, KL anchor, and documented failure modes.
- [x] Record reproduction constraints and licensing considerations in [RLCD feasibility](research/RLCD_FEASIBILITY.md).
- [ ] Build typed decision-outcome logging and a held-out calibration evaluation harness.
- [ ] Reproduce training only on suitable GPU hardware, after dataset/model license review.

The paper's experimental runs use a 48 GB GPU. The target RTX 2060 has 6 GB VRAM, so full reproduction is deferred. Do not claim AETHER is RLCD-trained or substitute an unrelated confidence heuristic.

## Stage 6 — Bounded autonomy

- [ ] Permission tiers and a policy-enforced action gateway.
- [ ] Read-only tools first.
- [ ] Explicit approval for destructive, external, financial, or irreversible operations.
- [ ] Resource budgets, deadlines, cancellation, and emergency stop.
- [ ] Auditable action receipts.

## Stage 7 — Self-engineering

- [ ] Read-only architecture inspection.
- [ ] Isolated patch proposals.
- [ ] Test, security scan, benchmark, and rollback pipeline.
- [ ] Human approval before production integration.
- [ ] Immutable audit trail for all accepted changes.

## Stage 8 — Learning and multimodal expansion

Investigate continual learning, transfer, meta-learning, vision, audio, and specialist agents independently. Integrate only where a reproducible evaluation demonstrates a benefit on the target hardware or an explicitly approved cloud budget.

## Engineering rules

1. Preserve existing user data.
2. Prefer root-cause fixes over symptom patches.
3. Never report an unexecuted test as passing.
4. Separate retrieval, procedural learning, policy adaptation, and model-weight updates.
5. Do not claim consciousness or general intelligence based on architecture labels.
6. Keep Windows 10/11 as the primary supported platform.
