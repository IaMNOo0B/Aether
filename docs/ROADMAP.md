# AETHER Engineering Roadmap

This roadmap translates the PROJECT AETHER — OMEGA directive into testable delivery stages. A folder or interface is not evidence that a capability works.

## Stage 0 — Repository and environment baseline

- [x] Initialize an empty repository with a Windows-first Python 3.12 foundation.
- [x] Add double-click setup and launch scripts.
- [x] Add standard-library runtime configuration and SQLite persistence.
- [x] Add an Ollama adapter that does not download models automatically.
- [x] Add a Windows CI test workflow.
- [ ] Confirm first green CI run.
- [ ] Perform a clean Windows 10/11 install and launch test.

## Stage 1 — Cognitive foundation

- [ ] Provider protocol and model registry.
- [ ] Robust provider health checks, timeouts, retries, and fallback policy.
- [ ] Structured task lifecycle with durable checkpoints.
- [ ] Context budgeting and deterministic history trimming.
- [ ] Structured logs and diagnostic report.
- [ ] Configuration schema validation and migration support.

## Stage 2 — Reliable memory

- [x] Persistent conversation history.
- [x] Persistent user memory records with source and verification status.
- [x] Basic lexical retrieval.
- [ ] Hybrid keyword/vector retrieval with an offline-capable embedding option.
- [ ] Memory correction, contradiction tracking, and version history.
- [ ] Explicit memory export, backup, restore, and retention controls.
- [ ] Retrieval relevance benchmark.

## Stage 3 — Planning and verification

- [ ] Task decomposition and dependency graph.
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

The master directive uses the term "Reinforcement Learning for Calibrated Decisions" but does not identify a unique paper or formal method. Do not implement a guessed algorithm. First identify the intended primary source and verify its equations, training protocol, datasets, benchmarks, and limitations. If the acronym is project-defined rather than an established named method, document that explicitly.

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
