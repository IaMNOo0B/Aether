# AETHER End-to-End Capability Workstreams

This execution map decomposes the authoritative Stage 0-9 roadmap into 25 implementation workstreams. These are work packages, not 25 invented roadmap phases. Existing code is reused; a workstream is complete only after integration tests and its acceptance evidence pass.

## Delivery rule

Workstreams are developed in dependency-aware batches on the consolidated integration branch. Independent work can be developed concurrently only where interfaces are already stable. Shared database, permission, provider, and runtime contracts are serialized to avoid conflicting changes. Every batch must preserve Windows 10/11 and Python 3.12 compatibility, preserve user data, and report tests truthfully.

## Workstreams

| ID | Workstream | Existing baseline | Remaining completion gate |
|---|---|---|---|
| W01 | Runtime/provider resilience | Ollama provider protocol and registry | Retry/backoff, fallback chain, capability metadata, provider contract tests |
| W02 | Durable task recovery | Persisted tasks and conservative recovery | Full restart matrix, idempotency receipts, migration tests |
| W03 | Configuration evolution | Validated TOML configuration | Versioned schema migrations and rollback tests |
| W04 | Memory retrieval quality | SQLite memory, lexical and optional vector retrieval | Realistic corpus benchmark, precision/recall and regression thresholds |
| W05 | Memory integrity and portability | Correction lineage, backup/restore | Restore fault injection, compatibility and recovery drills |
| W06 | Plan correctness | DAG validation and bounded executor | Held-out multi-step task benchmark and independent objective verification |
| W07 | Research ingestion | Scholarly/public search and bounded full-text ingestion | Robust source adapter tests, freshness and source-quality evaluation |
| W08 | Evidence/review integrity | Review protocol, citation checks, meta-analysis | Authenticated reviewer identity, validated instruments, certainty grading and end-to-end review benchmark |
| W09 | Experiment reproducibility | Preregistration and CSV analysis | Raw-run ingestion, effect/SE validation, leakage and rerun verification |
| W10 | Decision calibration | Decision ledger and held-out evaluator | Independent real-world dataset, adjudication and frozen test evaluation |
| W11 | RLCD research adapter | Feasibility analysis and post-hoc baseline | Faithful two-stage training reproduction on suitable compute; no claim of training until reproduced |
| W12 | Image understanding | Bounded image intake and provider routing | Vision-model integration and local/remote consent tests |
| W13 | Video understanding | Bounded frame sampling and vision calls | Windows FFmpeg integration, timestamps, long-video limits and end-to-end model validation |
| W14 | OCR and audio perception | Tesseract and ONNX audio-event pipeline | Windows dependency/model validation, accuracy fixtures and failure recovery |
| W15 | Goal discovery | Persistent proposals and explicit review | Broader dependency/research/event discovery with deduplication and provenance |
| W16 | Autonomy safety | Fixed tool gateway, approvals, emergency stop | OS-level isolation for untrusted execution and adversarial security suite |
| W17 | Self-engineering | Patch proposal and candidate test workflow | Isolation, static/security scans, resource limits, benchmark and human-reviewed integration |
| W18 | Continual procedural learning | Draft/validated skills and retrieval gate | Automated transfer, retention, forgetting and regression benchmark |
| W19 | Model adaptation | No model-weight training currently | Optional LoRA/fine-tuning experiments, dataset governance, rollback and catastrophic-forgetting evaluation |
| W20 | Specialist orchestration | Researcher/planner/critic roles | Independent model backends, bounded parallelism and evidence-based arbitration |
| W21 | Desktop control | Allowlisted app launch and approved input | Windows UI automation validation, focus/window recovery and action receipts |
| W22 | Voice interaction | Offline push-to-talk and speech output | Wake-word lifecycle, microphone validation, interruption handling and privacy controls |
| W23 | Unified multimodal context | Separate image/media/voice paths | Cross-modal session state, provenance, memory and end-to-end multimodal acceptance |
| W24 | Runtime observability and resource control | Dashboard and preflight CPU/RAM/VRAM checks | Per-process budgets, model-call accounting, telemetry reliability and load tests |
| W25 | Release and operational readiness | Windows CI and setup/launch scripts | Clean Windows install, upgrade/migration, backup/restore, smoke test, docs and release acceptance |

## Dependency-aware execution waves

1. **Core reliability (W01-W06):** provider contracts, persistence, migrations, memory, plan correctness.
2. **Knowledge and evidence (W07-W11):** research, evidence review, experiments, calibration, RLCD adapter.
3. **Perception and autonomy (W12-W16):** vision/media, goal discovery, isolation and security.
4. **Self-improvement and interaction (W17-W23):** self-engineering, continual learning, adaptation experiments, multi-agent, desktop, voice and multimodal integration.
5. **Production hardening (W24-W25):** observability, resource controls, clean install and full release acceptance.

## Hardware and honesty constraints

- Primary target: Windows 10/11, Python 3.12, RTX 2060 6 GB, Ryzen 5 3600, 16 GB RAM.
- Keep the local model path resource-bounded. Heavy training is an optional external-compute experiment, not a promise that the target GPU can perform it.
- Independent specialist roles may be scheduled concurrently only when memory/VRAM budgets allow; otherwise use controlled sequential execution.
- No real-world benchmark passes without independent data, reproducible commands, and recorded artifacts.
- No generated-code execution is called sandboxed until OS-enforced isolation is implemented and adversarially tested.
- AETHER does not claim consciousness, unrestricted general intelligence, or model-weight learning from storing procedural notes.

## Current integration batch

- [x] Create a consolidated integration branch on top of the current Phase 6 stack.
- [x] Add deterministic procedural retrieval benchmark (synthetic Recall@k and draft-skill exclusion).
- [x] Add benchmark tests to OMEGA acceptance group C.
- [ ] Complete Windows CI for the integration PR.
- [ ] Continue implementation across W01-W25 in dependency-aware batches.
