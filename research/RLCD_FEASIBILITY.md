# RLCD feasibility and reproduction boundary

## Reference implementation

Phase 5 investigates OpenJev-RLCD, not a proprietary Jev implementation:
- Paper: https://arxiv.org/abs/2609.38850
- Reference code: https://github.com/ZimmyGao/openjev-rlcd
- The reference repository declares MIT for its code; each dataset/model retains its own license.

The published recipe has two stages:
1. **Readout calibration:** sample on-policy rationales and train the post-rationale answer distribution with a strictly proper score using the pathwise gradient.
2. **Reasoning reinforcement:** initialize from Stage 1 and apply a down-weighted REINFORCE term to the same proper-score reward, with leave-one-out baselines and a KL anchor to the Stage 1 model.

The reference report uses Qwen3-1.7B, multiple seeds, and a single 48 GB GPU per run (RTX 6000 Ada is cited in the upstream run instructions). It downloads datasets and model weights from Hugging Face. Those requirements are materially different from AETHER's target PC.

## Target-machine feasibility

Target machine: RTX 2060 6 GB VRAM, Ryzen 5 3600, 16 GB system RAM.

| Workload | Local feasibility | Honest status |
|---|---|---|
| Decision ledger and Brier/log-loss/ECE/AURC | Yes | Implemented |
| Deterministic held-out evaluation of saved decisions | Yes, when enough independently resolved rows exist | Implemented |
| Post-hoc histogram calibration baseline | Yes | Implemented; not RLCD |
| Small CPU mathematical/estimator unit tests | Yes | Suitable for CI |
| Quantized Qwen3 inference | Potentially, model/context dependent | Not a training reproduction |
| Faithful two-stage Qwen3-1.7B RLCD training | Not supported by the documented 6 GB target | Requires a compatible external GPU environment |
| Paper-level benchmark reproduction | Not established by a local smoke test | Requires licensed datasets, pinned dependencies, multiple seeds and full evaluation |

A rough full-fine-tuning lower-bound calculation for 1.7B parameters illustrates the mismatch: BF16 weights (2 bytes/parameter) ≈ 3.4 GB; BF16 gradients (2 bytes) ≈ 3.4 GB; two FP32 Adam moments (8 bytes) ≈ 13.6 GB; optional FP32 master weights (4 bytes) ≈ 6.8 GB. That is roughly **20.4–27.2 GB before activations, sampled sequences, reference-policy state, logits, temporary buffers and framework overhead**. This is an illustrative optimizer-state estimate, not a measured run. It already exceeds 6 GB by a wide margin.

Do not estimate feasibility from parameter count alone. Training memory includes model weights, gradients, optimizer states, activations, sampled sequences, reference-policy state, and framework overhead. Quantization or LoRA may reduce memory but changes the training setup and must not be described as a faithful reproduction without matching the reference method and validating its outputs.

## Reproduction acceptance gate

A full reproduction may only be marked complete when all of the following are retained as artifacts:
- Exact upstream code commit and dependency lockfile.
- Dataset revisions, licenses, split manifests and SHA-256 hashes.
- Model identifier, revision, tokenizer revision and checkpoint hashes.
- Hardware model, VRAM, CUDA/driver versions and launch configuration.
- Stage 1 and Stage 2 configs, seeds, logs and checkpoints.
- Per-item dev/test distributions, calibration metrics and selective-risk metrics.
- Temperature scaling fitted only on development data.
- Paired multi-seed statistical comparison against the stated baselines.
- A clean rerun from documented commands.

AETHER currently supplies a deterministic held-out post-hoc calibration baseline. It does **not** implement the reference model's two-stage optimizer or update model weights. This document deliberately keeps that boundary explicit.

## Approved next execution path

1. Keep AETHER's decision ledger and post-hoc benchmark as the lightweight local baseline.
2. Use the upstream OpenJev-RLCD implementation in a separate, pinned research environment when a compatible GPU is available.
3. Import only the resulting provenance-rich per-item predictions and evaluation artifacts into AETHER.
4. Compare them against the same frozen held-out set; never train or tune on the final test split.
5. Do not download paid compute, create accounts, accept terms, or incur costs automatically.
