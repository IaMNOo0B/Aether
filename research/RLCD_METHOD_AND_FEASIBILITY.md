# RLCD methodology and AETHER feasibility

**Status:** research completed; training not implemented or claimed.

## 1. Meaning

In the AETHER master directive, RLCD means **Reinforcement Learning for Calibrated Decisions**. It is not a synonym for generic reinforcement learning, temperature scaling, or logging a model's self-reported confidence.

## 2. What is publicly reproducible

The primary open implementation inspected for this feasibility review is:

- Zhimin Gao and Pichao Wang, *OpenJev-RLCD: A Working RLCD Implementation*, arXiv:2609.38850 (2026): https://arxiv.org/abs/2609.38850
- Code and per-item results: https://github.com/ZimmyGao/openjev-rlcd

This is an independent open implementation. It explicitly makes no claim to reproduce TypeSafe/Jev's undisclosed internal training algorithm.

The inspected implementation describes a two-stage recipe:

1. **Calibrate the decision readout.** Sample reasoning rationales. After each rationale, read the model's explicit probability distribution over the allowed answer labels. Optimize that distribution with a strictly proper scoring rule using the pathwise gradient.
2. **Reinforce the reasoning.** Initialize from stage 1, then add a REINFORCE score-function term over sampled rationales. The paper's reported recipe uses a score-function coefficient of 0.3, leave-one-out baselines, and a KL anchor (beta = 0.04) to the stage-1 model.

The code distinguishes:
- A per-rationale objective, which rewards each rationale's own answer distribution.
- A mixture objective, which scores the average distribution across rationales and can reward disagreement between rationales.
- An interpolation parameter between those objectives.

The implementation evaluates GSM8K-Verify, MMLU-Pro and ChaosNLI, and reports proper-score and selective-prediction metrics. Its README reports Qwen3-1.7B experiments across three seeds.

## 3. Important limitations of the evidence

- The OpenJev repository is a research implementation, not proof that the proprietary Jev method has been replicated.
- Reported experiments use a 48 GB GPU (the repository identifies an RTX 6000 Ada setup), and model/data dependencies are downloaded from Hugging Face.
- Results are reported by the implementation authors; AETHER has not independently reproduced them.
- Calibration is distribution- and task-dependent. A calibrated probability on one task is not a universal confidence guarantee.
- Calibration metrics require real outcome labels. A model's verbal confidence score is not a substitute.

## 4. AETHER integration constraints

AETHER currently calls models through chat-completion interfaces (Ollama or an OpenAI-compatible API). Those interfaces do not expose the full differentiable token-level training path required by the OpenJev RLCD training loop.

The RTX 2060 6 GB development machine is not the hardware configuration used by the published OpenJev runs. AETHER must not start a heavyweight training run on this machine or claim RLCD training occurred.

## 5. Engineering decision

Until an external training environment is explicitly configured:

- Keep outcome logging and calibration evaluation separate from RLCD training.
- Do not treat calibration metrics alone as RLCD.
- Preserve the RLCD training integration as an optional external-compute capability.
- Require a reproducible run manifest, model/checkpoint identity, dataset versions, seeds, baseline, held-out evaluation and resource record before marking RLCD as implemented or verified.

## 6. Current AETHER status

| Capability | Status |
|---|---|
| Record probability predictions and resolved outcomes in SQLite | Implemented |
| Compute Brier score, log loss, ECE and selective-risk metrics | Implemented |
| Open-source RLCD methodology review | Research completed |
| Reproduce OpenJev training on AETHER hardware | Not feasible on the current 6 GB GPU configuration |
| External RLCD training adapter and reproducible run orchestration | Not implemented |
| Held-out AETHER RLCD training evaluation | Not implemented |

## References

- OpenJev-RLCD paper: https://arxiv.org/abs/2609.38850
- OpenJev-RLCD code, training recipe, tests and results: https://github.com/ZimmyGao/openjev-rlcd
- Related background on calibrated uncertainty in model-based RL (not the same RLCD method): https://arxiv.org/abs/1906.08312
