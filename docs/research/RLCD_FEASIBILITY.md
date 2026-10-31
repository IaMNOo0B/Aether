# RLCD research and AETHER feasibility note

**Status:** Method identified; reproduction deferred.  
**Reviewed:** 2026-11-01

## Finding

The master directive's phrase "Reinforcement Learning for Calibrated Decisions (RLCD)" now maps to a public, reproducible method described in Zhimin Gao and Pichao Wang, *OpenJev-RLCD: A Working RLCD Implementation*, arXiv:2609.38850 (2026).

This is an independent open implementation. It does not disclose or claim to reproduce TypeSafe AI's proprietary Jev training recipe.

Primary sources:
- Paper: https://arxiv.org/abs/2609.38850
- Implementation: https://github.com/ZimmyGao/openjev-rlcd

## Method, in implementation terms

The model samples one or more rationales for a typed decision. After each rationale, the model emits an `Answer:` marker. The method reads the model's next-token distribution over the allowed answer options and scores that distribution with a strictly proper scoring rule (the paper evaluates Brier and log scores).

Training is deliberately two-stage:

1. **Calibrate the decision readout.** Train the post-rationale answer distribution using the proper score and the pathwise gradient. The rationale is sampled on-policy but is not reinforced in this stage.
2. **Reinforce reasoning.** Start from the stage-one checkpoint. Use the same proper score as the reward for rationale policy updates, with a leave-one-out baseline and a KL anchor to the calibrated checkpoint. The reported implementation uses score-function coefficient 0.3 and KL coefficient 0.04.

The paper identifies two failure modes that AETHER must not ignore:
- Without a KL anchor, optimization can collapse reasoning into an empty rationale.
- The policy-gradient term can dominate the lower-variance readout gradient; the paper reports a 7–12x gradient-norm difference in its setup.

A group-mixture objective can reward disagreement between rationales even when that disagreement is not grounded. The paper therefore uses a per-rationale proper-score objective for its main recipe.

## Experimental scope and hardware

The paper evaluates Qwen3-1.7B on MMLU-Pro, GSM8K-Verify, and ChaosNLI. Its reported setup uses AdamW, learning rate 2e-6, eight questions per step, four rationales per question, 400 steps for stage one, and 400 further steps for stage two. It reports results across three seeds.

The authors state that experiments use one 48 GB GPU per run (RTX 6000 Ada in the paper). AETHER's target RTX 2060 has 6 GB VRAM. Therefore, reproducing the published training runs on the target machine is not a realistic current milestone. Do not attempt full RLCD fine-tuning on that machine or silently substitute a different algorithm.

The implementation repository is MIT licensed, but the benchmark datasets retain their own licenses. Any future reproduction must audit each dataset and model license separately.

## AETHER engineering decision

- Do not claim AETHER is RLCD-trained.
- Do not import the training stack into the default Windows runtime.
- Keep research reproduction isolated from the production assistant.
- First build decision-outcome logging and a calibration evaluation harness (Brier score, log loss, ECE, risk-coverage/AURC, and held-out evaluation) for clearly typed decisions with verifiable outcomes.
- Only consider training after securing suitable GPU resources, exact dataset/model licenses, a baseline, fixed seeds, reproducible scripts, and held-out tests.
- Treat confidence as a measured property of a decision model, not a free-form LLM self-rating.

## References

- Gao, Z. and Wang, P. (2026). *OpenJev-RLCD: A Working RLCD Implementation*. arXiv:2609.38850.
- The authors' implementation and reproduction instructions: https://github.com/ZimmyGao/openjev-rlcd
- Damani et al. (ICLR 2026), *Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty* (RLCR). This is a related but distinct calibration-reward approach, not interchangeable with the OpenJev-RLCD recipe.
