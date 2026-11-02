# Phase 5 real-world calibration benchmark protocol

This directory intentionally contains no fabricated benchmark observations. A dataset is eligible only after an operator supplies its provenance and independent outcome labels.

## Required dataset record

JSONL rows must contain:
- `id`: stable unique record identifier.
- `probability`: probability recorded before the outcome was known, in [0, 1].
- `outcome`: independently observed binary outcome (0 or 1).
- `domain`: one task domain; do not pool incomparable domains.
- `source_ref`: source record or experiment receipt.
- `prediction_time` and `outcome_time`: timezone-aware timestamps proving temporal order.

## Required dataset manifest

Record the following before running the benchmark:
- Dataset name/version, publisher/owner, canonical source URL and license.
- Collection window, inclusion/exclusion criteria and sampling procedure.
- Label definition, who adjudicated labels, adjudication procedure and inter-rater agreement where applicable.
- Number of records, missingness, class balance and known sampling biases.
- Dataset SHA-256, exact preprocessing code revision and environment lockfile.
- Confirmation that probability values were written before outcomes were observed.
- Confirmation that records do not overlap between train, calibration/dev and final test partitions.

## Evaluation

1. Freeze the protocol and manifest before evaluating.
2. Group by domain. Use separate datasets for materially different prediction tasks.
3. Split by time or independent entity where repeated observations could leak across partitions.
4. Fit any calibrator only on training/calibration data. Do not tune on final test outcomes.
5. Report Brier score, log loss, ECE with bin counts, AURC and coverage at fixed risk budgets, plus sample counts and confidence intervals.
6. Include uncalibrated baseline, post-hoc baseline and any trained candidate.
7. Run paired bootstrap or an appropriate paired test on identical held-out records.
8. Publish all per-item predictions, code, seeds, hashes and failure cases.

## Current status

AETHER's `aether.rlcd_evaluation` validates binary records and computes deterministic held-out metrics, but the current decision ledger does not yet enforce all manifest fields or independently authenticate label adjudicators. Until a real dataset and manifest pass the protocol, the Phase 5 real-world benchmark remains **pending**. Synthetic tests are software tests, not real-world evidence.
