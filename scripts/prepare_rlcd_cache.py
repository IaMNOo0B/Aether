"""Prepare the exact Hugging Face cache revisions required by AETHER's pinned RLCD plan.

Run with the Python environment in external/openjev-rlcd. This downloads several GB
for Qwen3-1.7B and, for MMLU-Pro, the dataset. It changes only the local HF cache
refs/main pointer for these repositories so upstream code using the default
revision resolves to the pinned snapshot. Run only on the external training host.
"""
from __future__ import annotations

import argparse
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DATASET_ID = "TIGER-Lab/MMLU-Pro"
DATASET_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"


def pin_main_ref(snapshot_path: str | Path, revision: str) -> Path:
    snapshot = Path(snapshot_path).resolve()
    if snapshot.name != revision or snapshot.parent.name != "snapshots":
        raise ValueError("snapshot_download did not return the requested immutable commit snapshot")
    repository_cache = snapshot.parent.parent
    refs = repository_cache / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    destination = refs / "main"
    temporary = refs / "main.aether-tmp"
    temporary.write_text(revision + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("mmlupro", "gsm8kv"), required=True)
    parser.add_argument("--confirm-download", action="store_true")
    args = parser.parse_args()
    if not args.confirm_download:
        parser.error("This operation downloads model/dataset files; pass --confirm-download explicitly.")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install the pinned upstream requirements before preparing the cache.") from exc

    model_snapshot = snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION)
    model_ref = pin_main_ref(model_snapshot, MODEL_REVISION)
    print(f"Pinned {MODEL_ID} main cache ref to {MODEL_REVISION}: {model_ref}")
    if args.task == "mmlupro":
        dataset_snapshot = snapshot_download(
            repo_id=DATASET_ID, repo_type="dataset", revision=DATASET_REVISION
        )
        dataset_ref = pin_main_ref(dataset_snapshot, DATASET_REVISION)
        print(f"Pinned {DATASET_ID} main cache ref to {DATASET_REVISION}: {dataset_ref}")
    else:
        print("GSM8K-Verify uses the constructed data/gsm8k_verify.json shipped in the pinned source checkout.")
    print("Cache is now pinned for offline training. Verify available disk space and GPU memory before starting runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
