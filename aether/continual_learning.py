"""External-adapter interface for continual learning with forgetting guards.

AETHER does not contain a local fine-tuning backend. A caller supplies a trainer
for a compatible external runtime and an evaluator over held-out task suites.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Protocol
import json
import math
import os
import tempfile


class LoRATrainer(Protocol):
    trainer_id: str

    def train(
        self, base_model: str, training_data: Path,
        output_directory: Path, config: dict[str, Any],
    ) -> Path: ...


@dataclass(frozen=True)
class ContinualLearningResult:
    run_id: str
    status: str
    base_model: str
    adapter_path: str
    adapter_sha256: str
    training_data_sha256: str
    evaluation_data_sha256: dict[str, str]
    scores_before: dict[str, float]
    scores_after: dict[str, float]
    forgetting_by_task: dict[str, float]
    maximum_forgetting: float
    new_task_score: float
    seed: int
    trainer_id: str
    config: dict[str, Any]
    limitations: tuple[str, ...]


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_tree(root: Path) -> str:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Trainer must return a regular adapter directory")
    digest = sha256()
    total = 0
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError("Trainer returned an empty adapter directory")
    for path in files:
        if path.is_symlink():
            raise ValueError("Adapter artifacts cannot contain symlinks")
        total += path.stat().st_size
        if total > 10_000_000_000:
            raise ValueError("Adapter artifacts exceed the 10 GB limit")
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(bytes.fromhex(_hash_file(path)))
    return digest.hexdigest()


class ContinualLearningManager:
    def __init__(self, artifact_root: str | Path):
        self.artifact_root = Path(artifact_root).resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)

    def train_adapter(
        self,
        *,
        base_model: str,
        training_data: str | Path,
        evaluation_suites: dict[str, str | Path],
        trainer: LoRATrainer,
        evaluator: Callable[[str, Path], float],
        config: dict[str, Any],
        seed: int,
        approved: bool = False,
        minimum_new_task_score: float = 0.8,
        maximum_forgetting: float = 0.05,
    ) -> ContinualLearningResult:
        if not approved:
            raise PermissionError("External model-weight training requires explicit approval")
        if not base_model.strip() or not getattr(trainer, "trainer_id", "").strip():
            raise ValueError("Base model and trainer identity are required")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("Training seed must be a non-negative integer")
        if not 0 <= minimum_new_task_score <= 1 or not 0 <= maximum_forgetting <= 1:
            raise ValueError("Learning thresholds must be between zero and one")
        if "new_task" not in evaluation_suites or len(evaluation_suites) < 2:
            raise ValueError("Evaluation requires a new_task suite and at least one retained task suite")
        training_path = Path(training_data).resolve()
        if not training_path.is_file() or training_path.is_symlink() or training_path.stat().st_size > 1_000_000_000:
            raise ValueError("Training data must be a regular file no larger than 1 GB")
        suite_paths = {name: Path(path).resolve() for name, path in evaluation_suites.items()}
        if any(not path.is_file() or path.is_symlink() for path in suite_paths.values()):
            raise ValueError("Every evaluation suite must be a regular file")
        training_hash = _hash_file(training_path)
        evaluation_hashes = {name: _hash_file(path) for name, path in suite_paths.items()}
        if training_hash in evaluation_hashes.values():
            raise ValueError("Training data must not be reused as a held-out evaluation suite")

        run_id = sha256(
            f"{base_model}|{training_hash}|{seed}|{trainer.trainer_id}".encode("utf-8")
        ).hexdigest()[:24]
        run_root = self.artifact_root / run_id
        if run_root.exists():
            raise FileExistsError("This deterministic training run already exists")
        run_root.mkdir(parents=True)
        output_directory = run_root / "adapter"
        output_directory.mkdir()
        candidate_path = Path(
            trainer.train(base_model, training_path, output_directory, dict(config))
        ).resolve()
        if output_directory not in candidate_path.parents and candidate_path != output_directory:
            raise ValueError("Trainer returned an artifact outside the run output directory")
        adapter_hash = _hash_tree(candidate_path)

        scores_before: dict[str, float] = {}
        scores_after: dict[str, float] = {}
        for name, suite in suite_paths.items():
            before = evaluator(base_model, suite) if name != "new_task" else 0.0
            after = evaluator(str(candidate_path), suite)
            for label, score in (("baseline", before), ("candidate", after)):
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)) or not 0 <= score <= 1:
                    raise ValueError(f"{label} evaluator score for {name} must be finite and in [0, 1]")
            if name != "new_task":
                scores_before[name] = float(before)
            scores_after[name] = float(after)

        forgetting = {
            name: max(0.0, scores_before[name] - scores_after[name])
            for name in scores_before
        }
        max_forgetting = max(forgetting.values(), default=0.0)
        new_score = scores_after["new_task"]
        status = (
            "accepted" if new_score >= minimum_new_task_score
            and max_forgetting <= maximum_forgetting else "rejected_regression"
        )
        result = ContinualLearningResult(
            run_id=run_id, status=status, base_model=base_model,
            adapter_path=str(candidate_path), adapter_sha256=adapter_hash,
            training_data_sha256=training_hash, evaluation_data_sha256=evaluation_hashes,
            scores_before=scores_before, scores_after=scores_after,
            forgetting_by_task=forgetting, maximum_forgetting=max_forgetting,
            new_task_score=new_score, seed=seed, trainer_id=trainer.trainer_id,
            config=dict(config),
            limitations=(
                "The caller-supplied trainer defines the actual fine-tuning algorithm and runtime.",
                "Scores are only as valid as the evaluator and held-out datasets supplied by the caller.",
                "AETHER does not merge or activate this adapter automatically.",
            ),
        )
        manifest = asdict(result)
        destination = run_root / "manifest.json"
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=run_root, prefix="manifest.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
        return result
