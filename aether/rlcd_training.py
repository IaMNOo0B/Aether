"""Reproducible external-compute adapter for the OpenJev-RLCD research recipe."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import json, os, subprocess, time


class RLCDConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class RLCDRunConfig:
    model_id: str
    dataset_path: str
    output_dir: str
    seed: int = 17
    stage1_epochs: int = 1
    stage2_epochs: int = 1
    rationale_samples: int = 4
    score_function_coefficient: float = 0.3
    kl_anchor_beta: float = 0.04
    interpolation: float = 0.5
    training_device: str = "cuda"
    min_gpu_memory_gb: float = 48.0

    def validate(self) -> None:
        if not self.model_id.strip() or not self.dataset_path.strip() or not self.output_dir.strip():
            raise RLCDConfigurationError("Model, dataset and output paths are required")
        if self.seed < 0 or self.stage1_epochs < 1 or self.stage2_epochs < 1:
            raise RLCDConfigurationError("Seed and epoch settings are invalid")
        if not 1 <= self.rationale_samples <= 64:
            raise RLCDConfigurationError("rationale_samples must be between 1 and 64")
        if not 0 <= self.score_function_coefficient <= 2:
            raise RLCDConfigurationError("score_function_coefficient must be in [0, 2]")
        if not 0 <= self.kl_anchor_beta <= 1 or not 0 <= self.interpolation <= 1:
            raise RLCDConfigurationError("KL beta and interpolation must be in [0, 1]")
        if self.min_gpu_memory_gb < 1:
            raise RLCDConfigurationError("min_gpu_memory_gb must be positive")


def _prompt_key(record: Mapping[str, Any]) -> str:
    prompt = " ".join(str(record.get("prompt", "")).casefold().split())
    if not prompt:
        raise RLCDConfigurationError("Every RLCD row requires a non-empty prompt")
    return sha256(prompt.encode("utf-8")).hexdigest()


def prepare_dataset(records: Iterable[Mapping[str, Any]], output_path: str | Path,
                    *, seed: int = 17) -> dict[str, Any]:
    """Deduplicate by prompt and deterministically stratify train/validation/test."""
    if seed < 0:
        raise RLCDConfigurationError("seed must be non-negative")
    unique: dict[str, dict[str, Any]] = {}
    duplicates = 0
    for raw in records:
        if not isinstance(raw, Mapping):
            raise RLCDConfigurationError("Dataset rows must be objects")
        row = dict(raw)
        key = _prompt_key(row)
        label = str(row.get("label", row.get("answer", ""))).strip()
        if not label:
            raise RLCDConfigurationError("Every RLCD row requires a non-empty label/answer")
        if key in unique:
            prior = str(unique[key].get("label", unique[key].get("answer", ""))).strip()
            if prior != label:
                raise RLCDConfigurationError("Conflicting labels for the same prompt; resolve before splitting")
            duplicates += 1
            continue
        unique[key] = {**row, "_aether_example_hash": key}
    if len(unique) < 10:
        raise RLCDConfigurationError("At least 10 unique examples are required")
    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for key, row in unique.items():
        grouped.setdefault(str(row.get("label", row.get("answer", ""))).strip(), []).append((key, row))
    splits = {"train": [], "validation": [], "test": []}
    for label, members in sorted(grouped.items()):
        if len(members) < 3:
            raise RLCDConfigurationError(f"Label {label!r} needs at least three examples")
        ordered = sorted(members, key=lambda pair: sha256(f"{seed}:{pair[0]}".encode()).hexdigest())
        for index, (_, row) in enumerate(ordered):
            bucket = index % 10
            splits["train" if bucket < 8 else "validation" if bucket == 8 else "test"].append(row)
    if any(not rows for rows in splits.values()):
        raise RLCDConfigurationError("Stratified split produced an empty partition")
    payload = {
        "schema": "aether-rlcd-dataset-v1", "seed": seed,
        "source_count": len(unique) + duplicates, "unique_count": len(unique),
        "duplicate_count": duplicates, "splits": splits,
        "split_counts": {key: len(value) for key, value in splits.items()},
        "label_counts": {key: len(value) for key, value in sorted(grouped.items())},
        "test_hashes": sorted(row["_aether_example_hash"] for row in splits["test"]),
        "warning": "Prompt-level deduplication and label stratification reduce leakage; temporal, subject and benchmark-specific leakage controls remain the dataset owner's responsibility.",
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, ensure_ascii=False).encode()
    destination.write_bytes(encoded)
    return {key: value for key, value in payload.items() if key != "splits"} | {
        "path": str(destination), "sha256": sha256(encoded).hexdigest()
    }


class RLCDTrainingAdapter:
    RECIPE = {
        "method": "OpenJev-RLCD two-stage research implementation",
        "stage_1": "Optimize rationale-conditioned answer distributions with a proper scoring rule.",
        "stage_2": "REINFORCE rationale sampling with leave-one-out baseline and KL anchor.",
        "reference": "https://arxiv.org/abs/2609.38850",
        "code": "https://github.com/ZimmyGao/openjev-rlcd",
        "not_claimed": "No claim of reproducing proprietary Jev internals or paper results.",
    }

    def __init__(self, project_root: str | Path):
        self.root = Path(project_root).resolve()

    def create_manifest(self, config: RLCDRunConfig, manifest_path: str | Path) -> dict[str, Any]:
        config.validate()
        dataset = (self.root / config.dataset_path).resolve()
        if self.root not in dataset.parents or not dataset.is_file():
            raise RLCDConfigurationError("Dataset must be an existing project file")
        output = (self.root / config.output_dir).resolve()
        if self.root not in output.parents:
            raise RLCDConfigurationError("Output directory must remain inside project root")
        data = json.loads(dataset.read_text(encoding="utf-8"))
        if data.get("schema") != "aether-rlcd-dataset-v1" or not data.get("splits", {}).get("test"):
            raise RLCDConfigurationError("A prepared dataset with held-out test split is required")
        manifest = {
            "schema": "aether-rlcd-run-v1", "recipe": self.RECIPE, "config": asdict(config),
            "dataset": {"path": str(dataset.relative_to(self.root)),
                        "sha256": sha256(dataset.read_bytes()).hexdigest(),
                        "split_counts": data["split_counts"], "test_hashes": data["test_hashes"]},
            "created_at_epoch": time.time(), "status": "prepared_not_executed",
            "baseline_required": True, "held_out_evaluation_required": True, "run_result": None,
        }
        destination = (self.root / manifest_path).resolve()
        if self.root not in destination.parents:
            raise RLCDConfigurationError("Manifest must remain inside project root")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    def run(self, manifest_path: str | Path, runner_argv: Sequence[str], *,
            approved: bool = False, external_compute_confirmed: bool = False,
            timeout_seconds: int = 86400) -> dict[str, Any]:
        if not approved or not external_compute_confirmed:
            raise PermissionError("RLCD training requires explicit approval and a provisioned external GPU")
        if not runner_argv or any(not isinstance(x, str) or not x for x in runner_argv):
            raise RLCDConfigurationError("runner_argv must be a non-empty argument list")
        if timeout_seconds < 1:
            raise RLCDConfigurationError("timeout_seconds must be positive")
        manifest = (self.root / manifest_path).resolve()
        if self.root not in manifest.parents or not manifest.is_file():
            raise RLCDConfigurationError("Manifest must exist inside project root")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("schema") != "aether-rlcd-run-v1" or payload.get("status") != "prepared_not_executed":
            raise RLCDConfigurationError("Only a new prepared manifest can be executed")
        dataset = (self.root / payload["dataset"]["path"]).resolve()
        if not dataset.is_file() or sha256(dataset.read_bytes()).hexdigest() != payload["dataset"]["sha256"]:
            raise RLCDConfigurationError("Dataset changed after manifest creation")
        output = (self.root / payload["config"]["output_dir"]).resolve()
        if self.root not in output.parents:
            raise RLCDConfigurationError("Output directory escapes project root")
        output.mkdir(parents=True, exist_ok=True)
        argv = [part.replace("{manifest}", str(manifest)).replace("{output}", str(output))
                for part in runner_argv]
        if any("{" in part or "}" in part for part in argv):
            raise RLCDConfigurationError("Unsupported runner placeholder")
        env = dict(os.environ)
        env["AETHER_RLCD_MANIFEST"] = str(manifest)
        started = time.time()
        try:
            result = subprocess.run(argv, cwd=self.root, env=env, capture_output=True,
                                    text=True, timeout=timeout_seconds, check=False, shell=False)
            status = "runner_completed" if result.returncode == 0 else "runner_failed"
            details = {"return_code": result.returncode, "stdout": result.stdout[-12000:],
                       "stderr": result.stderr[-8000:]}
        except subprocess.TimeoutExpired as exc:
            status = "runner_timed_out"
            details = {"timeout_seconds": timeout_seconds,
                       "stdout": str(exc.stdout or "")[-12000:],
                       "stderr": str(exc.stderr or "")[-8000:]}
        payload.update({"status": status, "started_at_epoch": started,
                        "finished_at_epoch": time.time(), "runner": argv, "run_result": details,
                        "note": "Successful process exit is not proof of RLCD reproduction; inspect checkpoint, baseline and held-out metrics."})
        manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
