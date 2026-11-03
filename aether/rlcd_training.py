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
            if index == 0:
                split = "train"
            elif index == 1:
                split = "validation"
            elif index == 2:
                split = "test"
            else:
                bucket = int(sha256(f"{seed}:{row['_aether_example_hash']}:split".encode()).hexdigest()[:8], 16) % 10
                split = "train" if bucket < 8 else "validation" if bucket == 8 else "test"
            splits[split].append(row)
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


@dataclass(frozen=True)
class RLCDExperimentConfig:
    """Exact two-stage OpenJev-RLCD experiment settings from scripts/two_stage.sh."""
    task: str = "mmlupro"
    model_id: str = "Qwen/Qwen3-1.7B"
    seeds: tuple[int, ...] = (17, 23, 29)
    run_root: str = "aether_runs/omega"
    checkpoint_root: str = "aether_checkpoints/omega"
    steps: int = 400
    items_per_step: int = 8
    rationales_per_item: int = 4
    evaluation_rationales: int = 4
    dev_cap: int = 300
    eval_every: int = 400

    def validate(self) -> None:
        if self.task not in {"mmlupro", "gsm8kv"}:
            raise RLCDConfigurationError("Published two-stage plan supports mmlupro or gsm8kv")
        if self.model_id != "Qwen/Qwen3-1.7B":
            raise RLCDConfigurationError(
                "The exact published two-stage run plan is pinned to Qwen/Qwen3-1.7B; "
                "use the generic adapter for other backbones."
            )
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise RLCDConfigurationError("Seeds must be a non-empty unique sequence")
        if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in self.seeds):
            raise RLCDConfigurationError("Seeds must be non-negative integers")
        if self.steps < 1 or self.items_per_step < 1 or self.rationales_per_item < 2:
            raise RLCDConfigurationError("Training steps, items and rationale counts are invalid")
        if self.evaluation_rationales < 1 or self.dev_cap < 1 or self.eval_every < 1:
            raise RLCDConfigurationError("Evaluation settings must be positive")
        for value in (self.run_root, self.checkpoint_root):
            if not value.strip() or Path(value).is_absolute() or ".." in Path(value).parts:
                raise RLCDConfigurationError("Run and checkpoint roots must be safe relative paths")


class RLCDTrainingAdapter:
    UPSTREAM_REPOSITORY = "https://github.com/ZimmyGao/openjev-rlcd.git"
    UPSTREAM_COMMIT = "5859d2b893314fb3581f06e4e36c303e372b3f48"
    RECIPE = {
        "method": "OpenJev-RLCD two-stage calibrate-then-reinforce",
        "stage_1": "PRDO per-rationale proper Brier readout optimization; --diversity-weight 0 --no-score-function; no token KL anchor.",
        "stage_2": "RLCD-RL REINFORCE on each rationale using its own readout's proper Brier score, leave-one-out baseline, sf coefficient 0.3, KL beta 0.04 anchored to stage-1 checkpoint.",
        "reference": "https://arxiv.org/abs/2609.38850",
        "code": "https://github.com/ZimmyGao/openjev-rlcd",
        "upstream_commit": UPSTREAM_COMMIT,
        "controls": ["readout-only continuation", "REINFORCE with no reader", "GRPO continuation"],
        "not_claimed": "This is an adapter for the published open implementation, not a reproduction result until all planned runs and held-out analysis are completed.",
    }

    def __init__(self, project_root: str | Path):
        self.root = Path(project_root).resolve()


    @staticmethod
    def _build_two_stage_runs(config: RLCDExperimentConfig) -> list[dict[str, Any]]:
        config.validate()
        prefix, max_tokens, test_cap = (
            ("gv", 320, 1260) if config.task == "gsm8kv" else ("mp", 256, 1000)
        )
        common = [
            "--task", config.task, "--model", config.model_id,
            "--items-per-step", str(config.items_per_step),
            "--m", str(config.rationales_per_item),
            "--m-eval", str(config.evaluation_rationales),
            "--max-new-tokens", str(max_tokens),
            "--steps", str(config.steps), "--dev-cap", str(config.dev_cap),
            "--test-cap", str(test_cap), "--diversity-weight", "0",
        ]
        runs = []
        for seed in config.seeds:
            stage1_name = f"{prefix}_warm{config.steps}_seed{seed}"
            checkpoint = (Path(config.checkpoint_root) / f"{prefix}_warm{config.steps}_s{seed}.pt").as_posix()
            stage1_output = (Path(config.run_root) / f"{stage1_name}.json").as_posix()
            stage1 = [
                "train", "--objective", "rlcd", *common, "--seed", str(seed),
                "--kl-beta", "0.04", "--eval-every", str(config.eval_every),
                "--no-score-function", "--save-checkpoint", checkpoint,
                "--output", stage1_output,
            ]
            arms = [
                ("fork_rl", ["--sf-coef", "0.3"]),
                ("fork_ro", ["--no-score-function"]),
                ("fork_anchor", ["--sf-coef", "0.3", "--sf-reader", "none"]),
                ("fork_grpo", ["--objective", "grpo"]),
            ]
            stage2 = []
            for arm, flags in arms:
                name = f"{prefix}_{arm}_seed{seed}"
                output = (Path(config.run_root) / f"{name}.json").as_posix()
                args = [
                    "train", *common, "--seed", str(seed + 1000),
                    "--eval-every", str(config.eval_every),
                    "--init-checkpoint", checkpoint, "--kl-beta", "0.04",
                    *flags, "--output", output,
                ]
                stage2.append({"name": name, "argv": args, "output": output, "arm": arm})
            runs.append({
                "seed": seed, "stage1": {"name": stage1_name, "argv": stage1,
                    "checkpoint": checkpoint, "output": stage1_output},
                "stage2": stage2,
            })
        return runs

    def create_two_stage_plan(
        self, config: RLCDExperimentConfig, plan_path: str | Path,
    ) -> dict[str, Any]:
        """Generate the exact published stage-1/stage-2 and fork-control command plan."""
        config.validate()
        plan = {
            "schema": "aether-openjev-rlcd-two-stage-plan-v1",
            "plan_id": sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()[:16],
            "upstream_repository": self.UPSTREAM_REPOSITORY,
            "upstream_commit": self.UPSTREAM_COMMIT,
            "recipe": self.RECIPE,
            "config": asdict(config),
            "model_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
            "dataset": ({
                "repo_id": "TIGER-Lab/MMLU-Pro",
                "revision": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
                "cache_ref_required": True,
            } if config.task == "mmlupro" else {
                "path": "data/gsm8k_verify.json",
                "sha256": sha256((self.root / "external" / "openjev-rlcd" / "data" / "gsm8k_verify.json").read_bytes()).hexdigest()
                    if (self.root / "external" / "openjev-rlcd" / "data" / "gsm8k_verify.json").is_file() else None,
            }),
            "python_module": "openjev_rlcd.train",
            "environment": {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                            "OPENJEV_DEVICE": "cuda", "PYTHONUNBUFFERED": "1",
                            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
            "hardware_requirement": {"minimum_gpu_memory_gib": 48,
                                     "cuda_required": True, "host_ram_gib": "not less than 32; 64 recommended"},
            "runs": self._build_two_stage_runs(config),
            "status": "planned_not_executed",
            "limitations": [
                "A plan is not a trained model or reproduced paper result.",
                "All upstream code, model weights and dataset caches must already be present.",
                "The upstream source commit is pinned and tracked training source must be clean.",
                "Three seeds and four forked stage-2 arms reproduce the published two_stage.sh run structure.",
            ],
        }
        destination = (self.root / plan_path).resolve()
        if self.root not in destination.parents:
            raise RLCDConfigurationError("Plan path must remain inside project root")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        return plan

    def run_two_stage_plan(
        self, plan_path: str | Path, upstream_path: str | Path, *,
        python_executable: str | Path | None = None,
        approved: bool = False, external_compute_confirmed: bool = False,
        timeout_seconds: int = 86400,
    ) -> dict[str, Any]:
        """Execute pinned upstream plan only after verifying source, dependencies and >=48 GiB CUDA."""
        if not approved or not external_compute_confirmed:
            raise PermissionError("RLCD experiment requires explicit approval and external GPU confirmation")
        if timeout_seconds < 1:
            raise RLCDConfigurationError("timeout_seconds must be positive")
        plan_file = (self.root / plan_path).resolve()
        if self.root not in plan_file.parents or not plan_file.is_file():
            raise RLCDConfigurationError("Plan file must exist inside project root")
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
        if plan.get("schema") != "aether-openjev-rlcd-two-stage-plan-v1":
            raise RLCDConfigurationError("Unsupported RLCD plan schema")
        if plan.get("status") != "planned_not_executed":
            raise RLCDConfigurationError("Only a new, unexecuted RLCD plan can run")
        if plan.get("upstream_commit") != self.UPSTREAM_COMMIT:
            raise RLCDConfigurationError("Plan upstream commit does not match the pinned implementation")
        config_data = dict(plan.get("config", {}))
        config_data["seeds"] = tuple(config_data.get("seeds", ()))
        try:
            config = RLCDExperimentConfig(**config_data)
            expected_runs = self._build_two_stage_runs(config)
        except (TypeError, ValueError) as exc:
            raise RLCDConfigurationError(f"Invalid RLCD experiment configuration: {exc}") from exc
        expected_id = sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()[:16]
        if plan.get("plan_id") != expected_id or plan.get("runs") != expected_runs:
            raise RLCDConfigurationError("RLCD plan was modified after creation; regenerate it from configuration")
        upstream = (self.root / upstream_path).resolve()
        if self.root not in upstream.parents or not (upstream / ".git").exists():
            raise RLCDConfigurationError("Pinned OpenJev-RLCD checkout must be a Git repository inside project root")
        python = Path(python_executable).expanduser().resolve() if python_executable else (
            upstream / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        )
        if upstream not in python.parents:
            raise RLCDConfigurationError("Training Python must be inside the pinned upstream checkout")
        if not python.is_file():
            raise RLCDConfigurationError(f"Upstream virtual-environment Python not found: {python}")
        def checked(argv: list[str], *, timeout: int = 30) -> subprocess.CompletedProcess:
            return subprocess.run(argv, cwd=upstream, capture_output=True, text=True,
                                  timeout=timeout, check=False, shell=False)
        revision = checked(["git", "rev-parse", "HEAD"])
        if revision.returncode or revision.stdout.strip() != plan["upstream_commit"]:
            raise RLCDConfigurationError(
                f"OpenJev checkout must be pinned to {plan['upstream_commit']}; found {revision.stdout.strip()}"
            )
        source_status = checked(["git", "status", "--porcelain", "--",
                                 "openjev_rlcd", "scripts", "requirements.txt"])
        if source_status.returncode or source_status.stdout.strip():
            raise RLCDConfigurationError("OpenJev tracked training source is modified; use a clean pinned checkout")
        probe_code = (
            "import json, torch, transformers, datasets, tokenizers, numpy; "
            "p=torch.cuda.is_available(); "
            "m=(torch.cuda.get_device_properties(0).total_memory/(1024**3) if p else 0); "
            "v={'torch':torch.__version__,'torch_cuda':torch.version.cuda,'transformers':transformers.__version__,'datasets':datasets.__version__,'tokenizers':tokenizers.__version__,'numpy':numpy.__version__}; "
            "expected={'torch':'2.6.0','torch_cuda':'12.4','transformers':'5.17.0','datasets':'5.0.1','tokenizers':'0.23.2','numpy':'2.5.2'}; "
            "ok=(v['torch'].startswith(expected['torch']) and all(v[k]==expected[k] for k in expected if k!='torch')); "
            "print(json.dumps({'cuda':p,'gpu':(torch.cuda.get_device_name(0) if p else None),'memory_gib':m,'versions':v,'versions_match':ok}))"
        )
        probe = checked([str(python), "-c", probe_code], timeout=120)
        if probe.returncode:
            raise RLCDConfigurationError(
                "Upstream environment dependencies are incomplete or CUDA probe failed: " + probe.stderr[-3000:]
            )
        try:
            hardware = json.loads(probe.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            raise RLCDConfigurationError("Could not parse upstream CUDA probe") from exc
        if not hardware.get("versions_match"):
            raise RLCDConfigurationError(
                "The upstream environment must match its pinned requirements exactly: "
                + json.dumps(hardware.get("versions", {}), sort_keys=True)
            )
        if not hardware.get("cuda") or float(hardware.get("memory_gib", 0)) < 48:
            raise RLCDConfigurationError(
                f"RLCD paper runs require >=48 GiB CUDA memory; detected {hardware.get('memory_gib', 0):.2f} GiB. "
                "This AETHER host's RTX 2060 6 GB cannot run the published experiment."
            )

        hf_cache = Path(os.environ.get("HF_HUB_CACHE", str(
            Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"
        ))).expanduser()
        model_ref = hf_cache / "models--Qwen--Qwen3-1.7B" / "refs" / "main"
        if not model_ref.is_file() or model_ref.read_text(encoding="utf-8").strip() != plan["model_revision"]:
            raise RLCDConfigurationError(
                "Pinned Qwen3-1.7B cache ref is missing or mismatched. Prepare the exact model revision "
                "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e before running offline."
            )
        if plan["config"]["task"] == "mmlupro":
            dataset_ref = hf_cache / "datasets--TIGER-Lab--MMLU-Pro" / "refs" / "main"
            if not dataset_ref.is_file() or dataset_ref.read_text(encoding="utf-8").strip() != plan["dataset"]["revision"]:
                raise RLCDConfigurationError(
                    "Pinned MMLU-Pro cache ref is missing or mismatched. Prepare revision "
                    "b189ec765aa7ed75c8acfea42df31fdae71f97be before running offline."
                )
        else:
            dataset_path = upstream / plan["dataset"]["path"]
            if not dataset_path.is_file() or sha256(dataset_path.read_bytes()).hexdigest() != plan["dataset"]["sha256"]:
                raise RLCDConfigurationError("Pinned GSM8K-Verify dataset file is missing or has changed")
        for seed_run in plan["runs"]:
            for run in [seed_run["stage1"], *seed_run["stage2"]]:
                for key in ("output", "checkpoint"):
                    if key in run and (upstream / run[key]).exists():
                        raise RLCDConfigurationError(
                            f"Refusing to overwrite existing RLCD artifact: {run[key]}"
                        )
        report_dir = self.root / "reports" / "rlcd" / plan["plan_id"]
        report_dir.mkdir(parents=True, exist_ok=True)
        report = {"plan_id": plan["plan_id"], "upstream_commit": revision.stdout.strip(),
                  "hardware": hardware, "status": "running", "runs": []}
        environment = dict(os.environ)
        environment.update(plan["environment"])
        started = time.time()
        for seed_run in plan["runs"]:
            for run in [seed_run["stage1"], *seed_run["stage2"]]:
                command = [str(python), "-m", "openjev_rlcd.train", *run["argv"]]
                log_path = report_dir / f"{run['name']}.log"
                try:
                    with log_path.open("w", encoding="utf-8") as log_handle:
                        result = subprocess.run(command, cwd=upstream, env=environment,
                            stdout=log_handle, stderr=subprocess.STDOUT, text=True,
                            timeout=timeout_seconds, check=False, shell=False)
                    record = {"name": run["name"], "return_code": result.returncode,
                              "status": "completed" if result.returncode == 0 else "failed",
                              "log": str(log_path.relative_to(self.root))}
                except subprocess.TimeoutExpired:
                    record = {"name": run["name"], "status": "timed_out",
                              "timeout_seconds": timeout_seconds,
                              "log": str(log_path.relative_to(self.root))}
                artifact = upstream / run.get("checkpoint", run["output"])
                if record["status"] == "completed" and not artifact.is_file():
                    record["status"] = "artifact_missing"
                if artifact.is_file():
                    record["artifact"] = str(artifact.relative_to(upstream))
                    record["artifact_sha256"] = sha256(artifact.read_bytes()).hexdigest()
                report["runs"].append(record)
                if record["status"] != "completed":
                    report["status"] = "failed"
                    break
            if report["status"] != "running":
                break
        if report["status"] == "running":
            report["status"] = "completed"
        report["started_at_epoch"], report["finished_at_epoch"] = started, time.time()
        report["note"] = "Execution status records process and artifact completion only. Run the upstream analysis plan and inspect per-item held-out results before claiming reproduction."
        (report_dir / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        plan["status"] = "executed"
        plan["execution_report"] = str((report_dir / "run_report.json").relative_to(self.root))
        plan_file.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        return report

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
