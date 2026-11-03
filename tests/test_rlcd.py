import json

import pytest

from aether.rlcd import RLCDRunManifest, save_manifest


def manifest():
    return RLCDRunManifest(
        method="OpenJev-RLCD",
        method_version="arXiv:2609.38850",
        source_commit="a" * 40,
        base_model="Qwen3-1.7B",
        trained_model_sha256="1" * 64,
        training_dataset_sha256="2" * 64,
        validation_dataset_sha256="3" * 64,
        test_dataset_sha256="4" * 64,
        seeds=(11, 22, 33),
        hardware="External GPU runner",
        baseline_metrics={"brier": 0.25, "ece": 0.12},
        held_out_metrics={"brier": 0.18, "ece": 0.04},
        training_config={"stage1": "proper-scoring-rule", "stage2": "reinforce"},
    )


def test_rlcd_manifest_requires_method_identity_and_separate_data_splits():
    item = manifest()
    item.validate()
    assert len(item.fingerprint()) == 64
    invalid = RLCDRunManifest(**{
        **item.__dict__,
        "method": "generic-reinforcement-learning",
    })
    with pytest.raises(ValueError, match="Unsupported RLCD methodology"):
        invalid.validate()
    same_split = RLCDRunManifest(**{
        **item.__dict__,
        "test_dataset_sha256": item.training_dataset_sha256,
    })
    with pytest.raises(ValueError, match="distinct hashes"):
        same_split.validate()


def test_rlcd_manifest_persists_only_completed_reproducible_runs(tmp_path):
    path = tmp_path / "runs" / "rlcd.json"
    result = save_manifest(manifest(), path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["manifest_sha256"] == result["manifest_sha256"]
    assert saved["trained_model_sha256"] == "1" * 64
    with pytest.raises(ValueError, match="completed"):
        save_manifest(RLCDRunManifest(**{
            **manifest().__dict__,
            "status": "running",
        }), tmp_path / "running.json")
