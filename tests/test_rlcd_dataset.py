import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest

from aether.rlcd_dataset import DatasetValidationError, evaluate_dataset, load_dataset


def _write_fixture(tmp_path, *, leak=False):
    rows = []
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for split in ("train", "dev", "test"):
        for index in range(5):
            source = "shared-source" if leak and split == "test" and index == 0 else f"{split}-{index}"
            rows.append({
                "id": f"{split}-{index}",
                "probability": 0.7 if index % 2 else 0.3,
                "outcome": index % 2,
                "domain": "binary-classification",
                "source_ref": source,
                "split": split,
                "prediction_time": (base + timedelta(days=index)).isoformat(),
                "outcome_time": (base + timedelta(days=index + 1)).isoformat(),
            })
    data = tmp_path / "records.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    manifest = {
        "name": "test-set", "version": "1", "owner": "research-team",
        "source_url": "https://example.org/dataset", "license": "CC-BY-4.0",
        "collection_start": base.isoformat(), "collection_end": (base + timedelta(days=10)).isoformat(),
        "inclusion_criteria": "All eligible examples", "exclusion_criteria": "Missing outcomes",
        "sampling_procedure": "Consecutive sample", "label_definition": "Observed binary outcome",
        "adjudicators": ["reviewer-a", "reviewer-b"],
        "prediction_precedes_outcome": True, "partitions_are_disjoint": True,
        "dataset_sha256": digest,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return data, manifest_path


def test_loads_provenance_dataset_and_evaluates_frozen_splits(tmp_path):
    data, manifest = _write_fixture(tmp_path)
    rows, metadata, digest = load_dataset(data, manifest)
    report = evaluate_dataset(rows)
    assert len(rows) == 15
    assert metadata["name"] == "test-set"
    assert report["domains"]["binary-classification"]["splits"]["test"]["count"] == 5
    assert len(digest) == 64


def test_rejects_cross_split_source_leakage(tmp_path):
    data, manifest = _write_fixture(tmp_path, leak=True)
    with pytest.raises(DatasetValidationError, match="leakage"):
        load_dataset(data, manifest)


def test_rejects_manifest_hash_mismatch(tmp_path):
    data, manifest = _write_fixture(tmp_path)
    metadata = json.loads(manifest.read_text())
    metadata["dataset_sha256"] = "0" * 64
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(DatasetValidationError, match="SHA-256"):
        load_dataset(data, manifest)


def test_rejects_prediction_after_outcome(tmp_path):
    data, manifest = _write_fixture(tmp_path)
    rows = [json.loads(line) for line in data.read_text().splitlines()]
    rows[0]["prediction_time"] = "2026-01-10T00:00:00+00:00"
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    metadata = json.loads(manifest.read_text())
    metadata["dataset_sha256"] = hashlib.sha256(data.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(DatasetValidationError, match="precede"):
        load_dataset(data, manifest)
