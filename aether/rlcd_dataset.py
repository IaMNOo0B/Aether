"""Validate provenance-rich real-world calibration datasets and evaluate frozen splits."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from aether.calibration import evaluate_binary_decisions
from aether.rlcd_evaluation import DecisionExample, HistogramCalibrator, _metric_dict

MAX_DATASET_BYTES = 50 * 1024 * 1024
MAX_ROWS = 1_000_000
REQUIRED_MANIFEST_FIELDS = (
    "name", "version", "owner", "source_url", "license", "collection_start",
    "collection_end", "inclusion_criteria", "exclusion_criteria",
    "sampling_procedure", "label_definition", "adjudicators",
    "prediction_precedes_outcome", "partitions_are_disjoint", "dataset_sha256",
)


class DatasetValidationError(ValueError):
    pass


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise DatasetValidationError(f"{field} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DatasetValidationError(f"{field} is not a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise DatasetValidationError(f"{field} must include a timezone")
    return parsed


def _manifest(path: Path, digest: str) -> dict[str, Any]:
    if path.stat().st_size > 128_000:
        raise DatasetValidationError("Manifest exceeds 128 KB")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(f"Invalid manifest: {exc}") from exc
    if not isinstance(value, dict):
        raise DatasetValidationError("Manifest must be a JSON object")
    missing = [field for field in REQUIRED_MANIFEST_FIELDS if field not in value]
    if missing:
        raise DatasetValidationError("Manifest is missing: " + ", ".join(missing))
    for field in ("name", "version", "owner", "license", "inclusion_criteria",
                  "exclusion_criteria", "sampling_procedure", "label_definition"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise DatasetValidationError(f"Manifest field {field} must be a non-empty string")
    url = value["source_url"]
    if not isinstance(url, str) or not url.startswith("https://"):
        raise DatasetValidationError("source_url must be an HTTPS URL")
    if value["prediction_precedes_outcome"] is not True or value["partitions_are_disjoint"] is not True:
        raise DatasetValidationError("Manifest must attest temporal ordering and disjoint partitions")
    adjudicators = value["adjudicators"]
    if not isinstance(adjudicators, list) or len(set(map(str, adjudicators))) < 2:
        raise DatasetValidationError("At least two distinct adjudicator identifiers are required")
    start = _timestamp(value["collection_start"], "collection_start")
    end = _timestamp(value["collection_end"], "collection_end")
    if end < start:
        raise DatasetValidationError("collection_end must not precede collection_start")
    if value["dataset_sha256"] != digest:
        raise DatasetValidationError("Dataset SHA-256 does not match the manifest")
    return value


def load_dataset(data_path: str | Path, manifest_path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    data_file = Path(data_path).expanduser().resolve(strict=True)
    manifest_file = Path(manifest_path).expanduser().resolve(strict=True)
    if not data_file.is_file() or data_file.stat().st_size < 1 or data_file.stat().st_size > MAX_DATASET_BYTES:
        raise DatasetValidationError("Dataset must be a regular file between 1 byte and 50 MB")
    digest = hashlib.sha256(data_file.read_bytes()).hexdigest()
    manifest = _manifest(manifest_file, digest)
    records: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    source_splits: dict[str, str] = {}
    with data_file.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if len(line.encode("utf-8")) > 64_000:
                raise DatasetValidationError(f"JSONL row {line_number} exceeds 64 KB")
            if not line.strip():
                continue
            if len(records) >= MAX_ROWS:
                raise DatasetValidationError("Dataset exceeds one million records")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(f"Invalid JSON on line {line_number}") from exc
            if not isinstance(row, dict):
                raise DatasetValidationError(f"Row {line_number} must be an object")
            record_id = row.get("id")
            if not isinstance(record_id, str) or not record_id.strip() or record_id in identifiers:
                raise DatasetValidationError(f"Row {line_number} has a missing or duplicate id")
            identifiers.add(record_id)
            probability = row.get("probability")
            if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not math.isfinite(probability) or not 0 <= probability <= 1:
                raise DatasetValidationError(f"Row {line_number} has an invalid probability")
            if row.get("outcome") not in (0, 1):
                raise DatasetValidationError(f"Row {line_number} outcome must be 0 or 1")
            if not isinstance(row.get("domain"), str) or not row["domain"].strip():
                raise DatasetValidationError(f"Row {line_number} requires a domain")
            if not isinstance(row.get("source_ref"), str) or not row["source_ref"].strip():
                raise DatasetValidationError(f"Row {line_number} requires source_ref")
            split = row.get("split")
            if split not in {"train", "dev", "test"}:
                raise DatasetValidationError(f"Row {line_number} split must be train, dev or test")
            prior_split = source_splits.setdefault(row["source_ref"], split)
            if prior_split != split:
                raise DatasetValidationError(f"Source leakage: {row['source_ref']} occurs in {prior_split} and {split}")
            predicted = _timestamp(row.get("prediction_time"), f"row {line_number} prediction_time")
            observed = _timestamp(row.get("outcome_time"), f"row {line_number} outcome_time")
            if predicted >= observed:
                raise DatasetValidationError(f"Row {line_number} prediction must precede observed outcome")
            records.append(row)
    if len(records) < 15:
        raise DatasetValidationError("At least 15 records are required (five per split)")
    for domain in {row["domain"] for row in records}:
        for split in ("train", "dev", "test"):
            count = sum(row["domain"] == domain and row["split"] == split for row in records)
            if count < 5:
                raise DatasetValidationError(f"Domain {domain!r} needs at least five {split} records")
    return records, manifest, digest


def evaluate_dataset(records: list[dict[str, Any]], *, bins: int = 10) -> dict[str, Any]:
    domains = sorted({row["domain"] for row in records})
    output: dict[str, Any] = {"domains": {}, "dataset_record_count": len(records)}
    for domain in domains:
        selected = [row for row in records if row["domain"] == domain]
        train = [DecisionExample(float(row["probability"]), int(row["outcome"]), row["id"])
                 for row in selected if row["split"] == "train"]
        calibrator = HistogramCalibrator(bins).fit(train)
        split_reports = {}
        for split in ("dev", "test"):
            held = [row for row in selected if row["split"] == split]
            baseline = evaluate_binary_decisions([
                {"probability": float(row["probability"]), "outcome": int(row["outcome"])}
                for row in held
            ])
            calibrated = evaluate_binary_decisions([
                {"probability": calibrator.predict(float(row["probability"])),
                 "outcome": int(row["outcome"])} for row in held
            ])
            split_reports[split] = {
                "count": len(held),
                "baseline": _metric_dict(baseline),
                "calibrated": _metric_dict(calibrated),
            }
        output["domains"][domain] = {
            "train_count": len(train), "splits": split_reports,
            "calibrator_bins": calibrator.summary(),
        }
    output["interpretation"] = (
        "The calibrator is fitted only on train rows. Dev and test are reported separately; "
        "test results must not be used to tune the method. Provenance fields are operator-supplied "
        "and do not authenticate the dataset owner or adjudicators."
    )
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and evaluate a real-world calibration JSONL dataset.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output")
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args(argv)
    if not 2 <= args.bins <= 100:
        raise ValueError("bins must be between 2 and 100")
    records, manifest, digest = load_dataset(args.data, args.manifest)
    report = {"dataset": manifest["name"], "version": manifest["version"],
              "sha256": digest, "evaluation": evaluate_dataset(records, bins=args.bins)}
    payload = json.dumps(report, indent=2)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
