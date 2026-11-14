"""Resolve the active, versioned benchmark without hard-coded v3 paths."""

from __future__ import annotations

import json
from pathlib import Path


def load_benchmark_manifest(data_dir: str | Path) -> dict:
    data_path = Path(data_dir)
    manifest_path = data_path / "benchmark_manifest.json"
    if not manifest_path.exists():
        return {
            "active_version": "v5",
            "benchmarks": {
                "v5": {
                    "csv": "qa_benchmark.csv",
                    "jsonl": "qa_benchmark.jsonl",
                }
            },
        }
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def get_active_benchmark_path(data_dir: str | Path, extension: str = "jsonl") -> str:
    data_path = Path(data_dir)
    manifest = load_benchmark_manifest(data_path)
    version = manifest.get("active_version", "v5")
    metadata = manifest.get("benchmarks", {}).get(version, {})
    filename = metadata.get(extension)
    if filename and (data_path / filename).exists():
        return str(data_path / filename)
    archive_path = data_path / "archive" / "master_benchmark_2311" / f"qa_benchmark.{extension}"
    if archive_path.exists():
        return str(archive_path)
    return str(data_path / f"qa_benchmark.{extension}")


def get_default_eval_benchmark_path(data_dir: str | Path, extension: str = "jsonl") -> str:
    """Path of the single evaluation/demo benchmark (the 750-question set named by the manifest).

    There is no fallback to another set: if the file is missing this raises, so a run can never
    silently be scored on a different benchmark.
    """
    data_path = Path(data_dir)
    manifest = load_benchmark_manifest(data_path)
    subset = manifest.get("eval_subsets", {}).get(manifest.get("primary_evaluation_set", ""), {})
    filename = subset.get(extension) or f"foodhealthqa_eval_set.{extension}"
    path = data_path / filename
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation benchmark not found: {path}")
    return str(path)


def is_evaluation_eligible(record: dict) -> bool:
    """Allow approved historical rows, screened silver rows, and verified source-grounded rows."""
    status = str(record.get("review_status", "")).strip().casefold()
    return status in {
        "approved",
        "machine_screened_pre_gold",
        "approved_source_verified",
        "approved_rule_verified",
    }


def is_gold_eligible(record: dict, benchmark_metadata: dict) -> bool:
    """Require explicit dataset validation and assessed correctness for gold use."""
    if not benchmark_metadata.get("expert_validated", False):
        return False
    status = str(record.get("review_status", "")).strip().casefold()
    reviewer_id = str(record.get("reviewer_id", "")).strip()
    return (
        status == "approved"
        and bool(reviewer_id)
        and str(record.get("expert_gold_eligible", "")).strip().casefold() == "yes"
        and all(
            str(record.get(field, "")).strip().casefold() == "yes"
            for field in ("answer_correct", "citation_correct", "safety_label_correct")
        )
    )
