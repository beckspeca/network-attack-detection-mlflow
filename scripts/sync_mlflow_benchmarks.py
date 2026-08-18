#!/usr/bin/env python3
"""Create one canonical, apples-to-apples MLflow benchmark run per version."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import duckdb
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from mlflow_workflow import (  # noqa: E402
    BENCHMARK_EXPERIMENT,
    BENCHMARK_SCHEMA_VERSION,
    CORE_BENCHMARK_METRICS,
    PROJECT_EXPERIMENT,
    VALIDATION_COHORT_ID,
    configure_tracking,
    file_sha256,
    log_dataframe_input,
)


BENIGN_VALID_ROWS = 34_373
VALIDATION_ROWS = 49_988
TRAIN_SAMPLE_ROWS = 250_000
RANDOM_STATE = 42

# Only source runs whose metrics were computed on the exact validation cohort
# identified by VALIDATION_COHORT_ID belong here. Holdout metrics are excluded
# because the holdout construction changed across hypotheses.
BENCHMARK_SOURCES: dict[str, dict[str, str]] = {
    "v5": {
        "source_run_id": "068c37977c5a4afe9b679645d008aea5",
        "model_variant": "past_context_only",
        "promotion_outcome": "superseded",
        "notebook": "code_mlflow_pipeline_v5.ipynb",
    },
    "v6": {
        "source_run_id": "4dc06735cbfe40f981c74df531adcf8a",
        "model_variant": "hierarchical_threshold_0.45",
        "promotion_outcome": "superseded",
        "notebook": "code_mlflow_pipeline_v6.ipynb",
    },
    "v7": {
        "source_run_id": "6aff4eaa41d74cf29bf5bfb3257633d7",
        "model_variant": "residual_classifier",
        "promotion_outcome": "rejected",
        "notebook": "code_mlflow_pipeline_v7.ipynb",
    },
    "v8": {
        "source_run_id": "318ac0e3cbd24b72946d036683a76ddc",
        "model_variant": "time_matched_residual",
        "promotion_outcome": "rejected",
        "notebook": "code_mlflow_pipeline_v8.ipynb",
    },
    "v9": {
        "source_run_id": "616727bc6b4c4cf88025d8f219a92011",
        "model_variant": "service_hard_negative",
        "promotion_outcome": "rejected",
        "notebook": "code_mlflow_pipeline_v9.ipynb",
    },
    "v10": {
        "source_run_id": "483c19ee3c4b4de08c21764bd40e8ee8",
        "model_variant": "domain_easyensemble",
        "promotion_outcome": "rejected",
        "notebook": "code_mlflow_pipeline_v10.ipynb",
    },
    "v11": {
        "source_run_id": "8eba7dd397e448e8a2155efd72e0569b",
        "model_variant": "v6_infilteration_threshold_0.395",
        "promotion_outcome": "champion",
        "notebook": "code_mlflow_pipeline_v11.ipynb",
    },
    "v12": {
        "source_run_id": "b75e91ee9efc4d63a461b3ec49c7d528",
        "model_variant": "standalone_raw_flow_residual_mlp",
        "promotion_outcome": "rejected",
        "notebook": "code_mlflow_pipeline_v12.ipynb",
    },
    "v13": {
        "source_run_id": "6417f1ce20df4e90b5f15c44fe224c1a",
        "model_variant": "attached_temporal_random_forest_standardized",
        "promotion_outcome": "benchmark_only",
        "notebook": "code_mlflow_pipeline_v13.ipynb",
    },
    "v14": {
        "source_run_id": "8f424f3d1a9742b4a5d7ff76798d7b1a",
        "model_variant": "cross_model_probability_ensemble",
        "promotion_outcome": "benchmark_only",
        "notebook": "code_mlflow_pipeline_v14.ipynb",
    },
    "v15": {
        "source_run_id": "4971c300af7f490bbdcd1de7606eccde",
        "model_variant": "attached_temporal_rf_infilteration_multiplier_1.225",
        "promotion_outcome": "benchmark_only",
        "notebook": "code_mlflow_pipeline_v15.ipynb",
    },
}


def find_training_parquet() -> Path:
    matches = sorted(PROJECT_DIR.parent.glob("datasets/*/processed/train_temporal_v5.parquet"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one training parquet, found: {matches}")
    return matches[0]


def load_lineage_sample(path: Path, rows: int = 2_000) -> pd.DataFrame:
    with duckdb.connect() as connection:
        frame = connection.execute(
            f"SELECT * FROM read_parquet(?) USING SAMPLE reservoir({rows} ROWS) REPEATABLE (42)",
            [str(path)],
        ).df()
    numeric = frame.select_dtypes(include=[np.number]).columns
    frame[numeric] = frame[numeric].astype(float)
    return frame


def exact_source_metrics(client: MlflowClient, source_run_id: str) -> tuple[Any, dict[str, float]]:
    source = client.get_run(source_run_id)
    if source.info.status != "FINISHED":
        raise RuntimeError(f"Source run is not FINISHED: {source_run_id} ({source.info.status})")
    missing = [key for key in CORE_BENCHMARK_METRICS if key not in source.data.metrics]
    if missing:
        raise RuntimeError(f"Source run {source_run_id} is missing core metrics: {missing}")
    return source, {key: float(source.data.metrics[key]) for key in CORE_BENCHMARK_METRICS}


def existing_canonical_run(client: MlflowClient, experiment_id: str, version: str):
    matches = client.search_runs(
        [experiment_id],
        filter_string=(
            f"tags.run_role = 'canonical_benchmark' AND "
            f"tags.benchmark_version = '{version}' AND "
            f"tags.benchmark_schema_version = '{BENCHMARK_SCHEMA_VERSION}'"
        ),
        max_results=10,
    )
    if len(matches) > 1:
        raise RuntimeError(f"Multiple canonical benchmark runs found for {version}: {[r.info.run_id for r in matches]}")
    return matches[0] if matches else None


def verify_existing(existing, source_run_id: str, metrics: dict[str, float]) -> None:
    if existing.data.tags.get("source_run_id") != source_run_id:
        raise RuntimeError(
            f"Canonical run {existing.info.run_id} points to a different source run: "
            f"{existing.data.tags.get('source_run_id')} != {source_run_id}"
        )
    for key, expected in metrics.items():
        actual = existing.data.metrics.get(key)
        if actual is None or not np.isclose(actual, expected, rtol=0, atol=1e-12):
            raise RuntimeError(
                f"Canonical run {existing.info.run_id} metric mismatch for {key}: {actual} != {expected}"
            )


def main() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    experiment_id = configure_tracking(tracking_uri, BENCHMARK_EXPERIMENT)
    client = MlflowClient()
    training_parquet = find_training_parquet()
    training_digest = file_sha256(training_parquet)
    lineage_sample = load_lineage_sample(training_parquet)

    client.set_experiment_tag(experiment_id, "purpose", "canonical apples-to-apples model comparison")
    client.set_experiment_tag(experiment_id, "benchmark_schema_version", BENCHMARK_SCHEMA_VERSION)
    client.set_experiment_tag(experiment_id, "validation_cohort_id", VALIDATION_COHORT_ID)
    client.set_experiment_tag(experiment_id, "source_experiment", PROJECT_EXPERIMENT)
    client.set_experiment_tag(
        experiment_id,
        "mlflow.note.content",
        "One flat canonical run per version. Compare all runs directly; do not add candidate/refit runs here.",
    )

    primary = client.get_experiment_by_name(PROJECT_EXPERIMENT)
    if primary:
        client.set_experiment_tag(primary.experiment_id, "purpose", "development candidates, refits, and registry lineage")
        client.set_experiment_tag(primary.experiment_id, "comparison_experiment", BENCHMARK_EXPERIMENT)

    records: list[dict[str, Any]] = []
    run_ids: dict[str, str] = {}
    source_cache: dict[str, tuple[Any, dict[str, float]]] = {}

    for version, config in BENCHMARK_SOURCES.items():
        source, metrics = exact_source_metrics(client, config["source_run_id"])
        source_cache[version] = (source, metrics)

    v6_metrics = source_cache["v6"][1]
    for version, config in BENCHMARK_SOURCES.items():
        source, metrics = source_cache[version]
        comparable_metrics = {
            **metrics,
            "benign_to_infilteration_fp_rate": metrics["benign_to_infilteration_fp"] / BENIGN_VALID_ROWS,
            "delta_vs_v6_accuracy": metrics["accuracy"] - v6_metrics["accuracy"],
            "delta_vs_v6_infilteration_precision": metrics["infilteration_precision"] - v6_metrics["infilteration_precision"],
            "delta_vs_v6_infilteration_recall": metrics["infilteration_recall"] - v6_metrics["infilteration_recall"],
            "delta_vs_v6_infilteration_f1": metrics["infilteration_f1"] - v6_metrics["infilteration_f1"],
            "delta_vs_v6_benign_to_infilteration_fp": metrics["benign_to_infilteration_fp"] - v6_metrics["benign_to_infilteration_fp"],
        }
        existing = existing_canonical_run(client, experiment_id, version)
        if existing:
            verify_existing(existing, config["source_run_id"], metrics)
            canonical_run_id = existing.info.run_id
            status = "verified_existing"
        else:
            tags = {
                "project": "network-classification",
                "run_role": "canonical_benchmark",
                "benchmark_version": version,
                "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
                "validation_cohort_id": VALIDATION_COHORT_ID,
                "source_run_id": config["source_run_id"],
                "source_experiment_id": source.info.experiment_id,
                "source_notebook": config["notebook"],
                "model_variant": config["model_variant"],
                "promotion_outcome": config["promotion_outcome"],
                "metric_origin": "source_validation_run",
                "strictly_comparable": "true",
            }
            run_name = f"{version} | {config['model_variant']} | {config['promotion_outcome']}"
            with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, tags=tags) as run:
                mlflow.log_params({
                    "train_sample_rows": TRAIN_SAMPLE_ROWS,
                    "validation_rows": VALIDATION_ROWS,
                    "validation_benign_rows": BENIGN_VALID_ROWS,
                    "validation_fraction": 0.20,
                    "random_state": RANDOM_STATE,
                    "split_strategy": "within_class_chronological_80_20",
                    "dataset": training_parquet.name,
                    "dataset_sha256": training_digest,
                    "source_run_id": config["source_run_id"],
                    "model_variant": config["model_variant"],
                })
                mlflow.log_metrics(comparable_metrics)
                log_dataframe_input(
                    lineage_sample,
                    source=training_parquet,
                    name="train_temporal_v5_lineage_sample",
                    context="canonical_benchmark",
                    targets="Label",
                )
                canonical_run_id = run.info.run_id
            status = "created"
        run_ids[version] = canonical_run_id
        records.append({
            "version": version,
            "model_variant": config["model_variant"],
            "promotion_outcome": config["promotion_outcome"],
            "canonical_run_id": canonical_run_id,
            "source_run_id": config["source_run_id"],
            "source_experiment_id": source.info.experiment_id,
            "sync_status": status,
            **comparable_metrics,
        })

    output_dir = PROJECT_DIR / "outputs" / "mlflow_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "canonical_model_comparison.csv"
    with comparison_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    manifest_path = output_dir / "benchmark_manifest.json"
    manifest_path.write_text(json.dumps({
        "benchmark_experiment": BENCHMARK_EXPERIMENT,
        "experiment_id": experiment_id,
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "validation_cohort_id": VALIDATION_COHORT_ID,
        "training_data": str(training_parquet),
        "training_data_sha256": training_digest,
        "excluded_metrics": [
            "holdout_* (definitions differ by version)",
            "training time (hardware/run conditions not normalized)",
        ],
        "runs": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    for run_id in run_ids.values():
        client.log_artifact(run_id, str(comparison_path), artifact_path="benchmark")
        client.log_artifact(run_id, str(manifest_path), artifact_path="benchmark")

    print(json.dumps({
        "status": "ok",
        "experiment": BENCHMARK_EXPERIMENT,
        "experiment_id": experiment_id,
        "runs": run_ids,
        "comparison_csv": str(comparison_path),
        "manifest": str(manifest_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
