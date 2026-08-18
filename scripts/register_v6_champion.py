#!/usr/bin/env python3
"""Package the validated v6 hierarchy and register it as the champion model."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from pathlib import Path

import duckdb
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from mlflow_workflow import (  # noqa: E402
    CHAMPION_ALIAS,
    PROJECT_EXPERIMENT,
    REGISTERED_MODEL_NAME,
    HierarchicalIDSModel,
    configure_tracking,
    file_sha256,
    hypothesis_run,
    log_dataframe_input,
    register_model_version,
)


V6_VALIDATION_RUN_ID = "4dc06735cbfe40f981c74df531adcf8a"
V6_REFIT_RUN_ID = "24bfd469769147f0b3d252f7589bc208"
V6_ATTACK_GATE_URI = "models:/m-097892a7ee9e4386821fc0b0f17d8b62"
V6_ATTACK_SUBTYPE_URI = "models:/m-ba1f13805e984953976021ee17cfa9af"
ATTACK_THRESHOLD = 0.45
CALENDAR_FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "elapsed_hours"]
ID_COL = "unique_id"
TARGET = "Label"


def find_training_parquet() -> Path:
    work_root = PROJECT_DIR.parent
    matches = sorted(work_root.glob("datasets/*/processed/train_temporal_v5.parquet"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one v5 training parquet, found: {matches}")
    return matches[0]


def add_ids_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    flag_cols = [
        "SYN Flag Cnt",
        "ACK Flag Cnt",
        "RST Flag Cnt",
        "FIN Flag Cnt",
        "URG Flag Cnt",
        "PSH Flag Cnt",
    ]
    values = {
        column: pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(float)
        for column in flag_cols
    }
    syn, ack = values["SYN Flag Cnt"], values["ACK Flag Cnt"]
    rst, fin = values["RST Flag Cnt"], values["FIN Flag Cnt"]
    urg, psh = values["URG Flag Cnt"], values["PSH Flag Cnt"]
    total_flags = syn + ack + rst + fin + urg + psh + 1.0
    frame["SYN_ACK_ratio"] = (syn + 1) / (ack + 1)
    frame["RST_ACK_ratio"] = (rst + 1) / (ack + 1)
    frame["SYN_RST_over_ACK_FIN"] = (syn + rst + 1) / (ack + fin + 1)
    frame["ACK_SYN_ratio"] = (ack + 1) / (syn + 1)
    frame["URG_ACK_ratio"] = (urg + 1) / (ack + 1)
    frame["SYN_share"] = (syn + 1) / total_flags
    frame["RST_share"] = (rst + 1) / total_flags
    numeric = frame.select_dtypes(include=[np.number]).columns
    frame[numeric] = frame[numeric].replace([np.inf, -np.inf], np.nan)
    return frame


def load_contract_sample(path: Path, rows: int = 200) -> tuple[pd.DataFrame, pd.DataFrame]:
    with duckdb.connect() as connection:
        raw = connection.execute(
            f"SELECT * FROM read_parquet(?) USING SAMPLE reservoir({int(rows)} ROWS) "
            "REPEATABLE (42)",
            [str(path)],
        ).df()
    enriched = add_ids_features(raw)
    features = enriched.drop(
        columns=[TARGET, ID_COL, "Timestamp", *CALENDAR_FEATURES], errors="ignore"
    )
    for column in ["Dst Port", "Protocol"]:
        if column in features:
            features[column] = features[column].astype("string")
    for column in features.columns:
        if column not in {"Dst Port", "Protocol"}:
            features[column] = pd.to_numeric(features[column], errors="coerce").astype(float)
    return raw, features


def package_versions() -> list[str]:
    packages = ["mlflow", "numpy", "pandas", "scikit-learn", "lightgbm"]
    return [f"{name}=={importlib.metadata.version(name)}" for name in packages]


def existing_champion(client: MlflowClient):
    try:
        return client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Create the model version and move champion alias")
    parser.add_argument("--force-new-version", action="store_true")
    parser.add_argument(
        "--allow-champion-rollback",
        action="store_true",
        help="Explicitly allow replacing a newer champion with the v6 model",
    )
    args = parser.parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    experiment_id = configure_tracking(tracking_uri, PROJECT_EXPERIMENT)
    client = MlflowClient()
    current = existing_champion(client)
    if current and current.tags.get("source_refit_run_id") == V6_REFIT_RUN_ID and not args.force_new_version:
        model_uri = f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}"
        model = mlflow.pyfunc.load_model(model_uri)
        training_parquet = find_training_parquet()
        _, sample = load_contract_sample(training_parquet, rows=10)
        prediction = model.predict(sample)
        print(json.dumps({
            "status": "already_registered",
            "experiment_id": experiment_id,
            "model_name": REGISTERED_MODEL_NAME,
            "version": current.version,
            "alias": CHAMPION_ALIAS,
            "smoke_rows": len(prediction),
        }, ensure_ascii=False, indent=2))
        return
    if (
        current
        and current.tags.get("source_refit_run_id") != V6_REFIT_RUN_ID
        and args.apply
        and not args.allow_champion_rollback
    ):
        print(json.dumps({
            "status": "blocked_newer_champion_present",
            "model_name": REGISTERED_MODEL_NAME,
            "current_champion_version": current.version,
            "current_champion_run_id": current.run_id,
            "message": "Use --allow-champion-rollback only for an intentional rollback to v6.",
        }, ensure_ascii=False, indent=2))
        return

    training_parquet = find_training_parquet()
    raw_sample, feature_sample = load_contract_sample(training_parquet)
    gate = mlflow.sklearn.load_model(V6_ATTACK_GATE_URI)
    subtype = mlflow.sklearn.load_model(V6_ATTACK_SUBTYPE_URI)
    feature_columns = list(gate.feature_names_in_)
    feature_sample = feature_sample.loc[:, feature_columns]
    wrapper = HierarchicalIDSModel(
        gate,
        subtype,
        attack_threshold=ATTACK_THRESHOLD,
        feature_columns=feature_columns,
    )
    input_example = feature_sample.iloc[:5].copy()
    output_example = wrapper.predict(None, input_example)
    signature = infer_signature(input_example, output_example)

    plan = {
        "tracking_uri": tracking_uri,
        "experiment": PROJECT_EXPERIMENT,
        "registered_model": REGISTERED_MODEL_NAME,
        "alias": CHAMPION_ALIAS,
        "source_validation_run_id": V6_VALIDATION_RUN_ID,
        "source_refit_run_id": V6_REFIT_RUN_ID,
        "feature_count": len(feature_columns),
        "training_data": str(training_parquet),
        "training_data_sha256": file_sha256(training_parquet),
        "input_columns": feature_columns,
    }
    if not args.apply:
        print(json.dumps({"status": "dry_run", **plan}, ensure_ascii=False, indent=2))
        return

    validation_run = client.get_run(V6_VALIDATION_RUN_ID)
    script_hash = file_sha256(__file__)
    with hypothesis_run(
        run_name="v6_registry_migration",
        hypothesis_id="v6",
        hypothesis="hierarchical_benign_attack_and_attack_subtype",
        stage="registry_migration",
        promotion_status="champion",
        validation_strategy="within_class_chronological_80_20",
        notebook="code_mlflow_pipeline_v6.ipynb",
        data_version="train_temporal_v5.parquet",
        feature_schema_version="v5-temporal-2",
        code_version=script_hash,
        extra_tags={
            "source_validation_run_id": V6_VALIDATION_RUN_ID,
            "source_refit_run_id": V6_REFIT_RUN_ID,
            "model_architecture": "hierarchical_pyfunc",
        },
    ) as run:
        mlflow.log_params({
            "attack_threshold": ATTACK_THRESHOLD,
            "feature_count": len(feature_columns),
            "random_state": 42,
            "source_validation_run_id": V6_VALIDATION_RUN_ID,
            "source_refit_run_id": V6_REFIT_RUN_ID,
            "training_data_sha256": plan["training_data_sha256"],
        })
        mlflow.log_metrics({key: float(value) for key, value in validation_run.data.metrics.items()})
        lineage_sample = raw_sample.copy()
        lineage_numeric = lineage_sample.select_dtypes(include=[np.number]).columns
        lineage_sample[lineage_numeric] = lineage_sample[lineage_numeric].astype(float)
        log_dataframe_input(
            lineage_sample,
            source=training_parquet,
            name="train_temporal_v5_contract_sample",
            context="registry_contract_sample",
            targets=TARGET,
        )
        mlflow.log_artifact(str(Path(__file__).resolve()), artifact_path="code")
        model_info = mlflow.pyfunc.log_model(
            name="network_classifier",
            python_model=wrapper,
            input_example=input_example,
            signature=signature,
            code_paths=[str(SRC_DIR / "mlflow_workflow.py")],
            pip_requirements=package_versions(),
            metadata={
                "attack_threshold": ATTACK_THRESHOLD,
                "source_validation_run_id": V6_VALIDATION_RUN_ID,
                "source_refit_run_id": V6_REFIT_RUN_ID,
            },
        )
        version = register_model_version(
            model_uri=model_info.model_uri,
            version_tags={
                "validation_status": "passed",
                "source_validation_run_id": V6_VALIDATION_RUN_ID,
                "source_refit_run_id": V6_REFIT_RUN_ID,
                "data_version": "train_temporal_v5.parquet",
                "feature_schema_version": "v5-temporal-2",
            },
            registered_model_tags={
                "project": "network-classification",
                "task": "multiclass-classification",
                "architecture": "hierarchical",
            },
        )
        champion = mlflow.pyfunc.load_model(
            f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}"
        )
        smoke_prediction = champion.predict(input_example)
        allowed = {"Benign", "Brute Force", "DDOS", "DoS", "Infilteration"}
        if len(smoke_prediction) != len(input_example):
            raise AssertionError("Registry smoke test returned the wrong number of rows")
        if not set(smoke_prediction["Label"]).issubset(allowed):
            raise AssertionError("Registry smoke test returned an unknown label")
        mlflow.log_metric("registry_smoke_test_passed", 1.0)
        migration_run_id = run.info.run_id

    print(json.dumps({
        "status": "registered",
        "experiment_id": experiment_id,
        "migration_run_id": migration_run_id,
        "logged_model_uri": model_info.model_uri,
        "registered_model": REGISTERED_MODEL_NAME,
        "version": version.version,
        "alias": CHAMPION_ALIAS,
        "smoke_rows": len(smoke_prediction),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
