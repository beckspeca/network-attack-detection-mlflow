#!/usr/bin/env python3
"""Attach the validated v20b rule layer to the full-pool v15 model."""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from mlflow_workflow import PROJECT_EXPERIMENT, configure_tracking, file_sha256, hypothesis_run  # noqa: E402
from v15_full_refit_submission import (  # noqa: E402
    ID_COL,
    REQUESTED_FLOW_FEATURES,
    ROLLING_SOURCE_COLUMNS,
    TARGET,
    TIMESTAMP,
    available_columns,
    build_test_features_from_train_history,
    read_columns,
    resolve_data_paths,
)
from v15_rule_veto_model import V15RuleVetoModel  # noqa: E402


HYPOTHESIS_ID = "v20b"
HYPOTHESIS = "full_pool_v15_plus_validated_error_specific_rule_layer"
SOURCE_VALIDATION_RUN_ID = "cdcf12aa9e684f5b8461e1200005dd44"
SOURCE_RULE_MODEL_URI = "models:/m-30e9cd37853e4035bf5200def48bbd7b"
SOURCE_FULL_V15_RUN_ID = "579cc15849f74cd6af1c0b3a4fb2c939"
SOURCE_FULL_V15_MODEL_URI = "models:/m-cbd07e83a26e48bc8b58351e6495bbb2"


def main() -> None:
    started = time.time()
    output_dir = PROJECT_DIR / "outputs" / HYPOTHESIS_ID
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path, test_path = resolve_data_paths(PROJECT_DIR)
    available = available_columns(train_path)
    selected_flow_features = [column for column in REQUESTED_FLOW_FEATURES if column in available]
    required = list(
        dict.fromkeys([ID_COL, TARGET, *ROLLING_SOURCE_COLUMNS, *selected_flow_features])
    )

    configure_tracking(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"), PROJECT_EXPERIMENT)
    full_v15 = mlflow.sklearn.load_model(SOURCE_FULL_V15_MODEL_URI)
    validated_hybrid = mlflow.sklearn.load_model(SOURCE_RULE_MODEL_URI)
    hybrid = V15RuleVetoModel(
        base_model=full_v15.base_model,
        rule_tree=validated_hybrid.rule_tree,
        label_names=full_v15.label_names,
        target_class_index=full_v15.target_class_index,
        multiplier=full_v15.multiplier,
        veto_leaves=validated_hybrid.veto_leaves,
    )

    full_train = read_columns(train_path, required).sort_values(
        [TIMESTAMP, ID_COL], kind="mergesort"
    ).reset_index(drop=True)
    train_history = full_train.loc[:, ROLLING_SOURCE_COLUMNS].copy()
    port_categories = sorted(full_train["Dst Port"].astype("string").unique().tolist())
    port_map = {value: index for index, value in enumerate(port_categories)}
    del full_train
    gc.collect()

    test_frame = read_columns(test_path, required)
    if test_frame[TARGET].notna().any():
        raise ValueError("Test labels must be empty")
    test_ids = test_frame[ID_COL].copy().reset_index(drop=True)
    test_features = build_test_features_from_train_history(
        train_history=train_history,
        test_frame=test_frame,
        selected_flow_features=selected_flow_features,
        port_map=port_map,
    )
    del train_history, test_frame
    gc.collect()

    base_prediction = full_v15.predict(test_features)
    hybrid_prediction = hybrid.predict(test_features)
    changed = base_prediction != hybrid_prediction
    invalid_change = changed & (
        (base_prediction != "Infilteration") | (hybrid_prediction == "Infilteration")
    )
    if invalid_change.any():
        raise AssertionError("Rule layer changed a prediction outside its allowed action")

    submission = pd.DataFrame({ID_COL: test_ids, TARGET: hybrid_prediction}).set_index(ID_COL)
    submission_path = output_dir / "submission_v20b_rule_hybrid.csv"
    submission.to_csv(submission_path, encoding="utf-8-sig")
    if len(submission) != len(test_ids) or submission[TARGET].isna().any():
        raise AssertionError("Submission row or label validation failed")
    if submission.index.duplicated().any() or submission.index.tolist() != test_ids.tolist():
        raise AssertionError("Submission ID validation failed")

    audit = pd.DataFrame(
        {
            "base_prediction": base_prediction,
            "hybrid_prediction": hybrid_prediction,
            "changed": changed,
        }
    )
    transition = (
        audit.loc[changed]
        .groupby(["base_prediction", "hybrid_prediction"])
        .size()
        .rename("count")
        .reset_index()
    )
    transition_path = output_dir / "v20b_test_rule_transitions.csv"
    transition.to_csv(transition_path, index=False, encoding="utf-8-sig")
    distribution_path = output_dir / "v20b_submission_distribution.csv"
    submission[TARGET].value_counts().rename("count").to_csv(
        distribution_path, encoding="utf-8-sig"
    )
    audit_summary = {
        "test_rows": int(len(submission)),
        "changed_predictions": int(changed.sum()),
        "changed_fraction": float(changed.mean()),
        "all_changes_from_infilteration": bool(
            np.all(base_prediction[changed] == "Infilteration")
        ),
        "all_changes_to_non_infilteration": bool(
            np.all(hybrid_prediction[changed] != "Infilteration")
        ),
        "source_validation_run_id": SOURCE_VALIDATION_RUN_ID,
        "source_full_v15_run_id": SOURCE_FULL_V15_RUN_ID,
        "source_rule": "flow_count_60s > 917 AND Pkt Size Avg <= 82.75",
        "submission_sha256": file_sha256(submission_path),
    }
    audit_path = output_dir / "v20b_full_deployment_manifest.json"
    audit_path.write_text(json.dumps(audit_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    input_example = test_features.iloc[:5].copy()
    expected_example = hybrid.predict(input_example)
    with hypothesis_run(
        run_name="v20b_full_pool_v15_rule_hybrid_deployment",
        hypothesis_id=HYPOTHESIS_ID,
        hypothesis=HYPOTHESIS,
        stage="full_pool_deployment_candidate",
        promotion_status="ids_deployment_candidate",
        validation_strategy="source_rule_selected_on_nested_time_split_and_passed_outer_holdout",
        notebook="scripts/v20b_full_deployment_submission.py",
        data_version=train_path.name,
        feature_schema_version="v15-error-specific-rule-v1",
        code_version=file_sha256(Path(__file__)),
        extra_tags={
            "source_validation_run_id": SOURCE_VALIDATION_RUN_ID,
            "source_full_v15_run_id": SOURCE_FULL_V15_RUN_ID,
            "competition_champion_replaced": "false",
            "test_label_used": "false",
            "operational_ids_objective": "true",
        },
    ) as run:
        mlflow.log_params(
            {
                "fit_rows_source": 4_638_804,
                "test_rows": len(submission),
                "feature_count": len(test_features.columns),
                "veto_rule_count": len(hybrid.veto_leaves),
                "source_full_v15_model_uri": SOURCE_FULL_V15_MODEL_URI,
                "source_rule_model_uri": SOURCE_RULE_MODEL_URI,
                "source_validation_run_id": SOURCE_VALIDATION_RUN_ID,
                "submission_sha256": audit_summary["submission_sha256"],
            }
        )
        mlflow.log_metrics(
            {
                "changed_predictions": int(changed.sum()),
                "changed_fraction": float(changed.mean()),
                "source_outer_ids_precision": 0.7916558424689562,
                "source_outer_ids_recall": 0.975792507204611,
                "source_outer_ids_f1": 0.8741322930411336,
                "source_outer_weighted_f1_attacks": 0.9566419289545457,
                "elapsed_seconds": time.time() - started,
            }
        )
        for artifact in (submission_path, transition_path, distribution_path, audit_path):
            mlflow.log_artifact(str(artifact))
        model_info = mlflow.sklearn.log_model(
            hybrid,
            name="full_pool_v15_rule_hybrid",
            input_example=input_example,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            metadata={
                "requires_past_train_history_features": True,
                "source_full_v15_model_uri": SOURCE_FULL_V15_MODEL_URI,
                "source_validation_run_id": SOURCE_VALIDATION_RUN_ID,
                "rule": audit_summary["source_rule"],
            },
        )
        run_id = run.info.run_id
        model_uri = model_info.model_uri

    reloaded = mlflow.sklearn.load_model(model_uri)
    if not np.array_equal(reloaded.predict(input_example), expected_example):
        raise RuntimeError("Reloaded full hybrid model predictions do not match")
    print("Test rows/features:", test_features.shape)
    print("Rule changes:", int(changed.sum()), float(changed.mean()))
    print("Transitions:\n", transition)
    print("Submission SHA256:", audit_summary["submission_sha256"])
    print("Deployment run:", run_id)
    print("Deployment model:", model_uri)
    print("Reload verification: ok")
    print("Submission:", submission_path)


if __name__ == "__main__":
    main()
