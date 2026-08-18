#!/usr/bin/env python3
"""Add targeted behavior features directly to the v15 multiclass RF."""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import duckdb
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from attached_temporal_features import ROLLING_SOURCE_COLUMNS, build_attachment_feature_frame  # noqa: E402
from mlflow_workflow import PROJECT_EXPERIMENT, configure_tracking, file_sha256, hypothesis_run  # noqa: E402
from probability_adjusted_model import ProbabilityAdjustedClassifier  # noqa: E402
from v16_infilteration_hard_negative_validation import (  # noqa: E402
    ID_COL,
    INFILTERATION,
    OUTER_VALID_FRACTION,
    RANDOM_STATE,
    REQUESTED_V15_FLOW_FEATURES,
    TARGET,
    TIMESTAMP,
    TRAIN_SAMPLE,
    class_chronological_masks,
    evaluate,
    resolve_dataset_paths,
)
from v17_infilteration_behavior_features_validation import (  # noqa: E402
    BEHAVIOR_SOURCE_COLUMNS,
    build_behavior_features,
)


HYPOTHESIS_ID = "v17b"
HYPOTHESIS = "v15_random_forest_plus_targeted_row_behavior_features"
SOURCE_V15_RUN_ID = "4971c300af7f490bbdcd1de7606eccde"
SOURCE_V15_MODEL_URI = "models:/m-b063a6360c3246aaa77c08454ec78a3f"
INNER_VALID_FRACTION = 0.10
INF_WEIGHT_MULTIPLIERS = (0.5, 0.75, 1.0, 1.25, 1.6, 2.0)
PROBABILITY_MULTIPLIERS = (0.5, 0.75, 1.0, 1.1, 1.225, 1.35, 1.5)


def build_rf(encoded_labels: np.ndarray, class_count: int, inf_index: int, inf_weight: float):
    counts = np.bincount(encoded_labels, minlength=class_count).astype(float)
    weights = {
        index: len(encoded_labels) / (class_count * count)
        for index, count in enumerate(counts)
    }
    weights[inf_index] *= float(inf_weight)
    return RandomForestClassifier(
        n_estimators=100,
        n_jobs=4,
        random_state=RANDOM_STATE,
        class_weight=weights,
    )


def predict_adjusted(base_model, frame: pd.DataFrame, classes: list[str], multiplier: float):
    probabilities = np.asarray(base_model.predict_proba(frame), dtype=float)
    encoded_classes = np.asarray(base_model.classes_).astype(int)
    inf_index = classes.index(INFILTERATION)
    inf_column = int(np.flatnonzero(encoded_classes == inf_index)[0])
    probabilities[:, inf_column] *= float(multiplier)
    return np.asarray(classes, dtype=object)[encoded_classes[probabilities.argmax(axis=1)]]


def main() -> None:
    np.random.seed(RANDOM_STATE)
    started = time.time()
    raw_path, cohort_path = resolve_dataset_paths()
    output_dir = PROJECT_DIR / "outputs" / HYPOTHESIS_ID
    output_dir.mkdir(parents=True, exist_ok=True)

    with duckdb.connect() as connection:
        available = connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{raw_path}')"
        ).df()["column_name"].tolist()
        selected_v15 = [c for c in REQUESTED_V15_FLOW_FEATURES if c in available]
        required = list(dict.fromkeys([ID_COL, TARGET, *ROLLING_SOURCE_COLUMNS, *selected_v15]))
        quoted = ", ".join(f'"{column}"' for column in required)
        full = connection.execute(
            f"SELECT {quoted} FROM read_parquet('{raw_path}') ORDER BY {TIMESTAMP}, {ID_COL}"
        ).df()
        cohort_ids = connection.execute(
            f"SELECT {ID_COL} FROM read_parquet('{cohort_path}') "
            f"USING SAMPLE reservoir({TRAIN_SAMPLE} ROWS) REPEATABLE ({RANDOM_STATE})"
        ).df()[ID_COL]
        full[TIMESTAMP] = pd.to_datetime(full[TIMESTAMP])
        full_v15, _ = build_attachment_feature_frame(full, selected_v15)
        cohort_mask = full[ID_COL].isin(set(cohort_ids.tolist()))
        metadata = full.loc[cohort_mask, [ID_COL, TIMESTAMP, TARGET]].reset_index(drop=True)
        v15_features = full_v15.loc[cohort_mask].reset_index(drop=True)
        del full, full_v15
        gc.collect()

        connection.register("v17b_ids", metadata[[ID_COL]])
        behavior_columns = [column for column in BEHAVIOR_SOURCE_COLUMNS if column in available]
        behavior_quoted = ", ".join(f'raw."{column}"' for column in behavior_columns)
        raw_sample = connection.execute(
            f"SELECT {behavior_quoted} FROM read_parquet('{raw_path}') raw "
            f"INNER JOIN v17b_ids ids USING ({ID_COL})"
        ).df()

    raw_sample = raw_sample.set_index(ID_COL).loc[metadata[ID_COL]].reset_index()
    behavior = build_behavior_features(raw_sample)
    behavior.columns = [f"behavior__{column}" for column in behavior.columns]
    features = pd.concat([v15_features, behavior], axis=1)
    del raw_sample, behavior
    gc.collect()

    labels = metadata[TARGET].astype(str)
    classes = sorted(labels.unique().tolist())
    attack_classes = [label for label in classes if label != "Benign"]
    class_to_index = {label: index for index, label in enumerate(classes)}
    encoded = labels.map(class_to_index).to_numpy(np.int64)
    inf_index = class_to_index[INFILTERATION]
    outer_train, outer_valid = class_chronological_masks(metadata, OUTER_VALID_FRACTION)
    inner_train, inner_valid = class_chronological_masks(
        metadata, INNER_VALID_FRACTION, eligible=outer_train
    )
    inner_train_pos = np.flatnonzero(inner_train.to_numpy())
    inner_valid_pos = np.flatnonzero(inner_valid.to_numpy())
    outer_train_pos = np.flatnonzero(outer_train.to_numpy())
    outer_valid_pos = np.flatnonzero(outer_valid.to_numpy())

    rows: list[dict[str, float]] = []
    fitted_inner: dict[float, RandomForestClassifier] = {}
    for inf_weight in INF_WEIGHT_MULTIPLIERS:
        model = build_rf(
            encoded[inner_train_pos], len(classes), inf_index, inf_weight
        )
        model.fit(features.iloc[inner_train_pos], encoded[inner_train_pos])
        fitted_inner[inf_weight] = model
        for probability_multiplier in PROBABILITY_MULTIPLIERS:
            prediction = predict_adjusted(
                model,
                features.iloc[inner_valid_pos],
                classes,
                probability_multiplier,
            )
            metrics = evaluate(labels.iloc[inner_valid_pos], prediction, attack_classes)
            rows.append(
                {
                    "infilteration_class_weight_multiplier": inf_weight,
                    "infilteration_probability_multiplier": probability_multiplier,
                    **metrics,
                }
            )

    candidates = pd.DataFrame(rows).sort_values(
        ["weighted_f1_attacks", "infilteration_f1", "infilteration_precision"],
        ascending=False,
    ).reset_index(drop=True)
    selected = candidates.iloc[0]
    selected_weight = float(selected["infilteration_class_weight_multiplier"])
    selected_probability = float(selected["infilteration_probability_multiplier"])
    print("Selected direct behavior model:", selected.to_dict())
    del fitted_inner
    gc.collect()

    direct_model = build_rf(
        encoded[outer_train_pos], len(classes), inf_index, selected_weight
    )
    direct_model.fit(features.iloc[outer_train_pos], encoded[outer_train_pos])
    direct_prediction = predict_adjusted(
        direct_model,
        features.iloc[outer_valid_pos],
        classes,
        selected_probability,
    )
    direct_metrics = evaluate(labels.iloc[outer_valid_pos], direct_prediction, attack_classes)

    configure_tracking(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"), PROJECT_EXPERIMENT)
    v15_model = mlflow.sklearn.load_model(SOURCE_V15_MODEL_URI)
    v15_prediction = v15_model.predict(v15_features.iloc[outer_valid_pos])
    v15_metrics = evaluate(labels.iloc[outer_valid_pos], v15_prediction, attack_classes)
    print("Outer v15 baseline:", v15_metrics)
    print("Outer v17b direct result:", direct_metrics)

    importance = pd.Series(
        direct_model.feature_importances_, index=features.columns, name="importance"
    ).sort_values(ascending=False)
    behavior_importance = importance[importance.index.str.startswith("behavior__")]
    importance_path = output_dir / "v17b_behavior_feature_importance.csv"
    behavior_importance.to_csv(importance_path, encoding="utf-8-sig")
    candidates_path = output_dir / "v17b_inner_weight_probability_candidates.csv"
    candidates.to_csv(candidates_path, index=False, encoding="utf-8-sig")
    comparison_path = output_dir / "v17b_outer_comparison.csv"
    pd.DataFrame(
        [
            {"model": "v15_baseline", **v15_metrics},
            {"model": "v17b_direct_behavior_multiclass", **direct_metrics},
        ]
    ).to_csv(comparison_path, index=False, encoding="utf-8-sig")
    manifest = {
        "selected_infilteration_class_weight_multiplier": selected_weight,
        "selected_infilteration_probability_multiplier": selected_probability,
        "v15_feature_count": int(v15_features.shape[1]),
        "behavior_feature_count": int(features.shape[1] - v15_features.shape[1]),
        "total_feature_count": int(features.shape[1]),
        "outer_v15_metrics": v15_metrics,
        "outer_v17b_metrics": direct_metrics,
        "source_v15_run_id": SOURCE_V15_RUN_ID,
    }
    manifest_path = output_dir / "v17b_validation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    improved = direct_metrics["weighted_f1_attacks"] > v15_metrics["weighted_f1_attacks"]
    status = "full_refit_candidate" if improved else "rejected_on_outer_validation"
    adjusted_model = ProbabilityAdjustedClassifier(
        base_model=direct_model,
        label_names=classes,
        target_class_index=inf_index,
        multiplier=selected_probability,
    )
    with hypothesis_run(
        run_name="v17b_direct_behavior_multiclass_validation",
        hypothesis_id=HYPOTHESIS_ID,
        hypothesis=HYPOTHESIS,
        stage="chronological_validation",
        promotion_status=status,
        validation_strategy="inner_weight_probability_selection_outer_chronological_evaluation",
        notebook="scripts/v17b_direct_behavior_multiclass_validation.py",
        data_version=raw_path.name,
        feature_schema_version="v15-plus-targeted-row-behavior-v1",
        code_version=file_sha256(Path(__file__)),
        extra_tags={
            "source_v15_run_id": SOURCE_V15_RUN_ID,
            "absolute_timestamp_features": "false",
            "outer_labels_used_for_selection": "false",
        },
    ) as run:
        mlflow.log_params(
            {
                "selected_infilteration_class_weight_multiplier": selected_weight,
                "selected_infilteration_probability_multiplier": selected_probability,
                "v15_feature_count": v15_features.shape[1],
                "behavior_feature_count": features.shape[1] - v15_features.shape[1],
                "total_feature_count": features.shape[1],
                "source_v15_run_id": SOURCE_V15_RUN_ID,
            }
        )
        mlflow.log_metrics(
            {
                **{f"outer_v17b_{key}": float(value) for key, value in direct_metrics.items()},
                **{f"outer_v15_{key}": float(value) for key, value in v15_metrics.items()},
                "delta_weighted_f1_attacks": direct_metrics["weighted_f1_attacks"]
                - v15_metrics["weighted_f1_attacks"],
                "delta_infilteration_f1": direct_metrics["infilteration_f1"]
                - v15_metrics["infilteration_f1"],
                "elapsed_seconds": time.time() - started,
            }
        )
        for artifact in (importance_path, candidates_path, comparison_path, manifest_path):
            mlflow.log_artifact(str(artifact))
        input_example = features.iloc[outer_valid_pos[:5]].copy()
        model_info = mlflow.sklearn.log_model(
            adjusted_model,
            name="direct_behavior_multiclass_model",
            input_example=input_example,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            metadata={
                "selected_infilteration_class_weight_multiplier": selected_weight,
                "selected_infilteration_probability_multiplier": selected_probability,
                "source_v15_model_uri": SOURCE_V15_MODEL_URI,
            },
        )
        run_id = run.info.run_id
        model_uri = model_info.model_uri

    print("Validation run:", run_id)
    print("Direct behavior model:", model_uri)
    print("Status:", status)


if __name__ == "__main__":
    main()
