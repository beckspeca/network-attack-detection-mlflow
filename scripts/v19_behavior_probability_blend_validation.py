#!/usr/bin/env python3
"""Soft-blend v15 with a pruned behavior-only Infilteration probability model."""

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
from lightgbm import LGBMClassifier


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from attached_temporal_features import ROLLING_SOURCE_COLUMNS, build_attachment_feature_frame  # noqa: E402
from mlflow_workflow import PROJECT_EXPERIMENT, configure_tracking, file_sha256, hypothesis_run  # noqa: E402
from v16_infilteration_hard_negative_validation import (  # noqa: E402
    ID_COL,
    INFILTERATION,
    OUTER_VALID_FRACTION,
    RANDOM_STATE,
    REQUESTED_V15_FLOW_FEATURES,
    TARGET,
    TIMESTAMP,
    TRAIN_SAMPLE,
    build_v15_base,
    class_chronological_masks,
    evaluate,
    resolve_dataset_paths,
)
from v17_infilteration_behavior_features_validation import (  # noqa: E402
    BEHAVIOR_SOURCE_COLUMNS,
    build_behavior_features,
)
from v18_pruned_behavior_importance_validation import PRUNED_BEHAVIOR_FEATURES  # noqa: E402


HYPOTHESIS_ID = "v19"
HYPOTHESIS = "v15_soft_blend_pruned_behavior_infilteration_probability"
SOURCE_V15_RUN_ID = "4971c300af7f490bbdcd1de7606eccde"
SOURCE_V15_MODEL_URI = "models:/m-b063a6360c3246aaa77c08454ec78a3f"
INNER_VALID_FRACTION = 0.10
POSITIVE_WEIGHTS = (1.0, 4.0, 8.0, 16.0)
BLEND_ALPHAS = (0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30)
FINAL_MULTIPLIERS = (0.75, 0.90, 1.0, 1.10, 1.225, 1.35, 1.50)

# Dst Port stays in v15.  Only its interaction with Protocol is retained in
# the independent behavior model to avoid duplicate standalone information.
V19_BEHAVIOR_FEATURES = [
    feature for feature in PRUNED_BEHAVIOR_FEATURES if feature != "dst_port_code"
]


def build_behavior_model() -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.035,
        num_leaves=21,
        min_child_samples=100,
        max_depth=7,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.8,
        reg_lambda=3.0,
        n_jobs=4,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )


def aligned_v15_probabilities(base_model, frame: pd.DataFrame, class_count: int) -> np.ndarray:
    raw = np.asarray(base_model.predict_proba(frame), dtype=float)
    encoded_classes = np.asarray(base_model.classes_).astype(int)
    aligned = np.zeros((len(frame), class_count), dtype=float)
    for column, encoded_class in enumerate(encoded_classes):
        aligned[:, encoded_class] = raw[:, column]
    return aligned


def blend_prediction(
    v15_probability: np.ndarray,
    behavior_probability: np.ndarray,
    classes: list[str],
    alpha: float,
    final_multiplier: float,
) -> np.ndarray:
    scores = np.asarray(v15_probability, dtype=float).copy()
    inf_index = classes.index(INFILTERATION)
    scores[:, inf_index] = (
        (1.0 - float(alpha)) * scores[:, inf_index]
        + float(alpha) * np.asarray(behavior_probability, dtype=float)
    ) * float(final_multiplier)
    return np.asarray(classes, dtype=object)[scores.argmax(axis=1)]


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

        connection.register("v19_ids", metadata[[ID_COL]])
        source_columns = [column for column in BEHAVIOR_SOURCE_COLUMNS if column in available]
        source_quoted = ", ".join(f'raw."{column}"' for column in source_columns)
        raw_sample = connection.execute(
            f"SELECT {source_quoted} FROM read_parquet('{raw_path}') raw "
            f"INNER JOIN v19_ids ids USING ({ID_COL})"
        ).df()

    raw_sample = raw_sample.set_index(ID_COL).loc[metadata[ID_COL]].reset_index()
    all_behavior = build_behavior_features(raw_sample)
    behavior = all_behavior[V19_BEHAVIOR_FEATURES].copy()
    del raw_sample, all_behavior
    gc.collect()

    labels = metadata[TARGET].astype(str)
    classes = sorted(labels.unique().tolist())
    attack_classes = [label for label in classes if label != "Benign"]
    class_to_index = {label: index for index, label in enumerate(classes)}
    encoded = labels.map(class_to_index).to_numpy(np.int64)
    inf_index = class_to_index[INFILTERATION]
    binary = labels.eq(INFILTERATION).astype(np.int8).to_numpy()
    outer_train, outer_valid = class_chronological_masks(metadata, OUTER_VALID_FRACTION)
    inner_train, inner_valid = class_chronological_masks(
        metadata, INNER_VALID_FRACTION, eligible=outer_train
    )
    inner_train_pos = np.flatnonzero(inner_train.to_numpy())
    inner_valid_pos = np.flatnonzero(inner_valid.to_numpy())
    outer_train_pos = np.flatnonzero(outer_train.to_numpy())
    outer_valid_pos = np.flatnonzero(outer_valid.to_numpy())

    inner_v15 = build_v15_base(encoded[inner_train_pos], classes)
    inner_v15.fit(v15_features.iloc[inner_train_pos], encoded[inner_train_pos])
    inner_v15_probability = aligned_v15_probabilities(
        inner_v15, v15_features.iloc[inner_valid_pos], len(classes)
    )
    candidate_rows: list[dict[str, float]] = []
    fitted_behavior: dict[float, LGBMClassifier] = {}
    for positive_weight in POSITIVE_WEIGHTS:
        behavior_model = build_behavior_model()
        sample_weight = np.where(binary[inner_train_pos] == 1, positive_weight, 1.0)
        behavior_model.fit(
            behavior.iloc[inner_train_pos],
            binary[inner_train_pos],
            sample_weight=sample_weight,
        )
        fitted_behavior[positive_weight] = behavior_model
        behavior_probability = behavior_model.predict_proba(
            behavior.iloc[inner_valid_pos]
        )[:, 1]
        for alpha in BLEND_ALPHAS:
            for final_multiplier in FINAL_MULTIPLIERS:
                prediction = blend_prediction(
                    inner_v15_probability,
                    behavior_probability,
                    classes,
                    alpha,
                    final_multiplier,
                )
                metrics = evaluate(labels.iloc[inner_valid_pos], prediction, attack_classes)
                candidate_rows.append(
                    {
                        "behavior_positive_weight": positive_weight,
                        "blend_alpha": alpha,
                        "infilteration_final_multiplier": final_multiplier,
                        **metrics,
                    }
                )

    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["weighted_f1_attacks", "infilteration_f1", "infilteration_precision"],
        ascending=False,
    ).reset_index(drop=True)
    selected = candidates.iloc[0]
    selected_weight = float(selected["behavior_positive_weight"])
    selected_alpha = float(selected["blend_alpha"])
    selected_multiplier = float(selected["infilteration_final_multiplier"])
    print("Selected soft blend:", selected.to_dict())
    del fitted_behavior, inner_v15
    gc.collect()

    behavior_model = build_behavior_model()
    behavior_model.fit(
        behavior.iloc[outer_train_pos],
        binary[outer_train_pos],
        sample_weight=np.where(binary[outer_train_pos] == 1, selected_weight, 1.0),
    )
    behavior_outer_probability = behavior_model.predict_proba(
        behavior.iloc[outer_valid_pos]
    )[:, 1]

    configure_tracking(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"), PROJECT_EXPERIMENT)
    v15_model = mlflow.sklearn.load_model(SOURCE_V15_MODEL_URI)
    outer_v15_probability = aligned_v15_probabilities(
        v15_model.base_model, v15_features.iloc[outer_valid_pos], len(classes)
    )
    v15_prediction = blend_prediction(
        outer_v15_probability,
        np.zeros(len(outer_valid_pos), dtype=float),
        classes,
        alpha=0.0,
        final_multiplier=1.225,
    )
    v19_prediction = blend_prediction(
        outer_v15_probability,
        behavior_outer_probability,
        classes,
        selected_alpha,
        selected_multiplier,
    )
    v15_metrics = evaluate(labels.iloc[outer_valid_pos], v15_prediction, attack_classes)
    v19_metrics = evaluate(labels.iloc[outer_valid_pos], v19_prediction, attack_classes)
    print("Outer v15 baseline:", v15_metrics)
    print("Outer v19 soft blend:", v19_metrics)

    importance = pd.Series(
        behavior_model.feature_importances_, index=behavior.columns, name="importance"
    ).sort_values(ascending=False)
    importance_path = output_dir / "v19_behavior_model_feature_importance.csv"
    importance.to_csv(importance_path, encoding="utf-8-sig")
    candidates_path = output_dir / "v19_inner_blend_candidates.csv"
    candidates.to_csv(candidates_path, index=False, encoding="utf-8-sig")
    comparison_path = output_dir / "v19_outer_comparison.csv"
    pd.DataFrame(
        [
            {"model": "v15_baseline", **v15_metrics},
            {"model": "v19_soft_behavior_probability_blend", **v19_metrics},
        ]
    ).to_csv(comparison_path, index=False, encoding="utf-8-sig")
    probability_diagnostic = pd.DataFrame(
        {
            "actual": labels.iloc[outer_valid_pos].reset_index(drop=True),
            "behavior_probability": behavior_outer_probability,
        }
    )
    diagnostic_path = output_dir / "v19_outer_behavior_probability_by_label.csv"
    probability_diagnostic.groupby("actual")["behavior_probability"].agg(
        ["count", "min", "median", "mean", "max"]
    ).to_csv(diagnostic_path, encoding="utf-8-sig")
    manifest = {
        "removed_duplicate_behavior_feature": "dst_port_code",
        "retained_port_feature": "protocol_x_dst_port",
        "behavior_features": V19_BEHAVIOR_FEATURES,
        "selected_behavior_positive_weight": selected_weight,
        "selected_blend_alpha": selected_alpha,
        "selected_infilteration_final_multiplier": selected_multiplier,
        "outer_v15_metrics": v15_metrics,
        "outer_v19_metrics": v19_metrics,
        "source_v15_run_id": SOURCE_V15_RUN_ID,
    }
    manifest_path = output_dir / "v19_validation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    improved = v19_metrics["weighted_f1_attacks"] > v15_metrics["weighted_f1_attacks"]
    status = "full_refit_candidate" if improved else "rejected_on_outer_validation"
    with hypothesis_run(
        run_name="v19_behavior_probability_soft_blend_validation",
        hypothesis_id=HYPOTHESIS_ID,
        hypothesis=HYPOTHESIS,
        stage="chronological_validation",
        promotion_status=status,
        validation_strategy="inner_blend_selection_outer_chronological_evaluation",
        notebook="scripts/v19_behavior_probability_blend_validation.py",
        data_version=raw_path.name,
        feature_schema_version="pruned-behavior-no-duplicate-dstport-v1",
        code_version=file_sha256(Path(__file__)),
        extra_tags={
            "source_v15_run_id": SOURCE_V15_RUN_ID,
            "soft_blend": "true",
            "absolute_timestamp_features": "false",
            "outer_labels_used_for_selection": "false",
        },
    ) as run:
        mlflow.log_params(
            {
                "selected_behavior_positive_weight": selected_weight,
                "selected_blend_alpha": selected_alpha,
                "selected_infilteration_final_multiplier": selected_multiplier,
                "behavior_feature_count": behavior.shape[1],
                "removed_duplicate_behavior_feature": "dst_port_code",
                "source_v15_run_id": SOURCE_V15_RUN_ID,
            }
        )
        mlflow.log_metrics(
            {
                **{f"outer_v19_{key}": float(value) for key, value in v19_metrics.items()},
                **{f"outer_v15_{key}": float(value) for key, value in v15_metrics.items()},
                "delta_weighted_f1_attacks": v19_metrics["weighted_f1_attacks"]
                - v15_metrics["weighted_f1_attacks"],
                "delta_infilteration_f1": v19_metrics["infilteration_f1"]
                - v15_metrics["infilteration_f1"],
                "elapsed_seconds": time.time() - started,
            }
        )
        for artifact in (
            importance_path,
            candidates_path,
            comparison_path,
            diagnostic_path,
            manifest_path,
        ):
            mlflow.log_artifact(str(artifact))
        input_example = behavior.iloc[outer_valid_pos[:5]].copy()
        model_info = mlflow.sklearn.log_model(
            behavior_model,
            name="behavior_probability_model",
            input_example=input_example,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            metadata={
                "selected_blend_alpha": selected_alpha,
                "selected_infilteration_final_multiplier": selected_multiplier,
                "source_v15_model_uri": SOURCE_V15_MODEL_URI,
            },
        )
        run_id = run.info.run_id
        model_uri = model_info.model_uri

    print("Validation run:", run_id)
    print("Behavior probability model:", model_uri)
    print("Status:", status)


if __name__ == "__main__":
    main()
