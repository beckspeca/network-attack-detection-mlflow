#!/usr/bin/env python3
"""Validate pruned behavior features and report all RF feature importances."""

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
from v17b_direct_behavior_multiclass_validation import build_rf, predict_adjusted  # noqa: E402


HYPOTHESIS_ID = "v18"
HYPOTHESIS = "v15_plus_pruned_useful_behavior_features_importance_review"
SOURCE_V15_RUN_ID = "4971c300af7f490bbdcd1de7606eccde"
SOURCE_V15_MODEL_URI = "models:/m-b063a6360c3246aaa77c08454ec78a3f"
INNER_VALID_FRACTION = 0.10
INF_WEIGHT_MULTIPLIERS = (0.5, 0.75, 1.0, 1.25, 1.6, 2.0)
PROBABILITY_MULTIPLIERS = (0.5, 0.75, 1.0, 1.1, 1.225, 1.35, 1.5)

# Explicitly removed at the user's request:
# active_to_idle_ratio, tcp_syn_present, tcp_syn_ack_joint, tcp_fin_present.
# Low-value single ACK and raw flag-sum indicators are also excluded; the TCP
# bitmask and RST indicator retain the useful flag interaction signal.
PRUNED_BEHAVIOR_FEATURES = [
    "bwd_to_fwd_packet_ratio",
    "bwd_to_fwd_byte_ratio",
    "bytes_per_packet",
    "tcp_flag_bitmask",
    "tcp_rst_present",
    "protocol_code",
    "dst_port_code",
    "protocol_x_dst_port",
    "packets_per_flow_duration",
    "total_packets",
    "total_bytes",
    "flow_duration",
    "log1p__bwd_to_fwd_packet_ratio",
    "log1p__bwd_to_fwd_byte_ratio",
    "log1p__bytes_per_packet",
    "log1p__packets_per_flow_duration",
    "log1p__total_packets",
    "log1p__total_bytes",
    "log1p__flow_duration",
]


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

        connection.register("v18_ids", metadata[[ID_COL]])
        source_columns = [column for column in BEHAVIOR_SOURCE_COLUMNS if column in available]
        source_quoted = ", ".join(f'raw."{column}"' for column in source_columns)
        raw_sample = connection.execute(
            f"SELECT {source_quoted} FROM read_parquet('{raw_path}') raw "
            f"INNER JOIN v18_ids ids USING ({ID_COL})"
        ).df()

    raw_sample = raw_sample.set_index(ID_COL).loc[metadata[ID_COL]].reset_index()
    all_behavior = build_behavior_features(raw_sample)
    missing = [column for column in PRUNED_BEHAVIOR_FEATURES if column not in all_behavior]
    if missing:
        raise RuntimeError(f"Missing requested pruned behavior features: {missing}")
    behavior = all_behavior[PRUNED_BEHAVIOR_FEATURES].copy()
    behavior.columns = [f"behavior__{column}" for column in behavior.columns]
    features = pd.concat([v15_features, behavior], axis=1)
    del raw_sample, all_behavior, behavior
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

    candidate_rows: list[dict[str, float]] = []
    for inf_weight in INF_WEIGHT_MULTIPLIERS:
        model = build_rf(encoded[inner_train_pos], len(classes), inf_index, inf_weight)
        model.fit(features.iloc[inner_train_pos], encoded[inner_train_pos])
        for probability_multiplier in PROBABILITY_MULTIPLIERS:
            prediction = predict_adjusted(
                model, features.iloc[inner_valid_pos], classes, probability_multiplier
            )
            metrics = evaluate(labels.iloc[inner_valid_pos], prediction, attack_classes)
            candidate_rows.append(
                {
                    "infilteration_class_weight_multiplier": inf_weight,
                    "infilteration_probability_multiplier": probability_multiplier,
                    **metrics,
                }
            )
        del model
        gc.collect()

    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["weighted_f1_attacks", "infilteration_f1", "infilteration_precision"],
        ascending=False,
    ).reset_index(drop=True)
    selected = candidates.iloc[0]
    selected_weight = float(selected["infilteration_class_weight_multiplier"])
    selected_probability = float(selected["infilteration_probability_multiplier"])
    print("Selected v18 settings:", selected.to_dict())

    model = build_rf(encoded[outer_train_pos], len(classes), inf_index, selected_weight)
    model.fit(features.iloc[outer_train_pos], encoded[outer_train_pos])
    prediction = predict_adjusted(
        model, features.iloc[outer_valid_pos], classes, selected_probability
    )
    v18_metrics = evaluate(labels.iloc[outer_valid_pos], prediction, attack_classes)

    configure_tracking(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"), PROJECT_EXPERIMENT)
    v15_model = mlflow.sklearn.load_model(SOURCE_V15_MODEL_URI)
    v15_prediction = v15_model.predict(v15_features.iloc[outer_valid_pos])
    v15_metrics = evaluate(labels.iloc[outer_valid_pos], v15_prediction, attack_classes)
    print("Outer v15 baseline:", v15_metrics)
    print("Outer v18 result:", v18_metrics)

    importance = pd.DataFrame(
        {
            "feature": features.columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    temporal_columns = set(c for c in v15_features.columns if c not in selected_v15)
    importance["feature_group"] = np.select(
        [
            importance["feature"].str.startswith("behavior__"),
            importance["feature"].isin(temporal_columns),
        ],
        ["behavior", "temporal"],
        default="general_flow",
    )
    importance["rank"] = np.arange(1, len(importance) + 1)
    importance = importance[["rank", "feature_group", "feature", "importance"]]
    group_importance = (
        importance.groupby("feature_group", as_index=False)
        .agg(feature_count=("feature", "size"), importance=("importance", "sum"))
        .sort_values("importance", ascending=False)
    )
    general_importance = importance.loc[importance["feature_group"] == "general_flow"]
    behavior_importance = importance.loc[importance["feature_group"] == "behavior"]

    all_importance_path = output_dir / "v18_all_feature_importance.csv"
    general_importance_path = output_dir / "v18_general_flow_feature_importance.csv"
    behavior_importance_path = output_dir / "v18_pruned_behavior_feature_importance.csv"
    group_importance_path = output_dir / "v18_feature_group_importance.csv"
    importance.to_csv(all_importance_path, index=False, encoding="utf-8-sig")
    general_importance.to_csv(general_importance_path, index=False, encoding="utf-8-sig")
    behavior_importance.to_csv(behavior_importance_path, index=False, encoding="utf-8-sig")
    group_importance.to_csv(group_importance_path, index=False, encoding="utf-8-sig")
    print("Feature group importance:\n", group_importance.to_string(index=False))
    print("Top general features:\n", general_importance.head(15).to_string(index=False))

    candidates_path = output_dir / "v18_inner_weight_probability_candidates.csv"
    candidates.to_csv(candidates_path, index=False, encoding="utf-8-sig")
    comparison_path = output_dir / "v18_outer_comparison.csv"
    pd.DataFrame(
        [
            {"model": "v15_baseline", **v15_metrics},
            {"model": "v18_pruned_behavior_multiclass", **v18_metrics},
        ]
    ).to_csv(comparison_path, index=False, encoding="utf-8-sig")
    manifest = {
        "removed_features": [
            "active_to_idle_ratio",
            "log1p__active_to_idle_ratio",
            "tcp_syn_present",
            "tcp_syn_ack_joint",
            "tcp_fin_present",
            "tcp_ack_present",
            "tcp_flag_sum",
            "log1p__tcp_flag_sum",
        ],
        "retained_behavior_features": PRUNED_BEHAVIOR_FEATURES,
        "selected_infilteration_class_weight_multiplier": selected_weight,
        "selected_infilteration_probability_multiplier": selected_probability,
        "outer_v15_metrics": v15_metrics,
        "outer_v18_metrics": v18_metrics,
        "feature_group_importance": group_importance.set_index("feature_group")["importance"].to_dict(),
        "source_v15_run_id": SOURCE_V15_RUN_ID,
    }
    manifest_path = output_dir / "v18_validation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    improved = v18_metrics["weighted_f1_attacks"] > v15_metrics["weighted_f1_attacks"]
    status = "full_refit_candidate" if improved else "rejected_on_outer_validation"
    adjusted_model = ProbabilityAdjustedClassifier(
        base_model=model,
        label_names=classes,
        target_class_index=inf_index,
        multiplier=selected_probability,
    )
    with hypothesis_run(
        run_name="v18_pruned_behavior_importance_validation",
        hypothesis_id=HYPOTHESIS_ID,
        hypothesis=HYPOTHESIS,
        stage="chronological_validation",
        promotion_status=status,
        validation_strategy="inner_weight_probability_selection_outer_chronological_evaluation",
        notebook="scripts/v18_pruned_behavior_importance_validation.py",
        data_version=raw_path.name,
        feature_schema_version="v15-plus-pruned-row-behavior-v1",
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
                "general_flow_feature_count": int((importance.feature_group == "general_flow").sum()),
                "temporal_feature_count": int((importance.feature_group == "temporal").sum()),
                "behavior_feature_count": int((importance.feature_group == "behavior").sum()),
                "total_feature_count": len(importance),
                "source_v15_run_id": SOURCE_V15_RUN_ID,
            }
        )
        mlflow.log_metrics(
            {
                **{f"outer_v18_{key}": float(value) for key, value in v18_metrics.items()},
                **{f"outer_v15_{key}": float(value) for key, value in v15_metrics.items()},
                **{
                    f"importance_group_{row.feature_group}": float(row.importance)
                    for row in group_importance.itertuples()
                },
                "delta_weighted_f1_attacks": v18_metrics["weighted_f1_attacks"]
                - v15_metrics["weighted_f1_attacks"],
                "delta_infilteration_f1": v18_metrics["infilteration_f1"]
                - v15_metrics["infilteration_f1"],
                "elapsed_seconds": time.time() - started,
            }
        )
        for artifact in (
            all_importance_path,
            general_importance_path,
            behavior_importance_path,
            group_importance_path,
            candidates_path,
            comparison_path,
            manifest_path,
        ):
            mlflow.log_artifact(str(artifact))
        input_example = features.iloc[outer_valid_pos[:5]].copy()
        model_info = mlflow.sklearn.log_model(
            adjusted_model,
            name="pruned_behavior_multiclass_model",
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
    print("Pruned behavior model:", model_uri)
    print("Status:", status)


if __name__ == "__main__":
    main()
