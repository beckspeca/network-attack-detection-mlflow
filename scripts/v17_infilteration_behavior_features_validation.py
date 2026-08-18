#!/usr/bin/env python3
"""Validate targeted row-behavior features for Infilteration verification.

The feature family is intentionally small and clock-free:
backward/forward packet and byte ratios, bytes per packet, TCP flag
combinations, Protocol x destination-port interaction, active/idle ratio, and
packets per flow duration.  A verifier only filters v15 Infilteration
candidates.  Selection uses grouped OOF predictions; the chronological outer
holdout is evaluated once.
"""

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
from sklearn.model_selection import StratifiedGroupKFold


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from attached_temporal_features import ROLLING_SOURCE_COLUMNS, build_attachment_feature_frame  # noqa: E402
from mlflow_workflow import PROJECT_EXPERIMENT, configure_tracking, file_sha256, hypothesis_run  # noqa: E402
from v16_candidate_meta_verifier_validation import augment_with_v15_meta, v15_outputs  # noqa: E402
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
    filter_infilteration,
    resolve_dataset_paths,
)


HYPOTHESIS_ID = "v17"
HYPOTHESIS = "targeted_infilteration_row_behavior_feature_verifier"
SOURCE_V15_RUN_ID = "4971c300af7f490bbdcd1de7606eccde"
SOURCE_V15_MODEL_URI = "models:/m-b063a6360c3246aaa77c08454ec78a3f"
INNER_VALID_FRACTION = 0.10
POSITIVE_WEIGHTS = (1.0, 2.0, 4.0, 8.0, 16.0)
FEATURE_SETS = ("behavior_only", "behavior_plus_v15_meta")
THRESHOLDS = np.round(np.linspace(0.0, 1.0, 101), 4)

BEHAVIOR_SOURCE_COLUMNS = [
    ID_COL,
    "Protocol",
    "Dst Port",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "SYN Flag Cnt",
    "ACK Flag Cnt",
    "RST Flag Cnt",
    "FIN Flag Cnt",
    "Active Mean",
    "Idle Mean",
    "Flow Duration",
]


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float)


def ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / (denominator.abs() + 1.0)


def signed_log1p(values: pd.Series) -> pd.Series:
    array = np.asarray(values, dtype=float)
    return pd.Series(np.sign(array) * np.log1p(np.abs(array)), index=values.index)


def build_behavior_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Build the seven requested behavior concepts and stable transforms."""
    protocol = numeric(raw, "Protocol")
    port_values = raw["Dst Port"].astype("string")
    port_categories = sorted(port_values.dropna().unique().tolist())
    port_map = {value: index for index, value in enumerate(port_categories)}
    port_code = port_values.map(port_map).fillna(-1).astype(float)
    fwd_packets = numeric(raw, "Tot Fwd Pkts")
    bwd_packets = numeric(raw, "Tot Bwd Pkts")
    fwd_bytes = numeric(raw, "TotLen Fwd Pkts")
    bwd_bytes = numeric(raw, "TotLen Bwd Pkts")
    syn = numeric(raw, "SYN Flag Cnt")
    ack = numeric(raw, "ACK Flag Cnt")
    rst = numeric(raw, "RST Flag Cnt")
    fin = numeric(raw, "FIN Flag Cnt")
    active = numeric(raw, "Active Mean")
    idle = numeric(raw, "Idle Mean")
    duration = numeric(raw, "Flow Duration")

    packet_ratio = ratio(bwd_packets, fwd_packets)
    byte_ratio = ratio(bwd_bytes, fwd_bytes)
    total_packets = fwd_packets + bwd_packets
    total_bytes = fwd_bytes + bwd_bytes
    bytes_per_packet = ratio(total_bytes, total_packets)
    flag_sum = syn + ack + rst + fin
    # Bitmask keeps the flag combination distinct instead of treating equal sums as equal.
    flag_bitmask = (
        (syn > 0).astype(np.int16)
        + 2 * (ack > 0).astype(np.int16)
        + 4 * (rst > 0).astype(np.int16)
        + 8 * (fin > 0).astype(np.int16)
    )
    protocol_port = protocol * (len(port_categories) + 1) + (port_code + 1)
    active_idle_ratio = ratio(active, idle)
    packets_per_duration = ratio(total_packets, duration)

    features = pd.DataFrame(
        {
            "bwd_to_fwd_packet_ratio": packet_ratio,
            "bwd_to_fwd_byte_ratio": byte_ratio,
            "bytes_per_packet": bytes_per_packet,
            "tcp_flag_sum": flag_sum,
            "tcp_flag_bitmask": flag_bitmask,
            "tcp_syn_present": (syn > 0).astype(np.int8),
            "tcp_ack_present": (ack > 0).astype(np.int8),
            "tcp_rst_present": (rst > 0).astype(np.int8),
            "tcp_fin_present": (fin > 0).astype(np.int8),
            "tcp_syn_ack_joint": ((syn > 0) & (ack > 0)).astype(np.int8),
            "protocol_code": protocol,
            "dst_port_code": port_code,
            "protocol_x_dst_port": protocol_port,
            "active_to_idle_ratio": active_idle_ratio,
            "packets_per_flow_duration": packets_per_duration,
            "total_packets": total_packets,
            "total_bytes": total_bytes,
            "flow_duration": duration,
        }
    )
    for column in [
        "bwd_to_fwd_packet_ratio",
        "bwd_to_fwd_byte_ratio",
        "bytes_per_packet",
        "tcp_flag_sum",
        "active_to_idle_ratio",
        "packets_per_flow_duration",
        "total_packets",
        "total_bytes",
        "flow_duration",
    ]:
        features[f"log1p__{column}"] = signed_log1p(features[column])
    return (
        features.replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
    )


def build_verifier() -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=220,
        learning_rate=0.035,
        num_leaves=15,
        min_child_samples=25,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.8,
        reg_lambda=3.0,
        n_jobs=4,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )


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

        connection.register("v17_ids", metadata[[ID_COL]])
        behavior_columns = [column for column in BEHAVIOR_SOURCE_COLUMNS if column in available]
        behavior_quoted = ", ".join(f'raw."{column}"' for column in behavior_columns)
        raw_sample = connection.execute(
            f"SELECT {behavior_quoted} FROM read_parquet('{raw_path}') raw "
            f"INNER JOIN v17_ids ids USING ({ID_COL})"
        ).df()

    raw_sample = raw_sample.set_index(ID_COL).loc[metadata[ID_COL]].reset_index()
    behavior_features = build_behavior_features(raw_sample)
    del raw_sample
    gc.collect()

    labels = metadata[TARGET].astype(str)
    classes = sorted(labels.unique().tolist())
    attack_classes = [label for label in classes if label != "Benign"]
    encoded = labels.map({label: index for index, label in enumerate(classes)}).to_numpy(np.int64)
    outer_train, outer_valid = class_chronological_masks(metadata, OUTER_VALID_FRACTION)
    inner_train, inner_valid = class_chronological_masks(
        metadata, INNER_VALID_FRACTION, eligible=outer_train
    )
    inner_train_pos = np.flatnonzero(inner_train.to_numpy())
    inner_valid_pos = np.flatnonzero(inner_valid.to_numpy())
    outer_valid_pos = np.flatnonzero(outer_valid.to_numpy())

    inner_model = build_v15_base(encoded[inner_train_pos], classes)
    inner_model.fit(v15_features.iloc[inner_train_pos], encoded[inner_train_pos])
    inner_prediction, inner_fallback, inner_probability, inner_scores = v15_outputs(
        inner_model, v15_features.iloc[inner_valid_pos], classes
    )
    inner_labels = labels.iloc[inner_valid_pos].reset_index(drop=True)
    inner_base_metrics = evaluate(inner_labels, inner_prediction, attack_classes)
    inner_candidate = inner_prediction == INFILTERATION
    candidate_indices = np.flatnonzero(inner_candidate)
    candidate_target = inner_labels.iloc[candidate_indices].eq(INFILTERATION).astype(np.int8).to_numpy()
    print("Inner baseline:", inner_base_metrics)
    print(
        "Inner candidates / true / false:",
        int(inner_candidate.sum()),
        int(candidate_target.sum()),
        int(len(candidate_target) - candidate_target.sum()),
    )

    inner_behavior = behavior_features.iloc[inner_valid_pos].reset_index(drop=True)
    inner_feature_sets = {
        "behavior_only": inner_behavior,
        "behavior_plus_v15_meta": augment_with_v15_meta(
            inner_behavior, inner_probability, inner_scores, classes
        ),
    }
    fingerprint_groups = pd.util.hash_pandas_object(
        inner_behavior.iloc[candidate_indices], index=False
    ).to_numpy(dtype=np.uint64)
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    result_rows: list[dict[str, object]] = []

    for feature_set in FEATURE_SETS:
        candidate_frame = inner_feature_sets[feature_set].iloc[candidate_indices].reset_index(drop=True)
        for positive_weight in POSITIVE_WEIGHTS:
            oof_probability = np.zeros(len(candidate_indices), dtype=float)
            for train_local, valid_local in splitter.split(
                candidate_frame, candidate_target, groups=fingerprint_groups
            ):
                verifier = build_verifier()
                verifier.fit(
                    candidate_frame.iloc[train_local],
                    candidate_target[train_local],
                    sample_weight=np.where(
                        candidate_target[train_local] == 1, positive_weight, 1.0
                    ),
                )
                oof_probability[valid_local] = verifier.predict_proba(
                    candidate_frame.iloc[valid_local]
                )[:, 1]
            full_probability = np.ones(len(inner_labels), dtype=float)
            full_probability[candidate_indices] = oof_probability
            for threshold in THRESHOLDS:
                prediction = filter_infilteration(
                    inner_prediction, inner_fallback, full_probability, float(threshold)
                )
                metrics = evaluate(inner_labels, prediction, attack_classes)
                result_rows.append(
                    {
                        "feature_set": feature_set,
                        "positive_weight": positive_weight,
                        "threshold": float(threshold),
                        **metrics,
                    }
                )

    candidates = pd.DataFrame(result_rows).sort_values(
        ["weighted_f1_attacks", "infilteration_f1", "infilteration_precision", "threshold"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    selected = candidates.iloc[0]
    feature_set = str(selected["feature_set"])
    positive_weight = float(selected["positive_weight"])
    threshold = float(selected["threshold"])
    print("Selected targeted verifier:", selected.to_dict())

    selected_inner_frame = inner_feature_sets[feature_set].iloc[candidate_indices].reset_index(drop=True)
    verifier = build_verifier()
    verifier.fit(
        selected_inner_frame,
        candidate_target,
        sample_weight=np.where(candidate_target == 1, positive_weight, 1.0),
    )

    configure_tracking(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"), PROJECT_EXPERIMENT)
    v15_model = mlflow.sklearn.load_model(SOURCE_V15_MODEL_URI)
    outer_prediction, outer_fallback, outer_probability, outer_scores = v15_outputs(
        v15_model.base_model, v15_features.iloc[outer_valid_pos], classes
    )
    outer_labels = labels.iloc[outer_valid_pos].reset_index(drop=True)
    outer_base_metrics = evaluate(outer_labels, outer_prediction, attack_classes)
    outer_behavior = behavior_features.iloc[outer_valid_pos].reset_index(drop=True)
    outer_frame = (
        outer_behavior
        if feature_set == "behavior_only"
        else augment_with_v15_meta(outer_behavior, outer_probability, outer_scores, classes)
    )
    outer_candidate = outer_prediction == INFILTERATION
    verifier_probability = np.ones(len(outer_labels), dtype=float)
    verifier_probability[outer_candidate] = verifier.predict_proba(
        outer_frame.loc[outer_candidate]
    )[:, 1]
    v17_prediction = filter_infilteration(
        outer_prediction, outer_fallback, verifier_probability, threshold
    )
    outer_metrics = evaluate(outer_labels, v17_prediction, attack_classes)
    print("Outer v15 baseline:", outer_base_metrics)
    print("Outer v17 result:", outer_metrics)

    importance = pd.Series(
        verifier.feature_importances_, index=selected_inner_frame.columns, name="importance"
    ).sort_values(ascending=False)
    importance_path = output_dir / "v17_behavior_feature_importance.csv"
    importance.to_csv(importance_path, encoding="utf-8-sig")
    candidates_path = output_dir / "v17_inner_grouped_oof_candidates.csv"
    candidates.to_csv(candidates_path, index=False, encoding="utf-8-sig")
    comparison_path = output_dir / "v17_outer_comparison.csv"
    pd.DataFrame(
        [
            {"model": "v15_baseline", **outer_base_metrics},
            {"model": "v17_targeted_behavior_verifier", **outer_metrics},
        ]
    ).to_csv(comparison_path, index=False, encoding="utf-8-sig")
    manifest = {
        "requested_behavior_concepts": [
            "Tot Bwd Pkts / (Tot Fwd Pkts + 1)",
            "TotLen Bwd Pkts / (TotLen Fwd Pkts + 1)",
            "bytes per packet",
            "SYN + ACK + RST + FIN combinations",
            "Protocol x Dst Port",
            "Active / (Idle + 1)",
            "packets / (Flow Duration + 1)",
        ],
        "selected_feature_set": feature_set,
        "selected_positive_weight": positive_weight,
        "selected_threshold": threshold,
        "feature_count": int(selected_inner_frame.shape[1]),
        "features": selected_inner_frame.columns.tolist(),
        "outer_base_metrics": outer_base_metrics,
        "outer_v17_metrics": outer_metrics,
        "source_v15_run_id": SOURCE_V15_RUN_ID,
    }
    manifest_path = output_dir / "v17_validation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    improved = outer_metrics["weighted_f1_attacks"] > outer_base_metrics["weighted_f1_attacks"]
    status = "full_refit_candidate" if improved else "rejected_on_outer_validation"
    with hypothesis_run(
        run_name="v17_targeted_infilteration_behavior_features_validation",
        hypothesis_id=HYPOTHESIS_ID,
        hypothesis=HYPOTHESIS,
        stage="chronological_validation",
        promotion_status=status,
        validation_strategy="inner_candidate_grouped_oof_threshold_outer_chronological_evaluation",
        notebook="scripts/v17_infilteration_behavior_features_validation.py",
        data_version=raw_path.name,
        feature_schema_version="targeted-row-behavior-v1",
        code_version=file_sha256(Path(__file__)),
        extra_tags={
            "source_v15_run_id": SOURCE_V15_RUN_ID,
            "filter_only": "true",
            "absolute_timestamp_features": "false",
            "outer_labels_used_for_selection": "false",
        },
    ) as run:
        mlflow.log_params(
            {
                "selected_feature_set": feature_set,
                "selected_positive_weight": positive_weight,
                "selected_threshold": threshold,
                "feature_count": selected_inner_frame.shape[1],
                "inner_candidate_count": int(inner_candidate.sum()),
                "inner_candidate_true_count": int(candidate_target.sum()),
                "outer_candidate_count": int(outer_candidate.sum()),
                "source_v15_run_id": SOURCE_V15_RUN_ID,
            }
        )
        mlflow.log_metrics(
            {
                **{f"outer_v17_{key}": float(value) for key, value in outer_metrics.items()},
                **{f"outer_v15_{key}": float(value) for key, value in outer_base_metrics.items()},
                **{f"inner_v15_{key}": float(value) for key, value in inner_base_metrics.items()},
                "delta_weighted_f1_attacks": outer_metrics["weighted_f1_attacks"]
                - outer_base_metrics["weighted_f1_attacks"],
                "delta_infilteration_f1": outer_metrics["infilteration_f1"]
                - outer_base_metrics["infilteration_f1"],
                "elapsed_seconds": time.time() - started,
            }
        )
        for artifact in (importance_path, candidates_path, comparison_path, manifest_path):
            mlflow.log_artifact(str(artifact))
        input_example = selected_inner_frame.iloc[:5].copy()
        model_info = mlflow.sklearn.log_model(
            verifier,
            name="targeted_behavior_verifier",
            input_example=input_example,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            metadata={
                "filter_only": True,
                "selected_threshold": threshold,
                "source_v15_model_uri": SOURCE_V15_MODEL_URI,
            },
        )
        run_id = run.info.run_id
        model_uri = model_info.model_uri

    print("Validation run:", run_id)
    print("Behavior verifier model:", model_uri)
    print("Status:", status)


if __name__ == "__main__":
    main()
