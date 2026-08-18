#!/usr/bin/env python3
"""Validate a v16 Infilteration hard-negative verifier on top of v15.

The v15 multiclass prediction is kept unless it predicts Infilteration.  A
binary verifier built from all row-level flow columns may then reject that
prediction and return the best non-Infilteration v15 class.  Threshold and
specialist settings are selected on an inner chronological split and are
evaluated once on the untouched outer chronological split.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from attached_temporal_features import (  # noqa: E402
    ROLLING_SOURCE_COLUMNS,
    build_attachment_feature_frame,
)
from mlflow_workflow import (  # noqa: E402
    PROJECT_EXPERIMENT,
    configure_tracking,
    file_sha256,
    hypothesis_run,
)


HYPOTHESIS_ID = "v16"
HYPOTHESIS = "v15_plus_infilteration_full_flow_hard_negative_verifier"
RANDOM_STATE = 42
TRAIN_SAMPLE = 250_000
OUTER_VALID_FRACTION = 0.20
INNER_VALID_FRACTION = 0.10
MINING_VALID_FRACTION = 0.20
INFILTERATION_MULTIPLIER = 1.225
SOURCE_V15_RUN_ID = "4971c300af7f490bbdcd1de7606eccde"
SOURCE_V15_MODEL_URI = "models:/m-b063a6360c3246aaa77c08454ec78a3f"

ID_COL = "unique_id"
TARGET = "Label"
TIMESTAMP = "Timestamp"
INFILTERATION = "Infilteration"
REQUESTED_V15_FLOW_FEATURES = [
    "Flow Duration",
    "Flow IAT Min",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Pkt Len Std",
    "Pkt Size Avg",
    "Init Fwd Win Byts",
    "Subflow Fwd Byts",
    "ACK Flag Cnt",
    "SYN Flag Cnt",
    "PSH Flag Cnt",
    "Dst Port",
    "TotLen Fwd Pkts",
    "Active Mean",
]
POSITIVE_WEIGHTS = (1.0, 2.0, 4.0, 8.0)
FEATURE_SETS = ("raw", "raw_plus_context")
THRESHOLDS = np.round(np.linspace(0.0, 1.0, 81), 4)
CAMPAIGN_DATES = {pd.Timestamp("2018-02-28").date(), pd.Timestamp("2018-03-01").date()}


def resolve_dataset_paths() -> tuple[Path, Path]:
    base = Path("/home/jovyan/work/datasets")
    raw = sorted(base.glob("*/interim/train.parquet"))
    cohort = sorted(base.glob("*/processed/train_temporal_v5.parquet"))
    if len(raw) != 1 or len(cohort) != 1:
        raise RuntimeError(f"Dataset resolution failed: raw={raw}, cohort={cohort}")
    return raw[0], cohort[0]


def sanitized_columns(columns: list[str]) -> list[str]:
    output: list[str] = []
    used: set[str] = set()
    for column in columns:
        name = (
            column.strip()
            .replace("/", "_per_")
            .replace(" ", "_")
            .replace("[", "_")
            .replace("]", "_")
            .replace("{", "_")
            .replace("}", "_")
            .replace(":", "_")
            .replace(",", "_")
        )
        candidate = name
        suffix = 2
        while candidate in used:
            candidate = f"{name}_{suffix}"
            suffix += 1
        used.add(candidate)
        output.append(candidate)
    return output


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return pd.to_numeric(numerator, errors="coerce") / (
        pd.to_numeric(denominator, errors="coerce").abs() + 1.0
    )


def build_specialist_raw_features(raw_sample: pd.DataFrame) -> pd.DataFrame:
    """Use every available row-level feature while excluding ID, label and clock."""
    excluded = {ID_COL, TARGET, TIMESTAMP}
    frame = raw_sample.loc[:, [c for c in raw_sample.columns if c not in excluded]].copy()
    if "Dst Port" in frame:
        categories = sorted(frame["Dst Port"].astype("string").dropna().unique().tolist())
        mapping = {value: index for index, value in enumerate(categories)}
        frame["Dst Port"] = frame["Dst Port"].astype("string").map(mapping).fillna(-1)
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    fwd_packets = frame.get("Tot Fwd Pkts", pd.Series(0.0, index=frame.index))
    bwd_packets = frame.get("Tot Bwd Pkts", pd.Series(0.0, index=frame.index))
    fwd_bytes = frame.get("TotLen Fwd Pkts", pd.Series(0.0, index=frame.index))
    bwd_bytes = frame.get("TotLen Bwd Pkts", pd.Series(0.0, index=frame.index))
    duration = frame.get("Flow Duration", pd.Series(0.0, index=frame.index))
    active = frame.get("Active Mean", pd.Series(0.0, index=frame.index))
    idle = frame.get("Idle Mean", pd.Series(0.0, index=frame.index))
    syn = frame.get("SYN Flag Cnt", pd.Series(0.0, index=frame.index))
    ack = frame.get("ACK Flag Cnt", pd.Series(0.0, index=frame.index))
    rst = frame.get("RST Flag Cnt", pd.Series(0.0, index=frame.index))
    fin = frame.get("FIN Flag Cnt", pd.Series(0.0, index=frame.index))
    psh = frame.get("PSH Flag Cnt", pd.Series(0.0, index=frame.index))

    frame["row_fwd_bwd_packet_ratio"] = safe_ratio(fwd_packets, bwd_packets)
    frame["row_fwd_bwd_byte_ratio"] = safe_ratio(fwd_bytes, bwd_bytes)
    frame["row_bwd_fwd_packet_ratio"] = safe_ratio(bwd_packets, fwd_packets)
    frame["row_bwd_fwd_byte_ratio"] = safe_ratio(bwd_bytes, fwd_bytes)
    frame["row_bytes_per_packet"] = safe_ratio(fwd_bytes + bwd_bytes, fwd_packets + bwd_packets)
    frame["row_packets_per_duration"] = safe_ratio(fwd_packets + bwd_packets, duration)
    frame["row_active_idle_ratio"] = safe_ratio(active, idle)
    frame["row_no_backward_packets"] = (pd.to_numeric(bwd_packets, errors="coerce") == 0).astype(float)
    frame["row_no_backward_bytes"] = (pd.to_numeric(bwd_bytes, errors="coerce") == 0).astype(float)
    frame["row_syn_without_backward"] = (
        (pd.to_numeric(syn, errors="coerce") > 0)
        & (pd.to_numeric(bwd_packets, errors="coerce") == 0)
    ).astype(float)
    frame["row_flag_total"] = syn + ack + rst + fin + psh
    frame["row_syn_ack_ratio"] = safe_ratio(syn, ack)
    frame["row_rst_ack_ratio"] = safe_ratio(rst, ack)

    frame.columns = sanitized_columns(frame.columns.tolist())
    return (
        frame.replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
    )


def class_chronological_masks(
    metadata: pd.DataFrame,
    valid_fraction: float,
    eligible: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series]:
    if eligible is None:
        eligible = pd.Series(True, index=metadata.index)
    eligible_meta = metadata.loc[eligible]
    cutoffs = eligible_meta.groupby(TARGET)[TIMESTAMP].quantile(1.0 - valid_fraction)
    valid = eligible & (metadata[TIMESTAMP] > metadata[TARGET].map(cutoffs))
    train = eligible & ~valid
    return train, valid


def build_v15_base(encoded_labels: np.ndarray, classes: list[str]) -> RandomForestClassifier:
    counts = np.bincount(encoded_labels, minlength=len(classes)).astype(float)
    class_weights = {
        index: len(encoded_labels) / (len(classes) * count)
        for index, count in enumerate(counts)
    }
    class_weights[classes.index(INFILTERATION)] *= 1.6
    return RandomForestClassifier(
        n_estimators=100,
        n_jobs=4,
        random_state=RANDOM_STATE,
        class_weight=class_weights,
    )


def multiclass_outputs(base_model, features: pd.DataFrame, classes: list[str]):
    probabilities = np.asarray(base_model.predict_proba(features), dtype=float)
    base_classes = np.asarray(base_model.classes_)
    inf_encoded = classes.index(INFILTERATION)
    inf_column = int(np.flatnonzero(base_classes == inf_encoded)[0])
    scores = probabilities.copy()
    scores[:, inf_column] *= INFILTERATION_MULTIPLIER
    label_names = np.asarray(classes, dtype=object)
    prediction = label_names[base_classes[scores.argmax(axis=1)].astype(int)]
    non_inf_scores = scores.copy()
    non_inf_scores[:, inf_column] = -np.inf
    fallback = label_names[base_classes[non_inf_scores.argmax(axis=1)].astype(int)]
    return prediction, fallback, scores[:, inf_column]


def evaluate(y_true: pd.Series | np.ndarray, prediction: np.ndarray, attack_classes: list[str]):
    y_array = np.asarray(y_true, dtype=object)
    result = {
        "weighted_f1_attacks": float(
            f1_score(y_array, prediction, labels=attack_classes, average="weighted", zero_division=0)
        ),
        "macro_f1_attacks": float(
            f1_score(y_array, prediction, labels=attack_classes, average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(y_array, prediction)),
    }
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_array, prediction, labels=[INFILTERATION], average=None, zero_division=0
    )
    result.update(
        {
            "infilteration_precision": float(precision[0]),
            "infilteration_recall": float(recall[0]),
            "infilteration_f1": float(f1[0]),
            "benign_to_infilteration_fp": int(
                np.sum((y_array == "Benign") & (prediction == INFILTERATION))
            ),
        }
    )
    return result


def hard_negative_weights(
    metadata: pd.DataFrame,
    labels: pd.Series,
    hard_negative_mask: np.ndarray,
    positive_weight: float,
) -> np.ndarray:
    weights = np.ones(len(metadata), dtype=np.float64)
    positive = labels.to_numpy() == INFILTERATION
    dates = metadata[TIMESTAMP].dt.date
    campaign_benign = labels.eq("Benign") & dates.isin(CAMPAIGN_DATES)
    weights[campaign_benign.to_numpy()] = 2.0
    weights[np.asarray(hard_negative_mask, dtype=bool)] = 8.0
    weights[positive] = float(positive_weight)
    return weights


def build_specialist() -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=350,
        learning_rate=0.04,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=1.0,
        n_jobs=4,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )


def filter_infilteration(
    base_prediction: np.ndarray,
    fallback: np.ndarray,
    specialist_probability: np.ndarray,
    threshold: float,
) -> np.ndarray:
    output = np.asarray(base_prediction, dtype=object).copy()
    rejected = (output == INFILTERATION) & (specialist_probability < threshold)
    output[rejected] = fallback[rejected]
    return output


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
        cohort_set = set(cohort_ids.tolist())
        cohort_mask = full[ID_COL].isin(cohort_set)
        metadata = full.loc[cohort_mask, [ID_COL, TIMESTAMP, TARGET]].reset_index(drop=True)
        v15_features = full_v15.loc[cohort_mask].reset_index(drop=True)
        del full_v15, full
        gc.collect()

        id_frame = metadata[[ID_COL]].copy()
        connection.register("v16_cohort_ids", id_frame)
        raw_sample = connection.execute(
            f"SELECT raw.* FROM read_parquet('{raw_path}') raw "
            f"INNER JOIN v16_cohort_ids ids USING ({ID_COL})"
        ).df()

    raw_sample = raw_sample.set_index(ID_COL).loc[metadata[ID_COL]].reset_index()
    specialist_raw = build_specialist_raw_features(raw_sample)
    temporal_columns = [c for c in v15_features.columns if c not in selected_v15]
    context = v15_features.loc[:, temporal_columns].copy().astype(np.float32)
    context.columns = [f"context__{c}" for c in context.columns]
    specialist_features = {
        "raw": specialist_raw,
        "raw_plus_context": pd.concat([specialist_raw, context], axis=1),
    }
    del raw_sample
    gc.collect()

    labels = metadata[TARGET].astype(str)
    classes = sorted(labels.unique().tolist())
    attack_classes = [label for label in classes if label != "Benign"]
    class_to_index = {label: index for index, label in enumerate(classes)}
    encoded = labels.map(class_to_index).to_numpy(dtype=np.int64)

    outer_train, outer_valid = class_chronological_masks(metadata, OUTER_VALID_FRACTION)
    inner_train, inner_valid = class_chronological_masks(
        metadata, INNER_VALID_FRACTION, eligible=outer_train
    )
    mining_train, mining_valid = class_chronological_masks(
        metadata, MINING_VALID_FRACTION, eligible=inner_train
    )
    positions = {
        "outer_train": np.flatnonzero(outer_train.to_numpy()),
        "outer_valid": np.flatnonzero(outer_valid.to_numpy()),
        "inner_train": np.flatnonzero(inner_train.to_numpy()),
        "inner_valid": np.flatnonzero(inner_valid.to_numpy()),
        "mining_train": np.flatnonzero(mining_train.to_numpy()),
        "mining_valid": np.flatnonzero(mining_valid.to_numpy()),
    }
    print("Split rows:", {key: len(value) for key, value in positions.items()})

    mining_model = build_v15_base(encoded[positions["mining_train"]], classes)
    mining_model.fit(
        v15_features.iloc[positions["mining_train"]],
        encoded[positions["mining_train"]],
    )
    mining_prediction, _, _ = multiclass_outputs(
        mining_model, v15_features.iloc[positions["mining_valid"]], classes
    )
    mining_labels = labels.iloc[positions["mining_valid"]].to_numpy()
    hard_inner_global = np.zeros(len(metadata), dtype=bool)
    hard_inner_global[positions["mining_valid"]] = (
        (mining_labels != INFILTERATION) & (mining_prediction == INFILTERATION)
    )
    print("Leak-free inner hard negatives:", int(hard_inner_global.sum()))
    del mining_model
    gc.collect()

    inner_model = build_v15_base(encoded[positions["inner_train"]], classes)
    inner_model.fit(
        v15_features.iloc[positions["inner_train"]],
        encoded[positions["inner_train"]],
    )
    inner_base_prediction, inner_fallback, _ = multiclass_outputs(
        inner_model, v15_features.iloc[positions["inner_valid"]], classes
    )
    inner_base_metrics = evaluate(
        labels.iloc[positions["inner_valid"]], inner_base_prediction, attack_classes
    )
    print("Inner v15 baseline:", inner_base_metrics)

    candidate_rows: list[dict[str, object]] = []
    fitted_candidates: dict[tuple[str, float], LGBMClassifier] = {}
    y_inner_binary = labels.iloc[positions["inner_train"]].eq(INFILTERATION).astype(np.int8)
    inner_metadata = metadata.iloc[positions["inner_train"]].reset_index(drop=True)
    hard_inner = hard_inner_global[positions["inner_train"]]

    for feature_set in FEATURE_SETS:
        all_features = specialist_features[feature_set]
        x_train = all_features.iloc[positions["inner_train"]]
        x_valid = all_features.iloc[positions["inner_valid"]]
        for positive_weight in POSITIVE_WEIGHTS:
            specialist = build_specialist()
            sample_weight = hard_negative_weights(
                inner_metadata,
                labels.iloc[positions["inner_train"]].reset_index(drop=True),
                hard_inner,
                positive_weight,
            )
            specialist.fit(x_train, y_inner_binary, sample_weight=sample_weight)
            probability = specialist.predict_proba(x_valid)[:, 1]
            for threshold in THRESHOLDS:
                prediction = filter_infilteration(
                    inner_base_prediction, inner_fallback, probability, float(threshold)
                )
                metrics = evaluate(
                    labels.iloc[positions["inner_valid"]], prediction, attack_classes
                )
                candidate_rows.append(
                    {
                        "feature_set": feature_set,
                        "positive_weight": positive_weight,
                        "threshold": float(threshold),
                        **metrics,
                    }
                )
            fitted_candidates[(feature_set, positive_weight)] = specialist

    candidate_frame = pd.DataFrame(candidate_rows)
    candidate_frame = candidate_frame.sort_values(
        ["weighted_f1_attacks", "infilteration_f1", "infilteration_precision", "threshold"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    best = candidate_frame.iloc[0]
    selected_feature_set = str(best["feature_set"])
    selected_positive_weight = float(best["positive_weight"])
    selected_threshold = float(best["threshold"])
    print("Selected on inner validation:", best.to_dict())

    inner_hard_from_model = (
        (labels.iloc[positions["inner_valid"]].to_numpy() != INFILTERATION)
        & (inner_base_prediction == INFILTERATION)
    )
    hard_outer_global = np.zeros(len(metadata), dtype=bool)
    hard_outer_global[positions["inner_valid"]] = inner_hard_from_model
    print("Leak-free outer hard negatives:", int(hard_outer_global.sum()))

    outer_feature_frame = specialist_features[selected_feature_set]
    outer_specialist = build_specialist()
    outer_weights = hard_negative_weights(
        metadata.iloc[positions["outer_train"]].reset_index(drop=True),
        labels.iloc[positions["outer_train"]].reset_index(drop=True),
        hard_outer_global[positions["outer_train"]],
        selected_positive_weight,
    )
    outer_specialist.fit(
        outer_feature_frame.iloc[positions["outer_train"]],
        labels.iloc[positions["outer_train"]].eq(INFILTERATION).astype(np.int8),
        sample_weight=outer_weights,
    )

    configure_tracking(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"), PROJECT_EXPERIMENT)
    v15_model = mlflow.sklearn.load_model(SOURCE_V15_MODEL_URI)
    outer_v15_base = v15_model.base_model
    outer_base_prediction, outer_fallback, _ = multiclass_outputs(
        outer_v15_base, v15_features.iloc[positions["outer_valid"]], classes
    )
    outer_base_metrics = evaluate(
        labels.iloc[positions["outer_valid"]], outer_base_prediction, attack_classes
    )
    specialist_probability = outer_specialist.predict_proba(
        outer_feature_frame.iloc[positions["outer_valid"]]
    )[:, 1]
    outer_prediction = filter_infilteration(
        outer_base_prediction,
        outer_fallback,
        specialist_probability,
        selected_threshold,
    )
    outer_metrics = evaluate(
        labels.iloc[positions["outer_valid"]], outer_prediction, attack_classes
    )
    print("Outer v15 baseline:", outer_base_metrics)
    print("Outer v16 result:", outer_metrics)

    candidate_path = output_dir / "v16_inner_threshold_candidates.csv"
    candidate_frame.to_csv(candidate_path, index=False, encoding="utf-8-sig")
    comparison = pd.DataFrame(
        [
            {"model": "v15_baseline", **outer_base_metrics},
            {"model": "v16_hard_negative_verifier", **outer_metrics},
        ]
    )
    comparison_path = output_dir / "v16_outer_comparison.csv"
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    manifest = {
        "selected_feature_set": selected_feature_set,
        "selected_positive_weight": selected_positive_weight,
        "selected_threshold": selected_threshold,
        "specialist_feature_count": int(outer_feature_frame.shape[1]),
        "specialist_features": outer_feature_frame.columns.tolist(),
        "inner_hard_negative_count": int(hard_inner_global.sum()),
        "outer_hard_negative_count": int(hard_outer_global.sum()),
        "source_v15_run_id": SOURCE_V15_RUN_ID,
        "source_v15_model_uri": SOURCE_V15_MODEL_URI,
        "outer_base_metrics": outer_base_metrics,
        "outer_v16_metrics": outer_metrics,
    }
    manifest_path = output_dir / "v16_validation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    improved = outer_metrics["weighted_f1_attacks"] > outer_base_metrics["weighted_f1_attacks"]
    status = "full_refit_candidate" if improved else "rejected_on_outer_validation"
    with hypothesis_run(
        run_name="v16_infilteration_hard_negative_verifier_validation",
        hypothesis_id=HYPOTHESIS_ID,
        hypothesis=HYPOTHESIS,
        stage="chronological_validation",
        promotion_status=status,
        validation_strategy="nested_class_chronological_inner_selection_outer_evaluation",
        notebook="scripts/v16_infilteration_hard_negative_validation.py",
        data_version=raw_path.name,
        feature_schema_version="all-row-flow-plus-optional-v15-context-v1",
        code_version=file_sha256(Path(__file__)),
        extra_tags={
            "source_v15_run_id": SOURCE_V15_RUN_ID,
            "source_v15_model_uri": SOURCE_V15_MODEL_URI,
            "hard_negative_policy": "same-campaign-benign-plus-leak-free-prior-split-v15-false-positives",
            "filter_only": "true",
        },
    ) as run:
        mlflow.log_params(
            {
                "train_sample": TRAIN_SAMPLE,
                "random_state": RANDOM_STATE,
                "selected_feature_set": selected_feature_set,
                "selected_positive_weight": selected_positive_weight,
                "selected_threshold": selected_threshold,
                "specialist_feature_count": outer_feature_frame.shape[1],
                "inner_hard_negative_count": int(hard_inner_global.sum()),
                "outer_hard_negative_count": int(hard_outer_global.sum()),
                "source_v15_run_id": SOURCE_V15_RUN_ID,
                "source_v15_model_uri": SOURCE_V15_MODEL_URI,
            }
        )
        mlflow.log_metrics(
            {
                **{f"outer_v16_{key}": float(value) for key, value in outer_metrics.items()},
                **{f"outer_v15_{key}": float(value) for key, value in outer_base_metrics.items()},
                **{f"inner_v15_{key}": float(value) for key, value in inner_base_metrics.items()},
                "delta_weighted_f1_attacks": outer_metrics["weighted_f1_attacks"]
                - outer_base_metrics["weighted_f1_attacks"],
                "delta_infilteration_f1": outer_metrics["infilteration_f1"]
                - outer_base_metrics["infilteration_f1"],
                "elapsed_seconds": time.time() - started,
            }
        )
        for artifact in (candidate_path, comparison_path, manifest_path):
            mlflow.log_artifact(str(artifact))
        input_example = outer_feature_frame.iloc[positions["outer_valid"][:5]].copy()
        model_info = mlflow.sklearn.log_model(
            outer_specialist,
            name="infilteration_verifier",
            input_example=input_example,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            metadata={
                "filter_only": True,
                "selected_threshold": selected_threshold,
                "source_v15_model_uri": SOURCE_V15_MODEL_URI,
            },
        )
        run_id = run.info.run_id
        model_uri = model_info.model_uri

    print("Validation run:", run_id)
    print("Specialist model:", model_uri)
    print("Status:", status)
    print("Artifacts:", output_dir)


if __name__ == "__main__":
    main()
