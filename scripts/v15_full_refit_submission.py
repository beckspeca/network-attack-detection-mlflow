#!/usr/bin/env python3
"""Full-pool v15 refit and competition submission generation."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

try:
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature
except ModuleNotFoundError:
    mlflow = None
    infer_signature = None


PROJECT_EXPERIMENT = "network-classification-ids"
HYPOTHESIS_ID = "v15"
HYPOTHESIS = "attached_temporal_random_forest_infilteration_multiplier"
SOURCE_VALIDATION_RUN_ID = "4971c300af7f490bbdcd1de7606eccde"
SOURCE_CANONICAL_RUN_ID = "33f2be741323482b8a7bddbcf68ed95f"
RANDOM_STATE = 42
INFILTERATION_MULTIPLIER = 1.225
N_ESTIMATORS = 100
N_JOBS = 4
MAX_SAMPLES_PER_TREE = 1_000_000

ID_COL = "unique_id"
TARGET = "Label"
TIMESTAMP = "Timestamp"
ROLLING_SOURCE_COLUMNS = [
    TIMESTAMP,
    "Dst Port",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "SYN Flag Cnt",
]
REQUESTED_FLOW_FEATURES = [
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


def resolve_project_dir() -> Path:
    cwd = Path.cwd()
    if cwd.name == "notebooks":
        return cwd.parent
    if (cwd / "notebooks").is_dir() and (cwd / "src").is_dir():
        return cwd
    if Path("/mnt/elice/dataset").is_dir():
        return cwd
    if "__file__" in globals():
        return Path(__file__).resolve().parents[1]
    return cwd


def resolve_data_paths(project_dir: Path) -> tuple[Path, Path]:
    """Support both the competition CSV layout and the local Parquet cache."""
    configured_train = os.getenv("TRAIN_DATA_PATH")
    configured_test = os.getenv("TEST_DATA_PATH")
    candidates = []
    if configured_train and configured_test:
        candidates.append((Path(configured_train), Path(configured_test)))
    candidates.extend(
        [
            (Path("/mnt/elice/dataset/train.csv"), Path("/mnt/elice/dataset/test.csv")),
            (Path.cwd() / "train.csv", Path.cwd() / "test.csv"),
        ]
    )
    local_train = sorted((project_dir.parent / "datasets").glob("*/interim/train.parquet"))
    local_test = sorted((project_dir.parent / "datasets").glob("*/interim/test.parquet"))
    if len(local_train) == 1 and len(local_test) == 1:
        candidates.append((local_train[0], local_test[0]))
    for train_path, test_path in candidates:
        if train_path.is_file() and test_path.is_file():
            return train_path, test_path
    raise RuntimeError(f"Could not find train/test data. Checked: {candidates}")


def available_columns(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, nrows=0).columns.tolist()
    if path.suffix.lower() in {".parquet", ".pq"}:
        import pyarrow.parquet as parquet

        return parquet.ParquetFile(path).schema.names
    raise ValueError(f"Unsupported dataset format: {path}")


def read_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, usecols=columns, low_memory=False)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=columns)
    raise ValueError(f"Unsupported dataset format: {path}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_past_window_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create v13-compatible train features from prior train rows only."""
    values = frame.loc[:, ROLLING_SOURCE_COLUMNS].copy()
    values[TIMESTAMP] = pd.to_datetime(values[TIMESTAMP])
    window = values.set_index(TIMESTAMP).sort_index(kind="mergesort")
    features = pd.DataFrame(index=values.index)
    features["time_since_prev_flow"] = window.index.to_series().diff().dt.total_seconds().to_numpy()
    features["flow_count_10s"] = window["Tot Fwd Pkts"].rolling("10s", closed="left").count().to_numpy()
    features["flow_count_60s"] = window["Tot Fwd Pkts"].rolling("60s", closed="left").count().to_numpy()
    total_packets = window["Tot Fwd Pkts"] + window["Tot Bwd Pkts"]
    total_bytes = window["TotLen Fwd Pkts"] + window["TotLen Bwd Pkts"]
    features["packet_count_10s"] = total_packets.rolling("10s", closed="left").sum().to_numpy()
    features["byte_count_10s"] = total_bytes.rolling("10s", closed="left").sum().to_numpy()
    features["syn_count_10s"] = window["SYN Flag Cnt"].rolling("10s", closed="left").sum().to_numpy()
    features["packet_rate_10s"] = features["packet_count_10s"] / 10.0
    features["byte_rate_10s"] = features["byte_count_10s"] / 10.0
    features["dst_flow_count_10s"] = window.groupby("Dst Port")["Tot Fwd Pkts"].transform(
        lambda values: values.rolling("10s", closed="left").count()
    ).to_numpy()
    fwd60 = window["Tot Fwd Pkts"].rolling("60s", closed="left").sum()
    bwd60 = window["Tot Bwd Pkts"].rolling("60s", closed="left").sum()
    fwd10 = window["Tot Fwd Pkts"].rolling("10s", closed="left").sum()
    bwd10 = window["Tot Bwd Pkts"].rolling("10s", closed="left").sum()
    packet60 = total_packets.rolling("60s", closed="left").sum()
    features["fwd_bwd_ratio_60s"] = fwd60.to_numpy() / (bwd60.to_numpy() + 1)
    features["fwd_bwd_ratio_10s"] = fwd10.to_numpy() / (bwd10.to_numpy() + 1)
    features["burst_ratio_10s"] = features["packet_count_10s"] / (packet60.to_numpy() + 1)
    categorical = pd.Categorical(window["Dst Port"])
    onehot = pd.DataFrame(
        {
            index: (categorical.codes == index).astype(np.int8)
            for index in range(len(categorical.categories))
        },
        index=window.index,
    )
    present10 = onehot.rolling("10s", closed="left").sum() > 0
    present60 = onehot.rolling("60s", closed="left").sum() > 0
    features["dstport_distinct_10s"] = present10.sum(axis=1).to_numpy()
    features["dstport_distinct_60s"] = present60.sum(axis=1).to_numpy()
    features["port_diversity_10s"] = features["dstport_distinct_10s"] / (
        features["flow_count_10s"] + 1
    )
    unanswered = (window["Tot Bwd Pkts"] == 0).astype(np.int8)
    window_flows = window["Tot Fwd Pkts"].rolling("10s", closed="left").count()
    unanswered_count = unanswered.rolling("10s", closed="left").sum()
    features["syn_unanswered_rate_10s"] = unanswered_count.to_numpy() / (
        window_flows.to_numpy() + 1
    )
    features["syn_per_bwd_10s"] = features["syn_count_10s"] / (bwd10.to_numpy() + 1)
    return features.replace([np.inf, -np.inf], np.nan).fillna(0)


def build_train_features(
    full_frame: pd.DataFrame,
    selected_flow_features: list[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    temporal = add_past_window_features(full_frame)
    port_categories = sorted(full_frame["Dst Port"].astype("string").unique().tolist())
    port_map = {value: index for index, value in enumerate(port_categories)}
    flow = full_frame.loc[:, selected_flow_features].copy()
    flow["Dst Port"] = (
        full_frame["Dst Port"].astype("string").map(port_map).fillna(-1).astype(np.int16)
    )
    combined = pd.concat([flow, temporal], axis=1)
    return combined.replace([np.inf, -np.inf], np.nan).fillna(0), port_map


def build_test_features_from_train_history(
    train_history: pd.DataFrame,
    test_frame: pd.DataFrame,
    selected_flow_features: list[str],
    port_map: dict[str, int],
) -> pd.DataFrame:
    """Query past train events for test rows; test rows never enter the history."""
    train_values = train_history.loc[:, ROLLING_SOURCE_COLUMNS].copy()
    test_values = test_frame.loc[:, ROLLING_SOURCE_COLUMNS].copy()
    train_values[TIMESTAMP] = pd.to_datetime(train_values[TIMESTAMP])
    test_values[TIMESTAMP] = pd.to_datetime(test_values[TIMESTAMP])
    train_values["__is_train"] = True
    train_values["__test_position"] = -1
    test_values["__is_train"] = False
    test_values["__test_position"] = np.arange(len(test_values), dtype=np.int64)
    combined = pd.concat([train_values, test_values], ignore_index=True)
    combined = combined.sort_values(TIMESTAMP, kind="mergesort").set_index(TIMESTAMP)
    is_train = combined["__is_train"].to_numpy(dtype=bool)
    test_positions = combined["__test_position"].to_numpy(dtype=np.int64)
    output_mask = test_positions >= 0
    output_positions = test_positions[output_mask]

    def take_test(values) -> np.ndarray:
        array = np.asarray(values)
        output = np.empty(len(test_frame), dtype=array.dtype)
        output[output_positions] = array[output_mask]
        return output

    temporal = pd.DataFrame(index=np.arange(len(test_frame)))
    train_timestamps = train_values[TIMESTAMP].sort_values().to_numpy(dtype="datetime64[ns]")
    query_timestamps = test_values[TIMESTAMP].to_numpy(dtype="datetime64[ns]")
    previous_positions = np.searchsorted(train_timestamps, query_timestamps, side="left") - 1
    time_since_previous = np.zeros(len(test_frame), dtype=np.float64)
    has_previous = previous_positions >= 0
    time_since_previous[has_previous] = (
        query_timestamps[has_previous] - train_timestamps[previous_positions[has_previous]]
    ) / np.timedelta64(1, "s")
    temporal["time_since_prev_flow"] = time_since_previous

    fwd = combined["Tot Fwd Pkts"].where(combined["__is_train"])
    bwd = combined["Tot Bwd Pkts"].where(combined["__is_train"])
    fwd_bytes = combined["TotLen Fwd Pkts"].where(combined["__is_train"])
    bwd_bytes = combined["TotLen Bwd Pkts"].where(combined["__is_train"])
    syn = combined["SYN Flag Cnt"].where(combined["__is_train"])
    total_packets = fwd + bwd
    total_bytes = fwd_bytes + bwd_bytes
    flow_count_10s = fwd.rolling("10s", closed="left").count()
    packet_count_10s = total_packets.rolling("10s", closed="left").sum()
    byte_count_10s = total_bytes.rolling("10s", closed="left").sum()
    syn_count_10s = syn.rolling("10s", closed="left").sum()
    temporal["flow_count_10s"] = take_test(flow_count_10s)
    temporal["flow_count_60s"] = take_test(fwd.rolling("60s", closed="left").count())
    temporal["packet_count_10s"] = take_test(packet_count_10s)
    temporal["byte_count_10s"] = take_test(byte_count_10s)
    temporal["syn_count_10s"] = take_test(syn_count_10s)
    temporal["packet_rate_10s"] = temporal["packet_count_10s"] / 10.0
    temporal["byte_rate_10s"] = temporal["byte_count_10s"] / 10.0
    dst_flow_count = pd.DataFrame(
        {"Dst Port": combined["Dst Port"], "__train_fwd": fwd}, index=combined.index
    ).groupby("Dst Port")["__train_fwd"].transform(
        lambda values: values.rolling("10s", closed="left").count()
    )
    temporal["dst_flow_count_10s"] = take_test(dst_flow_count)
    fwd60 = fwd.rolling("60s", closed="left").sum()
    bwd60 = bwd.rolling("60s", closed="left").sum()
    fwd10 = fwd.rolling("10s", closed="left").sum()
    bwd10 = bwd.rolling("10s", closed="left").sum()
    packet60 = total_packets.rolling("60s", closed="left").sum()
    temporal["fwd_bwd_ratio_60s"] = take_test(fwd60) / (take_test(bwd60) + 1)
    temporal["fwd_bwd_ratio_10s"] = take_test(fwd10) / (take_test(bwd10) + 1)
    temporal["burst_ratio_10s"] = temporal["packet_count_10s"] / (take_test(packet60) + 1)
    categorical = pd.Categorical(combined["Dst Port"])
    onehot = pd.DataFrame(
        {
            index: ((categorical.codes == index) & is_train).astype(np.int8)
            for index in range(len(categorical.categories))
        },
        index=combined.index,
    )
    present10 = onehot.rolling("10s", closed="left").sum() > 0
    present60 = onehot.rolling("60s", closed="left").sum() > 0
    temporal["dstport_distinct_10s"] = take_test(present10.sum(axis=1))
    temporal["dstport_distinct_60s"] = take_test(present60.sum(axis=1))
    temporal["port_diversity_10s"] = temporal["dstport_distinct_10s"] / (
        temporal["flow_count_10s"] + 1
    )
    unanswered = ((combined["Tot Bwd Pkts"] == 0) & combined["__is_train"]).astype(np.int8)
    unanswered_count = unanswered.rolling("10s", closed="left").sum()
    temporal["syn_unanswered_rate_10s"] = take_test(unanswered_count) / (
        temporal["flow_count_10s"] + 1
    )
    temporal["syn_per_bwd_10s"] = temporal["syn_count_10s"] / (take_test(bwd10) + 1)
    flow = test_frame.loc[:, selected_flow_features].copy().reset_index(drop=True)
    flow["Dst Port"] = (
        test_frame["Dst Port"].astype("string").map(port_map).fillna(-1).astype(np.int16).to_numpy()
    )
    output = pd.concat([flow, temporal], axis=1)
    return output.replace([np.inf, -np.inf], np.nan).fillna(0)


class ProbabilityAdjustedClassifier:
    """Apply a fixed class multiplier before choosing the predicted label."""

    def __init__(self, base_model, label_names, target_class_index: int, multiplier: float):
        self.base_model = base_model
        self.label_names = np.asarray(label_names, dtype=object)
        self.target_class_index = int(target_class_index)
        self.multiplier = float(multiplier)
        self.classes_ = self.label_names.copy()
        base_classes = np.asarray(self.base_model.classes_)
        matches = np.flatnonzero(base_classes == self.target_class_index)
        if len(matches) != 1:
            raise ValueError("Adjusted class is not uniquely present in base model classes")
        self._target_probability_column = int(matches[0])
        self.feature_names_in_ = self.base_model.feature_names_in_
        self.n_features_in_ = self.base_model.n_features_in_

    def predict_proba(self, features):
        return self.base_model.predict_proba(features)

    def predict(self, features):
        scores = np.asarray(self.predict_proba(features), dtype=float)
        scores[:, self._target_probability_column] *= self.multiplier
        base_classes = np.asarray(self.base_model.classes_)
        encoded = base_classes[scores.argmax(axis=1)].astype(int)
        return self.label_names[encoded]


def main() -> None:
    np.random.seed(RANDOM_STATE)
    project_dir = resolve_project_dir()
    train_path, test_path = resolve_data_paths(project_dir)
    output_dir = project_dir / "outputs" / HYPOTHESIS_ID
    output_dir.mkdir(parents=True, exist_ok=True)
    notebook_dir = project_dir / "notebooks" if (project_dir / "notebooks").is_dir() else project_dir
    submission_path = notebook_dir / "submission.csv"
    backup_submission_path = output_dir / "submission_v15.csv"

    available = available_columns(train_path)
    selected_flow_features = [column for column in REQUESTED_FLOW_FEATURES if column in available]
    missing_features = [column for column in REQUESTED_FLOW_FEATURES if column not in available]
    required = list(dict.fromkeys([ID_COL, TARGET, *ROLLING_SOURCE_COLUMNS, *selected_flow_features]))
    full_train = read_columns(train_path, required).sort_values(
        [TIMESTAMP, ID_COL], kind="mergesort"
    ).reset_index(drop=True)

    full_train[TIMESTAMP] = pd.to_datetime(full_train[TIMESTAMP])
    if full_train[TARGET].isna().any():
        raise ValueError("Training labels contain missing values")
    train_history = full_train.loc[:, ROLLING_SOURCE_COLUMNS].copy()
    train_lineage = full_train.loc[:, [ID_COL, TIMESTAMP, TARGET]].head(2000).copy()
    y_labels = full_train[TARGET].astype(str).reset_index(drop=True)
    classes = sorted(y_labels.unique().tolist())
    class_to_index = {label: index for index, label in enumerate(classes)}
    encoded = y_labels.map(class_to_index).to_numpy(dtype=np.int64)
    infilteration_label = next(
        label for label in classes if label.lower() in {"infiltration", "infilteration"}
    )

    feature_started = time.time()
    train_features, port_map = build_train_features(full_train, selected_flow_features)
    feature_seconds = time.time() - feature_started
    feature_columns = train_features.columns.tolist()
    counts = np.bincount(encoded, minlength=len(classes)).astype(float)
    class_weights = {
        index: len(encoded) / (len(classes) * count) for index, count in enumerate(counts)
    }
    class_weights[class_to_index[infilteration_label]] *= 1.6
    print("Train rows/features:", train_features.shape)
    print("Classes:", dict(zip(classes, counts.astype(int))))
    print("Feature seconds:", feature_seconds)

    del full_train
    gc.collect()
    base_model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        class_weight=class_weights,
        bootstrap=True,
        max_samples=min(MAX_SAMPLES_PER_TREE, len(train_features)),
        verbose=1,
    )
    train_started = time.time()
    base_model.fit(train_features, encoded)
    train_seconds = time.time() - train_started
    adjusted_model = ProbabilityAdjustedClassifier(
        base_model=base_model,
        label_names=classes,
        target_class_index=class_to_index[infilteration_label],
        multiplier=INFILTERATION_MULTIPLIER,
    )
    print("Refit seconds:", train_seconds)

    del train_features, encoded, y_labels
    gc.collect()
    test_frame = read_columns(test_path, required)
    if test_frame[TARGET].notna().any():
        raise ValueError("Test Label must be entirely empty")
    expected_test_rows = len(test_frame)
    test_ids = test_frame[ID_COL].copy().reset_index(drop=True)
    test_lineage = test_frame.loc[:, [ID_COL, TIMESTAMP]].head(2000).copy()
    test_feature_started = time.time()
    test_features = build_test_features_from_train_history(
        train_history=train_history,
        test_frame=test_frame,
        selected_flow_features=selected_flow_features,
        port_map=port_map,
    )
    test_feature_seconds = time.time() - test_feature_started
    if test_features.columns.tolist() != feature_columns:
        raise ValueError(
            f"Train/test feature mismatch: train={feature_columns}, test={test_features.columns.tolist()}"
        )
    print("Test rows/features:", test_features.shape)
    print("Test feature seconds:", test_feature_seconds)

    del train_history, test_frame
    gc.collect()
    predict_started = time.time()
    test_prediction = adjusted_model.predict(test_features)
    predict_seconds = time.time() - predict_started
    submission = pd.DataFrame({ID_COL: test_ids, TARGET: test_prediction}).set_index(ID_COL)
    submission.to_csv(submission_path, encoding="utf-8-sig")
    submission.to_csv(backup_submission_path, encoding="utf-8-sig")
    distribution = submission[TARGET].value_counts().reindex(classes, fill_value=0)
    distribution_path = output_dir / "v15_submission_distribution.csv"
    distribution.rename("count").to_csv(distribution_path, encoding="utf-8-sig")

    checks = {
        "rows": int(len(submission)),
        "expected_rows": expected_test_rows,
        "missing_labels": int(submission[TARGET].isna().sum()),
        "duplicate_ids": int(submission.index.duplicated().sum()),
        "unknown_labels": sorted(set(submission[TARGET]) - set(classes)),
        "test_order_preserved": bool(submission.index.tolist() == test_ids.tolist()),
    }
    if checks != {
        "rows": expected_test_rows,
        "expected_rows": expected_test_rows,
        "missing_labels": 0,
        "duplicate_ids": 0,
        "unknown_labels": [],
        "test_order_preserved": True,
    }:
        raise AssertionError(f"Submission validation failed: {checks}")
    print("Submission checks:", checks)
    print("Prediction distribution:\n", distribution)
    print("Prediction seconds:", predict_seconds)

    input_example = test_features.iloc[:5].copy()
    output_example = adjusted_model.predict(input_example)
    notebook_path = notebook_dir / "code.ipynb"
    script_path = project_dir / "scripts" / "v15_full_refit_submission.py"
    refit_run_id = None
    model_uri = None
    mlflow_error = None
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    mlflow_enabled = mlflow is not None and bool(tracking_uri)
    if mlflow_enabled:
        try:
            signature = infer_signature(input_example, output_example)
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(PROJECT_EXPERIMENT)
            with mlflow.start_run(
                run_name="v15_competition_full_pool_refit_submission",
                tags={
                    "project": "network-classification",
                    "hypothesis_id": HYPOTHESIS_ID,
                    "hypothesis": HYPOTHESIS,
                    "stage": "competition_full_pool_refit",
                    "promotion_status": "submission_candidate",
                    "source_validation_run_id": SOURCE_VALIDATION_RUN_ID,
                    "source_canonical_run_id": SOURCE_CANONICAL_RUN_ID,
                    "test_label_used": "false",
                    "test_history_policy": "prior_train_events_only",
                },
            ) as run:
                mlflow.log_params(
                    {
                        "fit_rows": int(counts.sum()),
                        "test_rows": len(test_features),
                        "feature_count": len(feature_columns),
                        "flow_feature_count": len(selected_flow_features),
                        "rolling_feature_count": len(feature_columns) - len(selected_flow_features),
                        "missing_requested_features": json.dumps(missing_features),
                        "n_estimators": N_ESTIMATORS,
                        "max_samples_per_tree": min(MAX_SAMPLES_PER_TREE, int(counts.sum())),
                        "n_jobs": N_JOBS,
                        "random_state": RANDOM_STATE,
                        "infilteration_class_weight_multiplier": 1.6,
                        "infilteration_probability_multiplier": INFILTERATION_MULTIPLIER,
                        "source_validation_run_id": SOURCE_VALIDATION_RUN_ID,
                        "source_canonical_run_id": SOURCE_CANONICAL_RUN_ID,
                        "train_data_sha256": file_sha256(train_path),
                        "test_data_sha256": file_sha256(test_path),
                        "submission_sha256": file_sha256(submission_path),
                        "code_sha256": file_sha256(notebook_path),
                    }
                )
                mlflow.log_metrics(
                    {
                        "feature_seconds": feature_seconds,
                        "train_seconds": train_seconds,
                        "test_feature_seconds": test_feature_seconds,
                        "predict_seconds": predict_seconds,
                        "source_validation_weighted_f1_attacks": 0.9565633450835647,
                        "source_validation_late_holdout_weighted_f1_attacks": 0.9779619459425275,
                        **{
                            f"prediction_count_{label.lower().replace(' ', '_')}": int(value)
                            for label, value in distribution.items()
                        },
                    }
                )
                mlflow.log_input(
                    mlflow.data.from_pandas(
                        train_lineage,
                        source=str(train_path),
                        name="competition_full_train_lineage_sample",
                        targets=TARGET,
                    ),
                    context="training",
                )
                mlflow.log_input(
                    mlflow.data.from_pandas(
                        test_lineage,
                        source=str(test_path),
                        name="competition_test_lineage_sample",
                    ),
                    context="inference",
                )
                mlflow.log_artifact(str(submission_path), artifact_path="submission")
                mlflow.log_artifact(str(distribution_path), artifact_path="submission")
                mlflow.log_artifact(str(notebook_path), artifact_path="code")
                if script_path.is_file():
                    mlflow.log_artifact(str(script_path), artifact_path="code")
                model_info = mlflow.sklearn.log_model(
                    adjusted_model,
                    name="competition_model",
                    input_example=input_example,
                    signature=signature,
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                    metadata={
                        "competition_candidate": True,
                        "requires_past_train_history_features": True,
                        "infilteration_probability_multiplier": INFILTERATION_MULTIPLIER,
                    },
                )
                refit_run_id = run.info.run_id
                model_uri = model_info.model_uri
            reloaded_model = mlflow.sklearn.load_model(model_uri)
            if not np.array_equal(reloaded_model.predict(input_example), output_example):
                raise AssertionError("Reloaded MLflow model predictions do not match")
        except Exception as error:
            mlflow_error = f"{type(error).__name__}: {error}"
            print("MLflow logging skipped; submission is still valid:", mlflow_error)
    elif mlflow is None:
        mlflow_error = "mlflow package is not installed"
        print("MLflow logging skipped; submission is still valid:", mlflow_error)
    else:
        mlflow_error = "MLFLOW_TRACKING_URI is not configured"
        print("MLflow logging skipped; submission is still valid:", mlflow_error)

    status_path = output_dir / "REFIT_SUBMISSION.md"
    status_path.write_text(
        "# v15 competition refit and submission\n\n"
        f"- Fit rows: `{int(counts.sum())}`\n"
        f"- Trees / max samples per tree: `{N_ESTIMATORS}` / `{min(MAX_SAMPLES_PER_TREE, int(counts.sum()))}`\n"
        f"- Infilteration probability multiplier: `{INFILTERATION_MULTIPLIER}`\n"
        f"- Submission rows: `{len(submission)}`\n"
        f"- MLflow refit run: `{refit_run_id or 'skipped'}`\n"
        f"- Model artifact: `{model_uri or 'skipped'}`\n"
        f"- MLflow note: `{mlflow_error or 'logged and reload-verified'}`\n"
        f"- Submission: `{submission_path}`\n",
        encoding="utf-8",
    )
    if mlflow is not None and refit_run_id is not None:
        mlflow.tracking.MlflowClient().log_artifact(
            refit_run_id, str(status_path), artifact_path="submission"
        )
    print("MLflow refit run:", refit_run_id or "skipped")
    print("Model URI:", model_uri or "skipped")
    print("Submission:", submission_path)
    print("Backup:", backup_submission_path)
    print("MLflow reload verification:", "ok" if model_uri else "skipped")


if __name__ == "__main__":
    main()
