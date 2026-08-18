"""Shared MLflow conventions for the network-classification project."""

from __future__ import annotations

from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient


DEFAULT_TRACKING_URI = "http://mlflow:5000"
PROJECT_EXPERIMENT = "network-classification-ids"
BENCHMARK_EXPERIMENT = "network-classification-ids-benchmark"
REGISTERED_MODEL_NAME = "network-classification-ids"
CHAMPION_ALIAS = "champion"
BENCHMARK_SCHEMA_VERSION = "ids-benchmark-v1"
VALIDATION_COHORT_ID = "temporal-v5_reservoir-250000_seed-42_class-chrono-80-20"
CORE_BENCHMARK_METRICS = (
    "accuracy",
    "weighted_f1_attacks",
    "macro_f1_attacks",
    "infilteration_precision",
    "infilteration_recall",
    "infilteration_f1",
    "benign_to_infilteration_fp",
)

STANDARD_TAG_KEYS = {
    "project",
    "hypothesis_id",
    "hypothesis",
    "stage",
    "validation_strategy",
    "promotion_status",
    "notebook",
    "code_version",
    "data_version",
    "feature_schema_version",
}


def configure_tracking(
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment_name: str = PROJECT_EXPERIMENT,
) -> str:
    """Connect to the shared tracking server and return the experiment id."""
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    return experiment.experiment_id


@contextmanager
def hypothesis_run(
    *,
    run_name: str,
    hypothesis_id: str,
    hypothesis: str,
    stage: str,
    promotion_status: str,
    validation_strategy: str,
    notebook: str,
    data_version: str,
    feature_schema_version: str,
    code_version: str = "unversioned",
    extra_tags: Mapping[str, Any] | None = None,
    nested: bool = False,
) -> Iterator[mlflow.ActiveRun]:
    """Start a run with the metadata required by this project."""
    tags = {
        "project": "network-classification",
        "hypothesis_id": hypothesis_id,
        "hypothesis": hypothesis,
        "stage": stage,
        "validation_strategy": validation_strategy,
        "promotion_status": promotion_status,
        "notebook": notebook,
        "code_version": code_version,
        "data_version": data_version,
        "feature_schema_version": feature_schema_version,
    }
    if extra_tags:
        tags.update({key: str(value) for key, value in extra_tags.items()})

    with mlflow.start_run(run_name=run_name, nested=nested, tags=tags) as run:
        yield run


def log_dataframe_input(
    frame: pd.DataFrame,
    *,
    source: str | Path,
    name: str,
    context: str,
    targets: str | None = None,
) -> None:
    """Log dataset source, digest, schema and profile to the active run."""
    source_path = Path(source)
    normalized_source = (
        source_path.resolve().as_uri()
        if source_path.is_absolute() or source_path.exists()
        else str(source)
    )
    dataset = mlflow.data.from_pandas(
        frame,
        source=normalized_source,
        name=name,
        targets=targets,
    )
    mlflow.log_input(dataset, context=context)


def log_sklearn_model_with_contract(
    model: Any,
    *,
    name: str,
    input_example: pd.DataFrame,
    code_paths: Sequence[str | Path] | None = None,
):
    """Log an sklearn-compatible model with an input/output contract."""
    predictions = model.predict(input_example)
    signature = infer_signature(input_example, predictions)
    return mlflow.sklearn.log_model(
        model,
        name=name,
        input_example=input_example,
        signature=signature,
        code_paths=[str(path) for path in code_paths] if code_paths else None,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    )


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a stable digest for a local dataset or code artifact."""
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def register_model_version(
    *,
    model_uri: str,
    registered_model_name: str = REGISTERED_MODEL_NAME,
    alias: str = CHAMPION_ALIAS,
    version_tags: Mapping[str, Any] | None = None,
    registered_model_tags: Mapping[str, Any] | None = None,
):
    """Register a logged model, attach governance tags, and move an alias."""
    version = mlflow.register_model(model_uri, registered_model_name)
    client = MlflowClient()
    if registered_model_tags:
        for key, value in registered_model_tags.items():
            client.set_registered_model_tag(registered_model_name, key, str(value))
    if version_tags:
        for key, value in version_tags.items():
            client.set_model_version_tag(
                registered_model_name, version.version, key, str(value)
            )
    client.set_registered_model_alias(registered_model_name, alias, version.version)
    return version


class HierarchicalIDSModel(mlflow.pyfunc.PythonModel):
    """Deployable wrapper around the v6 attack gate and subtype pipeline."""

    def __init__(
        self,
        attack_gate: Any,
        attack_subtype: Any,
        *,
        attack_threshold: float,
        feature_columns: Sequence[str],
        categorical_columns: Sequence[str] = ("Dst Port", "Protocol"),
    ) -> None:
        self.attack_gate = attack_gate
        self.attack_subtype = attack_subtype
        self.attack_threshold = float(attack_threshold)
        self.feature_columns = list(feature_columns)
        self.categorical_columns = list(categorical_columns)

    def _prepare(self, model_input: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)
        missing = [column for column in self.feature_columns if column not in model_input]
        if missing:
            raise ValueError(f"Missing required model columns: {missing}")
        frame = model_input.loc[:, self.feature_columns].copy()
        for column in self.categorical_columns:
            if column in frame:
                frame[column] = frame[column].astype("string")
        for column in self.feature_columns:
            if column not in self.categorical_columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
        return frame

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        frame = self._prepare(model_input)
        attack_probability = self.attack_gate.predict_proba(frame)[:, 1]
        attack_subtype = self.attack_subtype.predict(frame)
        labels = np.where(
            attack_probability >= self.attack_threshold,
            attack_subtype,
            "Benign",
        )
        return pd.DataFrame({"Label": labels}, index=frame.index)


class InfilterationThresholdIDSModel(HierarchicalIDSModel):
    """v6 hierarchy with a dedicated gate threshold for Infilteration.

    Known attack subtypes keep the validated v6 gate threshold. Only rows whose
    subtype prediction is Infilteration use ``infilteration_threshold``.
    """

    def __init__(
        self,
        attack_gate: Any,
        attack_subtype: Any,
        *,
        attack_threshold: float,
        infilteration_threshold: float,
        feature_columns: Sequence[str],
        categorical_columns: Sequence[str] = ("Dst Port", "Protocol"),
        infilteration_labels: Sequence[str] = ("Infilteration", "Infiltration"),
    ) -> None:
        super().__init__(
            attack_gate,
            attack_subtype,
            attack_threshold=attack_threshold,
            feature_columns=feature_columns,
            categorical_columns=categorical_columns,
        )
        self.infilteration_threshold = float(infilteration_threshold)
        self.infilteration_labels = tuple(infilteration_labels)

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        frame = self._prepare(model_input)
        attack_probability = self.attack_gate.predict_proba(frame)[:, 1]
        attack_subtype = self.attack_subtype.predict(frame)
        row_threshold = np.where(
            np.isin(attack_subtype, self.infilteration_labels),
            self.infilteration_threshold,
            self.attack_threshold,
        )
        labels = np.where(
            attack_probability >= row_threshold,
            attack_subtype,
            "Benign",
        )
        return pd.DataFrame({"Label": labels}, index=frame.index)
