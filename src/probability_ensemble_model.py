"""MLflow PyFunc wrapper for the v14 cross-model probability ensemble."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import mlflow.pyfunc
import numpy as np
import pandas as pd


class ProbabilityEnsembleIDSModel(mlflow.pyfunc.PythonModel):
    def __init__(
        self,
        *,
        v11_gate: Any,
        v11_subtype: Any,
        attachment_models: Mapping[str, Any],
        weights: Mapping[str, float],
        classes: Sequence[str],
        router_columns: Sequence[str],
        attachment_columns: Sequence[str],
        port_map: Mapping[str, int],
        infilteration_multiplier: float,
        categorical_columns: Sequence[str] = ("Dst Port", "Protocol"),
    ) -> None:
        self.v11_gate = v11_gate
        self.v11_subtype = v11_subtype
        self.attachment_models = dict(attachment_models)
        self.weights = {name: float(value) for name, value in weights.items()}
        self.classes = list(classes)
        self.class_to_index = {label: index for index, label in enumerate(self.classes)}
        self.router_columns = list(router_columns)
        self.attachment_columns = list(attachment_columns)
        self.port_map = dict(port_map)
        self.infilteration_multiplier = float(infilteration_multiplier)
        self.categorical_columns = list(categorical_columns)
        self.infilteration_index = next(
            index for index, label in enumerate(self.classes)
            if label.lower() in {"infiltration", "infilteration"}
        )

    def _v11_probability(self, frame: pd.DataFrame) -> np.ndarray:
        router = frame.loc[:, self.router_columns].copy()
        for column in self.categorical_columns:
            if column in router:
                router[column] = router[column].astype("string")
        gate_classes = list(self.v11_gate.classes_)
        attack_probability = self.v11_gate.predict_proba(router)[:, gate_classes.index(1)]
        subtype_probability = self.v11_subtype.predict_proba(router)
        output = np.zeros((len(frame), len(self.classes)), dtype=float)
        output[:, self.class_to_index["Benign"]] = 1.0 - attack_probability
        for local_index, label in enumerate(self.v11_subtype.classes_):
            output[:, self.class_to_index[str(label)]] = attack_probability * subtype_probability[:, local_index]
        return output / np.maximum(output.sum(axis=1, keepdims=True), 1e-12)

    def _attachment_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        attachment = frame.loc[:, self.attachment_columns].copy()
        attachment["Dst Port"] = (
            frame["Dst Port"].astype("string").map(self.port_map).fillna(-1).astype(np.int16)
        )
        return attachment

    def predict_proba(self, model_input: pd.DataFrame) -> np.ndarray:
        missing = [
            column for column in set(self.router_columns).union(self.attachment_columns)
            if column not in model_input
        ]
        if missing:
            raise ValueError(f"Missing required ensemble columns: {sorted(missing)}")
        probabilities = {"v11_hierarchy": self._v11_probability(model_input)}
        attachment = self._attachment_frame(model_input)
        for name, model in self.attachment_models.items():
            values = model.predict_proba(attachment)
            aligned = np.zeros((len(attachment), len(self.classes)), dtype=float)
            for local_index, class_index in enumerate(model.classes_):
                aligned[:, int(class_index)] = values[:, local_index]
            probabilities[name] = aligned
        combined = np.zeros((len(model_input), len(self.classes)), dtype=float)
        for name, weight in self.weights.items():
            combined += weight * probabilities[name]
        return combined / np.maximum(combined.sum(axis=1, keepdims=True), 1e-12)

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        probability = self.predict_proba(model_input)
        scores = probability.copy()
        scores[:, self.infilteration_index] *= self.infilteration_multiplier
        labels = np.asarray(self.classes, dtype=object)[scores.argmax(axis=1)]
        return pd.DataFrame({"Label": labels}, index=model_input.index)
