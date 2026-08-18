"""Standalone PyTorch tabular classifier and MLflow PyFunc wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, width),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.block(values)


class DeepTabularNet(nn.Module):
    """Compact residual MLP with learned embeddings for categorical columns."""

    def __init__(
        self,
        *,
        numeric_features: int,
        categorical_cardinalities: Sequence[int],
        classes: int,
        width: int = 128,
        blocks: int = 3,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.numeric_features = int(numeric_features)
        self.categorical_cardinalities = [int(value) for value in categorical_cardinalities]
        embedding_dims = [min(16, max(2, int(np.ceil(np.sqrt(cardinality))))) for cardinality in self.categorical_cardinalities]
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, dimension)
            for cardinality, dimension in zip(self.categorical_cardinalities, embedding_dims)
        ])
        input_width = self.numeric_features + sum(embedding_dims)
        self.input_layer = nn.Sequential(
            nn.Linear(input_width, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(width, dropout) for _ in range(blocks)
        ])
        self.output_layer = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, classes),
        )

    def forward(self, numeric: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        embedded = [layer(categorical[:, index]) for index, layer in enumerate(self.embeddings)]
        combined = torch.cat([numeric, *embedded], dim=1) if embedded else numeric
        hidden = self.input_layer(combined)
        hidden = self.residual_blocks(hidden)
        return self.output_layer(hidden)


@dataclass
class DeepIDSPreprocessor:
    numeric_columns: list[str]
    categorical_columns: list[str]
    medians: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    category_maps: dict[str, dict[str, int]]

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        *,
        categorical_columns: Sequence[str],
    ) -> "DeepIDSPreprocessor":
        categorical = list(categorical_columns)
        numeric = [column for column in frame.columns if column not in categorical]
        values = frame.loc[:, numeric].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        values[~np.isfinite(values)] = np.nan
        medians = np.nanmedian(values, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        values = np.where(np.isnan(values), medians, values)
        lower = np.nanquantile(values, 0.001, axis=0)
        upper = np.nanquantile(values, 0.999, axis=0)
        values = np.clip(values, lower, upper)
        values = np.sign(values) * np.log1p(np.abs(values))
        means = values.mean(axis=0)
        scales = values.std(axis=0)
        scales = np.where(scales < 1e-6, 1.0, scales)
        category_maps = {}
        for column in categorical:
            tokens = frame[column].astype("string").fillna("<NA>")
            category_maps[column] = {
                token: index + 1
                for index, token in enumerate(sorted(tokens.unique().tolist()))
            }
        return cls(
            numeric_columns=numeric,
            categorical_columns=categorical,
            medians=medians.astype(np.float64),
            lower_bounds=lower.astype(np.float64),
            upper_bounds=upper.astype(np.float64),
            means=means.astype(np.float64),
            scales=scales.astype(np.float64),
            category_maps=category_maps,
        )

    @property
    def feature_columns(self) -> list[str]:
        return [*self.numeric_columns, *self.categorical_columns]

    @property
    def categorical_cardinalities(self) -> list[int]:
        return [len(self.category_maps[column]) + 1 for column in self.categorical_columns]

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        missing = [column for column in self.feature_columns if column not in frame]
        if missing:
            raise ValueError(f"Missing required model columns: {missing}")
        values = frame.loc[:, self.numeric_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        values[~np.isfinite(values)] = np.nan
        values = np.where(np.isnan(values), self.medians, values)
        values = np.clip(values, self.lower_bounds, self.upper_bounds)
        values = np.sign(values) * np.log1p(np.abs(values))
        numeric = ((values - self.means) / self.scales).astype(np.float32)
        categorical = np.zeros((len(frame), len(self.categorical_columns)), dtype=np.int64)
        for index, column in enumerate(self.categorical_columns):
            mapping = self.category_maps[column]
            tokens = frame[column].astype("string").fillna("<NA>")
            categorical[:, index] = tokens.map(mapping).fillna(0).to_numpy(dtype=np.int64)
        return numeric, categorical


class DeepIDSPyFuncModel(mlflow.pyfunc.PythonModel):
    """Deployable raw-flow classifier with preprocessing and calibrated threshold."""

    def __init__(
        self,
        network: DeepTabularNet,
        preprocessor: DeepIDSPreprocessor,
        classes: Sequence[str],
        *,
        infilteration_threshold: float,
        batch_size: int = 8192,
    ) -> None:
        self.network = network.cpu().eval()
        self.preprocessor = preprocessor
        self.classes = list(classes)
        self.infilteration_threshold = float(infilteration_threshold)
        self.batch_size = int(batch_size)
        self.infilteration_index = next(
            index for index, label in enumerate(self.classes)
            if label.lower() in {"infiltration", "infilteration"}
        )

    def predict_proba(self, model_input: pd.DataFrame) -> np.ndarray:
        numeric, categorical = self.preprocessor.transform(model_input)
        chunks = []
        self.network.eval()
        with torch.inference_mode():
            for start in range(0, len(numeric), self.batch_size):
                stop = start + self.batch_size
                logits = self.network(
                    torch.from_numpy(numeric[start:stop]),
                    torch.from_numpy(categorical[start:stop]),
                )
                chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(chunks, axis=0) if chunks else np.empty((0, len(self.classes)), dtype=np.float32)

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        probabilities = self.predict_proba(model_input)
        non_inf = probabilities.copy()
        non_inf[:, self.infilteration_index] = -np.inf
        predicted_index = non_inf.argmax(axis=1)
        predicted_index = np.where(
            probabilities[:, self.infilteration_index] >= self.infilteration_threshold,
            self.infilteration_index,
            predicted_index,
        )
        labels = np.asarray(self.classes, dtype=object)[predicted_index]
        return pd.DataFrame({"Label": labels}, index=model_input.index)
