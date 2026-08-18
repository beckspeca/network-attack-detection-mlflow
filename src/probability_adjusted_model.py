"""Probability adjustment wrapper for a fitted multiclass classifier."""

from __future__ import annotations

import numpy as np


class ProbabilityAdjustedClassifier:
    """Apply a fixed class multiplier before choosing the predicted label."""

    def __init__(self, base_model, label_names, target_class_index: int, multiplier: float):
        if multiplier <= 0:
            raise ValueError("multiplier must be positive")

        self.base_model = base_model
        self.label_names = np.asarray(label_names, dtype=object)
        self.target_class_index = int(target_class_index)
        self.multiplier = float(multiplier)
        self.classes_ = self.label_names.copy()

        base_classes = np.asarray(self.base_model.classes_)
        matches = np.flatnonzero(base_classes == self.target_class_index)
        if len(matches) != 1:
            raise ValueError(
                f"Target class {self.target_class_index} is not uniquely present in "
                f"base model classes {base_classes.tolist()}"
            )
        self._target_probability_column = int(matches[0])

        if hasattr(self.base_model, "feature_names_in_"):
            self.feature_names_in_ = self.base_model.feature_names_in_
        if hasattr(self.base_model, "n_features_in_"):
            self.n_features_in_ = self.base_model.n_features_in_

    def predict_proba(self, features):
        """Return the original calibrated probabilities without score adjustment."""
        return self.base_model.predict_proba(features)

    def predict(self, features):
        """Return labels after multiplying the configured class score."""
        probabilities = np.asarray(self.predict_proba(features), dtype=float)
        scores = probabilities.copy()
        scores[:, self._target_probability_column] *= self.multiplier
        base_classes = np.asarray(self.base_model.classes_)
        encoded_predictions = base_classes[scores.argmax(axis=1)].astype(int)
        return self.label_names[encoded_predictions]
