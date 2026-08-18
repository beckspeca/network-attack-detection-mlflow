"""Serializable v15 model with an interpretable Infilteration veto layer."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class V15RuleVetoModel(BaseEstimator, ClassifierMixin):
    """Apply calibrated decision-tree leaves only to v15 Infilteration alerts."""

    def __init__(
        self,
        base_model,
        rule_tree,
        label_names,
        target_class_index: int,
        multiplier: float,
        veto_leaves,
    ):
        self.base_model = base_model
        self.rule_tree = rule_tree
        self.label_names = np.asarray(label_names, dtype=object)
        self.target_class_index = int(target_class_index)
        self.multiplier = float(multiplier)
        self.veto_leaves = tuple(int(value) for value in veto_leaves)
        self.classes_ = self.label_names
        if hasattr(base_model, "feature_names_in_"):
            self.feature_names_in_ = base_model.feature_names_in_
        if hasattr(base_model, "n_features_in_"):
            self.n_features_in_ = base_model.n_features_in_

    def _adjusted_scores(self, features):
        raw = np.asarray(self.base_model.predict_proba(features), dtype=float)
        encoded_classes = np.asarray(self.base_model.classes_, dtype=int)
        scores = np.zeros((len(raw), len(self.label_names)), dtype=float)
        scores[:, encoded_classes] = raw
        scores[:, self.target_class_index] *= self.multiplier
        return scores

    def predict_proba(self, features):
        scores = self._adjusted_scores(features)
        base_prediction = scores.argmax(axis=1)
        leaves = self.rule_tree.apply(features)
        veto = (base_prediction == self.target_class_index) & np.isin(
            leaves, self.veto_leaves
        )
        scores[veto, self.target_class_index] = 0.0
        denominator = scores.sum(axis=1, keepdims=True)
        return np.divide(scores, denominator, out=np.zeros_like(scores), where=denominator > 0)

    def predict(self, features):
        return self.label_names[self.predict_proba(features).argmax(axis=1)]
