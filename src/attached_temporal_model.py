"""Reusable hierarchical classifier adapted from the attached code.ipynb."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone


class HierarchicalClassifier(BaseEstimator, ClassifierMixin):
    """Benign/attack gate followed by an attack-subtype classifier."""

    def __init__(self, base1, base2, benign_index: int = 0):
        self.base1 = base1
        self.base2 = base2
        self.benign_index = benign_index

    def fit(self, X, y):
        labels = np.asarray(y, dtype=int)
        binary = (labels != int(self.benign_index)).astype(int)
        self.base1_ = clone(self.base1)
        self.base2_ = clone(self.base2)
        self.base1_.fit(X, binary)
        attack_mask = binary == 1
        self.attack_classes_ = np.unique(labels[attack_mask])
        local_map = {label: index for index, label in enumerate(self.attack_classes_)}
        local_labels = np.asarray([local_map[label] for label in labels[attack_mask]], dtype=int)
        self.base2_.fit(X.iloc[np.flatnonzero(attack_mask)], local_labels)
        self.classes_ = np.unique(labels)
        return self

    def predict(self, X):
        output = np.full(len(X), int(self.benign_index), dtype=int)
        binary = np.asarray(self.base1_.predict(X), dtype=int)
        attack_rows = np.flatnonzero(binary == 1)
        if len(attack_rows):
            local_prediction = np.asarray(self.base2_.predict(X.iloc[attack_rows]), dtype=int)
            output[attack_rows] = self.attack_classes_[local_prediction]
        return output

    def predict_proba(self, X):
        gate_probability = self.base1_.predict_proba(X)
        gate_classes = list(self.base1_.classes_)
        attack_probability = gate_probability[:, gate_classes.index(1)]
        subtype_probability = self.base2_.predict_proba(X)
        class_count = int(max(self.classes_)) + 1
        output = np.zeros((len(X), class_count), dtype=float)
        output[:, int(self.benign_index)] = 1.0 - attack_probability
        for local_index, global_class in enumerate(self.attack_classes_):
            output[:, int(global_class)] = attack_probability * subtype_probability[:, local_index]
        row_sum = output.sum(axis=1, keepdims=True)
        return output / np.maximum(row_sum, 1e-12)
