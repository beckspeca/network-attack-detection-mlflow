#!/usr/bin/env python3
"""Build a self-contained Elice code.ipynb for the validated v20b rule hybrid."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_DIR / "scripts" / "v15_full_refit_submission.py"
TARGET = PROJECT_DIR / "elice" / "code.ipynb"


def transformed_source() -> str:
    source = SOURCE.read_text(encoding="utf-8")
    replacements = {
        '"""Full-pool v15 refit and competition submission generation."""':
            '"""Elice submission: full-pool v15 plus validated v20b rule veto."""',
        'HYPOTHESIS_ID = "v15"': 'HYPOTHESIS_ID = "v20b"',
        'HYPOTHESIS = "attached_temporal_random_forest_infilteration_multiplier"':
            'HYPOTHESIS = "v15_plus_validated_rule_based_infilteration_veto"',
        'class ProbabilityAdjustedClassifier:': 'class RuleAdjustedClassifier:',
        'adjusted_model = ProbabilityAdjustedClassifier(': 'adjusted_model = RuleAdjustedClassifier(',
        'backup_submission_path = output_dir / "submission_v15.csv"':
            'backup_submission_path = output_dir / "submission_v20b_rule_hybrid.csv"',
        'distribution_path = output_dir / "v15_submission_distribution.csv"':
            'distribution_path = output_dir / "v20b_submission_distribution.csv"',
        'run_name="v15_competition_full_pool_refit_submission"':
            'run_name="v20b_elice_full_pool_rule_hybrid_submission"',
        '"# v15 competition refit and submission\\n\\n"':
            '"# v20b Elice competition refit and rule-hybrid submission\\n\\n"',
    }
    for old, new in replacements.items():
        if old not in source:
            raise RuntimeError(f"Expected source fragment not found: {old}")
        source = source.replace(old, new)

    original_predict = '''    def predict(self, features):
        scores = np.asarray(self.predict_proba(features), dtype=float)
        scores[:, self._target_probability_column] *= self.multiplier
        base_classes = np.asarray(self.base_model.classes_)
        encoded = base_classes[scores.argmax(axis=1)].astype(int)
        return self.label_names[encoded]
'''
    rule_predict = '''    def predict(self, features):
        scores = np.asarray(self.predict_proba(features), dtype=float)
        scores[:, self._target_probability_column] *= self.multiplier
        base_classes = np.asarray(self.base_model.classes_)
        encoded = base_classes[scores.argmax(axis=1)].astype(int)
        prediction = self.label_names[encoded].copy()

        # 시간순 홀드아웃에서 검증된 고정밀 Infilteration 오탐 억제 규칙.
        non_infilteration_scores = scores.copy()
        non_infilteration_scores[:, self._target_probability_column] = -np.inf
        fallback_encoded = base_classes[non_infilteration_scores.argmax(axis=1)].astype(int)
        fallback = self.label_names[fallback_encoded]
        target_label = self.label_names[self.target_class_index]
        rule_mask = (
            (prediction == target_label)
            & (features["flow_count_60s"].to_numpy() > 917.0)
            & (features["Pkt Size Avg"].to_numpy() <= 82.75)
        )
        prediction[rule_mask] = fallback[rule_mask]
        self.last_rule_change_count_ = int(rule_mask.sum())
        return prediction
'''
    if original_predict not in source:
        raise RuntimeError("Could not locate the v15 predict method")
    source = source.replace(original_predict, rule_predict)

    prediction_print = '    print("Prediction seconds:", predict_seconds)\n'
    replacement_print = (
        prediction_print
        + '    print("v20b rule changes:", adjusted_model.last_rule_change_count_)\n'
    )
    if prediction_print not in source:
        raise RuntimeError("Could not locate prediction reporting")
    source = source.replace(prediction_print, replacement_print)
    return source


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": [
            {
                "id": "v20b-overview",
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 네트워크 침입 분류 — v20b 엘리스 최종 제출\n",
                    "\n",
                    "전체 학습 데이터로 v15 RandomForest를 재학습하고 Infilteration 확률을 "
                    "1.225배 보정한다. 시간순 홀드아웃에서 검증된 규칙 "
                    "`flow_count_60s > 917 and Pkt Size Avg <= 82.75`를 사용해 "
                    "고밀도·소형 패킷 정상 흐름의 Infilteration 오탐을 차선 클래스로 되돌린다. "
                    "테스트 Label은 사용하지 않으며 실행 위치에 `submission.csv`를 생성한다.\n",
                ],
            },
            {
                "id": "v20b-pipeline",
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": transformed_source().splitlines(keepends=True),
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(TARGET)


if __name__ == "__main__":
    main()
