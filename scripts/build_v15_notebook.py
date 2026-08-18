#!/usr/bin/env python3
"""Build the v15 fixed-probability-adjustment notebook from v13."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_DIR / "notebooks" / "code_mlflow_pipeline_v13.ipynb"
TARGET = PROJECT_DIR / "notebooks" / "code_mlflow_pipeline_v15.ipynb"


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"Expected exactly one occurrence, found {source.count(old)}: {old[:80]!r}")
    return source.replace(old, new)


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        text = "".join(cell.get("source", []))
        text = text.replace("v13", "v15")
        cell["source"] = text.splitlines(keepends=True)
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    notebook["cells"][0]["source"] = (
        "# 네트워크 침입 탐지 — v15 Infilteration 확률 보정 Pipeline\n\n"
        "v13 RandomForest와 동일한 30개 과거 전용 피처를 사용하고, 최종 의사결정에서 "
        "Infilteration 확률만 1.225배 보정한다.\n\n"
        "- 모델 후보 선택과 검증 코호트는 v13의 표준 계약을 그대로 유지한다.\n"
        "- 보정계수는 사전 분석에서 고정하며 이 노트북 실행 중 재탐색하지 않는다.\n"
        "- 현재·미래 행과 테스트 Label은 사용하지 않는다.\n"
        "- rolling feature service가 없으므로 Registry champion은 변경하지 않는다.\n"
    ).splitlines(keepends=True)

    setup = "".join(notebook["cells"][1]["source"])
    setup = replace_once(
        setup,
        "from attached_temporal_model import HierarchicalClassifier\n",
        "from attached_temporal_model import HierarchicalClassifier\n"
        "from probability_adjusted_model import ProbabilityAdjustedClassifier\n",
    )
    setup = replace_once(
        setup,
        'HYPOTHESIS = "attached_code_temporal_hierarchical_standardized"',
        'HYPOTHESIS = "attached_temporal_random_forest_infilteration_multiplier"',
    )
    setup = replace_once(
        setup,
        "RANDOM_STATE = 42\nnp.random.seed(RANDOM_STATE)",
        "RANDOM_STATE = 42\nINFILTERATION_MULTIPLIER = 1.225\nnp.random.seed(RANDOM_STATE)",
    )
    notebook["cells"][1]["source"] = setup.splitlines(keepends=True)

    notebook["cells"][8]["source"] = (
        "## 4. 선택 모델 외부 시간 검증과 고정 확률 보정\n"
    ).splitlines(keepends=True)
    validation = "".join(notebook["cells"][9]["source"])
    old_validation = """outer_train_positions = np.flatnonzero(outer_train_mask.to_numpy())
outer_valid_positions = np.flatnonzero(outer_valid_mask.to_numpy())
selected_model = build_models(encoded[outer_train_positions])[selected_name]
started = time.time()
selected_model.fit(X_all.iloc[outer_train_positions], encoded[outer_train_positions])
outer_prediction_index = selected_model.predict(X_all.iloc[outer_valid_positions])
train_seconds = time.time() - started
outer_prediction = np.asarray(classes, dtype=object)[outer_prediction_index]
y_outer_valid = y_all.iloc[outer_valid_positions]
metrics = evaluate(y_outer_valid, outer_prediction)
print("Train seconds:", train_seconds)
display(pd.DataFrame([metrics], index=["v15_attached_standardized"]).T)
"""
    new_validation = """outer_train_positions = np.flatnonzero(outer_train_mask.to_numpy())
outer_valid_positions = np.flatnonzero(outer_valid_mask.to_numpy())
selected_base_model = build_models(encoded[outer_train_positions])[selected_name]
started = time.time()
selected_base_model.fit(X_all.iloc[outer_train_positions], encoded[outer_train_positions])
train_seconds = time.time() - started

adjusted_model = ProbabilityAdjustedClassifier(
    base_model=selected_base_model,
    label_names=classes,
    target_class_index=class_to_index[INFILTRATION_LABEL],
    multiplier=INFILTERATION_MULTIPLIER,
)
validation_features = X_all.iloc[outer_valid_positions]
base_prediction_index = selected_base_model.predict(validation_features)
base_prediction = np.asarray(classes, dtype=object)[base_prediction_index]
outer_prediction = adjusted_model.predict(validation_features)
y_outer_valid = y_all.iloc[outer_valid_positions].reset_index(drop=True)
base_metrics = evaluate(y_outer_valid, base_prediction)
metrics = evaluate(y_outer_valid, outer_prediction)

outer_valid_meta = sample_meta.iloc[outer_valid_positions].reset_index(drop=True)
late_cutoffs = outer_valid_meta.groupby(TARGET)[TIMESTAMP].quantile(0.50)
late_mask = outer_valid_meta[TIMESTAMP] > outer_valid_meta[TARGET].map(late_cutoffs)
late_positions = np.flatnonzero(late_mask.to_numpy())
late_metrics = evaluate(y_outer_valid.iloc[late_positions], outer_prediction[late_positions])

print("Train seconds:", train_seconds)
print("Fixed Infilteration multiplier:", INFILTERATION_MULTIPLIER)
display(pd.DataFrame([
    {"model": "v13_random_forest_base", **base_metrics},
    {"model": "v15_infilteration_multiplier_1.225", **metrics},
]).set_index("model").T)
display(pd.DataFrame([late_metrics], index=["v15_late_holdout"]).T)
"""
    validation = replace_once(validation, old_validation, new_validation)
    notebook["cells"][9]["source"] = validation.splitlines(keepends=True)

    tracking = "".join(notebook["cells"][11]["source"])
    tracking = replace_once(
        tracking,
        'pd.DataFrame([{"model": "v15_attached_standardized", **metrics}]).to_csv(summary_path, index=False, encoding="utf-8-sig")',
        'pd.DataFrame([\n'
        '    {"model": "v13_random_forest_base", **base_metrics},\n'
        '    {"model": "v15_infilteration_multiplier_1.225", **metrics},\n'
        ']).to_csv(summary_path, index=False, encoding="utf-8-sig")',
    )
    tracking = replace_once(
        tracking,
        'ax.set(title=f"v15 attached notebook standardized — {selected_name}", xlabel="Predicted", ylabel="Actual")',
        'ax.set(title=f"v15 fixed Infilteration multiplier {INFILTERATION_MULTIPLIER}", xlabel="Predicted", ylabel="Actual")',
    )
    tracking = replace_once(
        tracking,
        'signature = infer_signature(input_example, np.asarray(classes, dtype=object)[selected_model.predict(input_example)])',
        'signature = infer_signature(input_example, adjusted_model.predict(input_example))',
    )
    tracking = replace_once(
        tracking,
        'run_name="v15_attached_temporal_standardized_validation",',
        'run_name="v15_attached_temporal_rf_inf_multiplier_1_225_validation",',
    )
    tracking = replace_once(
        tracking,
        'validation_strategy="nested_time_selection_plus_class_chronological_80_20",',
        'validation_strategy="nested_time_selection_plus_fixed_inf_multiplier_class_chronological_80_20",',
    )
    tracking = replace_once(
        tracking,
        '"comparison_only_reason": "rolling_feature_service_not_implemented",',
        '"comparison_only_reason": "rolling_feature_service_not_implemented",\n'
        '        "adjustment_selection": "fixed_from_prior_v13_offline_analysis",',
    )
    tracking = replace_once(
        tracking,
        '"random_state": RANDOM_STATE,',
        '"random_state": RANDOM_STATE,\n'
        '        "infilteration_probability_multiplier": INFILTERATION_MULTIPLIER,\n'
        '        "base_model_source_version": "v13",',
    )
    tracking = replace_once(
        tracking,
        'mlflow.log_metrics({**{key: float(value) for key, value in metrics.items()}, "train_seconds": train_seconds})',
        'mlflow.log_metrics({\n'
        '        **{key: float(value) for key, value in metrics.items()},\n'
        '        **{f"base_{key}": float(value) for key, value in base_metrics.items()},\n'
        '        **{f"late_holdout_{key}": float(value) for key, value in late_metrics.items()},\n'
        '        "delta_vs_base_weighted_f1_attacks": metrics["weighted_f1_attacks"] - base_metrics["weighted_f1_attacks"],\n'
        '        "train_seconds": train_seconds,\n'
        '    })',
    )
    tracking = replace_once(
        tracking,
        'selected_model, name="benchmark_model",',
        'adjusted_model, name="benchmark_model",',
    )
    tracking = replace_once(
        tracking,
        'code_paths=[str(PROJECT_DIR / "src" / "attached_temporal_model.py")],',
        'code_paths=[\n'
        '            str(PROJECT_DIR / "src" / "attached_temporal_model.py"),\n'
        '            str(PROJECT_DIR / "src" / "probability_adjusted_model.py"),\n'
        '        ],',
    )
    tracking = replace_once(
        tracking,
        'metadata={"serving_ready": False, "requires_rolling_features": True},',
        'metadata={\n'
        '            "serving_ready": False,\n'
        '            "requires_rolling_features": True,\n'
        '            "infilteration_probability_multiplier": INFILTERATION_MULTIPLIER,\n'
        '        },',
    )
    tracking = replace_once(
        tracking,
        '        metadata={\n'
        '            "serving_ready": False,\n'
        '            "requires_rolling_features": True,\n'
        '            "infilteration_probability_multiplier": INFILTERATION_MULTIPLIER,\n'
        '        },',
        '        metadata={\n'
        '            "serving_ready": False,\n'
        '            "requires_rolling_features": True,\n'
        '            "infilteration_probability_multiplier": INFILTERATION_MULTIPLIER,\n'
        '        },\n'
        '        skops_trusted_types=["probability_adjusted_model.ProbabilityAdjustedClassifier"],',
    )
    tracking = replace_once(
        tracking,
        'f"- Selected model: {selected_name}\\n"',
        'f"- Selected base model: {selected_name}\\n"\n'
        '    f"- Infilteration probability multiplier: `{INFILTERATION_MULTIPLIER}`\\n"\n'
        '    f"- Attack Weighted F1 base / adjusted / delta: "\n'
        '    f"{base_metrics[\'weighted_f1_attacks\']:.6f} / {metrics[\'weighted_f1_attacks\']:.6f} / "\n'
        '    f"{metrics[\'weighted_f1_attacks\'] - base_metrics[\'weighted_f1_attacks\']:+.6f}\\n"',
    )
    tracking = replace_once(
        tracking,
        '"- Registry promotion disabled: online rolling feature service is not implemented.\\n",',
        'f"- Late-holdout attack Weighted F1: {late_metrics[\'weighted_f1_attacks\']:.6f}\\n"\n'
        '    "- Registry promotion disabled: online rolling feature service is not implemented.\\n",',
    )
    notebook["cells"][11]["source"] = tracking.splitlines(keepends=True)

    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(TARGET)


if __name__ == "__main__":
    main()
