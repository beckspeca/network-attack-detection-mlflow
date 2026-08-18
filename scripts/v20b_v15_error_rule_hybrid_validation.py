#!/usr/bin/env python3
"""Validate a v15 IDS rule layer learned only from out-of-time v15 errors."""

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
from sklearn.metrics import precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier, export_text


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from attached_temporal_features import ROLLING_SOURCE_COLUMNS, build_attachment_feature_frame  # noqa: E402
from mlflow_workflow import PROJECT_EXPERIMENT, configure_tracking, file_sha256, hypothesis_run  # noqa: E402
from v15_rule_veto_model import V15RuleVetoModel  # noqa: E402
from v16_infilteration_hard_negative_validation import (  # noqa: E402
    ID_COL,
    INFILTERATION,
    INFILTERATION_MULTIPLIER,
    OUTER_VALID_FRACTION,
    RANDOM_STATE,
    REQUESTED_V15_FLOW_FEATURES,
    TARGET,
    TIMESTAMP,
    TRAIN_SAMPLE,
    build_v15_base,
    class_chronological_masks,
    evaluate,
    multiclass_outputs,
    resolve_dataset_paths,
)


HYPOTHESIS_ID = "v20b"
HYPOTHESIS = "v15_plus_error_specific_interpretable_infilteration_veto_rules"
SOURCE_V15_RUN_ID = "4971c300af7f490bbdcd1de7606eccde"
SOURCE_V15_MODEL_URI = "models:/m-b063a6360c3246aaa77c08454ec78a3f"
INNER_VALID_FRACTION = 0.10
RULE_POOL_FRACTION = 0.40
RULE_CALIBRATION_FRACTION = 0.50
TREE_DEPTHS = (3, 4, 5, 6, 8)
VETO_PRECISION_THRESHOLDS = (0.60, 0.70, 0.80, 0.90, 0.95)
MIN_VETO_SUPPORTS = (10, 20, 50)
MAX_IDS_RECALL_DROP = 0.01


def ids_metrics(y_true, prediction) -> dict[str, float]:
    actual_attack = np.asarray(y_true, dtype=object) != "Benign"
    predicted_attack = np.asarray(prediction, dtype=object) != "Benign"
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual_attack,
        predicted_attack,
        average="binary",
        zero_division=0,
    )
    benign_count = int((~actual_attack).sum())
    false_positives = int((~actual_attack & predicted_attack).sum())
    return {
        "ids_attack_precision": float(precision),
        "ids_attack_recall": float(recall),
        "ids_attack_f1": float(f1),
        "ids_false_positive_rate": float(false_positives / benign_count) if benign_count else 0.0,
        "ids_false_positive_count": false_positives,
    }


def all_metrics(y_true, prediction, attack_classes) -> dict[str, float]:
    return {**evaluate(y_true, prediction, attack_classes), **ids_metrics(y_true, prediction)}


def fit_veto_tree(features, target, depth: int) -> DecisionTreeClassifier:
    tree = DecisionTreeClassifier(
        max_depth=int(depth),
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    tree.fit(features, target)
    return tree


def calibrate_veto_leaves(
    tree: DecisionTreeClassifier,
    features: pd.DataFrame,
    labels: pd.Series,
    base_prediction: np.ndarray,
    fallback_prediction: np.ndarray,
) -> dict[int, dict[str, float | int]]:
    candidate = np.asarray(base_prediction, dtype=object) == INFILTERATION
    if not candidate.any():
        return {}
    candidate_features = features.loc[candidate]
    leaves = tree.apply(candidate_features)
    actual = labels.to_numpy(dtype=object)[candidate]
    fallback = np.asarray(fallback_prediction, dtype=object)[candidate]
    frame = pd.DataFrame(
        {"leaf": leaves, "actual": actual, "fallback": fallback}
    )
    statistics: dict[int, dict[str, float | int]] = {}
    for leaf, group in frame.groupby("leaf"):
        support = len(group)
        statistics[int(leaf)] = {
            "candidate_support": int(support),
            "fallback_correct_precision": float(group["fallback"].eq(group["actual"]).mean()),
            "benign_fraction": float(group["actual"].eq("Benign").mean()),
            "true_infilteration_fraction": float(group["actual"].eq(INFILTERATION).mean()),
        }
    return statistics


def select_veto_leaves(statistics, precision_threshold: float, min_support: int) -> set[int]:
    return {
        int(leaf)
        for leaf, stats in statistics.items()
        if int(stats["candidate_support"]) >= int(min_support)
        and float(stats["fallback_correct_precision"]) >= float(precision_threshold)
    }


def apply_veto(
    tree,
    features,
    base_prediction,
    fallback_prediction,
    veto_leaves,
):
    original = np.asarray(base_prediction, dtype=object)
    output = original.copy()
    leaves = tree.apply(features)
    veto = (original == INFILTERATION) & np.isin(leaves, list(veto_leaves))
    output[veto] = np.asarray(fallback_prediction, dtype=object)[veto]
    return output, int(veto.sum())


def leaf_paths(tree: DecisionTreeClassifier, feature_names: list[str]) -> dict[int, str]:
    structure = tree.tree_
    paths: dict[int, str] = {}

    def visit(node: int, clauses: list[str]) -> None:
        left = int(structure.children_left[node])
        right = int(structure.children_right[node])
        if left == right:
            paths[node] = " AND ".join(clauses) if clauses else "TRUE"
            return
        feature = feature_names[int(structure.feature[node])]
        threshold = float(structure.threshold[node])
        visit(left, [*clauses, f"{feature} <= {threshold:.8g}"])
        visit(right, [*clauses, f"{feature} > {threshold:.8g}"])

    visit(0, [])
    return paths


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
        selected_v15 = [column for column in REQUESTED_V15_FLOW_FEATURES if column in available]
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
    cohort_mask = full[ID_COL].isin(set(cohort_ids.tolist()))
    metadata = full.loc[cohort_mask, [ID_COL, TIMESTAMP, TARGET]].reset_index(drop=True)
    features = full_v15.loc[cohort_mask].reset_index(drop=True)
    del full, full_v15
    gc.collect()

    labels = metadata[TARGET].astype(str)
    classes = sorted(labels.unique().tolist())
    attack_classes = [label for label in classes if label != "Benign"]
    mapping = {label: index for index, label in enumerate(classes)}
    encoded = labels.map(mapping).to_numpy(np.int64)

    outer_train, outer_valid = class_chronological_masks(metadata, OUTER_VALID_FRACTION)
    inner_train, inner_valid = class_chronological_masks(
        metadata, INNER_VALID_FRACTION, eligible=outer_train
    )
    development_base, rule_pool = class_chronological_masks(
        metadata, RULE_POOL_FRACTION, eligible=inner_train
    )
    rule_fit, rule_calibration = class_chronological_masks(
        metadata, RULE_CALIBRATION_FRACTION, eligible=rule_pool
    )
    positions = {
        "outer_valid": np.flatnonzero(outer_valid.to_numpy()),
        "inner_train": np.flatnonzero(inner_train.to_numpy()),
        "inner_valid": np.flatnonzero(inner_valid.to_numpy()),
        "development_base": np.flatnonzero(development_base.to_numpy()),
        "rule_fit": np.flatnonzero(rule_fit.to_numpy()),
        "rule_calibration": np.flatnonzero(rule_calibration.to_numpy()),
    }
    print("Split rows:", {key: len(value) for key, value in positions.items()})

    development_model = build_v15_base(encoded[positions["development_base"]], classes)
    development_model.fit(
        features.iloc[positions["development_base"]], encoded[positions["development_base"]]
    )
    fit_base, fit_fallback, _ = multiclass_outputs(
        development_model, features.iloc[positions["rule_fit"]], classes
    )
    calibration_base, calibration_fallback, _ = multiclass_outputs(
        development_model, features.iloc[positions["rule_calibration"]], classes
    )
    fit_candidate = fit_base == INFILTERATION
    fit_target = (
        fit_fallback[fit_candidate]
        == labels.iloc[positions["rule_fit"]].to_numpy(dtype=object)[fit_candidate]
    ).astype(np.int8)
    if len(np.unique(fit_target)) != 2:
        raise RuntimeError("The rule-fit candidate set does not contain both outcomes")
    print(
        "Rule-fit Infilteration candidates:",
        int(fit_candidate.sum()),
        "fallback correct rate:",
        float(fit_target.mean()),
    )

    inner_model = build_v15_base(encoded[positions["inner_train"]], classes)
    inner_model.fit(features.iloc[positions["inner_train"]], encoded[positions["inner_train"]])
    inner_base, inner_fallback, _ = multiclass_outputs(
        inner_model, features.iloc[positions["inner_valid"]], classes
    )
    inner_base_metrics = all_metrics(
        labels.iloc[positions["inner_valid"]], inner_base, attack_classes
    )
    print("Inner v15 baseline:", inner_base_metrics)

    candidates = [
        {
            "tree_depth": 0,
            "veto_precision_threshold": 1.0,
            "min_veto_support": 0,
            "veto_rule_count": 0,
            "veto_count": 0,
            **inner_base_metrics,
        }
    ]
    trees: dict[int, DecisionTreeClassifier] = {}
    statistics_by_depth = {}
    for depth in TREE_DEPTHS:
        tree = fit_veto_tree(
            features.iloc[positions["rule_fit"]].loc[fit_candidate], fit_target, depth
        )
        trees[depth] = tree
        statistics = calibrate_veto_leaves(
            tree,
            features.iloc[positions["rule_calibration"]],
            labels.iloc[positions["rule_calibration"]],
            calibration_base,
            calibration_fallback,
        )
        statistics_by_depth[depth] = statistics
        for threshold in VETO_PRECISION_THRESHOLDS:
            for min_support in MIN_VETO_SUPPORTS:
                veto_leaves = select_veto_leaves(statistics, threshold, min_support)
                prediction, veto_count = apply_veto(
                    tree,
                    features.iloc[positions["inner_valid"]],
                    inner_base,
                    inner_fallback,
                    veto_leaves,
                )
                candidates.append(
                    {
                        "tree_depth": depth,
                        "veto_precision_threshold": threshold,
                        "min_veto_support": min_support,
                        "veto_rule_count": len(veto_leaves),
                        "veto_count": veto_count,
                        **all_metrics(
                            labels.iloc[positions["inner_valid"]], prediction, attack_classes
                        ),
                    }
                )

    candidate_frame = pd.DataFrame(candidates)
    eligible = candidate_frame[
        candidate_frame["ids_attack_recall"]
        >= inner_base_metrics["ids_attack_recall"] - MAX_IDS_RECALL_DROP
    ].copy()
    selected = eligible.sort_values(
        ["ids_attack_precision", "ids_attack_f1", "weighted_f1_attacks", "veto_count"],
        ascending=[False, False, False, True],
    ).iloc[0]
    depth = int(selected["tree_depth"])
    if depth == 0:
        selected_tree = fit_veto_tree(
            features.iloc[positions["rule_fit"]].loc[fit_candidate], fit_target, 3
        )
        veto_leaves: set[int] = set()
        selected_statistics = calibrate_veto_leaves(
            selected_tree,
            features.iloc[positions["rule_calibration"]],
            labels.iloc[positions["rule_calibration"]],
            calibration_base,
            calibration_fallback,
        )
    else:
        selected_tree = trees[depth]
        selected_statistics = statistics_by_depth[depth]
        veto_leaves = select_veto_leaves(
            selected_statistics,
            float(selected["veto_precision_threshold"]),
            int(selected["min_veto_support"]),
        )
    print("Selected inner policy:", selected.to_dict())

    configure_tracking(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"), PROJECT_EXPERIMENT)
    v15_model = mlflow.sklearn.load_model(SOURCE_V15_MODEL_URI)
    outer_features = features.iloc[positions["outer_valid"]]
    outer_base, outer_fallback, _ = multiclass_outputs(
        v15_model.base_model, outer_features, classes
    )
    outer_base_metrics = all_metrics(
        labels.iloc[positions["outer_valid"]], outer_base, attack_classes
    )
    outer_prediction, outer_veto_count = apply_veto(
        selected_tree, outer_features, outer_base, outer_fallback, veto_leaves
    )
    outer_metrics = all_metrics(
        labels.iloc[positions["outer_valid"]], outer_prediction, attack_classes
    )
    print("Outer v15 baseline:", outer_base_metrics)
    print("Outer v20b rule hybrid:", outer_metrics)
    print("Outer veto count:", outer_veto_count)

    paths = leaf_paths(selected_tree, features.columns.tolist())
    rule_rows = [
        {
            "leaf": leaf,
            "condition": paths[leaf],
            "action": "replace Infilteration with v15 second-best class",
            **selected_statistics[leaf],
        }
        for leaf in sorted(veto_leaves)
    ]
    rules_path = output_dir / "v20b_selected_veto_rules.json"
    rules_path.write_text(json.dumps(rule_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    tree_path = output_dir / "v20b_veto_tree.txt"
    tree_path.write_text(
        export_text(selected_tree, feature_names=features.columns.tolist()), encoding="utf-8"
    )
    candidates_path = output_dir / "v20b_inner_candidates.csv"
    candidate_frame.sort_values("ids_attack_precision", ascending=False).to_csv(
        candidates_path, index=False, encoding="utf-8-sig"
    )
    comparison_path = output_dir / "v20b_outer_comparison.csv"
    pd.DataFrame(
        [
            {"model": "v15_baseline", **outer_base_metrics},
            {"model": "v20b_v15_rule_hybrid", **outer_metrics},
        ]
    ).to_csv(comparison_path, index=False, encoding="utf-8-sig")

    recall_safe = (
        outer_metrics["ids_attack_recall"]
        >= outer_base_metrics["ids_attack_recall"] - MAX_IDS_RECALL_DROP
    )
    improved = (
        outer_metrics["ids_attack_precision"] > outer_base_metrics["ids_attack_precision"]
        and recall_safe
    )
    status = "full_refit_candidate" if improved else "rejected_on_outer_validation"
    manifest = {
        "selection_objective": "maximize IDS alert precision with <=1 percentage-point recall loss",
        "selected_policy": selected.to_dict(),
        "selected_veto_leaves": sorted(veto_leaves),
        "selected_rules": rule_rows,
        "outer_veto_count": outer_veto_count,
        "outer_v15_metrics": outer_base_metrics,
        "outer_v20b_metrics": outer_metrics,
        "promotion_status": status,
        "source_v15_run_id": SOURCE_V15_RUN_ID,
    }
    manifest_path = output_dir / "v20b_validation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    hybrid_model = V15RuleVetoModel(
        base_model=v15_model.base_model,
        rule_tree=selected_tree,
        label_names=classes,
        target_class_index=classes.index(INFILTERATION),
        multiplier=INFILTERATION_MULTIPLIER,
        veto_leaves=sorted(veto_leaves),
    )
    with hypothesis_run(
        run_name="v20b_v15_error_specific_rule_hybrid_validation",
        hypothesis_id=HYPOTHESIS_ID,
        hypothesis=HYPOTHESIS,
        stage="chronological_validation",
        promotion_status=status,
        validation_strategy="nested_time_split_rule_fit_calibration_inner_selection_outer_test",
        notebook="scripts/v20b_v15_error_rule_hybrid_validation.py",
        data_version=raw_path.name,
        feature_schema_version="v15-error-specific-rule-v1",
        code_version=file_sha256(Path(__file__)),
        extra_tags={
            "source_v15_run_id": SOURCE_V15_RUN_ID,
            "rule_layer": "infilteration_veto",
            "outer_labels_used_for_selection": "false",
            "operational_ids_objective": "true",
        },
    ) as run:
        mlflow.log_params(
            {
                "selected_tree_depth": depth,
                "selected_veto_precision_threshold": float(selected["veto_precision_threshold"]),
                "selected_min_veto_support": int(selected["min_veto_support"]),
                "selected_veto_rule_count": len(veto_leaves),
                "max_ids_recall_drop": MAX_IDS_RECALL_DROP,
                "source_v15_run_id": SOURCE_V15_RUN_ID,
            }
        )
        mlflow.log_metrics(
            {
                **{f"outer_v20b_{key}": float(value) for key, value in outer_metrics.items()},
                **{f"outer_v15_{key}": float(value) for key, value in outer_base_metrics.items()},
                "outer_veto_count": float(outer_veto_count),
                "delta_ids_precision": outer_metrics["ids_attack_precision"]
                - outer_base_metrics["ids_attack_precision"],
                "delta_ids_recall": outer_metrics["ids_attack_recall"]
                - outer_base_metrics["ids_attack_recall"],
                "elapsed_seconds": time.time() - started,
            }
        )
        for artifact in (rules_path, tree_path, candidates_path, comparison_path, manifest_path):
            mlflow.log_artifact(str(artifact))
        model_info = mlflow.sklearn.log_model(
            hybrid_model,
            name="v15_rule_veto_hybrid",
            input_example=outer_features.iloc[:5],
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            metadata={
                "source_v15_model_uri": SOURCE_V15_MODEL_URI,
                "veto_rule_count": len(veto_leaves),
                "operational_objective": "IDS alert precision",
            },
        )
        run_id = run.info.run_id
        model_uri = model_info.model_uri

    reloaded = mlflow.sklearn.load_model(model_uri)
    if not np.array_equal(reloaded.predict(outer_features.iloc[:100]), outer_prediction[:100]):
        raise RuntimeError("Reloaded hybrid model predictions do not match")
    print("Validation run:", run_id)
    print("Hybrid model:", model_uri)
    print("Reload verification: ok")
    print("Status:", status)


if __name__ == "__main__":
    main()
