#!/usr/bin/env python3
"""Validate an interpretable high-precision rule layer around v15."""

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
from sklearn.tree import DecisionTreeClassifier, export_text


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from attached_temporal_features import ROLLING_SOURCE_COLUMNS, build_attachment_feature_frame  # noqa: E402
from mlflow_workflow import PROJECT_EXPERIMENT, configure_tracking, file_sha256, hypothesis_run  # noqa: E402
from v16_infilteration_hard_negative_validation import (  # noqa: E402
    ID_COL,
    INFILTERATION,
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


HYPOTHESIS_ID = "v20"
HYPOTHESIS = "v15_plus_high_precision_interpretable_rule_layer"
SOURCE_V15_RUN_ID = "4971c300af7f490bbdcd1de7606eccde"
SOURCE_V15_MODEL_URI = "models:/m-b063a6360c3246aaa77c08454ec78a3f"
INNER_VALID_FRACTION = 0.10
RULE_CALIBRATION_FRACTION = 0.20
TREE_DEPTHS = (4, 6, 8)
ATTACK_PRECISION_THRESHOLDS = (0.95, 0.975, 0.99, 0.995)
BENIGN_VETO_PRECISION_THRESHOLDS = (0.95, 0.975, 0.99, 0.995)
MIN_LEAF_SUPPORTS = (20, 50)
MIN_VETO_SUPPORTS = (10, 20)


def fit_rule_tree(
    features: pd.DataFrame,
    labels: np.ndarray,
    depth: int,
) -> DecisionTreeClassifier:
    tree = DecisionTreeClassifier(
        max_depth=int(depth),
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    tree.fit(features, labels)
    return tree


def calibrate_leaf_statistics(
    tree: DecisionTreeClassifier,
    features: pd.DataFrame,
    labels: pd.Series,
    base_prediction: np.ndarray,
    attack_classes: list[str],
) -> dict[int, dict[str, object]]:
    leaves = tree.apply(features)
    frame = pd.DataFrame(
        {
            "leaf": leaves,
            "actual": labels.to_numpy(dtype=object),
            "base_prediction": np.asarray(base_prediction, dtype=object),
        }
    )
    statistics: dict[int, dict[str, object]] = {}
    for leaf, group in frame.groupby("leaf"):
        counts = group["actual"].value_counts()
        attack_precision = {
            label: float(counts.get(label, 0) / len(group)) for label in attack_classes
        }
        best_attack = max(attack_precision, key=attack_precision.get)
        inf_candidates = group[group["base_prediction"] == INFILTERATION]
        benign_veto_precision = (
            float(inf_candidates["actual"].eq("Benign").mean())
            if len(inf_candidates)
            else 0.0
        )
        statistics[int(leaf)] = {
            "leaf_support": int(len(group)),
            "best_attack": best_attack,
            "best_attack_precision": attack_precision[best_attack],
            "attack_precision": attack_precision,
            "v15_infilteration_candidate_support": int(len(inf_candidates)),
            "benign_veto_precision": benign_veto_precision,
        }
    return statistics


def select_rules(
    leaf_statistics: dict[int, dict[str, object]],
    attack_precision: float,
    benign_veto_precision: float,
    min_leaf_support: int,
    min_veto_support: int,
) -> tuple[dict[int, str], set[int]]:
    attack_rules: dict[int, str] = {}
    benign_veto_rules: set[int] = set()
    for leaf, stats in leaf_statistics.items():
        if (
            int(stats["leaf_support"]) >= int(min_leaf_support)
            and float(stats["best_attack_precision"]) >= float(attack_precision)
        ):
            attack_rules[leaf] = str(stats["best_attack"])
        if (
            int(stats["v15_infilteration_candidate_support"]) >= int(min_veto_support)
            and float(stats["benign_veto_precision"]) >= float(benign_veto_precision)
        ):
            benign_veto_rules.add(leaf)
    return attack_rules, benign_veto_rules


def apply_rule_layer(
    tree: DecisionTreeClassifier,
    features: pd.DataFrame,
    base_prediction: np.ndarray,
    fallback_prediction: np.ndarray,
    attack_rules: dict[int, str],
    benign_veto_rules: set[int],
) -> tuple[np.ndarray, dict[str, int]]:
    leaves = tree.apply(features)
    original = np.asarray(base_prediction, dtype=object)
    output = original.copy()
    veto = (original == INFILTERATION) & np.isin(leaves, list(benign_veto_rules))
    output[veto] = np.asarray(fallback_prediction, dtype=object)[veto]
    promotions = np.zeros(len(output), dtype=bool)
    for leaf, label in attack_rules.items():
        mask = (original == "Benign") & (leaves == leaf)
        output[mask] = label
        promotions |= mask
    return output, {
        "benign_veto_count": int(veto.sum()),
        "attack_promotion_count": int(promotions.sum()),
        "total_changed": int(np.sum(output != original)),
    }


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
        selected_v15 = [c for c in REQUESTED_V15_FLOW_FEATURES if c in available]
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
    encoded = labels.map({label: index for index, label in enumerate(classes)}).to_numpy(np.int64)
    outer_train, outer_valid = class_chronological_masks(metadata, OUTER_VALID_FRACTION)
    inner_train, inner_valid = class_chronological_masks(
        metadata, INNER_VALID_FRACTION, eligible=outer_train
    )
    rule_train, rule_calibration = class_chronological_masks(
        metadata, RULE_CALIBRATION_FRACTION, eligible=inner_train
    )
    positions = {
        "outer_valid": np.flatnonzero(outer_valid.to_numpy()),
        "inner_train": np.flatnonzero(inner_train.to_numpy()),
        "inner_valid": np.flatnonzero(inner_valid.to_numpy()),
        "rule_train": np.flatnonzero(rule_train.to_numpy()),
        "rule_calibration": np.flatnonzero(rule_calibration.to_numpy()),
    }
    print("Split rows:", {key: len(value) for key, value in positions.items()})

    rule_base_model = build_v15_base(encoded[positions["rule_train"]], classes)
    rule_base_model.fit(features.iloc[positions["rule_train"]], encoded[positions["rule_train"]])
    calibration_base, calibration_fallback, _ = multiclass_outputs(
        rule_base_model, features.iloc[positions["rule_calibration"]], classes
    )
    del calibration_fallback

    inner_base_model = build_v15_base(encoded[positions["inner_train"]], classes)
    inner_base_model.fit(features.iloc[positions["inner_train"]], encoded[positions["inner_train"]])
    inner_base, inner_fallback, _ = multiclass_outputs(
        inner_base_model, features.iloc[positions["inner_valid"]], classes
    )
    inner_base_metrics = evaluate(
        labels.iloc[positions["inner_valid"]], inner_base, attack_classes
    )
    print("Inner v15 baseline:", inner_base_metrics)

    candidate_rows: list[dict[str, object]] = []
    trees: dict[int, DecisionTreeClassifier] = {}
    leaf_stats_by_depth: dict[int, dict[int, dict[str, object]]] = {}
    for depth in TREE_DEPTHS:
        tree = fit_rule_tree(
            features.iloc[positions["rule_train"]],
            encoded[positions["rule_train"]],
            depth,
        )
        trees[depth] = tree
        stats = calibrate_leaf_statistics(
            tree,
            features.iloc[positions["rule_calibration"]],
            labels.iloc[positions["rule_calibration"]],
            calibration_base,
            attack_classes,
        )
        leaf_stats_by_depth[depth] = stats
        for attack_precision in ATTACK_PRECISION_THRESHOLDS:
            for benign_precision in BENIGN_VETO_PRECISION_THRESHOLDS:
                for min_leaf_support in MIN_LEAF_SUPPORTS:
                    for min_veto_support in MIN_VETO_SUPPORTS:
                        attack_rules, benign_rules = select_rules(
                            stats,
                            attack_precision,
                            benign_precision,
                            min_leaf_support,
                            min_veto_support,
                        )
                        prediction, changes = apply_rule_layer(
                            tree,
                            features.iloc[positions["inner_valid"]],
                            inner_base,
                            inner_fallback,
                            attack_rules,
                            benign_rules,
                        )
                        metrics = evaluate(
                            labels.iloc[positions["inner_valid"]], prediction, attack_classes
                        )
                        candidate_rows.append(
                            {
                                "tree_depth": depth,
                                "attack_precision_threshold": attack_precision,
                                "benign_veto_precision_threshold": benign_precision,
                                "min_leaf_support": min_leaf_support,
                                "min_veto_support": min_veto_support,
                                "attack_rule_count": len(attack_rules),
                                "benign_veto_rule_count": len(benign_rules),
                                **changes,
                                **metrics,
                            }
                        )

    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["weighted_f1_attacks", "infilteration_f1", "accuracy", "total_changed"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    selected = candidates.iloc[0]
    depth = int(selected["tree_depth"])
    attack_precision = float(selected["attack_precision_threshold"])
    benign_precision = float(selected["benign_veto_precision_threshold"])
    min_leaf_support = int(selected["min_leaf_support"])
    min_veto_support = int(selected["min_veto_support"])
    selected_tree = trees[depth]
    selected_attack_rules, selected_benign_rules = select_rules(
        leaf_stats_by_depth[depth],
        attack_precision,
        benign_precision,
        min_leaf_support,
        min_veto_support,
    )
    print("Selected rule policy:", selected.to_dict())

    configure_tracking(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"), PROJECT_EXPERIMENT)
    v15_model = mlflow.sklearn.load_model(SOURCE_V15_MODEL_URI)
    outer_base, outer_fallback, _ = multiclass_outputs(
        v15_model.base_model, features.iloc[positions["outer_valid"]], classes
    )
    outer_base_metrics = evaluate(
        labels.iloc[positions["outer_valid"]], outer_base, attack_classes
    )
    outer_prediction, outer_changes = apply_rule_layer(
        selected_tree,
        features.iloc[positions["outer_valid"]],
        outer_base,
        outer_fallback,
        selected_attack_rules,
        selected_benign_rules,
    )
    outer_metrics = evaluate(
        labels.iloc[positions["outer_valid"]], outer_prediction, attack_classes
    )
    print("Outer v15 baseline:", outer_base_metrics)
    print("Outer v20 rule hybrid:", outer_metrics)
    print("Outer rule changes:", outer_changes)

    paths = leaf_paths(selected_tree, features.columns.tolist())
    selected_rule_rows: list[dict[str, object]] = []
    stats = leaf_stats_by_depth[depth]
    for leaf, attack_label in selected_attack_rules.items():
        selected_rule_rows.append(
            {
                "rule_type": "attack_promotion",
                "leaf": leaf,
                "action": attack_label,
                "condition": paths[leaf],
                **stats[leaf],
            }
        )
    for leaf in sorted(selected_benign_rules):
        selected_rule_rows.append(
            {
                "rule_type": "infilteration_benign_veto",
                "leaf": leaf,
                "action": "v15_non_infilteration_fallback",
                "condition": paths[leaf],
                **stats[leaf],
            }
        )
    rules_path = output_dir / "v20_selected_rules.json"
    rules_path.write_text(
        json.dumps(selected_rule_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tree_text_path = output_dir / "v20_rule_tree.txt"
    tree_text_path.write_text(
        export_text(selected_tree, feature_names=features.columns.tolist()), encoding="utf-8"
    )
    candidates_path = output_dir / "v20_inner_rule_policy_candidates.csv"
    candidates.to_csv(candidates_path, index=False, encoding="utf-8-sig")
    comparison_path = output_dir / "v20_outer_comparison.csv"
    pd.DataFrame(
        [
            {"model": "v15_baseline", **outer_base_metrics},
            {"model": "v20_v15_rule_hybrid", **outer_metrics},
        ]
    ).to_csv(comparison_path, index=False, encoding="utf-8-sig")
    manifest = {
        "selected_tree_depth": depth,
        "selected_attack_precision_threshold": attack_precision,
        "selected_benign_veto_precision_threshold": benign_precision,
        "selected_min_leaf_support": min_leaf_support,
        "selected_min_veto_support": min_veto_support,
        "attack_rule_count": len(selected_attack_rules),
        "benign_veto_rule_count": len(selected_benign_rules),
        "outer_rule_changes": outer_changes,
        "outer_v15_metrics": outer_base_metrics,
        "outer_v20_metrics": outer_metrics,
        "source_v15_run_id": SOURCE_V15_RUN_ID,
    }
    manifest_path = output_dir / "v20_validation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    improved = outer_metrics["weighted_f1_attacks"] > outer_base_metrics["weighted_f1_attacks"]
    status = "full_refit_candidate" if improved else "rejected_on_outer_validation"
    with hypothesis_run(
        run_name="v20_v15_high_precision_rule_hybrid_validation",
        hypothesis_id=HYPOTHESIS_ID,
        hypothesis=HYPOTHESIS,
        stage="chronological_validation",
        promotion_status=status,
        validation_strategy="rule_train_calibration_inner_selection_outer_chronological_evaluation",
        notebook="scripts/v20_v15_rule_hybrid_validation.py",
        data_version=raw_path.name,
        feature_schema_version="v15-interpretable-rule-layer-v1",
        code_version=file_sha256(Path(__file__)),
        extra_tags={
            "source_v15_run_id": SOURCE_V15_RUN_ID,
            "rule_layer": "true",
            "absolute_timestamp_features": "false",
            "outer_labels_used_for_selection": "false",
        },
    ) as run:
        mlflow.log_params(
            {
                "selected_tree_depth": depth,
                "selected_attack_precision_threshold": attack_precision,
                "selected_benign_veto_precision_threshold": benign_precision,
                "selected_min_leaf_support": min_leaf_support,
                "selected_min_veto_support": min_veto_support,
                "attack_rule_count": len(selected_attack_rules),
                "benign_veto_rule_count": len(selected_benign_rules),
                "source_v15_run_id": SOURCE_V15_RUN_ID,
            }
        )
        mlflow.log_metrics(
            {
                **{f"outer_v20_{key}": float(value) for key, value in outer_metrics.items()},
                **{f"outer_v15_{key}": float(value) for key, value in outer_base_metrics.items()},
                **{f"outer_rule_{key}": float(value) for key, value in outer_changes.items()},
                "delta_weighted_f1_attacks": outer_metrics["weighted_f1_attacks"]
                - outer_base_metrics["weighted_f1_attacks"],
                "delta_infilteration_f1": outer_metrics["infilteration_f1"]
                - outer_base_metrics["infilteration_f1"],
                "elapsed_seconds": time.time() - started,
            }
        )
        for artifact in (
            rules_path,
            tree_text_path,
            candidates_path,
            comparison_path,
            manifest_path,
        ):
            mlflow.log_artifact(str(artifact))
        input_example = features.iloc[positions["outer_valid"][:5]].copy()
        model_info = mlflow.sklearn.log_model(
            selected_tree,
            name="interpretable_rule_tree",
            input_example=input_example,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            metadata={
                "attack_rule_count": len(selected_attack_rules),
                "benign_veto_rule_count": len(selected_benign_rules),
                "source_v15_model_uri": SOURCE_V15_MODEL_URI,
            },
        )
        run_id = run.info.run_id
        model_uri = model_info.model_uri

    print("Validation run:", run_id)
    print("Rule tree model:", model_uri)
    print("Status:", status)


if __name__ == "__main__":
    main()
