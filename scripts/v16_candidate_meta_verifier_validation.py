#!/usr/bin/env python3
"""Second v16 experiment: verify only leak-free v15 Infilteration candidates.

The specialist is trained on rows that an earlier v15 model falsely or truly
raised as Infilteration.  Full row-level features are augmented with v15 class
scores and margins.  Specialist threshold selection uses grouped OOF
predictions inside the inner candidate set; the outer chronological holdout is
untouched until the final evaluation.
"""

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
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedGroupKFold


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from attached_temporal_features import ROLLING_SOURCE_COLUMNS, build_attachment_feature_frame  # noqa: E402
from mlflow_workflow import PROJECT_EXPERIMENT, configure_tracking, file_sha256, hypothesis_run  # noqa: E402
from v16_infilteration_hard_negative_validation import (  # noqa: E402
    CAMPAIGN_DATES,
    ID_COL,
    INFILTERATION,
    INFILTERATION_MULTIPLIER,
    OUTER_VALID_FRACTION,
    RANDOM_STATE,
    REQUESTED_V15_FLOW_FEATURES,
    TARGET,
    TIMESTAMP,
    TRAIN_SAMPLE,
    build_specialist_raw_features,
    build_v15_base,
    class_chronological_masks,
    evaluate,
    filter_infilteration,
    resolve_dataset_paths,
)


HYPOTHESIS_ID = "v16b"
HYPOTHESIS = "v15_candidate_only_full_flow_score_margin_meta_verifier"
SOURCE_V15_RUN_ID = "4971c300af7f490bbdcd1de7606eccde"
SOURCE_V15_MODEL_URI = "models:/m-b063a6360c3246aaa77c08454ec78a3f"
INNER_VALID_FRACTION = 0.10
POSITIVE_WEIGHTS = (1.0, 2.0, 4.0, 8.0, 16.0)
FEATURE_SETS = ("raw_meta", "raw_context_meta")
THRESHOLDS = np.round(np.linspace(0.0, 1.0, 101), 4)


def build_meta_model() -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        n_estimators=250,
        learning_rate=0.035,
        num_leaves=15,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.5,
        reg_lambda=2.0,
        n_jobs=4,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )


def v15_outputs(base_model, frame: pd.DataFrame, classes: list[str]):
    probabilities = np.asarray(base_model.predict_proba(frame), dtype=float)
    encoded_classes = np.asarray(base_model.classes_).astype(int)
    aligned = np.zeros((len(frame), len(classes)), dtype=float)
    for column, encoded_class in enumerate(encoded_classes):
        aligned[:, encoded_class] = probabilities[:, column]
    scores = aligned.copy()
    inf_index = classes.index(INFILTERATION)
    scores[:, inf_index] *= INFILTERATION_MULTIPLIER
    labels = np.asarray(classes, dtype=object)
    prediction = labels[scores.argmax(axis=1)]
    non_inf_scores = scores.copy()
    non_inf_scores[:, inf_index] = -np.inf
    fallback = labels[non_inf_scores.argmax(axis=1)]
    return prediction, fallback, aligned, scores


def augment_with_v15_meta(
    base_features: pd.DataFrame,
    probabilities: np.ndarray,
    scores: np.ndarray,
    classes: list[str],
) -> pd.DataFrame:
    output = base_features.reset_index(drop=True).copy()
    for index, label in enumerate(classes):
        slug = label.lower().replace(" ", "_")
        output[f"v15_probability__{slug}"] = probabilities[:, index].astype(np.float32)
        output[f"v15_adjusted_score__{slug}"] = scores[:, index].astype(np.float32)
    inf_index = classes.index(INFILTERATION)
    non_inf = scores.copy()
    non_inf[:, inf_index] = -np.inf
    best_other = non_inf.max(axis=1)
    inf_score = scores[:, inf_index]
    output["v15_inf_to_best_other_ratio"] = (inf_score / (best_other + 1e-8)).astype(np.float32)
    output["v15_inf_margin"] = (inf_score - best_other).astype(np.float32)
    clipped = np.clip(probabilities, 1e-12, 1.0)
    output["v15_probability_entropy"] = (-(clipped * np.log(clipped)).sum(axis=1)).astype(np.float32)
    return output


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
        v15_features = full_v15.loc[cohort_mask].reset_index(drop=True)
        del full, full_v15
        gc.collect()

        connection.register("v16b_ids", metadata[[ID_COL]])
        raw_sample = connection.execute(
            f"SELECT raw.* FROM read_parquet('{raw_path}') raw "
            f"INNER JOIN v16b_ids ids USING ({ID_COL})"
        ).df()

    raw_sample = raw_sample.set_index(ID_COL).loc[metadata[ID_COL]].reset_index()
    raw_features = build_specialist_raw_features(raw_sample)
    del raw_sample
    temporal_columns = [c for c in v15_features.columns if c not in selected_v15]
    context_features = v15_features[temporal_columns].astype(np.float32).copy()
    context_features.columns = [f"context__{column}" for column in temporal_columns]

    labels = metadata[TARGET].astype(str)
    classes = sorted(labels.unique().tolist())
    attack_classes = [label for label in classes if label != "Benign"]
    encoded = labels.map({label: index for index, label in enumerate(classes)}).to_numpy(np.int64)
    outer_train, outer_valid = class_chronological_masks(metadata, OUTER_VALID_FRACTION)
    inner_train, inner_valid = class_chronological_masks(
        metadata, INNER_VALID_FRACTION, eligible=outer_train
    )
    inner_train_pos = np.flatnonzero(inner_train.to_numpy())
    inner_valid_pos = np.flatnonzero(inner_valid.to_numpy())
    outer_valid_pos = np.flatnonzero(outer_valid.to_numpy())

    inner_model = build_v15_base(encoded[inner_train_pos], classes)
    inner_model.fit(v15_features.iloc[inner_train_pos], encoded[inner_train_pos])
    inner_prediction, inner_fallback, inner_probability, inner_scores = v15_outputs(
        inner_model, v15_features.iloc[inner_valid_pos], classes
    )
    inner_labels = labels.iloc[inner_valid_pos].reset_index(drop=True)
    inner_base_metrics = evaluate(inner_labels, inner_prediction, attack_classes)
    inner_candidate = inner_prediction == INFILTERATION
    print("Inner baseline:", inner_base_metrics)
    print(
        "Inner candidates / positives / negatives:",
        int(inner_candidate.sum()),
        int(np.sum(inner_candidate & inner_labels.eq(INFILTERATION).to_numpy())),
        int(np.sum(inner_candidate & ~inner_labels.eq(INFILTERATION).to_numpy())),
    )

    inner_raw = raw_features.iloc[inner_valid_pos].reset_index(drop=True)
    inner_context = context_features.iloc[inner_valid_pos].reset_index(drop=True)
    inner_sets = {
        "raw_meta": augment_with_v15_meta(
            inner_raw, inner_probability, inner_scores, classes
        ),
        "raw_context_meta": augment_with_v15_meta(
            pd.concat([inner_raw, inner_context], axis=1),
            inner_probability,
            inner_scores,
            classes,
        ),
    }
    candidate_indices = np.flatnonzero(inner_candidate)
    candidate_target = inner_labels.iloc[candidate_indices].eq(INFILTERATION).astype(np.int8).to_numpy()
    if candidate_target.sum() < 30:
        raise RuntimeError(f"Too few true Infilteration candidates: {candidate_target.sum()}")
    fingerprint_groups = pd.util.hash_pandas_object(
        inner_raw.iloc[candidate_indices], index=False
    ).to_numpy(dtype=np.uint64)
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    candidate_rows: list[dict[str, object]] = []
    for feature_set in FEATURE_SETS:
        candidate_frame = inner_sets[feature_set].iloc[candidate_indices].reset_index(drop=True)
        for positive_weight in POSITIVE_WEIGHTS:
            oof_probability = np.zeros(len(candidate_indices), dtype=float)
            for train_local, valid_local in splitter.split(
                candidate_frame, candidate_target, groups=fingerprint_groups
            ):
                model = build_meta_model()
                weights = np.where(candidate_target[train_local] == 1, positive_weight, 1.0)
                model.fit(
                    candidate_frame.iloc[train_local],
                    candidate_target[train_local],
                    sample_weight=weights,
                )
                oof_probability[valid_local] = model.predict_proba(
                    candidate_frame.iloc[valid_local]
                )[:, 1]
            full_probability = np.ones(len(inner_labels), dtype=float)
            full_probability[candidate_indices] = oof_probability
            for threshold in THRESHOLDS:
                prediction = filter_infilteration(
                    inner_prediction, inner_fallback, full_probability, float(threshold)
                )
                metrics = evaluate(inner_labels, prediction, attack_classes)
                candidate_rows.append(
                    {
                        "feature_set": feature_set,
                        "positive_weight": positive_weight,
                        "threshold": float(threshold),
                        **metrics,
                    }
                )

    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["weighted_f1_attacks", "infilteration_f1", "infilteration_precision", "threshold"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    selected = candidates.iloc[0]
    feature_set = str(selected["feature_set"])
    positive_weight = float(selected["positive_weight"])
    threshold = float(selected["threshold"])
    print("Selected candidate-only verifier:", selected.to_dict())

    final_inner_candidate_frame = inner_sets[feature_set].iloc[candidate_indices].reset_index(drop=True)
    specialist = build_meta_model()
    specialist.fit(
        final_inner_candidate_frame,
        candidate_target,
        sample_weight=np.where(candidate_target == 1, positive_weight, 1.0),
    )

    configure_tracking(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"), PROJECT_EXPERIMENT)
    v15_model = mlflow.sklearn.load_model(SOURCE_V15_MODEL_URI)
    outer_prediction, outer_fallback, outer_probability, outer_scores = v15_outputs(
        v15_model.base_model, v15_features.iloc[outer_valid_pos], classes
    )
    outer_labels = labels.iloc[outer_valid_pos].reset_index(drop=True)
    outer_base_metrics = evaluate(outer_labels, outer_prediction, attack_classes)
    outer_raw = raw_features.iloc[outer_valid_pos].reset_index(drop=True)
    outer_context = context_features.iloc[outer_valid_pos].reset_index(drop=True)
    outer_base_frame = (
        outer_raw
        if feature_set == "raw_meta"
        else pd.concat([outer_raw, outer_context], axis=1)
    )
    outer_meta = augment_with_v15_meta(
        outer_base_frame, outer_probability, outer_scores, classes
    )
    outer_candidate = outer_prediction == INFILTERATION
    outer_specialist_probability = np.ones(len(outer_labels), dtype=float)
    outer_specialist_probability[outer_candidate] = specialist.predict_proba(
        outer_meta.loc[outer_candidate]
    )[:, 1]
    v16_prediction = filter_infilteration(
        outer_prediction,
        outer_fallback,
        outer_specialist_probability,
        threshold,
    )
    outer_metrics = evaluate(outer_labels, v16_prediction, attack_classes)
    print("Outer v15 baseline:", outer_base_metrics)
    print("Outer v16b result:", outer_metrics)

    # Diagnostic only: describe score stability without using it for selection.
    diagnostic = pd.DataFrame(
        {
            "actual": outer_labels,
            "base_prediction": outer_prediction,
            "verifier_probability": outer_specialist_probability,
        }
    )
    diagnostic = diagnostic.loc[outer_candidate]
    diagnostic_summary = (
        diagnostic.assign(is_infilteration=diagnostic["actual"].eq(INFILTERATION))
        .groupby("is_infilteration")["verifier_probability"]
        .agg(["count", "min", "median", "mean", "max"])
        .reset_index()
    )

    candidates_path = output_dir / "v16b_inner_grouped_oof_candidates.csv"
    candidates.to_csv(candidates_path, index=False, encoding="utf-8-sig")
    comparison_path = output_dir / "v16b_outer_comparison.csv"
    pd.DataFrame(
        [
            {"model": "v15_baseline", **outer_base_metrics},
            {"model": "v16b_candidate_meta_verifier", **outer_metrics},
        ]
    ).to_csv(comparison_path, index=False, encoding="utf-8-sig")
    diagnostic_path = output_dir / "v16b_outer_probability_diagnostic.csv"
    diagnostic_summary.to_csv(diagnostic_path, index=False, encoding="utf-8-sig")
    manifest = {
        "selected_feature_set": feature_set,
        "selected_positive_weight": positive_weight,
        "selected_threshold": threshold,
        "inner_candidate_count": int(inner_candidate.sum()),
        "inner_candidate_true_count": int(candidate_target.sum()),
        "outer_candidate_count": int(outer_candidate.sum()),
        "outer_base_metrics": outer_base_metrics,
        "outer_v16b_metrics": outer_metrics,
        "source_v15_run_id": SOURCE_V15_RUN_ID,
        "source_v15_model_uri": SOURCE_V15_MODEL_URI,
    }
    manifest_path = output_dir / "v16b_validation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    improved = outer_metrics["weighted_f1_attacks"] > outer_base_metrics["weighted_f1_attacks"]
    status = "full_refit_candidate" if improved else "rejected_on_outer_validation"
    with hypothesis_run(
        run_name="v16b_candidate_only_score_margin_meta_verifier_validation",
        hypothesis_id=HYPOTHESIS_ID,
        hypothesis=HYPOTHESIS,
        stage="chronological_validation",
        promotion_status=status,
        validation_strategy="inner_candidate_grouped_oof_threshold_outer_chronological_evaluation",
        notebook="scripts/v16_candidate_meta_verifier_validation.py",
        data_version=raw_path.name,
        feature_schema_version="all-row-flow-v15-probability-margin-meta-v1",
        code_version=file_sha256(Path(__file__)),
        extra_tags={
            "source_v15_run_id": SOURCE_V15_RUN_ID,
            "filter_only": "true",
            "outer_labels_used_for_selection": "false",
        },
    ) as run:
        mlflow.log_params(
            {
                "selected_feature_set": feature_set,
                "selected_positive_weight": positive_weight,
                "selected_threshold": threshold,
                "inner_candidate_count": int(inner_candidate.sum()),
                "inner_candidate_true_count": int(candidate_target.sum()),
                "outer_candidate_count": int(outer_candidate.sum()),
                "source_v15_run_id": SOURCE_V15_RUN_ID,
            }
        )
        mlflow.log_metrics(
            {
                **{f"outer_v16b_{key}": float(value) for key, value in outer_metrics.items()},
                **{f"outer_v15_{key}": float(value) for key, value in outer_base_metrics.items()},
                **{f"inner_v15_{key}": float(value) for key, value in inner_base_metrics.items()},
                "delta_weighted_f1_attacks": outer_metrics["weighted_f1_attacks"]
                - outer_base_metrics["weighted_f1_attacks"],
                "delta_infilteration_f1": outer_metrics["infilteration_f1"]
                - outer_base_metrics["infilteration_f1"],
                "elapsed_seconds": time.time() - started,
            }
        )
        for artifact in (candidates_path, comparison_path, diagnostic_path, manifest_path):
            mlflow.log_artifact(str(artifact))
        input_example = final_inner_candidate_frame.iloc[:5].copy()
        model_info = mlflow.sklearn.log_model(
            specialist,
            name="candidate_meta_verifier",
            input_example=input_example,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            metadata={
                "filter_only": True,
                "selected_threshold": threshold,
                "source_v15_model_uri": SOURCE_V15_MODEL_URI,
            },
        )
        run_id = run.info.run_id
        model_uri = model_info.model_uri

    print("Validation run:", run_id)
    print("Meta verifier model:", model_uri)
    print("Status:", status)


if __name__ == "__main__":
    main()
