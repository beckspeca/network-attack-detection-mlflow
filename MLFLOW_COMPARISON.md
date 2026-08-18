# MLflow model comparison

Use the `network-classification-ids-benchmark` experiment for model-to-model
comparison. It contains exactly one flat canonical run for each comparable
version from v5 onward, including rejected experiments such as v12 and
comparison-only experiments such as the standardized attached notebook v13
and probability ensemble v14.

The main `network-classification-ids` experiment is the development and
lineage experiment. It intentionally contains parent validation runs, nested
candidates, refits, and Registry packaging runs; comparing every row there is
not meaningful.

## Comparison contract

Canonical runs must have all of these tags:

- `run_role=canonical_benchmark`
- `benchmark_schema_version=ids-benchmark-v1`
- `validation_cohort_id=temporal-v5_reservoir-250000_seed-42_class-chrono-80-20`
- `source_run_id=<actual validation run>`
- `strictly_comparable=true`

They use the same 250,000-row reservoir sample, seed 42, feature-data version,
and within-class chronological 80/20 split. Core metric names are identical for
every run. The canonical run copies metrics from the actual source validation
run and keeps that run ID as immutable provenance.

Do not compare `holdout_*` metrics across versions in this experiment. The
holdout construction changed during v7-v11, so those numbers are not the same
measurement even when the names look similar.

v13 is strictly comparable as an offline benchmark, but it is tagged
`promotion_outcome=benchmark_only`. Its model consumes 30 precomputed features,
including global past-window rolling features. It must not be promoted until an
online feature service reproduces those features with event-time ordering.

v14 combines class probabilities from the v13-family RandomForest, XGBoost,
LightGBM, and hierarchical LightGBM models. Ensemble weights and the
Infilteration multiplier are selected on the first half of the validation
period; the second half remains a late holdout. It is also
`promotion_outcome=benchmark_only` because it consumes the same offline rolling
features as v13.

v15 keeps the v13 RandomForest and feature contract, then applies a fixed
`1.225` multiplier to the Infilteration probability before `argmax`. The
multiplier is fixed from prior offline analysis rather than searched during the
v15 run. It remains `promotion_outcome=benchmark_only` for the same rolling
feature-service reason as v13 and v14.

## UI workflow

1. Open the `network-classification-ids-benchmark` experiment.
2. Select the versions to compare and choose **Compare**.
3. Use these columns first: `infilteration_recall`, `infilteration_precision`,
   `infilteration_f1`, `benign_to_infilteration_fp`, and `accuracy`.
4. Use the `delta_vs_v6_*` metrics for an immediate baseline comparison.
5. Follow `source_run_id` only when model artifacts, candidate thresholds, or
   detailed evaluation artifacts are needed.

## Sync

Run this from the Jupyter container after adding a new version's approved or
rejected validation run to the explicit source map:

```bash
python scripts/sync_mlflow_benchmarks.py
```

The sync is idempotent. It verifies existing canonical metrics and refuses to
create duplicate canonical runs for the same version and schema.
