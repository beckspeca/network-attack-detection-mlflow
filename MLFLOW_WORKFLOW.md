# MLflow workflow

All new hypotheses use `network-classification-ids` for development, lineage,
refit, and Registry runs. Model-to-model comparison uses the separate
`network-classification-ids-benchmark` experiment documented in
`MLFLOW_COMPARISON.md`.

## Run structure

- Parent run: one hypothesis/version (`hypothesis_id=v10`).
- Nested child runs: individual feature, weight, model, and threshold candidates.
- Validation run: logs chronological and date-holdout metrics.
- Refit run: created only after the promotion checks pass.
- Registry version: created only from an approved refit model.

Every run must include the standard tags from `src/mlflow_workflow.py`, dataset
lineage through `mlflow.log_input`, and a dataset/code digest. Every deployable
model must include an input example and model signature.

After a hypothesis receives its final validation decision, add its actual
validation run to `scripts/sync_mlflow_benchmarks.py` and run the sync. Candidate,
refit, migration, and Registry packaging runs must never be added as canonical
benchmark rows.

## Standalone deep-learning runs

v12 establishes the independent PyTorch path. It uses the raw 40 Flow features
from `interim/train.parquet`; benchmark-cohort `unique_id` values are used only
to reproduce the exact comparison rows. Prior model predictions, routers,
weights, engineered temporal features, and thresholds are not inputs. Rejected
deep models may log a signed candidate PyFunc artifact, but they must not create
a Registry version or move the `champion` alias.

## Champion model

The current champion is v11 (Registry version 2). It reuses the validated v6
gate and subtype models, keeps the known-attack gate threshold at `0.45`, and
uses `0.395` only when the subtype model predicts Infilteration.

The current canonical offline benchmark leader is v15. It is intentionally not
the Registry champion because its past-window rolling features are not yet
available through the online feature service.

The deployable hierarchy is a single PyFunc model, not two independently
selected model artifacts. Consumers load it through a stable alias:

```python
import mlflow.pyfunc

model = mlflow.pyfunc.load_model(
    "models:/network-classification-ids@champion"
)
prediction = model.predict(feature_frame)
```

Register or verify the historical v6 model from the Jupyter container:

```bash
python scripts/register_v6_champion.py
python scripts/register_v6_champion.py --apply
```

The first command is a dry run. Re-running `--apply` is idempotent while the
`champion` version has the same `source_refit_run_id`. If a newer champion is
present, `--apply` is blocked. An intentional rollback requires both `--apply`
and `--allow-champion-rollback`.

## Security and operations

- `.env` is ignored and must remain mode `600`.
- Restrict port 5000 with a host firewall, VPN, or reverse proxy.
- Add authentication and TLS before exposing MLflow beyond a trusted network.
- Back up both the PostgreSQL volume and `mlflow/artifacts`; one without the
  other is not a complete MLflow backup.
- Apply Compose healthcheck changes during a maintenance window with
  `docker compose up -d mlflow jupyter`.
