# Data directory

- `raw/`: original downloaded source files. Keep them unchanged and document source, license, retrieval date, and SHA-256.
- `interim/`: intermediate outputs such as merged or validated data.
- `processed/`: generated model-ready data. Never describe processed or synthetic data as raw.
- Never commit private, identifiable, or restricted health data.
- Do not merge unrelated datasets without a documented key, schema mapping, provenance, and target definition.

## Current candidate

The uploaded UCI obesity-level dataset is documented in `raw/README.md` and `reports/uci_dataset_assessment.md`. It is a mixed raw-and-synthetic 2,111-row, seven-class dataset, not a pure raw-only cohort. Its license is CC BY 4.0; retain attribution.

The legacy obesity datasets are retained under `data/raw/` for provenance investigation. They are not used by the current UCI seven-class training pipeline. Their original source lineage and
binary target mapping remain unverified.
