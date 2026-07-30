# TabICL (INRIA / soda-inria) — 🚧 PLACEHOLDER

> **Status: not yet implemented.** This vendor is a scaffold for a contributor to pick
> up. The structure below mirrors the `tabfm` vendor so the implementation should be a
> close parallel. Delete this banner once the notebooks are working.

[TabICLv2](https://arxiv.org/abs/2602.11139) is an in-context-learning tabular
foundation model from the soda-inria group. Like TabFM it is **self-hosted**: the
`tabicl` package downloads pretrained weights from Hugging Face and runs inference
locally. It notably scales to **million-row datasets under 50GB GPU memory**.

- Paper: https://arxiv.org/abs/2602.11139
- Code: https://github.com/soda-inria/tabicl
- Docs: https://tabicl.readthedocs.io/

## At a glance (from published docs — verify during implementation)

| | |
|---|---|
| **Consumption** | Self-hosted weights (Hugging Face) — runs locally |
| **Compute on Databricks** | **GPU cluster** (scales to very large datasets); inline in the notebook (v1) |
| **Auth** | None; downloads checkpoint from Hugging Face on first use |
| **Tasks covered** | Classification and Regression (see below re: forecast/shap extras) |
| **License** | **BSD 3-Clause** (`soda-inria/tabicl`); bundles Prior Labs-derived code under Apache-2.0 in `src/tabicl/forecast` |
| **Data limits** | Trained 300–100k samples / 2–100 features; generalizes to 600k+ rows |

Native API (sklearn-style, single-step):
```python
from tabicl import TabICLClassifier, TabICLRegressor
clf = TabICLClassifier()          # downloads checkpoint on first use
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)
```
Optional extras: `pip install tabicl[forecast]` (time series), `tabicl[shap]`,
`tabicl[finetune]`.

## What to implement

Mirror the `tabfm` vendor (`../tabfm/`). Concretely:

1. **`requirements.txt`** — pin `tabicl` (+ any extras you use), plus `scikit-learn`,
   `pandas`, `numpy`, `mlflow`, `matplotlib`. A stub is already in this folder.
2. **`notebooks/01_classification.ipynb`** — copy `../tabfm/notebooks/01_classification.ipynb`
   and swap the model layer:
   - Replace the two-step TabFM load with `from tabicl import TabICLClassifier`.
   - In `evaluate_classification`, replace the TabFM block with
     `clf = TabICLClassifier(); clf.fit(...)`.
   - Keep everything else identical (same tables, same targets, same
     `common/evaluation.py` helpers, `vendor="tabicl"` in `log_result`).
3. **`notebooks/02_regression.ipynb`** — same, with `TabICLRegressor`.
4. **(Optional) `notebooks/03_outlier_detection.ipynb` / `04_time_series_forecasting.ipynb`**
   — only if TabICL supports them well (`tabicl[forecast]` exists). Otherwise leave out
   and note the gap in the top-level vendor matrix.

### Contract to honor (so comparison stays apples-to-apples)

- Use the **shared datasets** from `shared/notebooks/00_data_preparation.ipynb` — do not
  regenerate or re-split differently. Use `split_xy(...)` from `common/evaluation.py`.
- Log every model (TabICL **and** the shared baselines) to the shared results table via
  `log_result(spark, vendor="tabicl", task=..., ...)`. The comparison notebook keys off
  the `vendor` / `task` / `model_name` columns.
- Task/target/table names are fixed by the shared data. Reuse the exact
  `table_name` + `target` pairs from the `tabfm` notebooks:
  - Classification: `supplier_delay_risk_train`/`is_delayed`,
    `material_shortage_train`/`shortage_risk`, `otif_risk_train`/`otif_risk`.
  - Regression: `price_elasticity_train`/`price_elasticity`,
    `supplier_lead_time_train`/`actual_lead_time_days`.

### Compute

Provision a GPU cluster. TabICL is designed for large data, so this is a good vendor to
also demonstrate scaling to a bigger synthetic dataset if you want (optional stretch).

### Definition of done

- `01_classification.ipynb` and `02_regression.ipynb` run end-to-end on a GPU cluster
  after the shared data-prep notebook.
- Results appear in `vendor_benchmark_results` under `vendor="tabicl"` and show up in
  `shared/notebooks/05_vendor_comparison.ipynb`.
- This README's placeholder banner is removed and the "At a glance" facts are verified.
- Update the vendor matrix in the [top-level README](../../README.md).
