# TabPFN (Prior Labs)

[TabPFN](https://docs.priorlabs.ai/) is a tabular foundation model from Prior Labs,
consumed through a **hosted API** via the `tabpfn-client` package.

## At a glance

| | |
|---|---|
| **Consumption** | Hosted API (Prior Labs) — thin HTTP client |
| **Compute on Databricks** | **Serverless CPU is sufficient** — the client just makes API calls |
| **Auth** | Prior Labs API token, stored as a Databricks Secret |
| **Tasks covered** | Classification, Regression, Outlier detection, Time-series forecasting |
| **License** | `tabpfn-client` is Apache 2.0; model use per Prior Labs terms — commercial use OK |
| **Data limits** | Practical up to ~10k rows / ~500 features per request (see Prior Labs docs) |

## Setup

1. Get an API token from [Prior Labs](https://docs.priorlabs.ai/).
2. Store it as a Databricks Secret:
   ```bash
   databricks secrets create-scope tabpfn-client
   databricks secrets put-secret tabpfn-client token
   ```
3. Run the shared data prep first: [`shared/notebooks/00_data_preparation.ipynb`](../../shared/notebooks/00_data_preparation.ipynb).

## Notebooks

| Notebook | Task |
|----------|------|
| `notebooks/01_classification.ipynb` | Binary & multi-class classification |
| `notebooks/02_regression.ipynb` | Regression with uncertainty |
| `notebooks/03_outlier_detection.ipynb` | Semi-supervised anomaly detection |
| `notebooks/04_time_series_forecasting.ipynb` | Lag-based demand forecasting |

Each notebook authenticates with:
```python
import tabpfn_client
token = dbutils.secrets.get(scope="tabpfn-client", key="token")
tabpfn_client.set_access_token(token)
```
and uses the native sklearn-style API: `TabPFNClassifier()` / `TabPFNRegressor()`
with `.fit()` / `.predict()` / `.predict_proba()`.

## Comparison

To include TabPFN results in the cross-vendor comparison
([`shared/notebooks/05_vendor_comparison.ipynb`](../../shared/notebooks/05_vendor_comparison.ipynb)),
log metrics to the shared results table via `common/evaluation.py`
(`log_result(..., vendor="tabpfn", ...)`). The existing notebooks log to MLflow;
adding a `log_result` call writes them into the shared benchmark table too.
