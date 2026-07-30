# Tabular Foundation Models on Databricks

[![Databricks](https://img.shields.io/badge/Databricks-Solution_Accelerator-FF3621?style=for-the-badge&logo=databricks)](https://databricks.com)
[![Unity Catalog](https://img.shields.io/badge/Unity_Catalog-Enabled-00A1C9?style=for-the-badge)](https://docs.databricks.com/en/data-governance/unity-catalog/index.html)
[![Serverless](https://img.shields.io/badge/Serverless-Compute-00C851?style=for-the-badge)](https://docs.databricks.com/en/compute/serverless.html)

A solution accelerator demonstrating how to use **tabular foundation models** (TFMs) on
Databricks. Tabular foundation models are pretrained models that make accurate predictions
on tabular data with little-to-no task-specific training — eliminating much of the model
training, hyperparameter tuning, and per-use-case maintenance overhead of traditional ML.

> **Multi-vendor scope.** This repository is intended to host examples across multiple
> tabular foundation model vendors. The current notebooks use
> [**TabPFN**](https://docs.priorlabs.ai/) (Prior Labs) as the first vendor; support for
> additional TFM vendors will be added over time. See [Adding a vendor](#adding-a-vendor).

## Overview

### The Challenge: Enterprise-Scale Predictive Analytics

Across industries, enterprises rely on large-scale tabular prediction to drive critical
business decisions — from forecasting demand to assessing risk to optimizing operations.
These workloads often span interconnected business processes, each requiring its own
models, features, and continuous maintenance. At enterprise scale organizations may
maintain **thousands or even millions of models** across products, SKUs, regions, and
customers — substantial, costly operational overhead.

### The Solution: Foundation Models for Tabular Data

**A pretrained tabular foundation model:**
- Works **out of the box** on tabular data
- Requires **minimal preprocessing**
- Eliminates most **model training, hyperparameter tuning, and experimentation**
- Delivers performance **comparable to, or better than**, many carefully tuned models
- Often provides **built-in uncertainty quantification** (prediction intervals, not just point estimates)

## Features

| Feature | Description |
|---------|-------------|
| **Classification** | Binary and multi-class classification with probability estimates |
| **Regression** | Continuous value prediction with uncertainty quantification |
| **Outlier Detection** | Anomaly scoring using semi-supervised learning |
| **Time Series Forecasting** | Lag-based forecasting |
| **Unity Catalog Integration** | Read/write data from Delta tables |
| **Realistic Datasets** | Synthetic retail/CPG data with business-relevant features |

## Supported vendors

| Vendor | Consumption | Compute on Databricks | Tasks | License | Status |
|--------|-------------|-----------------------|-------|---------|--------|
| **[TabPFN](vendors/tabpfn/)** (Prior Labs) | Hosted API (thin client) | Serverless **CPU** | Classification · Regression · Outlier · Forecasting | `tabpfn-client` Apache-2.0; hosted model per Prior Labs terms | ✅ Available |
| **[TabFM](vendors/tabfm/)** (Google) | Self-hosted weights (HF) | **GPU** (inline) | Classification · Regression | ⚠️ Weights **TabFM Non-Commercial License v1.0** (code Apache-2.0) | ✅ Available |
| **[TabICL](vendors/tabicl/)** (soda-inria) | Self-hosted weights (HF) | **GPU** (inline) | Classification · Regression | **BSD 3-Clause** | 🚧 Placeholder |

Each vendor lives in `vendors/<vendor>/` with its own README (setup, auth, license,
compute), `requirements.txt`, and task notebooks. Pick the folder for the vendor you
care about — the notebooks use each vendor's **native** API so they're realistic and
copy-pasteable — or run several and compare.

## Project Structure

```
tabular-fm-databricks/
├── shared/
│   ├── notebooks/
│   │   ├── 00_data_preparation.ipynb    # Generate retail/CPG datasets (run first)
│   │   └── 05_vendor_comparison.ipynb   # Aggregate + compare all vendors
│   └── scripts/
│       ├── util.py                       # Synthetic data generators
│       └── cleanup.sh                    # Resource cleanup
├── common/
│   └── evaluation.py                     # Shared split, metrics, baselines, results logging
├── vendors/
│   ├── tabpfn/    (README + requirements + notebooks 01–04)
│   ├── tabfm/     (README + requirements + notebooks 01–02)
│   └── tabicl/    (README placeholder for contributor)
├── databricks.yml                        # Databricks Asset Bundle configuration
├── requirements.txt                      # Base/shared dependencies
├── CONTRIBUTING.md · LICENSE.md · NOTICE.md · SECURITY.md
```

**How it fits together.** `shared/` generates one set of Delta tables that every vendor
reads — so comparisons are apples-to-apples. `common/evaluation.py` provides the
vendor-agnostic pieces (identical train/test split, sklearn baselines, metrics) and
writes results to a single `vendor_benchmark_results` Delta table. Each vendor notebook
runs its native model on that shared data and appends its results; the comparison
notebook reads them back.

## Prerequisites

1. **Databricks Workspace** with Unity Catalog enabled
2. For a given vendor: its credentials/compute (see the vendor's README — e.g. TabPFN needs an API token; TabFM/TabICL need a GPU cluster)
3. **Databricks CLI** (optional, for local development)

## Getting Started

1. **Generate the shared data** — run [`shared/notebooks/00_data_preparation.ipynb`](shared/notebooks/00_data_preparation.ipynb). This creates the Delta tables all vendors use.
2. **Pick a vendor** — open `vendors/<vendor>/` and follow its README for setup, then run its `01`/`02`… notebooks.
3. **Compare** (optional) — after running one or more vendors, run [`shared/notebooks/05_vendor_comparison.ipynb`](shared/notebooks/05_vendor_comparison.ipynb) for a side-by-side ranking against the shared baselines.

## Compute Requirements

Depends on the vendor: **TabPFN** runs on **Serverless CPU** (it's a hosted API);
**TabFM** and **TabICL** are self-hosted and need a **GPU cluster**. Because every vendor
writes to the same results table, you can run them on different clusters and still
compare them afterward.

## Adding a vendor

New vendors are self-contained under `vendors/<vendor>/`. The quickest path is to copy an
existing vendor (TabFM is a good template for a self-hosted model) and swap the model
layer, keeping the shared data, `common/evaluation.py` helpers, and results-logging
contract intact. See [`vendors/tabicl/README.md`](vendors/tabicl/README.md) for a
worked-through contributor checklist.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and CLA information.

## License

This project is provided subject to the [Databricks License](LICENSE.md).

---

© 2025 Databricks, Inc. All rights reserved.
