"""
Shared evaluation utilities for the multi-vendor tabular foundation models accelerator.

The goal of this module is to make every vendor comparable on the *same* footing:
same train/test split, same baselines, same metrics, and a single results schema
written to a Delta table that the comparison notebook (``05_vendor_comparison``) reads.

Vendor notebooks stay "native" — they instantiate and call each vendor's own model
directly (e.g. ``TabFMClassifier().fit(...)``). They only lean on this module for the
vendor-agnostic parts: baselines, metric computation, and logging results in a common
shape. Keeping that logic here (rather than copy-pasted per vendor) is what keeps the
cross-vendor comparison apples-to-apples.

Usage sketch (inside a vendor notebook)::

    from evaluation import (
        split_xy, classification_metrics, regression_metrics,
        train_baselines_classification, log_result, RESULTS_TABLE,
    )

    X_train, X_test, y_train, y_test = split_xy(df, target="is_delayed", test_size=0.2)

    # vendor model (native call)
    clf = TabFMClassifier(model=...); clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test); y_proba = clf.predict_proba(X_test)
    metrics = classification_metrics(y_test, y_pred, y_proba)

    log_result(spark, vendor="tabfm", task="supplier_delay_risk",
               problem_type="binary_classification", model_name="TabFMClassifier",
               metrics=metrics, n_train=len(X_train), n_test=len(X_test),
               n_features=X_train.shape[1])

    # baselines for the same split (logged under vendor="baseline")
    for name, m in train_baselines_classification(X_train, y_train, X_test, y_test).items():
        log_result(spark, vendor="baseline", task="supplier_delay_risk",
                   problem_type="binary_classification", model_name=name,
                   metrics=m, n_train=len(X_train), n_test=len(X_test),
                   n_features=X_train.shape[1])
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# Fully-qualified Delta table that all vendors append their results to.
# Keep in sync with CATALOG/SCHEMA used by shared/notebooks/00_data_preparation.
RESULTS_TABLE = "vendor_benchmark_results"

# Column order for the results table / DataFrame. One row per (vendor, task, model).
RESULT_COLUMNS = [
    "run_ts",           # UTC timestamp of the evaluation
    "vendor",           # e.g. "tabpfn", "tabfm", "tabicl", or "baseline"
    "task",             # e.g. "supplier_delay_risk"
    "problem_type",     # "binary_classification" | "multiclass_classification" | "regression"
    "model_name",       # e.g. "TabFMClassifier", "RandomForest"
    # classification metrics (null for regression)
    "accuracy",
    "roc_auc",
    "f1",
    # regression metrics (null for classification)
    "mae",
    "rmse",
    "r2",
    # shared context
    "n_train",
    "n_test",
    "n_features",
]

RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Data splitting
# --------------------------------------------------------------------------- #
def split_xy(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2,
    stratify: bool = False,
    random_state: int = RANDOM_STATE,
):
    """Split a pandas DataFrame into X_train, X_test, y_train, y_test.

    Deterministic by default so every vendor evaluates the exact same rows.
    Set ``stratify=True`` for classification with class imbalance.
    """
    X = df.drop(columns=[target])
    y = df[target]
    strat = y if stratify else None
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def classification_metrics(
    y_true,
    y_pred,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, Optional[float]]:
    """Compute a common classification metric set.

    ``roc_auc`` is computed when ``y_proba`` is provided:
      * binary -> uses the positive-class column (``y_proba[:, 1]``)
      * multiclass -> one-vs-rest, macro-averaged
    Returns ``None`` for roc_auc if it can't be computed.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_classes = len(np.unique(y_true))

    roc_auc: Optional[float] = None
    if y_proba is not None:
        try:
            if n_classes == 2:
                proba_pos = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                roc_auc = float(roc_auc_score(y_true, proba_pos))
            else:
                roc_auc = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
                )
        except (ValueError, IndexError):
            roc_auc = None

    average = "binary" if n_classes == 2 else "macro"
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": roc_auc,
        "f1": float(f1_score(y_true, y_pred, average=average)),
        "mae": None,
        "rmse": None,
        "r2": None,
    }


def regression_metrics(y_true, y_pred) -> Dict[str, Optional[float]]:
    """Compute a common regression metric set."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "accuracy": None,
        "roc_auc": None,
        "f1": None,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


# --------------------------------------------------------------------------- #
# Baselines (shared so every vendor is compared against identical references)
# --------------------------------------------------------------------------- #
def _preprocessor(X: pd.DataFrame, scale: bool = False):
    """Build a ColumnTransformer that encodes categoricals and (optionally) scales numerics.

    The shared datasets mix numeric and string/categorical columns (e.g.
    ``supplier_tier='Conditional'``). The tabular foundation models encode
    categoricals internally, but the sklearn baselines can't fit on raw strings —
    so wrap every baseline in this preprocessor to keep the comparison fair and
    the baselines from crashing on categorical inputs.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    cat_cols = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype) == "category"]
    num_cols = [c for c in X.columns if c not in cat_cols]
    num_tf = StandardScaler() if scale else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", num_tf, num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )


def train_baselines_classification(
    X_train, y_train, X_test, y_test, random_state: int = RANDOM_STATE
) -> Dict[str, Dict[str, Optional[float]]]:
    """Train standard classification baselines and return {model_name: metrics}.

    Each baseline is wrapped in a preprocessing pipeline (one-hot for categoricals,
    optional scaling for linear models) so it handles the mixed-type datasets.
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    # (name, estimator, needs_scaling)
    specs = [
        ("RandomForest", RandomForestClassifier(n_estimators=200, random_state=random_state), False),
        ("GradientBoosting", GradientBoostingClassifier(random_state=random_state), False),
        ("LogisticRegression", LogisticRegression(max_iter=1000, random_state=random_state), True),
    ]
    results: Dict[str, Dict[str, Optional[float]]] = {}
    for name, est, scale in specs:
        model = Pipeline([("pre", _preprocessor(X_train, scale=scale)), ("est", est)])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        results[name] = classification_metrics(y_test, y_pred, y_proba)
    return results


def train_baselines_regression(
    X_train, y_train, X_test, y_test, random_state: int = RANDOM_STATE
) -> Dict[str, Dict[str, Optional[float]]]:
    """Train standard regression baselines and return {model_name: metrics}.

    Each baseline is wrapped in a preprocessing pipeline (one-hot for categoricals,
    optional scaling for linear models) so it handles the mixed-type datasets.
    """
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline

    specs = [
        ("RandomForest", RandomForestRegressor(n_estimators=200, random_state=random_state), False),
        ("GradientBoosting", GradientBoostingRegressor(random_state=random_state), False),
        ("Ridge", Ridge(random_state=random_state), True),
    ]
    results: Dict[str, Dict[str, Optional[float]]] = {}
    for name, est, scale in specs:
        model = Pipeline([("pre", _preprocessor(X_train, scale=scale)), ("est", est)])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = regression_metrics(y_test, y_pred)
    return results


# --------------------------------------------------------------------------- #
# Results logging (common schema -> Delta)
# --------------------------------------------------------------------------- #
def result_row(
    vendor: str,
    task: str,
    problem_type: str,
    model_name: str,
    metrics: Dict[str, Optional[float]],
    n_train: int,
    n_test: int,
    n_features: int,
    run_ts: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble a single results row (dict) in the common schema."""
    row = {
        "run_ts": run_ts or _dt.datetime.utcnow().isoformat(),
        "vendor": vendor,
        "task": task,
        "problem_type": problem_type,
        "model_name": model_name,
        "n_train": int(n_train),
        "n_test": int(n_test),
        "n_features": int(n_features),
    }
    for key in ("accuracy", "roc_auc", "f1", "mae", "rmse", "r2"):
        row[key] = metrics.get(key)
    return {c: row[c] for c in RESULT_COLUMNS}


def log_result(
    spark,
    vendor: str,
    task: str,
    problem_type: str,
    model_name: str,
    metrics: Dict[str, Optional[float]],
    n_train: int,
    n_test: int,
    n_features: int,
    table: str = RESULTS_TABLE,
    run_ts: Optional[str] = None,
) -> None:
    """Append one result row to the shared Delta results table.

    ``spark`` is the active SparkSession (available as ``spark`` in Databricks
    notebooks). The table is created on first write. The comparison notebook reads
    from ``table`` and pivots by vendor/model.
    """
    from pyspark.sql.types import (
        StructType, StructField, StringType, DoubleType, LongType,
    )

    row = result_row(
        vendor, task, problem_type, model_name, metrics,
        n_train, n_test, n_features, run_ts=run_ts,
    )

    # Build the DataFrame with an EXPLICIT schema. Without it, Spark infers types
    # from the single row: a classification result has mae/rmse/r2 all None, which
    # Spark types as NullType and effectively drops from the saved table — so a
    # later regression write fails with "could not find mae". Pinning the schema
    # guarantees all 14 columns always exist with stable types.
    metric_cols = {"accuracy", "roc_auc", "f1", "mae", "rmse", "r2"}
    long_cols = {"n_train", "n_test", "n_features"}
    fields = []
    for c in RESULT_COLUMNS:
        if c in metric_cols:
            fields.append(StructField(c, DoubleType(), True))
        elif c in long_cols:
            fields.append(StructField(c, LongType(), True))
        else:
            fields.append(StructField(c, StringType(), True))
    schema = StructType(fields)

    # Coerce values to match (None stays None; metrics -> float; counts -> int).
    typed = {}
    for c in RESULT_COLUMNS:
        v = row[c]
        if v is None:
            typed[c] = None
        elif c in metric_cols:
            typed[c] = float(v)
        elif c in long_cols:
            typed[c] = int(v)
        else:
            typed[c] = str(v)

    (
        spark.createDataFrame([typed], schema=schema)
        .write.mode("append")
        .option("mergeSchema", "true")
        .saveAsTable(table)
    )
