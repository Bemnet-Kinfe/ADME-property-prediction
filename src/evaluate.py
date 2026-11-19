"""Metrics, chosen per task type.

Regression: MAE is the TDC leaderboard metric for Caco2_Wang; Spearman is
reported alongside it because it is scale-free and easier to compare across
tasks.

Classification: AUPRC is the headline because the positive class is the rare
one, and accuracy or ROC-AUC can look respectable while the model is useless on
the actives. The positive rate is reported next to it as the floor a random
model would reach.
"""

import warnings

import numpy as np
from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    roc_auc_score,
    root_mean_squared_error,
)


def regression_metrics(y_true, y_pred):
    # The trivial baseline predicts a constant, for which Spearman is undefined.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        rho = spearmanr(y_true, y_pred).statistic
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "spearman": float(rho) if np.isfinite(rho) else float("nan"),
    }


def classification_metrics(y_true, y_score):
    y_true = np.asarray(y_true)
    out = {"positive_rate": float(y_true.mean())}
    # A split can land with one class only; report NaN instead of crashing.
    if len(np.unique(y_true)) < 2:
        out["auprc"] = float("nan")
        out["roc_auc"] = float("nan")
        return out
    out["auprc"] = float(average_precision_score(y_true, y_score))
    out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return out


def metrics_for(task_type, y_true, y_pred):
    if task_type == "regression":
        return regression_metrics(y_true, y_pred)
    if task_type == "classification":
        return classification_metrics(y_true, y_pred)
    raise ValueError(f"unknown task type: {task_type}")


def aggregate(records, keys):
    """Mean and SD across seeds. Never report a single run."""
    out = {}
    for k in keys:
        vals = np.asarray([r[k] for r in records], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            out[k] = (float("nan"), float("nan"), 0)
        else:
            out[k] = (float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, len(vals))
    return out


def format_mean_sd(mean, sd, places=3):
    return f"{mean:.{places}f} ± {sd:.{places}f}"
