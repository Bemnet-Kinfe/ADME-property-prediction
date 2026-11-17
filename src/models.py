"""The baseline: gradient boosting on descriptors plus Morgan fingerprints.

This is not a formality. On ADME datasets of a few hundred to a few thousand
molecules, published comparisons repeatedly find this competitive with, and
sometimes better than, graph neural networks. If that is what the GNN phase
finds, that is the result.
"""

import warnings

import lightgbm as lgb
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

BASE_PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 10,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.6,
    "reg_lambda": 1.0,
    "verbose": -1,
    "n_jobs": -1,
}

EARLY_STOPPING_ROUNDS = 100


def fit_predict(task_type, X_train, y_train, X_valid, y_valid, X_test, seed):
    """Train with early stopping on the validation fold, predict the test fold."""
    params = dict(BASE_PARAMS, random_state=seed)

    if task_type == "regression":
        model = lgb.LGBMRegressor(objective="l2", **params)
        eval_metric = "l1"
    else:
        model = lgb.LGBMClassifier(objective="binary", **params)
        eval_metric = "average_precision"

    model.fit(
        X_train,
        y_train,
        eval_X=X_valid,
        eval_y=y_valid,
        eval_metric=eval_metric,
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )

    if task_type == "regression":
        pred = model.predict(X_test)
    else:
        pred = model.predict_proba(X_test)[:, 1]

    return np.asarray(pred), int(model.best_iteration_ or params["n_estimators"])


def predict_mean_baseline(task_type, y_train, n_test):
    """The trivial baseline: predict the training mean / prevalence for everything.

    Every reported number needs this floor next to it. A model that cannot beat
    it has learned nothing, and on small imbalanced sets that happens.
    """
    if task_type == "regression":
        return np.full(n_test, float(np.mean(y_train)))
    return np.full(n_test, float(np.mean(y_train)))
