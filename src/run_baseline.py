"""Phase 1 entry point. One command, fixed seeds, reproducible.

    python src/run_baseline.py --tasks caco2_wang herg
    python src/run_baseline.py --dev          # offline smoke test, no network

Every number printed is a mean +/- SD over the seeds, for both split types, with
the trivial baseline alongside it.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as data_mod
import evaluate as ev
import features as feat
import models
import splits as sp
import tdc_splits

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)

METRIC_KEYS = {
    "regression": ["mae", "rmse", "spearman"],
    "classification": ["auprc", "roc_auc", "positive_rate"],
}


def subsample_train(tr, n_train, y_all, task_type, seed):
    """Take a random subset of the training indices, for the learning curve.

    n_train=None means use everything. A request larger than the fold returns
    the whole fold.

    Classification subsamples are stratified, so the positive rate stays what
    the task's prevalence actually is. Without that, a small draw from an
    imbalanced set can land with almost no positives, and the resulting point
    on the curve measures the draw rather than the data scale.
    """
    if n_train is None or n_train >= len(tr):
        return tr

    rng = np.random.default_rng(seed)
    if task_type != "classification":
        return np.sort(rng.choice(tr, size=n_train, replace=False))

    y = y_all[tr]
    picked = []
    for cls in np.unique(y):
        pool = tr[y == cls]
        k = max(1, int(round(n_train * len(pool) / len(tr))))
        k = min(k, len(pool))
        picked.append(rng.choice(pool, size=k, replace=False))
    return np.sort(np.concatenate(picked))


def run_task(task_key, df, task_type, seeds, split_kinds, feature_mode,
             n_train=None):
    smiles = df["Drug"].tolist()
    y_all = df["Y"].to_numpy(dtype=float)

    t0 = time.time()
    X_all, names, valid = feat.featurize(
        smiles,
        use_descriptors=feature_mode in ("both", "descriptors"),
        use_morgan=feature_mode in ("both", "morgan"),
    )
    n_dropped = int((~valid).sum())
    if n_dropped:
        X_all, y_all = X_all[valid], y_all[valid]
        smiles = [s for s, keep in zip(smiles, valid) if keep]
    print(
        f"  featurized {X_all.shape[0]} molecules x {X_all.shape[1]} features "
        f"in {time.time() - t0:.1f}s"
        + (f" ({n_dropped} unparseable SMILES dropped)" if n_dropped else "")
    )

    rows, diagnostics = [], []
    for split_kind in split_kinds:
        # TDC's official protocol uses its own seeds, not ours, because the
        # leaderboard numbers were produced with those exact five.
        kind_seeds = tdc_splits.TDC_SEEDS if split_kind == "tdc" else seeds

        per_seed, per_seed_trivial = [], []
        for seed in kind_seeds:
            if split_kind == "tdc":
                tdc_name = data_mod.TASKS[task_key]["tdc_name"]
                tr, va, te = tdc_splits.tdc_official_split(tdc_name, df, seed)
            else:
                tr, va, te = sp.get_split(split_kind, smiles, seed)
            if len(te) == 0 or len(va) == 0:
                print(f"  !! {split_kind} seed {seed}: empty valid/test fold, skipped")
                continue

            # Learning curve: shrink the TRAINING set only. Validation and test
            # stay exactly as the split produced them, so every point on the
            # curve is scored on the same molecules and the points are
            # comparable. Subsampling the test set too would move the target
            # while measuring it.
            tr = subsample_train(tr, n_train, y_all, task_type, seed)

            pred, best_iter = models.fit_predict(
                task_type, X_all[tr], y_all[tr], X_all[va], y_all[va], X_all[te], seed
            )
            m = ev.metrics_for(task_type, y_all[te], pred)
            m["best_iteration"] = best_iter
            m["n_train_used"] = len(tr)
            per_seed.append(m)

            trivial = models.predict_mean_baseline(task_type, y_all[tr], len(te))
            per_seed_trivial.append(ev.metrics_for(task_type, y_all[te], trivial))

            if seed == kind_seeds[0]:
                diagnostics.append(
                    {"split": split_kind, "seed": int(seed),
                     **sp.split_report(smiles, tr, va, te)}
                )

        keys = METRIC_KEYS[task_type]
        agg = ev.aggregate(per_seed, keys)
        agg_trivial = ev.aggregate(per_seed_trivial, keys)
        for model_name, a, raw in (
            ("gradient boosting", agg, per_seed),
            ("trivial (train mean)", agg_trivial, per_seed_trivial),
        ):
            row = {"task": task_key, "split": split_kind, "model": model_name,
                   "n_train_requested": n_train if n_train is not None else -1,
                   "n_train_used": int(np.mean([r.get("n_train_used", 0) for r in raw]))
                   if raw and "n_train_used" in raw[0] else -1,
                   "n_seeds": len(raw)}
            for k in keys:
                mean, sd, _ = a[k]
                row[k] = mean
                row[f"{k}_sd"] = sd
            rows.append(row)

    return rows, diagnostics


def print_table(rows, task_type):
    keys = METRIC_KEYS[task_type]
    header = f"{'split':<10} {'model':<22} " + " ".join(f"{k:>18}" for k in keys)
    print("  " + header)
    print("  " + "-" * len(header))
    for r in rows:
        cells = " ".join(
            f"{ev.format_mean_sd(r[k], r[k + '_sd']):>18}" for k in keys
        )
        print(f"  {r['split']:<10} {r['model']:<22} {cells}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=data_mod.DEFAULT_TASKS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--splits", nargs="+", default=["random", "scaffold"],
                    choices=["random", "scaffold", "tdc"],
                    help="tdc uses TDC's official benchmark split and its own seeds 1-5")
    ap.add_argument("--features", choices=["both", "descriptors", "morgan"], default="both")
    ap.add_argument("--train-sizes", nargs="+", type=int, default=None,
                    metavar="N",
                    help="learning curve: run once per training-set size. "
                         "Only the training fold shrinks; valid and test are "
                         "held fixed so the points are comparable.")
    ap.add_argument("--prefer-cache", action="store_true",
                    help="read data/raw/<task>.csv before trying to reach TDC")
    ap.add_argument("--dev", action="store_true",
                    help="offline smoke test on RDKit's bundled NCI molecules")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_rows, all_diag = [], []

    if args.dev:
        reg, clf = data_mod.dev_dataset()
        jobs = [("dev_logp", reg, "regression"), ("dev_polar", clf, "classification")]
    else:
        jobs = []
        for task_key in args.tasks:
            df, source = data_mod.load_task(task_key, prefer_cache=args.prefer_cache)
            print(f"{task_key}: {len(df)} molecules (source: {source})")
            jobs.append((task_key, df, data_mod.TASKS[task_key]["task_type"]))

    sizes = args.train_sizes or [None]
    for task_key, df, task_type in jobs:
        for n_train in sizes:
            label = f"{task_key} ({task_type}, n={len(df)}"
            label += f", train<={n_train})" if n_train else ")"
            print(f"\n=== {label} ===")
            rows, diag = run_task(
                task_key, df, task_type, args.seeds, args.splits, args.features,
                n_train=n_train,
            )
            print_table(rows, task_type)
            all_rows += rows
            all_diag += diag

    print("\n=== split diagnostics (first seed) ===")
    print(pd.DataFrame(all_diag).to_string(index=False))

    out = args.out or os.path.join(
        RESULTS_DIR, "dev_baseline.csv" if args.dev else "phase1_baseline.csv"
    )
    pd.DataFrame(all_rows).to_csv(out, index=False)
    with open(out.replace(".csv", "_diagnostics.json"), "w") as fh:
        json.dump(all_diag, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
