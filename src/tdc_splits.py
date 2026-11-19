"""TDC's official benchmark splits.

These answer a different question from the splits in splits.py, and the project
needs both.

splits.py is the experiment. It resamples train/valid/test every seed, under
both a random and a scaffold rule, so the gap between the two measures how much
of a reported score is lookup rather than prediction. TDC's benchmark group
cannot produce that number, because it only ships scaffold splits.

This module is the calibration. Every number on the TDC leaderboard was produced
under this exact protocol, so it is the only way to check that our baseline is
in a sane range. A score computed on our own split is not comparable to anything
published.

One difference matters when reporting both, and it is easy to get wrong:

    TDC holds the test set FIXED and varies only the train/valid split across
    seeds 1-5. Our splits resample the test set too.

So TDC's error bars measure training variability and ours measure split
variability. They are different quantities. Quoting one beside the other as
though they were the same would overstate how comparable they are.

Needs network access on first use, since it downloads from Harvard Dataverse.
"""

import numpy as np

TDC_SEEDS = [1, 2, 3, 4, 5]


def _index_by_smiles(df):
    """Map SMILES -> row index in our dataframe.

    Keeps the first occurrence. TDC datasets can carry duplicate structures,
    and the duplicates are the same molecule, so collapsing them is correct.
    """
    mapping = {}
    for i, smi in enumerate(df["Drug"]):
        mapping.setdefault(smi, i)
    return mapping


def _to_indices(sub_df, mapping):
    idx, missing = [], 0
    for smi in sub_df["Drug"]:
        hit = mapping.get(smi)
        if hit is None:
            missing += 1
        else:
            idx.append(hit)
    return np.asarray(sorted(set(idx)), dtype=int), missing


def tdc_official_split(tdc_benchmark_name, df, seed, path="data/raw"):
    """Return (train, valid, test) index arrays into df, using TDC's own split.

    Follows the protocol in the TDC leaderboard guidelines: load the benchmark
    group, take the fixed train_val/test for the benchmark, then split
    train_val into train and valid for this seed.
    """
    from tdc.benchmark_group import admet_group

    group = admet_group(path=path)
    benchmark = group.get(tdc_benchmark_name)
    name = benchmark["name"]
    train_val, test = benchmark["train_val"], benchmark["test"]
    train, valid = group.get_train_valid_split(
        benchmark=name, split_type="default", seed=seed
    )

    mapping = _index_by_smiles(df)
    tr, miss_tr = _to_indices(train, mapping)
    va, miss_va = _to_indices(valid, mapping)
    te, miss_te = _to_indices(test, mapping)

    missing = miss_tr + miss_va + miss_te
    if missing:
        # A few unmatched SMILES mean the loaded CSV and the benchmark copy were
        # canonicalised differently. A lot of them mean they are different
        # datasets, and the resulting split would be silently wrong.
        total = len(train) + len(valid) + len(test)
        raise RuntimeError(
            f"{missing} of {total} benchmark molecules were not found in the "
            f"loaded dataset for {tdc_benchmark_name}. The CSV and the TDC "
            f"benchmark are probably not the same release; re-run fetch_data.py."
        )

    overlap = (set(tr) | set(va)) & set(te)
    if overlap:
        raise RuntimeError(
            f"{len(overlap)} molecules appear in both train/valid and test after "
            f"mapping by SMILES. Duplicate structures across TDC's own folds; "
            f"report this rather than scoring through it."
        )

    return tr, va, te
