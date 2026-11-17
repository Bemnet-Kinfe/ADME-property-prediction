# ADME property prediction: representation benchmark

When does a learned molecular representation beat engineered features, and when
does it not? ADME datasets are small, a few hundred to a few thousand measured
molecules, which is where that question is actually open.

Two tasks, two split types, five seeds, every number reported with error bars
and a trivial baseline alongside it.

| Task | Type | Endpoint | Why this one |
|---|---|---|---|
| `caco2_wang` | regression | Caco-2 permeability, log cm/s | Absorption. ~910 molecules, so it tests small-data behaviour. |
| `herg` | classification | hERG channel blocker | Cardiotoxicity, the canonical safety filter. Imbalanced. |

## Status

Phase 1 (baseline) code is complete and verified end to end on an offline
molecule set. It has not yet been run on the real endpoints, because the data
source was unreachable from the machine it was built on. See **Data** below.

- [x] Featurization: RDKit descriptors (~210) plus Morgan/ECFP4 fingerprints (2048 bits)
- [x] Random split and Bemis-Murcko scaffold split, seeded and reproducible
- [x] Gradient boosting baseline with early stopping, five seeds
- [x] Metrics: MAE/RMSE/Spearman for regression, AUPRC/ROC-AUC for classification
- [x] Trivial baseline (predict the training mean) reported next to every result
- [ ] Run on the real Caco-2 and hERG data
- [ ] Sanity check against the published TDC leaderboard
- [ ] Phase 2: graph neural network in PyTorch Geometric

## Data

Datasets come from Therapeutics Data Commons, which hosts on Harvard Dataverse.

```bash
pip install -r requirements.txt
python fetch_data.py                      # needs open network access
python src/run_baseline.py --prefer-cache
```

`fetch_data.py` writes `data/raw/caco2_wang.csv` and `data/raw/herg.csv`, each
with columns `Drug` and `Y`. Those two files are the only external input; once
they exist the pipeline is fully offline.

## Run

```bash
python src/run_baseline.py                       # both tasks, both splits, seeds 0-4
python src/run_baseline.py --tasks herg          # one task
python src/run_baseline.py --features morgan     # ablate the feature blocks
python src/run_baseline.py --dev                 # offline smoke test, no data needed
```

Results go to `results/phase1_baseline.csv`, with split diagnostics in
`results/phase1_baseline_diagnostics.json`.

## Why the scaffold split is the point

A scaffold is a molecule's core framework: its ring systems and the linkers
between them, with the decorations stripped off. Drug discovery programs pick a
promising scaffold and make dozens of close variants of it, so a dataset of
measured molecules is not a random sample of chemical space. It is clusters of
near-identical molecules around a handful of cores.

Under a random split, variants of the same scaffold land on both sides. The
model sees one molecule in training and is tested on the same molecule with an
extra methyl group. It scores well by doing lookup, not prediction. Deploy it on
a genuinely new scaffold, the only case anyone cares about, and it fails.

A scaffold split puts every molecule sharing a core on the same side. Scores
drop, often a lot, and that lower number is the honest one.

The pipeline reports both, and prints the diagnostic that explains the gap:
`frac_test_scaffold_seen_in_train`. On the offline molecule set that figure is
**0.80 for the random split and 0.00 for the scaffold split**, which is the
mechanism stated as a number rather than asserted.

### One choice worth knowing about

Acyclic molecules have no Murcko scaffold, so they all key to the empty string.
Whether to treat them as one group changes the result:

| | scaffold overlap, train vs test | fold sizes |
|---|---|---|
| pooled (default) | 0 shared scaffolds, 0.00 of test seen | can be unbalanced |
| not pooled | 1 shared group, 0.33–0.45 of test seen | balanced |

Pooling is the conservative choice and the default, because an empty scaffold
should not let a series of close acyclic analogs straddle the split. The cost is
that on a set with many acyclic molecules, one pooled group can be a large share
of the data and balanced folds become impossible; the splitter warns when a fold
comes out far below its target. This is a labeling choice that moves the
numbers, so it is reported rather than buried.

## Layout

```
fetch_data.py         one-time data download
requirements.txt
src/
  data.py             dataset loading, TDC with CSV fallback, offline dev set
  features.py         RDKit descriptors and Morgan fingerprints
  splits.py           random and scaffold splits, plus leakage diagnostics
  models.py           LightGBM baseline and the trivial baseline
  evaluate.py         metrics and aggregation across seeds
  run_baseline.py     phase 1 entry point
results/
```

## What this will not show

This is a benchmark of representations on two endpoints, not a general claim
about ADME prediction. It involves no molecular dynamics, no 3D conformers and
no generative chemistry. Measurements of these endpoints disagree between labs,
so there is a noise floor no model can beat; a model close to that floor is
doing as well as anything can, and the write-up should say where that floor is
rather than treating a lower error as strictly better.
