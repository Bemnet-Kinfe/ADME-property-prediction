# ADME property prediction: representation benchmark

**When does a learned molecular representation beat engineered features, and when does it not?**

That question turns almost entirely on data scale, and ADME is where it is genuinely open: the datasets are small because every label is a real experiment. Hand-designed fingerprints work on 500 molecules but saturate. A graph neural network has to learn its representation from labels, so it starts worse and in principle keeps climbing. Whether the two ever cross, and where, is not something you can reason your way to.

Two findings are targeted, in priority order:

1. **How much of a reported ADMET score is lookup rather than prediction.** Measured as the gap between a random and a scaffold split. This is the headline.
2. **Where, if anywhere, a learned representation overtakes engineered features.** Measured as a learning curve on a single endpoint, so data scale moves and nothing else does.

Every number is a mean ± standard deviation over five seeds, for every split type, with a trivial baseline printed beside it.

## Tasks

| Task | Molecules | Type | Role |
|---|---|---|---|
| `caco2_wang` | ~906 | regression, log cm/s | Caco-2 permeability. Absorption, and the small-data end of the scale. |
| `cyp3a4` | ~12,328 | classification | CYP3A4 inhibition. Metabolism, and the large-data end — 13× Caco-2. |
| `herg` | ~650 | classification | hERG blocking. Cardiotoxicity. **Off by default**, see below. |

The two defaults sit at opposite ends of the data-scale axis on purpose, because that axis is what the project measures. hERG is a fine task and the better story — it is the channel famous for killing drugs late in development — but at ~650 molecules it samples the same regime as Caco-2, so running both answers the question twice instead of twice over. Turn it on with `--tasks herg` once the main result exists, as a check that the finding is not peculiar to one endpoint.

## Status

Phase 1 is code-complete and verified end to end on an offline molecule set. **It has not been run on the real endpoints**, because the data source is unreachable from the machine it was built on. See [Data](#data).

- [x] Dataset loading, TDC with a cached-CSV fallback
- [x] Featurization: ~210 RDKit descriptors and 2048-bit Morgan fingerprints
- [x] Random and Bemis-Murcko scaffold splits, seeded, with leakage diagnostics
- [x] Gradient boosting baseline, trivial baseline, per-task metrics
- [x] CLI runner and an offline smoke test
- [x] TDC official benchmark splits *(written against the published protocol, not yet exercised)*
- [x] Learning curve over training-set size
- [ ] Run on the real Caco-2 and CYP3A4 data
- [ ] Sanity check against the published TDC leaderboard
- [ ] Phase 2: graph neural network in PyTorch Geometric

## Data

Datasets come from [Therapeutics Data Commons](https://tdcommons.ai/), which hosts on Harvard Dataverse. Not every environment can reach that host — some clusters and CI runners block it — so the download is a separate one-time script and the loader falls back to a cached CSV. After `fetch_data.py` has run once, the pipeline needs no network.

```bash
pip install -r requirements.txt
python fetch_data.py                      # one time, needs network
python src/run_baseline.py --prefer-cache
```

`fetch_data.py` writes `data/raw/<task>.csv`, each with columns `Drug` (SMILES) and `Y` (label). Those files are the only external input.

## Run

```bash
python src/run_baseline.py                          # defaults: both tasks, random + scaffold, seeds 0-4
python src/run_baseline.py --splits tdc             # TDC's official protocol, its own seeds 1-5
python src/run_baseline.py --features morgan        # ablate the feature blocks
python src/run_baseline.py --tasks cyp3a4 --train-sizes 900 2000 5000 12328   # learning curve
python src/run_baseline.py --dev                    # offline smoke test, no data, no network
```

Results go to `results/phase1_baseline.csv`; split diagnostics to `results/phase1_baseline_diagnostics.json`.

**Read the diagnostics before the metrics.** If a scaffold split reports anything other than zero shared scaffolds between train and test, the split is broken and the numbers below it do not mean what they say.

## Why the scaffold split is the point

A **scaffold** is a molecule's core framework: its ring systems and the linkers between them, with the decorations stripped off. Drug discovery programs pick a promising scaffold and make dozens of close variants of it, so a dataset of measured molecules is not a random sample of chemical space. It is clusters of near-identical molecules around a handful of cores.

Under a random split, variants of the same scaffold land on both sides. The model sees one molecule in training and is tested on the same molecule with an extra methyl group. It scores well by doing lookup, not prediction. Deploy it on a genuinely new scaffold — the only case anyone cares about — and it fails.

A scaffold split puts every molecule sharing a core on the same side. Scores drop, often a lot, and that lower number is the honest one.

The pipeline reports the diagnostic that quantifies it, `frac_test_scaffold_seen_in_train`. On the offline verification set:

| Split | Shared scaffolds, train vs test | Share of test scaffolds seen in train |
|---|---|---|
| Random | 24 | **0.80** |
| Scaffold | 0 | **0.00** |

Those are dev-harness figures on 1,200 molecules with synthetic labels, not a result. They are here because they show the mechanism works; the real figures replace them once the datasets are in hand.

### One choice that moves the numbers

Acyclic molecules have no Murcko scaffold, so they all key to the empty string. Whether to pool them is a real decision:

| Choice | Scaffold overlap, train vs test | Fold sizes |
|---|---|---|
| Pooled (default) | 0 shared, 0.00 of test seen | can be unbalanced |
| Not pooled | 1 shared group, 0.33–0.45 of test seen | balanced |

Pooling is the default because an empty scaffold means "no scaffold", not "the same scaffold", and pooling stops a series of close acyclic analogs from straddling the split. It also matches the conventional reference split, which retains the empty scaffold as one disclosed acyclic group ([Beyond Scaffold Splits, 2026](https://arxiv.org/pdf/2607.10729)).

The cost is that one pooled group can be a large share of a set, making balanced folds impossible; the splitter warns when a fold lands far below its target.

## Three split types, and why all three

| Split | What it is for |
|---|---|
| `random` | The inflated number. Only meaningful next to `scaffold`. |
| `scaffold` | The honest number. The gap against `random` is finding #1. |
| `tdc` | Calibration. The protocol every leaderboard number was produced under. |

The first two resample train/valid/test every seed, which is what makes the gap measurable. TDC's benchmark group cannot produce that gap, because it only ships scaffold splits — but it is the only way to check the baseline lands in a sane range against published work.

**One difference that is easy to get wrong:** TDC holds the test set fixed and varies only the train/valid split across its seeds. Ours resample the test set too. So TDC's error bars measure training variability and ours measure split variability. Those are different quantities and should not be quoted side by side as though they were the same.

## The learning curve

Comparing two different endpoints cannot cleanly answer the scale question, because swapping Caco-2 for CYP3A4 moves two variables at once: the dataset grows 13×, *and* the assay changes. A difference could be either.

So the curve holds the endpoint fixed and moves only the data:

```bash
python src/run_baseline.py --tasks cyp3a4 --train-sizes 900 2000 5000 12328
```

Only the training fold shrinks. Validation and test are held exactly as the split produced them, so every point is scored on the same molecules — subsampling the test set too would move the target while measuring it. Classification subsamples are stratified, because a small draw from an imbalanced set can land with almost no positives and measure the draw rather than the data.

The expected result is a negative one, and it is the useful one: in published ADMET work the GNN often never overtakes the baseline at all, even past ten thousand molecules. *"At 12,000 molecules, hand-engineered features still beat a learned representation"* tells a medicinal chemist not to reach for a GNN on their endpoint.

## Evaluation rules

- **Metrics per task, not by habit.** Regression leads with MAE, the TDC leaderboard metric for Caco-2, with Spearman alongside. Classification leads with AUPRC, because the positive class is the rare one and ROC-AUC can look respectable while the model is useless on the actives. The positive rate is printed next to AUPRC as the floor a random model reaches.
- **Five seeds, always.** On a few hundred molecules the difference between two models is often smaller than the variation between seeds. If the error bars overlap, the models are indistinguishable — say that, rather than declaring a winner.
- **The trivial baseline is reported beside every result.** A model that cannot beat predicting the training mean has learned nothing, and on small imbalanced sets that does happen.

## Layout

```
fetch_data.py         one-time data download
requirements.txt
src/
  data.py             dataset loading, TDC with CSV fallback, offline dev set
  features.py         RDKit descriptors and Morgan fingerprints
  splits.py           random and scaffold splits, plus leakage diagnostics
  tdc_splits.py       TDC's official benchmark splits, for calibration
  models.py           LightGBM baseline and the trivial baseline
  evaluate.py         metrics and aggregation across seeds
  run_baseline.py     entry point
results/
```

## What this does not show

A benchmark of molecular representations on two endpoints, not a general claim about ADME prediction. No molecular dynamics, no free energy calculations, no 3D conformers, no generative chemistry.

Measurements of these endpoints disagree between labs, so there is a noise floor no model can beat. A model close to that floor is doing as well as anything can; the write-up should say where that floor is rather than treating a lower error as strictly better.
