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

Data comes from [Therapeutics Data Commons](https://tdcommons.ai/).

## Planned work

- [ ] Dataset loading, with a cached-CSV fallback
- [ ] Featurization: RDKit descriptors and Morgan fingerprints
- [ ] Random and Bemis-Murcko scaffold splits, with leakage diagnostics
- [ ] Gradient boosting baseline and metrics
- [ ] CLI runner and offline smoke test
- [ ] Phase 2: graph neural network in PyTorch Geometric

## Install

```bash
pip install -r requirements.txt
```
