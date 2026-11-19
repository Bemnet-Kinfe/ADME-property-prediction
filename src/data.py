"""Dataset loading.

Primary source is Therapeutics Data Commons (PyTDC), which downloads from
Harvard Dataverse. When that host is unreachable, a cached CSV in data/raw/ is
used instead, so the pipeline runs identically once the file is present.

Expected CSV columns: Drug (SMILES), Y (label).
"""

import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")

# The two default tasks are chosen to sit at opposite ends of the data-scale
# axis, because that axis is what the project is actually measuring. Caco-2 is
# ~900 molecules and CYP3A4 is ~12,300, a 13x span.
#
# hERG is kept but off by default. It is a fine task and a better interview
# story, but at ~650 molecules it samples the same small-data regime as Caco-2,
# so running both answers the research question twice instead of twice over.
# Turn it on with --tasks herg once the main result exists, as a check that the
# finding is not peculiar to one endpoint.
TASKS = {
    "caco2_wang": {
        "tdc_name": "Caco2_Wang",
        "tdc_group": "ADME",
        "task_type": "regression",
        "label": "Caco-2 permeability (log cm/s)",
        "note": "Absorption. ~906 molecules. TDC leaderboard metric: MAE.",
        "default": True,
    },
    "cyp3a4": {
        "tdc_name": "CYP3A4_Veith",
        "tdc_group": "ADME",
        "task_type": "classification",
        "label": "CYP3A4 inhibitor (1 = inhibits)",
        "note": "Metabolism. ~12,328 molecules, imbalanced. Large enough to carry the learning curve.",
        "default": True,
    },
    "herg": {
        "tdc_name": "hERG",
        "tdc_group": "Tox",
        "task_type": "classification",
        "label": "hERG blocker (1 = blocks)",
        "note": "Cardiotoxicity. ~650 molecules, imbalanced. Off by default: same data regime as Caco-2.",
        "default": False,
    },
}

DEFAULT_TASKS = [k for k, v in TASKS.items() if v.get("default")]


def _from_tdc(task_key):
    cfg = TASKS[task_key]
    if cfg["tdc_group"] == "ADME":
        from tdc.single_pred import ADME as Loader
    else:
        from tdc.single_pred import Tox as Loader
    data = Loader(name=cfg["tdc_name"], path=RAW_DIR)
    return data.get_data()


def _from_cache(task_key):
    path = os.path.join(RAW_DIR, f"{task_key}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    smiles_col = cols.get("drug") or cols.get("smiles")
    label_col = cols.get("y") or cols.get("label")
    if smiles_col is None or label_col is None:
        raise ValueError(
            f"{path} must have a SMILES column (Drug/smiles) and a label column (Y/label); "
            f"found {list(df.columns)}"
        )
    return df.rename(columns={smiles_col: "Drug", label_col: "Y"})


def load_task(task_key, prefer_cache=False):
    """Return (df, source) where df has columns Drug and Y."""
    if task_key not in TASKS:
        raise ValueError(f"unknown task: {task_key}. Known: {sorted(TASKS)}")
    os.makedirs(RAW_DIR, exist_ok=True)

    order = ["cache", "tdc"] if prefer_cache else ["tdc", "cache"]
    errors = []
    for source in order:
        try:
            df = _from_tdc(task_key) if source == "tdc" else _from_cache(task_key)
            df = df[["Drug", "Y"]].dropna().reset_index(drop=True)
            return df, source
        except Exception as exc:  # noqa: BLE001 - we want to try the next source
            errors.append(f"{source}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        f"could not load {task_key}.\n  " + "\n  ".join(errors) +
        f"\nPlace a CSV with Drug and Y columns at {os.path.join(RAW_DIR, task_key + '.csv')}"
    )


def dev_dataset(n=1200, seed=0):
    """Offline smoke-test set: real NCI molecules shipped with RDKit.

    Labels here are physicochemical properties computed from structure, not
    measured ADME endpoints. This exists only to prove the pipeline runs and the
    splits behave; it says nothing about model quality on real endpoints.
    """
    import numpy as np
    from rdkit import Chem, RDConfig, RDLogger
    from rdkit.Chem import Crippen, Descriptors

    RDLogger.DisableLog("rdApp.*")
    path = os.path.join(RDConfig.RDDataDir, "NCI", "first_5K.smi")

    smiles = []
    with open(path) as fh:
        for line in fh:
            field = line.split()[0] if line.split() else ""
            if not field:
                continue
            mol = Chem.MolFromSmiles(field)
            if mol is None or mol.GetNumHeavyAtoms() < 5:
                continue
            smiles.append(Chem.MolToSmiles(mol))
            if len(smiles) >= n:
                break

    rng = np.random.default_rng(seed)
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    logp = np.array([Crippen.MolLogP(m) for m in mols])
    tpsa = np.array([Descriptors.TPSA(m) for m in mols])
    mw = np.array([Descriptors.MolWt(m) for m in mols])

    # A nonlinear mix of three properties plus noise, so no single descriptor
    # column equals the label. A label that is itself a descriptor scores near
    # perfectly and would hide bugs rather than expose them.
    signal = 0.6 * logp - 0.02 * tpsa + 1.5 * np.sin(mw / 60.0)
    reg = pd.DataFrame({"Drug": smiles, "Y": signal + rng.normal(0, 0.5, len(smiles))})
    noisy = signal + rng.normal(0, 1.0, len(smiles))
    clf = pd.DataFrame(
        {"Drug": smiles, "Y": (noisy > np.quantile(noisy, 0.8)).astype(int)}
    )
    return reg, clf
