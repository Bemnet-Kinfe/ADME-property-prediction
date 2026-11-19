"""Molecular featurization: RDKit descriptors and Morgan fingerprints."""

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

MORGAN_RADIUS = 2
MORGAN_BITS = 2048

_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS, fpSize=MORGAN_BITS
)

# Fixed descriptor order so feature columns are stable across runs.
DESCRIPTOR_NAMES = [name for name, _ in Descriptors.descList]


def parse_smiles(smiles_list):
    """Return (mols, valid_mask). Invalid SMILES become None and are masked out."""
    mols, valid = [], []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        mols.append(mol)
        valid.append(mol is not None)
    return mols, np.asarray(valid, dtype=bool)


def descriptors(mols):
    """~210 interpretable physicochemical descriptors per molecule."""
    rows = np.zeros((len(mols), len(DESCRIPTOR_NAMES)), dtype=np.float64)
    for i, mol in enumerate(mols):
        if mol is None:
            rows[i, :] = np.nan
            continue
        for j, name in enumerate(DESCRIPTOR_NAMES):
            try:
                rows[i, j] = Descriptors.descList[j][1](mol)
            except Exception:
                rows[i, j] = np.nan
    # Descriptors can return inf (e.g. Ipc on large molecules); treat as missing.
    rows[~np.isfinite(rows)] = np.nan
    return rows


def morgan(mols):
    """Binary Morgan/ECFP4 fingerprints."""
    rows = np.zeros((len(mols), MORGAN_BITS), dtype=np.uint8)
    for i, mol in enumerate(mols):
        if mol is None:
            continue
        fp = _morgan_gen.GetFingerprintAsNumPy(mol)
        rows[i, :] = fp
    return rows


def featurize(smiles_list, use_descriptors=True, use_morgan=True):
    """Build the feature matrix for the baseline.

    Returns (X, feature_names, valid_mask).
    """
    mols, valid = parse_smiles(smiles_list)
    blocks, names = [], []
    if use_descriptors:
        blocks.append(descriptors(mols))
        names += [f"desc:{n}" for n in DESCRIPTOR_NAMES]
    if use_morgan:
        blocks.append(morgan(mols).astype(np.float64))
        names += [f"morgan:{i}" for i in range(MORGAN_BITS)]
    if not blocks:
        raise ValueError("at least one feature block must be enabled")
    X = np.hstack(blocks)
    return X, names, valid
