"""Train/validation/test splits: random, and Bemis-Murcko scaffold.

The scaffold split is the point of this project. A random split scatters close
variants of the same core across train and test, so the model can score well by
recognising a scaffold it has already seen. A scaffold split keeps every molecule
sharing a core on one side, which is the only situation anyone deploys into.
"""

from collections import defaultdict

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")


def scaffold_of(smiles, include_chirality=False):
    """Bemis-Murcko scaffold as SMILES. Returns '' when it cannot be computed."""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(
            smiles=smiles, includeChirality=include_chirality
        )
    except Exception:
        return ""


def scaffold_groups(smiles_list, pool_acyclic=True):
    """Map scaffold SMILES -> list of indices.

    Acyclic molecules have no Murcko scaffold, so they all key to ''. Whether to
    treat them as one group is a real choice, not a detail:

    pool_acyclic=True (default, and what DeepChem and TDC do) is conservative.
    It never lets two acyclic molecules straddle the split, so a series of close
    acyclic analogs cannot leak. The cost is that one pooled group can be a large
    share of the set, which makes balanced folds impossible.

    pool_acyclic=False treats each acyclic molecule as its own group, on the
    grounds that an empty scaffold means "no scaffold", not "the same scaffold".
    Folds come out balanced, but close acyclic analogs can land on both sides.

    Which one is used moves the reported numbers, so report it.
    """
    groups = defaultdict(list)
    singletons = []
    for i, smi in enumerate(smiles_list):
        scaf = scaffold_of(smi)
        if scaf == "" and not pool_acyclic:
            singletons.append([i])
        else:
            groups[scaf].append(i)
    return list(groups.values()) + singletons


def random_split(n, seed, frac=(0.8, 0.1, 0.1)):
    """Shuffle indices and cut at the fractions."""
    assert abs(sum(frac) - 1.0) < 1e-9
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(round(frac[0] * n))
    n_valid = int(round(frac[1] * n))
    return (
        idx[:n_train],
        idx[n_train : n_train + n_valid],
        idx[n_train + n_valid :],
    )


def scaffold_split(smiles_list, seed, frac=(0.8, 0.1, 0.1), pool_acyclic=True):
    """Random scaffold split: whole scaffold groups are assigned to one side.

    Groups are shuffled with the seed, so repeated seeds give genuinely
    different splits rather than the single deterministic largest-first split.
    Molecules whose scaffold could not be computed are pooled under '' and
    treated as one group, which keeps them out of the test set's way.
    """
    assert abs(sum(frac) - 1.0) < 1e-9
    n = len(smiles_list)
    groups = scaffold_groups(smiles_list, pool_acyclic=pool_acyclic)

    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    n_train_target = int(round(frac[0] * n))
    n_valid_target = int(round(frac[1] * n))

    train, valid, test = [], [], []
    for group in groups:
        if len(train) + len(group) <= n_train_target:
            train.extend(group)
        elif len(valid) + len(group) <= n_valid_target:
            valid.extend(group)
        else:
            test.extend(group)

    return np.asarray(train), np.asarray(valid), np.asarray(test)


def get_split(kind, smiles_list, seed, frac=(0.8, 0.1, 0.1)):
    if kind == "random":
        return random_split(len(smiles_list), seed, frac)
    if kind == "scaffold":
        return scaffold_split(smiles_list, seed, frac)
    raise ValueError(f"unknown split kind: {kind}")


def split_report(smiles_list, train, valid, test):
    """Diagnostics worth printing: sizes, and scaffold overlap between train and test.

    For a scaffold split the overlap must be zero. For a random split it will be
    large, and that number is the explanation for the performance gap.
    """
    scaf = [scaffold_of(s) for s in smiles_list]
    train_scaf = {scaf[i] for i in train}
    test_scaf = {scaf[i] for i in test}
    shared = train_scaf & test_scaf
    n_test_seen = sum(1 for i in test if scaf[i] in train_scaf)
    return {
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test": len(test),
        "n_scaffolds_total": len(set(scaf)),
        # Acyclic molecules have no Murcko scaffold. They are pooled as one
        # group so they never straddle the split; if this is a large share of
        # the set, the scaffold split is weaker than it looks.
        "n_no_scaffold": sum(1 for s in scaf if s == ""),
        "n_shared_scaffolds": len(shared),
        "frac_test_scaffold_seen_in_train": (
            n_test_seen / len(test) if len(test) else float("nan")
        ),
    }
