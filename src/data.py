import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from . import config

def _canon(smi):
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol is not None else None

def load_dataset():
    """Loads a binary toxicity dataset from TDC (hERG or AMES)."""
    from tdc.single_pred import Tox
    df = Tox(name=config.DATASET).get_data()
    df = df.rename(columns={"Drug": "smiles", "Y": "label"})[["smiles", "label"]]
    df["smiles"] = [_canon(s) for s in df.smiles]
    df = df.dropna().drop_duplicates("smiles").reset_index(drop=True)
    df["label"] = df.label.astype(int)
    df.insert(0, "mol_id", [f"m{i:05d}" for i in range(len(df))])
    return df

def scaffold_split(df, seed, frac=config.SPLIT_FRAC):
    groups = {}
    for i, smi in zip(df.index, df.smiles):
        scaf = MurckoScaffold.MurckoScaffoldSmiles(smiles=smi, includeChirality=False)
        groups.setdefault(scaf, []).append(i)
    groups = list(groups.values())

    n = len(df)
    n_train, n_valid = int(frac[0] * n), int(frac[1] * n)
    rng = np.random.RandomState(seed)
    big = [g for g in groups if len(g) > n_valid / 2] 
    small = [g for g in groups if len(g) <= n_valid / 2]
    rng.shuffle(big)
    rng.shuffle(small)

    train, valid, test = [], [], []
    for g in big + small:
        if len(train) + len(g) <= n_train:
            train += g
        elif len(valid) + len(g) <= n_valid:
            valid += g
        else:
            test += g
    return df.loc[train], df.loc[valid], df.loc[test]