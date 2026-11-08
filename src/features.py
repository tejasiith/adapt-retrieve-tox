import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def morgan_np(smiles_list):
    arr = np.zeros((len(smiles_list), 2048), dtype=np.uint8)
    for i, s in enumerate(smiles_list):
        arr[i] = _gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s))
    return arr