import random
import numpy as np
import pandas as pd
import torch
from . import config

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_split(seed):
    d = config.DATA_DIR / "splits"
    return [pd.read_csv(d / f"seed{seed}_{n}.csv") for n in ("train", "valid", "test")]

def save_preds(method, seed, test_df, prob):
    p = config.RESULTS_DIR / "preds"
    p.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"mol_id": test_df.mol_id.values,
                  "y_true": test_df.label.values,
                  "prob": prob}).to_csv(p / f"{method}_seed{seed}.csv", index=False)