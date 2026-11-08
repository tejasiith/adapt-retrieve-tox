import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import config
from src.utils import load_split, save_preds
from src.baselines import run_rf
from src.metrics import compute_metrics

print("Dataset:", config.DATASET)
for seed in config.SEEDS:
    tr, va, te = load_split(seed)
    prob = run_rf(tr, te, seed)
    save_preds("rf_morgan", seed, te, prob)
    m = compute_metrics(te.label.values, prob)
    print(seed, {k: round(v, 3) for k, v in m.items()})