import sys
from pathlib import Path
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import config
from src.metrics import compute_metrics

rows = []
for f in sorted((config.RESULTS_DIR / "preds").glob("*_seed*.csv")):
    method, seed = f.stem.rsplit("_seed", 1)
    d = pd.read_csv(f)
    rows.append({"method": method, "seed": int(seed), **compute_metrics(d.y_true, d.prob)})

df = pd.DataFrame(rows)
summary = df.groupby("method")[["auroc", "auprc", "acc"]].agg(["mean", "std"]).round(3)
print("Dataset:", config.DATASET)
print(summary)
summary.to_csv(config.RESULTS_DIR / "summary_table.csv")