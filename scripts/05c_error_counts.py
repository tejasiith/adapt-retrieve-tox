import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import config

r = pd.read_json(config.RESULTS_DIR / "retrieval" / "lora_r8_fp_seed0.jsonl", lines=True)
r["top1_sim"] = r.neighbours.apply(lambda n: n[0]["sim"])
r["vote"] = r.neighbours.apply(lambda n: np.mean([x["label"] for x in n]))
r["conf"] = (r.pred_prob - 0.5).abs() * 2
err = r[r.pred_label != r.true_label].sort_values("conf", ascending=False).head(20)

print("false positives (true 0, pred 1):", ((err.true_label == 0) & (err.pred_label == 1)).sum())
print("false negatives (true 1, pred 0):", ((err.true_label == 1) & (err.pred_label == 0)).sum())
print("median top1_sim, these 20 errors:", round(err.top1_sim.median(), 2))
print("median top1_sim, all test molecules:", round(r.top1_sim.median(), 2))
print("errors where neighbours disagreed with the model:",
      ((err.vote >= 0.5).astype(int) != err.pred_label).sum(), "of 20")