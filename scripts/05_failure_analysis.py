import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import config

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
path = config.RESULTS_DIR / "retrieval" / f"lora_r8_fp_seed{seed}.jsonl"
r = pd.read_json(path, lines=True)

r["top1_sim"] = r.neighbours.apply(lambda n: n[0]["sim"])
r["vote"] = r.neighbours.apply(lambda n: np.mean([x["label"] for x in n]))
r["correct"] = r.pred_label == r.true_label
r["confidence"] = (r.pred_prob - 0.5).abs() * 2

r["sim_bin"] = pd.cut(r.top1_sim, [-0.01, 0.3, 0.5, 0.7, 1.0])
print("Dataset:", config.DATASET)
print("Accuracy by similarity to the closest training molecule:")
print(r.groupby("sim_bin", observed=True).correct.agg(["mean", "count"]).round(3))

errs = r[~r.correct].sort_values("confidence", ascending=False)
cols = ["mol_id", "smiles", "true_label", "pred_prob", "top1_sim", "vote"]
errs[cols].head(20).to_csv(config.RESULTS_DIR / f"top_errors_seed{seed}.csv", index=False)
print(f"\n{len(errs)} errors out of {len(r)}. Top 20 confident errors saved.")