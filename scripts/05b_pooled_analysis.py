import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import config

METHOD = "lora_r8"
N_BOOT = 2000
rng = np.random.RandomState(0)

def load(space):
    frames = []
    for s in config.SEEDS:
        p = config.RESULTS_DIR / "retrieval" / f"{METHOD}_{space}_seed{s}.jsonl"
        d = pd.read_json(p, lines=True)
        d["seed"] = s
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["vote"] = d.neighbours.apply(lambda n: np.mean([x["label"] for x in n]))
    d["agree"] = (d.vote >= 0.5).astype(int) == d.pred_label
    d["correct"] = d.pred_label == d.true_label
    d["top1_sim"] = d.neighbours.apply(lambda n: n[0]["sim"])
    return d

def boot_ci(d):
    a, c = d.agree.values, d.correct.values
    diffs = []
    for _ in range(N_BOOT):
        i = rng.randint(0, len(d), len(d))
        aa, cc = a[i], c[i]
        if aa.any() and (~aa).any():
            diffs.append(cc[aa].mean() - cc[~aa].mean())
    return np.percentile(diffs, [2.5, 97.5])

print("Dataset:", config.DATASET, "| method:", METHOD)
for space in ("fp", "emb"):
    d = load(space)
    lo, hi = boot_ci(d)
    print(f"\n[{space}] pooled over {len(d)} predictions ({len(config.SEEDS)} seeds)")
    print(f"  agree:    n={d.agree.sum():4d}  accuracy={d.correct[d.agree].mean():.3f}")
    print(f"  disagree: n={(~d.agree).sum():4d}  accuracy={d.correct[~d.agree].mean():.3f}")
    print(f"  difference (agree - disagree) 95% bootstrap CI: [{lo:.3f}, {hi:.3f}]")

d = load("fp")
d["sim_bin"] = pd.cut(d.top1_sim, [-0.01, 0.3, 0.5, 0.7, 1.0])
print("\n[fp] pooled accuracy by top-1 similarity:")
print(d.groupby("sim_bin", observed=True).correct.agg(["mean", "count"]).round(3))