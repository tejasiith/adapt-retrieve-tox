import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import config
from src.utils import load_split
from src.features import morgan_np
from src.retrieval import tanimoto, cosine, get_neighbours, neighbour_stats

ap = argparse.ArgumentParser()
ap.add_argument("--space", choices=["fp", "emb"], default="fp")
ap.add_argument("--method", default="lora_r8")
args = ap.parse_args()

out_dir = config.RESULTS_DIR / "retrieval"
out_dir.mkdir(parents=True, exist_ok=True)
rows = []

print("Dataset:", config.DATASET, "| method:", args.method, "| space:", args.space)
for seed in config.SEEDS:
    tr, va, te = load_split(seed)
    preds = pd.read_csv(config.RESULTS_DIR / "preds" / f"{args.method}_seed{seed}.csv")
    assert (preds.mol_id.values == te.mol_id.values).all(), "ID mismatch"

    if args.space == "fp":
        sim = tanimoto(morgan_np(te.smiles), morgan_np(tr.smiles))
    else:
        e = config.RESULTS_DIR / "emb"
        sim = cosine(np.load(e / f"{args.method}_seed{seed}_test.npy"),
                     np.load(e / f"{args.method}_seed{seed}_train.npy"))

    neigh = get_neighbours(sim, tr, k=config.K_NEIGHBOURS)   # neighbours come from TRAIN only
    records = []
    for i, nb in enumerate(neigh):
        p = float(preds.prob.iloc[i])
        records.append({"mol_id": te.mol_id.iloc[i], "smiles": te.smiles.iloc[i],
                        "true_label": int(te.label.iloc[i]),
                        "pred_prob": p, "pred_label": int(p >= 0.5),
                        "neighbours": nb})

    with open(out_dir / f"{args.method}_{args.space}_seed{seed}.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    stats = neighbour_stats(records)
    rows.append({"seed": seed, **stats})
    print(seed, {k: round(float(v), 3) for k, v in stats.items()})

df = pd.DataFrame(rows)
df.to_csv(config.RESULTS_DIR / f"retrieval_summary_{args.method}_{args.space}.csv", index=False)
print(df.drop(columns="seed").mean().round(3))