import argparse, sys
from pathlib import Path
import numpy as np
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import config, finetune
from src.utils import load_split, save_preds
from src.metrics import compute_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["lora", "full"], default="lora")
ap.add_argument("--rank", type=int, default=8)
args = ap.parse_args()

method = "full" if args.mode == "full" else f"lora_r{args.rank}"
emb_dir = config.RESULTS_DIR / "emb"
emb_dir.mkdir(parents=True, exist_ok=True)

print("Dataset:", config.DATASET, "| method:", method)
for seed in config.SEEDS:
    tr, va, te = load_split(seed)
    out = finetune.run(args.mode, tr, va, te, seed, lora_r=args.rank,
                       with_embeddings=(method == "lora_r8"))
    save_preds(method, seed, te, out["test_prob"])
    if "emb_train" in out:
        np.save(emb_dir / f"{method}_seed{seed}_train.npy", out["emb_train"])
        np.save(emb_dir / f"{method}_seed{seed}_test.npy", out["emb_test"])
    m = compute_metrics(te.label.values, out["test_prob"])
    print(method, seed, {k: round(v, 3) for k, v in m.items()},
          "| valid AUROC:", round(out["best_valid_auroc"], 3))