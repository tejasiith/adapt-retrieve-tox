import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src import config
from src.data import load_dataset, scaffold_split

out = config.DATA_DIR / "splits"
out.mkdir(parents=True, exist_ok=True)
config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

df = load_dataset()
df.to_csv(config.DATA_DIR / "clean.csv", index=False)
print(f"Dataset: {config.DATASET} | total molecules: {len(df)} | "
      f"positive rate: {df.label.mean():.3f}")

for seed in config.SEEDS:
    tr, va, te = scaffold_split(df, seed)
    for name, part in (("train", tr), ("valid", va), ("test", te)):
        part.to_csv(out / f"seed{seed}_{name}.csv", index=False)
    print(f"seed {seed}: train {len(tr)}, valid {len(va)}, test {len(te)}, "
          f"test positive rate {te.label.mean():.2f}")