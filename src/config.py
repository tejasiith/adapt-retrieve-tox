import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATASET = os.environ.get("DATASET", "hERG")
DATA_DIR = ROOT / "data" / DATASET.lower()
RESULTS_DIR = ROOT / "results" / DATASET.lower()

SEEDS = [0, 1, 2, 3, 4]
SPLIT_FRAC = (0.7, 0.1, 0.2)

MODEL_NAME = "DeepChem/ChemBERTa-77M-MLM"
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 20                 
LR_LORA = 5e-4            
LR_FULL = 3e-5
K_NEIGHBOURS = 5  