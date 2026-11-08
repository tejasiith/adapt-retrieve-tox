# Adapting a pretrained molecular model with LoRA, and checking predictions against retrieved neighbours (hERG)

A small study on the hERG cardiotoxicity task from Therapeutics Data Commons (TDC). I asked three questions:

1. How well does a pretrained molecular transformer (ChemBERTa) adapt to hERG with LoRA, compared with full fine-tuning and a random-forest baseline?
2. Can the nearest training molecules tell us when a prediction is unreliable?
3. Where does the model fail?

This is a study of adaptation and evidence checking, with hERG as the test case. It is not an attempt to build a better hERG predictor. It is separate from my earlier multimodal hERG model [(https://github.com/tejasiith/herg-multimodal-property-prediction)].

## Summary of findings

- LoRA with rank 16 reached the same mean test AUROC as full fine-tuning (0.822 vs 0.824), training about 6% of the parameters (this count includes the classification head).
- Both were about 0.02 AUROC above the random forest on average and beat it on 4 of 5 seeds. With about 131 test molecules per seed and a seed-to-seed spread of about 0.1, I treat this as suggestive, not proven.
- When the nearest training molecules disagreed with the model, accuracy dropped. This held for fingerprint retrieval (0.81 vs 0.53) and for retrieval in the model's own embedding space (0.84 vs 0.30).
- Neighbour agreement does not catch most confident errors: for 18 of the 20 most confident errors in seed 0, the neighbours agreed with the wrong prediction.

## Data and splits

- Dataset: hERG from TDC (`tdc.single_pred.Tox`), 648 molecules after SMILES canonicalisation and de-duplication. 69% are labelled as hERG blockers.
- Splits: scaffold splits (Murcko scaffolds), 70 / 10 / 20 → 453 train, 64 validation, 131 test molecules. Each seed (0 to 4) gives a different scaffold split and a different training run.
- Test positive rate varies from 0.63 to 0.78 across seeds.

The std values reported below therefore mix split variation and training variation.

## Methods

| Item | Setting |
|---|---|
| Pretrained model | `DeepChem/ChemBERTa-77M-MLM` (SMILES input, max length 128) |
| LoRA | targets `query` and `value`, alpha = 2 × rank, dropout 0.1, ranks 4 / 8 / 16 |
| Full fine-tuning | all weights trainable, learning rate 3e-5 |
| Training | AdamW, weight decay 0.01, batch size 32, 20 epochs, LoRA learning rate 5e-4 |
| Model selection | best epoch by validation AUROC; test set used once |
| Baseline | random forest (500 trees, balanced class weights) on Morgan fingerprints (radius 2, 2048 bits) |
| Retrieval | k = 5 nearest training molecules (never test molecules). Two spaces: Tanimoto on Morgan fingerprints, and cosine similarity on mean-pooled embeddings of the fine-tuned `lora_r8` model |

Hyperparameters were not tuned. The values above are starting points I chose before running.

## Results

### Test performance (mean ± std over 5 seeds)

| Method | AUROC | AUPRC | Accuracy | Trainable params* |
|---|---|---|---|---|
| Full fine-tuning | 0.824 ± 0.037 | 0.898 ± 0.062 | 0.777 ± 0.056 | 3,428,210 |
| LoRA r16 | 0.822 ± 0.018 | 0.896 ± 0.055 | 0.785 ± 0.043 | 222,338 |
| LoRA r8 | 0.808 ± 0.026 | 0.887 ± 0.053 | 0.774 ± 0.045 | 185,474 |
| Random forest (Morgan) | 0.802 ± 0.050 | 0.898 ± 0.051 | 0.794 ± 0.058 | n/a |
| LoRA r4 | 0.782 ± 0.029 | 0.879 ± 0.053 | 0.783 ± 0.050 | 167,042 |

\*Includes the classification head (about 150k parameters). The LoRA adapters alone are about 17k, 35k and 72k for r4, r8 and r16.

How I read this:

- AUROC is the main metric. About 69% of molecules are positive, so AUPRC and accuracy are inflated by the base rate. The random forest has the best accuracy, even though it has a lower AUROC than the full and r16 models.
- Full fine-tuning and LoRA r16 each beat the random forest on 4 of 5 seeds (both lose on seed 4). The mean gap is about 0.02, which is small relative to the spread.
- r16 had higher AUROC than r4 on all 5 seeds, so higher rank looks slightly better here. The differences between ranks are small relative to the noise.
- Validation AUROC is higher than test AUROC on most seeds because the best epoch is picked on 64 validation molecules. Only test numbers are reported above.

### Retrieval as a check on predictions

Pooled over 5 seeds (655 test predictions), using the `lora_r8` predictions:

| Retrieval space | Accuracy if neighbours agree | Accuracy if neighbours disagree | Difference, approx. 95% bootstrap CI |
|---|---|---|---|
| Morgan fingerprint (Tanimoto) | 0.813 (n = 563) | 0.533 (n = 92) | 0.28 [0.175, 0.390] |
| Fine-tuned embedding (cosine) | 0.841 (n = 574) | 0.296 (n = 81) | 0.55 [0.443, 0.644] |

"Agree" means the majority label of the 5 neighbours matches the model's predicted label.

- Fingerprint retrieval is independent of the model, so it is the cleaner evidence. The effect is weaker.
- The embedding result is partly circular, because the embeddings come from the same fine-tuned model that was trained on the training labels.
- The 655 rows come from 5 overlapping test sets, so the same molecule can appear more than once. The intervals are likely somewhat too narrow.
- I did not test whether using the neighbours improves predictions. I only tested whether disagreement flags errors.

Accuracy by similarity to the closest training molecule (pooled, fingerprint retrieval):

| Top-1 Tanimoto similarity | Accuracy | n |
|---|---|---|
| ≤ 0.3 | 0.750 | 236 |
| 0.3 to 0.5 | 0.739 | 211 |
| 0.5 to 0.7 | 0.817 | 126 |
| > 0.7 | 0.866 | 82 |

### Failure analysis

I examined the 20 most confident errors in seed 0 (`lora_r8`). 18 were false positives and 2 were false negatives. Their median top-1 similarity was 0.34, compared with 0.43 for all test molecules. For 18 of the 20, the neighbours agreed with the wrong prediction, so the neighbour-agreement flag would not have caught them. One pair of stereoisomers (m00449 and m00450) received identical scores. My Morgan fingerprints do not encode chirality, so retrieval cannot separate them.

This is one seed and 20 molecules, so it is descriptive only. Details are in [docs/failure_analysis.md](docs/failure_analysis.md).

## Limitations

- One small dataset (648 molecules) and one binary toxicity task. Differences of 0.01 to 0.02 AUROC between methods should not be trusted.
- One pretrained model (ChemBERTa-77M). I did not compare other molecular foundation models.
- Hyperparameters were not tuned, and the number of seeds is 5.
- Seed std mixes split variation and training variation.
- Fingerprints ignore chirality, so retrieval cannot distinguish stereoisomers.
- The retrieval analysis uses `lora_r8` only, and the embedding-space result is partly circular.
- The failure analysis covers one seed and 20 molecules.
- No satellite imagery, time series, or environmental data, and no comparison with my multimodal hERG model on these splits.
- A second dataset (AMES) is not yet run. The code supports it through the `DATASET` environment variable, but I have not tested it.

## Reproducing

Environment used: Python 3.12, PyTorch 2.6.0+cu124, transformers 4.57.6, peft 0.14.0, RDKit 2026.9.1, PyTDC 0.3.6, on an NVIDIA RTX 4050 (6 GB). Runtime: about 1 minute for 5 seeds of LoRA r8 on an NVIDIA RTX 4050 (6 GB), including model loading.

```
python scripts/01_prepare_data.py
python scripts/02_baselines.py
python scripts/03_finetune.py --mode lora --rank 8
python scripts/03_finetune.py --mode full
python scripts/03_finetune.py --mode lora --rank 4
python scripts/03_finetune.py --mode lora --rank 16
python scripts/04_retrieval.py
python scripts/04_retrieval.py --space emb
python scripts/05_failure_analysis.py 0
python scripts/05b_pooled_analysis.py
python scripts/05c_error_counts.py
python scripts/06_summary.py
```

To use another TDC dataset (for example AMES), set the `DATASET` environment variable before running the scripts. In PowerShell: `$env:DATASET="AMES"`.

## Repository layout

```
src/        data, features, baselines, fine-tuning, retrieval, metrics
scripts/    numbered scripts 01 to 06, plus 05b (pooled analysis) and 05c (error counts)
docs/       failure_analysis.md
results/    predictions, retrieval outputs, summary tables (per dataset)
```

## Contact

Tejas Lokhande · [tejas00lokhande@gmail.com] · github.com/tejasiith