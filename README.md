# Adapting a pretrained molecular model with LoRA, and checking predictions against retrieved neighbours (hERG and AMES)

A small study on two binary toxicity tasks from Therapeutics Data Commons (TDC): hERG (cardiotoxicity) and AMES (mutagenicity). I asked three questions:

1. How well does a pretrained molecular transformer (ChemBERTa) adapt to these tasks with LoRA, compared with full fine-tuning and a random-forest baseline?
2. Can the nearest training molecules tell us when a prediction is unreliable?
3. Where does the model fail?

This is a study of adaptation and evidence checking, with toxicity prediction as the test case. It is not an attempt to build a better hERG or AMES predictor. It is separate from my earlier multimodal hERG model [https://github.com/tejasiith/herg-multimodal-property-prediction].

## Summary of findings

- **LoRA matches full fine-tuning at a fraction of the cost.** On hERG, LoRA r16 and full fine-tuning had the same mean test AUROC (0.822 vs 0.824). On AMES, LoRA r16 was slightly ahead (0.830 vs 0.823). LoRA trains about 6% of the parameters (this count includes the classification head). On AMES, 5 seeds took about 7 minutes with LoRA r16 and about 12 minutes with full fine-tuning (approximate, timed by eye).
- **The random-forest baseline is strong.** On hERG, LoRA r16 and full fine-tuning were about 0.02 AUROC above it on average and beat it on 4 of 5 seeds, but with about 131 test molecules per seed and a seed-to-seed spread of about 0.1, I treat that as suggestive, not proven. On AMES, the random forest was ahead of every ChemBERTa variant on all 5 seeds (mean AUROC 0.847 vs 0.823 to 0.830).
- **Neighbour disagreement flags errors on both datasets.** When the nearest training molecules disagreed with the model, accuracy dropped. With fingerprint retrieval: 0.81 vs 0.53 on hERG, and 0.83 vs 0.48 on AMES.
- **The flag misses most confident errors.** For 18 of the 20 most confident errors in hERG seed 0, the neighbours agreed with the wrong prediction.

## Data and splits

| | hERG | AMES |
|---|---|---|
| Molecules after cleaning | 648 | 7,255 |
| Positive rate | 0.69 | 0.545 |
| Train / validation / test | 453 / 64 / 131 | 5,078 / 725 / 1,452 |
| Test positive rate across seeds | 0.63 to 0.78 | 0.59 to 0.61 |

- Source: TDC (`tdc.single_pred.Tox`). SMILES were canonicalised with RDKit and duplicates removed.
- Splits: scaffold splits (Murcko scaffolds), 70 / 10 / 20. Each seed (0 to 4) gives a different scaffold split and a different training run, so the std values below mix split variation and training variation.
- On AMES the test positive rate (about 0.60) is higher than the overall rate (0.545). The scaffold split fills the training set with large scaffold groups first, so the test set gets the small, rarer scaffolds.

## Methods

| Item | Setting |
|---|---|
| Pretrained model | `DeepChem/ChemBERTa-77M-MLM` (SMILES input, max length 128) |
| LoRA | targets `query` and `value`, alpha = 2 × rank, dropout 0.1, ranks 4 / 8 / 16 |
| Full fine-tuning | all weights trainable, learning rate 3e-5 |
| Training | AdamW, weight decay 0.01, batch size 32, 20 epochs, LoRA learning rate 5e-4 |
| Model selection | best epoch by validation AUROC; test set used once |
| Baseline | random forest (500 trees, balanced class weights) on Morgan fingerprints (radius 2, 2048 bits, no chirality) |
| Retrieval | k = 5 nearest training molecules (never test molecules). Two spaces: Tanimoto on Morgan fingerprints, and cosine similarity on mean-pooled embeddings of the fine-tuned `lora_r8` model |

Hyperparameters were not tuned. The values above are starting points I chose before running. The rank 4 ablation was run on hERG only. On AMES I ran r8, r16 and full fine-tuning.

## Results: hERG

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

- AUROC is the main metric. About 69% of hERG molecules are positive, so AUPRC and accuracy are inflated by the base rate. The random forest has the best accuracy even though its AUROC is lower than full and r16.
- Full fine-tuning and LoRA r16 each beat the random forest on 4 of 5 seeds (both lose on seed 4). The mean gap is about 0.02, small relative to the spread.
- r16 had higher AUROC than r4 on all 5 seeds, so higher rank looks slightly better here. The differences between ranks are small relative to the noise.
- Validation AUROC is higher than test AUROC on most seeds because the best epoch is picked on 64 validation molecules. Only test numbers are reported.
- Rerunning LoRA r8 reproduced the summary table to three decimals.

### Retrieval as a check on predictions (hERG)

Pooled over 5 seeds (655 test predictions), using `lora_r8` predictions. "Agree" means the majority label of the 5 neighbours matches the model's predicted label.

| Retrieval space | Accuracy if agree | Accuracy if disagree | Difference, approx. 95% bootstrap CI |
|---|---|---|---|
| Morgan fingerprint (Tanimoto) | 0.813 (n = 563) | 0.533 (n = 92) | 0.28 [0.175, 0.390] |
| Fine-tuned embedding (cosine) | 0.841 (n = 574) | 0.296 (n = 81) | 0.55 [0.443, 0.644] |

Accuracy by similarity to the closest training molecule (pooled, fingerprint retrieval):

| Top-1 Tanimoto similarity | Accuracy | n |
|---|---|---|
| ≤ 0.3 | 0.750 | 236 |
| 0.3 to 0.5 | 0.739 | 211 |
| 0.5 to 0.7 | 0.817 | 126 |
| > 0.7 | 0.866 | 82 |

### Failure analysis (hERG, seed 0)

I examined the 20 most confident errors in seed 0 (`lora_r8`). 18 were false positives and 2 were false negatives. Their median top-1 similarity was 0.34, compared with 0.43 for all test molecules. For 18 of the 20, the neighbours agreed with the wrong prediction, so the neighbour-agreement flag would not have caught them. One pair of stereoisomers (m00449 and m00450) received identical scores. My Morgan fingerprints do not encode chirality, so retrieval cannot separate them.

This is one seed and 20 molecules, so it is descriptive only. Details are in [docs/failure_analysis.md](docs/failure_analysis.md).

## Results: AMES

### Test performance (mean ± std over 5 seeds)

| Method | AUROC | AUPRC | Accuracy |
|---|---|---|---|
| Random forest (Morgan) | 0.847 ± 0.021 | 0.888 ± 0.015 | 0.777 ± 0.020 |
| LoRA r16 | 0.830 ± 0.020 | 0.877 ± 0.013 | 0.749 ± 0.018 |
| LoRA r8 | 0.828 ± 0.016 | 0.872 ± 0.010 | 0.743 ± 0.019 |
| Full fine-tuning | 0.823 ± 0.022 | 0.872 ± 0.014 | 0.746 ± 0.015 |

How I read this:

- The random forest was better than every ChemBERTa variant on all 5 seeds, by about 0.02 AUROC on average. With 1,452 test molecules per seed, and the gap in the same direction every time, this is a more reliable difference than anything on hERG.
- LoRA r8, LoRA r16 and full fine-tuning are within about 0.007 AUROC of each other, so I don't claim any ordering among them. Full fine-tuning did not improve on LoRA.
- Possible reasons for the gap are that ChemBERTa-77M is small, that my settings were not tuned, and that mutagenicity depends on specific substructures that Morgan fingerprints capture directly. I did not test any of these.
- Runtime on the RTX 4050: about 7 minutes for 5 seeds of LoRA r16 and about 12 minutes for full fine-tuning (approximate, timed by eye).

### Retrieval as a check on predictions (AMES)

Pooled over 5 seeds (7,260 test predictions), using `lora_r8` predictions.

| Retrieval space | Accuracy if agree | Accuracy if disagree | Difference, approx. 95% bootstrap CI |
|---|---|---|---|
| Morgan fingerprint (Tanimoto) | 0.830 (n = 5,436) | 0.484 (n = 1,824) | 0.35 [0.323, 0.371] |
| Fine-tuned embedding (cosine) | 0.813 (n = 6,032) | 0.398 (n = 1,228) | 0.41 [0.385, 0.444] |

Accuracy by similarity to the closest training molecule (pooled, fingerprint retrieval):

| Top-1 Tanimoto similarity | Accuracy | n |
|---|---|---|
| ≤ 0.3 | 0.721 | 728 |
| 0.3 to 0.5 | 0.718 | 3,377 |
| 0.5 to 0.7 | 0.769 | 2,504 |
| > 0.7 | 0.799 | 651 |

- The same pattern as hERG appears on a second, much larger dataset: accuracy is clearly lower when the neighbours disagree with the model.
- Disagreement is more frequent on AMES (25% of predictions with fingerprint retrieval, against 14% on hERG).
- The similarity trend is modest here: accuracy rises from 0.72 to 0.80, and the two lowest bins are about equal.
- For context, the neighbour vote alone (no model) had a mean AUROC of 0.808 with fingerprints and 0.823 with fine-tuned embeddings, compared with 0.828 for LoRA r8.
- I did not write a failure analysis for AMES.

### Caveats for both retrieval analyses

- Fingerprint retrieval is independent of the model, so it is the cleaner evidence. The embedding result is partly circular, because the embeddings come from the same fine-tuned model that was trained on the training labels.
- The pooled rows come from 5 overlapping test sets, so the same molecule can appear more than once. The intervals are likely somewhat too narrow.
- I did not test whether using the neighbours improves predictions. I only tested whether disagreement flags errors.

## Limitations

- Two binary toxicity datasets. On hERG (648 molecules), differences of 0.01 to 0.02 AUROC between methods should not be trusted.
- One pretrained model (ChemBERTa-77M). I did not compare other molecular foundation models.
- Hyperparameters were not tuned, and I used 5 seeds. The std mixes split variation and training variation.
- Fingerprints ignore chirality, so retrieval cannot distinguish stereoisomers.
- The retrieval analysis uses `lora_r8` only.
- The failure analysis covers hERG seed 0 and 20 molecules. There is none for AMES.
- The rank 4 ablation was run on hERG only.
- Runtimes on AMES are approximate.
- No satellite imagery, time series, or environmental data. No comparison with my multimodal hERG model on these splits.

## Reproducing

Environment used: Python 3.12, PyTorch 2.6.0+cu124, transformers 4.57.6, peft 0.14.0, RDKit 2026.9.1, PyTDC 0.3.6, on an NVIDIA RTX 4050 (6 GB). PyTorch 2.6.0 with CUDA 12.4 comes from the PyTorch wheel index (`--index-url https://download.pytorch.org/whl/cu124`). hERG: about 1 minute for 5 seeds of LoRA r8 (including model loading).

```
python scripts/01_prepare_data.py
python scripts/02_baselines.py
python scripts/03_finetune.py --mode lora --rank 8
python scripts/03_finetune.py --mode lora --rank 16
python scripts/03_finetune.py --mode full
python scripts/03_finetune.py --mode lora --rank 4     # hERG only
python scripts/04_retrieval.py
python scripts/04_retrieval.py --space emb
python scripts/05_failure_analysis.py 0
python scripts/05b_pooled_analysis.py
python scripts/05c_error_counts.py
python scripts/06_summary.py
```

The default dataset is hERG. To run AMES, set the `DATASET` environment variable first. In PowerShell: `$env:DATASET="AMES"` (and `Remove-Item Env:DATASET` to go back). Data goes to `data/<dataset>/` and results to `results/<dataset>/`, so the two datasets never overwrite each other.

## Repository layout

```
src/        data, features, baselines, fine-tuning, retrieval, metrics
scripts/    numbered scripts 01 to 06, plus 05b (pooled analysis) and 05c (error counts)
docs/       failure_analysis.md (hERG)
results/    predictions, retrieval outputs, summary tables (per dataset)
```

## Contact

Tejas Lokhande · [tejas00lokhande@gmail.com] · github.com/tejasiith

