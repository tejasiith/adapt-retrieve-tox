# AMES LoRA retrieval demo

Research demo for the adapt-retrieve-tox project: a LoRA-adapted ChemBERTa-77M predicts Ames mutagenicity, and the 5 nearest training molecules (Tanimoto similarity on Morgan fingerprints) are shown as an evidence check. Not for safety or regulatory decisions.

Code and write-up: https://github.com/tejasiith/adapt-retrieve-tox

## Data attribution

Ames mutagenicity dataset from Therapeutics Data Commons (TDC), licence CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/), aggregated from Xu et al., "In silico prediction of chemical Ames mutagenicity", J. Chem. Inf. Model. 52(11), 2012. Please also cite Huang et al., "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development", NeurIPS 2021.

Changes made: SMILES were canonicalised with RDKit, duplicate molecules were removed, and only the training portion of one scaffold split (seed 0) is included in train.csv. The TDC page lists the licence as "Not Specified. CC BY 4.0"; if you are a rights holder and have concerns, please contact the author.

## Licensing notes

- Code (app.py): MIT licence, copyright Tejas Lokhande.
- Base model: DeepChem/ChemBERTa-77M-MLM is loaded from the Hugging Face Hub at start-up and is not redistributed here. The checkpoint does not declare a licence, so its terms are unspecified.
- adapter/: LoRA adapter and classifier head trained by the author on top of that base model. Research demonstration only.
- Data: see Data attribution above.