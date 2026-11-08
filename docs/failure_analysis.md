# Failure analysis (hERG, seed 0)

**Setup:** LoRA ChemBERTa (`lora_r8`), seed 0 scaffold split, Morgan fingerprint retrieval (k = 5, neighbours from the training set only). I took the 20 most confident errors, ranked by |p − 0.5|. The numbers come from `scripts/05c_error_counts.py`, and the full list is in `results/herg/top_errors_seed0.csv`. This is one seed and 20 molecules, so everything below is descriptive, not a tested conclusion.

## Error types

I examined the 20 most confident prediction errors from seed 0. Of these, 18 were false positives and 2 were false negatives, so most of the errors came from molecules wrongly predicted as hERG blockers. About 69% of molecules in the dataset are hERG positive, so a model that leans positive would be expected to make mostly false-positive errors. I think this explains part of the 18 to 2 split, but I haven't tested it.

## Similarity to the training set

The median top-1 similarity of these 20 errors was 0.34, compared with 0.43 for all test molecules in seed 0. The errors were somewhat less similar to their closest training molecule, but this is a small gap from one seed. It is not enough to say that the errors come from unfamiliar chemical space. In the pooled results over five seeds, accuracy was also fairly high in the lowest similarity bin (0.75), so low similarity alone does not explain the mistakes.

## Do the neighbours flag these errors?

I also checked whether the nearest neighbours disagreed with the model's prediction. Only 2 of the 20 did. For the other 18, the neighbours' majority agreed with the model's wrong prediction, so the neighbour-agreement flag would not have caught them. This fits with the pooled result: disagreement is a useful warning on average (accuracy 0.81 when neighbours agree, 0.53 when they disagree, fingerprint retrieval), but most confident errors do not trigger it.

## Stereoisomers

Molecules m00450 and m00449 are stereoisomers. Both received the same probability (0.98849) and the same top-1 similarity (0.2375). My Morgan fingerprints do not encode chirality, so retrieval cannot tell these two apart. That is a limitation of my fingerprint setup. The identical probability is consistent with the model also not separating them, but I have not checked how the tokenizer handles the stereochemistry marks in the SMILES, so I can't say why. [Check: do both molecules have the same top-1 neighbour? Same similarity suggests so, but I haven't confirmed it.]

## What I would test next (not done yet)

Turn on chirality in the fingerprints and see whether pairs like m00449 and m00450 are then separated in retrieval.