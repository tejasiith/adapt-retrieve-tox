import numpy as np
from sklearn.metrics import roc_auc_score

def tanimoto(A, B):
    A, B = A.astype(np.float32), B.astype(np.float32)
    inter = A @ B.T
    union = A.sum(1)[:, None] + B.sum(1)[None, :] - inter
    return inter / np.maximum(union, 1e-9)

def cosine(A, B):
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    return A @ B.T

def get_neighbours(sim, ref_df, k=5):
    """sim: (n_query, n_ref) similarity matrix. Returns a list of neighbour lists."""
    top = np.argsort(-sim, axis=1)[:, :k]
    out = []
    for i, row in enumerate(top):
        out.append([{"mol_id": ref_df.mol_id.iloc[j],
                     "smiles": ref_df.smiles.iloc[j],
                     "label": int(ref_df.label.iloc[j]),
                     "sim": float(sim[i, j])} for j in row])
    return out

def neighbour_stats(records):
    y = np.array([r["true_label"] for r in records])
    pred = np.array([r["pred_label"] for r in records])
    vote = np.array([np.mean([n["label"] for n in r["neighbours"]]) for r in records])
    agree = (vote >= 0.5).astype(int) == pred
    correct = pred == y
    return {
        "knn_vote_auroc": roc_auc_score(y, vote),      # neighbours alone as a classifier
        "agree_rate": agree.mean(),
        "n_agree": int(agree.sum()),
        "n_disagree": int((~agree).sum()),
        "acc_if_agree": correct[agree].mean() if agree.any() else np.nan,
        "acc_if_disagree": correct[~agree].mean() if (~agree).any() else np.nan,
    }