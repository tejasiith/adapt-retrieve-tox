from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

def compute_metrics(y, prob):
    return {"auroc": roc_auc_score(y, prob),
            "auprc": average_precision_score(y, prob),
            "acc": accuracy_score(y, (prob >= 0.5).astype(int))}