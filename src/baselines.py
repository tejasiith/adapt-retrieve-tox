from sklearn.ensemble import RandomForestClassifier
from .features import morgan_np

def run_rf(train_df, test_df, seed):
    X_tr, X_te = morgan_np(train_df.smiles), morgan_np(test_df.smiles)
    clf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=seed,
                                 class_weight="balanced")
    clf.fit(X_tr, train_df.label.values)
    return clf.predict_proba(X_te)[:, 1]