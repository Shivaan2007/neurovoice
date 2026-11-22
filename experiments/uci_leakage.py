"""UCI tabular set: naive recording-level split vs subject-grouped split.

Each subject has 3 recordings. A naive split puts recordings of the same person in
train and test, so the model can match people instead of learning PD. The gap between
the two AUROCs is the leakage. We also report subject-level AUROC (each subject's
3 probabilities averaged), since the subject, not the recording, is the real unit.
"""
import csv
import datetime
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from neurovoice.config import ROOT, SEED
from neurovoice.evaluate import speaker_bootstrap_ci
from neurovoice.splits import grouped_splits, naive_splits
from neurovoice.uci import load_uci

warnings.filterwarnings("ignore")


def subject_level_auroc(y, p, groups):
    df = pd.DataFrame({"y": y, "p": p, "g": groups}).groupby("g").agg(y=("y", "first"),
                                                                       p=("p", "mean"))
    return float(roc_auc_score(df.y, df.p))


def run(X, y, groups):
    # Scaling sits inside the pipeline, so it is fit on training folds only.
    models = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        "svm_rbf": make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEED)),
        "random_forest": RandomForestClassifier(300, random_state=SEED, n_jobs=-1),
    }
    X = np.asarray(X)
    rows = []
    for name, model in models.items():
        r = {"model": name}
        for split, cv in (("naive_random", naive_splits(X, y)),
                          ("subject_grouped", grouped_splits(X, y, groups))):
            p = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
            r[f"auroc_{split}"] = round(roc_auc_score(y, p), 3)
            if split == "subject_grouped":
                _, lo, hi = speaker_bootstrap_ci(y, p, groups)
                r["grouped_ci95"] = f"{lo:.3f}-{hi:.3f}"
                r["auroc_grouped_subject_level"] = round(subject_level_auroc(y, p, groups), 3)
        r["inflation"] = round(r["auroc_naive_random"] - r["auroc_subject_grouped"], 3)
        rows.append(r)
        print(r)
    return pd.DataFrame(rows)


def main():
    X, y, groups = load_uci()
    print(f"{len(y)} recordings, {len(set(groups))} subjects, {X.shape[1]} features, "
          f"PD recordings={int(y.sum())}")
    out = run(X, y, groups)
    (ROOT / "results").mkdir(exist_ok=True)
    out.to_csv(ROOT / "results" / "uci_leakage.csv", index=False)
    log = ROOT / "experiments" / "log.csv"
    new = not log.exists()
    with log.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "train_sets", "test_set", "features", "model", "auroc", "notes"])
        for _, r in out.iterrows():
            for split in ("naive_random", "subject_grouped"):
                w.writerow([datetime.date.today(), "uci", "uci_cv", "uci 754 features",
                            r["model"], r[f"auroc_{split}"], split])


if __name__ == "__main__":
    main()
