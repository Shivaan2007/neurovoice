"""Recording-conditions check: can the same features tell which dataset a clip came from?

Trains a classifier to predict the DATASET (every dataset present; two today) from the same hand-built
features, with speaker-grouped CV so a person never sits in train and test. If it is near
perfect, the features carry recording information (microphone, room, session) that has
nothing to do with Parkinson's, and cross-dataset drops partly reflect that.

Outputs: results/dataset_classifier.csv (metrics) and results/dataset_classifier_confusion.csv.
Run: python -m experiments.dataset_classifier
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.cross_dataset import feature_columns, load_datasets
from neurovoice.config import ROOT, SEED
from neurovoice.evaluate import speaker_bootstrap_ci
from neurovoice.splits import grouped_splits

warnings.filterwarnings("ignore")


def main():
    datasets = load_datasets()
    if len(datasets) < 2:
        raise SystemExit("needs at least two datasets (e.g. data/raw/italian filled)")
    names = sorted(datasets)
    frames = []
    for i, name in enumerate(names):
        d = datasets[name].copy()
        d["dataset_label"] = i
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    cols = feature_columns(*datasets.values())
    X = df[cols].to_numpy(float)
    y, g = df["dataset_label"].to_numpy(), df["speaker_id"].to_numpy()
    chance = float(pd.Series(y).value_counts(normalize=True).max())

    models = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "random_forest": RandomForestClassifier(300, random_state=SEED, n_jobs=-1),
    }
    rows, conf = [], []
    for name, model in models.items():
        pred = cross_val_predict(model, X, y, cv=grouped_splits(X, y, g, 5))
        acc, lo, hi = speaker_bootstrap_ci(y, pred, g, metric=lambda a, b: float((a == b).mean()))
        bal, blo, bhi = speaker_bootstrap_ci(y, pred, g, metric=balanced_accuracy_score)
        spk = pd.DataFrame({"g": g, "y": y, "p": pred}).groupby("g").agg(
            y=("y", "first"), p=("p", lambda s: s.mode().iloc[0]))
        spk_acc = float((spk.p == spk.y).mean())
        rows.append({"model": name, "window_accuracy": round(acc, 3),
                     "accuracy_ci95": f"{lo:.3f}-{hi:.3f}",
                     "balanced_accuracy": round(bal, 3), "balanced_ci95": f"{blo:.3f}-{bhi:.3f}",
                     "speaker_accuracy": round(spk_acc, 3),
                     "chance_majority_class": round(float(chance), 3),
                     "windows": int(len(y)), "speakers": int(len(spk))})
        cm = confusion_matrix(y, pred)
        conf.append(pd.DataFrame(cm, index=[f"true_{n}" for n in names],
                                 columns=[f"pred_{n}" for n in names]).assign(model=name))
        print(rows[-1])
    res = ROOT / "results"
    pd.DataFrame(rows).to_csv(res / "dataset_classifier.csv", index=False)
    pd.concat(conf).to_csv(res / "dataset_classifier_confusion.csv")
    print(pd.concat(conf).to_string())
    np.seterr(all="ignore")


if __name__ == "__main__":
    main()
