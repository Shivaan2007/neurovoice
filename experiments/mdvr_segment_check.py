"""MDVR-KCL read text: 5 s windows, naive random split vs speaker-grouped split.

Shows how cutting long recordings into windows and splitting at random puts the same
person in train and test. This is a demo of one way numbers get inflated; we do not
claim it is what the original hackathon version did.
"""
import csv
import datetime
import warnings

import librosa
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from neurovoice.config import CACHE_DIR, MANIFEST_PATH, ROOT, SAMPLE_RATE
from neurovoice.evaluate import speaker_bootstrap_ci
from neurovoice.features import features_from_signal
from neurovoice.splits import grouped_splits, naive_splits

warnings.filterwarnings("ignore")
WIN = 5 * SAMPLE_RATE


def windows_for(row):
    y, _ = librosa.load(ROOT / row["file"], sr=SAMPLE_RATE, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    out = []
    for i in range(len(y) // WIN):
        f = features_from_signal(y[i * WIN:(i + 1) * WIN])
        f.update(speaker_id=row["speaker_id"], label=row["label"], window=i)
        out.append(f)
    return out


def main():
    cache = CACHE_DIR / "mdvr_read_text_5s_windows.csv"
    if cache.exists():
        df = pd.read_csv(cache)
    else:
        m = pd.read_csv(MANIFEST_PATH)
        m = m[(m.dataset == "mdvr_kcl") & (m.task == "read_text")]
        res = Parallel(n_jobs=-1)(delayed(windows_for)(r) for _, r in m.iterrows())
        df = pd.DataFrame([w for r in res for w in r])
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False)
    meta = ["speaker_id", "label", "window"]
    X, y, g = df.drop(columns=meta).values, df["label"].values, df["speaker_id"].values
    print(f"{len(df)} windows, {len(set(g))} speakers, PD windows={int(y.sum())}")

    models = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "svm_rbf": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
        "random_forest": RandomForestClassifier(300, random_state=42, n_jobs=-1),
    }
    rows = []
    for name, model in models.items():
        r = {"model": name}
        for split, cv in (("naive_random", naive_splits(X, y)),
                          ("speaker_grouped", grouped_splits(X, y, g))):
            p = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
            r[f"auroc_{split}"] = round(roc_auc_score(y, p), 3)
            if split == "speaker_grouped":
                _, lo, hi = speaker_bootstrap_ci(y, p, g)
                r["grouped_ci95"] = f"{lo:.3f}-{hi:.3f}"
        r["inflation"] = round(r["auroc_naive_random"] - r["auroc_speaker_grouped"], 3)
        rows.append(r)
        print(r)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "segment_check.csv", index=False)
    log = ROOT / "experiments" / "log.csv"
    new = not log.exists()
    with log.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "train_sets", "test_set", "features", "model", "auroc", "notes"])
        for r in rows:
            for split in ("naive_random", "speaker_grouped"):
                w.writerow([datetime.date.today(), "mdvr_kcl", "mdvr_kcl_cv", "5s windows",
                            r["model"], r[f"auroc_{split}"], split])


if __name__ == "__main__":
    main()
