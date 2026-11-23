"""Train the demo model on MDVR-KCL read text and save models/model.joblib + meta.json.

Steps (every number that guides a choice comes from speakers the model did not train on):
1. Features for each 5 s window of each read-text recording (cached in data/cache/).
2. For LR, SVM and RF: calibrated probabilities from speaker-grouped 5-fold CV, with
   calibration fit on inner grouped folds of each training split (nested, no leakage).
3. Pick the model with the best speaker-level AUROC (mean probability per speaker).
4. Choose the "not enough confidence" band from those held-out window probabilities.
5. Refit the chosen model on all speakers and save it with its metadata.

Run: python -m neurovoice.train
"""
import datetime
import json
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from neurovoice.config import (
    CACHE_DIR,
    MANIFEST_PATH,
    META_PATH,
    MODEL_PATH,
    NOTICE,
    ROOT,
    SEED,
)
from neurovoice.evaluate import calibration_report, speaker_bootstrap_ci
from neurovoice.features import WINDOW_SECONDS, load_audio, window_features
from neurovoice.splits import grouped_splits

warnings.filterwarnings("ignore")
META_COLS = ["speaker_id", "label", "window"]
WINDOW_CACHE = CACHE_DIR / "mdvr_read_text_5s_windows.csv"  # shared with the segment check
TARGET_PRECISION = 0.8  # the cut is TUNED to reach this on the held-out windows it is fit on
MIN_IN_BAND = 10
MIN_SPEAKERS_IN_BAND = 5  # windows from one person are correlated, so count people too


def make_models():
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "svm_rbf": make_pipeline(StandardScaler(), SVC(random_state=SEED)),
        "random_forest": RandomForestClassifier(300, random_state=SEED, n_jobs=-1),
    }


def load_windows():
    if WINDOW_CACHE.exists():
        return pd.read_csv(WINDOW_CACHE)
    from joblib import Parallel, delayed

    if not MANIFEST_PATH.exists():
        from neurovoice.manifest import build_manifest
        build_manifest()
    m = pd.read_csv(MANIFEST_PATH)
    m = m[(m.dataset == "mdvr_kcl") & (m.task == "read_text")]

    def one(row):
        feats = window_features(load_audio(str(ROOT / row["file"])))
        return [{**f, "speaker_id": row["speaker_id"], "label": row["label"], "window": i}
                for i, f in enumerate(feats)]

    res = Parallel(n_jobs=-1)(delayed(one)(r) for _, r in m.iterrows())
    df = pd.DataFrame([w for r in res for w in r])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(WINDOW_CACHE, index=False)
    return df


def calibrated(model, X, y, groups, n_splits=3):
    """Sigmoid calibration fit on speaker-grouped folds of (X, y)."""
    return CalibratedClassifierCV(model, method="sigmoid",
                                  cv=grouped_splits(X, y, groups, n_splits)).fit(X, y)


def oof_probabilities(name, X, y, groups, n_splits=5):
    """Held-out calibrated probability for every window (nested speaker-grouped CV)."""
    p = np.zeros(len(y))
    for tr, te in grouped_splits(X, y, groups, n_splits):
        model = calibrated(make_models()[name], X[tr], y[tr], groups[tr])
        p[te] = model.predict_proba(X[te])[:, 1]
    return p


def speaker_level(y, p, groups):
    df = pd.DataFrame({"y": y, "p": p, "g": groups}).groupby("g").agg(y=("y", "first"),
                                                                       p=("p", "mean"))
    return df.y.to_numpy(), df.p.to_numpy()


def pick_thresholds(y, p, target=TARGET_PRECISION, min_n=MIN_IN_BAND, groups=None,
                    min_speakers=MIN_SPEAKERS_IN_BAND):
    """low: largest cut where windows below it are healthy >= target of the time.
    high: smallest cut where windows above it are PD >= target of the time.
    If no cut reaches the target, that side stays closed (0 or 1), so the app says
    "not enough confidence" rather than guess.

    The precision reached here is IN-SAMPLE for the data the cut is fit on. Use
    crossfit_thresholds for an honest estimate. With `groups`, a band must also hold
    windows from at least `min_speakers` different people.
    """
    y, p = np.asarray(y), np.asarray(p)
    g = None if groups is None else np.asarray(groups)

    def enough(mask):
        return mask.sum() >= min_n and (g is None or len(set(g[mask])) >= min_speakers)

    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    low = max((t for t in grid if enough(p < t) and (y[p < t] == 0).mean() >= target),
              default=0.0)
    high = min((t for t in grid if enough(p > t) and (y[p > t] == 1).mean() >= target),
               default=1.0)
    if low > high:  # bands would overlap; collapse to a single cut
        low = high = round((low + high) / 2, 2)
    return float(low), float(high)


def crossfit_thresholds(y, p, groups, n_folds=5):
    """Pick the band cuts on some speakers and measure band precision on the others.

    Returns (per-fold table, pooled precision for "higher", pooled precision for "lower").
    """
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    rng = np.random.default_rng(SEED)
    spk = np.unique(groups)
    rng.shuffle(spk)
    rows = []
    for k, held in enumerate(np.array_split(spk, n_folds)):
        te = np.isin(groups, held)
        low, high = pick_thresholds(y[~te], p[~te], groups=groups[~te])
        up, dn = te & (p > high), te & (p < low)
        rows.append({
            "fold": k, "low": low, "high": high,
            "n_higher": int(up.sum()), "higher_speakers": int(len(set(groups[up]))),
            "higher_hits": int((y[up] == 1).sum()),
            "higher_precision": round(float((y[up] == 1).mean()), 3) if up.any() else None,
            "n_lower": int(dn.sum()), "lower_speakers": int(len(set(groups[dn]))),
            "lower_hits": int((y[dn] == 0).sum()),
            "lower_precision": round(float((y[dn] == 0).mean()), 3) if dn.any() else None,
        })
    table = pd.DataFrame(rows)
    pooled = {side: (round(float(table[f"{side}_hits"].sum() / table[f"n_{side}"].sum()), 3)
                     if table[f"n_{side}"].sum() else None)
              for side in ("higher", "lower")}
    return table, pooled["higher"], pooled["lower"]


def evaluate_models(X, y, groups, names=None):
    rows, oof = [], {}
    for name in names or make_models():
        p = oof_probabilities(name, X, y, groups)
        oof[name] = p
        point, lo, hi = speaker_bootstrap_ci(y, p, groups)
        sy, sp = speaker_level(y, p, groups)
        _, slo, shi = speaker_bootstrap_ci(sy, sp, np.arange(len(sy)))  # one row per speaker
        rows.append({"model": name,
                     "auroc_window": round(point, 3),
                     "window_ci95": f"{lo:.3f}-{hi:.3f}",
                     "auroc_speaker": round(roc_auc_score(sy, sp), 3),
                     "speaker_ci95": f"{slo:.3f}-{shi:.3f}",
                     **{k: round(v, 3) for k, v in calibration_report(y, p).items()}})
    return pd.DataFrame(rows), oof


def train(df, model_path=MODEL_PATH, meta_path=META_PATH, results_path=None, names=None):
    feature_names = [c for c in df.columns if c not in META_COLS]
    X = df[feature_names].to_numpy(dtype=float)
    y, groups = df["label"].to_numpy(), df["speaker_id"].to_numpy()

    results, oof = evaluate_models(X, y, groups, names)
    best = results.sort_values("auroc_speaker", ascending=False).iloc[0]
    low, high = pick_thresholds(y, oof[best.model], groups=groups)
    cf_table, cf_higher, cf_lower = crossfit_thresholds(y, oof[best.model], groups)
    in_sample = oof[best.model] > high
    bands = pd.Series(np.where(oof[best.model] < low, "lower",
                               np.where(oof[best.model] > high, "higher", "uncertain")))

    final = calibrated(make_models()[best.model], X, y, groups)
    import joblib
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, model_path)
    meta = {
        "model_version": f"mdvr-kcl-{best.model}-v1",
        "model": best.model,
        "trained_on": "MDVR-KCL read text (English, smartphone), 5 s windows",
        "training_datasets": ["mdvr_kcl"],
        "cross_dataset_headline": "pending: no unseen-dataset result yet "
                                  "(experiments/cross_dataset.py needs the Italian data)",
        "limits": NOTICE,
        "n_speakers": int(len(set(groups))),
        "n_windows": int(len(y)),
        "window_seconds": WINDOW_SECONDS,
        "feature_names": feature_names,
        "low": low,
        "high": high,
        "held_out_band_share": bands.value_counts(normalize=True).round(3).to_dict(),
        "band_precision": {
            "higher_in_sample": (round(float((y[in_sample] == 1).mean()), 3)
                                 if in_sample.any() else None),
            "higher_crossfit": cf_higher,
            "lower_crossfit": cf_lower,
            "crossfit_higher_per_fold": cf_table["higher_precision"].tolist(),
            "note": "in_sample is what the cut was tuned on and is optimistic; crossfit "
                    "picks the cut on other speakers and is the number to quote.",
        },
        "cv": best.drop("model").to_dict(),
        "selection_note": "Best of 3 models on the same CV, so its score is slightly "
                          "optimistic; all three are in results/train_cv.csv.",
        "trained": datetime.date.today().isoformat(),
        "sklearn": sklearn.__version__,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    if results_path:
        results.to_csv(results_path, index=False)
    return results, meta


def main():
    df = load_windows()
    print(f"{len(df)} windows, {df.speaker_id.nunique()} speakers, "
          f"PD windows={int(df.label.sum())}")
    (ROOT / "results").mkdir(exist_ok=True)
    results, meta = train(df, results_path=ROOT / "results" / "train_cv.csv")
    print(results.to_string(index=False))
    print(f"chosen: {meta['model']}  band: lower < {meta['low']} | "
          f"uncertain | higher > {meta['high']}  share: {meta['held_out_band_share']}")


if __name__ == "__main__":
    main()
