"""Unseen-dataset tests (train on one dataset, test on another) + calibration before/after.

Runs on whatever datasets are present: MDVR-KCL always (data/cache or data/raw), Italian
if its window cache or raw files exist (see experiments/italian_baseline.py). NeuroVoz and
wav2vec2 embeddings are not included yet.

Read the Italian rows with the recording confound in mind (docs/data_card.md): all
44.1 kHz files and the whole 2016 session are PD only, so Italian is tested three ways:
all speakers / without young controls / without young controls on 16 kHz recordings only.
A score on "all" mostly measures recording differences, not PD.

For each direction and subset, and each of (raw features, per-dataset z-normalised):
  * zero-shot: train on the source, test on every target speaker
  * adaptation: add 20% of target speakers to training, test on the other 80% (5 draws)
Outputs: results/cross_dataset.csv, results/calibration_before_after.csv.

Run: python -m experiments.cross_dataset
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from neurovoice.config import MANIFEST_PATH, SEED
from neurovoice.evaluate import calibration_report, expected_calibration_error, speaker_bootstrap_ci
from neurovoice.train import META_COLS, calibrated, load_windows, make_models, oof_probabilities, speaker_level

warnings.filterwarnings("ignore")
ROOT_RESULTS = None  # set in main() so tests can import without touching disk
ITALIAN_META = ["cohort", "age"]


def feature_columns(*frames):
    cols = set.intersection(*[set(f.columns) for f in frames])
    return sorted(c for c in cols if c not in META_COLS + ITALIAN_META)


def zscore_per_dataset(df, cols):
    """Standardise a dataset by its OWN mean and std (removes constant per-dataset offsets)."""
    out = df.copy()
    sd = df[cols].std().replace(0, 1.0)
    out[cols] = (df[cols] - df[cols].mean()) / sd
    return out


def fit_predict(name, train, test, cols):
    model = calibrated(make_models()[name], train[cols].to_numpy(float),
                       train["label"].to_numpy(), train["speaker_id"].to_numpy())
    return model.predict_proba(test[cols].to_numpy(float))[:, 1]


def score(test, p):
    y, g = test["label"].to_numpy(), test["speaker_id"].to_numpy()
    point, lo, hi = speaker_bootstrap_ci(y, p, g)
    sy, sp = speaker_level(y, p, g)
    cal = calibration_report(y, p)
    return {"auroc_window": round(point, 3), "ci95": f"{lo:.3f}-{hi:.3f}",
            "auroc_speaker": round(float(roc_auc_score(sy, sp)), 3),
            "brier": round(cal["brier"], 3), "ece": round(cal["ece"], 3),
            "target_speakers": int(len(sy))}


def zero_shot(name, src, tgt, cols):
    return score(tgt, fit_predict(name, src, tgt, cols))


def adaptation(name, src, tgt, cols, frac=0.2, draws=5):
    """Add a fraction of target speakers (stratified by label) to training; test on the rest."""
    spk = tgt.groupby("speaker_id").label.first()
    aucs, briers = [], []
    for d in range(draws):
        rng = np.random.default_rng(SEED + d)
        take = []
        for lab in (0, 1):
            ids = spk[spk == lab].index.to_numpy()
            rng.shuffle(ids)
            take += list(ids[:max(1, int(round(len(ids) * frac)))])
        in_train = tgt.speaker_id.isin(take)
        train = pd.concat([src, tgt[in_train]], ignore_index=True)
        test = tgt[~in_train]
        if test.label.nunique() < 2:
            continue
        p = fit_predict(name, train, test, cols)
        aucs.append(roc_auc_score(test.label, p))
        briers.append(calibration_report(test.label.to_numpy(), p)["brier"])
    return {"auroc_window": round(float(np.mean(aucs)), 3),
            "ci95": f"sd {np.std(aucs):.3f} over {len(aucs)} draws",
            "auroc_speaker": None, "brier": round(float(np.mean(briers)), 3), "ece": None,
            "target_speakers": int(len(spk) - len(take))}


def italian_subsets(it):
    """Subsets of Italian windows that progressively remove known confounds."""
    subsets = {"all": it}
    if "cohort" in it.columns:
        old = it[~it.cohort.astype(str).str.contains("young", case=False)]
        subsets["no_young"] = old
        if MANIFEST_PATH.exists():
            m = pd.read_csv(MANIFEST_PATH)
            m = m[(m.dataset == "italian") & (m.task == "read_text")]
            rates = m.groupby("speaker_id").sample_rate.agg(lambda s: set(s))
            ok = [s for s, r in rates.items() if r == {16000}]
            subsets["no_young_16k_only"] = old[old.speaker_id.isin(ok)]
    return subsets


def run(datasets, models=("random_forest", "logreg")):
    rows = []
    for src_name, tgt_name in [(a, b) for a in datasets for b in datasets if a != b]:
        tgt_sets = italian_subsets(datasets[tgt_name]) if tgt_name == "italian" \
            else {"all": datasets[tgt_name]}
        for subset, tgt in tgt_sets.items():
            if tgt.label.nunique() < 2:
                continue
            src = datasets[src_name]
            cols = feature_columns(src, tgt)
            for norm in ("raw", "per_dataset_z"):
                s = zscore_per_dataset(src, cols) if norm == "per_dataset_z" else src
                t = zscore_per_dataset(tgt, cols) if norm == "per_dataset_z" else tgt
                for name in models:
                    base = {"train": src_name, "test": tgt_name, "test_subset": subset,
                            "features": norm, "model": name}
                    rows.append({**base, "setting": "zero_shot", **zero_shot(name, s, t, cols)})
                    rows.append({**base, "setting": "adapt_20pct",
                                 **adaptation(name, s, t, cols)})
                    print(rows[-2], flush=True)
    return pd.DataFrame(rows)


def calibration_before_after(mdvr):
    """Random Forest on MDVR-KCL: uncalibrated vs sigmoid-calibrated held-out probabilities."""
    from sklearn.model_selection import cross_val_predict

    from neurovoice.splits import grouped_splits
    cols = feature_columns(mdvr)
    X, y, g = mdvr[cols].to_numpy(float), mdvr.label.to_numpy(), mdvr.speaker_id.to_numpy()
    raw = cross_val_predict(make_models()["random_forest"], X, y,
                            cv=grouped_splits(X, y, g, 5), method="predict_proba")[:, 1]
    cal = oof_probabilities("random_forest", X, y, g)
    rows = []
    for label, p in (("before (raw forest)", raw), ("after (sigmoid, nested)", cal)):
        r = calibration_report(y, p)
        rows.append({"probabilities": label, "brier": round(r["brier"], 3),
                     "ece": round(expected_calibration_error(y, p), 3),
                     "auroc_window": round(float(roc_auc_score(y, p)), 3)})
    return pd.DataFrame(rows)


def load_datasets():
    datasets = {"mdvr_kcl": load_windows()}
    try:
        from experiments.italian_baseline import CACHE, load_windows as load_italian
        if CACHE.exists() or MANIFEST_PATH.exists():
            it = load_italian()
            if len(it):
                datasets["italian"] = it
    except Exception as exc:  # no Italian data on this machine
        print(f"Italian data not available here ({type(exc).__name__}); skipping it.")
    try:
        from neurovoice.neurovoz import load_windows as load_neurovoz
        datasets["neurovoz"] = load_neurovoz()
    except Exception as exc:  # NeuroVoz needs a Data Usage Agreement; usually absent
        print(f"NeuroVoz not available here ({type(exc).__name__}); skipping it.")
    return datasets


def main():
    from neurovoice.config import ROOT
    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    datasets = load_datasets()
    cal = calibration_before_after(datasets["mdvr_kcl"])
    cal.to_csv(res / "calibration_before_after.csv", index=False)
    print(cal.to_string(index=False))
    if len(datasets) < 2:
        print("Only one dataset available: cross-dataset test skipped. Put the Italian "
              "data in data/raw/italian/ and rerun on that machine.")
        return
    out = run(datasets)
    out.to_csv(res / "cross_dataset.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
