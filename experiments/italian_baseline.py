"""Italian read text: speaker-grouped baseline, with and without the young controls.

The Italian healthy group mixes 22 elderly (60-77) and 15 young (19-29) controls, while
PD speakers are 40-80. A model could score well just by hearing "young voice = healthy".
We run the same calibrated, speaker-grouped CV as train.py on:
  * all speakers
  * without young controls (age-matched comparison: the honest number)
and an "age only" baseline (AUROC of age alone) to show how much age gives away.

Finding (2026-11-22): every subset scores ~0.99-1.0, and a single feature (mean of MFCC 8,
which mostly reflects microphone and room) separates PD from healthy perfectly even in
the matched subset (no young controls, 16 kHz files only). All 44.1 kHz files and the
2016 session are PD only. Hand-built features never separate real PD speech this cleanly,
so PD and control recordings differ in how they were made. Treat this dataset as a
recording-confound example, not as evidence of PD detection, and do not train the demo
model on it.
"""
import warnings

import pandas as pd
from sklearn.metrics import roc_auc_score

from neurovoice.config import CACHE_DIR, MANIFEST_PATH, ROOT
from neurovoice.features import load_audio, window_features
from neurovoice.train import evaluate_models

warnings.filterwarnings("ignore")
CACHE = CACHE_DIR / "italian_read_text_5s_windows.csv"
META = ["speaker_id", "label", "window", "cohort", "age"]


def load_windows():
    if CACHE.exists():
        return pd.read_csv(CACHE)
    from joblib import Parallel, delayed

    m = pd.read_csv(MANIFEST_PATH)
    m = m[(m.dataset == "italian") & (m.task == "read_text")]

    def one(row):
        feats = window_features(load_audio(str(ROOT / row["file"])))
        return [{**f, "speaker_id": row["speaker_id"], "label": row["label"],
                 "cohort": row["cohort"], "age": row["age"], "window": i}
                for i, f in enumerate(feats)]

    res = Parallel(n_jobs=-1)(delayed(one)(r) for _, r in m.iterrows())
    df = pd.DataFrame([w for r in res for w in r])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, index=False)
    return df


def age_only_auroc(df):
    spk = df.groupby("speaker_id").agg(label=("label", "first"), age=("age", "first"))
    return round(roc_auc_score(spk.label, spk.age), 3)


def best_single_feature(df, feats):
    """The one feature whose per-speaker mean best separates the classes (either way)."""
    spk = df.groupby("speaker_id")[feats + ["label"]].mean()
    scores = {c: roc_auc_score(spk.label, spk[c]) for c in feats}
    name = max(scores, key=lambda c: max(scores[c], 1 - scores[c]))
    return name, round(max(scores[name], 1 - scores[name]), 3)


def speakers_16k_only():
    m = pd.read_csv(MANIFEST_PATH)
    m = m[(m.dataset == "italian") & (m.task == "read_text")]
    top_rate = m.groupby("speaker_id").sample_rate.max()
    return set(top_rate[top_rate == 16000].index)


def main():
    df = load_windows()
    no_young = df[df.cohort != "young_hc"]
    matched = no_young[no_young.speaker_id.isin(speakers_16k_only())]
    out = []
    for subset, part in (("all_speakers", df),
                         ("without_young_controls", no_young),
                         ("matched_no_young_16k_only", matched)):
        feats = [c for c in part.columns if c not in META]
        X = part[feats].to_numpy(dtype=float)
        y, g = part["label"].to_numpy(), part["speaker_id"].to_numpy()
        print(f"{subset}: {len(part)} windows, {len(set(g))} speakers "
              f"({part[part.label == 1].speaker_id.nunique()} PD)")
        res, _ = evaluate_models(X, y, g)
        res.insert(0, "subset", subset)
        res["n_speakers"] = len(set(g))
        res["age_only_auroc"] = age_only_auroc(part)
        res["best_single_feature"], res["best_single_feature_auroc"] = \
            best_single_feature(part, feats)
        out.append(res)
    out = pd.concat(out, ignore_index=True)
    print(out.to_string(index=False))
    (ROOT / "results").mkdir(exist_ok=True)
    out.to_csv(ROOT / "results" / "italian_baseline.csv", index=False)


if __name__ == "__main__":
    main()
