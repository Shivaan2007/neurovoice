"""wav2vec2 embeddings + logistic regression vs hand-built features, MDVR-KCL read text.

Same 5 s windows, same speaker-grouped nested-calibrated CV as train.py, so the numbers
are directly comparable to results/train_cv.csv. Embeddings are cached in data/cache/.
Run: python -m experiments.embeddings_mdvr
"""
import warnings

import pandas as pd

from neurovoice.config import CACHE_DIR, MANIFEST_PATH, ROOT
from neurovoice.embeddings import window_embeddings
from neurovoice.features import load_audio
from neurovoice.train import evaluate_models

warnings.filterwarnings("ignore")
CACHE = CACHE_DIR / "mdvr_read_text_5s_wav2vec2.csv"


def load_embeddings():
    if CACHE.exists():
        return pd.read_csv(CACHE)
    m = pd.read_csv(MANIFEST_PATH)
    m = m[(m.dataset == "mdvr_kcl") & (m.task == "read_text")]
    rows = []
    for _, r in m.iterrows():
        for i, e in enumerate(window_embeddings(load_audio(str(ROOT / r["file"])))):
            rows.append({**e, "speaker_id": r["speaker_id"], "label": r["label"], "window": i})
        print(f"embedded {r['speaker_id']} ({len(rows)} windows)", flush=True)
    df = pd.DataFrame(rows)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, index=False)
    return df


def main():
    df = load_embeddings()
    cols = [c for c in df.columns if c.startswith("w2v_")]
    X, y, g = df[cols].to_numpy(float), df["label"].to_numpy(), df["speaker_id"].to_numpy()
    print(f"{len(df)} windows, {df.speaker_id.nunique()} speakers, {len(cols)} dims")
    res, _ = evaluate_models(X, y, g, names=["logreg", "random_forest"])
    res.insert(0, "features", "wav2vec2-base (mean-pooled)")
    print(res.to_string(index=False))
    res.to_csv(ROOT / "results" / "embeddings_mdvr.csv", index=False)


if __name__ == "__main__":
    main()
