"""predict(path) -> dict. Stub until a trained model exists (returns 0.5, 'uncertain').

With a model, the clip is cut into the same 5 s windows used in training, each window is
scored, and the probabilities are averaged. Bands come from thresholds in meta.json that
train.py chose on held-out predictions.
"""
import json

import numpy as np

from neurovoice.config import META_PATH, MODEL_PATH, NOTICE

STUB_VERSION = "stub-0"


def band_for(p: float, low: float, high: float) -> str:
    return "lower" if p < low else "higher" if p > high else "uncertain"


def predict(path: str, model_path=None, meta_path=None) -> dict:
    model_path, meta_path = model_path or MODEL_PATH, meta_path or META_PATH
    if not (model_path.exists() and meta_path.exists()):
        return {"probability": 0.5, "band": "uncertain",
                "model_version": STUB_VERSION, "notice": NOTICE}
    import joblib

    from neurovoice.features import features_from_signal, load_audio, window_features

    meta = json.loads(meta_path.read_text())
    model = joblib.load(model_path)
    y = load_audio(path)
    feats = window_features(y, meta.get("window_seconds", 5)) or [features_from_signal(y)]
    x = np.array([[f[k] for k in meta["feature_names"]] for f in feats])
    p = float(model.predict_proba(x)[:, 1].mean())
    return {"probability": p, "band": band_for(p, meta["low"], meta["high"]),
            "model_version": meta["model_version"], "notice": NOTICE}
