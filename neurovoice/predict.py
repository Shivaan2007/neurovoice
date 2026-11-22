"""predict(path) -> dict. Stub until the real model lands (returns 0.5, 'uncertain')."""
import json

from neurovoice.config import META_PATH, MODEL_PATH, NOTICE

STUB_VERSION = "stub-0"


def predict(path: str) -> dict:
    if not (MODEL_PATH.exists() and META_PATH.exists()):
        return {"probability": 0.5, "band": "uncertain",
                "model_version": STUB_VERSION, "notice": NOTICE}
    import joblib
    from neurovoice.features import extract_features

    meta = json.loads(META_PATH.read_text())
    model = joblib.load(MODEL_PATH)
    feats = extract_features(path)
    x = [[feats[k] for k in meta["feature_names"]]]
    p = float(model.predict_proba(x)[0, 1])
    band = "lower" if p < meta["low"] else "higher" if p > meta["high"] else "uncertain"
    return {"probability": p, "band": band,
            "model_version": meta["model_version"], "notice": NOTICE}
