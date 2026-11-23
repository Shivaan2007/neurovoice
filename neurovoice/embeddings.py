"""wav2vec2-base embeddings: one mean-pooled 768-d vector per 5 s window.

Windows are cut exactly like features.window_features (trim, then full 5 s windows) so
embedding rows line up one-to-one with the hand-built feature rows.
"""
import numpy as np

from neurovoice.config import SAMPLE_RATE
from neurovoice.features import WINDOW_SECONDS

MODEL_NAME = "facebook/wav2vec2-base"
_cache = {}


def _model():
    if "m" not in _cache:
        import torch
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
        _cache["fe"] = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
        _cache["m"] = Wav2Vec2Model.from_pretrained(MODEL_NAME).eval()
        torch.set_grad_enabled(False)
    return _cache["fe"], _cache["m"]


def window_embeddings(y: np.ndarray, seconds: float = WINDOW_SECONDS) -> list[dict[str, float]]:
    """Mean-pooled last hidden state for each full window. Returns [] if no full window."""
    win = int(seconds * SAMPLE_RATE)
    chunks = [y[i * win:(i + 1) * win] for i in range(len(y) // win)]
    if not chunks:
        return []
    fe, model = _model()
    out = []
    for c in chunks:
        inp = fe(c, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        h = model(**inp).last_hidden_state.mean(dim=1).squeeze(0)
        out.append({f"w2v_{i}": float(v) for i, v in enumerate(h)})
    return out
