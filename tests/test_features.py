import numpy as np
from neurovoice.config import SAMPLE_RATE
from neurovoice.features import features_from_signal


def test_fixed_keys_no_nans():
    t = np.arange(2 * SAMPLE_RATE) / SAMPLE_RATE
    feats = features_from_signal(np.sin(2 * np.pi * 150 * t).astype("float32"))
    assert "jitter_local" in feats and "f0_mean" in feats
    assert all(np.isfinite(v) for v in feats.values())
