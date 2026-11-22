"""extract_features(path) -> flat dict with fixed keys (16 kHz mono)."""
import numpy as np

from neurovoice.config import SAMPLE_RATE


def load_audio(path: str) -> np.ndarray:
    import librosa
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    return y


def features_from_signal(y: np.ndarray) -> dict[str, float]:
    import librosa
    sr = SAMPLE_RATE
    out: dict[str, float] = {}
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    for i in range(13):
        out[f"mfcc{i+1}_mean"] = float(mfcc[i].mean())
        out[f"mfcc{i+1}_std"] = float(mfcc[i].std())
        out[f"mfcc{i+1}_delta_mean"] = float(delta[i].mean())
    f0, _, _ = librosa.pyin(y, fmin=65, fmax=400, sr=sr)
    f0 = f0[~np.isnan(f0)]
    out["f0_mean"] = float(f0.mean()) if f0.size else 0.0
    out["f0_std"] = float(f0.std()) if f0.size else 0.0
    out["f0_range"] = float(f0.max() - f0.min()) if f0.size else 0.0
    out["rms_mean"] = float(librosa.feature.rms(y=y).mean())
    out["spectral_centroid_mean"] = float(
        librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    out["zcr_mean"] = float(librosa.feature.zero_crossing_rate(y).mean())
    out.update(_praat_features(y))
    return out


def _praat_features(y: np.ndarray) -> dict[str, float]:
    """Jitter, shimmer, HNR via Praat (parselmouth). Zeros if unvoiced."""
    keys = ["jitter_local", "shimmer_local", "hnr_mean"]
    try:
        import parselmouth
        from parselmouth.praat import call
        snd = parselmouth.Sound(y.astype("float64"), sampling_frequency=SAMPLE_RATE)
        pp = call(snd, "To PointProcess (periodic, cc)", 75, 500)
        jit = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shim = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        hnr = call(snd.to_harmonicity(), "Get mean", 0, 0)
        vals = [jit, shim, hnr]
    except Exception:
        vals = [0.0, 0.0, 0.0]
    return {k: (0.0 if v != v else float(v)) for k, v in zip(keys, vals)}


def extract_features(path: str) -> dict[str, float]:
    return features_from_signal(load_audio(path))


WINDOW_SECONDS = 5


def window_features(y: np.ndarray, seconds: float = WINDOW_SECONDS) -> list[dict[str, float]]:
    """Features for each full window of `seconds` (same cut as the segment check).

    Training and the app both use this, so a 20 s clip in the app is scored the same way
    the model was trained: per 5 s window, then averaged. Returns [] if no full window.
    """
    win = int(seconds * SAMPLE_RATE)
    return [features_from_signal(y[i * win:(i + 1) * win]) for i in range(len(y) // win)]
