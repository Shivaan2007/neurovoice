import json
import wave

import numpy as np
import pandas as pd

from neurovoice.config import SAMPLE_RATE
from neurovoice.features import features_from_signal
from neurovoice.predict import band_for, predict
from neurovoice.train import pick_thresholds, speaker_level, train


def test_thresholds_on_well_separated_probabilities():
    y = np.array([0] * 50 + [1] * 50)
    p = np.concatenate([np.linspace(0.05, 0.4, 50), np.linspace(0.6, 0.95, 50)])
    low, high = pick_thresholds(y, p)
    assert 0.4 < low <= high < 0.6


def test_thresholds_stay_closed_when_model_is_useless():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    low, high = pick_thresholds(y, rng.uniform(size=200), target=0.95)
    assert (low, high) == (0.0, 1.0)  # everything is "uncertain"


def test_band_for():
    assert [band_for(p, 0.3, 0.7) for p in (0.1, 0.5, 0.9)] == ["lower", "uncertain", "higher"]


def test_speaker_level_one_row_per_speaker():
    sy, sp = speaker_level([1, 1, 0], [0.2, 0.6, 0.1], ["a", "a", "b"])
    assert list(sy) == [1, 0] and list(sp) == [0.4, 0.1]


def _sine_wav(path, seconds):
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((np.sin(2 * np.pi * 150 * t) * 8000).astype("<i2").tobytes())


def test_train_saves_model_that_predict_uses(tmp_path):
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    names = list(features_from_signal(np.sin(2 * np.pi * 150 * t).astype("float32")))
    rng = np.random.default_rng(0)
    n_spk, per = 24, 8
    label = np.repeat(np.arange(n_spk) % 2, per)
    df = pd.DataFrame(rng.normal(size=(n_spk * per, len(names))) + label[:, None],
                      columns=names)
    df["speaker_id"], df["label"] = np.repeat(np.arange(n_spk), per), label
    df["window"] = np.tile(np.arange(per), n_spk)

    model_path, meta_path = tmp_path / "model.joblib", tmp_path / "meta.json"
    results, meta = train(df, model_path, meta_path, names=["logreg"])
    assert results.loc[0, "auroc_speaker"] > 0.9
    assert json.loads(meta_path.read_text())["feature_names"] == names
    assert 0.0 <= meta["low"] <= meta["high"] <= 1.0

    wav = tmp_path / "clip.wav"
    _sine_wav(wav, 11)  # two full 5 s windows
    out = predict(str(wav), model_path, meta_path)
    assert 0.0 <= out["probability"] <= 1.0
    assert out["band"] in {"lower", "uncertain", "higher"}
    assert out["model_version"] == "mdvr-kcl-logreg-v1"
