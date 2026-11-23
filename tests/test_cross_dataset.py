import numpy as np
import pandas as pd

from experiments.cross_dataset import (
    adaptation,
    feature_columns,
    run,
    zero_shot,
    zscore_per_dataset,
)


def _dataset(prefix, offset, n_spk=20, per=6, seed=0):
    rng = np.random.default_rng(seed)
    label = np.repeat(np.arange(n_spk) % 2, per)
    X = rng.normal(size=(n_spk * per, 3)) + label[:, None] * 1.5 + offset
    df = pd.DataFrame(X, columns=["f1", "f2", "f3"])
    df["label"] = label
    df["speaker_id"] = [f"{prefix}{i}" for i in np.repeat(np.arange(n_spk), per)]
    df["window"] = np.tile(np.arange(per), n_spk)
    return df


def test_zscore_removes_a_constant_dataset_offset():
    a, b = _dataset("a", 0.0), _dataset("b", 50.0, seed=1)
    cols = feature_columns(a, b)
    za, zb = zscore_per_dataset(a, cols), zscore_per_dataset(b, cols)
    assert abs(za[cols].mean().mean()) < 1e-9 and abs(zb[cols].mean().mean()) < 1e-9


def test_zero_shot_and_adaptation_report_metrics():
    src, tgt = _dataset("a", 0.0), _dataset("b", 0.0, seed=1)
    cols = feature_columns(src, tgt)
    zs = zero_shot("logreg", src, tgt, cols)
    assert zs["auroc_window"] > 0.8 and zs["target_speakers"] == 20
    ad = adaptation("logreg", src, tgt, cols, draws=2)
    assert ad["auroc_window"] > 0.8 and ad["target_speakers"] == 16  # 20% of 20 speakers held in


def test_normalisation_improves_calibration_on_an_offset_target():
    src, tgt = _dataset("a", 0.0), _dataset("b", 40.0, seed=1)
    out = run({"a": src, "b": tgt}, models=("logreg",))
    z = out[(out.train == "a") & (out.setting == "zero_shot")].set_index("features")
    assert z.loc["per_dataset_z", "brier"] < z.loc["raw", "brier"]  # a constant shift cannot move AUROC
