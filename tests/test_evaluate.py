import numpy as np
from sklearn.ensemble import RandomForestClassifier

from neurovoice.evaluate import (
    calibration_report,
    expected_calibration_error,
    leakage_gap,
    speaker_bootstrap_ci,
)


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(30), 3)
    y = np.repeat(rng.integers(0, 2, 30), 3)
    p = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 90), 0, 1)
    point, lo, hi = speaker_bootstrap_ci(y, p, groups, n_boot=200)
    assert lo <= point <= hi


def test_ece_zero_when_perfectly_calibrated():
    y = np.array([0, 0, 1, 1])
    assert expected_calibration_error(y, np.array([0.0, 0.0, 1.0, 1.0])) == 0.0
    assert set(calibration_report(y, np.array([0.1, 0.2, 0.8, 0.9]))) == {"brier", "ece"}


def test_leakage_gap_positive_when_speakers_carry_identity():
    # Labels are constant per speaker and features are speaker fingerprints,
    # so a naive split leaks identity and should score higher than a grouped one.
    rng = np.random.default_rng(1)
    n_spk, per = 30, 6
    groups = np.repeat(np.arange(n_spk), per)
    y = np.repeat(rng.integers(0, 2, n_spk), per)
    fingerprint = rng.normal(size=(n_spk, 5))
    X = np.repeat(fingerprint, per, axis=0) + rng.normal(0, 0.05, (n_spk * per, 5))
    res = leakage_gap(RandomForestClassifier(50, random_state=0), X, y, groups)
    assert res["naive"] > res["grouped"]
