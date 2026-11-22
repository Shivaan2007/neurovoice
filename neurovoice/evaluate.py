"""Evaluation helpers: speaker-level bootstrap CIs, calibration, leakage demo."""
import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import cross_val_predict

from neurovoice.config import SEED
from neurovoice.splits import grouped_splits, naive_splits


def speaker_bootstrap_ci(y, p, groups, metric=roc_auc_score, n_boot=1000, seed=SEED):
    """95% CI from resampling SPEAKERS (not recordings) with replacement."""
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    rng = np.random.default_rng(seed)
    speakers = np.unique(groups)
    idx_by_spk = {s: np.where(groups == s)[0] for s in speakers}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(speakers, size=len(speakers), replace=True)
        idx = np.concatenate([idx_by_spk[s] for s in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(metric(y[idx], p[idx]))
    point = metric(y, p)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def expected_calibration_error(y, p, n_bins=10):
    y, p = np.asarray(y), np.asarray(p)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def calibration_report(y, p):
    return {"brier": float(brier_score_loss(y, p)),
            "ece": expected_calibration_error(y, p)}


def leakage_gap(model, X, y, groups, n_splits=5):
    """Naive (speaker-leaking) vs speaker-grouped AUROC. The gap is the leakage demo."""
    out = {}
    for name, splits in (("naive", naive_splits(X, y, n_splits)),
                       ("grouped", grouped_splits(X, y, groups, n_splits))):
        p = cross_val_predict(model, X, y, cv=splits, method="predict_proba")[:, 1]
        out[name] = float(roc_auc_score(y, p))
    out["gap"] = out["naive"] - out["grouped"]
    return out
