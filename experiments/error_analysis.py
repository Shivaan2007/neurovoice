"""Error analysis on the headline model (MDVR-KCL read text, Random Forest).

Everything here uses held-out predictions (nested speaker-grouped CV from train.py), so
no number comes from a speaker the model trained on.

Outputs (results/):
  error_speakers.csv        one row per speaker: label, mean probability, wrong or not
  confident_errors.csv      windows wrong with probability >0.8 or <0.2, grouped by speaker
  threshold_crossfit.csv    precision of the app's bands when the cut is chosen on OTHER speakers
  roc_calibration.png       ROC curves (window and speaker level) + reliability diagram
  shap_summary.png          SHAP summary for the final Random Forest
  error_analysis.json       headline numbers with speaker-bootstrap 95% CIs

Run: python -m experiments.error_analysis
"""
import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from neurovoice.config import ROOT, SEED  # noqa: E402
from neurovoice.evaluate import speaker_bootstrap_ci  # noqa: E402
from neurovoice.train import (  # noqa: E402
    META_COLS,
    TARGET_PRECISION,
    crossfit_thresholds,
    load_windows,
    make_models,
    oof_probabilities,
    speaker_level,
)

warnings.filterwarnings("ignore")
RES = ROOT / "results"


def sens_spec(y, pred):
    tp, fn = int(((y == 1) & (pred == 1)).sum()), int(((y == 1) & (pred == 0)).sum())
    tn, fp = int(((y == 0) & (pred == 0)).sum()), int(((y == 0) & (pred == 1)).sum())
    return tp / max(tp + fn, 1), tn / max(tn + fp, 1)


def boot_ci(y, pred, groups, which, n_boot=1000):
    rng = np.random.default_rng(SEED)
    spk = np.unique(groups)
    idx = {s: np.where(groups == s)[0] for s in spk}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(spk, len(spk), replace=True)
        i = np.concatenate([idx[s] for s in pick])
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(sens_spec(y[i], pred[i])[which])
    return [round(float(v), 3) for v in np.percentile(vals, [2.5, 97.5])]


def main():
    RES.mkdir(exist_ok=True)
    df = load_windows()
    names = [c for c in df.columns if c not in META_COLS]
    X, y = df[names].to_numpy(float), df["label"].to_numpy()
    g = df["speaker_id"].to_numpy()
    meta = json.loads((ROOT / "models" / "meta.json").read_text())
    low, high = meta["low"], meta["high"]

    p = oof_probabilities("random_forest", X, y, g)
    point, lo, hi = speaker_bootstrap_ci(y, p, g)
    sy, sp = speaker_level(y, p, g)
    out = {"auroc_window": [round(point, 3), round(lo, 3), round(hi, 3)],
           "auroc_speaker": round(float(roc_auc_score(sy, sp)), 3),
           "n_speakers": int(len(sy)), "n_windows": int(len(y))}

    # Sensitivity / specificity: window level at 0.5, with speaker-bootstrap CIs.
    pred = (p >= 0.5).astype(int)
    se, sp_ = sens_spec(y, pred)
    out["window_at_0.5"] = {"sensitivity": round(se, 3), "sens_ci95": boot_ci(y, pred, g, 0),
                            "specificity": round(sp_, 3), "spec_ci95": boot_ci(y, pred, g, 1)}
    spred = (sp >= 0.5).astype(int)
    se, sp_ = sens_spec(sy, spred)
    out["speaker_at_0.5"] = {"sensitivity": round(se, 3), "specificity": round(sp_, 3),
                             "n_pd": int(sy.sum()), "n_healthy": int((sy == 0).sum())}

    # Per-speaker table.
    spk = pd.DataFrame({"speaker_id": g, "label": y, "p": p}).groupby("speaker_id").agg(
        label=("label", "first"), mean_p=("p", "mean"), windows=("p", "size"),
        share_windows_correct=("p", lambda s: 0.0))
    ok = pd.DataFrame({"speaker_id": g, "hit": (pred == y)}).groupby("speaker_id").hit.mean()
    spk["share_windows_correct"] = ok.round(3)
    spk["speaker_wrong"] = ((spk.mean_p >= 0.5).astype(int) != spk.label)
    spk.round(3).to_csv(RES / "error_speakers.csv")
    out["speakers_wrong"] = int(spk.speaker_wrong.sum())

    # Confidently wrong windows.
    bad = (p > 0.8) & (y == 0) | (p < 0.2) & (y == 1)
    ce = pd.DataFrame({"speaker_id": g[bad], "label": y[bad], "p": p[bad]})
    ce_sum = ce.groupby(["speaker_id", "label"]).agg(
        windows=("p", "size"), mean_p=("p", "mean")).reset_index().sort_values(
        "windows", ascending=False)
    ce_sum.round(3).to_csv(RES / "confident_errors.csv", index=False)
    out["confident_wrong_windows"] = int(bad.sum())
    out["confident_wrong_speakers"] = int(ce_sum.speaker_id.nunique())
    out["top_speaker_share_of_confident_errors"] = (
        round(float(ce_sum.windows.iloc[0] / max(bad.sum(), 1)), 3) if len(ce_sum) else 0.0)

    # Do the app's bands hold when the cuts are chosen on other speakers?
    cf, cf_higher, cf_lower = crossfit_thresholds(y, p, g)
    cf.to_csv(RES / "threshold_crossfit.csv", index=False)
    up, dn = p > high, p < low
    out["bands_as_deployed_in_sample"] = {
        "low": low, "high": high,
        "higher_precision": round(float((y[up] == 1).mean()), 3) if up.any() else None,
        "higher_windows": int(up.sum()), "higher_speakers": int(len(set(g[up]))),
        "lower_windows": int(dn.sum()), "lower_speakers": int(len(set(g[dn])))}
    out["target_precision"] = TARGET_PRECISION
    out["bands_crossfit_pooled"] = {"higher_precision": cf_higher, "lower_precision": cf_lower}

    # Figure: ROC (window + speaker) and reliability diagram.
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for lab, yy, pp in (("window", y, p), ("speaker", sy, sp)):
        fpr, tpr, _ = roc_curve(yy, pp)
        ax[0].plot(fpr, tpr, label=f"{lab} AUROC {roc_auc_score(yy, pp):.2f}")
    ax[0].plot([0, 1], [0, 1], "k:", lw=1)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate",
              title="Held-out ROC (speaker-grouped CV)")
    ax[0].legend()
    bins = np.minimum((p * 10).astype(int), 9)
    xs = [p[bins == b].mean() for b in range(10) if (bins == b).sum() >= 10]
    ys = [y[bins == b].mean() for b in range(10) if (bins == b).sum() >= 10]
    ax[1].plot([0, 1], [0, 1], "k:", lw=1)
    ax[1].plot(xs, ys, "o-")
    ax[1].set(xlabel="Mean predicted probability", ylabel="Fraction PD",
              title="Reliability (bins with 10+ windows)")
    fig.tight_layout()
    fig.savefig(RES / "roc_calibration.png", dpi=150)

    # SHAP on the final Random Forest (fit on all speakers; explains, does not evaluate).
    import shap
    rf = make_models()["random_forest"].fit(X, y)
    sv = shap.TreeExplainer(rf).shap_values(X)
    sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv)
    plt.figure()
    shap.summary_plot(sv, X, feature_names=names, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(RES / "shap_summary.png", dpi=150)
    imp = pd.Series(np.abs(sv).mean(0), index=names).sort_values(ascending=False)
    out["top_shap_features"] = [f"{k} ({v:.3f})" for k, v in imp.head(8).items()]

    (RES / "error_analysis.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(cf.to_string(index=False))


if __name__ == "__main__":
    main()
