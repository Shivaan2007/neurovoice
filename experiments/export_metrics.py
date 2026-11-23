"""Collect the headline numbers from results/*.csv|json into results/metrics.json.

This is the hand-off file: every number the README or model card quotes should come from
here, and every number here comes from a results file, never typed by hand.

Run: python -m experiments.export_metrics
"""
import json
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results"


def rf(df, **where):
    for k, v in where.items():
        df = df[df[k] == v]
    return df[df.model == "random_forest"].iloc[0]


def main():
    seg = rf(pd.read_csv(RES / "segment_check.csv"))
    uci = rf(pd.read_csv(RES / "uci_leakage.csv"))
    cv = rf(pd.read_csv(RES / "train_cv.csv"))
    ea = json.loads((RES / "error_analysis.json").read_text())
    cal = pd.read_csv(RES / "calibration_before_after.csv").set_index("probabilities")
    it = pd.read_csv(RES / "italian_baseline.csv")
    cross = RES / "cross_dataset.csv"

    metrics = {
        "leakage_demo": {
            "mdvr_kcl_5s_windows": {
                "naive_random_auroc": float(seg.auroc_naive_random),
                "speaker_grouped_auroc": float(seg.auroc_speaker_grouped),
                "speaker_grouped_ci95": seg.grouped_ci95,
                "inflation": float(seg.inflation), "source": "results/segment_check.csv"},
            "uci_3_recordings_per_subject": {
                "naive_random_auroc": float(uci.auroc_naive_random),
                "subject_grouped_auroc": float(uci.auroc_subject_grouped),
                "subject_grouped_ci95": uci.grouped_ci95, "inflation": float(uci.inflation), "source": "results/uci_leakage.csv"},
        },
        "unseen_speakers_mdvr_kcl_random_forest": {
            "auroc_window": float(cv.auroc_window), "window_ci95": cv.window_ci95,
            "auroc_speaker": float(cv.auroc_speaker), "brier": float(cv.brier),
            "ece": float(cv.ece), "source": "results/train_cv.csv"},
        "operating_point_at_0.5": {k: ea[k] for k in ("window_at_0.5", "speaker_at_0.5")},
        "confident_errors": {k: ea[k] for k in (
            "confident_wrong_windows", "confident_wrong_speakers",
            "top_speaker_share_of_confident_errors")},
        "higher_band": {"in_sample_precision": ea["bands_as_deployed_in_sample"]["higher_precision"],
                        "crossfit_precision_pooled": ea["bands_crossfit_pooled"]["higher_precision"],
                        "source": "results/threshold_crossfit.csv"},
        "calibration_mdvr_random_forest": cal.to_dict("index"),
        "italian_recording_confound": {
            "age_only_auroc_with_young_controls": float(it[it.subset == "all_speakers"].age_only_auroc.iloc[0]),
            "age_only_auroc_without_young_controls": float(it[it.subset == "without_young_controls"].age_only_auroc.iloc[0]),
            "best_single_feature_auroc_matched": float(it[it.subset == "matched_no_young_16k_only"].best_single_feature_auroc.iloc[0]),
            "model_auroc_matched": float(rf(it, subset="matched_no_young_16k_only").auroc_window),
            "note": "perfect scores reflect recording differences, not PD detection",
            "source": "results/italian_baseline.csv"},
        "unseen_dataset": ("see results/cross_dataset.csv" if cross.exists()
                           else "pending: run experiments/cross_dataset.py with Italian data"),
    }
    (RES / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2)[:1500])


if __name__ == "__main__":
    main()
