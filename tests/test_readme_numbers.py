"""Every number the README or model card quotes must match its results file.

The plan's rule: "check every number in the README matches results/". This turns it into
a test so a stale number fails CI instead of reaching a reader.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
README = (ROOT / "README.md").read_text()
CARD = (ROOT / "docs" / "model_card.md").read_text()


def rf(name):
    df = pd.read_csv(RES / name)
    return df[df.model == "random_forest"].iloc[0]


def has(text, *values):
    missing = [v for v in values if str(v) not in text]
    assert not missing, f"not found in text: {missing}"


def test_segment_leakage_row_matches_results():
    r = rf("segment_check.csv")
    has(README, f"{r.auroc_naive_random:.3f}", f"{r.auroc_speaker_grouped:.3f}",
        r.grouped_ci95.replace(" ", ""), f"{r.inflation:.3f}")


def test_uci_leakage_row_matches_results():
    r = rf("uci_leakage.csv")
    has(README, f"{r.auroc_naive_random:.3f}", f"{r.auroc_subject_grouped:.3f}",
        r.grouped_ci95, f"{r.inflation:.3f}")


def test_demo_model_table_matches_train_cv():
    df = pd.read_csv(RES / "train_cv.csv")
    for _, r in df.iterrows():
        has(README, f"{r.auroc_window:.3f} ({r.window_ci95})",
            f"{r.auroc_speaker:.3f} ({r.speaker_ci95})", f"{r.brier:.3f}")
    r = df[df.model == "random_forest"].iloc[0]
    has(CARD, f"{r.auroc_window:.3f} ({r.window_ci95})",
        f"{r.auroc_speaker:.3f} ({r.speaker_ci95})",
        f"{r.brier:.3f}", f"{r.ece:.3f}")


def test_band_claims_match_error_analysis():
    ea = json.loads((RES / "error_analysis.json").read_text())
    d = ea["bands_as_deployed_in_sample"]
    pooled = ea["bands_crossfit_pooled"]["higher_precision"]
    has(README, f"{round(pooled * 100)}%", f"{d['higher_speakers']} of {ea['n_speakers']} speakers")
    has(CARD, f"{round(pooled * 100)}%", f"{d['higher_speakers']} of {ea['n_speakers']} speakers")
    folds = pd.read_csv(RES / "threshold_crossfit.csv").higher_precision.dropna()
    has(README, f"{round(folds.min() * 100)}% and {round(folds.max() * 100)}%")
    share = d["higher_windows"] / ea["n_windows"]
    has(README, f"about {share * 100:.1f}% of windows")
    has(CARD, f"about {share * 100:.1f}% of held-out windows",
        f"About {(1 - share) * 100:.1f}% of held-out windows")


def test_model_meta_matches_the_cards():
    meta = json.loads((ROOT / "models" / "meta.json").read_text())
    has(CARD, f"{meta['high']}", f"{meta['n_speakers']} English speakers",
        f"{meta['n_windows']:,} windows")
    assert meta["low"] == 0.0, "the card says the model never says 'lower'; update it"


def test_sex_check_matches_results():
    df = pd.read_csv(RES / "uci_sex_check.csv").set_index(["model", "subgroup"])
    sex = df.loc[("sex only", "all")]
    has(README, f"{sex.auroc:.3f}", sex.ci95.replace("-", "-"))
    has(README, f"{df.loc[('all features incl. gender', 'all')].auroc:.3f}",
        f"{df.loc[('all features without gender', 'men')].auroc:.3f}",
        f"{df.loc[('all features without gender', 'women')].auroc:.3f}")


def test_wav2vec2_numbers_match_results():
    r = pd.read_csv(RES / "embeddings_mdvr.csv").set_index("model").loc["logreg"]
    has(README, f"{r.auroc_window:.3f} ({r.window_ci95})",
        f"{r.auroc_speaker:.3f} ({r.speaker_ci95})")


def test_metrics_json_agrees_with_results_files():
    m = json.loads((RES / "metrics.json").read_text())
    r = rf("segment_check.csv")
    assert m["leakage_demo"]["mdvr_kcl_5s_windows"]["inflation"] == float(r.inflation)
    cv = rf("train_cv.csv")
    assert m["unseen_speakers_mdvr_kcl_random_forest"]["auroc_window"] == float(cv.auroc_window)


def test_unseen_dataset_headline_matches_results():
    d = pd.read_csv(RES / "cross_dataset.csv")
    r = d[(d.train == "mdvr_kcl") & (d.test == "italian") & (d.test_subset == "no_young")
          & (d.features == "raw") & (d.model == "random_forest")
          & (d.setting == "zero_shot")].iloc[0]
    has(README, f"{r.auroc_window:.3f} ({r.ci95})", f"{r.auroc_speaker:.3f}")
    has(CARD, f"{r.auroc_window:.3f} (95% CI {r.ci95})", f"{r.auroc_speaker:.3f}")
    meta = json.loads((ROOT / "models" / "meta.json").read_text())
    has(meta["cross_dataset_headline"], f"{r.auroc_window:.3f}", r.ci95)


def test_dataset_classifier_matches_results():
    c = pd.read_csv(RES / "dataset_classifier.csv").set_index("model").loc["random_forest"]
    has(README, f"{c.window_accuracy:.1%}", c.accuracy_ci95, f"{c.chance_majority_class:.1%}")
