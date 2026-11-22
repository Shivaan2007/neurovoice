import numpy as np
import pandas as pd
import pytest

from experiments.uci_leakage import run, subject_level_auroc
from neurovoice.uci import load_uci


def _write(path, header_row=False, bad_label=False):
    df = pd.DataFrame({"id": [0, 0, 1, 1], "gender": [1, 1, 0, 0],
                       "f1": [0.1, 0.2, 0.3, 0.4], "class": [1, 1, 0, 0]})
    if bad_label:
        df.loc[1, "class"] = 0
    text = df.to_csv(index=False)
    if header_row:  # the real UCI file has a row of feature-group names on top
        text = ",Baseline Features,,\n" + text
    path.write_text(text)
    return path


@pytest.mark.parametrize("header_row", [False, True])
def test_load_uci_both_header_formats(tmp_path, header_row):
    X, y, g = load_uci(_write(tmp_path / "u.csv", header_row))
    assert list(X.columns) == ["gender", "f1"]
    assert list(y) == [1, 1, 0, 0] and list(g) == [0, 0, 1, 1]


def test_load_uci_rejects_subject_with_two_labels(tmp_path):
    with pytest.raises(ValueError, match="more than one label"):
        load_uci(_write(tmp_path / "u.csv", bad_label=True))


def test_subject_level_auroc_averages_per_subject():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.9, 0.1, 0.4, 0.4])  # subject 0 mean 0.5 > subject 1 mean 0.4
    assert subject_level_auroc(y, p, np.array([0, 0, 1, 1])) == 1.0


def test_run_shows_leakage_on_fingerprint_data():
    rng = np.random.default_rng(0)
    n_subj, per = 40, 3
    groups = np.repeat(np.arange(n_subj), per)
    y = np.repeat(rng.integers(0, 2, n_subj), per)
    X = np.repeat(rng.normal(size=(n_subj, 5)), per, axis=0) + rng.normal(0, 0.05, (120, 5))
    out = run(X, y, groups)
    assert set(out.model) == {"logreg", "svm_rbf", "random_forest"}
    rf = out.set_index("model").loc["random_forest"]
    assert rf.auroc_naive_random > rf.auroc_subject_grouped
