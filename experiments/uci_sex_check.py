"""UCI: does sex act as a shortcut? (gender is a column in this dataset, coded 1 = male.)

Counts sex per class, then compares subject-grouped Random Forest AUROC:
  * sex alone (how much the label is given away by sex)
  * all features including gender
  * all features without gender (the honest model)
and reports AUROC separately for men and women, with speaker-bootstrap 95% CIs.

Run: python -m experiments.uci_sex_check
"""
import json
import warnings

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from neurovoice.config import ROOT, SEED
from neurovoice.evaluate import speaker_bootstrap_ci
from neurovoice.splits import grouped_splits
from neurovoice.uci import load_uci

warnings.filterwarnings("ignore")


def cv_probs(X, y, g):
    model = RandomForestClassifier(300, random_state=SEED, n_jobs=-1)
    return cross_val_predict(model, X, y, cv=grouped_splits(X, y, g, 5),
                             method="predict_proba")[:, 1]


def main():
    X, y, g = load_uci()
    sex = X["gender"].to_numpy()  # 1 = male
    subj = pd.DataFrame({"id": g, "y": y, "male": sex}).groupby("id").first()
    counts = subj.groupby(["y", "male"]).size().unstack().rename(
        index={0: "healthy", 1: "PD"}, columns={0: "female", 1: "male"})
    share_male = (subj.groupby("y").male.mean()).round(3).to_dict()
    print(counts, "\nshare male:", share_male)

    # Sex alone: logistic regression on gender only, same grouped CV.
    p_sex = cross_val_predict(LogisticRegression(), X[["gender"]], y,
                              cv=grouped_splits(X, y, g, 5), method="predict_proba")[:, 1]
    p_with = cv_probs(X.to_numpy(), y, g)
    Xn = X.drop(columns=["gender"]).to_numpy()
    p_without = cv_probs(Xn, y, g)

    rows = []
    for name, p in (("sex only", p_sex), ("all features incl. gender", p_with),
                    ("all features without gender", p_without)):
        a, lo, hi = speaker_bootstrap_ci(y, p, g)
        rows.append({"model": name, "subgroup": "all", "auroc": round(a, 3),
                     "ci95": f"{lo:.3f}-{hi:.3f}", "subjects": int(len(set(g)))})
    for label, mask in (("men", sex == 1), ("women", sex == 0)):
        yy, pp, gg = y[mask], p_without[mask], g[mask]
        a, lo, hi = speaker_bootstrap_ci(yy, pp, gg)
        rows.append({"model": "all features without gender", "subgroup": label,
                     "auroc": round(a, 3), "ci95": f"{lo:.3f}-{hi:.3f}",
                     "subjects": int(len(set(gg)))})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    res = ROOT / "results"
    res.mkdir(exist_ok=True)
    out.to_csv(res / "uci_sex_check.csv", index=False)
    (res / "uci_sex_counts.json").write_text(json.dumps(
        {"counts": {k: {c: int(v) for c, v in r.items()} for k, r in counts.to_dict("index").items()},
         "share_male_by_class": {("PD" if k == 1 else "healthy"): v for k, v in share_male.items()},
         "n_subjects": int(len(subj))}, indent=2))
    print("sex-only AUROC", out.auroc[0], "| gap vs no-gender", round(out.auroc[1] - out.auroc[2], 3))


if __name__ == "__main__":
    main()
