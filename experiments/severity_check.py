"""MDVR-KCL: does the model do worse on milder Parkinson's?

MDVR-KCL file names carry clinician ratings (Zenodo record 2867216): IDnn_<hc|pd>_<H&Y>_
<UPDRS II-5>_<UPDRS III-18>. Age and sex are NOT in the release, so Hoehn & Yahr stage is
the only subgroup we can break the audio results down by.

Uses the held-out speaker predictions in results/error_speakers.csv (nested speaker-grouped
CV, Random Forest) and the read-text file names in the local, git-ignored manifest.
Only per-stage counts are written, never file names.

Run: python -m experiments.severity_check
"""
import re

import pandas as pd

from neurovoice.config import MANIFEST_PATH, ROOT

NAME = re.compile(r"ID(\d+)_(hc|pd)_(\d+)_(\d+)_(\d+)", re.IGNORECASE)


def ratings(manifest):
    rows = []
    for _, r in manifest[(manifest.dataset == "mdvr_kcl") & (manifest.task == "read_text")].iterrows():
        m = NAME.search(str(r["file"]))
        if m:
            rows.append({"speaker_id": r["speaker_id"], "hy_stage": int(m.group(3)),
                         "updrs_ii5": int(m.group(4)), "updrs_iii18": int(m.group(5))})
    return pd.DataFrame(rows).drop_duplicates("speaker_id")


def main():
    spk = pd.read_csv(ROOT / "results" / "error_speakers.csv")
    df = spk.merge(ratings(pd.read_csv(MANIFEST_PATH)), on="speaker_id", how="left")
    assert df.hy_stage.notna().all(), "some speakers have no rating in their file name"
    df["called_pd"] = df.mean_p >= 0.5
    pd_rows = df[df.label == 1]
    out = pd_rows.groupby("hy_stage").agg(
        pd_speakers=("speaker_id", "count"), mean_probability=("mean_p", "mean"),
        flagged_at_0_5=("called_pd", "sum")).reset_index()
    out["share_flagged"] = (out.flagged_at_0_5 / out.pd_speakers).round(3)
    out["mean_probability"] = out.mean_probability.round(3)
    out.insert(0, "group", "PD by Hoehn & Yahr stage")
    hc = df[df.label == 0]
    hc_row = {"group": "healthy controls", "hy_stage": "-", "pd_speakers": len(hc),
              "mean_probability": round(float(hc.mean_p.mean()), 3),
              "flagged_at_0_5": int(hc.called_pd.sum()),
              "share_flagged": round(float(hc.called_pd.mean()), 3)}
    out = pd.concat([out, pd.DataFrame([hc_row])], ignore_index=True)
    # Rating notes straight from the Zenodo description.
    notes = {"hc_with_nonzero_updrs": int(((df.label == 0) & (df.updrs_ii5 + df.updrs_iii18 > 0)).sum()),
             "pd_with_zero_updrs": int(((df.label == 1) & (df.updrs_ii5 + df.updrs_iii18 == 0)).sum())}
    out.to_csv(ROOT / "results" / "severity_check.csv", index=False)
    print(out.to_string(index=False))
    print(notes)


if __name__ == "__main__":
    main()
