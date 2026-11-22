"""Load the UCI Parkinson's Disease Classification set (Sakar et al., 2019).

756 recordings of sustained /a/ from 252 subjects (3 per subject), 754 pre-extracted
features, no audio. Columns: `id` (subject), `class` (1 = PD, 0 = healthy), features.
Download from https://archive.ics.uci.edu/dataset/470 and unzip into data/raw/uci/.
"""
import pandas as pd

from neurovoice.config import RAW_DIR

UCI_PATH = RAW_DIR / "uci" / "pd_speech_features.csv"


def load_uci(path=UCI_PATH):
    """Return (X, y, groups): feature frame, labels, subject IDs (one row per recording)."""
    df = pd.read_csv(path)
    if "id" not in df.columns:  # the UCI file has an extra row of feature-group names on top
        df = pd.read_csv(path, header=1)
    missing = {"id", "class"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if df.groupby("id")["class"].nunique().max() > 1:
        raise ValueError("A subject has more than one label.")
    X = df.drop(columns=["id", "class"]).astype(float)
    return X, df["class"].to_numpy(), df["id"].to_numpy()
