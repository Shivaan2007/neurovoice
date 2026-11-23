"""Build data/manifest.csv. One small parser per dataset (MDVR-KCL here, Italian in italian.py)."""
import hashlib
import re
import wave

import pandas as pd

from neurovoice.config import MANIFEST_COLUMNS, MANIFEST_PATH, RAW_DIR


def _sha1(path):
    return hashlib.sha1(path.read_bytes()).hexdigest()


def parse_mdvr_kcl(root=None):
    """MDVR-KCL: <Task>/<PD|HC>/ID##_<pd|hc>_*.wav, smartphone recordings, English.

    Age and gender are not in the file names, so they are left empty here.
    """
    root = root or RAW_DIR / "mdvr_kcl"
    rows = []
    for path in sorted(root.rglob("*.wav")):
        task_dir, group = path.parent.parent.name, path.parent.name
        m = re.match(r"ID(\d+)", path.name)
        if not m or group not in ("PD", "HC"):
            continue
        with wave.open(str(path), "rb") as w:
            rate, frames = w.getframerate(), w.getnframes()
        rows.append({
            # as_posix: forward slashes, so a manifest built on Windows also works on Linux
            "file": path.relative_to(root.parent.parent.parent).as_posix(),
            "speaker_id": f"mdvr_kcl_{int(m.group(1)):02d}",
            "label": int(group == "PD"),
            "dataset": "mdvr_kcl",
            "language": "en",
            "task": "read_text" if task_dir == "ReadText" else "spontaneous",
            "sample_rate": rate,
            "duration_s": round(frames / rate, 2),
            "age": None,
            "gender": None,
            "sha1": _sha1(path),
        })
    return rows


def check_one_label_per_speaker(df):
    """A speaker with two labels means an ID clash, which would corrupt grouped splits."""
    n_labels = df.groupby("speaker_id")["label"].nunique()
    bad = sorted(n_labels[n_labels > 1].index)
    if bad:
        raise ValueError(f"Speakers with more than one label: {bad}")


def build_manifest():
    from neurovoice.italian import parse_italian

    rows = parse_mdvr_kcl() + parse_italian()  # each parser skips a dataset not downloaded
    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    dup = df.sha1.duplicated()
    if dup.any():  # e.g. one Italian speaker's B1 and B2 are the same audio file
        # print speaker IDs, not paths: Italian paths contain participants' names
        print(f"dropping {int(dup.sum())} duplicate audio file(s) from:",
              sorted(set(df.speaker_id[dup])))
        df = df[~dup].reset_index(drop=True)
    check_one_label_per_speaker(df)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST_PATH, index=False)
    return df


if __name__ == "__main__":
    df = build_manifest()
    print(df.groupby(["dataset", "task", "label"]).agg(
        recordings=("file", "count"), speakers=("speaker_id", "nunique"),
        median_s=("duration_s", "median")))
    print("duplicate sha1:", int(df.sha1.duplicated().sum()))
    print("sample rates:", sorted(df.sample_rate.unique()))
