"""Reader for NeuroVoz (Castilian Spanish, Mendes-Laureano et al., Scientific Data 2024).

STATUS: written from the layout the paper documents; NOT yet run on the real files (access
is by request under a Data Usage Agreement). Check it against the real folders first and
fix whatever differs. The agreement forbids copying or sharing the audio and any use of it
(or things derived from it) commercially, so nothing from here is committed or deployed.

Documented layout (paper, "Data Records"): five folders "audio features", "audios", "grbas",
"transcriptions", "metadata". Audio files are named
    <Condition>_<AudioMaterial>[<repetition>]_<IDPatient>.wav
e.g. HC_A2_0034.wav (second /a/ of healthy patient 0034), PD_GANGA_0066.wav (listen-and-
repeat sentence GANGA, PD patient 0066). AudioMaterial is A/E/I/O/U (sustained vowels,
repetition 1-3), a keyword (listen-and-repeat sentence), FREE (monologue) or PATAKA (DDK).

Task mapping (CLAUDE.md): sentence keywords -> "read_text" (they are listen-and-repeat, not
read aloud, and we say so), vowels -> "vowel". Others keep their own name.

Sentences are only a few seconds long, shorter than our 5 s window, so load_windows joins a
speaker's sentences into one stream before cutting windows. Windows can therefore span a
sentence boundary.
"""
import hashlib
import re
import wave

import numpy as np
import pandas as pd

from neurovoice.config import CACHE_DIR, MANIFEST_PATH, RAW_DIR, ROOT

NAME_RE = re.compile(r"^(HC|PD)_([A-Za-z]+?)(\d)?_(\d{4})\.wav$", re.IGNORECASE)
VOWELS = {"A", "E", "I", "O", "U"}
MALE = {"m", "male", "man", "h", "hombre", "masculino"}
FEMALE = {"f", "female", "woman", "w", "mujer", "femenino"}
WINDOW_CACHE = CACHE_DIR / "neurovoz_read_text_5s_windows.csv"


def parse_name(filename):
    """Return (condition, material, repetition, patient_id) or None if the name doesn't fit."""
    m = NAME_RE.match(filename)
    if not m:
        return None
    return m.group(1).upper(), m.group(2).upper(), m.group(3), m.group(4)


def task_of(material):
    if material in VOWELS:
        return "vowel"
    if material == "FREE":
        return "monologue"
    if material == "PATAKA":
        return "ddk"
    return "read_text"  # a Table 3 keyword: listen-and-repeat sentence


def read_demographics(root):
    """Best-effort {patient_id: (age, gender)} from a metadata .csv; {} if none is found.

    The paper does not give the metadata columns, so this looks for an id column plus age
    and sex columns by name. Confirm against the real file.
    """
    for path in sorted(root.rglob("*.csv")):
        if "metadata" not in {p.lower() for p in path.parts}:
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        cols = {str(c).strip().lower(): c for c in df.columns}
        id_col = next((cols[c] for c in cols if c in ("id", "patient", "id_patient", "idpatient",
                                                      "subject", "participant")), None)
        age_col = next((cols[c] for c in cols if c in ("age", "edad")), None)
        sex_col = next((cols[c] for c in cols if c in ("sex", "gender", "sexo")), None)
        if id_col is None or (age_col is None and sex_col is None):
            continue
        out = {}
        for _, r in df.iterrows():
            pid = re.sub(r"\D", "", str(r[id_col])).zfill(4)
            sex = str(r[sex_col]).strip().lower() if sex_col else ""
            out[pid] = (
                float(r[age_col]) if age_col and pd.notna(r[age_col]) else None,
                "M" if sex in MALE else "F" if sex in FEMALE else None)
        if out:
            return out
    return {}


def parse_neurovoz(root=None):
    """Manifest rows for NeuroVoz; [] if it has not been downloaded."""
    root = root or RAW_DIR / "neurovoz"
    if not root.exists():
        return []
    demo = read_demographics(root)
    rows, skipped = [], 0
    for path in sorted(root.rglob("*.wav")):
        parsed = parse_name(path.name)
        if parsed is None:
            skipped += 1
            continue
        cond, material, _rep, pid = parsed
        try:
            with wave.open(str(path), "rb") as w:
                rate, frames = w.getframerate(), w.getnframes()
        except (wave.Error, EOFError):
            skipped += 1
            continue
        age, gender = demo.get(pid, (None, None))
        rows.append({
            "file": str(path.relative_to(ROOT)) if ROOT in path.parents else str(path),
            "speaker_id": f"neurovoz_{pid}",
            "label": int(cond == "PD"),
            "dataset": "neurovoz",
            "language": "es",
            "task": task_of(material),
            "sample_rate": rate,
            "duration_s": round(frames / rate, 2),
            "age": age,
            "gender": gender,
            "sha1": hashlib.sha1(path.read_bytes()).hexdigest(),
        })
    if skipped:
        print(f"neurovoz: skipped {skipped} .wav file(s) whose names did not match the "
              "documented pattern; check the real layout")
    return rows


def windows_from_manifest(manifest, base=ROOT):
    """5 s windows of features per speaker, joining that speaker's sentences into one stream."""
    from joblib import Parallel, delayed

    from neurovoice.features import load_audio, window_features

    m = manifest[(manifest.dataset == "neurovoz") & (manifest.task == "read_text")]

    def one(speaker, g):
        parts = [load_audio(str(base / f)) for f in sorted(g.file)]
        y = np.concatenate(parts) if parts else np.zeros(0, dtype="float32")
        return [{**f, "speaker_id": speaker, "label": int(g.label.iloc[0]), "window": i}
                for i, f in enumerate(window_features(y))]

    res = Parallel(n_jobs=-1)(delayed(one)(s, g) for s, g in m.groupby("speaker_id"))
    return pd.DataFrame([w for r in res for w in r])


def load_windows():
    """Cached window features for NeuroVoz read-text; raises if there is no data."""
    if WINDOW_CACHE.exists() and WINDOW_CACHE.stat().st_size > 1:
        return pd.read_csv(WINDOW_CACHE)
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("data/manifest.csv missing; run python -m neurovoice.manifest")
    df = windows_from_manifest(pd.read_csv(MANIFEST_PATH))
    if df.empty:  # never cache an empty result
        raise FileNotFoundError("no NeuroVoz read-text windows; is data/raw/neurovoz filled?")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(WINDOW_CACHE, index=False)
    return df
