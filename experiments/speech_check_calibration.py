"""Calibrate and test neurovoice.speech_check on real speech and on non-speech sounds.

Real speech: random 10 s clips from every MDVR-KCL recording (read text and dialogue)
and every Italian read-text recording, PD and healthy. The check must accept all of them;
rejecting a PD speaker's voice would be the worst failure.
Non-speech: synthetic sounds people might upload by mistake (tones, a melody, chords,
noise, near-silence, hum). The check should reject all of them.
Writes results/speech_check.csv (one row per clip) and prints accept rates.
"""
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from neurovoice.config import MANIFEST_PATH, ROOT, SAMPLE_RATE, SEED
from neurovoice.features import load_audio
from neurovoice.speech_check import failed_checks, speech_stats

warnings.filterwarnings("ignore")
CLIP_S = 10
SR = SAMPLE_RATE


def real_clips():
    m = pd.read_csv(MANIFEST_PATH)
    m = m[((m.dataset == "mdvr_kcl") & m.task.isin(["read_text", "spontaneous"]))
          | ((m.dataset == "italian") & (m.task == "read_text"))]
    rng = np.random.default_rng(SEED)
    for _, r in m.iterrows():
        yield (f"{r.dataset}/{r.task}", int(r.label), str(ROOT / r.file),
               int(rng.integers(0, 10_000)))


def one_real(source, label, path, seed):
    y = load_audio(path)
    n = CLIP_S * SR
    start = np.random.default_rng(seed).integers(0, max(1, len(y) - n))
    stats = speech_stats(y[start:start + n])
    return {"source": source, "label": label, "kind": "speech", **stats}


def synthetic():
    t = np.arange(CLIP_S * SR) / SR
    rng = np.random.default_rng(SEED)
    notes = 220 * 2 ** (rng.integers(0, 12, CLIP_S * 2) / 12)
    melody_f = np.repeat(notes, SR // 2)[: t.size]
    sounds = {
        "tone_150hz": np.sin(2 * np.pi * 150 * t),
        "tone_with_vibrato": np.sin(2 * np.pi * 150 * t + 0.5 * np.sin(2 * np.pi * 5 * t) * 30 / 5),
        "tone_pulsing_4hz": np.sin(2 * np.pi * 150 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t)),
        "melody_square_wave": np.sign(np.sin(2 * np.pi * np.cumsum(melody_f) / SR)),
        "chord": sum(np.sin(2 * np.pi * f * t) for f in (220, 277, 330)),
        "white_noise": rng.normal(size=t.size),
        "near_silence": rng.normal(size=t.size) * 1e-3,
        "mains_hum_50hz": np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 100 * t),
    }
    for name, y in sounds.items():
        y = (0.3 * y / np.max(np.abs(y))).astype("float32")
        yield {"source": f"synthetic/{name}", "label": -1, "kind": "non_speech", **speech_stats(y)}


def main():
    rows = Parallel(n_jobs=-1)(delayed(one_real)(*c) for c in real_clips())
    rows += list(synthetic())
    df = pd.DataFrame(rows)
    df["accepted"] = [not failed_checks(r) for r in df.to_dict("records")]
    cols = ["voiced_frac", "pitch_spread", "loudness_variation"]
    print(df[df.kind == "speech"].groupby("source")[cols].describe().T.round(3)
          .loc[(slice(None), ["min", "25%", "50%"]), :])
    print(df[df.kind == "non_speech"][["source"] + cols + ["accepted"]].round(3).to_string())
    print("speech accepted:", df[df.kind == "speech"].groupby(["source", "label"]).accepted.mean())
    print("non-speech rejected:", 1 - df[df.kind == "non_speech"].accepted.mean())
    df.round(4).to_csv(ROOT / "results" / "speech_check.csv", index=False)


if __name__ == "__main__":
    main()
