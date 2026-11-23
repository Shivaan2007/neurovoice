import wave

import numpy as np
import pandas as pd

from neurovoice.config import SAMPLE_RATE
from neurovoice.neurovoz import (
    parse_name,
    parse_neurovoz,
    read_demographics,
    task_of,
    windows_from_manifest,
)


def _wav(path, seconds=2.0, rate=SAMPLE_RATE):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(rate * seconds)) / rate
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.sin(2 * np.pi * 150 * t) * 8000).astype("<i2").tobytes())


def test_parse_name_matches_the_paper_examples():
    assert parse_name("HC_A2_0034.wav") == ("HC", "A", "2", "0034")
    assert parse_name("PD_GANGA_0066.wav") == ("PD", "GANGA", None, "0066")
    assert parse_name("PD_FREE_0001.wav") == ("PD", "FREE", None, "0001")
    assert parse_name("notes.wav") is None
    assert parse_name("XX_A1_0034.wav") is None


def test_task_mapping():
    assert [task_of(m) for m in ("A", "U", "FREE", "PATAKA", "GANGA", "ABLANDADA")] == [
        "vowel", "vowel", "monologue", "ddk", "read_text", "read_text"]


def test_parse_neurovoz_builds_manifest_rows(tmp_path):
    root = tmp_path / "neurovoz"
    for name in ("HC_A1_0001.wav", "HC_GANGA_0001.wav", "PD_FREE_0002.wav",
                 "PD_PATAKA_0002.wav", "PD_ABLANDADA_0002.wav"):
        _wav(root / "audios" / name)
    _wav(root / "audios" / "notes.wav")  # does not match the pattern: skipped
    rows = pd.DataFrame(parse_neurovoz(root))
    assert len(rows) == 5
    assert set(rows.dataset) == {"neurovoz"} and set(rows.language) == {"es"}
    assert set(rows[rows.speaker_id == "neurovoz_0002"].label) == {1}
    assert set(rows[rows.speaker_id == "neurovoz_0001"].label) == {0}
    assert sorted(rows.task) == ["ddk", "monologue", "read_text", "read_text", "vowel"]
    assert (rows.sample_rate == SAMPLE_RATE).all()


def test_parse_neurovoz_returns_nothing_when_not_downloaded(tmp_path):
    assert parse_neurovoz(tmp_path / "missing") == []


def test_demographics_are_read_when_a_metadata_csv_exists(tmp_path):
    root = tmp_path / "neurovoz"
    (root / "metadata").mkdir(parents=True)
    pd.DataFrame({"ID": [1, 2], "age": [64, 71], "sex": ["male", "female"]}).to_csv(
        root / "metadata" / "meta.csv", index=False)
    assert read_demographics(root) == {"0001": (64.0, "M"), "0002": (71.0, "F")}


def test_sentences_are_joined_before_windowing(tmp_path):
    # two 3 s sentences from one speaker -> one 6 s stream -> exactly one full 5 s window
    for name in ("HC_GANGA_0001.wav", "HC_ABLANDADA_0001.wav"):
        _wav(tmp_path / "audios" / name, seconds=3.0)
    manifest = pd.DataFrame({
        "file": ["audios/HC_ABLANDADA_0001.wav", "audios/HC_GANGA_0001.wav"],
        "speaker_id": "neurovoz_0001", "label": 0, "dataset": "neurovoz",
        "task": "read_text"})
    df = windows_from_manifest(manifest, base=tmp_path)
    assert len(df) == 1 and df.speaker_id.iloc[0] == "neurovoz_0001"
