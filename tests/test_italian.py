import wave

import numpy as np
import pandas as pd

from neurovoice.italian import birth_year, match_demographics, parse_italian, task_of


def test_task_codes():
    assert task_of("B1ABCDEF55F170320171104.wav") == "read_text"
    assert task_of("b2abcdef55F1606161659.wav") == "read_text"
    assert task_of("VA1ABCDEF55F170320171104.wav") == "vowel"
    assert task_of("D2ABCDEF55F170320171104.wav") == "syllables"
    assert task_of("FB1ABCDEF55F170320171104.wav") == "phrases"
    assert task_of("notes.wav") is None


def test_birth_year_both_file_name_styles():
    assert birth_year("B1VSIPTIOZ46M240120171920.wav") == 46  # 2017 style
    assert birth_year("B1vsiptioz46M1606161659.wav") == 46    # 2016 style


DEMO = [
    {"name": ["vito"], "surname": ["s"], "age": 71, "sex": "M"},
    {"name": ["vito"], "surname": ["s"], "age": 70, "sex": "M"},
    {"name": ["mario"], "surname": ["m"], "age": 72, "sex": "M"},  # sheet typo for Mario B
    {"name": ["antonio"], "surname": ["p"], "age": 75, "sex": "M"},
    {"name": ["antonella"], "surname": ["p"], "age": 61, "sex": "F"},
]


def test_match_full_key_with_birth_year_tie_break():
    assert match_demographics("Vito S", DEMO, yy=46) == (71, "M")
    assert match_demographics("Vito S", DEMO, yy=47) == (70, "M")


def test_match_unique_first_name_and_initials_with_sex():
    assert match_demographics("Mario B", DEMO) == (72, "M")
    # surname spelled out in the folder, only initials in the sheet; sex from file code
    assert match_demographics("PORCELLI A", DEMO, sex="F") == (61, "F")
    assert match_demographics("Nobody X", DEMO) == (None, None)


def _wav(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.random.default_rng(len(str(path))).normal(size=1600) * 1000)
                      .astype("<i2").tobytes())


def test_parse_italian_ids(tmp_path):
    root = tmp_path / "data" / "raw" / "italian"
    pd_dir = root / "28 People with Parkinson's disease"
    _wav(pd_dir / "6-10" / "Vito S" / "B1vsiptioz46M1606161659.wav")
    _wav(pd_dir / "11-16" / "Vito S" / "B1VSIPTIOZ46M240120171920.wav")  # same person
    _wav(pd_dir / "17-28" / "Vito S" / "B1VSIOTLOP47M100220171328.wav")  # different person
    yhc = root / "15 Young Healthy Control"
    _wav(yhc / "Luca S" / "B1LBULCAAS94M100120171015.wav")
    _wav(yhc / "Sara T" / "B1LBULCAAS94M100120171016.wav")  # same copied code, other person

    df = pd.DataFrame(parse_italian(root))
    ids = {f.split("/")[-3]: s for f, s in zip(df.file, df.speaker_id) if "Parkinson" in f}
    assert ids["6-10"] == ids["11-16"] != ids["17-28"]  # repeat session merges
    assert df[df.cohort == "young_hc"].speaker_id.nunique() == 2
    assert set(df.label[df.cohort == "pd"]) == {1}
    assert not df.speaker_id.str.contains("vito|luca|sara", case=False).any()
