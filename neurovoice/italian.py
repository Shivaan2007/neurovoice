"""Reader for Italian Parkinson's Voice and Speech (Dimauro & Girardi, IEEE DataPort).

Layout after unzipping into data/raw/italian/:
    15 Young Healthy Control/<Name S>/<TASK><code>.wav      + 15 YHC.xlsx
    22 Elderly Healthy Control/<NAME S>/<TASK><code>.wav    + Tab 3.xlsx
    28 People with Parkinson's disease/<batch>/<Name S>/... + TAB 5.xlsx

Task codes (FILE CODES.xlsx): B1/B2 read passage, FB1 phrases, D1/D2 "pa"/"ta" syllables,
VA..VU 1/2 sustained vowels. PR1 is not in the legend; kept as "pr1".

Privacy: folders are named after participants. Names are used only in memory to group
recordings and look up age/sex; speaker IDs are salted-free short hashes, and no name is
written to the manifest. File paths (which contain folder names) stay in the local,
git-ignored data/manifest.csv only.

Identity: some PD participants were recorded in two sessions and appear in two batch
folders (e.g. 6-10/ and 11-16/). One speaker = cohort + name folder + the most common
birth year found in that folder's file names, so repeat sessions share an ID and two
different people with the same short name do not. Young controls' file names carry a
copied code, so they use the folder alone.
"""
import hashlib
import re
import unicodedata
from collections import Counter

import pandas as pd

from neurovoice.config import RAW_DIR

COHORTS = {
    "15 Young Healthy Control": ("young_hc", 0, "15 YHC.xlsx"),
    "22 Elderly Healthy Control": ("elderly_hc", 0, "Tab 3.xlsx"),
    "28 People with Parkinson's disease": ("pd", 1, "TAB 5.xlsx"),
}
TASK_RE = re.compile(r"^(B1|B2|PR1|FB1|D1|D2|V[AEIOU][12])", re.IGNORECASE)
BIRTH_RE = re.compile(r"(\d{2})[MF]\d{6,}", re.IGNORECASE)
RECORDING_YEAR = 2017  # most sessions are 2016-2017; only used to break name ties


def task_of(filename):
    m = TASK_RE.match(filename)
    if not m:
        return None
    code = m.group(1).upper()
    if code in ("B1", "B2"):
        return "read_text"
    if code == "FB1":
        return "phrases"
    if code in ("D1", "D2"):
        return "syllables"
    if code.startswith("V"):
        return "vowel"
    return "pr1"


def birth_year(filename):
    m = BIRTH_RE.search(filename)
    return int(m.group(1)) if m else None


def norm(text):
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", "", text.lower()).split()


def folder_key(folder_name):
    words = norm(folder_name)
    return " ".join(words[:-1] + [words[-1][0]]) if words else ""


def read_demographics(xlsx):
    """Rows of {name, surname, age, sex} from a cohort sheet (header on the second row)."""
    df = pd.read_excel(xlsx, header=1)
    df.columns = [str(c).strip().lower() for c in df.columns]
    rows = []
    for _, r in df.dropna(subset=["name", "age"]).iterrows():
        name, surname = norm(r["name"]), norm(r["surname"])
        if name and surname:
            rows.append({"name": name, "surname": surname, "age": int(r["age"]),
                         "sex": str(r["sex"]).strip().upper()[:1] or None})
    return rows


def sex_from_code(filenames):
    sexes = Counter(m.group(0)[2].upper() for m in map(BIRTH_RE.search, filenames) if m)
    return sexes.most_common(1)[0][0] if sexes else None


def match_demographics(folder_name, demo, yy=None, sex=None):
    """(age, sex) for a name folder. Tries, in order: first name + surname initial; a
    unique first name; initials in either order (folders and sheets sometimes swap
    name/surname or spell out only one). Ties are broken by the sex and birth year
    encoded in the file names. Returns (None, None) if nothing matches."""
    words = norm(folder_name)
    if not words or not demo:
        return None, None
    first, last_initial = " ".join(words[:-1]), words[-1][0]
    initials = {(words[0][0], last_initial), (last_initial, words[0][0])}

    def full(r):
        return " ".join(r["name"]) == first and r["surname"][0][0] == last_initial

    def first_name(r):
        return " ".join(r["name"]) == first

    def by_initials(r):
        return (r["name"][0][0], r["surname"][0][0]) in initials

    for rule in (full, first_name, by_initials):
        options = [r for r in demo if rule(r)]
        if rule is first_name and len(options) != 1:
            continue
        if sex:
            options = [r for r in options if r["sex"] == sex] or options
        if options:
            if yy is not None:
                expected = RECORDING_YEAR - (1900 + yy)
                options.sort(key=lambda r: abs(r["age"] - expected))
            return options[0]["age"], options[0]["sex"]
    return None, None


def parse_italian(root=None):
    from neurovoice.manifest import _sha1  # avoid a circular import at module load
    import soundfile as sf

    root = root or RAW_DIR / "italian"
    rows = []
    for group_dir, (cohort, label, sheet) in COHORTS.items():
        base = root / group_dir
        if not base.exists():
            continue
        demo = read_demographics(base / sheet) if (base / sheet).exists() else []
        folders = sorted({p.parent for p in base.rglob("*.wav")})
        for folder in folders:
            wavs = sorted(p for p in folder.glob("*.wav") if task_of(p.name))
            years = Counter(y for y in map(birth_year, (p.name for p in wavs)) if y is not None)
            coded = cohort != "young_hc"  # young controls' file codes are copy-pasted
            yy = years.most_common(1)[0][0] if coded and years else None
            code_sex = sex_from_code([p.name for p in wavs]) if coded else None
            key = folder_key(folder.name)
            age, sex = match_demographics(folder.name, demo, yy, code_sex)
            spk = hashlib.sha1(f"{cohort}|{key}|{yy}".encode()).hexdigest()[:8]
            for path in wavs:
                info = sf.info(str(path))
                rows.append({
                    "file": path.relative_to(root.parent.parent.parent).as_posix(),
                    "speaker_id": f"italian_{spk}",
                    "label": label,
                    "dataset": "italian",
                    "language": "it",
                    "task": task_of(path.name),
                    "sample_rate": info.samplerate,
                    "duration_s": round(info.frames / info.samplerate, 2),
                    "age": age,
                    "gender": sex,
                    "sha1": _sha1(path),
                    "cohort": cohort,
                })
    return rows
