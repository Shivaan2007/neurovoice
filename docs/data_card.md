# Data card

Datasets are **never committed**. Download them yourself.

| Dataset | Language | Task used | Source | License note |
|---|---|---|---|---|
| MDVR-KCL | English | Read text (smartphone); spontaneous dialogue also present, unused | [Zenodo 2867216](https://zenodo.org/records/2867216): `26_29_09_2017_KCL.zip` (606 MB), unzip into `data/raw/mdvr_kcl/` | CC BY 4.0 (per Zenodo); trained weights may be shared |
| Italian Parkinson's Voice and Speech | Italian | Read text (B1/B2); vowels, syllables, phrases also parsed | [IEEE DataPort](https://ieee-dataport.org/open-access/italian-parkinsons-voice-and-speech) (free login, DOI 10.21227/aw6b-tg17): zip (593 MB), unzip into `data/raw/italian/` | Creative Commons Attribution (per the IEEE DataPort page, checked 2026-11-23; version not stated). **Recording confound, see below** |
| NeuroVoz | Spanish | Repeated sentences (listen-and-repeat) | NeuroVoz corpus | CC BY-NC-ND: evaluation only, no trained weights shared until confirmed |
| UCI Parkinson's Disease Classification | Turkish speakers, sustained /a/ | Tabular features (leakage demo): 756 recordings, 252 subjects (188 PD, 64 healthy), 3 per subject, 753 feature columns (incl. gender) besides `id` and `class`. The download is a zip holding a `.rar`; Windows `tar -xf` extracts it | [UCI #470](https://archive.ics.uci.edu/dataset/470); unzip `pd_speech_features.csv` into `data/raw/uci/` | CC BY 4.0 (per the UCI page, checked 2026-11-23) |

## MDVR-KCL (from `data/manifest.csv`, built 2026-11-22)

| Task | Label | Recordings | Speakers | Median length |
|---|---|---|---|---|
| Read text | Healthy | 21 | 21 | 148 s |
| Read text | PD | 16 | 16 | 130 s |
| Spontaneous dialogue | Healthy | 21 | 21 | 141 s |
| Spontaneous dialogue | PD | 15 | 15 | 141 s |

- All files are 44.1 kHz WAV; resampled to 16 kHz for features.
- No duplicate files (SHA-1 check); every speaker ID has one label.
- Age and sex are not in the file names or release, so they are blank in the manifest. The Zenodo record lists a single file (`26_29_09_2017_KCL.zip`) and no separate metadata sheet (checked 2026-11-23).
- Facts from the Zenodo description (record 2867216, CC BY 4.0, checked 2026-11-23): recorded at King's College London Hospital, 26-29 September 2017, in a roughly 10 m² examination room, on a Motorola Moto G4 phone held to the ear (44.1 kHz, 16 bit), by participants reading "The North Wind and the Sun" (some also a second passage), then a spontaneous dialogue.
- File names carry clinician ratings: `IDnn_<hc|pd>_<Hoehn & Yahr>_<UPDRS II-5>_<UPDRS III-18>`. These are the only subgroup labels we have for the audio; see `results/severity_check.csv`. One healthy control (ID31) has a non-zero rating and some PD speakers are rated 0, which the authors say means a rater could not hear voice change, not that labels are ambiguous.

## Italian Parkinson's Voice and Speech (from `data/manifest.csv`, built 2026-11-22)

| Cohort | Speakers | Age range (median) | Sex (F/M) | Read-text recordings |
|---|---|---|---|---|
| Young healthy | 15 | 19-29 (20) | 2 / 13 | 30 |
| Elderly healthy | 22 | 60-77 (68) | 12 / 10 | 42 |
| PD | 25 | 40-80 (70) | 9 / 16 | 51 |

- **Privacy:** folders are named after participants. Speaker IDs are short hashes and no name is written to the manifest or results; age and sex come from the cohort spreadsheets, matched in memory.
- **Repeat sessions:** 3 PD participants appear in two batch folders (2016 and 2017 sessions). They share one speaker ID, so the dataset's "28 PD" is **25 people**. Matching folder + birth year (from the file code) also keeps apart two different people with the same short name.
- **Duplicates:** one PD speaker's B1 and B2 are the same audio file; the manifest keeps one.
- **File codes are unreliable:** young controls' file names carry a copied code, and some codes have typos, so identity uses the folder.
- **PR1** is not in the dataset's legend; it is kept as task `pr1`.

**Recording confound (`experiments/italian_baseline.py`, `results/italian_baseline.csv`).**
- All 44.1 kHz files and the whole 2016 session are PD; every control is 16 kHz from 2017.
- Even on matched recordings (no young controls, 16 kHz only, 36 speakers), every model scores AUROC 1.0, and a single feature (mean of MFCC 8, mostly microphone and room) separates PD from healthy perfectly on its own.
- Real PD speech differences do not separate this cleanly with hand-built features, so PD and control recordings differ in how they were made.
- **Use:** a recording-confound example and a cross-dataset test set only. Do not report its within-dataset scores as evidence that the model recognises Parkinson's, and do not train the demo model on it.

Counts for NeuroVoz: to be filled when its loader lands.
