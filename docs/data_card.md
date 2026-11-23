# Data card

Datasets are **never committed**. Download them yourself.

| Dataset | Language | Task used | Source | License note |
|---|---|---|---|---|
| MDVR-KCL | English | Read text (smartphone); spontaneous dialogue also present, unused | [Zenodo 2867216](https://zenodo.org/records/2867216): `26_29_09_2017_KCL.zip` (606 MB), unzip into `data/raw/mdvr_kcl/` | CC BY 4.0 (per Zenodo); trained weights may be shared |
| Italian Parkinson's Voice and Speech | Italian | Read text | IEEE DataPort (free login) | TODO: confirm license |
| NeuroVoz | Spanish | Repeated sentences (listen-and-repeat) | NeuroVoz corpus | CC BY-NC-ND: evaluation only, no trained weights shared until confirmed |
| UCI Parkinson's Disease Classification | Turkish speakers, sustained /a/ | Tabular features (leakage demo): 756 recordings, 252 subjects (188 PD, 64 healthy), 3 per subject, 753 feature columns (incl. gender) besides `id` and `class`. The download is a zip holding a `.rar`; Windows `tar -xf` extracts it | [UCI #470](https://archive.ics.uci.edu/dataset/470); unzip `pd_speech_features.csv` into `data/raw/uci/` | TODO: confirm license on the UCI page |

## MDVR-KCL (from `data/manifest.csv`, built 2026-11-22)

| Task | Label | Recordings | Speakers | Median length |
|---|---|---|---|---|
| Read text | Healthy | 21 | 21 | 148 s |
| Read text | PD | 16 | 16 | 130 s |
| Spontaneous dialogue | Healthy | 21 | 21 | 141 s |
| Spontaneous dialogue | PD | 15 | 15 | 141 s |

- All files are 44.1 kHz WAV; resampled to 16 kHz for features.
- No duplicate files (SHA-1 check); every speaker ID has one label.
- Age and sex are not in the file names or release, so they are blank in the manifest.

Counts for the Italian and NeuroVoz sets: to be filled when their loaders land.
