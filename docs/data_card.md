# Data card

Datasets are **never committed**. Download them yourself.

| Dataset | Language | Task used | Source | License note |
|---|---|---|---|---|
| MDVR-KCL | English | Read text (smartphone) | Zenodo | TODO: confirm license |
| Italian Parkinson's Voice and Speech | Italian | Read text | IEEE DataPort (free login) | TODO: confirm license |
| NeuroVoz | Spanish | Repeated sentences (listen-and-repeat) | NeuroVoz corpus | CC BY-NC-ND: evaluation only, no trained weights shared until confirmed |
| UCI Parkinson's Disease Classification | Turkish speakers, sustained /a/ | Tabular features (leakage demo): 756 recordings, 252 subjects (188 PD, 64 healthy), 3 per subject, 753 feature columns (incl. gender) besides `id` and `class`. The download is a zip holding a `.rar`; Windows `tar -xf` extracts it | [UCI #470](https://archive.ics.uci.edu/dataset/470); unzip `pd_speech_features.csv` into `data/raw/uci/` | TODO: confirm license on the UCI page |

Counts, sample rates, durations, age and sex tables: to be filled from `data/manifest.csv`.
