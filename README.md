# NeuroVoice

**A Parkinson's voice screening-support research prototype that tests how well voice models hold up on speakers and datasets they've never seen.**

> Research prototype. Not a diagnosis, not clinically validated. Small cohorts, three languages.

## Results

Headline numbers come only from speaker-grouped and cross-dataset tests. Naive-split numbers are shown only to measure how much they inflate. AUROC: 0.5 is chance, 1.0 is perfect. CIs are 95%, bootstrapped over speakers.

**Leakage: the same people in train and test inflate scores** (Random Forest, best of LR / SVM / RF in each case)

| Dataset | Naive random split (leaks speakers) | Speaker-grouped split (95% CI) | Inflation | File |
|---|---|---|---|---|
| MDVR-KCL, 5 s windows of read text (37 speakers) | 0.972 | 0.738 (0.585-0.879) | 0.234 | [`results/segment_check.csv`](results/segment_check.csv) |
| UCI tabular, 3 recordings per subject (252 subjects) | 0.949 | 0.832 (0.779-0.882) | 0.117 | [`results/uci_leakage.csv`](results/uci_leakage.csv) |

Many windows per person leak more than three recordings per person. Every model in both files shows the same direction.

**Demo model: MDVR-KCL, held-out speakers only** ([`results/train_cv.csv`](results/train_cv.csv))

| Model | Window AUROC (95% CI) | Speaker-level AUROC | Brier |
|---|---|---|---|
| Logistic regression | 0.526 (0.359-0.696) | 0.577 | 0.247 |
| SVM | 0.660 (0.496-0.826) | 0.729 | 0.225 |
| **Random Forest (deployed)** | **0.716 (0.562-0.877)** | **0.774** | 0.204 |

Calibrated (sigmoid, nested inside speaker-grouped CV). The app says "higher" above 0.6, a cut tuned until it was right 80% of the time on the held-out windows it was fit on (about 19% of windows, from 23 of 37 speakers). That 80% is in-sample. When the cut is chosen on some speakers and checked on others, "higher" was right about 74% of the time overall, and between 15% and 100% depending on the fold, each fold resting on only 4 to 7 speakers (`results/threshold_crossfit.csv`). Treat "higher" as a weak signal. No cut reached 80% for "lower", so the model never says it; everything else is "not enough confidence". See [docs/model_card.md](docs/model_card.md).

**Unseen dataset (leave-one-dataset-out):** _pending_ (needs the Italian and NeuroVoz loaders).

## Why this project

Our Cal Hacks 12.0 (Oct 2025) prototype claimed 87%. The code is lost, so we cannot say exactly how that was measured. This rebuild focuses on honest evaluation: speaker-level splits, unseen-dataset tests, confound checks, calibrated probabilities and stated limits.

## What could fool us (validation checks)

| Check | How we test | Status |
|---|---|---|
| Age | Rerun without Italian young controls (19-29); age-only baseline | planned |
| Recording conditions | Classifier predicting which dataset a clip came from | planned |
| Segment leakage | 5 s windows of MDVR-KCL, grouped vs random split | done: 0.972 → 0.738 |
| Recording leakage | UCI, 3 recordings per subject, grouped vs random split | done: 0.949 → 0.832 |
| Sex balance | Sex counts per class; subgroup AUROC | planned |
| Medication state | Cannot test (NeuroVoz is medication-optimized) | limitation |
| Small numbers | Speaker-level bootstrap CIs | done for all results above |

## Data

MDVR-KCL (Zenodo), Italian Parkinson's Voice and Speech (IEEE DataPort), NeuroVoz (CC BY-NC-ND, evaluation only), UCI Parkinson's Disease Classification. Common task: connected speech. See [docs/data_card.md](docs/data_card.md). Datasets are never committed.

## Methods

Audio is resampled to 16 kHz mono, trimmed, and cut into 5 s windows. Features per window: MFCC 1-13 (mean/std/delta), pyin pitch, RMS, spectral centroid, ZCR, plus Praat jitter, shimmer, HNR (wav2vec2-base embeddings planned). Logistic regression, SVM and Random Forest; `StratifiedGroupKFold` by speaker; scaling and calibration fit inside training folds only. The app scores each 5 s window of a clip and averages, the same way the model was trained.

## Reproduce

```bash
make install
make reproduce   # manifest -> MDVR-KCL segment check -> UCI leakage -> train
```

Needs the datasets in `data/raw/` first; see [docs/data_card.md](docs/data_card.md).

## Run the app

```bash
pip install -r requirements.txt  # needs ffmpeg installed
uvicorn api.main:app --port 8000 &
streamlit run app/streamlit_app.py
```

The app uses `models/model.joblib` (Random Forest trained on MDVR-KCL). If the model files are missing it falls back to a stub (0.5, "uncertain"). Clips must be 5 to 30 seconds.

## Layout

`neurovoice/` library (manifest, features, splits, evaluate, train, predict), `experiments/` one-off studies, `results/` their output CSVs, `models/` the deployed model, `api/` FastAPI, `app/` Streamlit, `docs/`, `tests/`.

## Limitations and ethics

Small cohorts, three languages, different recording gear per dataset, medication state uncontrolled, no clinical validation. Not for diagnosis. Uploaded audio is deleted right after prediction. NeuroVoz weights are not shared until its license is confirmed.

## Status

Done: scaffold, both leakage demos, first trained model (MDVR-KCL) in the app. Next: Italian and NeuroVoz loaders, leave-one-dataset-out tests, wav2vec2 embeddings, error analysis, public deployment.

## License

MIT for code. Datasets keep their own licenses.
