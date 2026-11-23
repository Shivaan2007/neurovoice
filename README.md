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

Calibrated (sigmoid, nested inside speaker-grouped CV). The app says "higher" above 0.6, a cut tuned until it was right 80% of the time on the held-out windows it was fit on (about 18.5% of windows, from 23 of 37 speakers). That 80% is in-sample. When the cut is chosen on some speakers and checked on others, "higher" was right about 74% of the time overall, and between 15% and 100% depending on the fold, each fold resting on only 4 to 7 speakers (`results/threshold_crossfit.csv`). Treat "higher" as a weak signal. No cut reached 80% for "lower", so the model never says it; everything else is "not enough confidence". See [docs/model_card.md](docs/model_card.md).

**Unseen dataset (leave-one-dataset-out):** _pending_ (needs the Italian and NeuroVoz loaders).

## Why this project

Our Cal Hacks 12.0 (Oct 2025) prototype claimed 87%. The code is lost, so we cannot say exactly how that was measured. This rebuild focuses on honest evaluation: speaker-level splits, unseen-dataset tests, confound checks, calibrated probabilities and stated limits.

## What could fool us (validation checks)

| Check | How we test | Status |
|---|---|---|
| Age | Rerun without Italian young controls (19-29); age-only baseline | done (Italian): age alone scores 0.74 with young controls, 0.56 without |
| Recording conditions | Classifier predicting which dataset a clip came from | partly done: Italian PD and controls differ in recording; dataset classifier needs two datasets together |
| Segment leakage | 5 s windows of MDVR-KCL, grouped vs random split | done: 0.972 → 0.738 |
| Recording leakage | UCI, 3 recordings per subject, grouped vs random split | done: 0.949 → 0.832 |
| Sex balance | Sex counts per class; subgroup AUROC | done on UCI (sex is not a shortcut there); Italian counts only; no sex data for MDVR-KCL |
| Threshold tuned on test data | Choose the band cut on some speakers, score on others | done: "higher" 80% in-sample → 74% on other speakers |
| Medication state | Cannot test (NeuroVoz is medication-optimized) | limitation |
| Small numbers | Speaker-level bootstrap CIs | done for all results above |

All numbers are collected in `results/metrics.json` (built by `experiments/export_metrics.py` from the results files).

**Age.** The Italian healthy group mixes 22 elderly (60-77) and 15 young (19-29) controls, while PD speakers are 40-80, so a model could learn "young voice = healthy". Age alone separates the groups with AUROC 0.74 when young controls are included and 0.56 without them, so the headline runs exclude them. The models still score 0.96 to 1.0 either way, which means age is not what drives that score; the recording differences below are.

**Recording conditions.** In the Italian set every control was recorded at 16 kHz in 2017, while all the 44.1 kHz files and the whole 2016 session are PD. Even with young controls removed and only 16 kHz files kept (36 speakers), every model reaches AUROC 1.0 and a single feature (mean of MFCC 8, which mostly reflects microphone and room) separates PD from healthy on its own. Real speech differences do not separate this cleanly, so those scores reflect how the files were made. We use the Italian set only to show the confound and as a test set, never to train the demo model. We have not yet trained a classifier to guess which dataset a clip came from, which needs two datasets loaded together.

**Segment and recording leakage.** Cutting MDVR-KCL read text into 5 s windows and splitting at random puts the same person in train and test: Random Forest AUROC 0.972 against 0.738 when split by speaker, a gap of 0.23. UCI, with 3 recordings per subject, shows the same direction at a smaller size (0.949 against 0.832). This shows one common way hackathon-style numbers get inflated; we do not claim it is what our original version did.

**Sex balance.** Italian speakers by sex (F/M): young controls 2/13, elderly controls 12/10, PD 9/16, so sex differs by class and could act as a shortcut. MDVR-KCL ships without age or sex, so we cannot check it there. UCI does record sex, and it is imbalanced too: 57% of PD subjects are men against 36% of healthy ones. We tested it there with a subject-grouped Random Forest: sex alone scores AUROC 0.565 (95% CI 0.488-0.650, about chance), adding the gender column changes nothing (0.833 against 0.832 without it), and without gender the model scores 0.811 for men and 0.835 for women (`results/uci_sex_check.csv`). So sex is not a shortcut on UCI. That does not carry over to the audio datasets, where we have no sex labels for MDVR-KCL.

**Threshold tuned on test data.** The "higher" cut of 0.6 was tuned until it was right 80% of the time on the held-out windows it is scored on. When the cut is chosen on some speakers and checked on others it is right about 74% of the time, and between 15% and 100% by fold, each fold resting on 4 to 7 speakers.

**Small numbers.** MDVR-KCL has 16 PD speakers. Every headline number carries a 95% interval from resampling speakers, and the intervals are wide (for example window AUROC 0.716, interval 0.562 to 0.877).

**Medication state.** NeuroVoz patients were recorded in their medication-optimized state. We cannot test the effect of medication, so it is a limitation.

## Data

MDVR-KCL (Zenodo), Italian Parkinson's Voice and Speech (IEEE DataPort), NeuroVoz (CC BY-NC-ND, evaluation only), UCI Parkinson's Disease Classification. Common task: connected speech. See [docs/data_card.md](docs/data_card.md). Datasets are never committed.

## Methods

Audio is resampled to 16 kHz mono, trimmed, and cut into 5 s windows. Features per window: MFCC 1-13 (mean/std/delta), pyin pitch, RMS, spectral centroid, ZCR, plus Praat jitter, shimmer, HNR. As a comparison (not used by the app), wav2vec2-base embeddings mean-pooled per window with logistic regression score window AUROC 0.759 (0.617-0.881) and speaker AUROC 0.842 on the same speaker-grouped CV, against 0.716 (0.562-0.877) and 0.774 for the hand-built Random Forest; the intervals overlap heavily, so this is not a clear gain (`results/embeddings_mdvr.csv`; needs `pip install torch transformers`, not in `requirements.txt`). Logistic regression, SVM and Random Forest; `StratifiedGroupKFold` by speaker; scaling and calibration fit inside training folds only. The app scores each 5 s window of a clip and averages, the same way the model was trained.

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

Done: scaffold, both leakage demos, first trained model (MDVR-KCL) in the app. Also done: error analysis and SHAP, calibration before/after, wav2vec2 comparison on MDVR-KCL, Italian loader and recording-confound finding. Next: NeuroVoz loader, run the unseen-dataset test (`experiments/cross_dataset.py`) on a machine with the Italian data, sex and dataset-classifier checks, public deployment.

## License

MIT for code. Datasets keep their own licenses.
