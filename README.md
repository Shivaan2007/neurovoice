# NeuroVoice

**A Parkinson's voice screening-support research prototype that tests how well voice models hold up on speakers and datasets they've never seen.**

> Research prototype. Not a diagnosis, not clinically validated. Small cohorts, three languages.

## Results

| Evaluation | AUROC (95% CI) |
|---|---|
| Naive random split (inflated, leaks speakers) | _pending_ |
| Speaker-grouped split | _pending_ |
| Unseen dataset (leave-one-dataset-out) | _pending_ |

Filled in from `results/` after the numbers freeze. Headline numbers come only from speaker-grouped and cross-dataset tests.

## Why this project

Our Cal Hacks 12.0 (Oct 2025) prototype claimed 87%. The code is lost, so we cannot say exactly how that was measured. This rebuild focuses on honest evaluation: speaker-level splits, unseen-dataset tests, confound checks, calibrated probabilities and stated limits.

## What could fool us (validation checks)

| Check | How we test | Status |
|---|---|---|
| Age | Rerun without Italian young controls (19-29); age-only baseline | planned |
| Recording conditions | Classifier predicting which dataset a clip came from | planned |
| Segment leakage | 5 s windows of MDVR-KCL, grouped vs random split | planned |
| Sex balance | Sex counts per class; subgroup AUROC | planned |
| Medication state | Cannot test (NeuroVoz is medication-optimized) | limitation |
| Small numbers | Speaker-level bootstrap CIs | planned |

## Data

MDVR-KCL (Zenodo), Italian Parkinson's Voice and Speech (IEEE DataPort), NeuroVoz (CC BY-NC-ND, evaluation only), UCI Parkinson's Disease Classification. Common task: connected speech. See [docs/data_card.md](docs/data_card.md). Datasets are never committed.

## Methods

Features: MFCC 1-13 (mean/std/delta), pyin pitch, RMS, spectral centroid, ZCR, plus Praat jitter, shimmer, HNR; wav2vec2-base embeddings. Logistic regression, SVM and Random Forest; `StratifiedGroupKFold` by speaker; calibrated probabilities with a "not enough confidence" band.

## Run the app

```bash
pip install -r requirements.txt  # needs ffmpeg installed
uvicorn api.main:app --port 8000 &
streamlit run app/streamlit_app.py
```

Until a trained model is added in `models/`, the app returns a stub (0.5, "uncertain").

## Layout

`neurovoice/` library, `api/` FastAPI, `app/` Streamlit, `docs/`, `tests/`. Folders `experiments/`, `results/` and `models/` are added as the work lands.

## Limitations and ethics

Small cohorts, three languages, different recording gear per dataset, medication state uncontrolled, no clinical validation. Not for diagnosis. Uploaded audio is deleted right after prediction. NeuroVoz weights are not shared until its license is confirmed.

## Status

Scaffold complete: interfaces, stub predictor, API, app, CI, tests. Data pipeline, experiments and the real model are next.

## License

MIT for code. Datasets keep their own licenses.
