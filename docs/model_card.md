# Model card: `mdvr-kcl-random_forest-v1`

A screening-support **research prototype**. It is not a diagnostic tool and has no clinical validation.

## Model

- **Type:** Random Forest (300 trees) with sigmoid calibration (`CalibratedClassifierCV`, speaker-grouped folds).
- **Input:** a 5-30 s voice clip, read aloud. It is resampled to 16 kHz mono, trimmed, and cut into 5 s windows.
- **Features per window:** 48 hand-built acoustic features: MFCC 1-13 (mean, std, delta mean), pitch (mean, std, range via pyin), RMS, spectral centroid, zero-crossing rate, Praat jitter, shimmer, HNR.
- **Output:** the average calibrated probability over the clip's windows, plus a band:
  - `higher`: probability > 0.6
  - `uncertain`: everything else ("not enough confidence to say")
  - `lower`: never used by this version (see below)
- **Files:** `models/model.joblib`, `models/meta.json`. Produced by `python -m neurovoice.train`; trained 2026-11-22 with scikit-learn 1.9.1.

## Intended use

- Research and teaching on how voice models behave on unseen speakers and datasets.
- A public demo that shows a calibrated probability and admits uncertainty.

**Out of scope:** diagnosis, screening of real patients, or any clinical or personal health decision.

## Training data

MDVR-KCL read text: 37 English speakers (16 PD, 21 healthy) recorded on a smartphone, 1,001 windows of 5 s. License CC BY 4.0. Age and sex are not in the released files. See [data_card.md](data_card.md).

## Evaluation

Speaker-grouped 5-fold CV; calibration is fit on inner grouped folds of each training split, so no test speaker influences any fitted step. Full results for all three candidate models: [`results/train_cv.csv`](../results/train_cv.csv).

| Metric (held-out speakers) | Value |
|---|---|
| Window AUROC (95% CI, bootstrapped over speakers) | 0.716 (0.562-0.877) |
| Speaker-level AUROC (mean probability per speaker, 95% CI) | 0.777 (0.587-0.948) |
| Brier score | 0.204 |
| Expected calibration error | 0.105 |

The model was picked as the best of logistic regression, SVM and Random Forest on the same CV, so these numbers are slightly optimistic.

## How the confidence band was chosen

From held-out window probabilities:

- `higher` uses the smallest cut where windows above it were PD at least 80% of the time, and it must cover windows from at least 5 different speakers. That cut is 0.6, and it covers about 18.5% of held-out windows (23 of 37 speakers). **That 80% is in-sample:** the cut was tuned on the same held-out predictions it is scored on.
- **Honest estimate (cross-fitted):** choosing the cut on some speakers and measuring on the others, `higher` was right about 74% of the time overall. Per fold it was 91%, 100%, 60%, 15% and 62%, each fold resting on only 4 to 7 speakers. Quote 74%, not 80%, and treat `higher` as a weak signal. See `results/threshold_crossfit.csv` and `band_precision` in `models/meta.json`.
- `lower` would use the largest cut where windows below it were healthy at least 80% of the time. **No cut reached that**, so the model never says `lower`.
- The cut was tuned on single 5 s windows, but the app averages a clip's windows before applying it, so clip-level precision has not been measured.
- About 81.5% of held-out windows fall in `uncertain`.

In plain terms: when this model is unsure, it says so, and it currently cannot reliably reassure anyone.

## Known limits

- **Tiny cohort:** 37 speakers, one dataset, one language, one recording setup. The wide CI reflects this.
- **No test on other datasets yet:** performance on other languages, microphones or rooms is unknown and expected to be lower.
- **Not controlled:** age, sex and medication state.
- **Feature caveat:** jitter and shimmer are normally measured on sustained vowels; here they come from read speech.
- **No clinical validation.**

## Ethics

- Uploaded audio is deleted right after prediction.
- The app shows a research-prototype notice with every result.
- NeuroVoz (CC BY-NC-ND) is not used to train any shared weights.
