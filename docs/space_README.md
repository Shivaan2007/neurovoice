---
title: NeuroVoice
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Parkinson's voice screening-support research prototype
---

# NeuroVoice demo

**Research prototype. Not a diagnosis and not clinically validated.** Please talk to a doctor about any health concerns.

Record or upload 5 to 30 seconds of yourself reading aloud in English. The app returns a calibrated probability. Most clips come back as "not enough confidence to say"; "higher" is a weak signal, not a finding about you. Only speech works: other sounds give meaningless results.

- **Model:** Random Forest trained on MDVR-KCL (37 English speakers, smartphone recordings). On speakers it never saw: window AUROC 0.716 (95% CI 0.562-0.877), speaker-level AUROC 0.777 (95% CI 0.587-0.948). 0.5 is chance.
- **"Higher" band:** right about 74% of the time on unseen speakers, and much less reliable in some test groups. The model never says "lower".
- **Limits:** tiny cohort, one language and one recording setup; not validated on other datasets; milder Parkinson's scores close to healthy; age, sex and medication not controlled.
- **Privacy:** uploaded audio is deleted right after the prediction and is not stored.

Code, evaluation and model card: [github.com/Shivaan2007/neurovoice](https://github.com/Shivaan2007/neurovoice).
