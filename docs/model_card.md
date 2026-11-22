# Model card

**Status:** placeholder until Day 1 numbers freeze.

- **Intended use:** research on evaluating voice models on unseen speakers and datasets.
- **Out of scope:** any diagnosis, screening of real patients, clinical decisions.
- **Training data:** see `docs/data_card.md`.
- **Metrics:** AUROC with speaker-level bootstrap 95% CIs (to come, in `results/`).
- **Known limits:** small cohorts, three languages, recording conditions differ by dataset, medication state not controlled, no clinical validation.
