.PHONY: install test lint reproduce reproduce-italian metrics reproduce-all
install:
	pip install -r requirements.txt
test:
	pytest -q
lint:
	ruff check .
# Needs MDVR-KCL in data/raw/mdvr_kcl and UCI in data/raw/uci (see docs/data_card.md).
reproduce:
	python -m neurovoice.manifest && python -m neurovoice.train && \
	python -m experiments.mdvr_segment_check && python -m experiments.uci_leakage && \
	python -m experiments.uci_sex_check && python -m experiments.error_analysis && \
	python -m experiments.severity_check
# Needs the Italian data in data/raw/italian (and NeuroVoz in data/raw/neurovoz if you have it).
reproduce-italian:
	python -m neurovoice.manifest && python -m experiments.italian_baseline && \
	python -m experiments.cross_dataset && python -m experiments.dataset_classifier
# Collects every number into results/metrics.json.
metrics:
	python -m experiments.export_metrics
reproduce-all: reproduce reproduce-italian metrics
