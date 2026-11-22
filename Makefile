.PHONY: install test lint reproduce
install:
	pip install -r requirements.txt
test:
	pytest -q
lint:
	ruff check .
reproduce:
	python -m neurovoice.manifest && python -m experiments.mdvr_segment_check && \
	python -m experiments.uci_leakage && python -m neurovoice.train
