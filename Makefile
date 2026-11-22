.PHONY: install test lint reproduce
install:
	pip install -r requirements.txt
test:
	pytest -q
lint:
	ruff check .
# Runs every step that exists so far. Add `python -m neurovoice.train` here when it lands.
reproduce:
	python -m neurovoice.manifest && python -m experiments.mdvr_segment_check
