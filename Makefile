.PHONY: install test lint reproduce
install:
	pip install -r requirements.txt
test:
	pytest -q
lint:
	ruff check .
reproduce:
	python -m neurovoice.manifest && python -m neurovoice.train && python -m neurovoice.evaluate
