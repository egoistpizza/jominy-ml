PY ?= .venv/bin/python

.PHONY: all install check lint test results clean

all: check results

install:
	python3 -m venv .venv
	$(PY) -m pip install -r requirements-lock.txt
	$(PY) -m pip install --no-deps -e .

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests
	$(PY) -m mypy

test:
	$(PY) -m pytest

check: lint test

results:
	$(PY) -m jominy.cli run --out results

clean:
	rm -rf results
