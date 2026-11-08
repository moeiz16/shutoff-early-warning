PYTHON ?= python3.11
VENV   := .venv
PY     := $(VENV)/bin/python

.PHONY: all setup data test model tickets clean

all: data test model tickets

setup: $(VENV)/.installed

$(VENV)/.installed: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --quiet --upgrade pip
	$(PY) -m pip install --quiet -r requirements.txt
	touch $@

data: setup
	PYTHONPATH=src $(PY) -m earlywarning.simulate --out data
	$(PY) -m tickets.simulate --out data

test: setup
	$(PY) -m pytest -q

model: setup
	PYTHONPATH=src $(PY) -m earlywarning --data-dir data --out outputs

tickets: setup
	$(PY) -m tickets.extract --data-dir data --out tickets/extracted.jsonl
	$(PY) -m tickets.score --data-dir data --out tickets/agreement.csv

clean:
	rm -rf data outputs/tables outputs/figures outputs/summary.json .pytest_cache
	find . -name __pycache__ -prune -exec rm -rf {} +
