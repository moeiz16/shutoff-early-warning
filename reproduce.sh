#!/usr/bin/env bash
# Simulate the data, run the tests, and rebuild everything in outputs/ and tickets/.
# Usage: ./reproduce.sh
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  python3.11 -m venv .venv
  .venv/bin/python -m pip install --quiet --upgrade pip
  .venv/bin/python -m pip install --quiet -r requirements.txt
fi

PYTHONPATH=src .venv/bin/python -m earlywarning.simulate --out data
.venv/bin/python -m tickets.simulate --out data
.venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python -m earlywarning --data-dir data --out outputs
.venv/bin/python -m tickets.extract --data-dir data --out tickets/extracted.jsonl
.venv/bin/python -m tickets.score --data-dir data --out tickets/agreement.csv
