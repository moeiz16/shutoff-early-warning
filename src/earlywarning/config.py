"""Constants the analysis depends on. If a number in the README moves, it starts here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"

TARGET = "shutoff_next_6"

# Scoring happens at the end of billing cycle 12. I assume a bill falls due at
# the end of the cycle it was issued in, so a bill from cycle c has been due for
# (12 - c) * 30 days on the scoring date. See features.visible_at_cutoff.
CUTOFF_CYCLE = 12
DAYS_PER_CYCLE = 30

CALL_COST = 40.0        # one support call, needed or not (staff time for a proper conversation)
AVOIDABLE_LOSS = 500.0  # one shut-off a call would have prevented (lost revenue, crew visits, reconnection)

SEED = 11
TEST_SHARE = 0.2
N_FOLDS = 5
N_BOOT = 2000

# Fields kept out of the model, and why. tests/test_features.py checks that
# none of these ever reaches a feature matrix.
EXCLUDED_FIELDS = {
    "shutoff_notices": (
        "Counted on the day the extract was pulled (end of cycle 18), so it includes "
        "warning letters sent during the outcome window."
    ),
    "status_at_pull": "Connection status on the day the extract was pulled; 'shut_off' is the label restated.",
    "tariff": (
        "Recoded part-way through the year. Once old and new codes are mapped, CARE is exactly "
        "the hardship plan, and STD vs FLEX changes from cycle to cycle as if drawn at random."
    ),
    "customer_id": "Identifier.",
}
