"""The simulated extract really contains the problems the README says the pipeline finds."""
import pytest

from earlywarning import simulate
from earlywarning.data import Extract, check_extract, harmonize_tariffs


@pytest.fixture(scope="module")
def sim():
    customers, billing, labels = simulate.simulate(n_customers=8000, seed=5)
    return customers.merge(labels, on="customer_id"), billing


def test_simulation_passes_the_shape_checks():
    assert check_extract(Extract(*simulate.simulate(n_customers=500, seed=1))) == []


def test_same_seed_same_data():
    a = simulate.simulate(n_customers=300, seed=3)
    b = simulate.simulate(n_customers=300, seed=3)
    for x, y in zip(a, b):
        assert x.equals(y)


def test_notices_give_the_label_away(sim):
    c, _ = sim
    assert c.loc[c.shutoff_notices == 0, "shutoff_next_6"].max() == 0
    assert c.loc[c.shutoff_notices >= 2, "shutoff_next_6"].min() == 1


def test_status_at_pull_gives_the_label_away(sim):
    c, _ = sim
    assert c.loc[c.status_at_pull == "shut_off", "shutoff_next_6"].min() == 1


def test_care_tariff_is_exactly_the_hardship_plan(sim):
    c, billing = sim
    tariffs = harmonize_tariffs(billing)
    care = set(tariffs.loc[tariffs.tariff == "CARE", "customer_id"])
    assert care == set(c.loc[c.hardship_plan == 1, "customer_id"])


def test_days_overdue_piles_up_at_the_cap(sim):
    _, billing = sim
    counts = billing.days_overdue.value_counts()
    assert billing.days_overdue.max() == simulate.DAYS_OVERDUE_CAP
    assert counts[simulate.DAYS_OVERDUE_CAP] > 5 * counts[simulate.DAYS_OVERDUE_CAP - 1]


def test_missing_income_is_not_random(sim):
    c, _ = sim
    missing = c.income_tier.isna()
    assert c.loc[missing, "shutoff_next_6"].mean() > c.loc[~missing, "shutoff_next_6"].mean()
    assert c.loc[c.hardship_plan == 1, "income_tier"].isna().mean() > c.loc[c.hardship_plan == 0, "income_tier"].isna().mean()
