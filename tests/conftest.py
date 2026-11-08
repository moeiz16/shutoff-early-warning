"""Fixtures. Everything runs off the simulator, so the tests never need data/ to exist."""
import pandas as pd
import pytest

from earlywarning import simulate


@pytest.fixture(scope="session")
def small_extract():
    return simulate.simulate(n_customers=2000, seed=0)


@pytest.fixture(scope="session")
def small_data_dir(tmp_path_factory):
    return simulate.write(tmp_path_factory.mktemp("data"), n_customers=2000, seed=0)


def one_customer_history(rows):
    """Billing rows for customer 1 from (cycle, amount_due, amount_paid, days_overdue, channel)."""
    return pd.DataFrame([
        {"customer_id": 1, "cycle": c, "amount_due": due, "amount_paid": paid, "days_overdue": d,
         "tariff": "STD", "pay_channel": channel}
        for c, due, paid, d, channel in rows
    ])
