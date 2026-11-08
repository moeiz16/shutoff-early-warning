import numpy as np
import pandas as pd
import pytest

from earlywarning.config import EXCLUDED_FIELDS
from earlywarning.features import (
    build_features,
    feature_group,
    payment_features,
    visible_at_cutoff,
)
from tests.conftest import one_customer_history


def _history(cycle11=(100.0, 60.0, 45, "post"), cycle12=(100.0, 50.0, 20, "post")):
    rows = [(c, 100.0, 100.0, 0, "direct_debit") for c in range(1, 11)]
    rows.append((11, *cycle11))
    rows.append((12, *cycle12))
    return one_customer_history(rows)


def test_cycle_12_payment_is_treated_as_unknown():
    view = visible_at_cutoff(_history()).set_index("cycle")
    assert not view.loc[12, "payment_observed"]
    assert np.isnan(view.loc[12, "late"]) and np.isnan(view.loc[12, "paid_seen"])
    assert pd.isna(view.loc[12, "channel_seen"])
    assert view.loc[12, "amount_due"] == 100.0  # the bill itself is known


def test_payment_after_the_scoring_date_counts_as_outstanding():
    # Cycle 11 bill paid 45 days late: due at the end of cycle 11, so only 30 days had passed.
    view = visible_at_cutoff(_history()).set_index("cycle")
    assert view.loc[11, "outstanding"] == 1
    assert view.loc[11, "paid_seen"] == 0.0
    assert view.loc[11, "days_late_seen"] == 30
    assert view.loc[11, "late"] == 1


def test_payment_before_the_scoring_date_is_kept():
    view = visible_at_cutoff(_history(cycle11=(100.0, 60.0, 25, "post"))).set_index("cycle")
    assert view.loc[11, "outstanding"] == 0
    assert view.loc[11, "paid_seen"] == 60.0
    assert view.loc[11, "days_late_seen"] == 25


def test_payment_features_by_hand():
    f = payment_features(visible_at_cutoff(_history())).loc[1]
    # 11 observed bills: cycles 1-10 on time by direct debit, cycle 11 late by post and unpaid.
    assert f["overdue_share"] == pytest.approx(1 / 11)
    assert f["avg_days_overdue"] == pytest.approx(30 / 11)
    assert f["unpaid_share"] == pytest.approx(100 / 1100)
    assert f["avg_amount_due"] == pytest.approx(100.0)
    assert f["channel_share_post"] == pytest.approx(1 / 11)
    assert f["channel_share_app"] == 0 and f["channel_share_counter"] == 0


def test_features_ignore_anything_recorded_after_the_scoring_date(small_extract):
    customers, billing, _ = small_extract
    before = build_features(customers, billing)

    changed = billing.copy()
    rng = np.random.default_rng(1)
    last = changed["cycle"] == 12
    changed.loc[last, "days_overdue"] = rng.integers(0, 91, last.sum())
    changed.loc[last, "amount_paid"] = 0.0
    changed.loc[last, "pay_channel"] = "counter"
    # Late cycle-11 payments: change when (after the scoring date) and how much they paid.
    after = (changed["cycle"] == 11) & (changed["days_overdue"] > 30)
    changed.loc[after, "days_overdue"] = 90
    changed.loc[after, "amount_paid"] = changed.loc[after, "amount_due"]
    leaky = customers.assign(shutoff_notices=9, status_at_pull="shut_off")

    pd.testing.assert_frame_equal(before, build_features(leaky, changed))


def test_as_recorded_view_does_see_cycle_12(small_extract):
    customers, billing, _ = small_extract
    pit = build_features(customers, billing)
    raw = build_features(customers, billing, point_in_time=False)
    assert not np.allclose(pit["overdue_share"], raw["overdue_share"])


@pytest.mark.parametrize("extended", [False, True])
def test_no_excluded_field_reaches_the_model(small_extract, extended):
    customers, billing, _ = small_extract
    X = build_features(customers, billing, extended=extended)
    assert not set(X.columns) & set(EXCLUDED_FIELDS)
    assert not any("tariff" in c or "notice" in c or "status" in c for c in X.columns)
    assert X.notna().all().all()
    assert X.index.is_monotonic_increasing and X.index.is_unique


def test_income_is_one_hot_with_tier_3_as_reference(small_extract):
    customers, billing, _ = small_extract
    X = build_features(customers, billing)
    tier_cols = [c for c in X.columns if c.startswith("income_tier_")]
    c = customers.set_index("customer_id").loc[X.index]
    tier3_or_missing = (c["income_tier"] == 3) | c["income_tier"].isna()
    assert (X.loc[tier3_or_missing, tier_cols].sum(axis=1) == 0).all()
    assert (X["income_missing"] == c["income_tier"].isna().astype(int)).all()


def test_every_feature_has_a_readable_group(small_extract):
    customers, billing, _ = small_extract
    for col in build_features(customers, billing).columns:
        assert feature_group(col) != col, f"{col} has no readable group name"
