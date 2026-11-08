import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from earlywarning import evaluation as ev


def test_break_even_threshold_is_the_cost_ratio():
    assert ev.break_even_threshold() == pytest.approx(40 / 500)
    assert ev.break_even_threshold(effectiveness=0.5) == pytest.approx(0.16)


def test_policy_cost_by_hand():
    y = np.array([1, 0, 1, 0])
    called = np.array([True, True, False, False])
    # two calls at $40, one missed shut-off at $500
    assert ev.policy_cost(y, called) == 580
    # if calls only work half the time, a miss only "costs" the avoidable half
    assert ev.policy_cost(y, called, effectiveness=0.5) == 330


def test_operating_point_by_hand():
    y = np.array([1, 1, 0, 0, 0])
    op = ev.operating_point(y, np.array([1, 0, 1, 0, 0], dtype=bool))
    assert op["called"] == 2 and op["true_positives"] == 1
    assert op["recall"] == 0.5 and op["precision"] == 0.5
    assert op["false_positive_rate"] == pytest.approx(1 / 3)
    assert op["cost_per_customer"] == pytest.approx((2 * 40 + 500) / 5)


def test_calibrated_scores_put_the_cost_minimum_near_the_break_even_point():
    # If p is calibrated, calling everyone above 40/500 should be (close to) the cheapest list.
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 0.3, 200_000)
    y = (rng.random(p.size) < p).astype(int)
    curve = ev.cost_curve(y, p, np.round(np.arange(0.02, 0.2, 0.0025), 4))
    best = curve.loc[curve["cost_per_customer"].idxmin(), "threshold"]
    assert abs(best - 0.08) <= 0.01


def test_flat_band_contains_the_minimum():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 0.3, 20_000)
    y = (rng.random(p.size) < p).astype(int)
    curve = ev.cost_curve(y, p, np.round(np.arange(0.02, 0.2, 0.005), 4))
    lo, hi = ev.flat_band(curve, 0.02)
    best = curve.loc[curve["cost_per_customer"].idxmin(), "threshold"]
    assert lo <= best <= hi


def test_bootstrap_interval_brackets_the_estimate_and_is_reproducible():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.3 + rng.uniform(0, 0.7, 500), 0, 1)
    a = ev.bootstrap(y, p, ev.CORE_METRICS, n_boot=200, seed=7)
    b = ev.bootstrap(y, p, ev.CORE_METRICS, n_boot=200, seed=7)
    assert a.equals(b)
    assert (a["ci_low"] <= a["value"]).all() and (a["value"] <= a["ci_high"]).all()
    assert a.loc["auc", "value"] == pytest.approx(roc_auc_score(y, p))


def test_paired_difference_of_a_model_with_itself_is_zero():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 300)
    p = rng.random(300)
    assert ev.paired_auc_difference(y, p, p, n_boot=50, seed=0) == (0.0, 0.0, 0.0)


def test_wilson_interval_known_values():
    lo, hi = ev.wilson(30, 30)
    assert lo == pytest.approx(0.8865, abs=1e-4) and hi == pytest.approx(1.0)
    lo, hi = ev.wilson(0, 30)
    assert lo == pytest.approx(0.0, abs=1e-12) and hi == pytest.approx(0.1135, abs=1e-4)


def test_calibration_table_uses_equal_count_bins():
    rng = np.random.default_rng(4)
    p = rng.random(1000)
    y = (rng.random(1000) < p).astype(int)
    table = ev.calibration_table(y, p, n_bins=10)
    assert table["n"].tolist() == [100] * 10
    assert table["mean_predicted"].is_monotonic_increasing


def test_calibration_slope_is_one_for_calibrated_scores():
    rng = np.random.default_rng(5)
    p = rng.uniform(0.01, 0.6, 100_000)
    y = (rng.random(p.size) < p).astype(int)
    assert ev.calibration_slope(y, p) == pytest.approx(1.0, abs=0.05)
