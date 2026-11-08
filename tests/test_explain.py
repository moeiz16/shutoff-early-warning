import numpy as np
import pytest

from earlywarning import explain, models
from earlywarning.features import build_features


@pytest.fixture(scope="module")
def fitted(small_extract):
    customers, billing, labels = small_extract
    X = build_features(customers, billing)
    y = labels.set_index("customer_id").loc[X.index, "shutoff_next_6"]
    return models.logistic().fit(X, y), X, customers


def test_shap_values_add_up_to_the_models_log_odds(fitted):
    model, X, _ = fitted
    values, base = explain.linear_shap(model, X, X.iloc[:300])
    reconstructed = values.sum(axis=1).to_numpy() + base
    np.testing.assert_allclose(reconstructed, model.decision_function(X.iloc[:300]), atol=1e-8)


def test_linear_shap_matches_the_textbook_formula(fitted):
    # For a linear model, SHAP_j = coef_j * (z_j - mean z_j), in standardised units.
    model, X, _ = fitted
    values, _ = explain.linear_shap(model, X, X.iloc[:50])
    z = model[0].transform(X)
    expected = model[-1].coef_[0] * (z[:50] - z.mean(axis=0))
    np.testing.assert_allclose(values.to_numpy(), expected, atol=1e-8)


def test_grouping_keeps_each_rows_total(fitted):
    model, X, _ = fitted
    values, _ = explain.linear_shap(model, X, X.iloc[:100])
    by_concept = explain.grouped(values)
    np.testing.assert_allclose(by_concept.sum(axis=1), values.sum(axis=1), atol=1e-10)
    assert "Income tier" in by_concept.columns and "income_tier_1" not in by_concept.columns


def test_level_table_covers_every_customer(fitted):
    model, X, customers = fitted
    values, _ = explain.linear_shap(model, X, X)
    by_concept = explain.grouped(values)
    shown = explain.display_values(customers, X)
    p = model.predict_proba(X)[:, 1]
    table = explain.level_table("Income tier", by_concept["Income tier"], shown["Income tier"],
                                p, np.zeros(len(p)))
    assert table["n"].sum() == len(X)
    assert table["level"].tolist()[-1] == "missing"
