"""SHAP attributions for the logistic model, rolled up to the concepts people actually ask about."""
import numpy as np
import pandas as pd
import shap

from .config import CUTOFF_CYCLE
from .features import feature_group


def linear_shap(model, X_background: pd.DataFrame, X: pd.DataFrame):
    """Exact interventional SHAP values (log-odds) for a scaler + logistic regression pipeline.

    The masker gets the full background; shap's default would quietly subsample it to 100 rows.
    """
    scaler, clf = model[0], model[-1]
    background = scaler.transform(X_background)
    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.LinearExplainer(clf, masker)
    values = explainer.shap_values(scaler.transform(X))
    base = float(np.ravel(explainer.expected_value)[0])
    return pd.DataFrame(values, index=X.index, columns=X.columns), base


def grouped(values: pd.DataFrame) -> pd.DataFrame:
    """Sum the one-hot columns of each concept (e.g. four income-tier columns -> 'Income tier')."""
    names = [feature_group(c) for c in values.columns]
    return values.T.groupby(names, sort=False).sum().T


def importance(grouped_values: pd.DataFrame) -> pd.Series:
    return grouped_values.abs().mean().sort_values(ascending=False).rename("mean_abs_shap")


def display_values(customers: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """One human-readable value per concept, for plots and the level tables."""
    c = customers.set_index("customer_id").loc[X.index]
    shares = X[["channel_share_app", "channel_share_post", "channel_share_counter"]].copy()
    shares["channel_share_direct_debit"] = 1 - shares.sum(axis=1)
    main_channel = shares.idxmax(axis=1).str.replace("channel_share_", "", regex=False)
    n_bills = CUTOFF_CYCLE - 1
    return pd.DataFrame({
        "Bills paid late": np.rint(X["overdue_share"] * n_bills).astype(int),
        "Average days overdue": X["avg_days_overdue"],
        "Share of amount due left unpaid": X["unpaid_share"],
        "Average bill": X["avg_amount_due"],
        "How bills are paid": main_channel,
        "Income tier": c["income_tier"].map(lambda v: "missing" if pd.isna(v) else str(int(v))),
        "Home type": c["home_type"],
        "District": c["district"],
        "Hardship plan": c["hardship_plan"].map({0: "no", 1: "yes"}),
        "Occupants": c["occupants"],
        "Tenure": c["tenure_cycles"],
        "E-billing": c["ebill"].map({0: "no", 1: "yes"}),
    }, index=X.index)


CATEGORY_ORDER = {
    "How bills are paid": ["direct_debit", "app", "post", "counter"],
    "Income tier": ["1", "2", "3", "4", "5", "missing"],
}


def level_table(name: str, shap_values: pd.Series, shown: pd.Series, p, y) -> pd.DataFrame:
    """Mean SHAP, mean predicted risk and observed rate at each level of one concept."""
    if shown.dtype == object or shown.nunique() <= 12:
        level = shown
    else:
        level = pd.qcut(shown, 5, duplicates="drop")  # ordered bins, so groupby keeps them in order
    df = pd.DataFrame({"level": level, "shap": shap_values, "predicted": np.asarray(p),
                       "observed": np.asarray(y)}, index=shown.index)
    table = df.groupby("level", observed=True).agg(
        n=("shap", "size"), mean_shap=("shap", "mean"),
        mean_predicted=("predicted", "mean"), observed_rate=("observed", "mean"),
    ).reset_index()
    order = CATEGORY_ORDER.get(name)
    if order:
        present = set(table["level"])
        table["level"] = pd.Categorical(table["level"], [o for o in order if o in present])
        table = table.sort_values("level")
    table["level"] = table["level"].astype(str)
    table.insert(0, "feature", name)
    return table
