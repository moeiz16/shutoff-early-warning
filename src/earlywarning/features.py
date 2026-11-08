"""Features, restricted to what the utility could have known on the scoring date."""
import numpy as np
import pandas as pd

from .config import CUTOFF_CYCLE, DAYS_PER_CYCLE, EXCLUDED_FIELDS

CHANNELS = ["app", "post", "counter"]                       # 'direct_debit' is the reference
INCOME_TIERS = [1, 2, 4, 5]                                  # tier 3, the most common, is the reference
HOMES = ["apartment", "townhouse", "manufactured"]           # vs detached
DISTRICTS = ["harbour", "riverside", "uplands", "eastgate", "airport"]  # vs midtown


def visible_at_cutoff(billing: pd.DataFrame, cutoff_cycle: int = CUTOFF_CYCLE,
                      days_per_cycle: int = DAYS_PER_CYCLE) -> pd.DataFrame:
    """Billing history as the billing system would have shown it on the scoring date.

    amount_paid and days_overdue describe when a bill was eventually paid, and the
    extract was pulled six cycles later, so some of those payments happened after
    the scoring date. Assuming a bill falls due at the end of its own cycle:

    * the cutoff cycle's bill has only just fallen due, so its payment fields
      (amount, lateness, channel) are treated as unknown;
    * an earlier bill paid more days late than it had been open on the scoring
      date was still unpaid then, so it counts as outstanding: nothing paid, and
      lateness equal to the days it had been open.

    Amounts due are always kept; a bill is known once it is issued.
    """
    b = billing.loc[billing["cycle"] <= cutoff_cycle].copy()
    days_open = (cutoff_cycle - b["cycle"]) * days_per_cycle
    landed = b["days_overdue"] <= days_open
    observed = b["cycle"] < cutoff_cycle

    b["payment_observed"] = observed
    b["late"] = np.where(observed, b["days_overdue"] > 0, np.nan)
    b["outstanding"] = np.where(observed, ~landed, np.nan)
    b["days_late_seen"] = np.where(observed, np.minimum(b["days_overdue"], days_open), np.nan)
    b["paid_seen"] = np.where(observed, np.where(landed, b["amount_paid"], 0.0), np.nan)
    b["channel_seen"] = b["pay_channel"].where(observed)
    return b


def as_recorded(billing: pd.DataFrame) -> pd.DataFrame:
    """Same columns as visible_at_cutoff but taking the extract at face value.

    Only used to measure what the point-in-time rules cost; never for the model.
    """
    b = billing.copy()
    b["payment_observed"] = True
    b["late"] = (b["days_overdue"] > 0).astype(float)
    b["outstanding"] = 0.0
    b["days_late_seen"] = b["days_overdue"].astype(float)
    b["paid_seen"] = b["amount_paid"]
    b["channel_seen"] = b["pay_channel"]
    return b


def payment_features(view: pd.DataFrame, extended: bool = False) -> pd.DataFrame:
    obs = view.loc[view["payment_observed"]]
    by_cust = obs.groupby("customer_id")
    due = by_cust["amount_due"].sum()

    f = pd.DataFrame({
        "overdue_share": by_cust["late"].mean(),
        "avg_days_overdue": by_cust["days_late_seen"].mean(),
        "unpaid_share": 1 - by_cust["paid_seen"].sum() / due,
        "avg_amount_due": view.groupby("customer_id")["amount_due"].mean(),
    })
    channel_mix = pd.crosstab(obs["customer_id"], obs["channel_seen"], normalize="index")
    for ch in CHANNELS:
        f[f"channel_share_{ch}"] = channel_mix[ch] if ch in channel_mix else 0.0

    if extended:
        last = obs["cycle"].max()
        recent = obs.loc[obs["cycle"] > last - 3].groupby("customer_id")
        f["overdue_share_recent"] = recent["late"].mean()
        f["unpaid_share_recent"] = 1 - recent["paid_seen"].sum() / recent["amount_due"].sum()
        f["max_days_overdue"] = by_cust["days_late_seen"].max()
        f["bills_30plus_overdue"] = (obs["days_late_seen"] > 30).groupby(obs["customer_id"]).sum()
        f["outstanding_at_cutoff"] = by_cust["outstanding"].sum()
        f["amount_cv"] = view.groupby("customer_id")["amount_due"].std() / f["avg_amount_due"]
        first_channel = obs.loc[obs["cycle"] == obs["cycle"].min()].set_index("customer_id")["channel_seen"]
        last_channel = obs.loc[obs["cycle"] == last].set_index("customer_id")["channel_seen"]
        f["left_direct_debit"] = ((first_channel == "direct_debit")
                                  & (last_channel != "direct_debit")).astype(int)
    return f


def customer_features(customers: pd.DataFrame) -> pd.DataFrame:
    c = customers.set_index("customer_id")
    f = pd.DataFrame(index=c.index)
    f["hardship_plan"] = c["hardship_plan"]
    f["occupants"] = c["occupants"]
    f["log_tenure"] = np.log1p(c["tenure_cycles"])
    f["ebill"] = c["ebill"]
    # Missing income is informative here (higher shut-off rate), so it gets its own
    # indicator instead of being imputed away.
    f["income_missing"] = c["income_tier"].isna().astype(int)
    for tier in INCOME_TIERS:
        f[f"income_tier_{tier}"] = (c["income_tier"] == tier).astype(int)
    for h in HOMES:
        f[f"home_{h}"] = (c["home_type"] == h).astype(int)
    for d in DISTRICTS:
        f[f"district_{d}"] = (c["district"] == d).astype(int)
    return f


def build_features(customers: pd.DataFrame, billing: pd.DataFrame,
                   point_in_time: bool = True, extended: bool = False) -> pd.DataFrame:
    view = visible_at_cutoff(billing) if point_in_time else as_recorded(billing)
    X = customer_features(customers).join(payment_features(view, extended=extended), how="left")
    leaked = set(X.columns) & set(EXCLUDED_FIELDS)
    if leaked:
        raise RuntimeError(f"excluded fields reached the feature matrix: {sorted(leaked)}")
    if X.isna().any().any():
        raise RuntimeError("feature matrix has missing values: "
                           f"{X.columns[X.isna().any()].tolist()}")
    return X.sort_index()


# Readable names used to roll SHAP values up from columns to concepts.
_GROUP_PREFIXES = {
    "income_": "Income tier",
    "home_": "Home type",
    "district_": "District",
    "channel_share_": "How bills are paid",
}
_GROUP_NAMES = {
    "overdue_share": "Bills paid late",
    "avg_days_overdue": "Average days overdue",
    "unpaid_share": "Share of amount due left unpaid",
    "avg_amount_due": "Average bill",
    "hardship_plan": "Hardship plan",
    "occupants": "Occupants",
    "log_tenure": "Tenure",
    "ebill": "E-billing",
}


def feature_group(column: str) -> str:
    if column in _GROUP_NAMES:
        return _GROUP_NAMES[column]
    for prefix, name in _GROUP_PREFIXES.items():
        if column.startswith(prefix):
            return name
    return column
