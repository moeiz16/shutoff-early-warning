"""Data profiling.

Checks that don't involve the label run on every customer. Anything that looks
at the label runs on the development split only, so the test set stays unseen.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import plots
from .config import CUTOFF_CYCLE, DAYS_PER_CYCLE, TARGET
from .data import harmonize_tariffs

# When each customer field was recorded, as described in the extract notes (see README).
FIELD_TIMING = {
    "district": "static",
    "home_type": "static",
    "occupants": "at sign-up",
    "tenure_cycles": "end of cycle 12",
    "hardship_plan": "end of cycle 12",
    "income_tier": "estimated, date not given",
    "ebill": "not stated",
    "shutoff_notices": "day of extract (end of cycle 18)",
    "status_at_pull": "day of extract (end of cycle 18)",
}


def leakage_screen(customers: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """How well each raw field separates the outcome on its own, next to when it was recorded."""
    c = customers.loc[y.index]
    rows = []
    for col, recorded in FIELD_TIMING.items():
        x = c[col]
        if x.dtype == object:
            score = x.map(y.groupby(x).mean())
        else:
            score = x.fillna(x.median())
        auc = roc_auc_score(y, score)
        rows.append({"field": col, "recorded": recorded,
                     "single_field_auc": round(max(auc, 1 - auc), 4),
                     "missing_share": round(x.isna().mean(), 4)})
    return pd.DataFrame(rows).sort_values("single_field_auc", ascending=False)


def tariff_checks(billing: pd.DataFrame, customers: pd.DataFrame):
    by_cycle = pd.crosstab(billing["cycle"], billing["tariff"])
    raw = billing.pivot(index="customer_id", columns="cycle", values="tariff")
    cycles = list(raw.columns)
    changed = {b: float((raw[a] != raw[b]).mean()) for a, b in zip(cycles, cycles[1:])}
    recode_cycle = max(changed, key=changed.get)  # the cycle where every code changes at once

    wide = harmonize_tariffs(billing).pivot(index="customer_id", columns="cycle", values="tariff")
    on_care = wide.eq("CARE")
    care_ids = set(wide.index[on_care.any(axis=1)])
    members = set(customers.index[customers["hardship_plan"] == 1])
    others = wide.loc[~wide.index.isin(care_ids)]
    stay = np.mean([(others[a] == others[b]).mean() for a, b in zip(cycles, cycles[1:])])
    std_share = others.eq("STD").to_numpy().mean()
    return by_cycle, {
        "recode_cycle": int(recode_cycle),
        "share_of_codes_changing_at_recode": round(changed[recode_cycle], 4),
        "care_customers": len(care_ids),
        "care_is_hardship_plan": care_ids == members,
        "care_all_year": bool((on_care.all(axis=1) | ~on_care.any(axis=1)).all()),
        "std_flex_same_as_last_cycle": round(float(stay), 4),
        "same_if_drawn_at_random": round(float(std_share**2 + (1 - std_share) ** 2), 4),
    }


def payment_checks(billing: pd.DataFrame) -> tuple[pd.Series, dict]:
    days_open = (CUTOFF_CYCLE - billing["cycle"]) * DAYS_PER_CYCLE
    before_cutoff = billing["cycle"] < CUTOFF_CYCLE
    arrived_late = before_cutoff & (billing["days_overdue"] > days_open)
    on_time = billing["days_overdue"] == 0
    counts = billing["days_overdue"].value_counts().sort_index()
    cap = int(billing["days_overdue"].max())
    last = billing.loc[billing["cycle"] == CUTOFF_CYCLE]
    return counts, {
        "share_of_bills_late": round(float((~on_time).mean()), 4),
        "days_overdue_max": cap,
        "bills_at_max": int(counts.get(cap, 0)),
        "bills_at_max_minus_1": int(counts.get(cap - 1, 0)),
        "on_time_bills_paid_in_full": round(float(
            (billing.loc[on_time, "amount_paid"] == billing.loc[on_time, "amount_due"]).mean()), 4),
        "late_bills_mean_share_paid": round(float(
            (billing.loc[~on_time, "amount_paid"] / billing.loc[~on_time, "amount_due"]).mean()), 4),
        "nothing_paid_but_overdue": int(((billing["amount_paid"] == 0) & ~on_time).sum()),
        "overpaid_bills": int((billing["amount_paid"] > billing["amount_due"] + 0.005).sum()),
        "cutoff_cycle_bills_paid_late": int((last["days_overdue"] > 0).sum()),
        "earlier_bills_paid_after_cutoff": {
            int(k): int(v) for k, v in arrived_late.groupby(billing["cycle"]).sum().items() if v
        },
    }


def tenure_checks(billing: pd.DataFrame, customers: pd.DataFrame) -> dict:
    """Customers said to have joined during the year still have twelve cycles of bills."""
    joined = customers.index[customers["tenure_cycles"] < 12]
    b = billing.loc[billing["customer_id"].isin(joined)].merge(
        customers[["tenure_cycles"]], left_on="customer_id", right_index=True)
    before = b["cycle"] < (13 - b["tenure_cycles"])
    late = b["days_overdue"] > 0
    return {
        "customers_joined_during_year": int(len(joined)),
        "bills_dated_before_sign_up": int(before.sum()),
        "late_share_before_sign_up": round(float(late[before].mean()), 4),
        "late_share_after_sign_up": round(float(late[~before].mean()), 4),
        "late_share_all_customers": round(float((billing["days_overdue"] > 0).mean()), 4),
    }


def run(extract, dev_ids, tables_dir, figures_dir) -> dict:
    customers = extract.customers.set_index("customer_id").sort_index()
    billing = extract.billing
    y_dev = extract.labels.set_index("customer_id").loc[dev_ids, TARGET]
    dev = customers.loc[dev_ids]

    summary = {
        "customers": len(customers),
        "billing_rows": len(billing),
        "shutoff_rate_all": round(float(extract.labels[TARGET].mean()), 4),
        "hardship_share": round(float(customers["hardship_plan"].mean()), 4),
        "income_missing_share": round(float(customers["income_tier"].isna().mean()), 4),
        "income_missing_by_hardship_plan": customers.groupby("hardship_plan")["income_tier"]
        .apply(lambda s: round(float(s.isna().mean()), 4)).to_dict(),
        "income_missing_by_home": customers.groupby("home_type")["income_tier"]
        .apply(lambda s: round(float(s.isna().mean()), 4)).to_dict(),
    }

    screen = leakage_screen(customers, y_dev)
    screen.to_csv(tables_dir / "profile_leakage_screen.csv", index=False)

    notices = pd.crosstab(dev["shutoff_notices"], y_dev)
    status = pd.crosstab(dev["status_at_pull"], y_dev)
    notices.to_csv(tables_dir / "profile_notices_vs_label_dev.csv")
    status.to_csv(tables_dir / "profile_status_vs_label_dev.csv")
    summary["dev_shutoff_rate_by_notices"] = (
        y_dev.groupby(dev["shutoff_notices"]).mean().round(4).to_dict())
    summary["dev_shutoff_rate_when_status_shut_off"] = round(
        float(y_dev[dev["status_at_pull"] == "shut_off"].mean()), 4)
    summary["dev_shut_off_but_connected_at_pull"] = int(
        ((dev["status_at_pull"] == "connected") & (y_dev == 1)).sum())

    tariff_by_cycle, tariff_summary = tariff_checks(billing, customers)
    tariff_by_cycle.to_csv(tables_dir / "profile_tariff_by_cycle.csv")
    summary["tariffs"] = tariff_summary

    days_counts, pay_summary = payment_checks(billing)
    summary["payments"] = pay_summary
    summary["tenure"] = tenure_checks(billing, customers)

    dev_rows = billing.loc[billing["customer_id"].isin(dev_ids)].merge(
        y_dev.rename(TARGET), left_on="customer_id", right_index=True)
    trend = (dev_rows.assign(late=dev_rows["days_overdue"] > 0)
             .groupby(["cycle", TARGET])
             .agg(mean_due=("amount_due", "mean"), late_share=("late", "mean"))
             .unstack(TARGET))
    trend.round(4).to_csv(tables_dir / "profile_cycle_trend_by_outcome_dev.csv")

    missing = dev["income_tier"].isna()
    summary["dev_shutoff_rate_income_missing"] = round(float(y_dev[missing].mean()), 4)
    summary["dev_shutoff_rate_income_present"] = round(float(y_dev[~missing].mean()), 4)

    # Income mix inside and outside the hardship plan (0 = no estimate); it explains why the
    # model separates members less sharply than everyone else.
    mix = pd.crosstab(customers["hardship_plan"], customers["income_tier"].fillna(0).astype(int),
                      normalize="index")
    mix.round(4).to_csv(tables_dir / "profile_income_mix_by_hardship_plan.csv")
    summary["share_bottom_two_tiers_or_missing_by_hardship_plan"] = (
        mix[[0, 1, 2]].sum(axis=1).round(4).to_dict())

    plots.profile_overview(tariff_by_cycle, trend, days_counts, tariff_summary["recode_cycle"],
                           pay_summary["days_overdue_max"], figures_dir / "profile_overview.png")
    return summary
