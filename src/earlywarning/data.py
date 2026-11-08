"""Load the billing extract and fail loudly if it isn't the shape the analysis assumes."""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import TARGET

# Tariff codes before the recode, and the names that replaced them. The extract
# doesn't document the mapping; it's inferred from volumes (A9 and CARE are the
# same customers, and A1/A2 carry the same shares as STD/FLEX).
TARIFF_RENAMES = {"A1": "STD", "A2": "FLEX", "A9": "CARE"}


@dataclass
class Extract:
    customers: pd.DataFrame
    billing: pd.DataFrame
    labels: pd.DataFrame

    def labelled_customers(self) -> pd.DataFrame:
        merged = self.customers.merge(self.labels, on="customer_id", validate="one_to_one")
        return merged.set_index("customer_id").sort_index()


def load_extract(data_dir) -> Extract:
    data_dir = Path(data_dir)
    extract = Extract(
        customers=pd.read_csv(data_dir / "customers.csv"),
        billing=pd.read_csv(data_dir / "billing.csv"),
        labels=pd.read_csv(data_dir / "labels.csv"),
    )
    problems = check_extract(extract)
    if problems:
        raise ValueError("Extract failed basic checks:\n  " + "\n  ".join(problems))
    return extract


def check_extract(extract: Extract, n_cycles: int = 12) -> list[str]:
    customers, billing, labels = extract.customers, extract.billing, extract.labels
    problems = []

    for name, df in [("customers", customers), ("labels", labels)]:
        if df["customer_id"].duplicated().any():
            problems.append(f"{name}: duplicate customer_id")
    if billing.duplicated(["customer_id", "cycle"]).any():
        problems.append("billing: duplicate customer-cycle rows")

    ids = set(customers["customer_id"])
    if set(labels["customer_id"]) != ids or set(billing["customer_id"]) != ids:
        problems.append("customer_id differs between files")

    cycles_per_customer = billing.groupby("customer_id")["cycle"].nunique()
    if not billing["cycle"].between(1, n_cycles).all() or (cycles_per_customer != n_cycles).any():
        problems.append(f"billing: not every customer has cycles 1-{n_cycles}")

    for col in ["amount_due", "amount_paid", "days_overdue"]:
        if billing[col].isna().any() or (billing[col] < 0).any():
            problems.append(f"billing.{col}: missing or negative values")

    if not labels[TARGET].isin([0, 1]).all():
        problems.append(f"labels.{TARGET} is not 0/1")
    return problems


def harmonize_tariffs(billing: pd.DataFrame) -> pd.DataFrame:
    out = billing.copy()
    out["tariff"] = out["tariff"].replace(TARIFF_RENAMES)
    return out
