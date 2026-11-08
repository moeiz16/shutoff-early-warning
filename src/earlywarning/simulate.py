"""Simulated billing extract for a mid-sized water-and-power utility.

Writes customers.csv, billing.csv and labels.csv. Real billing extracts are
messy, and the interesting part of this project is noticing that, so the
simulation includes problems I've seen in real ones:

* two fields recorded on the day the extract was pulled (shut-off notices and
  connection status), long after the scoring date;
* payments recorded with the day they arrived, some of them after the scoring date;
* a tariff recode part-way through the year;
* income tiers that are missing more often for some customers than others;
* tenure that disagrees with the billing history;
* a cap on days overdue in the export, and a few bills with nothing paid but a
  lateness value.

Each customer has a hidden "struggling to pay" score. Lateness in every cycle and
the chance of a shut-off over the next six cycles both depend on it.

    PYTHONPATH=src python -m earlywarning.simulate --out data
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_CUSTOMERS = 25_000
SEED = 2024
N_CYCLES = 12
RECODE_CYCLE = 9          # tariff codes change name from this cycle on
DAYS_OVERDUE_CAP = 90     # the export stores anything later as 90

DISTRICTS = {"midtown": 0.0, "harbour": 0.15, "riverside": 0.05, "uplands": -0.10,
             "eastgate": 0.10, "airport": -0.05}
DISTRICT_SHARE = [0.22, 0.18, 0.17, 0.16, 0.15, 0.12]
HOMES = {"detached": 0.0, "apartment": 0.30, "townhouse": 0.10, "manufactured": 0.60}
HOME_SHARE = [0.44, 0.34, 0.10, 0.12]
INCOME = {1: 1.20, 2: 0.60, 3: 0.0, 4: -0.45, 5: -0.80}
INCOME_SHARE = [0.12, 0.22, 0.32, 0.22, 0.12]
CHANNELS = {"direct_debit": -0.50, "app": -0.15, "post": 0.45, "counter": 0.45}
OLD_TARIFF_CODES = {"STD": "A1", "FLEX": "A2", "CARE": "A9"}


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def simulate(n_customers: int = N_CUSTOMERS, seed: int = SEED):
    rng = np.random.default_rng(seed)
    n = n_customers
    ids = np.arange(100_001, 100_001 + n)

    income = rng.choice(list(INCOME), n, p=INCOME_SHARE)
    hardship = (rng.random(n) < _sigmoid(-0.3 - 1.0 * (income - 1))).astype(int)
    home = rng.choice(list(HOMES), n, p=HOME_SHARE)
    district = rng.choice(list(DISTRICTS), n, p=DISTRICT_SHARE)
    occupants = np.clip(1 + rng.poisson(1.6, n), 1, 9)
    tenure = np.clip(np.round(rng.gamma(2.0, 40.0, n)), 1, 300).astype(int)
    ebill = (rng.random(n) < 0.55).astype(int)

    # Better-off customers lean towards direct debit; others towards post and counter.
    weights = np.column_stack([
        np.clip(1.0 + 0.35 * (income - 3), 0.1, None),
        np.full(n, 1.0),
        np.clip(0.6 - 0.15 * (income - 3), 0.1, None),
        np.clip(0.5 - 0.15 * (income - 3), 0.1, None),
    ])
    weights /= weights.sum(axis=1, keepdims=True)
    channel_names = np.array(list(CHANNELS))
    preferred = channel_names[(weights.cumsum(axis=1) > rng.random(n)[:, None]).argmax(axis=1)]

    # Each household uses a different amount; a bigger bill is harder to keep up with.
    usage = rng.lognormal(0.0, 0.3, n)
    base_bill = (40 + 15 * occupants + np.where(home == "detached", 20, 0)) * usage

    struggle = (
        -1.45
        + pd.Series(income).map(INCOME).to_numpy()
        + pd.Series(home).map(HOMES).to_numpy()
        + pd.Series(district).map(DISTRICTS).to_numpy()
        + pd.Series(preferred).map(CHANNELS).to_numpy()
        + 0.30 * np.log(base_bill / 90)
        + 0.55 * (tenure < 12)
        + 0.35 * hardship
        + rng.normal(0.0, 0.75, n)
    )
    p_late = _sigmoid(struggle)

    cycles = []
    for cycle in range(1, N_CYCLES + 1):
        season = 1 + 0.28 * np.cos(2 * np.pi * (cycle - 1) / 12)  # winter peak, summer dip
        due = np.round(base_bill * season * rng.lognormal(0.0, 0.18, n), 2)
        late = rng.random(n) < p_late
        days = np.ceil(rng.lognormal(2.8 + 0.2 * (struggle - struggle.mean()), 0.65, n))
        days = np.where(late, np.clip(days, 1, DAYS_OVERDUE_CAP), 0).astype(int)
        share_paid = np.where(rng.random(n) < 0.06, 1.0, rng.beta(5.0, 1.8, n))
        share_paid = np.where(late & (rng.random(n) < 0.0015), 0.0, share_paid)
        paid = np.where(late, np.round(due * share_paid, 2), due)

        tariff = np.where(hardship == 1, "CARE", np.where(rng.random(n) < 0.65, "STD", "FLEX"))
        if cycle < RECODE_CYCLE:
            tariff = pd.Series(tariff).map(OLD_TARIFF_CODES).to_numpy()
        channel = np.where(rng.random(n) < 0.88, preferred, rng.choice(channel_names, n))

        cycles.append(pd.DataFrame({
            "customer_id": ids, "cycle": cycle, "amount_due": due, "amount_paid": paid,
            "days_overdue": days, "tariff": tariff, "pay_channel": channel,
        }))
    billing = pd.concat(cycles, ignore_index=True)

    # The hardship plan helps: at the same level of struggle, members are shut off less.
    p_shutoff = _sigmoid(-1.45 + 1.6 * struggle - 0.6 * hardship)
    y = (rng.random(n) < p_shutoff).astype(int)

    notices = np.where(y == 1, 1 + rng.binomial(2, 0.45, n), (rng.random(n) < 0.035).astype(int))
    status = np.where((y == 1) & (rng.random(n) > 0.12), "shut_off", "connected")

    # Income is estimated, and the estimate is missing more often for exactly the
    # customers who are more likely to struggle.
    missing = rng.random(n) < _sigmoid(-3.0 + 1.0 * hardship + 0.6 * (home == "manufactured")
                                       + 0.8 * (income <= 2))
    income_tier = np.where(missing, np.nan, income.astype(float))

    customers = pd.DataFrame({
        "customer_id": ids,
        "district": district,
        "home_type": home,
        "occupants": occupants,
        "tenure_cycles": tenure,
        "hardship_plan": hardship,
        "income_tier": income_tier,
        "ebill": ebill,
        "shutoff_notices": notices,
        "status_at_pull": status,
    })
    labels = pd.DataFrame({"customer_id": ids, "shutoff_next_6": y})
    return customers, billing, labels


def write(out_dir, n_customers: int = N_CUSTOMERS, seed: int = SEED) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    customers, billing, labels = simulate(n_customers, seed)
    customers.to_csv(out_dir / "customers.csv", index=False)
    billing.to_csv(out_dir / "billing.csv", index=False)
    labels.to_csv(out_dir / "labels.csv", index=False)
    return out_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description="Write the simulated billing extract.")
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--customers", type=int, default=N_CUSTOMERS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    write(args.out, args.customers, args.seed)
    print(f"{args.customers:,} customers x {N_CYCLES} cycles -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
