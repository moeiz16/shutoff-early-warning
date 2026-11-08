"""Scoring: bootstrap intervals, calibration, and the cost of a call list."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from .config import AVOIDABLE_LOSS, CALL_COST

CORE_METRICS = {"auc": roc_auc_score, "auprc": average_precision_score, "brier": brier_score_loss}


# --- cost of a call list --------------------------------------------------------

def break_even_threshold(effectiveness: float = 1.0, call_cost: float = CALL_COST,
                         loss: float = AVOIDABLE_LOSS) -> float:
    """Call when p * effectiveness * loss > call_cost. Only meaningful if p is calibrated."""
    return call_cost / (loss * effectiveness)


def policy_cost(y, called, effectiveness: float = 1.0, call_cost: float = CALL_COST,
                loss: float = AVOIDABLE_LOSS) -> float:
    """Money spent on calls plus the avoidable losses that weren't avoided.

    effectiveness is the share of would-be shut-offs that a call prevents. Shut-offs
    no call could prevent cost the same whatever the list looks like, so they are
    left out. With effectiveness=1 every missed shut-off counts in full.
    """
    y = np.asarray(y)
    called = np.asarray(called, dtype=bool)
    missed = np.count_nonzero(~called & (y == 1))
    return call_cost * np.count_nonzero(called) + effectiveness * loss * missed


def operating_point(y, called, effectiveness: float = 1.0) -> dict:
    y = np.asarray(y)
    called = np.asarray(called, dtype=bool)
    positives = np.count_nonzero(y == 1)
    negatives = len(y) - positives
    tp = np.count_nonzero(called & (y == 1))
    fp = np.count_nonzero(called & (y == 0))
    n_called = np.count_nonzero(called)
    cost = policy_cost(y, called, effectiveness)
    return {
        "n": len(y),
        "called": n_called,
        "called_share": n_called / len(y),
        "true_positives": tp,
        "recall": tp / positives if positives else np.nan,
        "precision": tp / n_called if n_called else np.nan,
        "false_positive_rate": fp / negatives if negatives else np.nan,
        "cost_total": cost,
        "cost_per_customer": cost / len(y),
    }


def cost_curve(y, p, thresholds, effectiveness: float = 1.0) -> pd.DataFrame:
    p = np.asarray(p)
    rows = [{"threshold": t, **operating_point(y, p >= t, effectiveness)} for t in thresholds]
    return pd.DataFrame(rows)


def flat_band(curve: pd.DataFrame, tolerance: float = 0.02) -> tuple[float, float]:
    """Range of thresholds whose cost is within `tolerance` of the minimum."""
    best = curve["cost_per_customer"].min()
    near = curve.loc[curve["cost_per_customer"] <= best * (1 + tolerance), "threshold"]
    return float(near.min()), float(near.max())


def at_threshold(threshold: float, key: str, effectiveness: float = 1.0):
    """Turn an operating-point quantity into a metric(y, p) for the bootstrap."""
    return lambda y, p: operating_point(y, p >= threshold, effectiveness)[key]


# --- uncertainty ----------------------------------------------------------------

def bootstrap(y, p, metrics: dict, n_boot: int, seed: int, alpha: float = 0.05) -> pd.DataFrame:
    """Percentile intervals. Every metric is computed on the same resamples."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in metrics}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yb, pb = y[idx], p[idx]
        if yb.min() == yb.max():  # AUC needs both classes
            continue
        for name, fn in metrics.items():
            draws[name].append(fn(yb, pb))
    rows = []
    for name, fn in metrics.items():
        lo, hi = np.nanpercentile(draws[name], [50 * alpha, 100 - 50 * alpha])
        rows.append({"metric": name, "value": fn(y, p), "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows).set_index("metric")


def paired_auc_difference(y, p_a, p_b, n_boot: int, seed: int) -> tuple[float, float, float]:
    """AUC(a) - AUC(b) with a 95% interval, resampling customers jointly."""
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(roc_auc_score(y[idx], p_a[idx]) - roc_auc_score(y[idx], p_b[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return roc_auc_score(y, p_a) - roc_auc_score(y, p_b), lo, hi


def wilson(k, n, z: float = 1.96):
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    phat = k / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2)) / denom
    # Clip so rounding can't push a bin with no positives a hair above zero.
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


# --- calibration ----------------------------------------------------------------

def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False)
    table = df.groupby("bin").agg(n=("y", "size"), mean_predicted=("p", "mean"),
                                  observed=("y", "mean"), positives=("y", "sum"))
    table["ci_low"], table["ci_high"] = wilson(table["positives"], table["n"])
    return table.reset_index()


def calibration_slope(y, p) -> float:
    """Slope of logit(observed) on logit(predicted). 1 is ideal; below 1 means overconfident."""
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    return float(LogisticRegression(penalty=None).fit(logit, np.asarray(y)).coef_[0][0])


def score_summary(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    base = y.mean()
    return {
        "n": len(y),
        "positives": int(y.sum()),
        "auc": roc_auc_score(y, p),
        "auprc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "brier_of_base_rate": base * (1 - base),
        "log_loss": log_loss(y, p),
        "observed_rate": base,
        "mean_predicted": p.mean(),
        "calibration_slope": calibration_slope(y, p),
    }


# --- subgroups ------------------------------------------------------------------

def subgroup_table(y, p, groups, threshold: float, n_boot: int, seed: int) -> pd.DataFrame:
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    metrics = {
        **CORE_METRICS,
        "recall": at_threshold(threshold, "recall"),
        "precision": at_threshold(threshold, "precision"),
        "cost_per_customer": at_threshold(threshold, "cost_per_customer"),
    }
    rows = []
    for g in sorted(np.unique(groups)):
        mask = groups == g
        yg, pg = y[mask], p[mask]
        boot = bootstrap(yg, pg, metrics, n_boot, seed)
        row = {
            "group": g,
            "n": int(mask.sum()),
            "positives": int(yg.sum()),
            "observed_rate": yg.mean(),
            "mean_predicted": pg.mean(),
            "called_share": float((pg >= threshold).mean()),
        }
        for name in metrics:
            row[name] = boot.loc[name, "value"]
            row[f"{name}_lo"] = boot.loc[name, "ci_low"]
            row[f"{name}_hi"] = boot.loc[name, "ci_high"]
        rows.append(row)
    return pd.DataFrame(rows)
