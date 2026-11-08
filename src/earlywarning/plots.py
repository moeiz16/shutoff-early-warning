"""Figures. Plain matplotlib; nothing in here changes a number."""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import shap  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from .explain import CATEGORY_ORDER  # noqa: E402

GROUPS = [("not shut off", 0), ("shut off in cycles 13-18", 1)]


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def profile_overview(tariff_by_cycle, trend, days_counts, recode_cycle, cap, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    ax = axes[0, 0]
    for label, col in GROUPS:
        ax.plot(trend.index, trend[("mean_due", col)], marker="o", label=label)
    ax.set(title="Average amount due by cycle (seasonal for everyone)", xlabel="billing cycle",
           ylabel="$")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for label, col in GROUPS:
        ax.plot(trend.index, trend[("late_share", col)], marker="o", label=label)
    ax.set(title="Share of bills paid late (flat all year)", xlabel="billing cycle",
           ylabel="share late", ylim=(0, 0.75))
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    bottom = np.zeros(len(tariff_by_cycle))
    for code in tariff_by_cycle.columns:
        ax.bar(tariff_by_cycle.index, tariff_by_cycle[code], bottom=bottom, label=code)
        bottom += tariff_by_cycle[code].to_numpy()
    ax.set(title=f"Tariff codes change name at cycle {recode_cycle}", xlabel="billing cycle",
           ylabel="customers")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1, 1]
    late = days_counts.loc[days_counts.index > 0]
    ax.bar(late.index, late.to_numpy(), width=1.0)
    ax.set(title=f"Days overdue on late bills (piles up at {cap}: an export cap?)",
           xlabel="days overdue", ylabel="bills")
    _save(fig, path)


def _error_bars(table):
    return [np.clip(table["observed"] - table["ci_low"], 0, None),
            np.clip(table["ci_high"] - table["observed"], 0, None)]


def calibration(overall, by_group, brier, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
    top = 1.05 * max(overall["ci_high"].max(), max(t["ci_high"].max() for t in by_group.values()))

    ax = axes[0]
    ax.errorbar(overall["mean_predicted"], overall["observed"], yerr=_error_bars(overall),
                fmt="o", capsize=3)
    ax.set_title(f"Test set, 10 equal-count bins (Brier {brier:.4f})")

    ax = axes[1]
    for (name, table), marker in zip(by_group.items(), ["o", "s"]):
        ax.errorbar(table["mean_predicted"], table["observed"], yerr=_error_bars(table),
                    fmt=marker, capsize=3, label=name, alpha=0.85)
    ax.set_title("By hardship plan, 5 bins each")
    ax.legend(fontsize=8)

    for ax in axes:
        ax.plot([0, top], [0, top], color="grey", lw=1, ls="--")
        ax.set(xlim=(0, top), ylim=(0, top), xlabel="mean predicted probability")
    axes[0].set_ylabel("observed shut-off rate")
    _save(fig, path)


def cost_curve(dev_curve, test_curve, chosen, band, nobody, everyone, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(dev_curve["threshold"], dev_curve["cost_per_customer"], label="development, out-of-fold")
    ax.plot(test_curve["threshold"], test_curve["cost_per_customer"], label="test", alpha=0.8)
    ax.axvspan(*band, color="grey", alpha=0.15, label="within 2% of the development minimum")
    ax.axvline(chosen, color="black", ls="--", lw=1, label=f"break-even: {chosen:.4f}")
    ax.axhline(nobody, color="tab:red", lw=1, ls=":", label="call nobody")
    ax.axhline(everyone, color="tab:purple", lw=1, ls=":", label="call everyone")
    ax.set(xlabel="call if predicted probability >= threshold",
           ylabel="cost per customer ($)", xlim=(0, 0.4))
    ax.legend(fontsize=8)
    _save(fig, path)


def roc_pr(y, p, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    fpr, tpr, _ = roc_curve(y, p)
    axes[0].plot(fpr, tpr)
    axes[0].plot([0, 1], [0, 1], color="grey", ls="--", lw=1)
    axes[0].set(title="ROC (test)", xlabel="false positive rate", ylabel="true positive rate")
    precision, recall, _ = precision_recall_curve(y, p)
    axes[1].plot(recall, precision)
    axes[1].axhline(np.mean(y), color="grey", ls="--", lw=1, label="base rate")
    axes[1].set(title="Precision-recall (test)", xlabel="recall", ylabel="precision")
    axes[1].legend(fontsize=8)
    _save(fig, path)


def shap_importance(importance, path):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    imp = importance.iloc[::-1]
    ax.barh(imp.index, imp.to_numpy())
    ax.set(xlabel="mean |SHAP value| (log-odds), test set",
           title="What moves the score, grouped by concept")
    _save(fig, path)


def shap_beeswarm(values, X, path):
    shap.summary_plot(values.to_numpy(), features=X.to_numpy(), feature_names=list(X.columns),
                      max_display=15, show=False, plot_size=(9, 6))
    fig = plt.gcf()
    _save(fig, path)


def shap_top_features(grouped_values, shown, names, path):
    fig, axes = plt.subplots(1, len(names), figsize=(4.4 * len(names), 4), sharey=True)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(0)
    for ax, name in zip(axes, names):
        x, s = shown[name], grouped_values[name]
        if x.dtype == object:
            cats = CATEGORY_ORDER.get(name, sorted(x.unique()))
            cats = [c for c in cats if c in set(x)]
            pos = x.map({c: i for i, c in enumerate(cats)}).to_numpy(dtype=float)
            ax.scatter(pos + rng.uniform(-0.25, 0.25, len(pos)), s, s=4, alpha=0.25)
            ax.set_xticks(range(len(cats)), cats)
        else:
            jitter = rng.uniform(-0.2, 0.2, len(x)) if x.nunique() <= 12 else 0
            ax.scatter(x + jitter, s, s=4, alpha=0.25)
        ax.axhline(0, color="grey", lw=0.8)
        xlabel = "late bills out of 11" if name == "Bills paid late" else name
        ax.set(title=name, xlabel=xlabel)
    axes[0].set_ylabel("SHAP value (log-odds)")
    _save(fig, path)
